# Copyright (c) ModelScope Contributors. All rights reserved.
"""Immutable sandbox profile manifests and least-privilege resolution."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .base import StepResult
from .seeds import CLEAR_WORKSPACE, LIBRARY_CANARIES, SEED_PROGRAMS

__all__ = [
    'SandboxProfile',
    'ProfileResolution',
    'ProfileRegistry',
    'default_profile_registry',
]

_PROFILE_NAME = re.compile(r'^[a-z][a-z0-9-]*$')
_SESSION_KINDS = frozenset({'no_tool', 'agentenv', 'lean', 'broker', 'legacy'})
_FORBIDDEN_CAPABILITIES = frozenset({
    'raw_network',
    'arbitrary_command',
    'custom_image',
    'secret_access',
    'physical_action',
})


def _unique(values: tuple[str, ...], field: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f'{field} must not contain duplicates')
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f'{field} must contain non-empty strings')


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class SandboxProfile:
    name: str
    version: int
    session_kind: str
    capabilities: tuple[str, ...]
    task_forms: tuple[str, ...]
    allowed_tools: tuple[str, ...]
    verifier_kinds: tuple[str, ...]
    input_kinds: tuple[str, ...]
    output_kinds: tuple[str, ...]
    setup_manifest: tuple[str, ...] = ()
    runtime_ref: str = ''
    available: bool = True
    unavailable_reason: str = ''
    enabled_for_new_tasks: bool = True

    def __post_init__(self) -> None:
        if not _PROFILE_NAME.fullmatch(self.name):
            raise ValueError(f'invalid profile name {self.name!r}')
        if not isinstance(self.version, int) or self.version < 1:
            raise ValueError('profile version must be a positive integer')
        if self.session_kind not in _SESSION_KINDS:
            raise ValueError(f'unknown session kind {self.session_kind!r}')
        for field in ('capabilities', 'task_forms', 'allowed_tools', 'verifier_kinds', 'input_kinds', 'output_kinds',
                      'setup_manifest'):
            _unique(getattr(self, field), field)
        if not self.capabilities or not self.task_forms or not self.verifier_kinds:
            raise ValueError('profile capabilities, task forms, and verifiers must be non-empty')
        forbidden = set(self.capabilities) & _FORBIDDEN_CAPABILITIES
        if forbidden:
            raise ValueError(f'profiles cannot grant forbidden capabilities: {sorted(forbidden)}')
        if self.session_kind == 'no_tool' and self.allowed_tools:
            raise ValueError('no-tool profiles cannot advertise tools')
        if self.available and self.unavailable_reason:
            raise ValueError('available profiles cannot have an unavailable reason')
        if not self.available and not self.unavailable_reason:
            raise ValueError('unavailable profiles require a reason')

    @property
    def profile_id(self) -> str:
        return f'{self.name}@{self.version}'

    def manifest(self) -> dict[str, object]:
        return {
            'profile_id': self.profile_id,
            'session_kind': self.session_kind,
            'capabilities': sorted(self.capabilities),
            'task_forms': sorted(self.task_forms),
            'allowed_tools': list(self.allowed_tools),
            'verifier_kinds': sorted(self.verifier_kinds),
            'input_kinds': sorted(self.input_kinds),
            'output_kinds': sorted(self.output_kinds),
            'setup_manifest': list(self.setup_manifest),
            'runtime_ref': self.runtime_ref,
            'enabled_for_new_tasks': self.enabled_for_new_tasks,
        }

    @property
    def digest(self) -> str:
        return _digest(self.manifest())


@dataclass(frozen=True)
class ProfileResolution:
    requested_profile_id: str
    profile: SandboxProfile
    registry_digest: str
    router_version: str
    reason: str


class ProfileRegistry:

    def __init__(self, router_version: str = 'sandbox-router-v1') -> None:
        if not router_version:
            raise ValueError('router_version must not be empty')
        self.router_version = router_version
        self._profiles: dict[str, SandboxProfile] = {}

    def register(self, profile: SandboxProfile) -> None:
        if profile.profile_id in self._profiles:
            raise ValueError(f'profile {profile.profile_id!r} is already registered')
        self._profiles[profile.profile_id] = profile

    def get(self, profile_id: str) -> SandboxProfile:
        try:
            return self._profiles[profile_id]
        except KeyError:
            raise ValueError(f'profile {profile_id!r} is not registered') from None

    def profiles(self) -> tuple[SandboxProfile, ...]:
        return tuple(self._profiles[key] for key in sorted(self._profiles))

    @property
    def digest(self) -> str:
        return _digest({
            'router_version': self.router_version,
            'profiles': [profile.manifest() for profile in self.profiles()],
        })

    @property
    def capabilities(self) -> frozenset[str]:
        return frozenset(capability for profile in self._profiles.values() for capability in profile.capabilities)

    def resolve(self, capabilities: Iterable[str], task_form: str, requested_profile_id: str = '') -> ProfileResolution:
        required = frozenset(capabilities)
        if not required:
            raise ValueError('an intent must request at least one capability')
        forbidden = required & _FORBIDDEN_CAPABILITIES
        if forbidden:
            raise ValueError(f'forbidden capabilities requested: {sorted(forbidden)}')
        unknown = required - self.capabilities
        if unknown:
            raise ValueError(f'unknown capabilities requested: {sorted(unknown)}')

        requested: SandboxProfile | None = None
        if requested_profile_id:
            requested = self.get(requested_profile_id)
            if not required.issubset(requested.capabilities) or task_form not in requested.task_forms:
                raise ValueError(f'intent would require escalation beyond requested profile '
                                 f'{requested_profile_id!r}')

        candidates = [
            profile for profile in self._profiles.values() if profile.available and profile.enabled_for_new_tasks
            and required.issubset(profile.capabilities) and task_form in profile.task_forms
        ]
        if not candidates:
            unavailable = [
                profile.profile_id for profile in self._profiles.values()
                if required.issubset(profile.capabilities) and task_form in profile.task_forms and not profile.available
            ]
            if unavailable:
                raise ValueError(f'sandbox unavailable for intent: {sorted(unavailable)}')
            raise ValueError(f'no profile supports task form {task_form!r} with capabilities '
                             f'{sorted(required)}')

        def priority(profile: SandboxProfile) -> tuple:
            requested_penalty = int(requested is not None and profile != requested)
            return (len(set(profile.capabilities) - required), len(profile.allowed_tools), requested_penalty,
                    profile.profile_id)

        resolved = min(candidates, key=priority)
        if requested is None:
            reason = 'least-privilege profile satisfying the declared task form and capabilities'
        elif resolved == requested:
            reason = 'requested profile is the least-privilege valid profile'
        else:
            reason = f'downgraded from {requested.profile_id} to the least-privilege valid profile'
        return ProfileResolution(
            requested_profile_id=requested_profile_id,
            profile=resolved,
            registry_digest=self.digest,
            router_version=self.router_version,
            reason=reason,
        )


def default_profile_registry() -> ProfileRegistry:
    registry = ProfileRegistry()
    registry.register(
        SandboxProfile(
            name='knowledge-answer',
            version=1,
            session_kind='no_tool',
            capabilities=('objective_answer', ),
            task_forms=('knowledge_answer', ),
            allowed_tools=(),
            verifier_kinds=('answer_equals', 'answer_one_of', 'answer_number'),
            input_kinds=('given_text', ),
            output_kinds=('answer', ),
            runtime_ref='host-no-tool-v1',
        ))
    registry.register(
        SandboxProfile(
            name='python-compute',
            version=1,
            session_kind='agentenv',
            capabilities=('numeric_compute', 'symbolic_compute', 'data_compute'),
            task_forms=('numeric_problem', 'symbolic_problem', 'given_data_analysis'),
            allowed_tools=('python_executor', ),
            verifier_kinds=('answer_equals', 'answer_one_of', 'answer_number', 'file_json', 'python'),
            input_kinds=('given_text', 'data_fixture'),
            output_kinds=('answer', 'workspace'),
            setup_manifest=('clear_workspace', 'python-numpy-sympy-canary'),
            runtime_ref='twinkle-rsi-msagent',
        ))
    registry.register(
        SandboxProfile(
            # agentenv, not a new `lean` session kind: the compiler runs inside the
            # sandbox through the same shell the other profiles use, so there is
            # nothing for a new kind to enable. `session_kind='lean'` was rejected by
            # open_profile_session outright, which made the profile unreachable
            # regardless of whether Lean itself worked.
            name='lean-proof',
            version=1,
            session_kind='agentenv',
            capabilities=('formal_proof', ),
            task_forms=('formal_proof', ),
            # No `lean_check` tool. Compiling is `lake env lean` and the verifier runs
            # it directly, so a separate tool would be a second way to do the same
            # thing -- and a tool the model can call is a tool the model can call
            # instead of thinking. `shell_executor` is what the profile needs.
            allowed_tools=('shell_executor', 'python_executor', 'read_file', 'write_file', 'edit_file', 'glob', 'grep'),
            verifier_kinds=('lean', 'file_contains', 'file_exists'),
            input_kinds=('given_text', ),
            output_kinds=('proof', 'workspace'),
            setup_manifest=('clear_workspace', 'python-numpy-sympy-canary'),
            runtime_ref='lean4-mathlib-pinned',
        ))
    registry.register(
        SandboxProfile(
            name='code-python',
            version=1,
            session_kind='agentenv',
            capabilities=('python_code', 'workspace_files'),
            task_forms=('python_implementation', ),
            allowed_tools=('shell_executor', 'python_executor', 'read_file', 'write_file', 'edit_file', 'glob', 'grep'),
            verifier_kinds=('file_exists', 'file_absent', 'file_contains', 'file_equals', 'file_json', 'python',
                            'hidden_python'),
            input_kinds=('given_text', 'repository_fixture'),
            output_kinds=('workspace', ),
            setup_manifest=('clear_workspace', 'python-test-runner-canary'),
            runtime_ref='twinkle-rsi-msagent',
        ))
    # ------------------------------------------------------------------
    # Seeded profiles. Each one starts the agent from a real state instead of an
    # empty directory, so the task can require reading before acting. The setup
    # manifest names the seed program that builds it (seeds.py) and then
    # the canary that proves the state is usable.
    # ------------------------------------------------------------------
    registry.register(
        SandboxProfile(
            name='sql-analytics',
            version=1,
            session_kind='agentenv',
            capabilities=('database_query', 'data_compute'),
            task_forms=('database_analysis', ),
            # No db_query tool: the sandbox already has sqlite3 through python_executor
            # and the shell, and adding a bespoke tool would need ms-agent changes for
            # no capability the agent lacks. The seed and the sql_query VERIFIER are
            # what change here -- the tool surface deliberately does not.
            allowed_tools=('shell_executor', 'python_executor', 'read_file', 'write_file', 'glob', 'grep'),
            verifier_kinds=('sql_query', 'file_exists', 'file_json', 'file_contains'),
            input_kinds=('given_text', 'data_fixture'),
            output_kinds=('answer', 'workspace'),
            setup_manifest=('clear_workspace', 'seed-sql-analytics', 'python-numpy-sympy-canary'),
            runtime_ref='twinkle-rsi-msagent',
        ))
    registry.register(
        SandboxProfile(
            name='data-wrangling',
            version=1,
            session_kind='agentenv',
            capabilities=('data_compute', 'text_processing'),
            task_forms=('data_cleaning', ),
            allowed_tools=('shell_executor', 'python_executor', 'read_file', 'write_file', 'glob', 'grep'),
            verifier_kinds=('file_json', 'file_contains', 'file_equals', 'python'),
            input_kinds=('given_text', 'data_fixture'),
            output_kinds=('answer', 'workspace'),
            setup_manifest=('clear_workspace', 'seed-csv-dirty', 'python-numpy-sympy-canary'),
            runtime_ref='twinkle-rsi-msagent',
        ))
    registry.register(
        SandboxProfile(
            name='docs-retrieval',
            version=1,
            session_kind='agentenv',
            capabilities=('document_retrieval', 'citation'),
            task_forms=('retrieval_qa', ),
            allowed_tools=('shell_executor', 'python_executor', 'read_file', 'write_file', 'glob', 'grep'),
            verifier_kinds=('answer_contains', 'answer_equals', 'answer_regex', 'file_contains', 'file_json'),
            input_kinds=('given_text', 'data_fixture'),
            output_kinds=('answer', 'citations'),
            setup_manifest=('clear_workspace', 'seed-docs-corpus', 'python-test-runner-canary'),
            runtime_ref='twinkle-rsi-msagent',
        ))
    registry.register(
        SandboxProfile(
            name='repo-patch',
            version=1,
            session_kind='agentenv',
            capabilities=('python_code', 'workspace_files', 'test_execution'),
            task_forms=('bug_fix', ),
            allowed_tools=('shell_executor', 'python_executor', 'read_file', 'write_file', 'edit_file', 'glob', 'grep'),
            verifier_kinds=('shell', 'python', 'hidden_python', 'file_absent', 'file_contains', 'file_exists'),
            input_kinds=('given_text', 'repository_fixture'),
            output_kinds=('workspace', ),
            setup_manifest=('clear_workspace', 'seed-broken-repo', 'python-test-runner-canary'),
            runtime_ref='twinkle-rsi-msagent',
        ))
    registry.register(
        SandboxProfile(
            name='log-forensics',
            version=1,
            session_kind='agentenv',
            capabilities=('log_analysis', 'shell_forensics'),
            task_forms=('incident_analysis', ),
            allowed_tools=('shell_executor', 'python_executor', 'read_file', 'write_file', 'glob', 'grep'),
            verifier_kinds=('file_json', 'file_contains', 'answer_equals', 'answer_number', 'shell'),
            input_kinds=('given_text', 'data_fixture'),
            output_kinds=('answer', 'workspace'),
            setup_manifest=('clear_workspace', 'seed-log-corpus', 'python-numpy-sympy-canary'),
            runtime_ref='twinkle-rsi-msagent',
        ))
    registry.register(
        SandboxProfile(
            name='world-forensics',
            version=1,
            session_kind='agentenv',
            # The seeded profile that also reads the SHARED world. The distinction
            # matters: sql-analytics copies a database into the workspace, so the model
            # reads a private file. This one reads a dataset it never receives -- the
            # call is answered on the training host, so nothing is copied in and the
            # sandbox still has no network.
            #
            # That is what makes the answer unfakeable. A seeded file can be opened,
            # edited, and reasoned about as local material; a shared one can only be
            # queried, and what comes back is what the world holds.
            capabilities=('database_query', 'document_retrieval', 'log_analysis'),
            task_forms=('incident_analysis', 'cross_source_reconciliation'),
            allowed_tools=('shell_executor', 'python_executor', 'read_file', 'write_file', 'glob', 'grep', 'world_sql',
                           'world_schema', 'world_doc'),
            verifier_kinds=('file_json', 'file_contains', 'answer_equals', 'answer_contains', 'answer_number', 'shell'),
            input_kinds=('given_text', ),
            output_kinds=('answer', 'workspace'),
            # No seed: the environment IS the shared world, and copying it into the
            # workspace would defeat the point. clear_workspace stays so the episode
            # starts from a known-empty directory either way.
            setup_manifest=('clear_workspace', 'python-numpy-sympy-canary'),
            runtime_ref='twinkle-rsi-msagent',
        ))
    registry.register(
        SandboxProfile(
            name='db-analyst',
            version=1,
            session_kind='agentenv',
            # Reads a REAL PostgreSQL service over the network, with an account the
            # service issues to this episode alone. Different from world-forensics,
            # which reads a SQLite file that ships with the run: here the model has to
            # connect to infrastructure and stay inside the grant it was handed, and
            # the grant is enforced by PostgreSQL rather than by the sandbox.
            capabilities=('database_query', 'database_schema'),
            task_forms=('database_analysis', 'grant_aware_query'),
            allowed_tools=('shell_executor', 'python_executor', 'read_file', 'write_file', 'glob', 'grep', 'db_query',
                           'db_schema'),
            verifier_kinds=('sql_query', 'file_json', 'file_contains', 'answer_number', 'answer_equals', 'shell'),
            input_kinds=('given_text', ),
            output_kinds=('answer', 'workspace'),
            # No seed: the environment IS the database service, and copying it in
            # would defeat the point of it being shared.
            setup_manifest=('clear_workspace', 'python-numpy-sympy-canary'),
            runtime_ref='twinkle-rsi-msagent',
        ))
    registry.register(
        SandboxProfile(
            name='web-research',
            version=1,
            session_kind='agentenv',
            # Now agentenv rather than broker: the search tools are answered on the
            # training host, exactly like the world tools, so there is no separate
            # session kind to enable and no sandbox network to open. The broker kind
            # was never implemented (`open_profile_session` rejects it outright), so
            # the profile used to be unavailable for a reason that had nothing to do
            # with search providers -- it was unreachable by construction.
            capabilities=('web_search', 'web_fetch', 'citation'),
            task_forms=('web_research', 'source_verification'),
            # Both search tools, not just `web_search`. An earlier design served
            # them from inside the sandbox, where only one existed, so `fetch_page`
            # was dropped from this list and no profile could reach it. They are
            # answered on the training host now (world.SEARCH_TOOL_SCHEMAS), so
            # both are advertised and both are allowed.
            #
            # `fetch_page` is the half that makes `web_receipt` worth scoring: a
            # search result carries an excerpt, and a policy can quote an excerpt
            # it was shown without ever reading the page. Requiring the source to
            # have been fetched is what separates "cited what it read" from
            # "cited what it was handed".
            allowed_tools=('shell_executor', 'python_executor', 'read_file', 'write_file', 'glob', 'grep', 'web_search',
                           'fetch_page'),
            # web_receipt is the one that matters here: it checks a cited url against
            # what the search tools actually served, which is what `task_contract`
            # already demands of this profile and what nothing implemented until now.
            verifier_kinds=('answer_equals', 'answer_contains', 'answer_number', 'file_json', 'file_contains', 'python',
                            'web_receipt'),
            input_kinds=('given_text', ),
            output_kinds=('answer', 'citations', 'workspace'),
            # No quota or domain policy steps: those belonged to the broker design.
            # The index is frozen and read-only, which is a stronger guarantee than a
            # domain allowlist -- there is nothing else to reach.
            setup_manifest=('clear_workspace', 'python-numpy-sympy-canary'),
            runtime_ref='twinkle-rsi-msagent',
        ))
    registry.register(
        SandboxProfile(
            name='legacy-agentic',
            version=1,
            session_kind='legacy',
            capabilities=('legacy_agentic', ),
            task_forms=('legacy_agentic', ),
            allowed_tools=('shell_executor', 'python_executor', 'read_file', 'write_file', 'edit_file', 'glob', 'grep',
                           'todo'),
            verifier_kinds=('legacy_check_script', ),
            input_kinds=('legacy_task', ),
            output_kinds=('workspace', 'answer'),
            # No manifest steps. This profile describes the environment every task
            # ran in before profiles existed, and resetting it is what the env's
            # own clear() already does -- naming a `legacy-reset` step here would
            # be a step nothing resolves, which clear() refuses at switch time.
            setup_manifest=(),
            runtime_ref='twinkle-rsi-msagent',
            enabled_for_new_tasks=False,
        ))
    return registry


# A profile names tools in the vocabulary of the agent runtime that served them,
# which is finer than what an Env declares: `shell_executor`, `python_executor`,
# `glob` and `grep` are four names there and one `run_command` here. The table is
# data rather than logic so the coarseness is visible in one place instead of
# being spread across conditionals.
#
# The consequence is worth stating plainly: on an Env whose only command tool is
# `run_command`, a profile that allows `glob` allows a shell. Least privilege at
# the granularity these profiles were written for needs an environment that
# declares the finer tools; what this mixin guarantees on a coarse one is that
# nothing outside the profile is reachable at all.
# The one placeholder a seed program may contain. Kept as a literal so the
# substitution below is exact and leaves every other brace alone.
_WORKSPACE_PLACEHOLDER = '{workspace!r}'

TOOL_ALIASES: dict[str, tuple[str, ...]] = {
    'shell_executor': ('run_command', ),
    'python_executor': ('run_command', ),
    'glob': ('run_command', ),
    'grep': ('run_command', ),
    'edit_file': ('write_file', ),
    'read_file': ('read_file', ),
    'write_file': ('write_file', ),
    'run_command': ('run_command', ),
}


def env_names_for(profile: SandboxProfile) -> frozenset:
    """The environment tool names a profile permits, after aliasing."""
    resolved = set()
    for name in profile.allowed_tools:
        resolved.update(TOOL_ALIASES.get(name, (name, )))
    return frozenset(resolved)


class ProfiledEnvMixin:
    """Restrict an environment to what a profile permits, and seed its workspace.

    Mix in front of any :class:`~twinkle_agentic.envs.base.Env`, and ahead of
    :class:`~twinkle_agentic.envs.world.WorldToolsMixin` when both are used, so
    that the world's tools are filtered by the profile too::

        class Slot(ProfiledEnvMixin, WorldToolsMixin, LocalEnv):
            pass

    Three things it does:

    * :meth:`tools` advertises only what the active profile allows, so a policy
      is never offered a tool it would be refused.
    * :meth:`step` and :meth:`step_batch` refuse anything else, because
      advertising is not enforcement -- a policy can name a tool it was not
      shown.
    * :meth:`switch_profile` moves the environment onto another profile and runs
      that profile's setup manifest, which is what puts the fixture in the
      workspace before the episode starts.

    With no profile set the environment behaves exactly as it did before: every
    declared tool is advertised and nothing is refused. That is what makes this
    safe to mix into an environment whose caller has no profiles.
    """

    def __init__(self,
                 *args: Any,
                 profile: SandboxProfile | None = None,
                 profiles: ProfileRegistry | None = None,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._profiles = profiles if profiles is not None else default_profile_registry()
        self._profile = profile
        self._manifest_results: list[tuple[str, int, str]] = []

    @property
    def profile(self) -> SandboxProfile | None:
        """The active profile, or None for an unrestricted environment."""
        return self._profile

    @property
    def profile_id(self) -> str:
        return self._profile.profile_id if self._profile is not None else ''

    @property
    def manifest_results(self) -> list[tuple[str, int, str]]:
        """``(step, exit_code, output)`` for the last manifest that ran."""
        return list(self._manifest_results)

    def switch_profile(self, profile_id: str) -> SandboxProfile:
        """Move onto ``profile_id`` and run its setup manifest.

        Raises rather than degrading: an environment left on the wrong profile
        produces tasks whose fixture is missing, and every one of them reads as a
        hard task rather than as a broken slot.
        """
        profile = self._profiles.get(profile_id)
        if not profile.available:
            raise ValueError(f'profile {profile_id!r} is not available: {profile.unavailable_reason}')
        if not profile.enabled_for_new_tasks:
            raise ValueError(f'profile {profile_id!r} is retired and cannot take new tasks')
        self._profile = profile
        self.clear()
        return profile

    def _allowed(self) -> frozenset | None:
        if self._profile is None:
            return None
        return env_names_for(self._profile)

    def _refusal(self, tool_name: str) -> str | None:
        allowed = self._allowed()
        if allowed is None:
            return None
        short = (tool_name or '').rsplit('---', 1)[-1]
        if short in allowed:
            return None
        return (f'Error: {tool_name!r} is not available in this environment. '
                f'Allowed tools: {sorted(allowed)}.')

    def tools(self) -> list[dict[str, Any]]:
        declared = list(super().tools())
        allowed = self._allowed()
        if allowed is None:
            return declared
        return [t for t in declared if (t.get('function') or {}).get('name') in allowed]

    def step(self, tool_name: str, arguments: dict[str, Any] = None) -> StepResult:
        refusal = self._refusal(tool_name)
        if refusal is not None:
            return StepResult(observation=refusal, reward=0.0, done=False)
        return super().step(tool_name, arguments)

    def step_batch(self, calls: Sequence[tuple[str, dict[str, Any]]]) -> list[StepResult]:
        allowed = self._allowed()
        if allowed is None:
            return super().step_batch(calls)
        # Refusals are answered here and only the permitted calls are forwarded,
        # positionally reassembled, so one bad call in a turn does not cost the
        # whole batch a round trip.
        kept_at: list[int] = []
        out: list[StepResult | None] = [None] * len(calls)
        for index, (name, _) in enumerate(calls):
            refusal = self._refusal(name)
            if refusal is None:
                kept_at.append(index)
            else:
                out[index] = StepResult(observation=refusal, reward=0.0, done=False)
        if kept_at:
            forwarded = super().step_batch([calls[i] for i in kept_at])
            if len(forwarded) != len(kept_at):
                raise RuntimeError(f'step_batch returned {len(forwarded)} results for '
                                   f'{len(kept_at)} forwarded call(s)')
            for index, result in zip(kept_at, forwarded):
                out[index] = result
        return [r if r is not None else StepResult(observation='Error: no result') for r in out]

    def clear(self) -> None:
        """Empty the workspace, then run the active profile's setup manifest.

        Order matters and is the reason this overrides rather than supplements:
        a seed written before the clear is a seed deleted by it.
        """
        super().clear()
        if self._profile is None:
            return
        workspace = self.workspace
        if not workspace:
            raise RuntimeError(f'profile {self._profile.profile_id} has a setup manifest but this '
                               'environment has no workspace to seed into')
        self._manifest_results = []
        for step in self._profile.setup_manifest:
            source = self._manifest_step(step)
            if source is None:
                # Raise rather than record and carry on: a manifest step that
                # does not resolve is a typo in the profile, and continuing
                # would leave the environment missing the fixture every task
                # built on it assumes. Each of those then reads as an unsolvable
                # task instead of as a broken profile.
                raise RuntimeError(f'profile {self._profile.profile_id} names an unknown setup '
                                   f'step {step!r}; known steps are clear_workspace, '
                                   f'{sorted(SEED_PROGRAMS)} and {sorted(LIBRARY_CANARIES)}')
            # The seed programs are templates, not source: each names the
            # workspace it is to build in. Substituted by exact replacement
            # rather than str.format, because the programs are real python and
            # contain set and dict comprehensions whose braces format() would
            # read as replacement fields.
            exit_code, output = self.run_script(source.replace(_WORKSPACE_PLACEHOLDER, repr(workspace)), 'python')
            self._manifest_results.append((step, exit_code, output))
            if exit_code != 0:
                raise RuntimeError(f'profile {self._profile.profile_id} setup step {step!r} '
                                   f'failed with exit {exit_code}: {output[-300:]}')

    @staticmethod
    def _manifest_step(step: str) -> str | None:
        if step == 'clear_workspace':
            return CLEAR_WORKSPACE
        if step in SEED_PROGRAMS:
            return SEED_PROGRAMS[step]
        if step in LIBRARY_CANARIES:
            return LIBRARY_CANARIES[step]
        return None
