# Copyright (c) ModelScope Contributors. All rights reserved.
"""Building the environment each concurrent job runs in.

``rsi_grpo.py`` used to construct ``AgentEnv`` or ``LocalEnv`` directly. That is
still exactly what it gets when nothing below is configured: the classes built
here are those same classes, or those classes with mixins in front.

Three things can be added, independently:

``TWINKLE_WORLD_ROOT``
    Points at a host-served read-only world: a SQLite database, a document set,
    and optionally a search corpus. Its tools are answered on the training host
    rather than in the sandbox, so an environment with no network keeps that
    property and the data never enters a snapshot. See
    :mod:`twinkle_agentic.envs.world`.

``TWINKLE_RSI_PROFILES``
    A comma-separated list of profile ids to rotate the slots through, e.g.
    ``sql-analytics@1,docs-retrieval@1``. Each slot is switched onto its profile
    before it is handed out, which runs that profile's setup manifest -- the
    seed program that builds the fixture the tasks will be graded against. Without
    a profile every slot boots unrestricted and a task about a database has no
    database. See :mod:`twinkle_agentic.envs.profiles`.

``TWINKLE_TOPIC_AFFINITIES``
    Read only when the list above is empty. Points at the table saying which
    profiles each knowledge card is eligible for; the rotation then becomes every
    profile this machine can actually run, best first. This is what drops a
    profile whose backing service is absent here, so a run moved to a host without
    that service degrades to the sandbox-only profile instead of scoring zero on a
    quarter of its groups. See :mod:`twinkle_agentic.envs.topic_profile`.

Order of the mixins matters and is the reason this is a module rather than two
lines inline: ``ProfiledEnvMixin`` has to precede ``WorldToolsMixin`` so the
world contributes its tools first and the profile then decides which survive.
Reversed, a profile could not restrict the world's tools at all.
"""
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

from twinkle_agentic.envs import (AgentEnv, Env, HostWorld, LocalEnv, ProfiledEnvMixin, TopicProfiles,
                                  WorldToolsMixin, default_profile_registry)

__all__ = ['build_slot_envs', 'configured_profiles', 'configured_world']


def configured_world() -> Optional[HostWorld]:
    """The host world this run was pointed at, or None.

    Loaded here rather than per slot: the world is read-only and shared, and
    loading it once is also what makes a bank that fails its manifest check fail
    once, at startup, instead of on the first slot that touches it.
    """
    root = os.environ.get('TWINKLE_WORLD_ROOT', '')
    if not root:
        return None
    world = HostWorld(root)
    world.load()
    return world


def configured_profiles() -> List[str]:
    """The profile ids to rotate the slots through, best first. Empty means none.

    An explicit list wins over the affinity table: naming profiles is how a run
    pins what it measured, and a run that pinned them should not have the list
    change underneath it because a service happened to be down. The table is the
    default for a run that did not decide, because "whatever works here" is still
    a better answer than "unrestricted".
    """
    explicit = os.environ.get('TWINKLE_RSI_PROFILES', '')
    ids = [part.strip() for part in explicit.split(',') if part.strip()]
    if ids:
        return ids
    if os.environ.get('TWINKLE_TOPIC_AFFINITIES', ''):
        return list(TopicProfiles().rotation())
    return []


def _slot_type(index: int, bases: Tuple[type, ...], env_class: Type[Env]) -> Type[Env]:
    """The environment class for one slot, composed from what this run asked for.

    ``env_class`` itself when there is nothing to mix in, so an unconfigured run
    builds the class it built before rather than an empty subclass of it.

    Built per call rather than cached because the bases differ by configuration,
    and a cached class from a differently-configured call would be the kind of
    mistake that shows up as a profile that silently does not restrict anything.
    """
    if not bases:
        return env_class
    return type(f'RsiSlot{index}', bases + (env_class, ), {})


def build_slot_envs(*,
                    num_envs: int,
                    sandbox_template: str = '',
                    sandbox_api_url: Optional[str] = None,
                    sandbox_timeout: int = 300,
                    workspace_root: str = '/tmp/twinkle_rsi_slots',
                    pre_tool_call: Any = None,
                    world: Optional[HostWorld] = None,
                    profile_ids: Sequence[str] = (),
                    command_timeout: int = 120) -> List[Env]:
    """One environment per concurrent job, optionally world-backed and profiled.

    Args:
        num_envs: how many jobs run at once; one environment each, because a job
            that acts in an environment is also verified in it and must not share.
        sandbox_template: an AgentENV template. Empty means local directories,
            which has no isolation -- fine for a check that is a few asserts,
            wrong for training a policy to run commands it wrote itself.
        world: a loaded :class:`HostWorld`, or None for no world tools.
        profile_ids: profiles to rotate the slots through. Empty means no
            profile, which leaves every tool the environment declares available.

    Returns:
        The environments, in slot order. A slot assigned a profile has already
        been switched onto it, so its fixture exists before the first job.

    Raises:
        ValueError: if a profile id is unknown, unavailable or retired. Raised
            here rather than on the first episode, so a run with a typo in its
            profile list never starts.
    """
    registry = default_profile_registry() if profile_ids else None
    bases: Tuple[type, ...] = ()
    if registry is not None:
        bases += (ProfiledEnvMixin, )
    if world is not None:
        bases += (WorldToolsMixin, )

    envs: List[Env] = []
    for index in range(num_envs):
        kwargs: Dict[str, Any] = {}
        if world is not None:
            # One world object shared by every slot: it is read-only, and its
            # receipts are per episode, so sharing the data is safe while sharing
            # a receipt log would not be. Each slot gets its own view.
            kwargs['world'] = world
            # Released by the last slot to close, so a pool torn down in any order
            # drops its per-episode database role exactly once.
            kwargs['release_world_on_close'] = index == num_envs - 1
        if registry is not None:
            kwargs['profiles'] = registry

        if sandbox_template:
            agent_kwargs: Dict[str, Any] = {
                'template': sandbox_template,
                'api_url': sandbox_api_url or None,
                'sandbox_timeout': sandbox_timeout,
                'command_timeout': command_timeout,
                'metadata': {
                    'run': 'rsi_grpo',
                    'slot': str(index)
                },
            }
            if pre_tool_call is not None:
                agent_kwargs['pre_tool_call'] = pre_tool_call
            env_class: Type[Env] = AgentEnv
            init_kwargs: Dict[str, Any] = agent_kwargs
        else:
            env_class = LocalEnv
            init_kwargs = {
                'workspace': os.path.join(workspace_root, f'slot_{index}'),
                'command_timeout': command_timeout,
            }
        env = _slot_type(index, bases, env_class)(**init_kwargs, **kwargs)

        if registry is not None:
            # Rotate rather than assign one profile to every slot: a run whose
            # slots all hold the same fixture can only produce one kind of task,
            # and the policy learns that the fixture is part of the world rather
            # than part of the question.
            env.switch_profile(profile_ids[index % len(profile_ids)])
        envs.append(env)
    return envs
