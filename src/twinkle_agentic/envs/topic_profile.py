# Copyright (c) ModelScope Contributors. All rights reserved.
"""Pick the sandbox profile a topic calls for.

WHY THIS EXISTS
---------------
The sandbox pool used to boot one kind of environment -- `legacy-agentic`, with
every tool and no profile -- regardless of what the topic was about. A topic on
quantum coherence and a topic on office-chair ergonomics both landed in the same
empty workspace, so the proposer could only ever write tasks made of files it
created itself.

A :class:`~twinkle_agentic.envs.profiles.SandboxProfile` registry says which
profiles *exist*. This module answers the other question -- which one a given
topic should run in -- from a table an offline classifier writes. The table is
JSON Lines, one record per knowledge card::

    {"card_id": "...", "topic_path": {"root": "...", "leaf": "..."},
     "eligible_profiles": ["world-forensics@1", "sql-analytics@1"],
     "advisory_only": true}

Nothing in this repository builds it; it is a run's own artefact, pointed at by
``TWINKLE_TOPIC_AFFINITIES``. This module is deliberately small: read the table,
look up a card, return a choice.

WHAT "A CHOICE" MEANS HERE
--------------------------
A topic usually maps to several profiles, and one group can only use one sandbox
at a time. The order below prefers the profile that gives the proposer the most to
work with: a task that can be computed is more verifiable than one that can only
be asserted, so compute beats code beats retrieval beats prose. `knowledge-answer`
is last because it is the baseline every topic carries -- choosing it means the
topic offered nothing more specific.

The table is advisory by construction (`advisory_only: True` in each record). A
missing table, a missing card, or an unknown profile name all fall back to the
baseline rather than failing the run: a training job that has already loaded its
weights should not die because a lookup missed.
"""
from __future__ import annotations

import json
import logging
import os

from . import world
from .profiles import SandboxProfile, default_profile_registry

logger = logging.getLogger(__name__)

# Most specific first. `knowledge-answer` is the baseline every card carries, so
# it sorts last and is also the fallback.
#
# SWITCHABLE is derived from this tuple, so a profile absent from here is not
# merely deprioritised -- it is unreachable, and _load() drops it from every
# card's candidate list before anything gets to choose it. That is what happened
# to the profiles added to the registry after this list was written: regenerating
# the affinity table so that it names db-analyst on 91 cards and docs-retrieval on
# 88 changes nothing at runtime until they are listed here too.
#
# Order within the list, and why:
#   1. A capability the card was mined for beats a generic one. A card labelled
#      `domain: sql` also matches the `data` regexes ("...Group-Level Analysis"),
#      so python-compute is eligible for it too; ranking compute first routes
#      every database card into a plain python slot. That is how a bank built to
#      be a third database and a third web search produced 89.5% "write a CSV"
#      tasks and no query tasks at all.
#   2. A profile backed by a real service beats one that simulates it inside the
#      sandbox. db-analyst queries PostgreSQL through `world` with a
#      per-episode provisioned account; sql-analytics answers the same
#      database_analysis form with sqlite through python_executor. Both stay
#      eligible, so the sandbox-only one remains the fallback when the service is
#      not provisioned, and `_service_reachable` below settles that once at load
#      rather than per group.
#   3. lean-proof stays first by the existing convention: it is the most specific
#      form, and UNAVAILABLE below is what keeps it from being chosen until its
#      runtime exists.
PREFERENCE: tuple = (
    'lean-proof@1',
    'world-forensics@1',
    'db-analyst@1',
    'web-research@1',
    'sql-analytics@1',
    'docs-retrieval@1',
    'log-forensics@1',
    'repo-patch@1',
    'data-wrangling@1',
    'python-compute@1',
    'code-python@1',
    'knowledge-answer@1',
)

BASELINE = 'knowledge-answer@1'

# Profiles a slot can actually be switched to.
#
# `knowledge-answer` is not here even though it is the baseline every card
# carries: its session kind is `no_tool`, so it is a *statement that the topic
# needs no tools*, not a sandbox a slot can move into. A card whose only entry is
# the baseline therefore stays on whatever the slot already has -- which is the
# intent, since the classifier is saying the topic offers nothing more specific.
#
# `lean-proof` is left out because its runtime has never been built; switching to
# it would fail on every card that carries it. It goes back in the moment the
# image exists, and the entry in PREFERENCE is already in the right place.
SWITCHABLE = frozenset(p for p in PREFERENCE if p not in ('knowledge-answer@1', 'lean-proof@1'))

# Profiles named here but not switchable, kept visible so the reason is in one
# place rather than spread across the callers.
UNAVAILABLE = (
    ('knowledge-answer@1', 'session kind is no_tool: not a sandbox to switch into'),
    ('lean-proof@1', 'runtime lean4-mathlib-pinned has not been built'),
)

# Where a task's data already is when the episode starts. A profile's workspace is
# not empty when one of the `seeds` programs plants data in it, and its data is not
# in the workspace at all when the tools that reach it are answered on the training
# host. Which of the three applies is a property of the profile, so it is derived
# here from the registry rather than guessed per call site -- the same fact is
# needed both when a task author is told what it is building in and when the
# problem statement is written.
START_HOST = 'host'
START_SEEDED = 'seeded'
START_EMPTY = 'empty'


def _profile_or_none(profile_id: str | None, registry=None) -> SandboxProfile | None:
    """The named profile, or None when the id is empty or is not registered.

    ``ProfileRegistry.get`` raises rather than returning None. Every caller below
    is on the advisory path, where an unknown name has to degrade instead of
    stopping a run that has already loaded its weights -- so the translation lives
    here once. Catching it at three call sites is how they came to catch
    ``KeyError``, which ``get`` does not raise, and the documented fallback was
    therefore unreachable.
    """
    if not profile_id:
        return None
    reg = registry or default_profile_registry()
    try:
        return reg.get(profile_id)
    except ValueError:
        return None


def starting_state(profile_id: str | None, registry=None) -> str:
    """Which of host / seeded / empty describes this profile's starting state.

    Host wins over seeded: a profile whose tools are answered outside the sandbox
    gets its data from there even if a seed also ran, and telling the author the
    data is local would send it looking in the workspace.
    """
    profile = _profile_or_none(profile_id, registry)
    if profile is None:
        return START_EMPTY
    if set(profile.allowed_tools or ()) & set(world.TOOL_NAMES):
        return START_HOST
    if any(str(step).startswith('seed-') for step in (profile.setup_manifest or ())):
        return START_SEEDED
    return START_EMPTY


def host_tools_of(profile_id: str | None, registry=None) -> tuple[str, ...]:
    """The host-answered tools a profile grants, in the registry's own order."""
    profile = _profile_or_none(profile_id, registry)
    if profile is None:
        return ()
    return tuple(name for name in (profile.allowed_tools or ()) if name in world.TOOL_NAMES)


def seed_steps_of(profile_id: str | None, registry=None) -> tuple[str, ...]:
    """The `seeds` programs a profile runs before the agent sees anything."""
    profile = _profile_or_none(profile_id, registry)
    if profile is None:
        return ()
    return tuple(step for step in (profile.setup_manifest or ()) if str(step).startswith('seed-'))


# Profiles whose tools are answered by a service that may not exist on whichever
# machine is running the trainer. db-analyst's db_query/db_schema go through
# `world` to a PostgreSQL server at TWINKLE_WORLD_DB_HOST, which defaults to
# 127.0.0.1 -- so "is it there" depends entirely on which host the trainer landed
# on. Nothing crashes when it is missing: world.db_query catches the connection
# failure and returns "cannot reach the shared database" as an observation. But
# then every query fails, every rollout scores 0, and the group lands out of band
# and is dropped. Routing 91 of 332 switchable cards there would spend roughly a
# quarter of the groups and teach the proposer that database topics do not pay,
# which is the opposite of what the card bank was built for.
#
# Probing once at load and dropping the profile makes the router fall through to
# sql-analytics, which answers the same database_analysis task form with sqlite
# inside the sandbox and needs no service at all. Fail closed: an unreachable
# service is treated as unavailable rather than assumed present.
_SERVICE_BACKED = frozenset({'db-analyst@1'})


def _service_reachable(profile_id: str, timeout: float = 2.0) -> bool:
    """Whether the service behind a profile's host tools accepts a connection.

    A bare TCP connect, not a query: provisioning a real role just to test
    reachability would create and leak a database account per import, and the
    distinction this has to make is "is there a server here" and nothing finer.
    """
    if profile_id not in _SERVICE_BACKED:
        return True
    if profile_id == 'db-analyst@1':
        # The same two variables `world` connects with. Reading a different pair
        # here would probe one host and then query another.
        host = os.environ.get('TWINKLE_WORLD_DB_HOST', '127.0.0.1')
        port = int(os.environ.get('TWINKLE_WORLD_DB_PORT', '5432'))
    else:
        return True
    import socket
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


class TopicProfiles:
    """A card id -> eligible profile list lookup, loaded from the affinity table."""

    def __init__(self, path: str | None = None) -> None:
        self.path = path or os.environ.get('TWINKLE_TOPIC_AFFINITIES', '')
        self.by_card: dict[str, list[str]] = {}
        self.by_topic: dict[str, list[str]] = {}
        self.loaded = False
        # Computed before _load, which filters candidates through it.
        self.unreachable = tuple(sorted(p for p in SWITCHABLE if not _service_reachable(p)))
        self.switchable = SWITCHABLE - set(self.unreachable)
        if self.unreachable:
            logger.warning(
                '[topic-profile] service unreachable, not switchable here: %s '
                '(their cards fall through to the next eligible profile)', ', '.join(self.unreachable))
        self._load()

    def _load(self) -> None:
        if not self.path:
            return
        if not os.path.isfile(self.path):
            # Not "every topic uses the baseline": for_card returns None, so every
            # slot stays on whatever it was built with. Saying otherwise in the log
            # would send whoever reads it looking for a switch that never happens.
            logger.warning(
                '[topic-profile] affinity table missing at %s; '
                'every card will stay on its slot\'s own profile', self.path)
            return
        n = 0
        with open(self.path, encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                chosen = [p for p in record.get('eligible_profiles', ()) if p in self.switchable]
                # Kept even when empty. `for_card` answers None for it either way,
                # but describe() counts what is here, and "N cards offered nothing
                # this machine can run" is the number a run's log most needs --
                # dropping the record hides exactly the case worth noticing.
                card_id = record.get('card_id')
                if card_id:
                    self.by_card[str(card_id)] = chosen
                topic = record.get('topic_path') or {}
                if topic.get('root') and topic.get('leaf'):
                    self.by_topic[f"{topic['root']} > {topic['leaf']}"] = chosen
                n += 1
        self.loaded = True
        logger.info('[topic-profile] loaded %d affinity record(s) from %s: %s', n, self.path, self.describe())

    def rotation(self) -> list[str]:
        """Every profile this machine can actually run, best first.

        What a slot pool rotates through when a run configured an affinity table
        but no explicit profile list: PREFERENCE order, minus the profiles whose
        backing service is not reachable here. Derived from the same two sources
        ``for_card`` uses rather than written out again, so registering a profile
        and listing it in PREFERENCE is enough -- a rotation kept in a second
        place is a rotation that silently stops including new profiles.
        """
        return [profile for profile in PREFERENCE if profile in self.switchable]

    def for_card(self, card) -> str | None:
        """Which profile this card should move its slot to, or None to stay put.

        `None` is the answer in two cases, and both mean the same thing to the
        caller -- do not switch:

          * the card carries only the baseline, so the classifier is saying the
            topic needs no tools;
          * the card names a profile that cannot be switched into yet (see
            UNAVAILABLE), so switching would fail.

        Returning the baseline's name instead would send the caller into a switch
        that cannot succeed, which is the failure this return type exists to
        prevent.
        """
        candidates: list[str] = []
        card_id = getattr(card, 'card_id', None)
        if card_id and str(card_id) in self.by_card:
            candidates = self.by_card[str(card_id)]
        if not candidates:
            cue = getattr(card, 'source_cue', None)
            if cue and cue in self.by_topic:
                candidates = self.by_topic[cue]
        for profile in PREFERENCE:
            if profile in candidates and profile in self.switchable:
                return profile
        return None

    def describe(self) -> dict[str, int]:
        """How many cards route to each profile, for the run's log.

        ``stay-on-slot`` counts the cards that offered nothing switchable -- only
        the baseline, or a profile this machine cannot run. Reported rather than
        omitted: it is the number that says an affinity table has stopped matching
        the registry.
        """
        counts: dict[str, int] = {}
        for profiles in self.by_card.values():
            chosen = next((p for p in PREFERENCE if p in profiles and p in self.switchable), 'stay-on-slot')
            counts[chosen] = counts.get(chosen, 0) + 1
        return counts
