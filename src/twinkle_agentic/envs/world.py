# Copyright (c) ModelScope Contributors. All rights reserved.
"""A read-only world served from the training host, not from the sandbox.

An agentic RSI run needs tasks whose answers the policy cannot invent. A task
that says "report the failure rate in the ops database" is only a real task if
the database exists and the policy has to read it; otherwise the policy writes
plausible numbers and a check that compares strings passes either way.

This module is that world: a SQLite database, a document set, and a search
corpus, exposed as tools the model calls exactly like the sandbox's own.

Why host-side
-------------
A tool call already round-trips. :meth:`Env.step_batch` hands the call to the
environment and brings the answer back, so the training host sees every call
before and after the sandbox does. A tool that lives on the host therefore needs
no network path *into* the sandbox, which is what makes this the right place:

* no egress from the sandbox. An environment configured with no network keeps
  that property: a model that could reach the host's network could scan it, a
  model that can only call a named tool cannot;
* no reverse tunnel and no new listening port to defend;
* the data never enters the sandbox, so it cannot be edited, copied out of
  scope, or left behind in a snapshot.

The cost is one extra hop, and it is not on the hot path: a tool call is already
a sandbox round trip, and this one answers from a local read-only handle.

What the model sees
-------------------
Seven tools, in three groups, advertised like any other:

``world_sql`` / ``world_schema`` / ``world_doc``
    read-only SQL over ``ops.db``, its schema, and the document set.
``web_search`` / ``fetch_page``
    retrieval over a frozen corpus, or a live engine if one is configured.
``db_query`` / ``db_schema``
    a real SQL server, on a per-episode role rather than a shared login.

Every retrieval answer is recorded as a *receipt* (:attr:`HostWorld.receipts`).
That is what makes a citation checkable: ``web_receipt`` in
:mod:`twinkle_agentic.verifier.result_check` can ask "was this url actually
served to this episode?" and get an answer from what happened rather than from
what the policy wrote down.

Configuration is by environment variable, because a library does not guess where
its data lives:

=============================== ============================================
``TWINKLE_WORLD_ROOT``          directory holding ``manifest.json``, ``ops.db``
                                and ``docs/``. Required; without it
                                :attr:`HostWorld.available` is False.
``TWINKLE_WORLD_SEARCH_ROOT``   directory holding ``corpus.json`` and its
                                ``manifest.json``. Optional; without it the two
                                search tools report themselves unavailable.
``TWINKLE_WORLD_SEARCH_BACKEND````corpus`` (default) or ``live``.
``TWINKLE_WORLD_SEARCH_MODULE`` directory holding ``search_backend.py``.
                                Required when the backend is ``live``.
``TWINKLE_WORLD_DB_MODULE``     directory holding the ``provision`` module that
                                mints and drops per-episode database roles.
                                Required only when ``db_enabled``.
``TWINKLE_WORLD_DB_HOST``       database host, default ``127.0.0.1``.
``TWINKLE_WORLD_DB_PORT``       database port, default ``5432``.
``TWINKLE_WORLD_DB_LEASE``      role lease seconds, default ``21600``.
=============================== ============================================

A live search backend is a plug-in rather than a dependency: the world imports
``search_backend`` from ``TWINKLE_WORLD_SEARCH_MODULE`` and calls
``Backend().web_search(query, num_results, freshness)`` and
``Backend().fetch_page(url, query)``, both returning the observation string the
tool shows the model. Which provider sits behind that class is the deployment's
choice, and nothing in this repository names one.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Sequence

from twinkle.data_format.message import Tool as ToolInfo
from .base import StepResult

MAX_ROWS = 5000

# Deliberately the same crude overlap the corpus builder uses. It is a stand-in
# for a ranker, not a ranker: a good one would hide exactly the failures the
# corpus exists to reproduce.
_WORD = re.compile(r'[a-z0-9]+')
_STOP = {
    'the', 'a', 'an', 'of', 'in', 'is', 'are', 'and', 'or', 'to', 'for', 'on', 'by', 'with', 'as', 'at', 'it', 'its',
    'that', 'this', 'be', 'from'
}


def _terms(text: str) -> set:
    return {word for word in _WORD.findall((text or '').lower()) if word not in _STOP and len(word) > 2}


# The schemas are the contract the model is shown. Written here rather than read
# from the world service because the host side must be able to advertise a tool
# before the service is up -- and a tool that disappears from the prompt when the
# service is down would make an outage look like a different task.
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        'type': 'function',
        'function': {
            'name':
            'world_sql',
            'description': ('Run a read-only SQL query against the shared operations database. '
                            'Tables: regions, sites, machines, incidents, procedures. The '
                            'database is shared and read-only; writes are rejected. Use this to '
                            'obtain the real values instead of assuming them.'),
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {
                        'type': 'string',
                        'description': 'A single SELECT statement.'
                    },
                    'params': {
                        'type': 'array',
                        'description': 'Optional bound parameters for ? placeholders.',
                        'items': {}
                    },
                },
                'required': ['query'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name':
            'world_schema',
            'description': ('Describe the shared database: call with no name to list the tables, '
                            'or with a table name to get its columns and types. Cheaper and more '
                            'reliable than guessing column names.'),
            'parameters': {
                'type': 'object',
                'properties': {
                    'name': {
                        'type': 'string',
                        'description': 'Optional table name.'
                    }
                },
                'required': [],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name':
            'world_doc',
            'description': ('Read one document from the shared procedures corpus. Call with no '
                            'name to list what is available. The documents state the rules the '
                            'database alone does not, such as how an incident should have been '
                            'classified.'),
            'parameters': {
                'type': 'object',
                'properties': {
                    'name': {
                        'type': 'string',
                        'description': 'Document name, e.g. disk.md.'
                    }
                },
                'required': [],
            },
        },
    },
]

# Search tools. Named and shaped after ms-agent's own web_search/fetch_page so
# swapping the frozen corpus for a live engine is a change of one implementation
# and nothing else -- no task, no profile, no schema moves.
#
# The ranking is not trustworthy and the description says so. An engine that
# returned only good results would train a policy that believes the first hit;
# a task can then be written to require verification, and the policy meets the
# unreliability where it belongs -- in the episode, not in production.
SEARCH_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        'type': 'function',
        'function': {
            'name':
            'web_search',
            'description': ('Search the shared document index. Returns ranked results with a url, '
                            'title, publication date and short quoted excerpts. The ranking is NOT '
                            'reliable: a result can be on-topic and out of date, and the top hit is '
                            'sometimes superseded. Fetch the url before trusting its content.'),
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {
                        'type': 'string',
                        'description': 'Search query.'
                    },
                    'num_results': {
                        'type': 'integer',
                        'description': 'How many results to return (1-25).'
                    },
                    'freshness': {
                        'type': 'string',
                        'description': "Optional 'YYYY-MM-DD'; results on or after this date first."
                    },
                },
                'required': ['query'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name':
            'fetch_page',
            'description': ('Fetch the full text of a url returned by web_search. Use this on every '
                            'result whose content matters: search excerpts are shortened and are not '
                            'the document. A long page is returned in part, not whole; pass `query` '
                            'to get the part that bears on what you are looking for.'),
            'parameters': {
                'type': 'object',
                'properties': {
                    'url': {
                        'type': 'string',
                        'description': 'A url from web_search.'
                    },
                    # Optional and not required: a policy that does not know what
                    # it is looking for should still be able to read the page.
                    'query': {
                        'type':
                        'string',
                        'description':
                        'Optional. What you are looking for on '
                        'the page; the returned text is the part '
                        'that mentions it.'
                    },
                },
                'required': ['url'],
            },
        },
    },
]

# The shared PostgreSQL service. Separate from the world_* tools above: those
# read a SQLite file that ships with the run, these talk to a real server over
# TCP with a per-sandbox account. Both exist because they are different exercises
# -- reading a file you were handed, versus connecting to infrastructure and
# working within the grant you were given.
DB_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        'type': 'function',
        'function': {
            'name':
            'db_query',
            'description': ('Run a read-only SQL query against the shared operations database '
                            '(PostgreSQL). Your account can SELECT and create temporary tables; '
                            'it cannot change anything. Tables: regions, sites, machines, '
                            'incidents, procedures, incident_notes.'),
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {
                        'type': 'string',
                        'description': 'One SQL statement (SELECT).'
                    },
                    'params': {
                        'type': 'array',
                        'description': 'Optional bound parameters for %s placeholders.',
                        'items': {}
                    },
                },
                'required': ['query'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name':
            'db_schema',
            'description': ('List the tables in the shared database, or describe one table. '
                            'Cheaper and more reliable than guessing column names.'),
            'parameters': {
                'type': 'object',
                'properties': {
                    'table': {
                        'type': 'string',
                        'description': 'Optional table name.'
                    }
                },
                'required': [],
            },
        },
    },
]

TOOL_NAMES = tuple(schema['function']['name'] for schema in TOOL_SCHEMAS + SEARCH_TOOL_SCHEMAS + DB_TOOL_SCHEMAS)


def _world_name(tool_name: str | None) -> str | None:
    """The world tool this call names, or None if it is not one of ours.

    Strips the ``---<id>`` suffix a harness adds to disambiguate same-named
    tools, so a suffixed call still resolves.
    """
    if not tool_name:
        return None
    short = tool_name.rsplit('---', 1)[-1]
    return short if short in TOOL_NAMES else None


class WorldUnavailable(RuntimeError):
    """The world is not configured for this run."""


class HostWorld:
    """Read-only access to the built world, straight off the filesystem.

    Deliberately does NOT go through world_service.py. The service exists for the
    case where something other than this process needs to read the world; the host
    already has the files, so routing an in-process call through HTTP would add a
    port to secure and a failure mode to handle for nothing. Both read the same
    database with the same read-only handle, so a query behaves identically either
    way.
    """

    def __init__(self, root: str | None = None):
        # A library does not guess where its data lives. Without an explicit root
        # the world reports itself unavailable rather than reading whatever
        # happens to sit at a path baked in at import time.
        configured = root or os.environ.get('TWINKLE_WORLD_ROOT', '')
        self.root: Path | None = Path(configured) if configured else None
        self.manifest: dict[str, Any] = {}
        self.db_path = self.root / 'ops.db' if self.root is not None else None
        self.docs: dict[str, Path] = {}
        # The search corpus is optional: a run that only uses world_* tools should
        # not fail because the search index has not been built.
        self.search_root: Path | None = None
        self.search_documents: list[dict[str, Any]] = []
        # The episode's own database account, created on first use and dropped on
        # close. `db_enabled` is set from the profile's tool list, so a run that
        # never names db_* never touches the service at all.
        self.db_enabled = False
        self._db_creds: dict[str, Any] | None = None
        # Which engine answers web_search/fetch_page. The corpus is the default
        # and stays the default: it is deterministic, so a run built on it
        # replays, and most task development happens against it. A live engine
        # is opted into per run because its answers change by the hour.
        self.backend = 'corpus'
        self.live = None
        # The retrieval calls this world actually served, in order -- `web_search`
        # and `fetch_page` only, which is what `_receipt` records. The point is
        # that a citation check can ask "did you really retrieve this?" and get an
        # answer from what happened rather than from what the policy wrote down. A
        # policy that invents a url has nothing here to match against. The SQL and
        # document tools leave no receipt: they return values, not sources, so
        # there is no citation to check them against.
        self.receipts: list[dict[str, Any]] = []

    def _load_search(self) -> None:
        # Only the explicit env var is honoured. Walking up from __file__ to find
        # a corpus was tried and dropped: how many parents to count depends on
        # where a run checked the tree out, and getting it wrong silently reads
        # somebody else's index.
        self.backend = (os.environ.get('TWINKLE_WORLD_SEARCH_BACKEND') or 'corpus').strip().lower()
        if self.backend != 'corpus':
            self._load_live_search()
            return
        # No corpus root means no search tools, which is a supported
        # configuration: a run that only reads the database should not have to
        # build a search index it will never query.
        search_root = os.environ.get('TWINKLE_WORLD_SEARCH_ROOT', '')
        if not search_root:
            return
        root = Path(search_root)
        corpus = root / 'corpus.json'
        manifest = root / 'manifest.json'
        if not corpus.is_file() or not manifest.is_file():
            return
        expected = json.loads(manifest.read_text(encoding='utf-8'))['files']['corpus.json']
        import hashlib
        actual = hashlib.sha256(corpus.read_bytes()).hexdigest()
        if actual != expected:
            raise WorldUnavailable(f'search corpus does not match its manifest: expected {expected}, '
                                   f'got {actual}. Rebuild the corpus and its manifest together -- '
                                   'a drifted corpus would silently change what every search task '
                                   'retrieves, which is the same run producing a different score.')
        self.search_documents = json.loads(corpus.read_text(encoding='utf-8'))['documents']
        self.search_root = root

    def _load_live_search(self) -> None:
        """Attach a live engine.

        Imported lazily so a corpus-only run does not depend on the plug-in
        existing, and so a broken live backend cannot take down a run that never
        asked for it. The contract is two methods returning observation strings;
        see the module docstring.
        """
        module_dir = os.environ.get('TWINKLE_WORLD_SEARCH_MODULE', '')
        if not module_dir:
            raise WorldUnavailable('TWINKLE_WORLD_SEARCH_BACKEND is set to '
                                   f'{self.backend!r} but TWINKLE_WORLD_SEARCH_MODULE is not: a live '
                                   'engine is a deployment choice, so the directory holding its '
                                   'backend module has to be named rather than assumed.')
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)
        import search_backend
        self.live = search_backend.Backend()

    @property
    def available(self) -> bool:
        if self.root is None or self.db_path is None:
            return False
        return self.db_path.is_file() and bool(self.docs or (self.root / 'docs').is_dir())

    def schemas(self) -> list[ToolInfo]:
        """The tools this world can actually answer right now.

        Advertising a tool that always answers "not configured" is worse than not
        advertising it: the policy spends a turn discovering that, and a
        difficulty measurement built on the episode counts the wasted turn as
        part of the task. So each group is offered only when its backing store
        is present.
        """
        out: list[ToolInfo] = []
        if self.available:
            out.extend(TOOL_SCHEMAS)
        if self.search_documents or self.live is not None:
            out.extend(SEARCH_TOOL_SCHEMAS)
        if self.db_enabled:
            out.extend(DB_TOOL_SCHEMAS)
        return out

    def load(self) -> None:
        """Read the manifest and verify the files still match it.

        Checked once per process at startup. The world is read-only, so a match
        now is a match for the run; hashing a 100 KB database per query would be
        paid on every tool call for no added safety.
        """
        if self.root is None:
            raise WorldUnavailable('no world root configured: pass root=... to HostWorld or set '
                                   'TWINKLE_WORLD_ROOT to the directory holding manifest.json, ops.db '
                                   'and docs/')
        manifest_path = self.root / 'manifest.json'
        if not manifest_path.is_file():
            raise WorldUnavailable(f'no world manifest at {manifest_path}')
        import hashlib
        self.manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        for name, expected in sorted(self.manifest.get('files', {}).items()):
            path = self.root / name
            if not path.is_file():
                raise WorldUnavailable(f'world file missing: {path}')
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                raise WorldUnavailable(f'{name} does not match the manifest\n  expected {expected}\n'
                                       f'  actual   {actual}\nRebuild with build_world.py. Serving a world '
                                       'that drifted would silently change every task built on it.')
        docs_dir = self.root / 'docs'
        if docs_dir.is_dir():
            self.docs = {path.name: path for path in sorted(docs_dir.glob('*.md'))}
        self._load_search()

    def _connect(self) -> sqlite3.Connection:
        # mode=ro is the enforcement, exactly as in world_service.py: a property of
        # the handle rather than a filter over SQL text, so there is no keyword list
        # to get clever around.
        return sqlite3.connect(f'file:{self.db_path}?mode=ro', uri=True)

    def sql(self, query: str, params: Any) -> str:
        if not isinstance(query, str) or not query.strip():
            return 'world_sql: `query` must be a non-empty string.'
        if not isinstance(params, (list, tuple)):
            return 'world_sql: `params` must be a list.'
        head = query.strip().lstrip('(').split(None, 1)[0].upper()
        # Refused as a class. Measured: `PRAGMA journal_mode=WAL` and
        # `PRAGMA user_version=99` are rejected by the read-only handle, but
        # `PRAGMA journal_mode=DELETE` returns a row and reads like success -- a
        # no-op only because the file is already in delete mode, which is a
        # coincidence. None of them are needed to answer a question about the data.
        if head == 'PRAGMA':
            return 'world_sql: PRAGMA is not available on the shared world.'
        try:
            conn = self._connect()
        except sqlite3.Error as error:
            return f'world_sql error: database unavailable: {error}'
        try:
            cursor = conn.execute(query, list(params))
            if cursor.description is None:
                return 'world_sql: statement produced no result set.'
            columns = [c[0] for c in cursor.description]
            rows = cursor.fetchmany(MAX_ROWS + 1)
            truncated = len(rows) > MAX_ROWS
            rows = rows[:MAX_ROWS]
        except sqlite3.Error as error:
            return f'world_sql error: {type(error).__name__}: {error}'
        finally:
            conn.close()
        lines = [' | '.join(str(c) for c in columns), '-+-'.join('-' * len(str(c)) for c in columns)]
        for row in rows:
            lines.append(' | '.join('NULL' if v is None else str(v) for v in row))
        if truncated:
            lines.append(f'... truncated at {MAX_ROWS} rows')
        if not rows:
            lines.append('(no rows)')
        return '\n'.join(lines)

    def schema(self, name: str) -> str:
        name = (name or '').strip()
        try:
            conn = self._connect()
        except sqlite3.Error as error:
            return f'world_schema error: database unavailable: {error}'
        try:
            tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
            if name and name not in tables:
                return f"world_schema error: unknown table {name!r}. Tables: {', '.join(tables)}"
            if not name:
                return 'Tables: ' + ', '.join(tables)
            columns = [(r[1], r[2], bool(r[3]), bool(r[5])) for r in conn.execute(f'PRAGMA table_info({name})')]
        except sqlite3.Error as error:
            return f'world_schema error: {type(error).__name__}: {error}'
        finally:
            conn.close()
        lines = [f'table {name}:']
        for column, kind, notnull, pk in columns:
            flags = []
            if pk:
                flags.append('primary key')
            if notnull:
                flags.append('not null')
            lines.append(f'  {column} {kind}' + (f" ({', '.join(flags)})" if flags else ''))
        return '\n'.join(lines)

    def doc(self, name: str) -> str:
        name = (name or '').strip()
        if not name:
            return ('Available documents: ' + ', '.join(sorted(self.docs)) + '. Call again with one name.')
        path = self.docs.get(name)
        if path is None:
            return (f'world_doc error: unknown document {name!r}. '
                    f'Available: {", ".join(sorted(self.docs))}')
        text = path.read_text(encoding='utf-8')
        return text if len(text) <= 20000 else text[:20000] + '\n... truncated'

    def _rank(self, query: str) -> list[tuple[float, dict[str, Any]]]:
        want = _terms(query)
        if not want:
            return []
        scored = []
        for doc in self.search_documents:
            have = _terms(doc['title'] + ' ' + doc['body'])
            score = len(want & have) + 2.0 * len(want & _terms(doc['title']))
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda pair: -pair[0])
        return scored

    def web_search(self, query: str, num_results: Any = 5, freshness: str = '') -> str:
        """Ranked results, rendered the way a tool observation is.

        The output is JSON rather than prose so the fields a policy has to reason
        about -- `published`, `url`, `rank` -- stay separable. A paragraph would
        bury the date, and the date is what decides between two documents that
        contradict each other.
        """
        if not isinstance(query, str) or not query.strip():
            return 'web_search: `query` must be a non-empty string.'
        if self.live is not None:
            return self.live.web_search(query, num_results, freshness)
        if not self.search_documents:
            return 'web_search: no index is configured for this run.'
        try:
            limit = int(num_results or 5)
        except (TypeError, ValueError):
            limit = 5
        limit = max(1, min(limit, 25))
        ranked = self._rank(query)
        if freshness:
            # A freshness filter without a stable sort is a different answer every
            # call, so rank is the tiebreak and order is total.
            ranked.sort(key=lambda pair: (pair[1].get('published') or '', pair[0]), reverse=True)
        results = []
        for rank, (_score, doc) in enumerate(ranked[:limit], 1):
            quoted = []
            want = _terms(query)
            for sentence in re.split(r'(?<=\.)\s+', doc['body'].replace('\n', ' ')):
                sentence = sentence.strip(' #')
                if sentence and want & _terms(sentence):
                    quoted.append(sentence)
                if len(quoted) >= 2:
                    break
            results.append({
                'rank': rank,
                'url': doc['url'],
                'title': doc['title'],
                'published': doc.get('published'),
                'excerpts': quoted,
            })
        if not results:
            # Saying so is the answer. A policy trained to expect results will
            # invent one here, and that is exactly the failure to catch.
            return ('web_search: no results for that query. Try different terms, or '
                    'conclude the information is not in the index.')
        return json.dumps({
            'query': query,
            'num_results': len(results),
            'results': results
        },
                          ensure_ascii=False,
                          indent=2)

    def fetch_page(self, url: str, query: str = '') -> str:
        if not isinstance(url, str) or not url.strip():
            return 'fetch_page: `url` must be a string.'
        if self.live is not None:
            return self.live.fetch_page(url, query if isinstance(query, str) else '')
        for doc in self.search_documents:
            if doc['url'] == url:
                return (f"# {doc['title']}\n"
                        f"published: {doc.get('published')}\n"
                        f"url: {doc['url']}\n\n{doc['body']}")
        return (f'fetch_page: no document at {url!r}. Call web_search again to get a '
                'url that exists.')

    # ---- the shared PostgreSQL service --------------------------------
    def _db_credentials(self) -> dict[str, Any] | None:
        """This episode's own database account, created on first use.

        One account per episode, not one shared read-only login: with a shared
        login, "read-only" is a promise about every sandbox's behaviour. With a
        per-episode role it is a grant the server enforces, and the blast radius
        of a bug is one episode.

        Created lazily so that a run which never uses the database never creates
        an account, and dropped in `close()`.
        """
        if self._db_creds is not None:
            return self._db_creds
        if not self.db_enabled:
            return None
        provision_dir = os.environ.get('TWINKLE_WORLD_DB_MODULE', '')
        if not provision_dir:
            raise WorldUnavailable('db_enabled is set but TWINKLE_WORLD_DB_MODULE is not: the '
                                   'per-episode role provisioning is deployment-specific, so the '
                                   'directory holding its `provision` module has to be named.')
        if provision_dir not in sys.path:
            sys.path.insert(0, provision_dir)
        import provision
        self._db_creds = provision.provision(
            os.environ.get('TWINKLE_WORLD_DB_HOST', '127.0.0.1'),
            int(os.environ.get('TWINKLE_WORLD_DB_PORT', '5432')),
            ttl=int(os.environ.get('TWINKLE_WORLD_DB_LEASE', '21600')))
        return self._db_creds

    def _db_connect(self):
        creds = self._db_credentials()
        if creds is None:
            raise RuntimeError('no database is configured for this run')
        import psycopg2
        conn = psycopg2.connect(
            host=creds['host'],
            port=creds['port'],
            dbname=creds['database'],
            user=creds['role'],
            password=creds['password'],
            connect_timeout=10)
        # autocommit on, and NOT set_session(readonly=True).
        #
        # readonly=True was the obvious belt-and-braces move and it was wrong: a
        # read-only transaction refuses CREATE TEMP TABLE AS, which the grant
        # deliberately allows so a solver has somewhere to work. The two guards
        # disagreed, and the one that fired was the one that removed a capability
        # the profile promised -- the model would have been told it could use a
        # scratch table and then been refused.
        #
        # The GRANT is the enforcement, and it is the stronger one: it is a
        # property of the role recorded in the catalogue, it survives this process
        # dying, and it cannot be lifted by anything the model sends. Verified in
        # test_db.py -- insert, update, delete, truncate, drop, create table,
        # alter and create schema are all refused by PostgreSQL itself.
        conn.autocommit = True
        return conn

    def db_query(self, query: str, params: Any = None) -> str:
        if not isinstance(query, str) or not query.strip():
            return 'db_query: `query` must be a non-empty string.'
        if params is not None and not isinstance(params, (list, tuple)):
            return 'db_query: `params` must be a list.'
        try:
            conn = self._db_connect()
        except Exception as error:  # noqa: BLE001
            return f'db_query: cannot reach the shared database: {error}'
        try:
            with conn.cursor() as cur:
                cur.execute(query, list(params) if params else None)
                if cur.description is None:
                    # A statement that returns nothing is allowed to succeed (a
                    # temp table, a SET) but the model is told, so it does not read
                    # silence as a result.
                    return 'db_query: statement completed with no result set.'
                columns = [d[0] for d in cur.description]
                rows = cur.fetchmany(MAX_ROWS + 1)
                truncated = len(rows) > MAX_ROWS
                rows = rows[:MAX_ROWS]
        except Exception as error:  # noqa: BLE001
            return f'db_query error: {type(error).__name__}: {str(error).strip()}'
        finally:
            conn.close()
        lines = [' | '.join(str(c) for c in columns), '-+-'.join('-' * len(str(c)) for c in columns)]
        for row in rows:
            lines.append(' | '.join('NULL' if v is None else str(v) for v in row))
        if truncated:
            lines.append(f'... truncated at {MAX_ROWS} rows')
        if not rows:
            lines.append('(no rows)')
        return '\n'.join(lines)

    def db_schema(self, table: str = '') -> str:
        table = (table or '').strip()
        try:
            conn = self._db_connect()
        except Exception as error:  # noqa: BLE001
            return f'db_schema: cannot reach the shared database: {error}'
        try:
            with conn.cursor() as cur:
                cur.execute('SELECT table_name FROM information_schema.tables '
                            "WHERE table_schema = 'public' ORDER BY table_name")
                tables = [r[0] for r in cur.fetchall()]
                if table and table not in tables:
                    return (f'db_schema error: unknown table {table!r}. '
                            f'Tables: {", ".join(tables)}')
                if not table:
                    return 'Tables: ' + ', '.join(tables)
                cur.execute(
                    'SELECT column_name, data_type, is_nullable '
                    'FROM information_schema.columns '
                    "WHERE table_schema = 'public' AND table_name = %s "
                    'ORDER BY ordinal_position', (table, ))
                columns = cur.fetchall()
        except Exception as error:  # noqa: BLE001
            return f'db_schema error: {type(error).__name__}: {str(error).strip()}'
        finally:
            conn.close()
        lines = [f'table {table}:']
        for name, kind, nullable in columns:
            suffix = '' if nullable == 'YES' else ' not null'
            lines.append(f'  {name} {kind}{suffix}')
        return '\n'.join(lines)

    def release(self) -> None:
        """Drop this episode's database account, if one was made.

        On HostWorld rather than only on the env that owns it: the account belongs
        to this object, and every caller that can create one has to be able to
        release it. Keeping it on the env alone meant a second caller would have
        had to reimplement the drop.

        Safe to call more than once. An episode killed before it gets here never
        calls it at all -- that is what the lease and the reaper are for.
        """
        if self._db_creds is None:
            return
        role = self._db_creds['role']
        self._db_creds = None
        try:
            provision_dir = os.environ.get('TWINKLE_WORLD_DB_MODULE', '')
            if provision_dir and provision_dir not in sys.path:
                sys.path.insert(0, provision_dir)
            import provision
            provision.drop(role)
        except Exception as error:  # noqa: BLE001
            # Not fatal: the account is read-only and leased, so a failed drop
            # leaves an inert role rather than a live hazard.
            print(f'[world] failed to drop database role {role}: {error}')

    def _receipt(self, tool_name: str, arguments: dict[str, Any], result: str) -> None:
        """Record what was served, parsed enough to be checkable.

        The urls are extracted from the tool's own output rather than from its
        arguments: `fetch_page` takes one url, but `web_search` is asked a query
        and decides which urls to return, so the urls that were actually served
        are only knowable from the result.
        """
        entry = {'tool': tool_name, 'args': dict(arguments), 'urls': [], 'query': ''}
        if tool_name == 'web_search':
            entry['query'] = (arguments.get('query') or '').strip()
            try:
                payload = json.loads(result)
            except (TypeError, ValueError):
                payload = {}
            for hit in payload.get('results') or []:
                if isinstance(hit, dict) and hit.get('url'):
                    entry['urls'].append({
                        'url': hit['url'],
                        'published': hit.get('published') or '',
                        'title': hit.get('title') or ''
                    })
        elif tool_name == 'fetch_page':
            url = (arguments.get('url') or '').strip()
            # A fetch that failed is not a receipt for the page it named.
            if url and not result.startswith('fetch_page:'):
                entry['urls'].append({'url': url, 'published': '', 'title': ''})
        else:
            return
        self.receipts.append(entry)

    def _answer(self, tool_name: str, arguments: dict[str, Any]) -> str | None:
        if tool_name == 'world_sql':
            return self.sql(arguments.get('query'), arguments.get('params') or [])
        if tool_name == 'world_schema':
            return self.schema(arguments.get('name'))
        if tool_name == 'world_doc':
            return self.doc(arguments.get('name'))
        if tool_name == 'web_search':
            return self.web_search(
                arguments.get('query'), arguments.get('num_results'),
                arguments.get('freshness') or '')
        if tool_name == 'fetch_page':
            return self.fetch_page(arguments.get('url'), arguments.get('query') or '')
        if tool_name == 'db_query':
            return self.db_query(arguments.get('query'), arguments.get('params'))
        if tool_name == 'db_schema':
            return self.db_schema(arguments.get('table') or '')
        return None

    def call(self, tool_name: str, arguments: dict[str, Any]) -> str | None:
        """Answer one world tool call, or None if the name is not ours.

        The receipt is written here rather than in the caller so that every path
        that answers a search tool records it: a check that trusted the caller to
        remember would silently stop working the first time a second caller
        appeared. A failed call records nothing for the url it failed on, which is
        what makes "I fetched this" a claim with evidence behind it.
        """
        arguments = arguments or {}
        result = self._answer(tool_name, arguments)
        if result is not None:
            self._receipt(tool_name, arguments, result)
        return result


def split_calls(calls: Sequence[tuple[str, dict[str, Any]]],
                world: HostWorld) -> tuple[list[tuple[int, str]], list[tuple[str, dict[str, Any]]]]:
    """Partition one turn's calls into (host-answered, sandbox-bound).

    Batch-aware on purpose: a turn can mix a world call with shell commands, and
    the sandbox request must carry only the calls it can actually run. Order is
    preserved by index so the results can be reassembled positionally -- GRPO
    rewards do not care about order, but a transcript that pairs an observation
    with the wrong call teaches the policy something false.
    """
    host: list[tuple[int, str]] = []
    sandbox: list[tuple[str, dict[str, Any]]] = []
    for index, (name, args) in enumerate(calls):
        short = name.rsplit('---', 1)[-1]
        if short in TOOL_NAMES:
            host.append((index, short))
        else:
            sandbox.append((name, args))
    return host, sandbox


class WorldToolsMixin:
    """Serve host-side world tools alongside whatever an environment provides.

    Mix in front of any :class:`~twinkle_agentic.envs.base.Env`::

        class WorldLocalEnv(WorldToolsMixin, LocalEnv):
            pass

        env = WorldLocalEnv(workspace='/tmp/slot0', world=HostWorld(root))

    The mixin answers the world's own tool names from the training host and
    forwards everything else to the environment untouched. It adds one attribute
    (:attr:`world`) to the public surface, so an environment that already works
    keeps working -- and because it is a mixin rather than a change to
    ``AgentEnv``, it applies to ``LocalEnv`` and ``OpenEnv`` as well, which is
    what makes it testable without a sandbox host.

    Two things that are easy to get wrong:

    * **The mixin goes first.** ``class X(WorldToolsMixin, LocalEnv)``, not the
      reverse: it has to precede the environment in the MRO for its ``tools``,
      ``step`` and ``step_batch`` to be the ones called.
    * **Tool names may arrive suffixed.** A harness that disambiguates
      same-named tools across environments appends ``---<id>``. Both
      :meth:`step` and :meth:`step_batch` strip that before matching, so a
      suffixed world call is answered here instead of being forwarded to a
      sandbox that has no such tool.

    The world's lifetime is the caller's, not the mixin's: several environments
    may share one read-only world, so ``close()`` does not release it. Pass
    ``release_world_on_close=True`` on the environment that owns it.
    """

    def __init__(self,
                 *args: Any,
                 world: HostWorld | None = None,
                 release_world_on_close: bool = False,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._world = world
        self._release_world_on_close = release_world_on_close

    @property
    def world(self) -> HostWorld | None:
        """The host world this environment reads from, or None."""
        return self._world

    def tools(self) -> list[ToolInfo]:
        declared = list(super().tools())
        if self._world is None:
            return declared
        # A name the environment already owns wins. Shadowing it silently would
        # promise the model one behaviour and hand it another.
        names = {(t.get('function') or {}).get('name') for t in declared}
        return declared + [
            schema for schema in self._world.schemas() if (schema.get('function') or {}).get('name') not in names
        ]

    def step(self, tool_name: str, arguments: dict[str, Any] = None) -> StepResult:
        answer = self._answer_world(tool_name, arguments or {})
        if answer is not None:
            return StepResult(observation=answer, reward=0.0, done=False)
        return super().step(tool_name, arguments)

    def step_batch(self, calls: Sequence[tuple[str, dict[str, Any]]]) -> list[StepResult]:
        """Answer the world's calls here, and send the rest as one batch.

        Worth overriding rather than leaving to the serial default: the point of
        ``step_batch`` is keeping tool execution off the generation critical
        path, and a turn that mixes one world call with three shell commands
        should still send those three as a single request.

        Positions are tracked explicitly rather than inferred from order, so a
        transcript cannot pair an observation with the wrong call -- which would
        teach the policy something false without anything downstream noticing.
        """
        if self._world is None:
            return super().step_batch(calls)
        host_at: list[int] = []
        sandbox_at: list[int] = []
        for index, (name, _) in enumerate(calls):
            if _world_name(name) is not None:
                host_at.append(index)
            else:
                sandbox_at.append(index)
        out: list[StepResult | None] = [None] * len(calls)
        for index in host_at:
            name = _world_name(calls[index][0])
            answer = self._world.call(name, calls[index][1] or {})
            out[index] = StepResult(
                observation=(answer if answer is not None else f'Error: {name} is not available in this environment.'),
                reward=0.0,
                done=False)
        if sandbox_at:
            forwarded = super().step_batch([calls[i] for i in sandbox_at])
            if len(forwarded) != len(sandbox_at):
                raise RuntimeError(f'step_batch returned {len(forwarded)} results for '
                                   f'{len(sandbox_at)} forwarded call(s); cannot pair observations '
                                   'with calls, so refusing to guess')
            for index, result in zip(sandbox_at, forwarded):
                out[index] = result
        return [r if r is not None else StepResult(observation='Error: no result') for r in out]

    def _answer_world(self, tool_name: str, arguments: dict[str, Any]) -> str | None:
        """The world's answer, or None when this call is not the world's."""
        if self._world is None:
            return None
        name = _world_name(tool_name)
        if name is None:
            return None
        answer = self._world.call(name, arguments)
        if answer is None:
            return f'Error: {name} is not available in this environment.'
        return answer

    def close(self) -> None:
        try:
            super().close()
        finally:
            if self._release_world_on_close and self._world is not None:
                self._world.release()
