# rsi — the model proposes its own tasks, and that work trains it

`rsi_grpo.py` is the entry point. It runs one loop: a challenger acts in a
workspace, writes a check script that verifies what it built, states the task
someone else would be given, and only the statements whose pass rate lands inside
`--pass-band` are kept. Both halves — the proposing and the solving — train in the
same optimizer step.

For the sandbox host and the agent that runs inside it, see
[agentic/README.md](agentic/README.md). This file covers what a run can add on
top. All of it is optional and all of it is off by default.

## What a run can add

| layer | what it adds | turn it on with | code |
|---|---|---|---|
| host world | tools answered on the training host, from data the sandbox never receives | `TWINKLE_WORLD_ROOT` | `twinkle_agentic.envs.world` |
| profiles | one restricted tool set and one seeded fixture per slot | `TWINKLE_RSI_PROFILES` or `TWINKLE_TOPIC_AFFINITIES` | `twinkle_agentic.envs.profiles`, `.topic_profile` |
| knowledge bank | topics retrieved from a mined bank instead of invented | `TWINKLE_KNOWLEDGE_CARDS` | `twinkle_agentic.challenger.knowledge` |
| task bank | the record of kept tasks, so novelty has a referent | `TWINKLE_TASK_BANK` | `twinkle_agentic.challenger.task_bank` |
| novelty score | a within-group novelty a program computes, instead of a judge's | see below | `twinkle_agentic.novelty_metric` |

Slot construction lives in [`agentic/slot.py`](agentic/slot.py). With none of the
variables above set it builds exactly the `AgentEnv` / `LocalEnv` list
`rsi_grpo.py` built before these existed, so a run that wants none of this pays
for none of it.

## Environment variables

| variable | default | meaning |
|---|---|---|
| `TWINKLE_WORLD_ROOT` | — | directory holding `manifest.json`, `ops.db` and `docs/` |
| `TWINKLE_WORLD_SEARCH_ROOT` | — | directory holding `corpus.json` and its `manifest.json`; without it the two search tools are not offered |
| `TWINKLE_WORLD_SEARCH_BACKEND` | `corpus` | `corpus`, or anything else to load a live engine |
| `TWINKLE_WORLD_SEARCH_MODULE` | — | directory holding `search_backend.py`; required for a live engine |
| `TWINKLE_WORLD_DB_MODULE` | — | directory holding the `provision` module that mints and drops per-episode database roles |
| `TWINKLE_WORLD_DB_HOST` / `_PORT` / `_LEASE` | `127.0.0.1` / `5432` / `21600` | the shared SQL service |
| `TWINKLE_RSI_PROFILES` | — | comma-separated profile ids to rotate the slots through |
| `TWINKLE_TOPIC_AFFINITIES` | — | JSONL table of eligible profiles per card; read only when the list above is empty |
| `TWINKLE_KNOWLEDGE_CARDS` | — | path to a knowledge-card bank |
| `TWINKLE_KNOWLEDGE_CARDS_SHA256` | — | pins that file; a substituted bank fails the run instead of training on it |
| `TWINKLE_KNOWLEDGE_SEED` | `0` | which permutation of the bank this run walks |
| `TWINKLE_TASK_BANK` | — | JSONL file of kept statements, appended as the run goes |

Every file the world reads is hashed against a `manifest.json` beside it, at load.
A drifted corpus or database raises rather than serving different answers than the
ones the run's tasks were written against.

## Running it

Unchanged from before:

```bash
python cookbook/rsi/rsi_grpo.py --sandbox-template twinkle-rsi-msagent \
    --sandbox-api-url http://<host>:<port> --num-envs 32 --num-keywords 0
```

With all four layers:

```bash
TWINKLE_WORLD_ROOT=/data/world \
TWINKLE_WORLD_SEARCH_ROOT=/data/search \
TWINKLE_RSI_PROFILES=world-forensics@1,web-research@1,sql-analytics@1,docs-retrieval@1 \
TWINKLE_KNOWLEDGE_CARDS=/data/bank/knowledge_cards.jsonl \
TWINKLE_KNOWLEDGE_CARDS_SHA256=<sha256 of that file> \
TWINKLE_TASK_BANK=$SAVE_DIR/tasks.jsonl \
python cookbook/rsi/rsi_grpo.py --sandbox-template twinkle-rsi-msagent \
    --sandbox-api-url http://<host>:<port> --save-dir $SAVE_DIR --num-envs 32
```

Both halves of a comparison have to set these identically. A run with a world and
a run without one are not measuring the same task distribution, and the difference
is invisible in the reward curves.

## Profiles

A profile is a tool allow-list plus a setup manifest. `default_profile_registry()`
ships these:

| id | `starting_state` | tools beyond shell/python/files | `seed-*` program |
|---|---|---|---|
| `knowledge-answer@1` | empty | — (no-tool session) | — |
| `python-compute@1` | empty | — | — |
| `code-python@1` | empty | — | — |
| `lean-proof@1` | empty | — | — |
| `sql-analytics@1` | seeded | — | `seed-sql-analytics` |
| `data-wrangling@1` | seeded | — | `seed-csv-dirty` |
| `docs-retrieval@1` | seeded | — | `seed-docs-corpus` |
| `log-forensics@1` | seeded | — | `seed-log-corpus` |
| `repo-patch@1` | seeded | — | `seed-broken-repo` |
| `world-forensics@1` | host | `world_sql`, `world_schema`, `world_doc` | — |
| `web-research@1` | host | `web_search`, `fetch_page` | — |
| `db-analyst@1` | host | `db_query`, `db_schema` | — |
| `legacy-agentic@1` | empty | `todo`; retired, takes no new tasks | — |

`starting_state` is derived, not declared: `host` when any allowed tool is answered
on the training host, `seeded` when the manifest names a `seed-*` program,
otherwise `empty`. It is what a task author has to be told, because "read the
database" means three different things across those rows.

Every profile with a setup manifest ends it with a canary —
`python-numpy-sympy-canary` or `python-test-runner-canary` — which imports what
tasks in that profile will need. A canary that fails raises out of
`switch_profile`, so a slot with a broken interpreter never produces tasks that
read as merely hard. The two profiles with no manifest, `knowledge-answer@1` and
`legacy-agentic@1`, run none: the first has no tools to be broken, the second is
retired and takes no new tasks.

A seeded profile differs from a host one in a way that matters for grading. A
fixture planted in the workspace can be opened, edited and reasoned about as local
material; a host-served one can only be queried, and what comes back is what the
world holds. That is what makes the answer unfakeable.

## Check kinds added

`twinkle_agentic.verifier.result_check` gained six, on top of the ten upstream:

| kind | what it asserts |
|---|---|
| `hidden_python` | a python check the proposer never saw, so it cannot be read off the task |
| `answer_one_of` | the answer is one of an exact set, with no string-matching slack |
| `answer_number` | the answer is a number within a tolerance |
| `sql_query` | a query the verifier runs itself returns the expected rows |
| `web_receipt` | a cited url was actually served to this episode, per the world's receipts |
| `lean` | a `lake env lean` compile of the submitted proof succeeds |

`web_receipt` is the one that only works because the world is host-served: the
receipt is written when the tool answers, so a policy that invents a url has
nothing to match against.

## Scoring novelty

`twinkle_agentic.novelty_metric.NoveltyScorer` scores a group of sibling proposals
against each other. It exists because a novelty term an LLM judge supplies can be
measured, and on two recorded runs it measured badly: 93.3% and 98.5% of
proposals scored exactly zero (1439 and 4589 judged), and 66.2% and 91.6% of
groups (225 and 783) scored identically on every sibling. GRPO subtracts the group mean and divides by the group standard
deviation, so a term constant across a group contributes **exactly zero
gradient**. Those runs had a novelty term in the reward and it did nothing.

`constant_group_rate` is that measurement, exported so any replacement is judged
the same way.

Two terms, both computed inside the group because that is the contrast GRPO uses:
`distance` is `1 - max cosine` to the closest sibling, and `spread` is
`1 / cluster size` over siblings within `cluster_threshold` of each other. The
vectoriser is fitted on the whole corpus of statements, not on one group — these
statements share their scaffolding verbatim, and IDF fitted per group would weight
that boilerplate above the content.

Its calibration set is `TaskBank.sibling_groups()`, which is why that method
keeps a group id per statement:

```python
bank = TaskBank(os.path.join(SAVE_DIR, 'tasks.jsonl'))
bank.ingest_recorder(SAVE_DIR)
scorer = NoveltyScorer(bank.statements(), calibration_groups=bank.sibling_groups())
scores = scorer.score(group_statements)   # one per proposal, input order
```

Needs `scikit-learn` and `numpy`, imported inside the methods that use them. A run
that never scores novelty pays neither.

## Things that will bite

- **Mixin order.** `ProfiledEnvMixin` has to precede `WorldToolsMixin`, which has
  to precede the environment. `slot.py` gets this right; hand-composing a class
  the other way round leaves the world's tools unfiltered.
- **A profile that cannot run here is dropped, not retried.** `db-analyst@1` needs
  a reachable SQL service. `topic_profile._service_reachable` probes it once at
  load, so cards routed there fall through to `sql-analytics@1`. Without the probe
  they would score zero and read as "database topics are too hard".
- **The knowledge bank's control arm returns `None`.** One group in each matched
  pair is handed a card and the other is told to propose from scratch. If other
  seeders are also configured, the control arm still gets their text — run with
  `--num-keywords 0` and no `--seed-dataset` for a clean comparison.
- **The task bank is filled from the recorder.** `AgenticChallenger` writes every
  proposal to `<save-dir>/trajs/index.jsonl`; `TaskBank.ingest_recorder` reads the
  kept ones back each iteration. No `--save-dir` means no bank.
- **The novelty scorer cannot see words its corpus never used.** `TfidfVectorizer`
  drops out-of-vocabulary tokens, so fit it on the run being scored. Fitting on
  one run and scoring another gives a scorer blind to the difference.
