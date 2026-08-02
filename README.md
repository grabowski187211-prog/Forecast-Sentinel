# Forecast Model Sentinel

**An agent that stops silent ML model invalidation, using DataHub's end-to-end ML lineage.**

Built for [Build with DataHub: The Agent Hackathon](https://datahub.devpost.com) — Challenge 3, *Production ML Agents*.

---

## The problem

A demand-forecasting model goes into production. Six weeks later, an upstream
data engineer changes `holiday_flag` from an integer to a string in the source
sales table. Nothing breaks. No pipeline fails. No alert fires.

The model keeps serving predictions. They are quietly wrong, because the
encoding it learned at training time no longer matches the encoding it receives
at inference time. The planning team acts on those numbers for a month before
anyone notices the forecast error has drifted.

This failure is invisible to the tools teams already have:

- **Monitoring** watches the model endpoint. The endpoint is healthy.
- **Data quality tests** watch the table. The table is valid — a string column
  is a perfectly good string column.
- **Drift detection** watches the prediction distribution, and reports it weeks
  later, after the damage, without saying why.

The missing link is *lineage*: nobody connected "this column changed" to "that
deployed model depends on it". DataHub is the one system that holds that link.

## What the sentinel does

It answers the question no dashboard currently answers: **this upstream thing
changed — is my deployed model still valid?** The terminal transcript below is
illustrative until a checked-in live run replaces it in `examples/`.

```
$ sentinel check "urn:li:mlModel:(urn:li:dataPlatform:mlflow,demand-forecast-v3,PROD)"

╭─ BLOCK — demand-forecast-v3 ─────────────────────────────────────────────────╮
│ holiday_flag changed from int to string in the model's primary training       │
│ input; the deployed artefact's learned encoding no longer matches inference   │
│ data, so predictions are silently wrong.                                      │
╰──────────────────────────────────────────────────────────────────────────────╯
lineage: 14 upstream, 6 downstream, max hops 4

Upstream changes (2)
┏━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ sev    ┃ change                            ┃ dataset                        ┃
┡━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ high   │ holiday_flag: dtype INT -> VARCHAR │ urn:li:dataset:(…,raw_sales,…) │
│ low    │ promo_id: added (BIGINT)           │ urn:li:dataset:(…,raw_sales,…) │
└────────┴───────────────────────────────────┴────────────────────────────────┘

Risks
  HIGH  Feature encoding mismatch on holiday_flag
    path: raw_sales.holiday_flag → feat_seasonality → demand-forecast-v3

Recommended actions
  1. Stop the nightly batch scoring job before the next 02:00 run.
  2. Pin feat_seasonality to the previous dtype, or retrain on the new encoding.
  3. Notify the owners of exec_demand_dashboard — last 3 days of figures suspect.

Written back to DataHub
  ✓ remove_tags — cleared obsolete sentinel status tags
  ✓ add_tags — tagged urn:li:tag:model-invalidated
  ✓ update_description — appended verdict status to the description
  ✓ save_document — saved full findings document linked to the model

BLOCK verdict — failing with exit code 2.
```

Exit code 2 means this drops into CI as a gate in front of a scheduled retrain
or batch-scoring job. The verdict is also written back into DataHub, so the next
person who opens that model in the catalog sees the finding without being told.

## How it works

The design splits the problem in two, because the halves need different tools.

```
     ┌─────────────── deterministic (code) ───────────────┐
     │                                                     │
  resolve model ──> walk ML lineage ──> snapshot training  │
  (get_entities)    (get_lineage,       input schemas      │
                     bounded server-     (list_schema_fields)
                     side traversal)
                                              │            │
                                     diff vs recorded      │
                                        baseline           │
     └─────────────────────────────────────┬───────────────┘
                                           │  facts, not opinions
     ┌────── agentic (OpenAI, Anthropic fallback) ─▼───────┐
     │  Verify the lineage paths that matter               │
     │  (get_lineage_paths_between, get_dataset_queries)   │
     │  Judge consequence per change                       │
     │  Establish blast radius downstream                  │
     │  emit_verdict → BLOCK | WARN | OK | UNKNOWN         │
     └─────────────────────────────────────┬───────────────┘
                                           │
                     write back: add_tags, update_description,
                                 save_document
```

**Why the split matters.** Detecting that a column changed type is a diff — code
does it faster, cheaper and more reliably than a model, and the result is
reproducible. Deciding whether that change *invalidates a trained artefact*
requires reading the lineage, understanding how a specific model consumes a
specific feature, and weighing consequence. That is genuine judgement, and it is
the only part the model is asked to do.

The agent never detects drift. It is handed the facts and asked what they mean.

OpenAI's Responses API is the primary judgement path. The same read-only tools
and typed `emit_verdict` contract are available through the Anthropic tool
runner as a fallback. Provider failure can therefore be retried without
duplicating catalog mutations: all writes remain outside the model loop.

**Evidence discipline.** Every risk must cite the lineage path it travels and
facts retrieved from DataHub in that session. Anything the agent asserted but
could not confirm goes in an explicit `unverified_claims` field rather than
being dropped silently. An ML-lineage agent that speculates is worse than no
agent, because its output gets pasted into an incident channel.

## DataHub integration

The sentinel talks to DataHub **only** through the DataHub MCP server — the
agent calls DataHub's real tools rather than a reimplementation of them.

| Tool | Used for |
|---|---|
| `get_entities` | Resolve the model, its features, owners, deployment |
| `get_lineage` | Breadth-first walk upstream (training data) and downstream (blast radius) |
| `get_lineage_paths_between` | *Prove* a changed column reaches the model, rather than assume it |
| `list_schema_fields` | Snapshot and re-read training-input schemas |
| `get_dataset_queries` | See how a feature is actually computed |
| `search` | Discover mlModel entities |
| `add_tags` | Mark the model `model-invalidated` / `model-needs-review` |
| `update_description` | Put the verdict where a human will see it |
| `save_document` | Attach the full findings report to the catalog |

It reads **and writes**. A read-only lineage report is a nice diagram; writing
the verdict back into the catalog is what makes the next person's search results
carry the warning.

Both DataHub deployment shapes are supported from one config layer:

- **self-hosted** — spawns `uvx mcp-server-datahub@latest` over stdio against GMS
- **DataHub Cloud** — connects to the tenant's streamable-HTTP MCP endpoint

Nothing above `forecast_sentinel/datahub/mcp_client.py` knows which is live.

## Install

Requires Python 3.10–3.12 (`acryl-datahub` and the MCP server do not yet ship
wheels for 3.13+), [uv](https://docs.astral.sh/uv/), and Docker if you want a
local DataHub.

```bash
git clone <this-repo> && cd forecast-model-sentinel
uv venv --python 3.12
uv pip install -e ".[dev,cli]"
cp .env.example .env      # then edit
```

### Bring up a local DataHub

```bash
# DataHub CLI (either global via Homebrew, or into this venv):
brew install datahub-project/tap/datahub
datahub docker quickstart                        # UI on :9002, GMS on :8080
datahub init --username datahub --password datahub
datahub datapack load showcase-ecommerce         # ~1,050 entities with lineage
```

The showcase datapack is rich in datasets but thin on `mlModel` entities, so the
repo ships a seeder that emits a small demand-forecasting ML graph
(dataset → feature → model → deployment) plus a reproducible dtype change:

```bash
python scripts/seed_ml_demo.py                   # emit the ML lineage
python scripts/seed_ml_demo.py --break-schema    # flip holiday_flag INT -> VARCHAR
```

### Verify connectivity

```bash
sentinel doctor
```

## Use

```bash
sentinel models                       # find mlModel URNs in the catalog
sentinel baseline <model-urn>         # record training-input schemas as the baseline
sentinel check <model-urn>            # detect drift, judge it, write back, report
sentinel watch                        # check every model that has a baseline
```

`check` writes a self-contained HTML report to `.sentinel/runs/` — static, no
scripts, no external assets, so it opens from a `file://` URL and attaches
cleanly to a ticket.

### As a CI gate

```yaml
- name: Validate forecast model before retrain
  run: sentinel check "$MODEL_URN" --no-html
  # exit 2 on BLOCK fails the job
```

## Configuration

All configuration is environment-based; see [`.env.example`](.env.example).

| Variable | Purpose |
|---|---|
| `DATAHUB_MODE` | `selfhosted` or `cloud` |
| `DATAHUB_GMS_URL` / `DATAHUB_GMS_TOKEN` | Self-hosted GMS endpoint and PAT |
| `DATAHUB_TENANT_URL` / `DATAHUB_TOKEN` | Cloud tenant and PAT |
| `TOOLS_IS_MUTATION_ENABLED` | Gates the MCP server's write tools |
| `SENTINEL_PROVIDER` | `auto` (OpenAI then Anthropic), `openai`, or `anthropic` |
| `OPENAI_API_KEY` / `SENTINEL_OPENAI_MODEL` | Primary provider credentials and model |
| `ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN` | Fallback credentials |
| `SENTINEL_ANTHROPIC_MODEL` / `SENTINEL_EFFORT` | Fallback model and shared reasoning effort |
| `SENTINEL_FAIL_ON_BLOCK` | Whether a BLOCK verdict exits non-zero |

With `SENTINEL_PROVIDER=auto` (the default), the Sentinel uses
`OPENAI_API_KEY` first. If OpenAI is unavailable, errors, or returns no typed
verdict, it retries through Anthropic when `ANTHROPIC_API_KEY`,
`ANTHROPIC_AUTH_TOKEN`, or an SDK profile is available. Set the provider to
`anthropic` to skip OpenAI. `sentinel doctor` reports both credential paths.

## Project layout

```
src/forecast_sentinel/
├── config.py              dual-deployment configuration
├── snapshots.py           schema baselines + deterministic drift detection
├── cli.py                 the `sentinel` command
├── datahub/
│   ├── mcp_client.py      MCP transport (stdio | streamable HTTP)
│   ├── urns.py            DataHub URN parsing (nesting-aware)
│   └── ml_lineage.py      bounded server-side ML-lineage traversal
├── agent/
│   ├── prompts.py         system prompt + per-run turns
│   ├── schemas.py         the verdict contract
│   └── sentinel.py        run orchestration, agent loop, write-back
└── report/html.py         self-contained HTML report
```

There is also a **Claude Code plugin** in [`plugin/`](plugin/) that exposes the
same capability conversationally (`/sentinel:check`, `/sentinel:baseline`) for
people who would rather ask than run a CLI.

## Testing

```bash
pytest
```

The deterministic half and provider orchestration are unit-tested without a
live DataHub: URN parsing, schema-payload normalisation, drift diffing, the
OpenAI function-call continuation loop, Anthropic fallback, and the shared
read-only tool boundary.

## Design notes and limitations

- **Lineage traversal is bounded** (`max_hops=4`, `max_results=200`). DataHub
  performs the transitive walk server-side. When it reports more results than
  the bound allows, the run is marked `truncated` rather than silently reporting
  a partial blast radius as complete.
- **A missing baseline is not drift.** The first run on a model records a
  baseline and reviews lineage *coverage* instead — whether the catalog is
  complete enough to catch a future change. Reporting every field of a
  newly-linked table as a change would bury the ones that matter.
- **Governance gaps are reported as governance gaps**, at low severity. "No owner
  set on the feature table" is not evidence the forecast is wrong, and the prompt
  says so explicitly.
- **Distribution drift is out of scope.** The sentinel reasons about *structural*
  change (schema, lineage, ownership) because that is what DataHub knows.
  Statistical drift needs the data itself, not the catalog.
- **DataHub payload shapes vary** across versions and platforms — keys appear as
  camelCase or snake_case, lineage entries as objects or bare URN strings. The
  parsers are deliberately tolerant, and every parser has a test.

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Author

Jan-Philipp Grabowski — [inphronesys.com](https://inphronesys.com) ·
[LinkedIn](https://www.linkedin.com/in/jpgrabowski/)
