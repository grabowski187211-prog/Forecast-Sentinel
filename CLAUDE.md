# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`forecast-sentinel` — an agent that decides whether an upstream data change has
invalidated a **deployed** ML model, by walking DataHub's end-to-end ML lineage
(training data → features → model → deployment). Built for the DataHub Agent
Hackathon, Challenge 3 (Production ML Agents). Deadline **2026-08-10**.

Requirements and scoring criteria are digested in
`01_Reference_Material/hackathon_brief.md`; open submission items are tracked in
`docs/SUBMISSION.md`.

## Commands

The venv is Python 3.12 and **not** activated by default. Prefix commands or
activate first.

```bash
.venv/bin/pytest                              # no DataHub needed
.venv/bin/pytest tests/test_snapshots.py      # one file
.venv/bin/pytest -k drift                     # one pattern
.venv/bin/pytest tests/test_urns.py::TestNestedUrns::test_dataset_urn_keeps_platform_intact

.venv/bin/ruff check src/ tests/ scripts/     # lint (must be clean)
.venv/bin/ruff check --fix src/

uv pip install -e ".[dev,cli]"                # tests + DataHub SDK/CLI
```

CLI (needs a live DataHub — see `docs/SETUP.md`):

```bash
.venv/bin/sentinel doctor                     # config + MCP connectivity probe
.venv/bin/sentinel models                     # list mlModel URNs
.venv/bin/sentinel baseline <model-urn>       # record training-input schemas
.venv/bin/sentinel check <model-urn>          # detect, judge, write back (exit 2 on BLOCK)
.venv/bin/sentinel watch                      # every model with a baseline

.venv/bin/python scripts/seed_ml_demo.py --dry-run   # verify DataHub SDK compat, emit nothing
./scripts/bootstrap_datahub.sh                       # quickstart + datapack + seed
```

**Python must be 3.10–3.12.** `acryl-datahub` and the MCP server publish no
3.13+ wheels, and the system Python here is 3.14 — always create the venv with
`uv venv --python 3.12`.

## Architecture: the load-bearing decision

The work is split in two, and the split is the design:

- **Deterministic half** (`snapshots.py`, `datahub/ml_lineage.py`) answers *what
  changed*. Pure Python: bounded server-side `get_lineage` traversal plus a schema diff
  against a recorded baseline. Reproducible, no model involved.
- **Agentic half** (`agent/`) answers *does it matter*. OpenAI is primary and
  Anthropic is the `auto` fallback; an explicit Gemini free-tier path uses the
  same read-only tools and typed verdict contract. Each reasons over DataHub's
  real MCP tools about whether a dtype change breaks a trained artefact.

**Do not move drift detection into the agent.** A diff is not a judgement call;
making it one costs money and makes the answer vary between runs. Conversely, do
not try to encode consequence rules in Python — whether a nullability change
matters depends on lineage context the code does not have.

Full flow diagram and extension points: `docs/ARCHITECTURE.md`.

## Constraints that will bite you

**The `mcp` pin is deliberate: `>=1.8,<2`.** `anthropic.lib.tools.mcp.async_mcp_tool`
takes a 1.x `ClientSession`. The MCP SDK's 2.x line replaces that with a
higher-level `Client` facade (`Client(stdio_client(params))`), and its published
docs describe 2.x — so the docs and the code intentionally disagree. If you
"modernise" `datahub/mcp_client.py` to the 2.x API, the Anthropic bridge breaks.
Verified signatures are recorded in `01_Reference_Material/datahub_api_notes.md`.

**DataHub payload shapes vary** across versions and platforms: keys arrive
camelCase *or* snake_case, lineage entries as objects *or* bare URN strings,
owners as strings *or* nested dicts. The parsers in `ml_lineage.py` and
`snapshots.py` accept every observed shape and **drop** what they cannot
interpret rather than raising. Every tolerance has a test — if you tighten a
parser, expect `TestPayloadParsing` / `TestParseSchemaFields` to tell you why it
was loose.

**URN keys contain nested URNs.** A `schemaField` URN embeds a whole `dataset`
URN, which embeds a `dataPlatform` URN. `split(",")` corrupts all of them; use
`datahub.urns.parse_urn`, which tracks parenthesis depth. Key arity also varies
by entity type, so `Urn.parts` stays positional rather than assuming a shape.

**The agent gets read tools only.** OpenAI and Gemini use
`mcp.openai_tools(include_writes=False)`; Anthropic uses
`mcp.anthropic_tools(include_writes=False)`. Both adapters share the same
`WRITE_TOOLS` safety boundary. Write-back happens afterwards in
`_record_verdict`, in code, from the structured verdict. The model decides
*what* to record, not how many catalog objects to touch. Keep it that way;
fallback is safe only while this remains true.

**Writes are double-gated.** The MCP server hides mutation tools unless
`TOOLS_IS_MUTATION_ENABLED=true`, and `DataHubMCP.inventory.has_write_access`
probes for them at runtime. Never assume a write tool exists — `sentinel doctor`
reports what is actually exposed.

**Lineage traversal is bounded** (`max_hops=4`, `max_results=200`) and sets
`graph.truncated` when DataHub returns `hasMore`. That flag reaches the report on
purpose: a partial blast radius presented as complete is worse than none. Do not
silently raise the bounds.

**Snapshot reads fail closed.** If any `list_schema_fields` call fails,
`capture_snapshot` raises `SnapshotCaptureError`. Never turn a partial schema
read into "no drift" or save it as a baseline.

**A missing baseline is not drift.** First run on a model records a baseline and
reviews lineage *coverage* instead. Datasets present in only one snapshot are
skipped, not reported as wholesale add/remove — otherwise linking a new table
buries the findings that matter.

## Conventions

- **The verdict is a typed contract.** `agent/schemas.py` `Verdict` is what the
  CLI exits on, the report renders, and the write-back persists. The agent
  produces it by calling the local `emit_verdict` tool; a schema-invalid verdict
  is fed back to the model to retry rather than dropped.
- **Prompts encode the rubric.** `agent/prompts.py` holds the BLOCK/WARN/OK/UNKNOWN
  definitions and the evidence-discipline rules (cite lineage paths, list
  `unverified_claims`, do not dress up governance gaps as model risks). Behaviour
  changes belong there, not scattered through `sentinel.py`.
- **The HTML report is static and self-contained** — no scripts, no external
  assets, Jinja2 autoescaping on. Catalog strings are user-controlled and the
  report opens from `file://` URLs. `tests/test_report.py` asserts both.
- **`ruff` ignores `UP042`/`UP017`** (`StrEnum` and `datetime.UTC` are 3.11+,
  and this package supports 3.10) and **`B008`** (typer's `Option()`-in-default
  is the framework's intended API). Don't "fix" those.
- `datahub.sdk` is flagged experimental by DataHub itself, so
  `scripts/seed_ml_demo.py --dry-run` exists to catch signature moves without
  needing a live instance.

## Current verification status

Verified: unit and orchestration-contract tests green, `ruff` clean, all
third-party signatures checked against installed packages, seeder dry-run
constructs every entity, and a prior live DataHub 1.5.0.6 run confirmed lineage
and schema drift detection.

Verified end to end against live DataHub: the explicit Gemini free-tier path
detected the `INT → VARCHAR` drift, emitted a typed `BLOCK` with zero unverified
claims, and completed status-tag cleanup/addition, description append, and
linked-document save. JSON, static HTML, and terminal evidence are checked in
under `examples/`. OpenAI remains the primary `auto` path and Anthropic its
fallback; neither paid-provider judgement path is claimed as live-verified.
