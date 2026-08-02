# Forecast Sentinel contributor guide

## Product

`forecast-sentinel` decides whether an upstream structural data change has
invalidated a deployed ML model by combining DataHub ML lineage with a recorded
training-schema baseline. It is a Build with DataHub Agent Hackathon entry for
Challenge 3, Production ML Agents. The submission deadline is 2026-08-10.

The load-bearing design boundary is:

- Deterministic Python answers **what changed**: bounded DataHub lineage reads,
  schema snapshots, and reproducible diffs.
- The model answers **does it matter**: it verifies relevant paths through
  read-only DataHub MCP tools and emits one typed verdict.
- The orchestrator performs catalog mutations after the verdict. The model must
  never receive mutation tools.

Do not move drift detection into the model or encode model-consequence rules as
hard-coded Python heuristics.

## Commands

The supported interpreter range is Python 3.10-3.12. This checkout uses 3.12.
The environment must be editable-installed from this directory:

```bash
uv venv --python 3.12
uv pip install -e ".[dev,cli]"

.venv/bin/pytest
.venv/bin/ruff check src/ tests/ scripts/
.venv/bin/python scripts/seed_ml_demo.py --dry-run
```

Live DataHub commands:

```bash
.venv/bin/sentinel doctor
.venv/bin/sentinel models
.venv/bin/sentinel baseline <model-urn>
.venv/bin/sentinel check <model-urn>
.venv/bin/sentinel watch
```

Never read or commit `.env`. Runtime baselines, reports, PIDs, and MCP logs in
`.sentinel/` are ignored because they can contain catalog metadata.

## Compatibility constraints

- Keep `mcp>=1.8,<2`. Anthropic's `async_mcp_tool` bridge takes the MCP 1.x
  `ClientSession`; the documented 2.x `Client` API is not interchangeable.
- `get_lineage` uses `upstream: bool`, `max_hops`, and `max_results`. DataHub
  performs the transitive walk server-side. The defaults are four hops and 200
  results; `hasMore` must set and surface `graph.truncated`.
- DataHub payloads vary across versions. Preserve the tested camelCase,
  snake_case, nested-object, and bare-URN tolerances.
- Nested DataHub URNs cannot be parsed with `split(",")`; use `parse_urn`.
- Any failed `list_schema_fields` call must raise `SnapshotCaptureError`. A
  partial snapshot must never be saved or interpreted as no drift.

## Agent and write-back contracts

- `Verdict` in `agent/schemas.py` is the contract shared by CI, terminal/HTML
  reports, JSON output, and DataHub write-back.
- Keep the default non-streaming output budget at 8,000 tokens or below unless
  all provider runners are migrated to streaming; the Anthropic SDK rejects
  longer estimated requests before sending them.
- OpenAI is primary in `auto` mode and Anthropic is the fallback. Both receive
  only their `*_tools(include_writes=False)` adapter plus the local
  `emit_verdict` tool. Keep `WRITE_TOOLS` as their shared safety boundary.
- Explicit `gemini` mode uses Google's OpenAI-compatible Chat Completions API,
  the same `openai_tools(include_writes=False)` adapter, and no paid-provider
  fallback. Keep free-tier catalog inputs synthetic or non-sensitive.
- Provider fallback must remain inside the read-only judgement phase. Catalog
  writes happen once, after a valid verdict, so retrying another provider is
  side-effect safe.
- Live DataHub mutation arguments are:
  - `add_tags` / `remove_tags`: `tag_urns`, `entity_urns`
  - `update_description`: `entity_urn`, `operation`, `description`
  - `save_document`: `document_type`, `title`, `content`; use `related_assets`
    to link the report to the checked model
- Clear obsolete Sentinel status tags before adding the current status.
- Append the timestamped status rather than replacing the model's existing
  human-authored description.
- Treat every mutation as independently fallible and surface failures.

## Reporting and submission truth

- The HTML report stays static, self-contained, autoescaped, and free of
  scripts or external assets.
- `README.md` is the main judge-facing document; `docs/BUILD_LOG.md` does not
  exist here, so git history and `docs/SUBMISSION.md` are the evidence trail.
- The README terminal transcript must stay synchronized with the checked-in
  JSON, HTML, and terminal output under `examples/`.
- A live DataHub 1.5.0.6 run with `gemini-3.6-flash` verified the full path:
  lineage, baseline, deterministic dtype drift, typed `BLOCK`, and all four
  write-back mutations. Do not claim the OpenAI or Anthropic judgement paths
  are live-verified until an authenticated run proves each one.
- Keep `README.md`, `docs/SETUP.md`, `docs/ARCHITECTURE.md`,
  `docs/SUBMISSION.md`, and the ignored `docs/DEVPOST_STORY.md` consistent with
  the implemented and verified state.
