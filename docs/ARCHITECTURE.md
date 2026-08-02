# Architecture

## The one design decision that matters

The work splits into two halves with different tools:

| | Deterministic half | Agentic half |
|---|---|---|
| Question | *What changed?* | *Does it matter?* |
| Implementation | Python: BFS + schema diff | OpenAI, with Anthropic fallback + DataHub MCP tools |
| Properties | Reproducible, free, fast | Reasoned, cited, judged |
| Files | `snapshots.py`, `datahub/ml_lineage.py` | `agent/` |

Detecting that `holiday_flag` went from `INT` to `VARCHAR` is a diff. Sending a
model to do it would be slower, more expensive, and *less* reliable — the answer
would vary between runs. Deciding whether that change invalidates a trained
gradient-boosting model requires reading lineage, understanding how the feature
is consumed, and weighing consequence. That is judgement, and it is the only
thing the model is asked for.

The agent is never asked to detect drift. It is handed facts and asked what they
mean.

## Flow

```
sentinel check <model-urn>
  │
  ├─ DataHubMCP.__aenter__()            stdio subprocess or streamable HTTP
  │    └─ list_tools()                  probe what this server actually exposes
  │
  ├─ build_ml_lineage(mcp, urn)         DETERMINISTIC
  │    └─ one get_lineage call per direction; DataHub walks transitively
  │       bounded: max_hops=4, max_results=200 → sets graph.truncated
  │
  ├─ capture_snapshot(mcp, urn, datasets)
  │    └─ list_schema_fields per training input
  │
  ├─ diff_snapshots(baseline, current)  → list[DriftEvent]
  │    severity: removed/dtype = high, nullability = medium, added = low
  │
  ├─ Sentinel._judge(...)               AGENTIC
  │    ├─ OpenAI Responses function-tool loop (primary)
  │    └─ Anthropic MCP tool runner (fallback)
  │       both receive DataHub read tools + the same emit_verdict contract
  │       agent verifies paths, judges consequence, calls emit_verdict once
  │
  ├─ Sentinel._record_verdict(...)      write-back
  │    add_tags · update_description · save_document
  │
  └─ render: rich terminal + self-contained HTML + optional JSON
     exit 2 on BLOCK
```

## Module responsibilities

```
config.py            One config object covering both deployment shapes.
                     Selects OpenAI primary / Anthropic fallback independently.

datahub/
  mcp_client.py      Owns the MCP session. Two transports, one interface.
                     Adapts read tools to both providers and probes availability.
  urns.py            URN parsing. Depth-aware comma splitting; positional parts.
  ml_lineage.py      Bounded BFS. Tolerant payload parsing.

snapshots.py         Baselines on disk, one JSON per model. Deterministic diff.

agent/
  prompts.py         System prompt + per-run turns. Encodes the verdict rubric
                     and evidence discipline.
  schemas.py         The verdict contract: Decision, RiskItem, Verdict.
  sentinel.py        Orchestration, agent loop, write-back.

report/html.py       Static self-contained HTML. Autoescaped.
cli.py               typer app. Exit 2 = BLOCK.
```

## Why the specific choices

**Bounded traversal, explicitly marked.** DataHub's `get_lineage` performs the
multi-hop walk server-side. In a real catalog, an unbounded walk from a central
table enumerates the warehouse. The client requests `max_hops=4` and
`max_results=200`; a `hasMore` response sets `graph.truncated`, which is surfaced
in the report. A partial blast radius presented as complete is worse than no
blast radius.

**Tolerant parsers, tested.** DataHub payloads vary across versions and
platforms: keys appear camelCase or snake_case, lineage entries as objects or
bare URN strings, owners as strings or nested dicts. The parsers accept all
observed shapes and drop what they cannot interpret rather than raising. Every
tolerance has a test — see `tests/test_ml_lineage.py::TestPayloadParsing`.

**A one-direction lineage failure is visible but survivable.** If upstream or
downstream retrieval succeeds while the other direction fails, the useful half
is retained and the error is surfaced on the graph. Failure in both directions
raises `LineageError`; it must never look like an empty lineage result.

**A failing schema read does abort snapshotting.** A partial current snapshot
cannot safely mean "no drift". Every failed `list_schema_fields` call is
collected and surfaced as a `SnapshotCaptureError`; no partial baseline is saved.

**Direct tool calls for the deterministic path.** `DataHubMCP.call()` exists so
snapshotting and lineage traversal can hit MCP tools without a model in the loop.
The agent reaches the same tools through the tool runner.

**Read tools only, in the agent loop.** Both
`openai_tools(include_writes=False)` and
`anthropic_tools(include_writes=False)` filter through the same `WRITE_TOOLS`
safety boundary. Write-back happens after the verdict, in code, from structured
fields. The model decides *what* to record; it does not get to decide *how many*
catalog objects to touch.

**Provider fallback is side-effect safe.** OpenAI is attempted first in `auto`
mode. If its API call fails or it ends without a valid typed verdict, the same
evidence is retried through Anthropic. Because model-visible tools are read-only,
the retry cannot duplicate a tag, description, or document write. Token usage is
aggregated and the provider that produced the verdict is recorded in run notes.

**Write-back follows the live MCP schemas.** The orchestrator removes obsolete
Sentinel status tags, adds the current one, appends a timestamped status to the
model description, and saves an `Analysis` document with the model in
`related_assets`. Each mutation is independently recorded as succeeded or
failed.

**A missing baseline is not drift.** The first run on a model records a baseline
and reviews lineage *coverage* instead. Likewise, a dataset that appears in only
one snapshot is skipped rather than reported as a wholesale add/remove —
otherwise linking a new table would drown the real findings.

**The verdict is a typed contract.** `emit_verdict` is a local tool with a strict
schema, and a rejected verdict is fed back to the model to retry. That is why the
CLI can exit 2 and the write-back has structured fields to persist.

## Extension points

| To add | Where |
|---|---|
| A new drift signal (freshness, assertions, volume) | `snapshots.py` — emit `DriftEvent`s; the agent half needs no change |
| A new write-back target (Slack, PagerDuty, Jira) | `Sentinel._record_verdict` |
| A new model provider | Add a read-only tool adapter and provider loop in `Sentinel._judge` |
| Cloud-only assertions and incidents | New MCP tool names in `mcp_client.WRITE_TOOLS` / `READ_TOOLS`; the runtime probe handles absence |
| A different report format | `report/` — `SentinelRun.to_dict()` is the stable surface |

## Known limitations

- **Structural change only.** The sentinel reasons about schema, lineage and
  ownership because that is what a catalog knows. Statistical distribution drift
  needs the data itself.
- **`mlFeature` entities are not required.** The demo seeder models features as a
  materialised dataset, because the installed DataHub SDK exposes no `mlFeatures`
  setter. Real catalogs using Feast will have proper `mlFeature` entities, and
  the lineage walk handles them — it is the *seeder* that is limited, not the
  agent.
- **Baselines are local files.** Fine for CI with a cached `.sentinel/`
  directory; a shared deployment would want them in object storage or as DataHub
  structured properties.
- **One model per check.** `sentinel watch` loops, but there is no cross-model
  deduplication: two models sharing a broken input each get their own verdict.
