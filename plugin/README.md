# Forecast Sentinel — Claude Code plugin

The same capability as the `sentinel` CLI, reachable conversationally. Use this
when you are *investigating* ("what depends on this column?", "is this model
safe to retrain?") rather than *gating* — the CLI is the right tool in CI.

## Install

```bash
claude plugin install forecast-sentinel --from ./plugin
```

Requires the DataHub MCP server to be configured in Claude Code:

```bash
# self-hosted
claude mcp add datahub \
  -e DATAHUB_GMS_URL="http://localhost:8080" \
  -e DATAHUB_GMS_TOKEN="$DATAHUB_GMS_TOKEN" \
  -e TOOLS_IS_MUTATION_ENABLED="true" \
  -- uvx mcp-server-datahub@latest

# DataHub Cloud (OAuth)
claude mcp add --transport http datahub https://mcp.datahub.com/mcp
```

Verify with `claude mcp list`.

## Skills

| Skill | Use for |
|---|---|
| `ml-lineage-audit` | Trace a model's full ML lineage and find gaps that would hide a future break |
| `model-invalidation-check` | Decide whether a specific upstream change invalidates a deployed model |
| `blast-radius` | Given a dataset or column, find every model and dashboard downstream |

## Relationship to the Python agent

Both call the same DataHub MCP tools and apply the same verdict rubric. The
difference is where the deterministic half lives: the CLI computes schema drift
from recorded baselines before the model reasons, while the plugin asks you what
changed. Use the CLI when you want reproducible, unattended checks.
