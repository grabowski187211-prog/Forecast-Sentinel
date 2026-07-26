---
name: blast-radius
description: Given a dataset, table or column, find every ML model, deployment and dashboard downstream of it in DataHub. Use when the user asks "what depends on this", "what breaks if I change this column", "who consumes this table", "what's downstream of", or is planning a schema migration and wants the impact first. Requires the DataHub MCP server.
---

# Blast radius

Given an asset, enumerate what depends on it — before someone changes it.

This is the question asked *before* a migration, whereas
`model-invalidation-check` is asked after something already changed.

## Method

1. **Resolve the asset.** If the user gave a name rather than a URN, `search`
   for it and confirm which one. If they named a *column*, resolve the dataset
   first, then confirm the column exists via `list_schema_fields` — a
   misremembered column name silently produces an empty blast radius, which
   reads as "safe to change".
2. **Walk downstream.** `get_lineage` with downstream direction. Repeat on each
   result to follow multi-hop paths: a table feeds a feature, which feeds a
   model, which feeds a deployment. One hop is not the answer.
3. **Classify what you find**, because the consequences differ:
   - `mlModel` / `mlModelDeployment` — a trained artefact and live serving
   - `dataset` — a downstream table that will carry the change forward
   - `dashboard` / `chart` — humans reading numbers
   - `dataJob` / `dataFlow` — pipelines that may fail outright
4. **Find the owners.** `get_entities` on the affected assets for ownership. A
   blast radius without owners is not actionable — nobody knows who to tell.
5. **Note where lineage is thin.** If a critical asset has no downstream edges
   at all, that is suspicious rather than reassuring: it usually means lineage
   was never captured, not that nothing consumes it. Say which it is.

## Reporting

Lead with the count and the most severe category: *"9 assets downstream,
including 2 production models and the exec demand dashboard."*

Then group by category, and within each name the owner. Give the column-level
path where you have it — "changing `holiday_flag` specifically affects…" is far
more useful than a table-level list.

Close with what you could not determine. Bounded traversals and missing lineage
both produce incomplete pictures, and an incomplete blast radius presented as
complete is how migrations break production.
