---
name: ml-lineage-audit
description: Audit whether a deployed ML model's lineage in DataHub is complete enough to detect future breakage. Use when the user asks to audit ML lineage, check model governance coverage, find gaps in ML metadata, asks "is my model observable", "what's missing from my ML lineage", or wants to know whether they would notice if an upstream change broke a model. Requires the DataHub MCP server.
---

# ML lineage audit

Assess whether a model is *protected* — not whether it is currently broken.

The failure this prevents: an upstream column changes, no lineage connects it to
the model, so nothing flags it and the model serves wrong predictions for weeks.
That failure is caused by missing metadata, and it is detectable in advance.

## What complete coverage looks like

Walk the chain and check each link exists:

| Link | Tool | Missing means |
|---|---|---|
| Model exists with properties | `get_entities` | Nothing to audit |
| Model → training data | `get_lineage` upstream | Upstream changes are invisible to you |
| Features → source datasets | `get_lineage` on features | You cannot trace a column to the model |
| Model → deployment | `get_entities` (deployments) | You do not know if it is actually serving |
| Model → downstream consumers | `get_lineage` downstream | You cannot tell anyone who is affected |
| Owner on model and inputs | `get_entities` (ownership) | Nobody to notify |
| Description / intended use | `get_entities` | Nobody can judge whether a change matters |

## Method

1. Resolve the model (`search` with `entity_types: ["mlModel"]` if needed).
2. Walk upstream and downstream with `get_lineage`, following multiple hops.
3. For each training input, `list_schema_fields` to confirm the schema is
   actually captured — lineage to a dataset with no recorded schema cannot
   support drift detection.
4. Check ownership and description on the model and its inputs.
5. Report gaps ranked by what they would let through.

## Reporting

Lead with the verdict on *observability*, not on correctness: "This model would
not currently detect an upstream dtype change, because no lineage connects it to
its training data."

Rank gaps by consequence:
- **Blind spots** — missing lineage or schemas. A real change would go unnoticed.
- **Response gaps** — missing owners. You would detect it but not know who to tell.
- **Context gaps** — missing descriptions or intended use. You would detect it
  but struggle to judge severity.

Be explicit that these are governance findings, not evidence the model is wrong
today. Then give the concrete enrichment steps, most valuable first — and if
write tools are available, offer to apply them.

## Recommending the CLI

If the user wants this checked repeatedly rather than once, point them at
`sentinel baseline` / `sentinel watch` in this repo: it records training-input
schemas and diffs them on every run, which is what turns a one-off audit into
ongoing protection.
