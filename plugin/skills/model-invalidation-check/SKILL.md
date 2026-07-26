---
name: model-invalidation-check
description: Decide whether a specific upstream data change invalidates a deployed ML model, using DataHub ML lineage. Use when the user says a column/table/feature changed and asks whether a model, forecast, or scoring job is still safe — or asks "is this model affected by", "did this break my model", "can I still trust these predictions". Requires the DataHub MCP server.
---

# Model invalidation check

Answer one question: **does this upstream change invalidate that deployed model?**

The change is a given. Your job is consequence, not detection.

## Before you start

Confirm the DataHub MCP tools are available (`search`, `get_entities`,
`get_lineage`). If they are not, stop and tell the user to configure the DataHub
MCP server — do not answer from general knowledge about ML. A guess here gets
pasted into an incident channel.

If the user named a model in prose rather than by URN, resolve it with `search`
(`entity_types: ["mlModel"]`) and confirm which one they mean before proceeding.

## Method

1. **Establish what the model consumes.** `get_entities` on the model URN, then
   `get_lineage` upstream. A change to a column the model never reads is not a
   risk, and saying so is a valid, useful answer.
2. **Prove the path.** Use `get_lineage_paths_between` from the changed asset to
   the model. Do not assume a connection because the names look related.
3. **Judge the mechanism.** Reason about *how* the change reaches the trained
   artefact:
   - dtype change (int→str, float→int) breaks encoding learned at training time
   - removed field the model consumes → null features or a pipeline error
   - widened nullability → nulls the model never saw; depends on imputation
   - added field the model does not consume → not a risk to *this* model
   - semantic change with the same dtype (units, currency, timezone) → the most
     dangerous case, because nothing errors
4. **Establish blast radius.** `get_lineage` downstream from the model for
   deployments, dashboards and datasets consuming its predictions.
5. **Check how the feature is computed.** `get_dataset_queries` on the feature
   dataset often reveals a cast or default that changes the answer.

## Verdict

State one of these plainly, in the first sentence:

- **BLOCK** — the deployed artefact is invalid; predictions are wrong or the
  pipeline will fail. Serving should stop or the model retrain first.
- **WARN** — still serving usable output, but needs human attention.
- **OK** — the change is real and provably does not affect this model.
- **UNKNOWN** — the catalog lacks the context to decide. Name exactly what
  metadata is missing. Do not guess to avoid this outcome.

Then: the lineage path the risk travels, the evidence you retrieved, and ordered
next steps.

## Evidence discipline

Every claim cites a tool result from this session. If you asserted something you
could not confirm, say so explicitly rather than dropping it.

Do not dress up catalog gaps as model risks. "No owner is set on the feature
table" is a governance finding, not a reason the forecast is wrong.

## Recording the finding

If write tools are available (`add_tags`, `update_description`), offer to record
the verdict on the model so the next person sees it. Ask first — tagging a
production model is a visible change to a shared catalog.
