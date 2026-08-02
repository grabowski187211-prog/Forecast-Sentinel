# Sample outputs

These are genuine outputs from the authenticated Gemini free-tier run against
the local DataHub 1.5.0.6 demo on 2026-08-02 — see
[`../docs/SUBMISSION.md`](../docs/SUBMISSION.md) for the remaining submission
work.

The run detected `holiday_flag: INT → VARCHAR`, received a typed `BLOCK` verdict
from `gemini-3.6-flash`, and completed status-tag cleanup, invalidation tagging,
description append, and linked-document save through DataHub MCP. The JSON
records zero unverified claims.

## Checked-in outputs

| File | Produced by |
|---|---|
| `verdict_block.json` | Structured run, drift, verdict, write-backs, notes, and usage |
| `sentinel_demand-forecast-v3_block.html` | Static self-contained report copied from `.sentinel/runs/` |
| `terminal_block.txt` | Clean transcript of the blocking run |

## Regenerating

```bash
./scripts/bootstrap_datahub.sh
MODEL='urn:li:mlModel:(urn:li:dataPlatform:mlflow,demand-forecast-v3,PROD)'

python scripts/seed_ml_demo.py
sentinel baseline "$MODEL"
python scripts/seed_ml_demo.py --break-schema
sentinel check "$MODEL" --json examples/verdict_block.json
cp .sentinel/runs/sentinel_mlflow_demand-forecast-v3__PROD_.html \
  examples/sentinel_demand-forecast-v3_block.html
```

## Before committing anything here

Verdicts embed catalog content — dataset names, descriptions, owner URNs. Read
each file before committing. Scrub anything from a real internal catalog; the
demo graph from `seed_ml_demo.py` is synthetic and safe to publish as-is.
