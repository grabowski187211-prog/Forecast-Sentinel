# Sample outputs

The hackathon rules recommend including sample outputs. **These are placeholders
until an end-to-end run against a live DataHub has happened** — see
[`../docs/SUBMISSION.md`](../docs/SUBMISSION.md).

Do not treat anything here as evidence of a complete agent run. The verified
claims today are: the local tests and lint pass, the seeder constructs every
entity, and a prior live DataHub run confirmed lineage, baseline capture, and
deterministic schema-drift detection. Authenticated model judgement (OpenAI
primary, Anthropic fallback) and catalog write-back still need fresh checked-in
output; the four mutation schemas have separately passed a live typed-verdict
smoke test.

## What will land here

| File | Produced by |
|---|---|
| `verdict_block.json` | `sentinel check "$MODEL" --json examples/verdict_block.json` after `--break-schema` |
| `verdict_ok.json` | Same command with the schema intact |
| `sentinel_demand-forecast-v3.html` | Copied from `.sentinel/runs/` after a check |
| `terminal_block.txt` | `sentinel check "$MODEL" | tee examples/terminal_block.txt` |

## Regenerating

```bash
./scripts/bootstrap_datahub.sh
MODEL='urn:li:mlModel:(urn:li:dataPlatform:mlflow,demand-forecast-v3,PROD)'

sentinel baseline "$MODEL"
sentinel check "$MODEL" --json examples/verdict_ok.json

python scripts/seed_ml_demo.py --break-schema
sentinel check "$MODEL" --json examples/verdict_block.json
cp .sentinel/runs/*.html examples/
```

## Before committing anything here

Verdicts embed catalog content — dataset names, descriptions, owner URNs. Read
each file before committing. Scrub anything from a real internal catalog; the
demo graph from `seed_ml_demo.py` is synthetic and safe to publish as-is.
