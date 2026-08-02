# Setup

## Prerequisites

| Requirement | Why | Check |
|---|---|---|
| Python 3.10–3.12 | `acryl-datahub` and the MCP server lack 3.13+ wheels | `python3 --version` |
| [uv](https://docs.astral.sh/uv/) | venv + `uvx` runs the MCP server | `uv --version` |
| Docker | Only for a **local** DataHub | `docker info` |
| OpenAI API key | Primary agent path | `$OPENAI_API_KEY` |
| Anthropic credentials | Optional fallback | `$ANTHROPIC_API_KEY`, `$ANTHROPIC_AUTH_TOKEN`, or an SDK profile |

Docker needs ≥2 CPUs, 8GB RAM, 2GB swap and 13GB free disk allocated to it. If
you are using DataHub Cloud, Docker is not needed at all.

## 1. Install

```bash
uv venv --python 3.12
uv pip install -e ".[dev,cli]"      # tests + DataHub SDK/CLI + demo seeder
source .venv/bin/activate           # or prefix commands with .venv/bin/
```

Verify:

```bash
pytest        # unit and orchestration-contract tests; no DataHub required
```

## 2. Configure

```bash
cp .env.example .env
```

### Self-hosted

```ini
DATAHUB_MODE=selfhosted
DATAHUB_GMS_URL=http://localhost:8080
DATAHUB_GMS_TOKEN=            # empty is fine for quickstart defaults
TOOLS_IS_MUTATION_ENABLED=true
```

### DataHub Cloud

```ini
DATAHUB_MODE=cloud
DATAHUB_TENANT_URL=https://<tenant>.acryl.io
DATAHUB_TOKEN=<personal access token>
TOOLS_IS_MUTATION_ENABLED=true
```

Generate the token in the DataHub UI under **Settings → Access Tokens**.

`TOOLS_IS_MUTATION_ENABLED=true` is what lets the sentinel write verdicts back.
Without it the run still works, but findings stay in the terminal and the HTML
report — `sentinel doctor` will tell you so.

### Model provider

The default `auto` mode prefers OpenAI and retries with Anthropic if the OpenAI
request fails or does not emit a valid verdict:

```ini
SENTINEL_PROVIDER=auto
OPENAI_API_KEY=<openai-api-key>
SENTINEL_OPENAI_MODEL=gpt-5.6

# Optional fallback
ANTHROPIC_API_KEY=
ANTHROPIC_AUTH_TOKEN=
SENTINEL_ANTHROPIC_MODEL=claude-opus-5
SENTINEL_EFFORT=high
```

Set `SENTINEL_PROVIDER=anthropic` to skip OpenAI. The old `SENTINEL_MODEL`
variable remains compatible: Claude model names are routed to Anthropic and
other model names to OpenAI, but the provider-specific names are clearer.

## 3. Bring up a local DataHub (self-hosted only)

Scripted:

```bash
./scripts/bootstrap_datahub.sh
```

Or by hand:

```bash
brew install datahub-project/tap/datahub
datahub docker quickstart                      # first run pulls several GB
datahub init --username datahub --password datahub
datahub datapack load showcase-ecommerce       # ~1,050 entities with lineage
```

UI at <http://localhost:9002> (`datahub` / `datahub`); GMS at
<http://localhost:8080>.

> Quickstart is a **development** deployment: default credentials, all ports
> exposed, no horizontal scaling. Do not point it at anything real.

### Seed the ML demo graph

The showcase datapack is rich in datasets but thin in `mlModel` entities, so the
repo emits its own minimal ML path:

```bash
python scripts/seed_ml_demo.py --dry-run    # verify SDK compatibility, emit nothing
python scripts/seed_ml_demo.py              # emit the graph
```

This creates `raw_sales` → `feat_seasonality` → `train_demand_forecast` →
`demand-forecast-v3` → `planning-api`, plus the three tags used for Sentinel
status write-back.

## 4. Verify connectivity

```bash
sentinel doctor
```

Expected: the configuration table, then `Connected. N MCP tools exposed.` plus
the write-tool state and separate OpenAI/Anthropic credential checks.

## 5. Run the demo

```bash
MODEL='urn:li:mlModel:(urn:li:dataPlatform:mlflow,demand-forecast-v3,PROD)'

sentinel models                              # confirm the model is in the catalog
sentinel baseline "$MODEL"                   # record training-input schemas
python scripts/seed_ml_demo.py --break-schema   # holiday_flag INT -> VARCHAR
sentinel check "$MODEL"                      # detect, judge, write back
```

The last command should produce a **BLOCK** verdict and exit 2.

## Troubleshooting

**`uvx not found`** — install uv; the self-hosted MCP server runs via
`uvx mcp-server-datahub@latest`.

**`could not connect to the DataHub MCP server (selfhosted)`** — is
`datahub docker quickstart` up? `curl http://localhost:8080/health`. The first
`uvx` invocation also downloads the server, which can take a minute.

**`tool 'add_tags' is not exposed`** — `TOOLS_IS_MUTATION_ENABLED` is not `true`
on the MCP server. It is read from `.env` and passed to the subprocess, so
restart the run after changing it. Write-back also uses `remove_tags`,
`update_description`, and optionally `save_document`; failed mutations are
listed individually in the terminal and report.

**`No mlModel entities found`** — run `python scripts/seed_ml_demo.py`. The
showcase datapack alone does not include ML models.

**`No upstream datasets found for this model`** — the model exists but has no
training-data lineage in DataHub. There is nothing to baseline; the sentinel will
review lineage coverage instead and tell you what is missing.

**Verdict is always UNKNOWN** — usually genuine: the catalog lacks the lineage to
decide. Check `sentinel check --json out.json` and read the agent's
`unverified_claims`, which name exactly what metadata was missing.

**`No OpenAI credentials`** — set `OPENAI_API_KEY` in `.env`. In `auto` mode the
Sentinel will otherwise try Anthropic as the fallback.

**`No Anthropic credentials`** — this is acceptable when OpenAI is configured.
To enable fallback, set `ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN`, or
configure an SDK profile. DataHub connectivity alone is sufficient for `models`
and `baseline`; `check` and `watch` need at least one model provider.

**Python 3.13+ resolution errors** — create the venv with
`uv venv --python 3.12`. `acryl-datahub` does not publish 3.13 wheels yet.

## Claude Code plugin (optional)

```bash
claude mcp add datahub \
  -e DATAHUB_GMS_URL="http://localhost:8080" \
  -e TOOLS_IS_MUTATION_ENABLED="true" \
  -- uvx mcp-server-datahub@latest

claude mcp list                          # confirm it registered
claude plugin install forecast-sentinel --from ./plugin
```
