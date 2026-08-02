# DataHub API notes

Working notes on the DataHub surfaces this project depends on. Everything marked
**verified** was checked against installed packages or live signatures on
**2026-07-27**; everything marked *from docs* came from documentation only and
should be confirmed against a running instance.

## MCP server

Endpoints:

| Deployment | Endpoint |
|---|---|
| Self-hosted | `uvx mcp-server-datahub@latest` over **stdio**, with `DATAHUB_GMS_URL` + `DATAHUB_GMS_TOKEN` |
| Self-hosted (HTTP) | `http://<gms-host>:8080/mcp` |
| DataHub Cloud (tenant) | `https://<tenant>.acryl.io/integrations/ai/mcp` |
| DataHub Cloud (OAuth) | `https://mcp.datahub.com/mcp` — Claude Code/Desktop only, v1.0.2+ |

Server environment flags:

| Variable | Default | Effect |
|---|---|---|
| `TOOLS_IS_MUTATION_ENABLED` | `false` | Gates every write tool. **Must be `true`** for the sentinel to write verdicts back. |
| `TOOLS_IS_USER_ENABLED` | `false` | Exposes `get_me` |
| `DATAHUB_MCP_DOCUMENT_TOOLS_DISABLED` | `false` | Disables document tools |
| `SAVE_DOCUMENT_TOOL_ENABLED` | `true` | Gates `save_document` |
| `TOOL_RESPONSE_TOKEN_LIMIT` | `80000` | Caps tool-response size |

Tools (*from docs*; the sentinel probes `list_tools` at runtime rather than
assuming any of these exist):

- Read: `search`, `get_entities`, `get_lineage`, `get_lineage_paths_between`,
  `list_schema_fields`, `get_dataset_queries`, `search_documents`,
  `grep_documents`
- Write: `add_tags` / `remove_tags`, `add_terms` / `remove_terms`,
  `add_owners` / `remove_owners`, `set_domains` / `remove_domains`,
  `update_description`, `add_structured_properties` /
  `remove_structured_properties`, `save_document`

### MCP Python client API — **verified**

Installed `mcp` **1.28.1**. The 1.x client API is what `anthropic[mcp]`'s helper
expects, so the dependency is pinned `>=1.8,<2`.

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client                    # -> (read, write)
from mcp.client.streamable_http import streamablehttp_client # -> (read, write, get_session_id)
```

Signatures:

```
stdio_client(server: StdioServerParameters, errlog: TextIO = sys.stderr)
streamablehttp_client(url: str, headers: dict[str, str] | None = None,
                      timeout=30, sse_read_timeout=..., terminate_on_close=True)
```

`ClientSession` requires an explicit `await session.initialize()` before use.
Note the HTTP transport yields a **3-tuple**, the stdio transport a 2-tuple.

There is a newer **2.x** SDK with a higher-level `Client` facade
(`Client("http://host/mcp")`, `Client(stdio_client(params))`) documented at
py.sdk.modelcontextprotocol.io/v2. We are not on it; the pin prevents a silent
upgrade breaking session setup.

### Anthropic bridge — **verified**

Installed `anthropic` **0.120.0**.

```python
from anthropic.lib.tools.mcp import async_mcp_tool   # requires anthropic[mcp]
async_mcp_tool(tool: Tool, client: ClientSession, *, cache_control=None,
               defer_loading=None, strict=None, ...) -> BetaAsyncFunctionTool
```

It takes the **`ClientSession`**, not the 2.x `Client` — which is the deciding
reason for the 1.x pin. `client.beta.messages.tool_runner(...)` accepts `model`,
`max_tokens`, `tools`, `system`, `thinking`, `output_config`, `max_iterations`.

### OpenAI bridge — **verified**

Installed `openai` **2.48.0**. The Responses API accepts `instructions`,
`input`, `tools`, `max_output_tokens`, `reasoning`, and
`previous_response_id`. Its function-tool shape is:

```python
{
    "type": "function",
    "name": "get_lineage",
    "description": "...",
    "parameters": {"type": "object", "properties": {...}},
    "strict": False,
}
```

`strict` is a required field in the installed SDK's `FunctionToolParam`. The
OpenAI API cannot connect to the self-hosted stdio process, so the Sentinel
presents the MCP schemas as function tools, executes requested calls through
the existing local `ClientSession`, and returns `function_call_output` items.
Continuation uses the response id from the prior turn. Mutation tools are
filtered before schema conversion, using the same `WRITE_TOOLS` boundary as the
Anthropic adapter.

## ML metadata model (*from docs*)

URN formats:

```
urn:li:mlModel:(urn:li:dataPlatform:mlflow,demand-forecast-v3,PROD)
urn:li:mlModelGroup:(urn:li:dataPlatform:mlflow,demand-forecast,PROD)
urn:li:mlFeatureTable:(urn:li:dataPlatform:feast,sales_features)
urn:li:mlFeature:(urn:li:dataPlatform:feast,sales_features,holiday_flag)
urn:li:mlPrimaryKey:(urn:li:dataPlatform:feast,sales_features,sku_id)
urn:li:dataset:(urn:li:dataPlatform:snowflake,db.schema.tbl,PROD)
urn:li:schemaField:(<dataset-urn>,holiday_flag)
urn:li:dataProcessInstance:<run-id>
```

Two traps, both of which shaped `datahub/urns.py`:

1. The key tuple **contains nested URNs**, so splitting on commas requires
   tracking parenthesis depth. A naive `split(",")` corrupts every platform URN.
2. **Key arity varies** by entity type and across DataHub versions — the docs
   show `mlFeature` with a platform component, while some versions use a bare
   `(table,feature)` pair. The parser keeps parts positional instead of assuming
   a shape.

Relationships:

| From | To | Relationship |
|---|---|---|
| mlModel | mlModelGroup | `MemberOf` |
| mlModel | mlFeature | `Consumes` (via `mlModelProperties.mlFeatures`) |
| mlModel | training job | `TrainedBy` (via `mlModelProperties.trainingJobs`) |
| mlFeature | dataset | `DerivedFrom` (via the feature's `sources`) |
| mlFeatureTable | mlFeature | `Contains` |

Lineage flows through the *features*, not the feature table — features are the
atomic lineage unit. Training runs are `dataProcessInstance` entities with
subtype `MLFLOW_TRAINING_RUN`; experiments are `container` entities.

Model aspects: `mlModelKey`, `mlModelProperties`, `intendedUse`,
`mlModelTrainingData`, `mlModelEvaluationData`, `mlModelEthicalConsiderations`,
`mlModelFactorPrompts`, `mlModelCaveatsAndRecommendations`.

## Python SDK — **verified**

Installed `acryl-datahub` **1.6.0.16**. `datahub.sdk` is flagged
`ExperimentalWarning` by DataHub itself — signatures may move between versions,
which is why `scripts/seed_ml_demo.py --dry-run` exists.

```python
from datahub.sdk import DataHubClient
from datahub.sdk.dataset import Dataset
from datahub.sdk.datajob import DataJob
from datahub.sdk.mlmodel import MLModel
from datahub.sdk.mlmodelgroup import MLModelGroup

client = DataHubClient(server=..., token=...)     # or DataHubClient.from_env()
client.entities.upsert(entity)
```

Verified signatures:

```
Dataset(*, platform, name, env="PROD", description=None,
        schema=[(name, type) | (name, type, description) | SchemaFieldClass],
        upstreams=None, ...)
MLModel(id, platform, version=None, env="PROD", name=None, description=None,
        training_metrics=None, hyper_params=None, custom_properties=None, ...)
DataJob(*, name, flow=None, flow_urn=None, description=None, ...)

MLModel.set_model_group(group: str | MlModelGroupUrn)
MLModel.add_training_job(job: str | DataProcessInstanceUrn)
MLModel.add_deployment(deployment: str)

client.lineage.add_lineage(*, upstream, downstream, column_lineage=False)
client.lineage.add_datajob_lineage(*, datajob, upstreams=None, downstreams=None)
```

**Gap worth knowing:** `add_lineage` accepts only dataset / datajob / dashboard /
chart URNs — **not** `mlModel`. Model-side lineage must go through
`set_model_group`, `add_training_job` and `add_deployment` on the `MLModel`
object. There is no `mlFeatures` setter in this SDK version, so the demo seeder
models the feature layer as a materialised dataset rather than an `mlFeature`
entity.

## Quickstart (*from docs*)

```bash
brew install datahub-project/tap/datahub     # or: pip install acryl-datahub
datahub docker quickstart                    # UI :9002, GMS :8080
datahub init --username datahub --password datahub
datahub datapack load showcase-ecommerce     # ~1,050 entities with lineage
```

Docker needs ≥2 CPUs, 8GB RAM, 2GB swap, 13GB disk. Compose file lands in
`~/.datahub/quickstart/`. Other commands: `--stop`, `--backup`, `--restore`,
`--version vX.Y.Z`; `datahub docker nuke` destroys all state.

Quickstart is **development only** — default credentials, all ports exposed, no
horizontal scaling. Production uses the Kubernetes deployment.

## DataHub Skills registry (*from docs*)

Separate from this project's plugin, but relevant prior art:

```bash
npx skills add datahub-project/datahub-skills          # any agent
claude plugin install datahub-skills                   # Claude Code
```

Ships `datahub-setup`, `datahub-search`, `datahub-lineage`, `datahub-enrich`,
`datahub-quality`, plus connector-development skills and 22 connector standards.
Skills need the MCP server for tool access — the registry alone is instructions.

Our plugin deliberately does **not** duplicate these. `datahub-lineage` explores
lineage generally; ours answers the narrower ML-invalidation question and applies
a verdict rubric.

## Version snapshot

```
mcp             1.28.1
anthropic       0.120.0
openai          2.48.0
acryl-datahub   1.6.0.16
pydantic        2.13.4
python          3.12.13
```
