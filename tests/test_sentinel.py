"""Regression tests for the live Sentinel orchestration path.

The first live DataHub run exposed failures that unit tests around the parsers
could not catch. These tests pin the contracts between lineage summaries, the
orchestrator, CLI rendering, and DataHub's mutation-tool schemas.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from forecast_sentinel.agent.prompts import build_no_drift_prompt
from forecast_sentinel.agent.schemas import Decision, Verdict
from forecast_sentinel.agent.sentinel import (
    BLOCK_TAG,
    SENTINEL_TAG,
    WARN_TAG,
    AgentExecutionError,
    Sentinel,
    SentinelRun,
    _render_lineage,
)
from forecast_sentinel.cli import _render_run
from forecast_sentinel.config import (
    AgentConfig,
    AgentProvider,
    DataHubConfig,
    Mode,
    SentinelConfig,
)
from forecast_sentinel.datahub.mcp_client import READ_TOOLS, ToolInventory
from forecast_sentinel.datahub.ml_lineage import Asset, MLLineageGraph
from forecast_sentinel.datahub.urns import dataset_urn, ml_model_urn, parse_urn
from forecast_sentinel.snapshots import DriftKind, FieldSpec, Snapshot, SnapshotStore

MODEL = ml_model_urn("mlflow", "demand-forecast-v3")
RAW = dataset_urn("snowflake", "sales.raw_sales")
DEPLOYMENT = "urn:li:mlModelDeployment:(urn:li:dataPlatform:sagemaker,planning-api,PROD)"


def _config(tmp_path) -> SentinelConfig:
    return SentinelConfig(
        datahub=DataHubConfig(mode=Mode.SELFHOSTED, gms_url="http://localhost:8080"),
        agent=AgentConfig(),
        state_dir=tmp_path,
    )


def _verdict(decision: Decision = Decision.BLOCK) -> Verdict:
    return Verdict(
        model_urn=MODEL,
        decision=decision,
        headline="The deployed model is invalidated.",
        reasoning="holiday_flag changed from INT to VARCHAR on a training input.",
        recommended_actions=["Retrain before the next scoring run."],
    )


def _run() -> SentinelRun:
    graph = MLLineageGraph(root=parse_urn(MODEL), max_hops=4)
    graph.add(Asset(urn=parse_urn(RAW), name="raw_sales"), direction="upstream")
    return SentinelRun(
        model_urn=MODEL,
        model_label="demand-forecast-v3",
        started_at="2026-07-27T10:00:00+00:00",
        graph=graph,
        verdict=_verdict(),
    )


def test_lineage_render_uses_the_current_summary_contract():
    rendered = _render_lineage(_run().graph)
    assert "search bounded at 4 hops" in rendered


def test_cli_render_uses_the_current_summary_contract(capsys):
    _render_run(_run())
    assert "max hops 4" in capsys.readouterr().out


def test_first_run_prompt_does_not_claim_a_drift_comparison_happened():
    prompt = build_no_drift_prompt(
        model_urn=MODEL,
        model_label="demand-forecast-v3",
        lineage_summary="one upstream dataset",
        baseline_created=True,
        comparison_performed=False,
    )
    assert "no drift comparison was possible" in prompt
    assert "No upstream schema drift was detected" not in prompt


def test_default_agent_budget_is_safe_for_non_streaming_tool_runner(tmp_path):
    assert _config(tmp_path).agent.max_tokens == 8_000


def test_openai_is_primary_and_anthropic_is_retained_as_fallback(tmp_path):
    agent = _config(tmp_path).agent
    assert agent.provider is AgentProvider.AUTO
    assert agent.openai_model == "gpt-5.6"
    assert agent.gemini_model == "gemini-3.6-flash"
    assert agent.anthropic_model == "claude-opus-5"


def test_provider_specific_model_environment_is_loaded(tmp_path, monkeypatch):
    monkeypatch.setenv("DATAHUB_MODE", "selfhosted")
    monkeypatch.setenv("DATAHUB_GMS_URL", "http://localhost:8080")
    monkeypatch.setenv("SENTINEL_PROVIDER", "openai")
    monkeypatch.setenv("SENTINEL_OPENAI_MODEL", "gpt-test")
    monkeypatch.setenv("SENTINEL_GEMINI_MODEL", "gemini-test")
    monkeypatch.setenv("SENTINEL_ANTHROPIC_MODEL", "claude-test")
    monkeypatch.delenv("SENTINEL_MODEL", raising=False)

    config = SentinelConfig.from_env(tmp_path / "missing.env")

    assert config.agent.provider is AgentProvider.OPENAI
    assert config.agent.openai_model == "gpt-test"
    assert config.agent.gemini_model == "gemini-test"
    assert config.agent.anthropic_model == "claude-test"


class FakeOpenAIResponses:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    async def create(self, **request):
        self.requests.append(request)
        if len(self.requests) == 1:
            return SimpleNamespace(
                id="response-1",
                usage=SimpleNamespace(input_tokens=10, output_tokens=2),
                output=[
                    SimpleNamespace(
                        type="function_call",
                        name="search",
                        call_id="call-search",
                        arguments=json.dumps({"query": MODEL}),
                    )
                ],
            )
        return SimpleNamespace(
            id="response-2",
            usage=SimpleNamespace(input_tokens=4, output_tokens=3),
            output=[
                SimpleNamespace(
                    type="function_call",
                    name="emit_verdict",
                    call_id="call-verdict",
                    arguments=json.dumps(
                        {
                            "decision": "BLOCK",
                            "headline": "The deployed model is invalidated.",
                            "reasoning": "The changed input type breaks its encoder.",
                            "recommended_actions": ["Retrain the model."],
                        }
                    ),
                )
            ],
        )


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.responses = FakeOpenAIResponses()


class ReadOnlyMCP:
    def __init__(self) -> None:
        self.inventory = ToolInventory(names=("search",))
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def openai_tools(self, *, include_writes: bool = True) -> list[dict[str, Any]]:
        assert not include_writes
        return [
            {
                "type": "function",
                "name": "search",
                "description": "Search DataHub.",
                "parameters": {"type": "object", "properties": {}},
            }
        ]

    async def call(self, name: str, arguments: dict[str, Any] | None = None) -> dict:
        self.calls.append((name, arguments or {}))
        return {"urn": MODEL, "name": "demand-forecast-v3"}


async def test_openai_responses_loop_executes_local_tools_and_emits_typed_verdict(tmp_path):
    client = FakeOpenAIClient()
    sentinel = Sentinel(_config(tmp_path), openai=client)  # type: ignore[arg-type]
    mcp = ReadOnlyMCP()

    verdict, usage = await sentinel._judge_openai(mcp, _run(), "Assess the model.")

    assert verdict is not None
    assert verdict.decision is Decision.BLOCK
    assert usage == {"input_tokens": 14, "output_tokens": 5}
    assert mcp.calls == [("search", {"query": MODEL})]
    continuation = client.responses.requests[1]
    assert continuation["previous_response_id"] == "response-1"
    assert continuation["input"] == [
        {
            "type": "function_call_output",
            "call_id": "call-search",
            "output": json.dumps({"urn": MODEL, "name": "demand-forecast-v3"}),
        }
    ]


class FakeGeminiCompletions:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    async def create(self, **request):
        self.requests.append(request)
        if len(self.requests) == 1:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-search",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": json.dumps({"query": MODEL}),
                        },
                    }
                ],
            }
            usage = SimpleNamespace(prompt_tokens=10, completion_tokens=2)
        else:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-verdict",
                        "type": "function",
                        "function": {
                            "name": "emit_verdict",
                            "arguments": json.dumps(
                                {
                                    "decision": "BLOCK",
                                    "headline": "The deployed model is invalidated.",
                                    "reasoning": "The changed input type breaks its encoder.",
                                    "recommended_actions": ["Retrain the model."],
                                }
                            ),
                        },
                    }
                ],
            }
            usage = SimpleNamespace(prompt_tokens=4, completion_tokens=3)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=usage,
        )


class FakeGeminiClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=FakeGeminiCompletions())


async def test_gemini_chat_loop_reuses_read_only_tools_and_emits_typed_verdict(tmp_path):
    client = FakeGeminiClient()
    sentinel = Sentinel(_config(tmp_path), gemini=client)  # type: ignore[arg-type]
    mcp = ReadOnlyMCP()

    verdict, usage = await sentinel._judge_gemini(mcp, _run(), "Assess the model.")

    assert verdict is not None
    assert verdict.decision is Decision.BLOCK
    assert usage == {"input_tokens": 14, "output_tokens": 5}
    assert mcp.calls == [("search", {"query": MODEL})]
    first_request = client.chat.completions.requests[0]
    assert first_request["model"] == "gemini-3.6-flash"
    assert first_request["max_tokens"] == 8_000
    assert {tool["function"]["name"] for tool in first_request["tools"]} == {
        "search",
        "emit_verdict",
    }
    continuation = client.chat.completions.requests[1]["messages"]
    assert {
        "role": "tool",
        "tool_call_id": "call-search",
        "content": json.dumps({"urn": MODEL, "name": "demand-forecast-v3"}),
    } in continuation


async def test_explicit_gemini_mode_does_not_attempt_paid_providers(tmp_path, monkeypatch):
    base = _config(tmp_path)
    config = SentinelConfig(
        datahub=base.datahub,
        agent=AgentConfig(provider=AgentProvider.GEMINI),
        state_dir=base.state_dir,
    )
    sentinel = Sentinel(config, gemini=object())  # type: ignore[arg-type]
    attempted: list[str] = []

    async def pass_gemini(*args, **kwargs):
        attempted.append("gemini")
        return _verdict(Decision.WARN), {"input_tokens": 8, "output_tokens": 2}

    monkeypatch.setattr(sentinel, "_judge_gemini", pass_gemini)

    run = _run()
    verdict = await sentinel._judge(object(), run, allow_writes=False)  # type: ignore[arg-type]

    assert verdict is not None
    assert verdict.decision is Decision.WARN
    assert attempted == ["gemini"]
    assert "Judgement provider: Gemini" in run.notes[0]


async def test_openai_failure_retries_with_anthropic_fallback(tmp_path, monkeypatch):
    config = _config(tmp_path)
    sentinel = Sentinel(config, openai=object(), anthropic=object())  # type: ignore[arg-type]
    attempted: list[str] = []

    async def fail_openai(*args, **kwargs):
        attempted.append("openai")
        raise AgentExecutionError("temporary OpenAI failure")

    async def pass_anthropic(*args, **kwargs):
        attempted.append("anthropic")
        return _verdict(Decision.WARN), {"input_tokens": 8, "output_tokens": 2}

    monkeypatch.setattr(sentinel, "_judge_openai", fail_openai)
    monkeypatch.setattr(sentinel, "_judge_anthropic", pass_anthropic)

    run = _run()
    verdict = await sentinel._judge(object(), run, allow_writes=False)  # type: ignore[arg-type]

    assert verdict is not None
    assert verdict.decision is Decision.WARN
    assert attempted == ["openai", "anthropic"]
    assert run.token_usage == {"input_tokens": 8, "output_tokens": 2}
    assert "trying Anthropic fallback" in run.notes[0]
    assert "Judgement provider: Anthropic" in run.notes[1]


class RecordingMCP:
    def __init__(self) -> None:
        self.inventory = ToolInventory(
            names=("remove_tags", "add_tags", "update_description", "save_document")
        )
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call(self, name: str, arguments: dict[str, Any] | None = None) -> dict:
        self.calls.append((name, arguments or {}))
        return {"success": True}


async def test_write_back_matches_live_datahub_tool_schemas(tmp_path):
    sentinel = Sentinel(_config(tmp_path), anthropic=object())  # type: ignore[arg-type]
    mcp = RecordingMCP()

    writes = await sentinel._record_verdict(mcp, _run())  # type: ignore[arg-type]

    by_name = {name: arguments for name, arguments in mcp.calls}
    assert by_name["remove_tags"] == {
        "tag_urns": [SENTINEL_TAG, WARN_TAG],
        "entity_urns": [MODEL],
    }
    assert by_name["add_tags"] == {
        "tag_urns": [BLOCK_TAG],
        "entity_urns": [MODEL],
    }
    assert by_name["update_description"]["entity_urn"] == MODEL
    assert by_name["update_description"]["operation"] == "append"
    assert by_name["save_document"]["document_type"] == "Analysis"
    assert by_name["save_document"]["related_assets"] == [MODEL]
    assert all(write.succeeded for write in writes)


async def test_check_runs_drift_to_write_back_without_contract_errors(tmp_path, monkeypatch):
    import forecast_sentinel.agent.sentinel as sentinel_module

    mutation_calls: list[tuple[str, dict[str, Any]]] = []

    class FakeDataHubMCP:
        def __init__(self, *args, **kwargs) -> None:
            self.inventory = ToolInventory(
                names=(
                    *READ_TOOLS,
                    "remove_tags",
                    "add_tags",
                    "update_description",
                    "save_document",
                )
            )

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info) -> None:
            return None

        async def call(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
            arguments = arguments or {}
            if name == "get_lineage":
                if arguments["upstream"]:
                    return {
                        "upstreams": {
                            "searchResults": [{"entity": {"urn": RAW, "name": "raw_sales"}}],
                            "hasMore": False,
                        }
                    }
                return {
                    "downstreams": {
                        "searchResults": [{"entity": {"urn": DEPLOYMENT, "name": "planning-api"}}],
                        "hasMore": False,
                    }
                }
            if name == "list_schema_fields":
                return {"fields": [{"fieldPath": "holiday_flag", "nativeDataType": "VARCHAR"}]}
            mutation_calls.append((name, arguments))
            return {"success": True}

        def anthropic_tools(self, *, include_writes: bool = True) -> list[Any]:
            return []

    class StubJudgeSentinel(Sentinel):
        async def _judge(self, mcp, run, *, allow_writes):
            self.rendered_lineage = _render_lineage(run.graph)
            return _verdict()

    monkeypatch.setattr(sentinel_module, "DataHubMCP", FakeDataHubMCP)
    config = _config(tmp_path)
    SnapshotStore(config.baseline_dir).save(
        Snapshot(
            model_urn=MODEL,
            captured_at="2026-07-26T10:00:00+00:00",
            datasets={RAW: [FieldSpec("holiday_flag", native_type="INT")]},
        )
    )
    sentinel = StubJudgeSentinel(config, anthropic=object())  # type: ignore[arg-type]

    run = await sentinel.check(MODEL, write_back=True)

    assert run.decision is Decision.BLOCK
    assert len(run.drift) == 1
    assert run.drift[0].kind is DriftKind.TYPE_CHANGED
    assert "search bounded at 4 hops" in sentinel.rendered_lineage
    assert [name for name, _ in mutation_calls] == [
        "remove_tags",
        "add_tags",
        "update_description",
        "save_document",
    ]
