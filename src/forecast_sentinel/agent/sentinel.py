"""The sentinel run: deterministic detection, then agentic judgement.

The split is deliberate. Detecting that a column changed type is a diff — code
does it faster and more reliably than a model. Deciding whether that change
invalidates a *trained artefact* requires reading the lineage, understanding how
the feature is consumed, and weighing consequence. That is the agent's half.

Flow for `Sentinel.check(model_urn)`:

    resolve model -> walk ML lineage -> snapshot training inputs
        -> diff vs baseline -> [drift?] -> agent judges -> verdict
        -> write verdict back into DataHub
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from anthropic import AsyncAnthropic, beta_async_tool
from openai import AsyncOpenAI

from forecast_sentinel.agent.prompts import (
    SYSTEM_PROMPT,
    build_check_prompt,
    build_no_drift_prompt,
)
from forecast_sentinel.agent.schemas import Decision, Verdict, WriteBack
from forecast_sentinel.config import AgentProvider, SentinelConfig
from forecast_sentinel.datahub.mcp_client import (
    WRITE_TOOLS,
    DataHubMCP,
    MCPConnectionError,
)
from forecast_sentinel.datahub.ml_lineage import Asset, MLLineageGraph, build_ml_lineage
from forecast_sentinel.datahub.urns import parse_urn
from forecast_sentinel.snapshots import (
    DriftEvent,
    SnapshotStore,
    capture_snapshot,
    diff_snapshots,
)

SENTINEL_TAG = "urn:li:tag:sentinel-reviewed"
BLOCK_TAG = "urn:li:tag:model-invalidated"
WARN_TAG = "urn:li:tag:model-needs-review"
STATUS_TAGS = (SENTINEL_TAG, BLOCK_TAG, WARN_TAG)
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

_OPENAI_VERDICT_TOOL: dict[str, Any] = {
    "type": "function",
    "name": "emit_verdict",
    "strict": False,
    "description": "Record the final typed verdict for this model. Call exactly once.",
    "parameters": {
        "type": "object",
        "properties": {
            "decision": {"type": "string", "enum": ["BLOCK", "WARN", "OK", "UNKNOWN"]},
            "headline": {"type": "string"},
            "reasoning": {"type": "string"},
            "risks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "severity": {
                            "type": "string",
                            "enum": ["low", "medium", "high", "critical"],
                        },
                        "mechanism": {"type": "string"},
                        "affected_urn": {"type": ["string", "null"]},
                        "evidence": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["title", "severity", "mechanism"],
                },
            },
            "recommended_actions": {"type": "array", "items": {"type": "string"}},
            "downstream_at_risk": {"type": "array", "items": {"type": "string"}},
            "confidence": {
                "type": ["string", "null"],
                "enum": ["low", "medium", "high", "critical", None],
            },
            "unverified_claims": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["decision", "headline", "reasoning"],
    },
}


class AgentExecutionError(RuntimeError):
    """Raised when the model tool-runner cannot complete a judgement."""

    def __init__(self, message: str, *, usage: dict[str, int] | None = None) -> None:
        super().__init__(message)
        self.usage = usage or {"input_tokens": 0, "output_tokens": 0}


@dataclass
class SentinelRun:
    """Everything one check produced — the report and the CLI render from this."""

    model_urn: str
    model_label: str
    started_at: str
    graph: MLLineageGraph | None = None
    drift: list[DriftEvent] = field(default_factory=list)
    verdict: Verdict | None = None
    write_backs: list[WriteBack] = field(default_factory=list)
    baseline_created: bool = False
    baseline_captured_at: str | None = None
    notes: list[str] = field(default_factory=list)
    token_usage: dict[str, int] = field(default_factory=dict)

    @property
    def decision(self) -> Decision:
        if self.verdict is None:
            return Decision.UNKNOWN
        return self.verdict.decision

    @property
    def should_fail_build(self) -> bool:
        return self.decision is Decision.BLOCK

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_urn": self.model_urn,
            "model_label": self.model_label,
            "started_at": self.started_at,
            "decision": self.decision.value,
            "baseline_created": self.baseline_created,
            "baseline_captured_at": self.baseline_captured_at,
            "lineage": self.graph.summary() if self.graph else None,
            "drift": [event.to_dict() for event in self.drift],
            "verdict": self.verdict.to_dict() if self.verdict else None,
            "write_backs": [wb.model_dump(mode="json") for wb in self.write_backs],
            "notes": self.notes,
            "token_usage": self.token_usage,
        }


class Sentinel:
    """Orchestrates one or more model checks against a DataHub instance."""

    def __init__(
        self,
        config: SentinelConfig,
        *,
        openai: AsyncOpenAI | None = None,
        gemini: AsyncOpenAI | None = None,
        anthropic: AsyncAnthropic | None = None,
    ) -> None:
        self._config = config
        # Clients are lazy so baseline-only commands do not require model
        # credentials. Injection keeps both provider loops independently testable.
        self._openai = openai
        self._gemini = gemini
        self._anthropic = anthropic
        self._store = SnapshotStore(config.baseline_dir)
        self._server_log = config.mcp_server_log

    # --- public API ---------------------------------------------------------

    async def capture_baseline(self, model_urn: str) -> SentinelRun:
        """Record the current schema of a model's training inputs as the baseline."""
        run = _new_run(model_urn)
        async with DataHubMCP(self._config.datahub, server_log=self._server_log) as mcp:
            graph = await build_ml_lineage(mcp, model_urn)
            run.graph = graph
            run.model_label = _label_for(graph, model_urn)

            dataset_urns = _training_dataset_urns(graph)
            if not dataset_urns:
                run.notes.append(
                    "No upstream datasets found for this model. Either the model has no "
                    "training-data lineage recorded in DataHub, or the URN is wrong. "
                    "Nothing to baseline."
                )
                return run

            snapshot = await capture_snapshot(mcp, model_urn, dataset_urns)
            path = self._store.save(snapshot)
            run.baseline_created = True
            run.baseline_captured_at = snapshot.captured_at
            run.notes.append(
                f"Baseline recorded for {len(snapshot.datasets)} training input(s) -> {path}"
            )
        return run

    async def check(self, model_urn: str, *, write_back: bool | None = None) -> SentinelRun:
        """Detect drift, judge consequence, and optionally record the verdict."""
        run = _new_run(model_urn)

        async with DataHubMCP(self._config.datahub, server_log=self._server_log) as mcp:
            if missing := mcp.inventory.missing_read_tools():
                run.notes.append(
                    "DataHub MCP server is missing expected read tools: "
                    + ", ".join(missing)
                )

            allow_writes = (
                mcp.inventory.has_write_access if write_back is None else bool(write_back)
            )
            if write_back and not mcp.inventory.has_write_access:
                allow_writes = False
                run.notes.append(
                    "Write-back requested but the MCP server exposes no mutation tools. "
                    "Set TOOLS_IS_MUTATION_ENABLED=true on the server to enable it."
                )

            graph = await build_ml_lineage(mcp, model_urn)
            run.graph = graph
            run.model_label = _label_for(graph, model_urn)

            dataset_urns = _training_dataset_urns(graph)
            baseline = self._store.load(model_urn)

            if baseline is None:
                if dataset_urns:
                    snapshot = await capture_snapshot(mcp, model_urn, dataset_urns)
                    self._store.save(snapshot)
                    run.baseline_created = True
                    run.baseline_captured_at = snapshot.captured_at
                    run.notes.append(
                        "No baseline existed — recorded one now. This run reviews lineage "
                        "coverage only; the next run can detect drift."
                    )
                else:
                    run.notes.append(
                        "No baseline and no upstream datasets to snapshot. Reviewing "
                        "lineage coverage only."
                    )
            else:
                current = await capture_snapshot(mcp, model_urn, dataset_urns)
                run.baseline_captured_at = baseline.captured_at
                run.drift = diff_snapshots(baseline, current)
                if not run.drift:
                    run.notes.append(
                        f"No schema drift vs baseline captured {baseline.captured_at}."
                    )

            run.verdict = await self._judge(mcp, run, allow_writes=allow_writes)

            if allow_writes and run.verdict is not None:
                run.write_backs = await self._record_verdict(mcp, run)

        return run

    # --- internals ---------------------------------------------------------

    async def _judge(
        self, mcp: DataHubMCP, run: SentinelRun, *, allow_writes: bool
    ) -> Verdict | None:
        """Prefer OpenAI and retry with Anthropic when that path cannot judge."""
        user_turn = _build_judgement_prompt(run, allow_writes=allow_writes)
        if self._config.agent.provider is AgentProvider.ANTHROPIC:
            providers = [AgentProvider.ANTHROPIC]
        elif self._config.agent.provider is AgentProvider.GEMINI:
            providers = [AgentProvider.GEMINI]
        else:
            providers = [AgentProvider.OPENAI, AgentProvider.ANTHROPIC]
        failures: list[str] = []
        completed_without_verdict = False
        total_usage = {"input_tokens": 0, "output_tokens": 0}

        for index, provider in enumerate(providers):
            try:
                if provider is AgentProvider.OPENAI:
                    verdict, usage = await self._judge_openai(mcp, run, user_turn)
                    model = self._config.agent.openai_model
                elif provider is AgentProvider.GEMINI:
                    verdict, usage = await self._judge_gemini(mcp, run, user_turn)
                    model = self._config.agent.gemini_model
                else:
                    verdict, usage = await self._judge_anthropic(mcp, run, user_turn)
                    model = self._config.agent.anthropic_model
            except AgentExecutionError as exc:
                failures.append(f"{provider.value}: {exc}")
                _merge_usage(total_usage, exc.usage)
                if index + 1 < len(providers):
                    run.notes.append(
                        f"{_provider_label(provider)} judgement failed; trying "
                        f"{_provider_label(providers[index + 1])} fallback."
                    )
                continue

            completed_without_verdict = verdict is None
            _merge_usage(total_usage, usage)
            if verdict is not None:
                run.token_usage = total_usage
                run.notes.append(
                    f"Judgement provider: {_provider_label(provider)} ({model})."
                )
                return verdict
            if index + 1 < len(providers):
                run.notes.append(
                    f"{_provider_label(provider)} finished without a verdict; trying "
                    f"{_provider_label(providers[index + 1])} fallback."
                )

        run.token_usage = total_usage
        if completed_without_verdict:
            run.notes.append(
                "The agent finished without emitting a verdict. Treating as UNKNOWN."
            )
            return None

        detail = "; ".join(failures) or "no provider was attempted"
        raise AgentExecutionError(f"no model provider could complete the judgement ({detail})")

    async def _judge_openai(
        self, mcp: DataHubMCP, run: SentinelRun, user_turn: str
    ) -> tuple[Verdict | None, dict[str, int]]:
        """Run a local function-tool loop through OpenAI's Responses API."""
        client = self._openai
        owns_client = client is None
        if client is None:
            try:
                client = AsyncOpenAI()
            except Exception as exc:  # noqa: BLE001 - normalize credential/config failures
                raise AgentExecutionError(f"OpenAI client unavailable: {exc}") from exc

        captured: dict[str, Verdict] = {}
        tools = [*mcp.openai_tools(include_writes=False), _OPENAI_VERDICT_TOOL]
        usage = {"input_tokens": 0, "output_tokens": 0}
        next_input: str | list[dict[str, Any]] = user_turn
        previous_response_id: str | None = None

        try:
            for _ in range(self._config.agent.max_iterations):
                request: dict[str, Any] = {
                    "model": self._config.agent.openai_model,
                    "instructions": SYSTEM_PROMPT,
                    "input": next_input,
                    "tools": tools,
                    "max_output_tokens": self._config.agent.max_tokens,
                    "reasoning": {"effort": self._config.agent.effort},
                }
                if previous_response_id is not None:
                    request["previous_response_id"] = previous_response_id

                try:
                    response = await client.responses.create(**request)
                except Exception as exc:  # noqa: BLE001 - provider errors enable fallback
                    raise AgentExecutionError(
                        f"OpenAI Responses API failed: {exc}", usage=usage
                    ) from exc

                _merge_usage(usage, _response_usage(response))
                function_calls = [
                    item
                    for item in (_field(response, "output", []) or [])
                    if _field(item, "type") == "function_call"
                ]
                if not function_calls:
                    return captured.get("verdict"), usage

                tool_outputs: list[dict[str, Any]] = []
                for call in function_calls:
                    call_id = str(_field(call, "call_id", ""))
                    name = str(_field(call, "name", ""))
                    arguments, parse_error = _parse_tool_arguments(
                        _field(call, "arguments", "{}")
                    )
                    if parse_error:
                        output = parse_error
                    elif name == "emit_verdict":
                        output = _capture_verdict(run, captured, arguments)
                    elif name in WRITE_TOOLS:
                        output = f"Tool call rejected: {name} is a mutation tool."
                    elif name not in mcp.inventory.names:
                        output = f"Tool call rejected: unknown DataHub tool {name!r}."
                    else:
                        try:
                            output = _render_tool_output(await mcp.call(name, arguments))
                        except Exception as exc:  # noqa: BLE001 - return read failure to model
                            output = f"DataHub tool {name!r} failed: {exc}"
                    tool_outputs.append(
                        {
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": output,
                        }
                    )

                # A valid typed verdict is the terminal condition. Catalog
                # mutations still happen later in the orchestrator.
                if "verdict" in captured:
                    return captured["verdict"], usage

                response_id = _field(response, "id")
                if not response_id:
                    raise AgentExecutionError(
                        "OpenAI response requested tools but returned no response id",
                        usage=usage,
                    )
                previous_response_id = str(response_id)
                next_input = tool_outputs

            raise AgentExecutionError(
                f"OpenAI exceeded {self._config.agent.max_iterations} tool iterations",
                usage=usage,
            )
        finally:
            if owns_client:
                await client.close()

    async def _judge_gemini(
        self, mcp: DataHubMCP, run: SentinelRun, user_turn: str
    ) -> tuple[Verdict | None, dict[str, int]]:
        """Run Gemini's free-tier tool loop through its OpenAI-compatible API."""
        client = self._gemini
        owns_client = client is None
        if client is None:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise AgentExecutionError(
                    "Gemini client unavailable: set GEMINI_API_KEY from Google AI Studio"
                )
            client = AsyncOpenAI(api_key=api_key, base_url=GEMINI_OPENAI_BASE_URL)

        captured: dict[str, Verdict] = {}
        tools = _chat_completion_tools(
            [*mcp.openai_tools(include_writes=False), _OPENAI_VERDICT_TOOL]
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_turn},
        ]
        usage = {"input_tokens": 0, "output_tokens": 0}

        try:
            for _ in range(self._config.agent.max_iterations):
                try:
                    response = await client.chat.completions.create(
                        model=self._config.agent.gemini_model,
                        messages=messages,
                        tools=tools,
                        tool_choice="auto",
                        reasoning_effort=self._config.agent.effort,
                        max_tokens=self._config.agent.max_tokens,
                    )
                except Exception as exc:  # noqa: BLE001 - normalize provider failures
                    raise AgentExecutionError(
                        f"Gemini OpenAI-compatible API failed: {exc}", usage=usage
                    ) from exc

                _merge_usage(usage, _chat_completion_usage(response))
                choices = _field(response, "choices", []) or []
                if not choices:
                    raise AgentExecutionError(
                        "Gemini returned no chat-completion choices", usage=usage
                    )
                message = _field(choices[0], "message")
                if message is None:
                    raise AgentExecutionError(
                        "Gemini returned a choice without a message", usage=usage
                    )
                messages.append(_chat_message_payload(message))
                tool_calls = _field(message, "tool_calls", []) or []
                if not tool_calls:
                    return captured.get("verdict"), usage

                for call in tool_calls:
                    call_id = str(_field(call, "id", ""))
                    function = _field(call, "function", {}) or {}
                    name = str(_field(function, "name", ""))
                    arguments, parse_error = _parse_tool_arguments(
                        _field(function, "arguments", "{}")
                    )
                    if parse_error:
                        output = parse_error
                    elif name == "emit_verdict":
                        output = _capture_verdict(run, captured, arguments)
                    elif name in WRITE_TOOLS:
                        output = f"Tool call rejected: {name} is a mutation tool."
                    elif name not in mcp.inventory.names:
                        output = f"Tool call rejected: unknown DataHub tool {name!r}."
                    else:
                        try:
                            output = _render_tool_output(await mcp.call(name, arguments))
                        except Exception as exc:  # noqa: BLE001 - return read failure to model
                            output = f"DataHub tool {name!r} failed: {exc}"
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": output,
                        }
                    )

                if "verdict" in captured:
                    return captured["verdict"], usage

            raise AgentExecutionError(
                f"Gemini exceeded {self._config.agent.max_iterations} tool iterations",
                usage=usage,
            )
        finally:
            if owns_client:
                await client.close()

    async def _judge_anthropic(
        self, mcp: DataHubMCP, run: SentinelRun, user_turn: str
    ) -> tuple[Verdict | None, dict[str, int]]:
        """Run the original Anthropic tool runner as the fallback provider."""
        client = self._anthropic
        owns_client = client is None
        if client is None:
            try:
                client = AsyncAnthropic()
            except Exception as exc:  # noqa: BLE001 - normalize credential/config failures
                raise AgentExecutionError(f"Anthropic client unavailable: {exc}") from exc

        captured: dict[str, Verdict] = {}

        @beta_async_tool
        async def emit_verdict(
            decision: str,
            headline: str,
            reasoning: str,
            risks: list[dict] | None = None,
            recommended_actions: list[str] | None = None,
            downstream_at_risk: list[str] | None = None,
            confidence: str | None = None,
            unverified_claims: list[str] | None = None,
        ) -> str:
            """Record the final verdict for this model. Call exactly once.

            Args:
                decision: One of BLOCK, WARN, OK, UNKNOWN.
                headline: One sentence an on-call engineer can act on.
                reasoning: Why this decision follows from the lineage and change.
                risks: Each with title, severity (low|medium|high|critical),
                    mechanism (the lineage path), optional affected_urn, evidence.
                recommended_actions: Concrete next steps, most important first.
                downstream_at_risk: URNs of affected downstream consumers.
                confidence: low|medium|high|critical.
                unverified_claims: Anything you could not confirm from DataHub.
            """
            return _capture_verdict(
                run,
                captured,
                {
                    "decision": decision,
                    "headline": headline,
                    "reasoning": reasoning,
                    "risks": risks,
                    "recommended_actions": recommended_actions,
                    "downstream_at_risk": downstream_at_risk,
                    "confidence": confidence,
                    "unverified_claims": unverified_claims,
                },
            )

        usage = {"input_tokens": 0, "output_tokens": 0}
        try:
            try:
                tools = [*mcp.anthropic_tools(include_writes=False), emit_verdict]
                runner = client.beta.messages.tool_runner(
                    model=self._config.agent.anthropic_model,
                    max_tokens=self._config.agent.max_tokens,
                    thinking={"type": "adaptive"},
                    output_config={"effort": self._config.agent.effort},
                    system=SYSTEM_PROMPT,
                    tools=tools,
                    messages=[{"role": "user", "content": user_turn}],
                    max_iterations=self._config.agent.max_iterations,
                )
                async for message in runner:
                    message_usage = getattr(message, "usage", None)
                    if message_usage is not None:
                        usage["input_tokens"] += (
                            getattr(message_usage, "input_tokens", 0) or 0
                        )
                        usage["output_tokens"] += (
                            getattr(message_usage, "output_tokens", 0) or 0
                        )
            except Exception as exc:  # noqa: BLE001 - provider errors enable fallback
                raise AgentExecutionError(
                    f"Anthropic Messages API failed: {exc}", usage=usage
                ) from exc
            return captured.get("verdict"), usage
        finally:
            if owns_client:
                await client.close()

    async def _record_verdict(self, mcp: DataHubMCP, run: SentinelRun) -> list[WriteBack]:
        """Persist the verdict into DataHub so it is visible in the catalog."""
        verdict = run.verdict
        if verdict is None:
            return []

        writes: list[WriteBack] = []
        tag = {
            Decision.BLOCK: BLOCK_TAG,
            Decision.WARN: WARN_TAG,
            Decision.UNKNOWN: WARN_TAG,
        }.get(verdict.decision, SENTINEL_TAG)

        stale_tags = [candidate for candidate in STATUS_TAGS if candidate != tag]
        if stale_tags and "remove_tags" in mcp.inventory.names:
            writes.append(
                await self._try_write(
                    mcp,
                    "remove_tags",
                    {
                        "tag_urns": stale_tags,
                        "entity_urns": [run.model_urn],
                    },
                    target=run.model_urn,
                    detail="cleared obsolete sentinel status tags",
                )
            )

        writes.append(
            await self._try_write(
                mcp,
                "add_tags",
                {"tag_urns": [tag], "entity_urns": [run.model_urn]},
                target=run.model_urn,
                detail=f"tagged {tag}",
            )
        )

        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        status = f"[Sentinel {verdict.decision.value} @ {stamp}] {verdict.headline}"
        writes.append(
            await self._try_write(
                mcp,
                "update_description",
                {
                    "entity_urn": run.model_urn,
                    "operation": "append",
                    "description": f"\n\n{status}",
                },
                target=run.model_urn,
                detail="appended verdict status to the description",
            )
        )

        if "save_document" in mcp.inventory.names:
            writes.append(
                await self._try_write(
                    mcp,
                    "save_document",
                    {
                        "document_type": "Analysis",
                        "title": f"Sentinel report: {run.model_label}",
                        "content": _render_markdown_report(run),
                        "topics": ["forecast-sentinel", verdict.decision.value.lower()],
                        "related_assets": [run.model_urn],
                    },
                    target=run.model_urn,
                    detail="saved full findings document linked to the model",
                )
            )

        return [w for w in writes if w is not None]

    async def _try_write(
        self,
        mcp: DataHubMCP,
        tool: str,
        arguments: dict[str, Any],
        *,
        target: str,
        detail: str,
    ) -> WriteBack:
        try:
            await mcp.call(tool, arguments)
        except MCPConnectionError as exc:
            return WriteBack(
                tool=tool, target_urn=target, detail=detail, succeeded=False, error=str(exc)
            )
        return WriteBack(tool=tool, target_urn=target, detail=detail)


# --- provider helpers --------------------------------------------------------


def _build_judgement_prompt(run: SentinelRun, *, allow_writes: bool) -> str:
    lineage_summary = _render_lineage(run.graph)
    if run.drift:
        return build_check_prompt(
            model_urn=run.model_urn,
            model_label=run.model_label,
            lineage_summary=lineage_summary,
            drift_summary=_render_drift(run.drift),
            write_enabled=allow_writes,
        )
    return build_no_drift_prompt(
        model_urn=run.model_urn,
        model_label=run.model_label,
        lineage_summary=lineage_summary,
        baseline_created=run.baseline_created,
        comparison_performed=run.baseline_captured_at is not None,
    )


def _capture_verdict(
    run: SentinelRun,
    captured: dict[str, Verdict],
    arguments: dict[str, Any],
) -> str:
    """Validate the shared tool payload before any provider can end the run."""
    try:
        decision = str(arguments.get("decision", "")).strip().upper()
        verdict = Verdict(
            model_urn=run.model_urn,
            decision=Decision(decision),
            headline=arguments.get("headline", ""),
            reasoning=arguments.get("reasoning", ""),
            risks=arguments.get("risks") or [],
            recommended_actions=arguments.get("recommended_actions") or [],
            downstream_at_risk=arguments.get("downstream_at_risk") or [],
            confidence=arguments.get("confidence"),
            unverified_claims=arguments.get("unverified_claims") or [],
        )
    except Exception as exc:  # noqa: BLE001 - return validation detail to the model
        return f"Verdict rejected: {exc}. Fix the fields and call emit_verdict again."
    captured["verdict"] = verdict
    return f"Verdict recorded: {verdict.decision.value}."


def _parse_tool_arguments(raw: Any) -> tuple[dict[str, Any], str | None]:
    if isinstance(raw, dict):
        return raw, None
    try:
        parsed = json.loads(raw or "{}")
    except (TypeError, json.JSONDecodeError) as exc:
        return {}, f"Tool arguments rejected: invalid JSON ({exc})."
    if not isinstance(parsed, dict):
        return {}, "Tool arguments rejected: expected a JSON object."
    return parsed, None


def _render_tool_output(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    return json.dumps(payload, default=str)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _response_usage(response: Any) -> dict[str, int]:
    usage = _field(response, "usage")
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0}
    return {
        "input_tokens": int(_field(usage, "input_tokens", 0) or 0),
        "output_tokens": int(_field(usage, "output_tokens", 0) or 0),
    }


def _chat_completion_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Responses API function schemas to Chat Completions schemas."""
    converted = []
    for tool in tools:
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters")
                    or {"type": "object", "properties": {}},
                },
            }
        )
    return converted


def _chat_message_payload(message: Any) -> dict[str, Any]:
    """Preserve provider-specific tool-call metadata when continuing a chat."""
    if isinstance(message, dict):
        return dict(message)
    if hasattr(message, "model_dump"):
        return message.model_dump(mode="json", exclude_none=True)

    payload: dict[str, Any] = {
        "role": _field(message, "role", "assistant"),
        "content": _field(message, "content"),
    }
    tool_calls = _field(message, "tool_calls", []) or []
    if tool_calls:
        payload["tool_calls"] = [
            {
                "id": _field(call, "id", ""),
                "type": _field(call, "type", "function"),
                "function": {
                    "name": _field(_field(call, "function", {}), "name", ""),
                    "arguments": _field(
                        _field(call, "function", {}), "arguments", "{}"
                    ),
                },
            }
            for call in tool_calls
        ]
    return payload


def _chat_completion_usage(response: Any) -> dict[str, int]:
    usage = _field(response, "usage")
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0}
    return {
        "input_tokens": int(_field(usage, "prompt_tokens", 0) or 0),
        "output_tokens": int(_field(usage, "completion_tokens", 0) or 0),
    }


def _merge_usage(total: dict[str, int], addition: dict[str, int]) -> None:
    total["input_tokens"] += addition.get("input_tokens", 0)
    total["output_tokens"] += addition.get("output_tokens", 0)


def _provider_label(provider: AgentProvider) -> str:
    return {
        AgentProvider.OPENAI: "OpenAI",
        AgentProvider.GEMINI: "Gemini",
        AgentProvider.ANTHROPIC: "Anthropic",
    }.get(provider, provider.value)


# --- rendering helpers -------------------------------------------------------


def _new_run(model_urn: str) -> SentinelRun:
    return SentinelRun(
        model_urn=model_urn,
        model_label=parse_urn(model_urn).short(),
        started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def _label_for(graph: MLLineageGraph, model_urn: str) -> str:
    for asset in graph.all_assets.values():
        if asset.urn.raw == model_urn and asset.name:
            return asset.name
    return parse_urn(model_urn).short()


def _training_dataset_urns(graph: MLLineageGraph) -> list[str]:
    """Upstream datasets are what we can snapshot a schema for."""
    return [a.urn.raw for a in graph.upstream.values() if a.entity_type == "dataset"]


def _render_lineage(graph: MLLineageGraph | None) -> str:
    if graph is None or not graph.all_assets:
        return "_No lineage returned by DataHub for this model._"

    lines: list[str] = []
    summary = graph.summary()
    lines.append(
        f"{summary['upstream_count']} upstream / {summary['downstream_count']} downstream "
        f"assets, search bounded at {summary['max_hops']} hops"
        + (" (truncated)" if summary["truncated"] else "")
    )
    if errors := summary["errors"]:
        lines.append("Lineage retrieval warnings: " + "; ".join(errors))
    if owners := summary["owners"]:
        lines.append(f"Owners: {', '.join(owners[:8])}")

    def block(title: str, assets: list[Asset]) -> None:
        if not assets:
            return
        lines.append(f"\n{title}:")
        for asset in sorted(assets, key=lambda a: a.urn.raw)[:25]:
            lines.append(f"  - [{asset.entity_type}] {asset.label} — `{asset.urn.raw}`")
        if len(assets) > 25:
            lines.append(f"  … and {len(assets) - 25} more")

    block("Upstream (training inputs and features)", list(graph.upstream.values()))
    block("Downstream (consumers of this model's output)", list(graph.downstream.values()))
    return "\n".join(lines)


def _render_drift(events: list[DriftEvent]) -> str:
    if not events:
        return "_None._"
    lines = []
    for event in events:
        lines.append(
            f"- **{event.severity.upper()}** {event.describe()}  \n"
            f"  in `{event.dataset_urn}`"
        )
    return "\n".join(lines)


def _render_markdown_report(run: SentinelRun) -> str:
    """The document written back to DataHub via `save_document`."""
    verdict = run.verdict
    parts = [
        f"# Forecast Model Sentinel — {run.model_label}",
        "",
        f"- **Decision:** {run.decision.value}",
        f"- **Model:** `{run.model_urn}`",
        f"- **Checked:** {run.started_at}",
    ]
    if run.baseline_captured_at:
        parts.append(f"- **Baseline:** {run.baseline_captured_at}")
    if verdict:
        parts += ["", f"**{verdict.headline}**", "", verdict.reasoning]
        if verdict.risks:
            parts += ["", "## Risks"]
            for risk in verdict.risks:
                parts.append(f"### {risk.severity.value.upper()} — {risk.title}")
                parts.append(f"Path: {risk.mechanism}")
                if risk.affected_urn:
                    parts.append(f"Affected: `{risk.affected_urn}`")
                for item in risk.evidence:
                    parts.append(f"- {item}")
                parts.append("")
        if verdict.recommended_actions:
            parts += ["", "## Recommended actions"]
            parts += [f"{i}. {a}" for i, a in enumerate(verdict.recommended_actions, 1)]
        if verdict.unverified_claims:
            parts += ["", "## Could not verify"]
            parts += [f"- {c}" for c in verdict.unverified_claims]

    if run.drift:
        parts += ["", "## Detected upstream changes", _render_drift(run.drift)]
    if run.notes:
        parts += ["", "## Notes"] + [f"- {n}" for n in run.notes]

    parts += ["", "---", "_Generated by Forecast Model Sentinel._"]
    return "\n".join(parts)


def dump_run(run: SentinelRun) -> str:
    return json.dumps(run.to_dict(), indent=2, default=str)
