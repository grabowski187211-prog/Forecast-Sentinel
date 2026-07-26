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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from anthropic import AsyncAnthropic, beta_async_tool

from forecast_sentinel.agent.prompts import (
    SYSTEM_PROMPT,
    build_check_prompt,
    build_no_drift_prompt,
)
from forecast_sentinel.agent.schemas import Decision, Verdict, WriteBack
from forecast_sentinel.config import SentinelConfig
from forecast_sentinel.datahub.mcp_client import DataHubMCP, MCPConnectionError
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

    def __init__(self, config: SentinelConfig, *, anthropic: AsyncAnthropic | None = None) -> None:
        self._config = config
        self._client = anthropic or AsyncAnthropic()
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
        """Run the agent loop over DataHub's tools and capture its verdict."""
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
            try:
                verdict = Verdict(
                    model_urn=run.model_urn,
                    decision=Decision(decision.strip().upper()),
                    headline=headline,
                    reasoning=reasoning,
                    risks=risks or [],  # type: ignore[arg-type]
                    recommended_actions=recommended_actions or [],
                    downstream_at_risk=downstream_at_risk or [],
                    confidence=confidence,  # type: ignore[arg-type]
                    unverified_claims=unverified_claims or [],
                )
            except Exception as exc:  # noqa: BLE001 - feed the error back to the model
                return (
                    f"Verdict rejected: {exc}. Fix the fields and call emit_verdict again."
                )
            captured["verdict"] = verdict
            return f"Verdict recorded: {verdict.decision.value}."

        lineage_summary = _render_lineage(run.graph)
        if run.drift:
            user_turn = build_check_prompt(
                model_urn=run.model_urn,
                model_label=run.model_label,
                lineage_summary=lineage_summary,
                drift_summary=_render_drift(run.drift),
                write_enabled=allow_writes,
            )
        else:
            user_turn = build_no_drift_prompt(
                model_urn=run.model_urn,
                model_label=run.model_label,
                lineage_summary=lineage_summary,
            )

        tools = [*mcp.anthropic_tools(include_writes=False), emit_verdict]

        runner = self._client.beta.messages.tool_runner(
            model=self._config.agent.model,
            max_tokens=self._config.agent.max_tokens,
            thinking={"type": "adaptive"},
            output_config={"effort": self._config.agent.effort},
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=[{"role": "user", "content": user_turn}],
            max_iterations=self._config.agent.max_iterations,
        )

        usage = {"input_tokens": 0, "output_tokens": 0}
        async for message in runner:
            message_usage = getattr(message, "usage", None)
            if message_usage is not None:
                usage["input_tokens"] += getattr(message_usage, "input_tokens", 0) or 0
                usage["output_tokens"] += getattr(message_usage, "output_tokens", 0) or 0
        run.token_usage = usage

        verdict = captured.get("verdict")
        if verdict is None:
            run.notes.append(
                "The agent finished without emitting a verdict. Treating as UNKNOWN."
            )
        return verdict

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

        writes.append(
            await self._try_write(
                mcp,
                "add_tags",
                {"urn": run.model_urn, "tags": [tag]},
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
                {"urn": run.model_urn, "description": status},
                target=run.model_urn,
                detail="updated description with verdict status",
            )
        )

        if "save_document" in mcp.inventory.names:
            writes.append(
                await self._try_write(
                    mcp,
                    "save_document",
                    {
                        "title": f"Sentinel report: {run.model_label}",
                        "content": _render_markdown_report(run),
                    },
                    target=run.model_urn,
                    detail="saved full findings document",
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
        f"assets, depth {summary['depth_reached']}"
        + (" (truncated)" if summary["truncated"] else "")
    )
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
