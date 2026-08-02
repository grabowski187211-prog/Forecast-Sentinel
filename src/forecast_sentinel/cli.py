"""`sentinel` — the command-line surface.

    sentinel doctor                  check config + DataHub MCP connectivity
    sentinel models                  list mlModel entities in the catalog
    sentinel baseline <model-urn>    record the current training-input schemas
    sentinel check <model-urn>       detect drift, judge it, write back
    sentinel watch                   check every model that has a baseline

`check` exits 2 on a BLOCK verdict, so it drops straight into CI as a gate in
front of a scheduled retrain or a batch scoring job.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import typer
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from forecast_sentinel.agent.schemas import Decision
from forecast_sentinel.agent.sentinel import AgentExecutionError, Sentinel, SentinelRun
from forecast_sentinel.config import AgentProvider, ConfigError, SentinelConfig
from forecast_sentinel.datahub.mcp_client import DataHubMCP, MCPConnectionError
from forecast_sentinel.datahub.ml_lineage import LineageError
from forecast_sentinel.datahub.urns import UrnParseError, parse_urn
from forecast_sentinel.report.html import write_html_report
from forecast_sentinel.snapshots import SnapshotCaptureError, SnapshotStore

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Guard production ML models using DataHub's end-to-end ML lineage.",
)
console = Console()

EXIT_BLOCK = 2
EXIT_ERROR = 1

_DECISION_STYLE = {
    Decision.OK: "green",
    Decision.WARN: "yellow",
    Decision.BLOCK: "bold red",
    Decision.UNKNOWN: "dim",
}


def _load_config(env_file: Path | None) -> SentinelConfig:
    try:
        return SentinelConfig.from_env(env_file)
    except ConfigError as exc:
        console.print(f"[bold red]Configuration error:[/] {exc}")
        console.print("Copy [cyan].env.example[/] to [cyan].env[/] and fill it in.")
        raise typer.Exit(EXIT_ERROR) from exc


def _validate_urn(value: str, *, expect: str | None = None) -> str:
    try:
        urn = parse_urn(value)
    except UrnParseError as exc:
        console.print(f"[bold red]Invalid URN:[/] {exc}")
        raise typer.Exit(EXIT_ERROR) from exc
    if expect and urn.entity_type != expect:
        console.print(
            f"[yellow]Warning:[/] expected a {expect} URN but got {urn.entity_type}."
        )
    return urn.raw


@app.command()
def doctor(
    env_file: Path | None = typer.Option(None, "--env-file", help="Path to a .env file."),
) -> None:
    """Verify configuration and connectivity to the DataHub MCP server."""
    config = _load_config(env_file)
    dh = config.datahub

    table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
    table.add_row("mode", dh.mode.value)
    if dh.mode.value == "selfhosted":
        table.add_row("GMS URL", dh.gms_url or "[red]unset[/]")
        table.add_row("token", "set" if dh.gms_token else "[dim]empty (ok for quickstart)[/]")
    else:
        table.add_row("tenant", dh.tenant_url or "[red]unset[/]")
        table.add_row("MCP endpoint", dh.cloud_mcp_endpoint if dh.tenant_url else "[red]n/a[/]")
        table.add_row("token", "set" if dh.token else "[red]unset[/]")
    table.add_row("writes requested", "yes" if dh.mutation_enabled else "no")
    table.add_row("provider", config.agent.provider.value)
    table.add_row("OpenAI model", config.agent.openai_model)
    table.add_row("Gemini model", config.agent.gemini_model)
    table.add_row("Anthropic fallback", config.agent.anthropic_model)
    table.add_row("effort", config.agent.effort)
    table.add_row("state dir", str(config.state_dir))
    console.print(Panel(table, title="Configuration", border_style="cyan"))

    async def _probe() -> None:
        async with DataHubMCP(dh, server_log=config.mcp_server_log) as mcp:
            inv = mcp.inventory
            console.print(f"[green]Connected.[/] {len(inv.names)} MCP tools exposed.")
            if missing := inv.missing_read_tools():
                console.print(f"[yellow]Missing read tools:[/] {', '.join(missing)}")
            else:
                console.print("[green]All expected read tools present.[/]")
            if inv.has_write_access:
                console.print("[green]Write tools available[/] — verdicts can be written back.")
            else:
                console.print(
                    "[yellow]No write tools.[/] Set TOOLS_IS_MUTATION_ENABLED=true "
                    "to let the sentinel record findings in the catalog."
                )

            openai_ready = False
            try:
                openai_client = AsyncOpenAI()
            except Exception:  # noqa: BLE001 - missing key is the expected failure
                pass
            else:
                openai_ready = True
                await openai_client.close()
            if openai_ready:
                role = (
                    "available (not selected)"
                    if config.agent.provider
                    in {AgentProvider.ANTHROPIC, AgentProvider.GEMINI}
                    else "available — primary provider"
                )
                console.print(f"[green]OpenAI credentials {role}.[/]")
            elif config.agent.provider in {
                AgentProvider.ANTHROPIC,
                AgentProvider.GEMINI,
            }:
                console.print(
                    f"[dim]OpenAI credentials not required in "
                    f"{config.agent.provider.value} mode.[/]"
                )
            else:
                console.print(
                    "[yellow]No OpenAI credentials.[/] Set OPENAI_API_KEY; "
                    "Anthropic will be tried as the fallback."
                )

            gemini_ready = bool(os.getenv("GEMINI_API_KEY"))
            if gemini_ready:
                role = (
                    "selected provider"
                    if config.agent.provider is AgentProvider.GEMINI
                    else "available (not selected)"
                )
                console.print(f"[green]Gemini credentials available — {role}.[/]")
            elif config.agent.provider is AgentProvider.GEMINI:
                console.print(
                    "[yellow]No Gemini credentials.[/] Create a free key in Google "
                    "AI Studio and set GEMINI_API_KEY."
                )
            else:
                console.print("[dim]Gemini credentials not configured.[/]")

            anthropic_client = AsyncAnthropic()
            try:
                anthropic_ready = bool(
                    anthropic_client.api_key
                    or anthropic_client.auth_token
                    or anthropic_client.credentials
                )
            finally:
                await anthropic_client.close()
            if anthropic_ready:
                role = (
                    "selected provider"
                    if config.agent.provider is AgentProvider.ANTHROPIC
                    else "fallback available"
                )
                console.print(f"[green]Anthropic credentials available — {role}.[/]")
            elif (
                openai_ready or gemini_ready
            ) and config.agent.provider is not AgentProvider.ANTHROPIC:
                console.print("[dim]Anthropic fallback credentials not configured.[/]")
            else:
                console.print(
                    "[yellow]No Anthropic credentials.[/] Set ANTHROPIC_API_KEY or "
                    "ANTHROPIC_AUTH_TOKEN to enable the fallback."
                )

    try:
        asyncio.run(_probe())
    except MCPConnectionError as exc:
        console.print(f"[bold red]Connection failed:[/] {exc}")
        raise typer.Exit(EXIT_ERROR) from exc


@app.command()
def models(
    limit: int = typer.Option(25, "--limit", "-n", help="Maximum models to list."),
    env_file: Path | None = typer.Option(None, "--env-file"),
) -> None:
    """List mlModel entities in the catalog, so you have URNs to work with."""
    config = _load_config(env_file)

    async def _search() -> object:
        async with DataHubHelper(config) as mcp:
            # The DataHub MCP `search` tool takes only query/filter/num_results/
            # sort_by/sort_order/offset — there is no `entity_types` parameter,
            # and `entity_type:mlModel` in the query string does not filter
            # either (verified against DataHub 1.5.0.6). So over-fetch and filter
            # by URN prefix here, which is exact.
            return await mcp.call("search", {"query": "*", "num_results": max(limit * 10, 50)})

    try:
        payload = asyncio.run(_search())
    except MCPConnectionError as exc:
        console.print(f"[bold red]{exc}[/]")
        raise typer.Exit(EXIT_ERROR) from exc

    urns = [u for u in _collect_urns(payload) if u.startswith("urn:li:mlModel:")][:limit]
    if not urns:
        console.print(
            "[yellow]No mlModel entities found.[/] Load demo ML metadata first:\n"
            "  [cyan]python scripts/seed_ml_demo.py[/]"
        )
        return
    table = Table(title=f"mlModel entities ({len(urns)})")
    table.add_column("#", justify="right", style="dim")
    table.add_column("URN")
    for i, urn in enumerate(urns, 1):
        table.add_row(str(i), urn)
    console.print(table)


@app.command()
def baseline(
    model_urn: str = typer.Argument(..., help="mlModel URN to baseline."),
    env_file: Path | None = typer.Option(None, "--env-file"),
) -> None:
    """Record the current schema of a model's training inputs."""
    config = _load_config(env_file)
    urn = _validate_urn(model_urn, expect="mlModel")
    sentinel = Sentinel(config)
    try:
        run = asyncio.run(sentinel.capture_baseline(urn))
    except (MCPConnectionError, LineageError, SnapshotCaptureError) as exc:
        console.print(f"[bold red]{exc}[/]")
        raise typer.Exit(EXIT_ERROR) from exc
    for note in run.notes:
        console.print(f"[dim]•[/] {note}")
    if run.baseline_created:
        console.print(f"[green]Baseline recorded[/] at {run.baseline_captured_at}")
    else:
        console.print("[yellow]No baseline recorded.[/]")
        raise typer.Exit(EXIT_ERROR)


@app.command()
def check(
    model_urn: str = typer.Argument(..., help="mlModel URN to assess."),
    write_back: bool = typer.Option(
        True, "--write-back/--no-write-back", help="Record the verdict in DataHub."
    ),
    html: bool = typer.Option(True, "--html/--no-html", help="Write an HTML report."),
    json_out: Path | None = typer.Option(None, "--json", help="Also dump the run as JSON."),
    env_file: Path | None = typer.Option(None, "--env-file"),
) -> None:
    """Detect upstream drift, judge whether it invalidates the model, write back."""
    config = _load_config(env_file)
    urn = _validate_urn(model_urn, expect="mlModel")
    sentinel = Sentinel(config)

    try:
        run = asyncio.run(sentinel.check(urn, write_back=write_back))
    except (MCPConnectionError, LineageError, SnapshotCaptureError, AgentExecutionError) as exc:
        console.print(f"[bold red]{exc}[/]")
        raise typer.Exit(EXIT_ERROR) from exc

    _render_run(run)

    if html:
        path = write_html_report(run, config.run_dir)
        console.print(f"\n[cyan]HTML report:[/] {path}")
    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(run.to_dict(), indent=2, default=str), encoding="utf-8")
        console.print(f"[cyan]JSON:[/] {json_out}")

    if run.should_fail_build and config.fail_on_block:
        console.print("\n[bold red]BLOCK verdict — failing with exit code 2.[/]")
        raise typer.Exit(EXIT_BLOCK)


@app.command()
def watch(
    write_back: bool = typer.Option(True, "--write-back/--no-write-back"),
    env_file: Path | None = typer.Option(None, "--env-file"),
) -> None:
    """Check every model that already has a recorded baseline."""
    config = _load_config(env_file)
    store = SnapshotStore(config.baseline_dir)
    urns = store.list_models()
    if not urns:
        console.print(
            "[yellow]No baselines recorded yet.[/] Run "
            "[cyan]sentinel baseline <model-urn>[/] first."
        )
        raise typer.Exit(EXIT_ERROR)

    console.print(f"Checking {len(urns)} model(s) with baselines.\n")
    sentinel = Sentinel(config)
    blocked: list[str] = []

    for urn in urns:
        console.rule(urn)
        try:
            run = asyncio.run(sentinel.check(urn, write_back=write_back))
        except (
            MCPConnectionError,
            LineageError,
            SnapshotCaptureError,
            AgentExecutionError,
        ) as exc:
            console.print(f"[bold red]{exc}[/]")
            raise typer.Exit(EXIT_ERROR) from exc
        _render_run(run)
        write_html_report(run, config.run_dir)
        if run.should_fail_build:
            blocked.append(urn)

    console.rule("Summary")
    if blocked:
        console.print(f"[bold red]{len(blocked)} model(s) invalidated:[/]")
        for urn in blocked:
            console.print(f"  • {urn}")
        if config.fail_on_block:
            raise typer.Exit(EXIT_BLOCK)
    else:
        console.print("[green]No models invalidated.[/]")


# --- rendering ---------------------------------------------------------------


def _render_run(run: SentinelRun) -> None:
    style = _DECISION_STYLE[run.decision]
    verdict = run.verdict

    headline = verdict.headline if verdict else "No verdict produced."
    console.print(
        Panel(
            headline,
            title=f"[{style}]{run.decision.value}[/] — {run.model_label}",
            border_style=style,
        )
    )

    if run.graph:
        s = run.graph.summary()
        console.print(
            f"[dim]lineage:[/] {s['upstream_count']} upstream, "
            f"{s['downstream_count']} downstream, max hops {s['max_hops']}"
            + (" [yellow](truncated)[/]" if s["truncated"] else "")
        )
        for error in s["errors"]:
            console.print(f"[yellow]lineage warning:[/] {error}")

    if run.drift:
        table = Table(title=f"Upstream changes ({len(run.drift)})", title_justify="left")
        table.add_column("sev")
        table.add_column("change")
        table.add_column("dataset", overflow="fold")
        for event in run.drift:
            colour = {"high": "red", "medium": "yellow", "low": "dim"}[event.severity]
            table.add_row(f"[{colour}]{event.severity}[/]", event.describe(), event.dataset_urn)
        console.print(table)

    if verdict and verdict.risks:
        console.print("\n[bold]Risks[/]")
        for risk in verdict.risks:
            colour = {
                "critical": "bold red",
                "high": "red",
                "medium": "yellow",
                "low": "dim",
            }[risk.severity.value]
            console.print(f"  [{colour}]{risk.severity.value.upper()}[/] {risk.title}")
            console.print(f"    [dim]path:[/] {risk.mechanism}")

    if verdict and verdict.recommended_actions:
        console.print("\n[bold]Recommended actions[/]")
        for i, action in enumerate(verdict.recommended_actions, 1):
            console.print(f"  {i}. {action}")

    if verdict and verdict.unverified_claims:
        console.print("\n[yellow]Could not verify[/]")
        for claim in verdict.unverified_claims:
            console.print(f"  • {claim}")

    if run.write_backs:
        console.print("\n[bold]Written back to DataHub[/]")
        for wb in run.write_backs:
            mark = "[green]✓[/]" if wb.succeeded else "[red]✗[/]"
            console.print(f"  {mark} {wb.tool} — {wb.detail}")
            if wb.error:
                console.print(f"      [dim]{wb.error}[/]")

    for note in run.notes:
        console.print(f"[dim]note:[/] {note}")


def _collect_urns(payload: object) -> list[str]:
    """Pull URNs out of a search payload without assuming its exact shape."""
    found: list[str] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in {"urn", "entityUrn", "entity_urn"} and isinstance(value, str):
                    if value not in found:
                        found.append(value)
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(payload)
    return found


class DataHubHelper:
    """Thin async-context wrapper so `models` can reuse DataHubMCP directly."""

    def __init__(self, config: SentinelConfig) -> None:
        self._mcp = DataHubMCP(
            config.datahub, server_log=config.mcp_server_log
        )

    async def __aenter__(self) -> DataHubMCP:
        return await self._mcp.__aenter__()

    async def __aexit__(self, *exc_info: object) -> None:
        await self._mcp.__aexit__(*exc_info)  # type: ignore[arg-type]


if __name__ == "__main__":
    app()
