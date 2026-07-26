"""Transport to the DataHub MCP server.

One class handles both deployment shapes so nothing else in the codebase needs
to know which one is live:

* self-hosted -> `uvx mcp-server-datahub@latest` as a stdio subprocess
* cloud       -> the tenant's streamable-HTTP MCP endpoint, bearer-authenticated

The connected `ClientSession` is handed to the Anthropic tool runner via
`anthropic.lib.tools.mcp.async_mcp_tool`, so the model calls DataHub's real
tools (`search`, `get_lineage`, `add_tags`, ...) rather than a reimplementation.

API note: this targets the `mcp` 1.x client API (`ClientSession` +
`stdio_client` / `streamablehttp_client`), which is what `anthropic[mcp]`'s
helper expects. The 2.x SDK introduces a higher-level `Client` facade; the
dependency is pinned to `<2` so that change cannot silently break the session
setup below.
"""

from __future__ import annotations

import json
import os
import shutil
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

from forecast_sentinel.config import DataHubConfig, Mode

# Tools the DataHub MCP server exposes. Read tools are always present; mutation
# tools require TOOLS_IS_MUTATION_ENABLED=true on the server.
#
# Verified against a live self-hosted server (DataHub 1.5.0.6), which exposed 18
# tools. `search_documents` / `grep_documents` are absent there because the
# server filters document tools out when the catalog holds no documents — so a
# missing read tool is not necessarily a misconfiguration.
READ_TOOLS = (
    "search",
    "get_entities",
    "get_lineage",
    "get_lineage_paths_between",
    "list_schema_fields",
    "get_dataset_queries",
    "search_documents",
    "grep_documents",
)

# Every tool that mutates the catalog. This list is a safety boundary, not
# documentation: `anthropic_tools(include_writes=False)` filters on it to keep
# mutation tools out of the agent's hands. A tool missing here is silently
# handed to the model, so the remove_* counterparts matter as much as the add_*
# ones — they are what could strip ownership or domains off an entity.
WRITE_TOOLS = (
    "add_tags",
    "remove_tags",
    "add_terms",
    "remove_terms",
    "add_owners",
    "remove_owners",
    "set_domains",
    "remove_domains",
    "update_description",
    "add_structured_properties",
    "remove_structured_properties",
    "save_document",
)


class MCPConnectionError(RuntimeError):
    """Raised when the DataHub MCP server cannot be reached or is unusable."""


@dataclass
class ToolInventory:
    """What the connected MCP server actually offers."""

    names: tuple[str, ...] = ()
    raw: list[Any] = field(default_factory=list)

    @property
    def has_write_access(self) -> bool:
        return any(name in self.names for name in WRITE_TOOLS)

    def missing_read_tools(self) -> tuple[str, ...]:
        return tuple(name for name in READ_TOOLS if name not in self.names)


class DataHubMCP:
    """Async context manager owning the MCP session for one sentinel run."""

    def __init__(self, config: DataHubConfig, *, server_log: Path | None = None) -> None:
        """
        server_log: where the stdio MCP server's stderr goes. The DataHub MCP
            server logs every GraphQL query at DEBUG, which floods the terminal
            and buries the sentinel's own output. Routing it to a file keeps it
            available for debugging without making the CLI unreadable. `None`
            discards it.
        """
        config.validate()
        self._config = config
        self._server_log = server_log
        self._session: ClientSession | None = None
        self._stack: AsyncExitStack | None = None
        self._inventory = ToolInventory()

    @property
    def session(self) -> ClientSession:
        if self._session is None:
            raise MCPConnectionError("DataHubMCP is not connected; use `async with`")
        return self._session

    @property
    def inventory(self) -> ToolInventory:
        return self._inventory

    def _open_server_log(self) -> TextIO:
        """Open the sink for the MCP server subprocess's stderr."""
        if self._server_log is None:
            return open(os.devnull, "w")  # noqa: SIM115 - closed by the exit stack
        self._server_log.parent.mkdir(parents=True, exist_ok=True)
        return open(self._server_log, "a", encoding="utf-8")  # noqa: SIM115 - ditto

    async def __aenter__(self) -> DataHubMCP:
        stack = AsyncExitStack()
        self._stack = stack
        try:
            if self._config.mode is Mode.SELFHOSTED:
                if shutil.which("uvx") is None:
                    raise MCPConnectionError(
                        "`uvx` not found on PATH. Install uv (https://docs.astral.sh/uv/) — "
                        "the self-hosted DataHub MCP server runs via `uvx mcp-server-datahub`."
                    )
                params = StdioServerParameters(
                    command="uvx",
                    args=["mcp-server-datahub@latest"],
                    env=self._config.mcp_server_env(),
                )
                errlog = stack.enter_context(self._open_server_log())
                read, write = await stack.enter_async_context(
                    stdio_client(params, errlog=errlog)
                )
            else:
                streams = await stack.enter_async_context(
                    streamablehttp_client(
                        self._config.cloud_mcp_endpoint,
                        headers={"Authorization": f"Bearer {self._config.token}"},
                    )
                )
                # streamablehttp_client yields (read, write, get_session_id).
                read, write = streams[0], streams[1]

            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self._session = session
        except MCPConnectionError:
            await stack.aclose()
            self._stack = None
            raise
        except Exception as exc:  # noqa: BLE001 - surface transport detail to the user
            await stack.aclose()
            self._stack = None
            raise MCPConnectionError(
                f"could not connect to the DataHub MCP server "
                f"({self._config.mode.value}): {exc}"
            ) from exc

        listed = await self._session.list_tools()
        tools = list(listed.tools or [])
        self._inventory = ToolInventory(
            names=tuple(tool.name for tool in tools),
            raw=tools,
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self._session = None
        stack, self._stack = self._stack, None
        if stack is not None:
            await stack.aclose()

    # --- direct tool calls (deterministic paths, no model in the loop) -------

    async def call(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """Call an MCP tool and return its decoded payload.

        Used by the deterministic parts of the sentinel — snapshotting schemas,
        walking lineage — where sending the model is pure overhead. The agent
        reaches the same tools through the tool runner.
        """
        if name not in self._inventory.names:
            raise MCPConnectionError(
                f"tool {name!r} is not exposed by this DataHub MCP server. "
                f"Available: {', '.join(sorted(self._inventory.names)) or '(none)'}"
            )
        result = await self.session.call_tool(name, arguments or {})
        if getattr(result, "isError", False) or getattr(result, "is_error", False):
            raise MCPConnectionError(f"DataHub tool {name!r} failed: {_render(result)}")
        return _decode(result)

    def anthropic_tools(self, *, include_writes: bool = True) -> list[Any]:
        """Wrap the MCP tools for the Anthropic tool runner."""
        from anthropic.lib.tools.mcp import async_mcp_tool

        selected = []
        for tool in self._inventory.raw:
            if not include_writes and tool.name in WRITE_TOOLS:
                continue
            selected.append(async_mcp_tool(tool, self.session))
        return selected


def _decode(result: Any) -> Any:
    """Prefer MCP structured content; fall back to parsing text blocks as JSON."""
    for attr in ("structuredContent", "structured_content"):
        structured = getattr(result, attr, None)
        if structured is not None:
            return structured

    chunks: list[str] = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if text:
            chunks.append(text)
    if not chunks:
        return None
    joined = "\n".join(chunks)
    try:
        return json.loads(joined)
    except json.JSONDecodeError:
        return joined


def _render(result: Any) -> str:
    payload = _decode(result)
    if isinstance(payload, str):
        return payload
    return json.dumps(payload, default=str)[:500]
