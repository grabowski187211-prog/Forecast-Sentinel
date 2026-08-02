"""Configuration for both DataHub deployment shapes.

The sentinel talks to DataHub only through the DataHub MCP server, but that
server is reached two different ways:

* self-hosted -> spawn `uvx mcp-server-datahub@latest` over stdio, pointed at GMS
* cloud       -> connect to the tenant's HTTP MCP endpoint with a bearer token

Everything downstream of this module is deployment-agnostic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv


class Mode(str, Enum):
    SELFHOSTED = "selfhosted"
    CLOUD = "cloud"


class AgentProvider(str, Enum):
    """Which model API judges the deterministic DataHub evidence."""

    AUTO = "auto"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class ConfigError(RuntimeError):
    """Raised when the environment cannot produce a usable configuration."""


@dataclass(frozen=True)
class DataHubConfig:
    mode: Mode
    gms_url: str | None = None
    gms_token: str | None = None
    tenant_url: str | None = None
    token: str | None = None
    mutation_enabled: bool = True
    tool_response_token_limit: int = 80_000

    @property
    def cloud_mcp_endpoint(self) -> str:
        """The tenant-scoped MCP endpoint, e.g. https://acme.acryl.io/integrations/ai/mcp."""
        if not self.tenant_url:
            raise ConfigError("DATAHUB_TENANT_URL is required when DATAHUB_MODE=cloud")
        return f"{self.tenant_url.rstrip('/')}/integrations/ai/mcp"

    @property
    def ui_url(self) -> str:
        """Best-effort link to the catalog UI, used in reports."""
        if self.mode is Mode.CLOUD and self.tenant_url:
            return self.tenant_url.rstrip("/")
        return "http://localhost:9002"

    def validate(self) -> None:
        if self.mode is Mode.SELFHOSTED:
            if not self.gms_url:
                raise ConfigError("DATAHUB_GMS_URL is required when DATAHUB_MODE=selfhosted")
        else:
            if not self.tenant_url:
                raise ConfigError("DATAHUB_TENANT_URL is required when DATAHUB_MODE=cloud")
            if not self.token:
                raise ConfigError("DATAHUB_TOKEN is required when DATAHUB_MODE=cloud")

    def mcp_server_env(self) -> dict[str, str]:
        """Environment handed to the stdio MCP server subprocess."""
        env = {
            "TOOLS_IS_MUTATION_ENABLED": "true" if self.mutation_enabled else "false",
            "TOOL_RESPONSE_TOKEN_LIMIT": str(self.tool_response_token_limit),
        }
        if self.gms_url:
            env["DATAHUB_GMS_URL"] = self.gms_url
        if self.gms_token:
            env["DATAHUB_GMS_TOKEN"] = self.gms_token
        return env


@dataclass(frozen=True)
class AgentConfig:
    provider: AgentProvider = AgentProvider.AUTO
    openai_model: str = "gpt-5.6"
    anthropic_model: str = "claude-opus-5"
    effort: str = "high"
    # Both provider paths use bounded non-streaming responses. A structured
    # verdict and its read-only tool calls fit comfortably within this budget.
    max_tokens: int = 8_000
    max_iterations: int = 40


@dataclass(frozen=True)
class SentinelConfig:
    datahub: DataHubConfig
    agent: AgentConfig
    state_dir: Path
    fail_on_block: bool = True

    @property
    def baseline_dir(self) -> Path:
        return self.state_dir / "baselines"

    @property
    def run_dir(self) -> Path:
        return self.state_dir / "runs"

    @property
    def mcp_server_log(self) -> Path:
        """Where the stdio MCP server's stderr is parked.

        The DataHub MCP server logs every GraphQL query at DEBUG level. Left on
        the terminal it buries the sentinel's own output, so it goes here and
        stays available for debugging.
        """
        return self.state_dir / "mcp-server.log"

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> SentinelConfig:
        load_dotenv(dotenv_path=env_file, override=False)

        raw_mode = os.getenv("DATAHUB_MODE", "selfhosted").strip().lower()
        try:
            mode = Mode(raw_mode)
        except ValueError as exc:
            raise ConfigError(
                f"DATAHUB_MODE must be 'selfhosted' or 'cloud', got {raw_mode!r}"
            ) from exc

        datahub = DataHubConfig(
            mode=mode,
            gms_url=_clean(os.getenv("DATAHUB_GMS_URL")),
            gms_token=_clean(os.getenv("DATAHUB_GMS_TOKEN")),
            tenant_url=_clean(os.getenv("DATAHUB_TENANT_URL")),
            token=_clean(os.getenv("DATAHUB_TOKEN")),
            mutation_enabled=_flag(os.getenv("TOOLS_IS_MUTATION_ENABLED"), default=True),
            tool_response_token_limit=int(os.getenv("TOOL_RESPONSE_TOKEN_LIMIT", "80000")),
        )
        raw_provider = os.getenv("SENTINEL_PROVIDER", "auto").strip().lower()
        try:
            provider = AgentProvider(raw_provider)
        except ValueError as exc:
            raise ConfigError(
                "SENTINEL_PROVIDER must be 'auto', 'openai', or 'anthropic', "
                f"got {raw_provider!r}"
            ) from exc

        # SENTINEL_MODEL was the original Anthropic-only setting. Keep old
        # environments working by routing a legacy Claude slug to Anthropic and
        # any other legacy slug to OpenAI. Provider-specific variables win.
        legacy_model = _clean(os.getenv("SENTINEL_MODEL"))
        legacy_openai = (
            legacy_model
            if legacy_model and not legacy_model.startswith("claude")
            else None
        )
        legacy_anthropic = (
            legacy_model if legacy_model and legacy_model.startswith("claude") else None
        )
        agent = AgentConfig(
            provider=provider,
            openai_model=(
                _clean(os.getenv("SENTINEL_OPENAI_MODEL"))
                or legacy_openai
                or "gpt-5.6"
            ),
            anthropic_model=(
                _clean(os.getenv("SENTINEL_ANTHROPIC_MODEL"))
                or legacy_anthropic
                or "claude-opus-5"
            ),
            effort=os.getenv("SENTINEL_EFFORT", "high"),
        )
        return cls(
            datahub=datahub,
            agent=agent,
            state_dir=Path(os.getenv("SENTINEL_STATE_DIR", ".sentinel")),
            fail_on_block=_flag(os.getenv("SENTINEL_FAIL_ON_BLOCK"), default=True),
        )


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _flag(value: str | None, *, default: bool) -> bool:
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
