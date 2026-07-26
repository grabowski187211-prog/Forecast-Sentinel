"""The verdict contract.

The agent's output is not prose — it is a decision another system can act on.
Modelling it explicitly means the CLI can exit non-zero, CI can gate a
deployment, and the catalog write-back has structured fields to persist.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# `model_urn` collides with pydantic's protected `model_` namespace. The field
# name matters more than the warning: DataHub calls the entity an mlModel and
# renaming it here would desync the schema from the catalog vocabulary.
_ALLOW_MODEL_PREFIX = ConfigDict(protected_namespaces=())


class Decision(str, Enum):
    """What the sentinel concluded about the deployed model."""

    OK = "OK"
    """Change detected, but the deployed model remains valid."""

    WARN = "WARN"
    """Model still serves, but something needs human attention soon."""

    BLOCK = "BLOCK"
    """The deployed artefact is invalidated — stop serving or retrain."""

    UNKNOWN = "UNKNOWN"
    """Not enough catalog context to decide. Treated as a WARN by callers."""

    @property
    def is_actionable(self) -> bool:
        return self in {Decision.WARN, Decision.BLOCK, Decision.UNKNOWN}


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskItem(BaseModel):
    """One concrete way the change threatens the deployed model."""

    title: str = Field(description="One-line statement of the risk.")
    severity: Severity
    affected_urn: str | None = Field(
        default=None,
        description="The DataHub URN of the asset this risk attaches to, if any.",
    )
    mechanism: str = Field(
        description=(
            "How the upstream change reaches the model. Cite the lineage path, "
            "e.g. 'raw_sales.holiday_flag -> feat_seasonality -> demand_model_v3'."
        )
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="Facts from DataHub that support this risk. No speculation.",
    )


class Verdict(BaseModel):
    """The sentinel's decision about one model."""

    model_config = _ALLOW_MODEL_PREFIX

    model_urn: str
    decision: Decision
    headline: str = Field(
        description="One sentence a on-call engineer can act on without reading further."
    )
    reasoning: str = Field(
        description="Why this decision follows from the lineage and the observed change."
    )
    risks: list[RiskItem] = Field(default_factory=list)
    recommended_actions: list[str] = Field(
        default_factory=list,
        description="Ordered, concrete next steps. Most important first.",
    )
    downstream_at_risk: list[str] = Field(
        default_factory=list,
        description="URNs of downstream consumers affected if this model is wrong.",
    )
    confidence: Severity | None = Field(
        default=None, description="How confident the agent is in this decision."
    )
    unverified_claims: list[str] = Field(
        default_factory=list,
        description=(
            "Anything asserted that could not be confirmed from DataHub. "
            "Empty is the good case."
        ),
    )

    @property
    def max_severity(self) -> Severity | None:
        if not self.risks:
            return None
        order = [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]
        return max((r.severity for r in self.risks), key=order.index)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class WriteBack(BaseModel):
    """A record of one mutation the sentinel made to the catalog."""

    tool: str
    target_urn: str
    detail: str
    succeeded: bool = True
    error: str | None = None
