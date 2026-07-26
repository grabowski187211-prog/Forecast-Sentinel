"""DataHub URN parsing.

DataHub identifies every entity with a URN. The ones the sentinel walks:

    dataset            urn:li:dataset:(urn:li:dataPlatform:snowflake,db.schema.tbl,PROD)
    mlModel            urn:li:mlModel:(urn:li:dataPlatform:mlflow,demand-forecast-v3,PROD)
    mlModelGroup       urn:li:mlModelGroup:(urn:li:dataPlatform:mlflow,demand-forecast,PROD)
    mlFeatureTable     urn:li:mlFeatureTable:(urn:li:dataPlatform:feast,sales_features)
    mlFeature          urn:li:mlFeature:(urn:li:dataPlatform:feast,sales_features,holiday_flag)
    mlPrimaryKey       urn:li:mlPrimaryKey:(urn:li:dataPlatform:feast,sales_features,sku_id)
    schemaField        urn:li:schemaField:(<dataset-urn>,holiday_flag)
    dataProcessInstance    urn:li:dataProcessInstance:<run-id>

The tuple body can contain nested URNs, so splitting on commas requires
tracking parenthesis depth — a naive `split(",")` corrupts every platform URN.
Entity-key arity has also varied across DataHub versions, so the parser keeps
the parts positional rather than hard-coding a shape per entity type.
"""

from __future__ import annotations

from dataclasses import dataclass


class UrnParseError(ValueError):
    """Raised when a string is not a well-formed DataHub URN."""


# DataHub fabric types. Used to decide whether a trailing key component is an
# environment or part of the entity name.
_FABRICS = frozenset(
    {"PROD", "DEV", "QA", "STAGING", "TEST", "UAT", "EI", "NON_PROD", "CORP"}
)


@dataclass(frozen=True)
class Urn:
    """A parsed DataHub URN."""

    raw: str
    entity_type: str
    parts: tuple[str, ...]

    @property
    def platform(self) -> str | None:
        """The dataPlatform name, when the key starts with a platform URN."""
        if not self.parts:
            return None
        first = self.parts[0]
        prefix = "urn:li:dataPlatform:"
        return first[len(prefix) :] if first.startswith(prefix) else None

    @property
    def name(self) -> str | None:
        """The human-meaningful name: the part after the platform, else the first part."""
        parts = list(self.parts)
        if not parts:
            return None
        if self.platform and len(parts) > 1:
            return parts[1]
        return parts[0]

    @property
    def env(self) -> str | None:
        """Fabric/environment (PROD, DEV, QA, ...) when the key carries one."""
        parts = self.parts
        if not parts:
            return None
        last = parts[-1]
        return last if last in _FABRICS else None

    @property
    def is_production(self) -> bool:
        return self.env == "PROD"

    def short(self) -> str:
        """A compact label for reports and log lines."""
        label = self.name or self.raw
        if self.platform:
            label = f"{self.platform}:{label}"
        if self.env:
            label = f"{label} [{self.env}]"
        return label

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.raw


def parse_urn(value: object) -> Urn:
    """Parse a DataHub URN, raising UrnParseError on anything malformed.

    Accepts `object` rather than `str` on purpose: URNs arrive from MCP tool
    payloads, so the type guard is load-bearing at runtime.
    """
    if not isinstance(value, str):
        raise UrnParseError(f"expected a string URN, got {type(value).__name__}")
    text = value.strip()
    if not text.startswith("urn:li:"):
        raise UrnParseError(f"not a DataHub URN (missing 'urn:li:' prefix): {value!r}")

    remainder = text[len("urn:li:") :]
    entity_type, sep, body = remainder.partition(":")
    if not sep or not entity_type:
        raise UrnParseError(f"URN has no entity type: {value!r}")

    if body.startswith("(") and body.endswith(")"):
        parts = _split_top_level(body[1:-1])
    elif body.startswith("(") or body.endswith(")"):
        raise UrnParseError(f"unbalanced parentheses in URN: {value!r}")
    else:
        # Simple single-part key, e.g. urn:li:dataProcessInstance:<id> or a tag.
        parts = (body,)

    if not parts or any(part == "" for part in parts):
        raise UrnParseError(f"URN key has an empty component: {value!r}")

    return Urn(raw=text, entity_type=entity_type, parts=parts)


def _split_top_level(body: str) -> tuple[str, ...]:
    """Split a URN tuple body on commas that are not inside nested parentheses."""
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for char in body:
        if char == "(":
            depth += 1
            current.append(char)
        elif char == ")":
            depth -= 1
            if depth < 0:
                raise UrnParseError(f"unbalanced parentheses in URN key: {body!r}")
            current.append(char)
        elif char == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if depth != 0:
        raise UrnParseError(f"unbalanced parentheses in URN key: {body!r}")
    parts.append("".join(current).strip())
    return tuple(parts)


def is_urn(value: object) -> bool:
    """True when `value` parses as a DataHub URN."""
    if not isinstance(value, str):
        return False
    try:
        parse_urn(value)
    except UrnParseError:
        return False
    return True


# --- constructors (handy for tests, seeding and CLI ergonomics) --------------


def _platform(platform: str) -> str:
    return f"urn:li:dataPlatform:{platform}"


def dataset_urn(platform: str, name: str, env: str = "PROD") -> str:
    return f"urn:li:dataset:({_platform(platform)},{name},{env})"


def ml_model_urn(platform: str, name: str, env: str = "PROD") -> str:
    return f"urn:li:mlModel:({_platform(platform)},{name},{env})"


def ml_model_group_urn(platform: str, name: str, env: str = "PROD") -> str:
    return f"urn:li:mlModelGroup:({_platform(platform)},{name},{env})"


def ml_feature_table_urn(platform: str, table: str) -> str:
    return f"urn:li:mlFeatureTable:({_platform(platform)},{table})"


def ml_feature_urn(platform: str, table: str, feature: str) -> str:
    return f"urn:li:mlFeature:({_platform(platform)},{table},{feature})"


def schema_field_urn(dataset: str, field_path: str) -> str:
    return f"urn:li:schemaField:({dataset},{field_path})"
