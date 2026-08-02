"""Schema baselines and drift detection.

The sentinel needs a *trigger*: something changed upstream of a deployed model.
DataHub knows the current schema of every training input, but not what the
schema looked like when the model was trained. So the sentinel records a
baseline at model-validation time and diffs against it on every later run.

Drift detection is deliberately deterministic and model-free — a dtype change
is a fact, not a judgement. The agent's job starts afterwards: deciding whether
a given fact invalidates a given deployed model.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from forecast_sentinel.datahub.mcp_client import DataHubMCP


class DriftKind(str, Enum):
    FIELD_REMOVED = "field_removed"
    FIELD_ADDED = "field_added"
    TYPE_CHANGED = "type_changed"
    NULLABILITY_CHANGED = "nullability_changed"


class SnapshotCaptureError(RuntimeError):
    """Raised when any training-input schema could not be captured.

    A partial current snapshot cannot safely be interpreted as "no drift". The
    caller must surface the incomplete read and retry rather than silently
    skipping the dataset that failed.
    """

    def __init__(self, failures: dict[str, str]) -> None:
        self.failures = failures
        detail = "; ".join(f"{urn}: {error}" for urn, error in failures.items())
        super().__init__(f"could not capture {len(failures)} training-input schema(s): {detail}")


# How dangerous each drift kind is to a *trained* model, before any context.
# Removals and type changes break feature encoding; additions do not.
DRIFT_SEVERITY: dict[DriftKind, str] = {
    DriftKind.FIELD_REMOVED: "high",
    DriftKind.TYPE_CHANGED: "high",
    DriftKind.NULLABILITY_CHANGED: "medium",
    DriftKind.FIELD_ADDED: "low",
}


@dataclass(frozen=True)
class FieldSpec:
    """One column of a training input, as DataHub reports it."""

    path: str
    native_type: str | None = None
    data_type: str | None = None
    nullable: bool | None = None

    def type_label(self) -> str:
        return self.native_type or self.data_type or "unknown"


@dataclass(frozen=True)
class DriftEvent:
    """A single observed schema change."""

    dataset_urn: str
    field_path: str
    kind: DriftKind
    before: str | None = None
    after: str | None = None

    @property
    def severity(self) -> str:
        return DRIFT_SEVERITY[self.kind]

    def describe(self) -> str:
        if self.kind is DriftKind.TYPE_CHANGED:
            return f"{self.field_path}: dtype {self.before} -> {self.after}"
        if self.kind is DriftKind.FIELD_REMOVED:
            return f"{self.field_path}: removed (was {self.before})"
        if self.kind is DriftKind.FIELD_ADDED:
            return f"{self.field_path}: added ({self.after})"
        return f"{self.field_path}: nullability {self.before} -> {self.after}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_urn": self.dataset_urn,
            "field_path": self.field_path,
            "kind": self.kind.value,
            "severity": self.severity,
            "before": self.before,
            "after": self.after,
            "description": self.describe(),
        }


@dataclass
class Snapshot:
    """The recorded schema of every training input of one model."""

    model_urn: str
    captured_at: str
    datasets: dict[str, list[FieldSpec]] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "model_urn": self.model_urn,
                "captured_at": self.captured_at,
                "datasets": {
                    urn: [asdict(spec) for spec in specs]
                    for urn, specs in self.datasets.items()
                },
            },
            indent=2,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, text: str) -> Snapshot:
        payload = json.loads(text)
        datasets = {
            urn: [FieldSpec(**spec) for spec in specs]
            for urn, specs in (payload.get("datasets") or {}).items()
        }
        return cls(
            model_urn=payload["model_urn"],
            captured_at=payload.get("captured_at", ""),
            datasets=datasets,
        )


class SnapshotStore:
    """Baselines on disk, one JSON file per model."""

    def __init__(self, directory: Path) -> None:
        self._dir = directory

    def _path(self, model_urn: str) -> Path:
        return self._dir / f"{_slug(model_urn)}.json"

    def exists(self, model_urn: str) -> bool:
        return self._path(model_urn).is_file()

    def save(self, snapshot: Snapshot) -> Path:
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._path(snapshot.model_urn)
        path.write_text(snapshot.to_json(), encoding="utf-8")
        return path

    def load(self, model_urn: str) -> Snapshot | None:
        path = self._path(model_urn)
        if not path.is_file():
            return None
        return Snapshot.from_json(path.read_text(encoding="utf-8"))

    def list_models(self) -> list[str]:
        if not self._dir.is_dir():
            return []
        models = []
        for path in sorted(self._dir.glob("*.json")):
            try:
                models.append(Snapshot.from_json(path.read_text(encoding="utf-8")).model_urn)
            except (json.JSONDecodeError, KeyError, OSError):
                continue
        return models


async def capture_snapshot(
    mcp: DataHubMCP, model_urn: str, dataset_urns: list[str]
) -> Snapshot:
    """Read every training-input schema, raising if any input cannot be read."""
    datasets: dict[str, list[FieldSpec]] = {}
    failures: dict[str, str] = {}
    for dataset_urn in dataset_urns:
        try:
            payload = await mcp.call("list_schema_fields", {"urn": dataset_urn})
        except Exception as exc:  # noqa: BLE001 - collect every failed input before aborting
            failures[dataset_urn] = str(exc) or type(exc).__name__
            continue
        datasets[dataset_urn] = parse_schema_fields(payload)
    if failures:
        raise SnapshotCaptureError(failures)
    return Snapshot(
        model_urn=model_urn,
        captured_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        datasets=datasets,
    )


def parse_schema_fields(payload: Any) -> list[FieldSpec]:
    """Normalise a `list_schema_fields` payload into FieldSpecs."""
    if payload is None:
        return []

    entries: Any
    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload, dict):
        for key in ("fields", "schemaFields", "schema_fields", "results", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                entries = value
                break
        else:
            return []
    else:
        return []

    specs: list[FieldSpec] = []
    for entry in entries:
        if isinstance(entry, str):
            specs.append(FieldSpec(path=entry))
            continue
        if not isinstance(entry, dict):
            continue
        path = (
            entry.get("fieldPath")
            or entry.get("field_path")
            or entry.get("path")
            or entry.get("name")
        )
        if not isinstance(path, str):
            continue
        nullable = entry.get("nullable")
        specs.append(
            FieldSpec(
                path=path,
                native_type=_as_str(entry.get("nativeDataType") or entry.get("native_data_type")),
                data_type=_as_str(entry.get("type") or entry.get("dataType")),
                nullable=nullable if isinstance(nullable, bool) else None,
            )
        )
    return specs


def diff_snapshots(baseline: Snapshot, current: Snapshot) -> list[DriftEvent]:
    """Compare two snapshots and return every field-level change.

    Datasets present in only one snapshot are skipped rather than reported as a
    wholesale add/remove: lineage itself changing is a different signal, and
    reporting every field of a newly-linked table as drift would bury the
    changes that actually matter.
    """
    events: list[DriftEvent] = []

    for dataset_urn, base_fields in baseline.datasets.items():
        current_fields = current.datasets.get(dataset_urn)
        if current_fields is None:
            continue

        base_by_path = {spec.path: spec for spec in base_fields}
        current_by_path = {spec.path: spec for spec in current_fields}

        for path, base_spec in base_by_path.items():
            current_spec = current_by_path.get(path)
            if current_spec is None:
                events.append(
                    DriftEvent(
                        dataset_urn=dataset_urn,
                        field_path=path,
                        kind=DriftKind.FIELD_REMOVED,
                        before=base_spec.type_label(),
                    )
                )
                continue

            if base_spec.type_label() != current_spec.type_label():
                events.append(
                    DriftEvent(
                        dataset_urn=dataset_urn,
                        field_path=path,
                        kind=DriftKind.TYPE_CHANGED,
                        before=base_spec.type_label(),
                        after=current_spec.type_label(),
                    )
                )

            if (
                base_spec.nullable is not None
                and current_spec.nullable is not None
                and base_spec.nullable != current_spec.nullable
            ):
                events.append(
                    DriftEvent(
                        dataset_urn=dataset_urn,
                        field_path=path,
                        kind=DriftKind.NULLABILITY_CHANGED,
                        before=str(base_spec.nullable),
                        after=str(current_spec.nullable),
                    )
                )

        for path, current_spec in current_by_path.items():
            if path not in base_by_path:
                events.append(
                    DriftEvent(
                        dataset_urn=dataset_urn,
                        field_path=path,
                        kind=DriftKind.FIELD_ADDED,
                        after=current_spec.type_label(),
                    )
                )

    order = {"high": 0, "medium": 1, "low": 2}
    events.sort(key=lambda e: (order[e.severity], e.dataset_urn, e.field_path))
    return events


def _as_str(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, dict):
        for key in ("type", "name"):
            nested = value.get(key)
            if isinstance(nested, str) and nested.strip():
                return nested.strip()
    return None


def _slug(urn: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in urn)[:180]
