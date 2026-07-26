"""Drift detection is the sentinel's trigger, so it gets the closest scrutiny.

These tests deliberately feed the several payload shapes DataHub returns for
`list_schema_fields` across versions and platforms — the parser being tolerant
is a correctness requirement, not a nicety.
"""

from __future__ import annotations

from forecast_sentinel.datahub.urns import dataset_urn, ml_model_urn
from forecast_sentinel.snapshots import (
    DriftKind,
    FieldSpec,
    Snapshot,
    SnapshotStore,
    diff_snapshots,
    parse_schema_fields,
)

DATASET = dataset_urn("snowflake", "sales.raw_sales")
MODEL = ml_model_urn("mlflow", "demand-forecast-v3")


def _snapshot(
    fields: list[FieldSpec], *, captured_at: str = "2026-06-01T00:00:00+00:00"
) -> Snapshot:
    return Snapshot(model_urn=MODEL, captured_at=captured_at, datasets={DATASET: fields})


class TestParseSchemaFields:
    def test_camel_case_payload(self):
        specs = parse_schema_fields(
            {
                "fields": [
                    {"fieldPath": "holiday_flag", "nativeDataType": "INT", "nullable": False},
                    {"fieldPath": "units", "nativeDataType": "FLOAT"},
                ]
            }
        )
        assert [s.path for s in specs] == ["holiday_flag", "units"]
        assert specs[0].type_label() == "INT"
        assert specs[0].nullable is False
        assert specs[1].nullable is None

    def test_snake_case_payload(self):
        specs = parse_schema_fields(
            {"schema_fields": [{"field_path": "sku_id", "native_data_type": "VARCHAR"}]}
        )
        assert specs[0].path == "sku_id"
        assert specs[0].type_label() == "VARCHAR"

    def test_bare_list_of_strings(self):
        specs = parse_schema_fields(["a", "b"])
        assert [s.path for s in specs] == ["a", "b"]
        assert specs[0].type_label() == "unknown"

    def test_nested_type_object(self):
        specs = parse_schema_fields({"fields": [{"path": "ts", "type": {"type": "TIMESTAMP"}}]})
        assert specs[0].type_label() == "TIMESTAMP"

    def test_unusable_payloads_return_empty(self):
        assert parse_schema_fields(None) == []
        assert parse_schema_fields({}) == []
        assert parse_schema_fields("nope") == []
        # Entries without any recognisable path are skipped, not guessed at.
        assert parse_schema_fields({"fields": [{"description": "no path here"}]}) == []


class TestDiff:
    def test_dtype_change_is_high_severity(self):
        before = _snapshot([FieldSpec("holiday_flag", native_type="INT")])
        after = _snapshot([FieldSpec("holiday_flag", native_type="VARCHAR")])
        events = diff_snapshots(before, after)
        assert len(events) == 1
        assert events[0].kind is DriftKind.TYPE_CHANGED
        assert events[0].severity == "high"
        assert "INT -> VARCHAR" in events[0].describe()

    def test_removed_field_is_high_severity(self):
        before = _snapshot([FieldSpec("holiday_flag", native_type="INT")])
        after = _snapshot([])
        events = diff_snapshots(before, after)
        assert events[0].kind is DriftKind.FIELD_REMOVED
        assert events[0].severity == "high"

    def test_added_field_is_low_severity(self):
        before = _snapshot([])
        after = _snapshot([FieldSpec("promo_id", native_type="BIGINT")])
        events = diff_snapshots(before, after)
        assert events[0].kind is DriftKind.FIELD_ADDED
        assert events[0].severity == "low"

    def test_nullability_change_is_medium(self):
        before = _snapshot([FieldSpec("units", native_type="FLOAT", nullable=False)])
        after = _snapshot([FieldSpec("units", native_type="FLOAT", nullable=True)])
        events = diff_snapshots(before, after)
        assert events[0].kind is DriftKind.NULLABILITY_CHANGED
        assert events[0].severity == "medium"

    def test_unknown_nullability_is_not_reported_as_a_change(self):
        # A payload that omits `nullable` must not look like a nullability flip.
        before = _snapshot([FieldSpec("units", native_type="FLOAT", nullable=False)])
        after = _snapshot([FieldSpec("units", native_type="FLOAT", nullable=None)])
        assert diff_snapshots(before, after) == []

    def test_identical_snapshots_produce_nothing(self):
        fields = [FieldSpec("a", native_type="INT"), FieldSpec("b", native_type="STRING")]
        assert diff_snapshots(_snapshot(fields), _snapshot(list(fields))) == []

    def test_events_sort_high_severity_first(self):
        before = _snapshot(
            [FieldSpec("keep", native_type="INT"), FieldSpec("drop", native_type="INT")]
        )
        after = _snapshot(
            [FieldSpec("keep", native_type="STRING"), FieldSpec("new", native_type="INT")]
        )
        events = diff_snapshots(before, after)
        severities = [e.severity for e in events]
        assert severities == sorted(severities, key=["high", "medium", "low"].index)
        assert severities[0] == "high"

    def test_dataset_missing_from_current_is_skipped_not_reported(self):
        """A dataset that dropped out of lineage is a different signal.

        Reporting every field of it as "removed" would bury the real changes.
        """
        before = _snapshot([FieldSpec("a", native_type="INT")])
        after = Snapshot(model_urn=MODEL, captured_at="2026-07-01T00:00:00+00:00", datasets={})
        assert diff_snapshots(before, after) == []

    def test_newly_linked_dataset_is_not_all_additions(self):
        before = Snapshot(model_urn=MODEL, captured_at="2026-06-01T00:00:00+00:00", datasets={})
        after = _snapshot([FieldSpec("a"), FieldSpec("b"), FieldSpec("c")])
        assert diff_snapshots(before, after) == []


class TestSnapshotStore:
    def test_round_trip(self, tmp_path):
        store = SnapshotStore(tmp_path)
        assert not store.exists(MODEL)

        snapshot = _snapshot([FieldSpec("holiday_flag", native_type="INT", nullable=False)])
        store.save(snapshot)

        assert store.exists(MODEL)
        loaded = store.load(MODEL)
        assert loaded is not None
        assert loaded.model_urn == MODEL
        assert loaded.datasets[DATASET][0].native_type == "INT"
        assert loaded.datasets[DATASET][0].nullable is False

    def test_load_missing_returns_none(self, tmp_path):
        assert SnapshotStore(tmp_path).load(MODEL) is None

    def test_list_models(self, tmp_path):
        store = SnapshotStore(tmp_path)
        store.save(_snapshot([FieldSpec("a")]))
        assert store.list_models() == [MODEL]

    def test_corrupt_baseline_is_skipped(self, tmp_path):
        store = SnapshotStore(tmp_path)
        store.save(_snapshot([FieldSpec("a")]))
        (tmp_path / "garbage.json").write_text("{not json", encoding="utf-8")
        # One bad file must not break `sentinel watch`.
        assert store.list_models() == [MODEL]

    def test_urn_becomes_a_safe_filename(self, tmp_path):
        store = SnapshotStore(tmp_path)
        store.save(_snapshot([FieldSpec("a")]))
        written = list(tmp_path.glob("*.json"))
        assert len(written) == 1
        assert "/" not in written[0].name and "(" not in written[0].name
