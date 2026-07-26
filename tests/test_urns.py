"""URN parsing must survive nested platform URNs and varying key arity."""

from __future__ import annotations

import pytest

from forecast_sentinel.datahub.urns import (
    UrnParseError,
    dataset_urn,
    is_urn,
    ml_feature_urn,
    ml_model_urn,
    parse_urn,
    schema_field_urn,
)


class TestNestedUrns:
    """The whole point: a naive split(",") corrupts every platform URN."""

    def test_dataset_urn_keeps_platform_intact(self):
        urn = parse_urn("urn:li:dataset:(urn:li:dataPlatform:snowflake,db.schema.tbl,PROD)")
        assert urn.entity_type == "dataset"
        assert urn.platform == "snowflake"
        assert urn.name == "db.schema.tbl"
        assert urn.env == "PROD"
        assert urn.is_production

    def test_schema_field_urn_contains_a_whole_dataset_urn(self):
        dataset = dataset_urn("snowflake", "sales.raw_sales")
        urn = parse_urn(schema_field_urn(dataset, "holiday_flag"))
        assert urn.entity_type == "schemaField"
        assert len(urn.parts) == 2
        # The nested dataset URN must come back whole, not split on its commas.
        assert urn.parts[0] == dataset
        assert urn.parts[1] == "holiday_flag"

    def test_ml_feature_three_part_key(self):
        urn = parse_urn(ml_feature_urn("feast", "sales_features", "holiday_flag"))
        assert urn.entity_type == "mlFeature"
        assert urn.platform == "feast"
        assert urn.name == "sales_features"
        # No fabric on a feature key, so env must be None rather than the feature name.
        assert urn.env is None


class TestArityTolerance:
    def test_two_part_key_without_fabric(self):
        urn = parse_urn("urn:li:mlFeatureTable:(urn:li:dataPlatform:feast,sales_features)")
        assert urn.name == "sales_features"
        assert urn.env is None

    def test_single_part_key(self):
        urn = parse_urn("urn:li:dataProcessInstance:abc123")
        assert urn.entity_type == "dataProcessInstance"
        assert urn.parts == ("abc123",)
        assert urn.name == "abc123"
        assert urn.platform is None

    def test_non_prod_fabric_recognised(self):
        urn = parse_urn(dataset_urn("bigquery", "proj.ds.tbl", env="DEV"))
        assert urn.env == "DEV"
        assert not urn.is_production


class TestRejections:
    @pytest.mark.parametrize(
        "bad",
        [
            "not-a-urn",
            "urn:li:",
            "urn:li:dataset",
            "urn:li:dataset:(urn:li:dataPlatform:snowflake,tbl,PROD",
            "urn:li:dataset:urn:li:dataPlatform:snowflake,tbl,PROD)",
            "urn:li:dataset:(,tbl,PROD)",
        ],
    )
    def test_malformed_raises(self, bad):
        with pytest.raises(UrnParseError):
            parse_urn(bad)

    def test_non_string_raises(self):
        # URNs arrive from MCP payloads, so the type guard is load-bearing.
        with pytest.raises(UrnParseError):
            parse_urn({"urn": "urn:li:dataset:(x,y,PROD)"})

    def test_is_urn_never_raises(self):
        assert is_urn(ml_model_urn("mlflow", "demand-forecast-v3"))
        assert not is_urn("nonsense")
        assert not is_urn(None)
        assert not is_urn(42)


def test_short_label_is_human_readable():
    urn = parse_urn(ml_model_urn("mlflow", "demand-forecast-v3", env="PROD"))
    assert urn.short() == "mlflow:demand-forecast-v3 [PROD]"


def test_whitespace_is_tolerated():
    urn = parse_urn("  urn:li:dataset:(urn:li:dataPlatform:hive,tbl,PROD)  ")
    assert urn.name == "tbl"
