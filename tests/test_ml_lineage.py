"""Lineage assembly, tested against the payload shapes DataHub actually returns.

The canonical fixtures below were captured from a live self-hosted DataHub
1.5.0.6 MCP server. An earlier version of this module failed three ways at once
against that server, so these tests pin all three:

* `get_lineage` takes `upstream: bool` + `max_hops`, not `direction: "UPSTREAM"`.
* Entities are nested at `{"upstreams": {"searchResults": [{"entity": ...}]}}`.
* A failing call must raise, not present itself as "no lineage found".
"""

from __future__ import annotations

from typing import Any

import pytest

from forecast_sentinel.datahub.ml_lineage import (
    LineageError,
    _parse_lineage_payload,
    build_ml_lineage,
)
from forecast_sentinel.datahub.urns import dataset_urn, ml_model_urn

MODEL = ml_model_urn("mlflow", "demand-forecast-v3")
FEAT = dataset_urn("snowflake", "sales.feat_seasonality")
RAW = dataset_urn("snowflake", "sales.raw_sales")
JOB = "urn:li:dataJob:(urn:li:dataFlow:(airflow,demand_planning,PROD),train_demand_forecast)"
DEPLOYMENT = "urn:li:mlModelDeployment:(urn:li:dataPlatform:sagemaker,planning-api,PROD)"
DASHBOARD = "urn:li:dashboard:(looker,exec_demand)"


def live_envelope(*entities: dict, direction: str = "upstream", has_more: bool = False) -> dict:
    """The exact response shape observed from DataHub 1.5.0.6."""
    return {
        f"{direction}s": {
            "total": len(entities),
            "facets": [{"field": "degree", "aggregations": [{"value": "1", "count": 0}]}],
            "searchResults": [{"entity": e} for e in entities],
            "offset": 0,
            "returned": len(entities),
            "hasMore": has_more,
        }
    }


def entity(urn: str, name: str | None = None, **extra: Any) -> dict:
    node: dict[str, Any] = {"urn": urn}
    if name:
        node["name"] = name
    node.update(extra)
    return node


class FakeMCP:
    """Stands in for DataHubMCP, asserting the call shape as it goes."""

    def __init__(self, upstream: dict | None = None, downstream: dict | None = None) -> None:
        self._up = upstream
        self._down = downstream
        self.calls: list[dict] = []

    async def call(self, name: str, arguments: dict | None = None) -> Any:
        arguments = arguments or {}
        self.calls.append({"tool": name, **arguments})
        assert name == "get_lineage", f"unexpected tool: {name}"
        # The live server rejects `direction`; guard against a regression.
        assert "direction" not in arguments, "get_lineage takes `upstream`, not `direction`"
        assert isinstance(arguments.get("upstream"), bool)
        return self._up if arguments["upstream"] else self._down


class TestCallShape:
    async def test_uses_upstream_bool_and_max_hops(self):
        mcp = FakeMCP(live_envelope(entity(RAW)), live_envelope(direction="downstream"))
        await build_ml_lineage(mcp, MODEL, max_hops=3, max_results=50)
        assert len(mcp.calls) == 2, "one call per direction — the server does the multi-hop walk"
        up = next(c for c in mcp.calls if c["upstream"] is True)
        assert up["max_hops"] == 3
        assert up["max_results"] == 50
        assert up["urn"] == MODEL


class TestLivePayload:
    async def test_parses_the_real_shape_end_to_end(self):
        """The full observed upstream response: a data job plus two datasets."""
        mcp = FakeMCP(
            live_envelope(
                entity(JOB, type="DATA_JOB", dataFlow={"flowId": "demand_planning"}),
                entity(FEAT, "sales.feat_seasonality", type="DATASET"),
                entity(RAW, "sales.raw_sales", type="DATASET"),
            ),
            live_envelope(
                entity(DEPLOYMENT, "planning-api"),
                entity(DASHBOARD, "exec_demand"),
                direction="downstream",
            ),
        )
        graph = await build_ml_lineage(mcp, MODEL)

        assert set(graph.upstream) == {JOB, FEAT, RAW}
        assert set(graph.downstream) == {DEPLOYMENT, DASHBOARD}
        assert graph.blast_radius == 2

    async def test_training_datasets_are_reachable_for_snapshotting(self):
        """The bug that broke `sentinel baseline`: datasets must be findable."""
        mcp = FakeMCP(
            live_envelope(
                entity(JOB, type="DATA_JOB"),
                entity(FEAT, "sales.feat_seasonality"),
                entity(RAW, "sales.raw_sales"),
            ),
            live_envelope(direction="downstream"),
        )
        graph = await build_ml_lineage(mcp, MODEL)
        datasets = sorted(
            a.urn.raw for a in graph.upstream.values() if a.entity_type == "dataset"
        )
        assert datasets == sorted([FEAT, RAW])

    async def test_entity_type_comes_from_the_urn_not_the_payload(self):
        """Payload `type` is SCREAMING_SNAKE (DATA_JOB) and must not be trusted."""
        mcp = FakeMCP(
            live_envelope(entity(JOB, type="DATA_JOB")),
            live_envelope(direction="downstream"),
        )
        graph = await build_ml_lineage(mcp, MODEL)
        assert graph.upstream[JOB].entity_type == "dataJob"

    async def test_editable_description_is_preferred(self):
        mcp = FakeMCP(
            live_envelope(
                entity(RAW, "raw_sales", editableProperties={"description": "edited"},
                       properties={"description": "original"})
            ),
            live_envelope(direction="downstream"),
        )
        graph = await build_ml_lineage(mcp, MODEL)
        assert graph.upstream[RAW].description == "edited"

    async def test_has_more_marks_truncated(self):
        mcp = FakeMCP(
            live_envelope(entity(RAW), has_more=True),
            live_envelope(direction="downstream"),
        )
        graph = await build_ml_lineage(mcp, MODEL)
        assert graph.truncated

    async def test_root_is_not_listed_as_its_own_neighbour(self):
        mcp = FakeMCP(
            live_envelope(entity(MODEL), entity(RAW)),
            live_envelope(direction="downstream"),
        )
        graph = await build_ml_lineage(mcp, MODEL)
        assert MODEL not in graph.upstream


class TestErrorsSurface:
    async def test_total_failure_raises_rather_than_reporting_empty(self):
        """The worst bug this module had: a broken API call looked like empty lineage."""

        class Broken(FakeMCP):
            async def call(self, name, arguments=None):
                raise RuntimeError("unexpected keyword argument")

        with pytest.raises(LineageError, match="both directions"):
            await build_ml_lineage(Broken(), MODEL)

    async def test_one_sided_failure_is_recorded_but_survivable(self):
        class HalfBroken(FakeMCP):
            async def call(self, name, arguments=None):
                if arguments and arguments.get("upstream") is False:
                    raise RuntimeError("GMS timeout")
                return live_envelope(entity(RAW, "raw_sales"))

        graph = await build_ml_lineage(HalfBroken(), MODEL)
        assert RAW in graph.upstream
        assert graph.downstream == {}
        assert any("downstream" in e for e in graph.errors)
        assert graph.summary()["errors"]


class TestPayloadTolerance:
    @pytest.mark.parametrize(
        "key", ["searchResults", "relationships", "entities", "results", "items"]
    )
    def test_alternative_entity_list_keys(self, key):
        assets, _ = _parse_lineage_payload({"upstreams": {key: [{"entity": entity(RAW, "raw")}]}})
        assert [a.urn.raw for a in assets] == [RAW]

    def test_bare_list_of_urn_strings(self):
        assets, _ = _parse_lineage_payload([RAW, MODEL])
        assert {a.urn.raw for a in assets} == {RAW, MODEL}

    def test_flat_entity_without_nesting(self):
        assets, _ = _parse_lineage_payload({"upstreams": {"searchResults": [entity(RAW, "raw")]}})
        assert assets[0].name == "raw"

    def test_entity_urn_alias_key(self):
        assets, _ = _parse_lineage_payload({"upstreams": {"searchResults": [{"entityUrn": RAW}]}})
        assert assets[0].urn.raw == RAW

    def test_garbage_entries_are_dropped_not_raised(self):
        assets, _ = _parse_lineage_payload(
            {
                "upstreams": {
                    "searchResults": [
                        {"urn": "not-a-urn"}, {"x": 1}, 42, None, entity(RAW),
                    ]
                }
            }
        )
        assert [a.urn.raw for a in assets] == [RAW]

    def test_owners_and_tags_in_several_shapes(self):
        assets, _ = _parse_lineage_payload(
            {
                "upstreams": {
                    "searchResults": [
                        entity(
                            RAW,
                            owners=[{"owner": {"urn": "urn:li:corpuser:alice"}},
                                    "urn:li:corpuser:bob"],
                            tags={"tags": [{"tag": {"urn": "urn:li:tag:pii"}}]},
                        )
                    ]
                }
            }
        )
        assert assets[0].owners == ("urn:li:corpuser:alice", "urn:li:corpuser:bob")
        assert assets[0].tags == ("urn:li:tag:pii",)

    def test_none_and_unusable_payloads(self):
        assert _parse_lineage_payload(None) == ([], False)
        assert _parse_lineage_payload({}) == ([], False)
        assert _parse_lineage_payload("nope") == ([], False)


class TestSummary:
    async def test_summary_is_serialisable_and_counts_by_urn_type(self):
        mcp = FakeMCP(
            live_envelope(entity(JOB, type="DATA_JOB"), entity(RAW, "raw_sales")),
            live_envelope(entity(DEPLOYMENT), direction="downstream"),
        )
        graph = await build_ml_lineage(mcp, MODEL, max_hops=4)
        s = graph.summary()
        assert s["root"] == MODEL
        assert s["upstream_count"] == 2
        assert s["downstream_count"] == 1
        assert s["entity_counts"]["dataset"] == 1
        assert s["entity_counts"]["dataJob"] == 1
        assert s["max_hops"] == 4
        assert s["truncated"] is False
