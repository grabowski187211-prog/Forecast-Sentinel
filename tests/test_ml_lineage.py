"""Lineage assembly runs against fake MCP payloads.

`get_lineage` returns one hop, so the multi-hop ML path (table -> feature ->
model -> deployment) depends on the BFS here being correct and bounded. A fake
MCP client lets that be tested without a DataHub instance.
"""

from __future__ import annotations

from typing import Any

import pytest

from forecast_sentinel.datahub.ml_lineage import (
    _parse_lineage_payload,
    build_ml_lineage,
)
from forecast_sentinel.datahub.urns import (
    dataset_urn,
    ml_feature_urn,
    ml_model_urn,
)

MODEL = ml_model_urn("mlflow", "demand-forecast-v3")
FEATURE = ml_feature_urn("feast", "sales_features", "holiday_flag")
RAW = dataset_urn("snowflake", "sales.raw_sales")
DEPLOYMENT = "urn:li:mlModelDeployment:(urn:li:dataPlatform:sagemaker,planning-api,PROD)"
DASHBOARD = "urn:li:dashboard:(looker,exec_demand)"


class FakeMCP:
    """Stands in for DataHubMCP, serving a canned one-hop lineage graph."""

    def __init__(self, upstream: dict[str, list[Any]], downstream: dict[str, list[Any]]) -> None:
        self._up = upstream
        self._down = downstream
        self.calls: list[tuple[str, dict]] = []

    async def call(self, name: str, arguments: dict | None = None) -> Any:
        arguments = arguments or {}
        self.calls.append((name, arguments))
        if name != "get_lineage":
            raise AssertionError(f"unexpected tool call: {name}")
        table = self._up if arguments["direction"] == "UPSTREAM" else self._down
        return {"relationships": table.get(arguments["urn"], [])}


def _entity(urn: str, name: str | None = None) -> dict:
    node: dict[str, Any] = {"urn": urn}
    if name:
        node["name"] = name
    return {"entity": node}


class TestPayloadParsing:
    @pytest.mark.parametrize("key", ["relationships", "entities", "results", "data", "items"])
    def test_all_known_envelope_keys(self, key):
        assets = _parse_lineage_payload({key: [_entity(RAW, "raw_sales")]})
        assert len(assets) == 1
        assert assets[0].name == "raw_sales"

    def test_bare_list_of_urn_strings(self):
        assets = _parse_lineage_payload([RAW, MODEL])
        assert {a.urn.raw for a in assets} == {RAW, MODEL}

    def test_flat_entity_without_nesting(self):
        assets = _parse_lineage_payload({"entities": [{"urn": RAW, "name": "raw_sales"}]})
        assert assets[0].name == "raw_sales"

    def test_entity_url_alias_keys(self):
        assets = _parse_lineage_payload({"entities": [{"entityUrn": RAW}]})
        assert assets[0].urn.raw == RAW

    def test_garbage_entries_are_dropped_not_raised(self):
        assets = _parse_lineage_payload(
            {"entities": [{"urn": "not-a-urn"}, {"no_urn": 1}, 42, None, _entity(RAW)]}
        )
        assert [a.urn.raw for a in assets] == [RAW]

    def test_owners_and_tags_in_several_shapes(self):
        assets = _parse_lineage_payload(
            {
                "entities": [
                    {
                        "urn": RAW,
                        "owners": [{"urn": "urn:li:corpuser:alice"}, "urn:li:corpuser:bob"],
                        "tags": {"tags": [{"tag": "urn:li:tag:pii"}]},
                    }
                ]
            }
        )
        assert assets[0].owners == ("urn:li:corpuser:alice", "urn:li:corpuser:bob")
        assert assets[0].tags == ("urn:li:tag:pii",)

    def test_none_payload(self):
        assert _parse_lineage_payload(None) == []


class TestBuildLineage:
    @pytest.fixture
    def mcp(self) -> FakeMCP:
        # raw_sales -> holiday_flag feature -> model -> deployment + dashboard
        return FakeMCP(
            upstream={
                MODEL: [_entity(FEATURE, "holiday_flag")],
                FEATURE: [_entity(RAW, "raw_sales")],
            },
            downstream={
                MODEL: [_entity(DEPLOYMENT, "planning-api"), _entity(DASHBOARD, "exec_demand")],
            },
        )

    async def test_walks_multiple_hops_upstream(self, mcp):
        graph = await build_ml_lineage(mcp, MODEL)
        assert set(graph.upstream) == {FEATURE, RAW}
        assert graph.depth_reached >= 2

    async def test_finds_serving_surfaces_downstream(self, mcp):
        graph = await build_ml_lineage(mcp, MODEL)
        assert set(graph.downstream) == {DEPLOYMENT, DASHBOARD}
        assert graph.blast_radius == 2
        serving = {a.urn.raw for a in graph.serving_surfaces}
        assert DEPLOYMENT in serving
        assert DASHBOARD in serving

    async def test_training_dataset_is_reachable_for_snapshotting(self, mcp):
        graph = await build_ml_lineage(mcp, MODEL)
        datasets = [a.urn.raw for a in graph.upstream.values() if a.entity_type == "dataset"]
        assert datasets == [RAW]

    async def test_edges_point_in_the_direction_of_data_flow(self, mcp):
        graph = await build_ml_lineage(mcp, MODEL)
        assert (RAW, FEATURE) in graph.edges
        assert (FEATURE, MODEL) in graph.edges
        assert (MODEL, DEPLOYMENT) in graph.edges

    async def test_depth_bound_is_respected(self, mcp):
        graph = await build_ml_lineage(mcp, MODEL, max_depth=1)
        assert set(graph.upstream) == {FEATURE}
        assert RAW not in graph.upstream

    async def test_node_bound_marks_truncated(self, mcp):
        graph = await build_ml_lineage(mcp, MODEL, max_nodes=1)
        assert graph.truncated

    async def test_cycle_does_not_hang(self):
        # A ↔ B lineage loop must terminate via the visited set.
        a, b = dataset_urn("hive", "a"), dataset_urn("hive", "b")
        mcp = FakeMCP(upstream={a: [_entity(b)], b: [_entity(a)]}, downstream={})
        graph = await build_ml_lineage(mcp, a, max_depth=10)
        assert set(graph.upstream) == {b}

    async def test_failing_branch_does_not_kill_the_walk(self):
        class PartlyBroken(FakeMCP):
            async def call(self, name: str, arguments: dict | None = None) -> Any:
                if (arguments or {}).get("urn") == FEATURE:
                    raise RuntimeError("GMS timeout on this branch")
                return await super().call(name, arguments)

        mcp = PartlyBroken(
            upstream={MODEL: [_entity(FEATURE)], FEATURE: [_entity(RAW)]},
            downstream={MODEL: [_entity(DEPLOYMENT)]},
        )
        graph = await build_ml_lineage(mcp, MODEL)
        # The feature is still recorded; only its own expansion was lost.
        assert FEATURE in graph.upstream
        assert DEPLOYMENT in graph.downstream

    async def test_summary_is_serialisable(self, mcp):
        graph = await build_ml_lineage(mcp, MODEL)
        summary = graph.summary()
        assert summary["root"] == MODEL
        assert summary["upstream_count"] == 2
        assert summary["downstream_count"] == 2
        assert summary["entity_counts"]["dataset"] == 1
