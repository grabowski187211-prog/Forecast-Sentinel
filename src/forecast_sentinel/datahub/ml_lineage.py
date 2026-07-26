"""Assemble the ML lineage neighbourhood of a deployed model.

DataHub models the ML path as a chain of distinct entity types:

    dataset ──DerivedFrom──> mlFeature ──Consumes──> mlModel ──MemberOf──> mlModelGroup
                                            │
                          dataProcessInstance (training run)
                                            │
                                       mlModelDeployment

The sentinel's whole premise is that this chain is queryable, so a change to a
column in a warehouse table can be connected to a model serving predictions in
production. This module turns raw `get_lineage` payloads into a typed graph the
rest of the code can reason about, tolerating the shape drift you get across
DataHub versions and platforms (keys appear as camelCase or snake_case, and
lineage entries are sometimes bare URN strings rather than objects).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from forecast_sentinel.datahub.mcp_client import DataHubMCP
from forecast_sentinel.datahub.urns import Urn, UrnParseError, parse_urn

# Entity types that terminate an upstream walk — going further just enumerates
# the warehouse.
UPSTREAM_STOP_TYPES = frozenset({"dataPlatformInstance", "corpuser", "corpGroup"})

# Entity types that represent something *serving* predictions.
SERVING_TYPES = frozenset({"mlModelDeployment", "dashboard", "chart", "dataProduct"})


@dataclass(frozen=True)
class Asset:
    """A node in the ML lineage graph."""

    urn: Urn
    name: str | None = None
    description: str | None = None
    owners: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    domain: str | None = None

    @property
    def entity_type(self) -> str:
        return self.urn.entity_type

    @property
    def label(self) -> str:
        return self.name or self.urn.short()

    @property
    def is_model(self) -> bool:
        return self.entity_type in {"mlModel", "mlModelGroup"}

    @property
    def is_feature(self) -> bool:
        return self.entity_type in {"mlFeature", "mlFeatureTable", "mlPrimaryKey"}

    @property
    def is_serving(self) -> bool:
        return self.entity_type in SERVING_TYPES


@dataclass
class MLLineageGraph:
    """The lineage neighbourhood of one model, split by direction."""

    root: Urn
    upstream: dict[str, Asset] = field(default_factory=dict)
    downstream: dict[str, Asset] = field(default_factory=dict)
    edges: set[tuple[str, str]] = field(default_factory=set)
    depth_reached: int = 0
    truncated: bool = False

    def add(self, asset: Asset, *, direction: str) -> None:
        target = self.upstream if direction == "upstream" else self.downstream
        existing = target.get(asset.urn.raw)
        # Later payloads are usually richer (get_entities vs get_lineage stubs).
        if existing is None or (asset.name and not existing.name):
            target[asset.urn.raw] = asset

    @property
    def all_assets(self) -> dict[str, Asset]:
        return {**self.upstream, **self.downstream}

    def of_type(self, *entity_types: str) -> list[Asset]:
        wanted = set(entity_types)
        return [a for a in self.all_assets.values() if a.entity_type in wanted]

    @property
    def training_inputs(self) -> list[Asset]:
        """Datasets and features the model was trained on."""
        wanted = {"dataset", "mlFeature", "mlFeatureTable"}
        return [a for a in self.upstream.values() if a.entity_type in wanted]

    @property
    def serving_surfaces(self) -> list[Asset]:
        """Everything downstream that exposes the model's output."""
        return [a for a in self.downstream.values() if a.is_serving]

    @property
    def blast_radius(self) -> int:
        return len(self.downstream)

    def owners(self) -> tuple[str, ...]:
        seen: list[str] = []
        for asset in self.all_assets.values():
            for owner in asset.owners:
                if owner not in seen:
                    seen.append(owner)
        return tuple(seen)

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for asset in self.all_assets.values():
            counts[asset.entity_type] = counts.get(asset.entity_type, 0) + 1
        return {
            "root": self.root.raw,
            "upstream_count": len(self.upstream),
            "downstream_count": len(self.downstream),
            "entity_counts": counts,
            "depth_reached": self.depth_reached,
            "truncated": self.truncated,
            "owners": list(self.owners()),
        }


async def build_ml_lineage(
    mcp: DataHubMCP,
    root_urn: str,
    *,
    max_depth: int = 4,
    max_nodes: int = 250,
) -> MLLineageGraph:
    """Walk lineage both ways from `root_urn`, breadth-first.

    `get_lineage` returns one hop at a time, so a multi-hop path (table ->
    feature -> model -> deployment) needs an explicit traversal. Bounded by
    `max_depth` and `max_nodes` so a densely connected catalog cannot turn one
    check into an unbounded crawl.
    """
    root = parse_urn(root_urn)
    graph = MLLineageGraph(root=root)

    for direction in ("upstream", "downstream"):
        visited: set[str] = {root.raw}
        queue: deque[tuple[str, int]] = deque([(root.raw, 0)])

        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue
            if len(graph.all_assets) >= max_nodes:
                graph.truncated = True
                break

            try:
                payload = await mcp.call(
                    "get_lineage", {"urn": current, "direction": direction.upper()}
                )
            except Exception:  # noqa: BLE001 - a dead branch must not kill the walk
                continue

            for asset in _parse_lineage_payload(payload):
                if asset.urn.raw in visited:
                    continue
                visited.add(asset.urn.raw)
                graph.add(asset, direction=direction)
                graph.depth_reached = max(graph.depth_reached, depth + 1)

                if direction == "upstream":
                    graph.edges.add((asset.urn.raw, current))
                else:
                    graph.edges.add((current, asset.urn.raw))

                if direction == "upstream" and asset.entity_type in UPSTREAM_STOP_TYPES:
                    continue
                queue.append((asset.urn.raw, depth + 1))

    return graph


def _parse_lineage_payload(payload: Any) -> list[Asset]:
    """Extract assets from a `get_lineage` response.

    Shapes seen in the wild:
        {"relationships": [{"entity": {"urn": ..., "name": ...}}]}
        {"entities": [{"urn": ...}]}
        {"results": [...]}
        [ "urn:li:dataset:(...)", ... ]
    """
    if payload is None:
        return []

    entries: Iterable[Any]
    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload, dict):
        for key in ("relationships", "entities", "results", "data", "lineage", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                entries = value
                break
        else:
            entries = [payload]
    else:
        return []

    assets: list[Asset] = []
    for entry in entries:
        asset = _to_asset(entry)
        if asset is not None:
            assets.append(asset)
    return assets


def _to_asset(entry: Any) -> Asset | None:
    if isinstance(entry, str):
        try:
            return Asset(urn=parse_urn(entry))
        except UrnParseError:
            return None
    if not isinstance(entry, dict):
        return None

    # The entity may be nested one level down.
    node = entry
    for key in ("entity", "node", "asset"):
        nested = entry.get(key)
        if isinstance(nested, dict):
            node = nested
            break

    raw_urn = node.get("urn") or node.get("entityUrn") or node.get("entity_urn")
    if not isinstance(raw_urn, str):
        return None
    try:
        urn = parse_urn(raw_urn)
    except UrnParseError:
        return None

    properties = node.get("properties")
    if not isinstance(properties, dict):
        properties = {}

    return Asset(
        urn=urn,
        name=_first_str(node.get("name"), properties.get("name"), node.get("displayName")),
        description=_first_str(node.get("description"), properties.get("description")),
        owners=_string_tuple(node.get("owners") or node.get("ownership")),
        tags=_string_tuple(node.get("tags") or node.get("globalTags")),
        domain=_first_str(node.get("domain")),
    )


def _first_str(*candidates: Any) -> str | None:
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _string_tuple(value: Any) -> tuple[str, ...]:
    """Coerce the several shapes DataHub uses for owner/tag collections."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, dict):
        for key in ("owners", "tags"):
            nested = value.get(key)
            if isinstance(nested, list):
                return _string_tuple(nested)
        return ()
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                label = _first_str(
                    item.get("urn"),
                    item.get("owner"),
                    item.get("tag"),
                    item.get("name"),
                )
                if label:
                    out.append(label)
        return tuple(out)
    return ()
