"""Assemble the ML lineage neighbourhood of a deployed model.

DataHub models the ML path as a chain of distinct entity types:

    dataset ──> dataset ──> dataJob (training run) ──> mlModel ──> deployment

The sentinel's whole premise is that this chain is queryable, so a change to a
column in a warehouse table can be connected to a model serving predictions in
production.

Verified against a live self-hosted DataHub 1.5.0.6 MCP server. Two things that
the documentation does not make obvious, and which an earlier version of this
module got wrong:

* `get_lineage` takes `upstream: bool` + `max_hops: int` — **not** a
  `direction: "UPSTREAM"` string. It performs the multi-hop walk server-side, so
  one call per direction returns the whole transitive neighbourhood. There is no
  need to breadth-first search hop by hop from the client.
* The response nests entities as
  `{"upstreams": {"searchResults": [{"entity": {...}}], "hasMore": bool}}`.
  The `entity.type` field is SCREAMING_SNAKE (`DATASET`, `DATA_JOB`), which does
  not match DataHub's URN entity types (`dataset`, `dataJob`) — so entity type is
  derived from the URN, never from that field.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from forecast_sentinel.datahub.mcp_client import DataHubMCP
from forecast_sentinel.datahub.urns import Urn, UrnParseError, parse_urn

# Entity types that represent something *serving* predictions or numbers.
SERVING_TYPES = frozenset(
    {"mlModelDeployment", "dashboard", "chart", "dataProduct", "mlPrimaryKey"}
)


class LineageError(RuntimeError):
    """Raised when lineage cannot be retrieved at all.

    Distinct from an empty result: a model with no recorded lineage is a valid,
    reportable finding, whereas a failing `get_lineage` call is a defect that
    must not be presented as "no lineage found".
    """


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
        """Derived from the URN, not from the payload's `type` field."""
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
    max_hops: int = 0
    truncated: bool = False
    errors: list[str] = field(default_factory=list)

    def add(self, asset: Asset, *, direction: str) -> None:
        target = self.upstream if direction == "upstream" else self.downstream
        existing = target.get(asset.urn.raw)
        # Later payloads are sometimes richer; prefer the one carrying a name.
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
            "max_hops": self.max_hops,
            "truncated": self.truncated,
            "owners": list(self.owners()),
            "errors": list(self.errors),
        }


async def build_ml_lineage(
    mcp: DataHubMCP,
    root_urn: str,
    *,
    max_hops: int = 4,
    max_results: int = 200,
) -> MLLineageGraph:
    """Fetch the upstream and downstream neighbourhood of `root_urn`.

    One `get_lineage` call per direction; DataHub does the multi-hop walk.
    `max_hops` and `max_results` bound it so a densely connected catalog cannot
    turn one check into a warehouse-wide crawl, and `hasMore` from the server
    sets `truncated` so a partial blast radius is never reported as complete.

    Raises LineageError if lineage could not be retrieved in either direction —
    a systematic failure must not masquerade as "this model has no lineage".
    """
    root = parse_urn(root_urn)
    graph = MLLineageGraph(root=root, max_hops=max_hops)

    for direction, upstream in (("upstream", True), ("downstream", False)):
        try:
            payload = await mcp.call(
                "get_lineage",
                {
                    "urn": root.raw,
                    "upstream": upstream,
                    "max_hops": max_hops,
                    "max_results": max_results,
                },
            )
        except Exception as exc:  # noqa: BLE001 - recorded, and re-raised below if total
            graph.errors.append(f"{direction}: {exc}")
            continue

        assets, has_more = _parse_lineage_payload(payload, direction=direction)
        for asset in assets:
            if asset.urn.raw == root.raw:
                continue
            graph.add(asset, direction=direction)
        if has_more:
            graph.truncated = True

    if len(graph.errors) == 2:
        raise LineageError(
            "get_lineage failed in both directions for "
            f"{root.raw}: " + "; ".join(graph.errors)
        )
    return graph


def _parse_lineage_payload(
    payload: Any, *, direction: str = "upstream"
) -> tuple[list[Asset], bool]:
    """Extract assets and the has-more flag from a `get_lineage` response.

    Canonical live shape:
        {"upstreams": {"total": 3, "searchResults": [{"entity": {...}}],
                       "hasMore": false}}

    Older/alternative shapes are still accepted because DataHub payloads vary by
    version and platform, and a tolerant parser here is cheaper than a broken
    run. Anything unrecognisable is dropped rather than raised on.
    """
    if payload is None:
        return [], False

    envelope: Any = payload
    has_more = False

    if isinstance(payload, dict):
        # Unwrap the directional envelope: upstreams / downstreams.
        for key in (f"{direction}s", "upstreams", "downstreams", "lineage"):
            nested = payload.get(key)
            if isinstance(nested, dict):
                envelope = nested
                break

    if isinstance(envelope, dict):
        has_more = bool(envelope.get("hasMore") or envelope.get("has_more"))

    entries: Iterable[Any]
    if isinstance(envelope, list):
        entries = envelope
    elif isinstance(envelope, dict):
        for key in (
            "searchResults",
            "search_results",
            "relationships",
            "entities",
            "results",
            "items",
            "data",
        ):
            value = envelope.get(key)
            if isinstance(value, list):
                entries = value
                break
        else:
            return [], has_more
    else:
        return [], has_more

    assets: list[Asset] = []
    for entry in entries:
        asset = _to_asset(entry)
        if asset is not None:
            assets.append(asset)
    return assets, has_more


def _to_asset(entry: Any) -> Asset | None:
    if isinstance(entry, str):
        try:
            return Asset(urn=parse_urn(entry))
        except UrnParseError:
            return None
    if not isinstance(entry, dict):
        return None

    # The entity is usually nested one level down.
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
    editable = node.get("editableProperties")
    if not isinstance(editable, dict):
        editable = {}

    return Asset(
        urn=urn,
        name=_first_str(
            node.get("name"),
            properties.get("name"),
            node.get("displayName"),
            _nested_str(node, "dataFlow", "flowId"),
        ),
        description=_first_str(
            editable.get("description"),
            node.get("description"),
            properties.get("description"),
        ),
        owners=_string_tuple(node.get("owners") or node.get("ownership")),
        tags=_string_tuple(node.get("tags") or node.get("globalTags")),
        domain=_first_str(node.get("domain")),
    )


def _nested_str(node: dict[str, Any], *path: str) -> str | None:
    current: Any = node
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current if isinstance(current, str) and current.strip() else None


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
                    _nested_str(item, "owner", "urn"),
                    _nested_str(item, "tag", "urn"),
                    item.get("urn"),
                    item.get("owner"),
                    item.get("tag"),
                    item.get("name"),
                )
                if label:
                    out.append(label)
        return tuple(out)
    return ()
