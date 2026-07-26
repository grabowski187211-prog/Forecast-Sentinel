"""WRITE_TOOLS is a safety boundary, so it gets tested like one.

`anthropic_tools(include_writes=False)` filters on WRITE_TOOLS to keep mutation
tools out of the agent's hands. A tool missing from that tuple is silently handed
to the model — which is exactly what happened: `remove_owners`, `remove_domains`
and `remove_structured_properties` were absent, so the agent would have been
given tools that strip ownership and domains off catalog entities.

The tool names below were captured from a live self-hosted DataHub 1.5.0.6 MCP
server (18 tools, TOOLS_IS_MUTATION_ENABLED=true).
"""

from __future__ import annotations

import pytest

from forecast_sentinel.datahub.mcp_client import READ_TOOLS, WRITE_TOOLS, ToolInventory

# Exactly what the live server exposed.
LIVE_TOOLS = (
    "add_owners",
    "add_structured_properties",
    "add_tags",
    "add_terms",
    "get_dataset_queries",
    "get_entities",
    "get_lineage",
    "get_lineage_paths_between",
    "list_schema_fields",
    "remove_domains",
    "remove_owners",
    "remove_structured_properties",
    "remove_tags",
    "remove_terms",
    "save_document",
    "search",
    "set_domains",
    "update_description",
)


class TestWriteToolCoverage:
    @pytest.mark.parametrize(
        "tool",
        [
            "add_tags",
            "remove_tags",
            "add_terms",
            "remove_terms",
            "add_owners",
            "remove_owners",
            "set_domains",
            "remove_domains",
            "update_description",
            "add_structured_properties",
            "remove_structured_properties",
            "save_document",
        ],
    )
    def test_every_mutating_tool_is_classified_as_a_write(self, tool):
        assert tool in WRITE_TOOLS, (
            f"{tool} mutates the catalog but is not in WRITE_TOOLS, so it would be "
            "handed to the agent by anthropic_tools(include_writes=False)"
        )

    def test_every_remove_counterpart_is_present(self):
        """A regression guard: the original bug was add_* present, remove_* absent."""
        for tool in WRITE_TOOLS:
            if tool.startswith("add_"):
                counterpart = "remove_" + tool[len("add_") :]
                if counterpart in LIVE_TOOLS:
                    assert counterpart in WRITE_TOOLS, (
                        f"{tool} is classified as a write but {counterpart} is not"
                    )

    def test_no_live_mutating_tool_escapes_classification(self):
        """Nothing from the live server should be treated as read-only by accident."""
        unclassified = [
            t
            for t in LIVE_TOOLS
            if t not in WRITE_TOOLS
            and (t.startswith(("add_", "remove_", "set_", "update_", "save_")))
        ]
        assert unclassified == [], f"mutating tools not in WRITE_TOOLS: {unclassified}"

    def test_reads_and_writes_do_not_overlap(self):
        assert set(READ_TOOLS).isdisjoint(WRITE_TOOLS)

    def test_every_live_tool_is_known(self):
        """Surfaces new tools a DataHub upgrade introduces, so they get classified."""
        known = set(READ_TOOLS) | set(WRITE_TOOLS)
        assert set(LIVE_TOOLS) <= known, f"unrecognised: {set(LIVE_TOOLS) - known}"


class TestInventory:
    def test_write_access_detected_from_live_tool_set(self):
        assert ToolInventory(names=LIVE_TOOLS).has_write_access

    def test_write_access_false_when_mutations_disabled(self):
        read_only = tuple(t for t in LIVE_TOOLS if t not in WRITE_TOOLS)
        assert not ToolInventory(names=read_only).has_write_access

    def test_document_tools_reported_missing_on_an_empty_catalog(self):
        """The live server filters document tools out when no documents exist.

        That is expected server behaviour, not a misconfiguration — the check
        exists so the CLI can say so rather than implying something is broken.
        """
        missing = ToolInventory(names=LIVE_TOOLS).missing_read_tools()
        assert set(missing) == {"search_documents", "grep_documents"}

    def test_nothing_missing_when_all_read_tools_present(self):
        assert ToolInventory(names=READ_TOOLS).missing_read_tools() == ()
