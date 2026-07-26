"""The HTML report must be self-contained and must escape catalog strings.

Dataset names and descriptions come from the catalog, which means they are
user-controlled. The report gets opened from file:// URLs and attached to
tickets, so an injected <script> is a real risk, not a theoretical one.
"""

from __future__ import annotations

from forecast_sentinel.agent.schemas import Decision, RiskItem, Severity, Verdict, WriteBack
from forecast_sentinel.agent.sentinel import SentinelRun
from forecast_sentinel.datahub.ml_lineage import Asset, MLLineageGraph
from forecast_sentinel.datahub.urns import dataset_urn, ml_model_urn, parse_urn
from forecast_sentinel.report.html import render_html_report, write_html_report
from forecast_sentinel.snapshots import DriftEvent, DriftKind

MODEL = ml_model_urn("mlflow", "demand-forecast-v3")
RAW = dataset_urn("snowflake", "sales.raw_sales")


def _run(*, decision: Decision = Decision.BLOCK, label: str = "demand-forecast-v3") -> SentinelRun:
    graph = MLLineageGraph(root=parse_urn(MODEL))
    graph.add(Asset(urn=parse_urn(RAW), name="raw_sales"), direction="upstream")

    return SentinelRun(
        model_urn=MODEL,
        model_label=label,
        started_at="2026-07-26T09:00:00+00:00",
        graph=graph,
        drift=[
            DriftEvent(
                dataset_urn=RAW,
                field_path="holiday_flag",
                kind=DriftKind.TYPE_CHANGED,
                before="INT",
                after="VARCHAR",
            )
        ],
        verdict=Verdict(
            model_urn=MODEL,
            decision=decision,
            headline="holiday_flag changed dtype; the deployed encoding no longer matches.",
            reasoning="The model consumes holiday_flag as an integer feature.",
            risks=[
                RiskItem(
                    title="Feature encoding mismatch",
                    severity=Severity.HIGH,
                    affected_urn=RAW,
                    mechanism="raw_sales.holiday_flag -> feat_seasonality -> demand-forecast-v3",
                    evidence=["dtype INT -> VARCHAR observed vs baseline"],
                )
            ],
            recommended_actions=["Stop the nightly scoring job."],
            downstream_at_risk=["urn:li:dashboard:(looker,exec_demand)"],
            confidence=Severity.HIGH,
            unverified_claims=["Could not confirm whether the pipeline casts the column."],
        ),
        write_backs=[WriteBack(tool="add_tags", target_urn=MODEL, detail="tagged invalidated")],
        notes=["Baseline captured 2026-06-01."],
        token_usage={"input_tokens": 1200, "output_tokens": 800},
    )


class TestSelfContained:
    def test_no_external_resources(self):
        html = render_html_report(_run())
        for forbidden in ("<script", "src=", "https://", "http://", "@import", "cdn"):
            assert forbidden not in html, f"report must not reference {forbidden}"

    def test_supports_light_and_dark(self):
        html = render_html_report(_run())
        assert "prefers-color-scheme: dark" in html

    def test_wide_content_scrolls_rather_than_breaking_the_page(self):
        html = render_html_report(_run())
        assert "overflow-x: auto" in html


class TestContent:
    def test_verdict_and_evidence_are_present(self):
        html = render_html_report(_run())
        assert "BLOCK" in html
        assert "Feature encoding mismatch" in html
        assert "feat_seasonality" in html
        assert "Stop the nightly scoring job." in html
        assert "dtype INT -&gt; VARCHAR" in html or "dtype INT -> VARCHAR" in html

    def test_unverified_claims_are_surfaced_not_hidden(self):
        html = render_html_report(_run())
        assert "Could not verify" in html
        assert "Could not confirm whether the pipeline casts" in html

    def test_write_backs_are_listed(self):
        assert "tagged invalidated" in render_html_report(_run())

    def test_renders_without_a_verdict(self):
        run = _run()
        run.verdict = None
        html = render_html_report(run)
        assert "No verdict was produced" in html
        assert "UNKNOWN" in html


class TestEscaping:
    def test_injected_script_in_a_catalog_name_is_escaped(self):
        run = _run(label="<script>alert('xss')</script>")
        html = render_html_report(run)
        assert "<script>alert" not in html
        assert "&lt;script&gt;" in html

    def test_injected_markup_in_verdict_text_is_escaped(self):
        run = _run()
        assert run.verdict is not None
        run.verdict.headline = "<img src=x onerror=alert(1)>"
        html = render_html_report(run)
        assert "<img src=x" not in html
        assert "&lt;img" in html


class TestWriteFile:
    def test_writes_a_readable_file(self, tmp_path):
        path = write_html_report(_run(), tmp_path)
        assert path.is_file()
        assert path.suffix == ".html"
        assert "BLOCK" in path.read_text(encoding="utf-8")

    def test_filename_is_sanitised(self, tmp_path):
        path = write_html_report(_run(label="a/b:c*d"), tmp_path)
        assert "/" not in path.name
        assert ":" not in path.name
