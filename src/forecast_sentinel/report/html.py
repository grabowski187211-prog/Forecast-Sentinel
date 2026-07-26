"""Render a sentinel run as a single self-contained HTML file.

Deliberately static: no scripts, no external assets. The report gets attached
to incident tickets and opened from file:// URLs, where anything requiring a
network fetch or a CSP exemption is a liability. Jinja2 autoescaping handles the
catalog strings, which are user-controlled (dataset names, descriptions).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import Environment, select_autoescape

if TYPE_CHECKING:  # pragma: no cover
    from forecast_sentinel.agent.sentinel import SentinelRun

_TEMPLATE = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sentinel — {{ run.model_label }}</title>
<style>
  :root {
    --bg: #fbfaf8; --panel: #ffffff; --ink: #1c1b19; --muted: #6b6862;
    --line: #e3dfd8; --accent: #7c5cff;
    --ok: #2f7d51; --warn: #a86400; --block: #b3261e; --unknown: #5a5a5a;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #17161a; --panel: #1f1e24; --ink: #ecebe8; --muted: #9b978f;
      --line: #322f38; --accent: #a58cff;
      --ok: #6fd39b; --warn: #f0b429; --block: #ff6b60; --unknown: #a0a0a0;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 2rem 1.25rem 4rem;
    background: var(--bg); color: var(--ink);
    font: 16px/1.6 ui-sans-serif, -apple-system, "Segoe UI", system-ui, sans-serif;
  }
  main { max-width: 62rem; margin: 0 auto; }
  header { border-bottom: 1px solid var(--line); padding-bottom: 1.25rem; margin-bottom: 2rem; }
  .eyebrow {
    text-transform: uppercase; letter-spacing: .12em; font-size: .72rem;
    color: var(--muted); margin: 0 0 .4rem;
  }
  h1 { font-size: 1.7rem; margin: 0 0 .75rem; line-height: 1.25; }
  h2 { font-size: 1.15rem; margin: 2.25rem 0 .85rem; }
  h3 { font-size: .98rem; margin: 1.25rem 0 .4rem; }
  code, .urn {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: .82em; word-break: break-all;
  }
  .verdict {
    display: inline-flex; align-items: baseline; gap: .5rem;
    font-weight: 650; font-size: .82rem; letter-spacing: .06em;
    padding: .3rem .7rem; border-radius: 999px; border: 1px solid currentColor;
  }
  .v-OK { color: var(--ok); } .v-WARN { color: var(--warn); }
  .v-BLOCK { color: var(--block); } .v-UNKNOWN { color: var(--unknown); }
  .headline { font-size: 1.12rem; margin: 1.25rem 0 0; }
  .meta { display: flex; flex-wrap: wrap; gap: .4rem 1.5rem; margin-top: 1rem;
          color: var(--muted); font-size: .84rem; }
  .panel {
    background: var(--panel); border: 1px solid var(--line);
    border-radius: .6rem; padding: 1rem 1.15rem; margin: .85rem 0;
  }
  .tiles { display: grid; gap: .75rem; grid-template-columns: repeat(auto-fit, minmax(9rem, 1fr)); }
  .tile { background: var(--panel); border: 1px solid var(--line);
          border-radius: .6rem; padding: .85rem 1rem; }
  .tile .n { font-size: 1.5rem; font-weight: 650; }
  .tile .k { color: var(--muted); font-size: .76rem; text-transform: uppercase;
             letter-spacing: .08em; }
  .risk { border-left: 3px solid var(--line); padding-left: .9rem; margin: 1.1rem 0; }
  .risk.critical, .risk.high { border-left-color: var(--block); }
  .risk.medium { border-left-color: var(--warn); }
  .risk.low { border-left-color: var(--muted); }
  .sev { font-size: .7rem; font-weight: 650; letter-spacing: .08em;
         text-transform: uppercase; color: var(--muted); }
  .path { color: var(--muted); font-size: .86rem; margin: .3rem 0; }
  ul, ol { margin: .5rem 0; padding-left: 1.35rem; }
  li { margin: .25rem 0; }
  .scroll { overflow-x: auto; }
  table { border-collapse: collapse; width: 100%; font-size: .88rem; min-width: 34rem; }
  th, td { text-align: left; padding: .5rem .65rem; border-bottom: 1px solid var(--line); }
  th { color: var(--muted); font-weight: 600; font-size: .75rem;
       text-transform: uppercase; letter-spacing: .07em; }
  .pill { font-size: .68rem; font-weight: 650; text-transform: uppercase;
          letter-spacing: .06em; padding: .12rem .45rem; border-radius: .3rem;
          border: 1px solid currentColor; }
  .p-high { color: var(--block); } .p-medium { color: var(--warn); }
  .p-low { color: var(--muted); }
  .muted { color: var(--muted); }
  footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--line);
           color: var(--muted); font-size: .8rem; }
</style>
</head>
<body>
<main>
  <header>
    <p class="eyebrow">Forecast Model Sentinel</p>
    <h1>{{ run.model_label }}</h1>
    <span class="verdict v-{{ run.decision.value }}">{{ run.decision.value }}</span>
    {% if verdict and verdict.confidence %}
      <span class="muted" style="font-size:.82rem">confidence: {{ verdict.confidence.value }}</span>
    {% endif %}
    {% if verdict %}<p class="headline">{{ verdict.headline }}</p>{% endif %}
    <div class="meta">
      <span class="urn">{{ run.model_urn }}</span>
      <span>checked {{ run.started_at }}</span>
      {% if run.baseline_captured_at %}
        <span>baseline {{ run.baseline_captured_at }}</span>
      {% endif %}
    </div>
  </header>

  <div class="tiles">
    <div class="tile">
      <div class="n">{{ upstream_count }}</div><div class="k">Upstream assets</div>
    </div>
    <div class="tile">
      <div class="n">{{ downstream_count }}</div><div class="k">Blast radius</div>
    </div>
    <div class="tile">
      <div class="n">{{ run.drift | length }}</div><div class="k">Schema changes</div>
    </div>
    <div class="tile">
      <div class="n">{{ verdict.risks | length if verdict else 0 }}</div>
      <div class="k">Risks</div>
    </div>
  </div>

  {% if verdict %}
    <h2>Assessment</h2>
    <div class="panel">{{ verdict.reasoning }}</div>

    {% if verdict.risks %}
      <h2>Risks</h2>
      {% for risk in verdict.risks %}
        <div class="risk {{ risk.severity.value }}">
          <div class="sev">{{ risk.severity.value }}</div>
          <h3>{{ risk.title }}</h3>
          <p class="path">Path: {{ risk.mechanism }}</p>
          {% if risk.affected_urn %}<p class="path urn">{{ risk.affected_urn }}</p>{% endif %}
          {% if risk.evidence %}
            <ul>{% for item in risk.evidence %}<li>{{ item }}</li>{% endfor %}</ul>
          {% endif %}
        </div>
      {% endfor %}
    {% endif %}

    {% if verdict.recommended_actions %}
      <h2>Recommended actions</h2>
      <ol>{% for action in verdict.recommended_actions %}<li>{{ action }}</li>{% endfor %}</ol>
    {% endif %}

    {% if verdict.downstream_at_risk %}
      <h2>Downstream at risk</h2>
      <ul>{% for urn in verdict.downstream_at_risk %}<li class="urn">{{ urn }}</li>{% endfor %}</ul>
    {% endif %}

    {% if verdict.unverified_claims %}
      <h2>Could not verify</h2>
      <div class="panel">
        <ul>{% for claim in verdict.unverified_claims %}<li>{{ claim }}</li>{% endfor %}</ul>
      </div>
    {% endif %}
  {% else %}
    <h2>Assessment</h2>
    <div class="panel muted">No verdict was produced for this run.</div>
  {% endif %}

  {% if run.drift %}
    <h2>Detected upstream changes</h2>
    <div class="scroll">
      <table>
        <thead><tr><th>Severity</th><th>Change</th><th>Dataset</th></tr></thead>
        <tbody>
        {% for event in run.drift %}
          <tr>
            <td><span class="pill p-{{ event.severity }}">{{ event.severity }}</span></td>
            <td>{{ event.describe() }}</td>
            <td class="urn">{{ event.dataset_urn }}</td>
          </tr>
        {% endfor %}
        </tbody>
      </table>
    </div>
  {% endif %}

  {% if run.write_backs %}
    <h2>Written back to DataHub</h2>
    <ul>
      {% for wb in run.write_backs %}
        <li>
          <code>{{ wb.tool }}</code> — {{ wb.detail }}
          {% if not wb.succeeded %}<span class="pill p-high">failed</span>
            <span class="muted">{{ wb.error }}</span>{% endif %}
        </li>
      {% endfor %}
    </ul>
  {% endif %}

  {% if run.notes %}
    <h2>Notes</h2>
    <ul class="muted">{% for note in run.notes %}<li>{{ note }}</li>{% endfor %}</ul>
  {% endif %}

  <footer>
    Generated by Forecast Model Sentinel.
    {% if run.token_usage %}
      Model usage: {{ run.token_usage.get('input_tokens', 0) }} in /
      {{ run.token_usage.get('output_tokens', 0) }} out.
    {% endif %}
  </footer>
</main>
</body>
</html>
"""


def render_html_report(run: SentinelRun) -> str:
    """Render one run to a standalone HTML document."""
    env = Environment(autoescape=select_autoescape(default_for_string=True, default=True))
    template = env.from_string(_TEMPLATE)
    summary = run.graph.summary() if run.graph else {}
    return template.render(
        run=run,
        verdict=run.verdict,
        upstream_count=summary.get("upstream_count", 0),
        downstream_count=summary.get("downstream_count", 0),
    )


def write_html_report(run: SentinelRun, directory: Path) -> Path:
    """Write the report and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in run.model_label)[:80]
    path = directory / f"sentinel_{slug}.html"
    path.write_text(render_html_report(run), encoding="utf-8")
    return path
