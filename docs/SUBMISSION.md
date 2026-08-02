# Submission checklist

Deadline: **2026-08-10, 5:00 PM EDT**. Requirements digested in
[`01_Reference_Material/hackathon_brief.md`](../01_Reference_Material/hackathon_brief.md).

## Required

| Item | Status |
|---|---|
| Working project using DataHub (MCP Server) | Done — authenticated Gemini run live-verified detection, judgement, and all writes |
| Public repository with source + setup instructions | Done — public `grabowski187211-prog/Forecast-Sentinel` repository |
| Apache 2.0 licence, detectable at repo root | Done — `LICENSE` |
| Text description of features and functionality | Done — `README.md` |
| Demo video, <3 min, public, YouTube/Vimeo/Youku | **Not yet recorded** |
| Sample outputs in `examples/` | Done — case-study dashboard plus JSON, static report, and terminal transcript from the live run |

## Before recording anything

Current verification:

- [x] Unit and orchestration-contract tests pass without DataHub
- [x] `ruff check src/ tests/ scripts/` clean
- [x] All third-party API signatures verified against installed packages
- [x] `seed_ml_demo.py --dry-run` constructs every entity against the real SDK
- [x] Live DataHub 1.5.0.6: 3 upstream / 1 downstream assets, both schemas captured, deliberate `INT → VARCHAR` drift detected
- [x] Live write-back smoke test: obsolete tags removed, `model-invalidated` added, status appended, linked `Analysis` document saved
- [x] Fresh full run: `gemini-3.6-flash` emitted a typed `BLOCK`; all four writes succeeded

The recording gate is clear. Use the case-study dashboard as the video's visual
spine; its claims trace to the checked-in JSON, run report, and terminal
transcript under `examples/`.

```bash
# start Docker/Colima and then:
./scripts/bootstrap_datahub.sh
sentinel doctor
MODEL='urn:li:mlModel:(urn:li:dataPlatform:mlflow,demand-forecast-v3,PROD)'
sentinel baseline "$MODEL"
python scripts/seed_ml_demo.py --break-schema
sentinel check "$MODEL" --json examples/verdict_block.json
```

The README sample now matches the checked-in run. Keep it synchronized if a
later recording run produces different lineage counts or wording.

## Video plan (<3 min)

Equal weighting means the video is worth as much as the code. Structure:

| Time | Beat |
|---|---|
| 0:00–0:25 | **The problem, concretely.** Open the case-study dashboard on “Everything was green. The forecast was not.” State that the endpoint and table remain healthy while `holiday_flag` breaks the model's feature contract. |
| 0:25–0:45 | Flip the schema. Show that DataHub records the change but nothing connects it to the model. Monitoring is green. |
| 0:45–1:45 | `sentinel check`. Let the agent's tool calls scroll — this is the DataHub integration, visible. Land on the BLOCK verdict, the lineage path, the recommended actions, exit code 2. |
| 1:45–2:15 | Refresh the DataHub UI: the model now carries `model-invalidated`, the description shows the verdict, the findings document is attached. **This is the differentiator** — most entries only read. |
| 2:15–2:40 | Return to the dashboard's evidence path and proof links; show the HTML report and mention the CI gate. |
| 2:40–2:55 | Close on the dashboard's deterministic Python → read-only agent → controlled write-back boundary. |

Record the terminal at a legible font size. Do not narrate the architecture
before showing the failure — lead with the problem.

## Scoring self-assessment

Six equally-weighted criteria:

| Criterion | Confidence | Gap |
|---|---|---|
| Use of DataHub | Strong | 9 tools, read **and** write, both deployment shapes |
| Technical execution | Strong | Gemini full path live-verified; OpenAI/Anthropic paths contract-tested; bounded traversal, typed verdict, CI exit codes |
| Originality | Good | Write-back and the deterministic/agentic split are the distinctive parts |
| Real-world usefulness | Strong | Silent invalidation is a real, expensive, common failure |
| Submission quality | Pending | Entirely dependent on the video |
| OSS contribution bonus | **Not addressed** | See below |

## The open-source bonus

Judging awards a bonus for meaningful contributions to DataHub. Candidates
surfaced while building, in order of effort:

1. **Docs fix — MCP client API version.** The Agent Context Kit docs show the
   Python MCP integration without noting that `anthropic[mcp]`'s helper takes a
   1.x `ClientSession`, while the MCP SDK's own docs now describe the 2.x
   `Client`. That cost real time here and would cost every Python entrant the
   same. Small, concrete, genuinely useful.
2. **Docs fix — `mlFeature` URN arity.** The metamodel page shows
   `urn:li:mlFeature:(platform,table,feature)`; other DataHub sources show a
   two-part key. Worth reporting even if the answer is "version-dependent".
3. **SDK gap — ML lineage.** `client.lineage.add_lineage` rejects `mlModel`
   URNs, and `datahub.sdk` exposes no `mlFeatures` setter, so the
   dataset→feature→model path cannot be built with the high-level SDK. That is a
   real gap for exactly the ML-lineage use case this challenge is about. An issue
   with a reproduction is cheap; a PR is not.

Do (1) and (2) as documentation PRs and file (3) as an issue. Reference them in
the submission description.

## Disclosure

The rules require disclosing pre-existing code. Nothing in this repository is
pre-existing: it was written during the submission period. Third-party
dependencies (`openai`, `anthropic`, `mcp`, `acryl-datahub`, `pydantic`,
`typer`, `rich`, `jinja2`, `httpx`) are standard libraries used as-is, which the
rules permit without disclosure. Claude Code and OpenAI Codex were used as coding
assistants, which the rules explicitly permit.

## Pre-submit checks

- [ ] `pytest` green
- [ ] `ruff check src/ tests/ scripts/` clean
- [ ] `.env` is **not** committed (`git status --ignored | grep .env`)
- [ ] No tokens in `examples/`, `docs/`, or committed reports
- [ ] `LICENSE` renders as Apache 2.0 on the repo landing page
- [ ] README terminal output matches a real run
- [ ] Video is public, not unlisted-only, and under 3:00
- [ ] Repo clones clean into a fresh directory and `SETUP.md` works verbatim
