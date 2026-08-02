# Build with DataHub: The Agent Hackathon — brief

Digest of <https://datahub.devpost.com>, its Rules and Resources tabs, as read on
**2026-07-26**. Kept here so the requirements do not have to be re-fetched, and
so drift between what we built and what is required is visible.

## Facts

| | |
|---|---|
| Organiser | DataHub (Devpost-hosted, online) |
| Registered participants | 1,930 at time of reading |
| Submission window | 2026-07-06 → **2026-08-10, 5:00 PM EDT** |
| Judging | 2026-08-17 → 2026-08-31 |
| Winners announced | ~2026-09-08 |
| Prize pool | $20,500 |

Prize split: $6,000 grand prize (1) · $3,000 per challenge winner (4) · $1,000
honourable mention (2) · $50 feedback-survey prize (10).

Judges include representatives from Cloudflight, Pinterest, DataHub and OpenAI.

## The four challenges

1. **Agents That Do Real Work** — read DataHub via MCP Server or Agent Context
   Kit, take action, write results back.
2. **Metadata-Aware Code Generation** — generate production data code
   (transformations, DAGs, ingestion, migrations) using DataHub Skills or MCP.
3. **Production ML Agents** ← *we are here* — protect ML models in production
   using end-to-end ML lineage: "the path from training data to features to
   models to deployments".
4. **Open / Wildcard** — creative use of DataHub as a foundation (supply chain,
   forecasting, regulatory automation, other domains).

## Submission requirements

- [ ] Working project using DataHub via at least one of: MCP Server, Agent
      Context Kit, DataHub Skills, Analytics Agent
- [ ] **Public repository** with full source and setup instructions
- [ ] **Apache 2.0 licence file**, detectable at the top of the repository page
- [ ] Text description of features and functionality
- [ ] **Demo video under 3 minutes**, public, on YouTube / Vimeo / Youku
- [ ] Sample outputs recommended (an `examples/` folder)

Rules constraints worth noting:

- Projects must be **newly created during the submission period**. Frameworks,
  libraries, starter templates and AI coding assistants are all permitted, but
  any other pre-existing code must be **disclosed**.
- Entrants must be 18+ / at legal majority. Teams are allowed, with no stated
  maximum size; an individual may join multiple teams.

## Judging criteria

Stage one screens for baseline viability. Stage two applies six
**equally-weighted** criteria:

1. **Use of DataHub** — meaningful integration with the context graph, MCP
   Server, Agent Context Kit, DataHub Skills or Analytics Agent
2. **Technical execution** — quality, robustness, end-to-end functionality
3. **Originality** — creative approaches beyond shipped features
4. **Real-world usefulness** — solving actual practitioner problems
5. **Submission quality** — video, description, README clarity
6. **Bonus** — meaningful open-source contributions to DataHub

Equal weighting is the key planning fact: a brilliant agent with a poor README
scores no better than a mediocre one with a good README. Video and README are
each worth as much as the code.

## Resources published by the organisers

- Docs: <https://docs.datahub.com> · [Quickstart](https://docs.datahub.com/docs/quickstart)
  · [DataHub Skills](https://docs.datahub.com/docs/dev-guides/agent-context/skills)
  · [Agent Context Kit](https://docs.datahub.com/docs/dev-guides/agent-context/agent-context)
  · [Analytics Agent](https://docs.datahub.com/docs/features/feature-guides/analytics-agent)
- Repos: [datahub-project/datahub](https://github.com/datahub-project/datahub)
  · [datahub-project/datahub-skills](https://github.com/datahub-project/datahub-skills)
  · [acryldata/mcp-server-datahub](https://github.com/acryldata/mcp-server-datahub)
- Sample data: `showcase-ecommerce` datapack (~1,049 entities), `bootstrap`
  (lightweight), plus static datasets for
  [NYC Taxi](https://github.com/datahub-project/static-assets/tree/main/datasets/nyc-taxi),
  [Healthcare](https://github.com/datahub-project/static-assets/tree/main/datasets/healthcare),
  [Fiction Retail](https://github.com/datahub-project/static-assets/tree/main/datasets/fiction-retail)
- Community: [DataHub Slack](https://join.slack.com/t/datahubspace/shared_invite/zt-3rxzw3uww-7F2k5mDpjKXIGLskiQPwLQ),
  channel `#agent-hackathon` · [Town Halls](https://datahub.com/community/datahub-town-halls/)
- Support: support@devpost.com

Note: the Slack link in `Links_for_Hackathon.md` is a workspace-internal
`app.slack.com` URL that only resolves for an already-joined member. The public
invite above is the one to share.

## How this project maps to the criteria

| Criterion | Where it is addressed |
|---|---|
| Use of DataHub | Nine MCP tools, read **and** write. Verdicts are persisted via `add_tags` / `update_description` / `save_document`. Both self-hosted and Cloud transports. |
| Technical execution | Deterministic drift detection separated from agentic judgement; bounded lineage traversal; unit and orchestration-contract tests; exits non-zero for CI. |
| Originality | Uses ML lineage to answer a question no monitoring tool answers — connecting a schema change to a *trained artefact*, not just to a table. Writes the verdict back into the catalog. |
| Real-world usefulness | Silent model invalidation from upstream dtype changes is a real, common, expensive failure. The CI gate is the practitioner-facing form. |
| Submission quality | README leads with the problem; `docs/` covers architecture and setup; `examples/` is ready for checked-in live outputs. |
| OSS contribution bonus | Not yet addressed — see `docs/SUBMISSION.md`. |

## Open items

Tracked in `docs/SUBMISSION.md`: the authenticated full agent run, demo outputs
and video, and a decision on whether to attempt the open-source-contribution
bonus.
