# Runtime-backed web research

This example adapts one structured autonomous runtime to AgentRig's
provider-neutral `SearchProvider` contract. Callers receive ordinary
`SearchResult`, `SearchHit`, and `SearchCitation` values and do not depend on a
provider SDK.

The reusable composition in `workflow.py`:

1. preflights the request's citation and result-capacity requirements;
2. grants one web-search tool but no workspace writes;
3. strictly decodes at most three distinct HTTPS sources;
4. rechecks the decoded hits against the caller's `max_results`; and
5. derives retrieval timestamps and duration from the injected run clock.

`scripted.py` runs the adapter deterministically through
`ScriptedAgentRuntime`:

```sh
uv run python -m examples.capabilities.web_research.scripted
```

Run its focused tests with:

```sh
uv run python -m unittest -v \
  examples.capabilities.web_research.test_scripted
./buck2 test //examples/capabilities/web_research:test
```

## Explicit live Codex run

`live.py` is the only provider-specific composition root. It uses a read-only
workspace, enables only Codex web search, grants network access, denies
interactive approvals, and returns at most two citation-ready sources.

```sh
uv sync --locked --extra codex
AGENTRIG_RUN_LIVE=1 \
  uv run --extra codex python -m \
    examples.capabilities.web_research.live
```

Override the default model only with an explicit environment value:

```sh
AGENTRIG_RUN_LIVE=1 \
AGENTRIG_CODEX_LIVE_MODEL=gpt-5.6-terra \
  uv run --extra codex python -m \
    examples.capabilities.web_research.live
```

The fixed live query asks for public OpenAI Codex documentation. Output is
limited to citation URLs and titles, result count, model, runtime, and safe
lifecycle event kinds. Prompts, summaries, raw tool payloads, and provider
errors are never printed.
