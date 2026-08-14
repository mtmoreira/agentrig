# Structured agent with scripted and Codex runtimes

This example configures one typed decision agent once, then runs it through two
runtime implementations:

```text
DecisionRequest
  -> ConfiguredAgent[DecisionRequest, DecisionBrief]
       -> ScriptedAgentRuntime  (offline and deterministic)
       -> CodexAgentRuntime     (explicit live execution)
  -> DecisionBrief
```

`workflow.py` owns the provider-neutral request and result types, strict JSON
codecs, output schema, instructions, and agent builder. The builder accepts an
`AgentRuntime` and its capability ID, so no Codex SDK type crosses into the
composition layer. The provider schema enforces the structured domain shape;
the typed decoder independently revalidates rules such as at least one risk
and the bounded recommendation vocabulary.

`scripted.py` injects a deterministic scenario. It is the default executable
and the basis of the focused tests. The tests prove typed decoding, exact
runtime authority, sanitized schema failures, and that request/result content
does not enter lifecycle events.

`live.py` is the explicit provider composition root. It installs the Codex
capability ID, official SDK bridge, an ephemeral thread, no tools, a 120-second
deadline, and a read-only sandbox with network access disabled. It requires
`AGENTRIG_RUN_LIVE=1` and uses existing Codex authentication.

## Files

- `workflow.py` — typed domain, strict codecs/schema, and agent builder
- `scripted.py` — deterministic runtime, context, and offline CLI
- `live.py` — bounded Codex runtime wiring and opt-in live CLI
- `test_scripted.py` — offline behavior, authority, schema, and privacy tests
- `BUCK` — separate scripted and live binaries plus the offline test target

## Run offline

```sh
uv run python -m examples.agents.codex_structured_agent.scripted
./buck2 run //examples/agents/codex_structured_agent:scripted
```

## Run live

Authenticate Codex first, then opt in explicitly:

```sh
AGENTRIG_RUN_LIVE=1 \
  uv run python -m examples.agents.codex_structured_agent.live
```

The live entry point defaults to `gpt-5.6-terra`. Override it with
`AGENTRIG_CODEX_LIVE_MODEL` when validating another supported model.
If a live run fails, the CLI reports only AgentRig's normalized failure kind,
safe code, and exception class. It never prints the raw provider error.

## Verify

```sh
uv run mypy examples/agents/codex_structured_agent
uv run python -m unittest -v \
  examples.agents.codex_structured_agent.test_scripted
./buck2 test //examples/agents/codex_structured_agent:test
./buck2 build //examples/agents/codex_structured_agent:live
```
