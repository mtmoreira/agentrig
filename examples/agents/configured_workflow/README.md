# Configured agent inside a workflow

This example follows one typed request across both AgentRig composition
directions:

```text
ScriptedAgentRuntime
  -> ConfiguredAgent[ResearchRequest, ResearchBrief]
  -> AgentStep[ResearchRequest, ResearchBrief]
  -> Sequence[..., DeliveredBrief]
  -> WorkflowAgent[ResearchRequest, DeliveredBrief]
```

`workflow.py` contains only provider-neutral contracts and composition. A
`ConfiguredAgent` binds an injected `AgentRuntime` to a versioned agent contract,
private typed input, strict JSON codecs, instructions, tool allowlist, limits,
and permissions. `AgentStep` then makes that agent an ordinary workflow step.
After deterministic post-processing, `WorkflowAgent` exposes the complete
workflow through a second typed agent contract.

The outer workflow-agent contract repeats the search allowlist, network
permission, and tool-call bound because wrapping the workflow must not understate
the authority exercised by its configured child agent.

`scripted.py` supplies a deterministic runtime scenario with progress and one
allowed search tool call. The runtime returns structured JSON, the output codec
validates it, and the final CLI prints a stable typed summary. Provider session
metadata stays at the runtime boundary and never appears on the portable agent
result.

The focused tests also prove that malformed runtime output becomes a sanitized
schema failure and that a tool outside the agent contract's allowlist is
rejected before any tool lifecycle event is emitted.

## Files

- `workflow.py` — domain types, codecs, contracts, and composition builders
- `scripted.py` — deterministic runtime, execution context, and CLI
- `test_scripted.py` — success, schema, authority, lineage, and privacy tests
- `BUCK` — runnable binary and offline integration test

## Run

From the repository root:

```sh
uv run python -m examples.agents.configured_workflow.scripted
./buck2 run //examples/agents/configured_workflow:scripted
```

## Verify

```sh
uv run mypy examples/agents/configured_workflow
uv run python -m unittest -v examples.agents.configured_workflow.test_scripted
./buck2 test //examples/agents/configured_workflow:test
```
