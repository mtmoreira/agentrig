# Compose with the Codex runtime

Keep Codex at the live composition root. Portable workflow modules should depend on `AgentRuntime`, an agent contract, or a capability protocol rather than importing the Codex SDK.

## Configure a bounded live path

1. Install AgentRig with its `codex` extra and use the repository's pinned dependency set.
2. Construct `CodexAgentRuntime` only in the live entry point or dependency-injection layer.
3. Give each `ConfiguredAgent` an explicit contract, strict codecs, capability identity, tool allowlist, approval policy, sandbox policy, and limits.
4. Prefer ephemeral threads unless continuation is a deliberate application feature. Resume only an explicit thread without expanding authority.
5. Grant the smallest canonical writable scopes. Keep network, workspace writes, and approvals disabled unless the operation requires them.
6. Apply a deadline and cancellation token. Ensure cancellation interrupts the turn and closes the client.
7. Consume only AgentRig's safe lifecycle, usage, tool identity, and normalized failure models. Do not log raw SDK notifications or transport exception messages.

## Choose a reference example

- Strict structured result without tools: `examples/agents/codex_structured_agent/`
- Workspace-bounded coding with host validation: `examples/capabilities/bounded_coding/`
- Web research with validated citations: `examples/capabilities/web_research/`
- Runtime contract and shutdown behavior: `tests/live/test_codex_runtime_live.py`

Build and test the scripted variant first. Keep live execution an explicit command and assert domain output plus safe event kinds, not provider prose or incidental event counts.
