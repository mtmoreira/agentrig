# ADR 0001: Select Codex for the first autonomous runtime

- Status: Accepted for implementation
- Date: 2026-08-13
- Scope: AgentRig Milestone 8 only

## Context

AgentRig needs one autonomous runtime integration to prove that its portable
`AgentRuntime` boundary can support real sessions, streamed lifecycle events,
structured output, cancellation, approvals, and workspace isolation. The
choice must be based on a small current-product spike rather than implied by
the provider-neutral architecture.

This decision selects the first runtime implemented in AgentRig. It does not
make Codex the default for applications, select providers for direct
capabilities, or prevent a later Claude integration.

The spike compared the current official Python surfaces for Codex and the
Claude Agent SDK against the Milestone 8 criteria.

## Comparison

| Criterion | Codex Python SDK and app server | Claude Agent SDK for Python |
| --- | --- | --- |
| Programmatic interface | Stable `openai-codex` package for Python 3.10+; the async client controls a bundled, pinned Codex app-server runtime over JSON-RPC. | Python 3.10+ SDK that supervises a bundled, pinned Claude CLI subprocess. |
| Structured output | A turn accepts a JSON Schema through `outputSchema`. | `output_format` accepts JSON Schema and supports Pydantic-derived schemas. |
| Stream fidelity | Thread, turn, item, approval, and token-usage notifications closely match AgentRig's provider, tool, progress, approval, and usage events. | Async messages can include partial raw API events plus result, tool, and permission messages. |
| Session continuation | Threads can be started, resumed, and forked by identifier. | Sessions can be continued or resumed by identifier; transcript state is stored locally. |
| Tool configuration | Stable built-in command, file-change, and MCP surfaces are available. Dynamic custom tools are experimental and are outside the initial adapter scope. | Built-in tools, custom tools, and MCP servers are supported. `allowed_tools` controls automatic approval rather than forming a complete deny-by-default tool boundary, so `tools` and `disallowed_tools` also require configuration. |
| Cancellation | A running turn can be interrupted and still reaches a terminal turn notification. | `ClaudeSDKClient.interrupt()` supports streaming mode; callers must continue draining messages to observe termination. |
| Approval handling | The app server makes explicit server requests for command execution, file changes, MCP calls, and supported tool approvals. | Permission callbacks and permission modes can mediate tool calls. |
| Workspace and sandbox | Per-turn `readOnly`, `workspaceWrite`, and unrestricted sandbox modes are exposed, including writable roots and network access. | Sandbox and filesystem/network restrictions are configurable, but secure deployment requires explicit hardening and defense in depth. |
| Application authentication | ChatGPT login and API-key authentication are supported; official guidance recommends API keys for programmatic CI. Credentials are read from the environment or Codex credential storage, not request data. | Application API keys and supported cloud-provider credentials are available through the execution environment. |
| Error normalization | JSON-RPC request failures and terminal turn states provide a bounded integration point; AgentRig must map them to its own failure vocabulary. | SDK process, connection, message-parsing, permission, and terminal result failures provide a bounded integration point; AgentRig must map them too. |
| Testability and CI | The pinned runtime, headless protocol, injected SDK client boundary, and API-key CI path support deterministic adapter tests plus a separately marked live lane. | The pinned subprocess runtime, typed messages, and API-key environment also support deterministic adapter tests plus a live lane. |

Both runtimes satisfy the core feasibility bar. Codex is the better first fit
because its thread/turn lifecycle, per-turn schema and sandbox configuration,
interrupt endpoint, and granular app-server notifications map directly onto
AgentRig's existing `AgentExecutionRequest`, `RunContext`, and event vocabulary.
This is an architectural inference from the documented surfaces, not a claim
that Codex is universally preferable.

## Decision

Implement the first autonomous runtime with the stable official
`openai-codex` Python SDK.

The integration will live under `agentrig.integrations.openai` and implement
the existing provider-neutral `AgentRuntime` protocol. Provider types must not
cross that package boundary. `ConfiguredAgent` remains responsible for typed
agent contracts and codecs; the integration translates encoded requests,
provider events, and terminal results.

The initial adapter has these constraints:

- Depend only on a stable, pinned `openai-codex` release. Adding that production
  dependency and updating the lockfile requires explicit approval.
- Put the SDK behind a small injected client protocol so unit and contract tests
  never start the bundled runtime or require credentials.
- Do not depend on the experimental dynamic-tool API. Start with stable Codex
  tool and MCP configuration, and prove deny-by-default tool behavior before
  exposing additional tool paths.
- Translate only an allowlisted set of safe event attributes. Prompts, model
  output, environment values, authentication data, and raw provider payloads
  must not be copied into AgentRig events.
- Configure sandbox, writable roots, network access, and approval policy
  explicitly for every turn. The adapter must never infer unrestricted access
  or silently elevate a contract.
- Keep provider thread identifiers and usage details in provider metadata or
  normalized usage events. Local provider transcripts and credential caches are
  not AgentRig artifacts and must not enter fixtures, reports, or packages.
- Normalize interruption, deadline expiry, refusal, malformed structured
  output, provider/process failure, and approval-required states without
  retaining unsafe exception text.

## Implementation sequence

1. With explicit dependency approval, add a pinned stable `openai-codex`
   dependency and verify uv locking, wheel construction, Buck2 resolution, and
   the supported Python 3.12 and 3.13 platforms.
2. Add the integration package, capability descriptor, injected client
   protocol, and provider-neutral test messages.
3. Implement request configuration, structured-result decoding, cancellation,
   safe event translation, usage capture, and normalized failures.
4. Run the same adapter through two `ConfiguredAgent` contracts and cover
   timeout, interruption, refusal, malformed output, approval, and sandbox
   policy paths in deterministic contract tests.
5. Add an explicitly marked minimal live test plus separate synthetic coding
   and research/search examples using the same runtime.

## Reversal criteria

Reopen this decision before expanding the adapter if any of these conditions is
confirmed during implementation:

- The stable package or bundled runtime cannot be resolved and packaged
  reproducibly by uv and Buck2 on AgentRig's supported Python/platform matrix.
- Stream interruption, JSON Schema output, or per-turn sandbox policy is not
  available through a stable Python SDK surface.
- Deny-by-default tool and approval behavior would require the experimental
  dynamic-tool API or an unsafe provider configuration.
- Authentication or local transcript behavior cannot satisfy AgentRig's
  credential, logging, fixture, report, and retention constraints.
- The event stream cannot be translated without leaking raw prompt, output, or
  credential-bearing provider data.

If a reversal criterion is met, repeat the same bounded dependency and contract
spike against the Claude Agent SDK before selecting another runtime.

## Consequences

- Milestone 8 can proceed without changing the agent or capability contracts.
- The integration owns all Codex SDK imports, process behavior, and provider
  metadata.
- Applications retain explicit provider selection at their composition roots.
- A later Claude adapter can implement the same `AgentRuntime` protocol while
  preserving provider-specific configuration inside its integration package.
- The first implementation slice is gated on approval for a new production
  dependency; this ADR itself adds no runtime dependency.

## Official sources reviewed

- [Codex SDK](https://developers.openai.com/codex/sdk)
- [Codex app server](https://developers.openai.com/codex/app-server)
- [Codex authentication](https://developers.openai.com/codex/auth)
- [Claude Agent SDK for Python](https://code.claude.com/docs/en/agent-sdk/python)
- [Claude Agent SDK overview and terms](https://code.claude.com/docs/en/agent-sdk)
- [Claude structured outputs](https://code.claude.com/docs/en/agent-sdk/structured-outputs)
- [Claude SDK hosting](https://code.claude.com/docs/en/agent-sdk/hosting)
- [Claude secure deployment](https://code.claude.com/docs/en/agent-sdk/secure-deployment)
