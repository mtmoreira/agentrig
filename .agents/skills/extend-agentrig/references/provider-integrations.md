# Provider integrations and dependencies

## Isolate the SDK

- Define an injected AgentRig-owned client, thread, turn, or transport seam around the SDK. Unit tests should supply that seam without importing the provider package.
- Keep provider types and options inside `agentrig.integrations.<provider>`. Translate them into portable contracts and allowlisted events at the boundary.
- Import optional SDK modules only when constructing the provider bridge. Package initialization and base-wheel imports must remain provider-free.
- Preserve an explicitly named provider-options escape hatch without adding provider flags to portable request types.

## Enforce authority before transport

- Validate capability identity, tools, permissions, workspace roots, network, approvals, and execution limits before client construction and on every resumed turn.
- Repeat bounded authority rather than relying on provider session memory.
- Emit only safe lifecycle, tool identity, and aggregate usage fields. Never forward raw notifications, reasoning, prompts, tool arguments, repository contents, or transport messages.
- Interrupt on cancellation, deadline, or authority violation. Close processes, streams, pipes, and background readers on every terminal path.

## Change dependencies deliberately

Obtain approval before adding or replacing a production dependency. Update `pyproject.toml` through the established optional-extra boundary, regenerate `uv.lock`, then regenerate `third_party/python/locked_deps.bzl` with `tools/generate_buck_python_deps.py`. External Buck consumers depend on the public `extra-<name>` facade, not generated package-private wheel targets.

Verify four distinct surfaces where applicable:

1. Base wheel installs and imports without the SDK.
2. Optional extra installs the exact locked SDK and runtime closure.
3. Buck imports the same closure through its public facade.
4. A real offline lifecycle test initializes and shuts down cleanly; authenticated behavior remains in a separate live target.

Record a provider-selection or protocol-level decision in `docs/adr/` when it changes the architectural commitment.
