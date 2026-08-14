# Choose the narrowest abstraction

Select one primary abstraction for each responsibility. Compose them only when the workflow needs the additional semantics.

| Need | Prefer | Why |
| --- | --- | --- |
| Adapt one deterministic async callable | `FunctionStep` | Adds a typed step boundary without inventing an agent |
| Run typed steps in order | `Sequence` | Checks adjacent types and stops on failure |
| Configure model behavior independently of a provider | `AgentContract` with `ConfiguredAgent` | Separates instructions, schemas, tools, permissions, limits, and runtime |
| Invoke an agent inside a workflow | `AgentStep` | Derives step identity and preserves agent failures |
| Expose a workflow through an agent contract | `WorkflowAgent` | Makes an existing workflow substitutable as an agent |
| Express portable generation, search, retrieval, image, coding, or tool behavior | The matching capability protocol | Keeps consumers independent of providers |
| Back a capability with a general runtime | A small application adapter over `AgentRuntime` | Localizes request encoding and result decoding |
| Evaluate output before continuing | `GradeStep` and a `GradePolicy` | Keeps grades separate from control-flow decisions |
| Repair a failing subject with bounded attempts | `RepairLoop` | Carries relevant evidence and enforces a repair budget |
| Gate an effect on a decision | `ApprovalStep` | Ensures denial never invokes the action |
| Run a repeatable regression dataset | `EvalRunner` with an `EvalTarget` | Isolates cases and records deterministic reports |

## Selection rules

- Prefer a capability protocol when the application needs a semantic operation such as search or coding. Prefer `ConfiguredAgent` when the application needs configurable agent behavior with explicit instructions, codecs, tools, and permissions.
- Keep a workflow as a workflow unless consumers specifically need the agent interface. Add `WorkflowAgent` only at that boundary.
- Use `FunctionStep` for deterministic transformations and validations. Do not ask a model to perform a state transition that ordinary code can perform exactly.
- Use one outer authority envelope. Verify that it includes every nested capability and effect without silently expanding permissions.
- Introduce `Sequence` for linear composition. Use an application-owned coordinator when branching or domain state transitions would be clearer as explicit code.

Public workflow exports live in `src/agentrig/workflow/__init__.py`; agent contracts and runtimes live in `src/agentrig/agents/`; portable operations live in `src/agentrig/capabilities/`.
