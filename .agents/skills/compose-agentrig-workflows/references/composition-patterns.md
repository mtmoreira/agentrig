# Composition patterns

## Typed linear execution

Build each step around one input and output type, then compose with `Sequence`. Let type checking reject incompatible adjacent steps. Create the root `RunContext` at the application boundary and allow workflow execution to derive child contexts so lineage, cancellation, deadlines, events, and metadata remain coherent.

Use `ExecutionOutcome` or the corresponding agent/capability result at boundaries. Preserve normalized failures and artifacts instead of raising raw provider exceptions.

## Retry

Attach `RetryPolicy` only to work whose `EffectProfile` permits automatic repetition. Retry classified transient failures within explicit attempt and delay bounds. Do not retry validation failures, policy denials, cancellations, exhausted deadlines, or non-repeatable effects.

## Grade and repair

Treat a failing grade as control-flow data, distinct from grader execution failure.

1. Grade the current typed subject.
2. Let a policy convert the full grade set into continue, repair, approval, or block.
3. If repair is allowed, pass only the current subject and relevant evidence to the repair step.
4. Re-grade the repaired subject and enforce attempt and cost budgets across the loop.
5. Preserve partial evidence when the loop blocks or fails.

Use `examples/workflows/review_repair_approve/` for the complete pattern.

## Approval before effects

Make the approval request describe one scoped action and its authority. Resolve it before invoking the action. A denial is terminal and must not call the action; an unresolved request remains a normalized blocking result. Validate that the resolution matches the original request.

## Capability pipelines

Pass typed results between portable capabilities. When one result depends on another, validate provenance deterministically—for example, require generated citations to refer to returned search hits. Keep provider metadata separate from portable output.

## Runtime-backed agents

Use application codecs to encode domain input into JSON-safe runtime input and decode runtime output into a strict domain type. Reject extra or malformed fields and re-check domain rules that provider schemas cannot express. The same configured agent should run against scripted and live runtimes without changing its contract.
