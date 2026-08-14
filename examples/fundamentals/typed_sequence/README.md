# Typed sequence, retry, and events

This fundamentals example shows how AgentRig composes ordinary typed Python
functions with an injected step:

```text
RawRequest
  -> normalize: RawRequest -> NormalizedRequest
  -> classify:  NormalizedRequest -> ClassifiedRequest
  -> render:    ClassifiedRequest -> RequestSummary
```

`workflow.py` owns the provider-neutral composition. The classifier is supplied
as a `Step`, so it can be implemented by deterministic code, a direct model
capability, or an agent without changing either adjacent handoff.

`scripted.py` injects an idempotent classifier that fails once with a normalized
transient-provider failure and succeeds on its second attempt. The sequence's
bounded retry policy retries that implementation using the same child context.
The next step receives a separate sibling context, and all lifecycle events
retain the root run as their parent.

The focused tests also prove two important negative boundaries:

- changing the classifier to `NON_REPEATABLE` prevents automatic retry;
- an already-cancelled root context never reaches a step implementation.

Inputs and outputs remain absent from lifecycle events. Only safe step identity,
attempt, effect, status, and normalized failure attributes are emitted.

## Files

- `workflow.py` — typed domain values and provider-neutral sequence builder
- `scripted.py` — transient classifier, deterministic context, and CLI
- `test_scripted.py` — retry, effect, cancellation, lineage, and privacy tests
- `BUCK` — runnable binary and offline integration test

## Run

From the repository root:

```sh
uv run python -m examples.fundamentals.typed_sequence.scripted
./buck2 run //examples/fundamentals/typed_sequence:scripted
```

## Verify

```sh
uv run mypy examples/fundamentals/typed_sequence
uv run python -m unittest -v examples.fundamentals.typed_sequence.test_scripted
./buck2 test //examples/fundamentals/typed_sequence:test
```
