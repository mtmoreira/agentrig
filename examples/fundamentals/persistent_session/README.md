# Persistent process session

This example keeps one trusted Python worker alive for two serial requests. It
demonstrates AgentRig-owned lifecycle and transport behavior without assigning
meaning to the opaque request payload:

```text
spawn -> ready -> health -> request 1 -> request 2 -> shutdown -> reap
```

The worker returns a process-local request counter. The final report shows
counts `1` and `2`, proving that both calls reached the same process, while the
session's final state is `closed`. AgentRig supplies versioning, correlation,
bounds, cancellation/deadline handling and cleanup. The application still owns
the payload schema and any transaction or recovery policy.

The example uses an absolute executable, exact argv, empty environment, a
private temporary working directory and no shell or network. The integration
contract separately covers corrupt frames, wrong IDs, output/input bounds,
crash/EOF, deadlines, cancellation, failed shutdown and explicit recovery.

## Files

- `worker.py` — minimal versioned JSONL worker with process-local state
- `scripted.py` — session composition and inspectable report
- `test_scripted.py` — deterministic state-continuity and cleanup assertions
- `BUCK` — runnable binary and focused integration target

## Run

```sh
uv run python -m examples.fundamentals.persistent_session.scripted
./buck2 run //examples/fundamentals/persistent_session:scripted
```

## Verify

```sh
uv run mypy examples/fundamentals/persistent_session
uv run python -m unittest -v examples.fundamentals.persistent_session.test_scripted
./buck2 test //examples/fundamentals/persistent_session:test
```
