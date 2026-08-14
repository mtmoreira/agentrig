# Review, repair, approve

This example builds one typed release workflow:

```text
Draft
  -> grade against a release policy
  -> repair only the failed evidence (bounded to two grading attempts)
  -> prepare a scoped publication request
  -> obtain explicit approval
  -> publish
  -> ReleaseResult
```

The composition in `workflow.py` is provider-neutral. Its grader, repair step,
approval authority, and publisher are injected through AgentRig protocols. The
`scripted.py` entry point supplies deterministic local implementations: the
first grade fails, the repair adds a rollback instruction, the second grade
passes, and the approval authority permits publication.

This separation is intentional. Replacing the scripted grader or repair step
with an agent-backed implementation does not change the workflow's control
flow, repair limit, grading budget, approval boundary, or result type.

## Files

- `workflow.py` — domain values and provider-neutral workflow builder
- `scripted.py` — deterministic dependencies, execution context, and CLI
- `test_scripted.py` — approved and denied publication paths
- `BUCK` — runnable binary and offline integration test

## Run

From the repository root:

```sh
uv run python -m examples.workflows.review_repair_approve.scripted
./buck2 run //examples/workflows/review_repair_approve:scripted
```

The scripted CLI prints a stable JSON summary containing the repaired revision,
final grade, approval, publication destination, and emitted event kinds.

## Verify

```sh
uv run python -m unittest \
  examples.workflows.review_repair_approve.test_scripted
uv run mypy examples/workflows/review_repair_approve
./buck2 test //examples/workflows/review_repair_approve:test
```
