# Bounded coding capability

This example exposes one structured autonomous runtime through AgentRig's
provider-neutral `CodingAgent` contract. The task authorizes exactly one
ephemeral workspace, one changed file, one turn, eight tool events, and no
network access.

The reusable composition in `workflow.py` does not import Codex. It receives an
`AgentRuntime`, runtime capability ID, tool ID, and retention policy. It then:

1. checks the portable coding descriptor before runtime execution;
2. encodes the private task and its workspace authorization;
3. executes one non-repeatable agent turn;
4. strictly decodes changed-file and validation evidence; and
5. constructs `CodingResult`, which rechecks paths and changed-file bounds
   against the original task.

`scripted.py` runs that same adapter against `ScriptedAgentRuntime`, making the
example deterministic and credential-free:

```sh
uv run python -m examples.capabilities.bounded_coding.scripted
```

Run its focused tests with:

```sh
uv run python -m unittest -v \
  examples.capabilities.bounded_coding.test_scripted
./buck2 test //examples/capabilities/bounded_coding:test
```

## Explicit live Codex run

`live.py` is the only provider-specific composition root. It creates a fresh
temporary workspace, grants workspace-write authority only to that directory,
denies network access and interactive approvals, runs the task, verifies that
only `greeting.py` was reported, and independently compiles that file before
the temporary workspace is removed.

The live run requires the optional dependency and exact opt-in:

```sh
uv sync --locked --extra codex
AGENTRIG_RUN_LIVE=1 \
  uv run --extra codex python -m \
    examples.capabilities.bounded_coding.live
```

Override the default model only with an explicit environment value:

```sh
AGENTRIG_RUN_LIVE=1 \
AGENTRIG_CODEX_LIVE_MODEL=gpt-5.6-terra \
  uv run --extra codex python -m \
    examples.capabilities.bounded_coding.live
```

The example prints only safe status, model, event kinds, changed relative
paths, and host-validation state. It never prints prompts, generated source,
tool arguments, raw validation output, or provider errors.
