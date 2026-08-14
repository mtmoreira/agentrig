# AgentRig repository guidance

## Purpose and boundaries

- Build a provider-neutral, typed Python SDK for composing deterministic code,
  AI capabilities, autonomous agents, graders, and evals.
- Keep application domain schemas and business rules outside AgentRig.
- Implement the smallest coherent vertical slice. Do not scaffold unused future
  packages or broaden a requested change into unrelated cleanup.
- Treat `docs/architecture.md` as the architectural source of truth and
  `docs/development-plan.md` as delivery context. Update them only when a change
  alters an established decision or roadmap claim.

## Repository map

- `src/agentrig/core/`: provider-free execution, events, failures, artifacts,
  grading, policy, and redaction.
- `src/agentrig/workflow/`: typed composition over `core` and `agents`.
- `src/agentrig/agents/`: portable agent and runtime contracts.
- `src/agentrig/capabilities/`: narrow provider-neutral capability contracts.
- `src/agentrig/evals/`: reusable eval execution, reports, and baselines.
- `src/agentrig/testing/`: scripted fakes and reusable contract suites.
- `src/agentrig/integrations/`: optional provider adapters.
- `examples/`: self-contained executable documentation, excluded from the wheel.
- `evals/`: AgentRig's repository-owned datasets, graders, suites, and baselines.
- `third_party/python/locked_deps.bzl`: generated from `uv.lock`; never edit it
  manually.

## Cross-cutting invariants

- Keep dependency direction explicit in package-level BUCK files. `core` must
  not import another AgentRig package or a provider SDK.
- Prefer immutable typed values. Copy and freeze caller-owned collections at
  contract boundaries and validate impossible states during construction.
- Keep private prompts, outputs, evidence, tool arguments, credentials, and raw
  provider errors out of reprs, events, reports, fixtures, and exception text.
- Normalize failures at execution boundaries without retaining unexpected
  exception messages. Preserve only deliberately safe codes and metadata.
- Check capability, authority, cancellation, and deadline constraints before
  invoking an implementation or consuming a scripted outcome.
- Make retries bounded and effect-aware. Never retry a non-repeatable operation
  automatically.
- Use injected clocks and ID generators for deterministic behavior.
- Export supported public symbols from the owning package `__init__.py`; do not
  add convenience exports to the empty root `agentrig.__init__`.

## Dependencies and generated files

- Ask before adding or replacing a production dependency.
- Use `uv.lock` as the dependency source of truth. Regenerate the Buck bridge
  with `uv run python tools/generate_buck_python_deps.py` after approved changes.
- Consumers outside `third_party/python` depend on public `extra-<name>` targets,
  not generated package-private wheel targets.
- Keep optional provider SDK imports behind their integration boundary so the
  base package imports without provider extras.

## Validation

Run the narrowest relevant checks first, then the repository lanes affected by
the change:

```sh
uv sync --locked --extra codex
uv lock --check
uv run python tools/generate_buck_python_deps.py --check
uv run python tools/validate_agent_context.py
uv run python tools/typecheck.py
uv run python -m unittest discover -s tests/unit -t .
./buck2 test //... --exclude live --always-exclude
uv build
```

- Typecheck each changed example leaf explicitly because the root mypy source
  set does not include `examples/`.
- The incompatible-sequence typing fixture intentionally prints a mypy error;
  require the harness success message and zero exit status.
- Run live tests only with explicit opt-in. They must be least-authority,
  deadline-bounded, safe to log, and fail rather than skip when authentication
  is unavailable.
- End text files with exactly one newline and no blank EOF line.
- Before handoff, review the complete diff and report commands, results, live
  lanes not run, and remaining risks.

## Scoped guidance

- Follow `src/agentrig/AGENTS.md` when changing framework packages.
- Follow `examples/AGENTS.md` when adding or changing examples.
- Follow `tests/AGENTS.md` for repository test behavior.
- Follow `evals/AGENTS.md` for repository-owned evaluation assets.
