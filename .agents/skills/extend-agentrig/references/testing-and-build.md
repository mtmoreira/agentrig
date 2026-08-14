# Testing and build evidence

## Build evidence from the invariant outward

1. Add focused unit tests for valid values and every important impossible state.
2. Test preflight ordering: invalid authority, unsupported requirements, cancellation, and expired deadlines must not call implementations or consume scripts.
3. Test raw exception normalization and assert that private messages do not survive.
4. Add valid typing coverage for public protocols and generic composition. Add an isolated negative fixture only when static rejection is promised behavior.
5. Extend scripted fakes and reusable contract suites for semantics every implementation must share.
6. Add offline integration coverage for SDK packaging, imports, and lifecycle boundaries.
7. Add a separately opted-in live test only when a real provider behavior cannot be established offline.

Prefer exact ordered assertions for calls, events, lineage, artifacts, grades, and outcomes. Inject clocks, IDs, clients, contexts, and scripts; unit tests must not use network, subprocesses, wall time, or credentials.

## Run validation in layers

Start with the changed module's unit tests and type fixture. Then run the affected Buck target before broader lanes.

The standard repository lane is:

```sh
uv sync --locked --extra codex
uv lock --check
uv run python tools/generate_buck_python_deps.py --check
uv run python tools/typecheck.py
uv run python -m unittest discover -s tests/unit -t .
./buck2 test //... --exclude live --always-exclude
uv build
```

Also inspect the built wheel when changing exports, package data, optional dependencies, or repository-only context. Confirm the base installation imports without optional provider SDKs. Typecheck each changed example leaf explicitly because the root typecheck source set excludes examples.

The incompatible-sequence fixture intentionally prints a mypy error; require its success message and zero harness exit status. Run live targets only with exact opt-in, least authority, a deadline, safe logs, and explicit authentication failure.
