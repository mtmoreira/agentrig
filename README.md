# AgentRig

AgentRig is a typed Python SDK for composing deterministic code, AI agents,
model capabilities, retrieval systems, tools, graders, and evaluations into
observable workflows.

The project is pre-alpha. Its public API will emerge through small,
test-supported vertical slices; empty future package trees are intentionally
not scaffolded.

## Development prerequisites

- Python 3.13.14 for the local development environment. The published package
  supports Python 3.12 and newer.
- [uv 0.12.3](https://github.com/astral-sh/uv/releases/tag/0.12.3).
- [DotSlash](https://dotslash-cli.com/docs/installation/) to run the
  repository-pinned Buck2 launcher.

`uv` refuses to run with a different version because its lockfile format and
resolution behavior are part of the reproducibility contract. The checked-in
`./buck2` DotSlash launcher verifies and selects Buck2 `2026-08-01` for the host
platform. That binary carries its matching prelude; `.buckconfig` uses the
bundled, versioned copy rather than a moving branch.

## Bootstrap and verify

From the repository root:

```sh
uv --version
./buck2 --version
uv python install 3.13.14
uv sync --locked --extra codex
uv lock --check
uv run python tools/generate_buck_python_deps.py --check
uv run python tools/typecheck.py
uv run python -m unittest discover -s tests/unit -t .
./buck2 test //... --exclude live --always-exclude
uv build
uv run --isolated --no-project \
  --with ./dist/agentrig-0.1.0-py3-none-any.whl \
  python -c 'import agentrig; print(agentrig.__name__)'
```

The first two commands must report `uv 0.12.3` and a Buck2 build from
`2026-08-01`, respectively. Stop rather than accepting a version mismatch.

The final command proves the built wheel imports from an isolated environment,
not from the source tree or development virtual environment. The earlier `uv`
test command independently checks the editable installation. Buck2 remains the
authoritative build and test front door. Provider-backed tests use the explicit
live execution mode described below.

Optional production extras are resolved once in `uv.lock`. The checked-in Buck2
bridge under `third_party/python` selects the matching hashed wheel for each
supported host; regenerate it with
`uv run python tools/generate_buck_python_deps.py` after changing an extra.

## Test lanes

Every Python test target declares exactly one scope: `unit`, `contract`,
`integration`, or `eval`. Networked provider tests additionally carry the
orthogonal `live` label; all other targets carry `offline`. Unit tests are always
offline and install a process-wide guard before test-module imports. The guard
rejects external Python socket operations and child-process creation while
allowing local `asyncio` wakeup sockets.

Run the complete offline suite by default:

```sh
./buck2 test //... --exclude live --always-exclude
```

Select one scope when iterating:

```sh
./buck2 test //... --include unit
./buck2 test //... --include contract
./buck2 test //... --include integration
./buck2 test //... --include eval
```

Live tests are never silently enabled and must fail when provider credentials
are absent. Opt in explicitly through Buck2's test executor:

```sh
./buck2 test //... --include live -- \
  --env AGENTRIG_RUN_LIVE=1
```

See [the architecture](docs/architecture.md) and
[development plan](docs/development-plan.md) for the design and incremental
delivery sequence.
