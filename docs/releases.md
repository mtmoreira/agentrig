# AgentRig release contract

AgentRig releases are immutable, reproducible distribution identities. A
release binds one semantic version, one Git tag, one full source commit, one
wheel, and one source distribution through a checked manifest.

This contract distinguishes two kinds of stability:

- **Artifact stability** means consumers can resolve the same immutable source
  and verify the exact files they received.
- **API stability** describes which compatibility promises apply between
  versions.

AgentRig can provide artifact-stable `0.x` releases while the project remains
pre-alpha. A version in `pyproject.toml` or an editable/path installation alone
is not a release.

## Version and API policy

AgentRig follows Semantic Versioning using `MAJOR.MINOR.PATCH` versions and
matching `vMAJOR.MINOR.PATCH` Git tags.

Until `1.0.0`:

- a minor release may make documented breaking changes to public APIs;
- a patch release must remain backward-compatible with the preceding release
  in the same minor line, except for an explicitly documented security or
  correctness issue that cannot be fixed compatibly;
- every compatibility break must be called out in release notes and must not be
  hidden in a patch release.

After `1.0.0`, normal Semantic Versioning compatibility rules apply.

A symbol is public when it is exported by the owning package's `__init__.py`.
The root `agentrig` package intentionally exports no convenience API today;
consumers import contracts from their owning packages. Examples, tests, tools,
private names, and package internals are not public APIs.

Released versions and tags are never moved, deleted, or reused. A correction
uses a new version.

## Unreleased image runtime compatibility

The next minor candidate adds explicit `ImageInput` roles, image-native usage,
a bounded selected-route executor, and an optional OpenAI image client seam.
Existing `ImageGenerationRequest(reference_images=..., mask=...)` construction
and results without an explicit usage argument remain supported. New edits
should migrate to role-bound `inputs`; callers must not mix legacy and explicit
forms. No release version or immutable tag is assigned until separately
authorized release preparation completes.

## AgentRig 0.2.2

AgentRig 0.2.2 adds a backward-compatible production boundary for strict
multimodal generation through the OpenAI Responses API:

- exported application-owned artifact resolution with private, digest-verified
  bytes;
- an injected Responses client contract and optional official OpenAI Python SDK
  bridge;
- strict JSON-schema output, bounded image and output limits, cancellation and
  deadlines, portable usage, and safe normalized failures;
- stateless, non-streaming, tool-free SDK requests with `store=false`; and
- offline contract/SDK tests plus a separately opted-in synthetic-only live
  image test.

The adapter declares `provider_managed` retention. `store=false` prevents API
response storage for later retrieval but does not claim an organization-level
zero-data-retention policy or bypass provider safety handling. Existing 0.2.1
imports and runtime behavior remain supported.

## AgentRig 0.2.1

AgentRig 0.2.1 adds a backward-compatible portable usage contract to the
autonomous-runtime surface introduced in 0.2.0:

- exported immutable `AgentRuntimeUsage` with optional input, cached-input, and
  output token counts;
- normalized usage attached to successful and failed `AgentExecutionResult`
  values without changing existing constructor call sites;
- stable `usage_reporting` capability identity;
- matching normalized result usage and safe usage events from the scripted,
  Codex, and Ollama runtimes; and
- bounded live Codex and Ollama assertions that verify the result/event usage
  relationship without logging credentials, prompts, reasoning, or output,
  with explicit API-key or ambient Codex authentication selection.

The release adds no provider selection, ranking, fallback, credential storage,
or base dependency. Missing provider counts remain `None`, so downstream
applications can distinguish unknown usage from measured zero usage. Existing
0.2.0 imports and constructor calls remain supported.

## AgentRig 0.2.0

AgentRig 0.2.0 adds application-scoped runtime composition while preserving the
provider-neutral execution boundary established in 0.1.0:

- `agent_runtime` capability identity for portable autonomous-runtime
  requirements;
- an immutable, application-scoped runtime catalog with exact registration and
  resolution semantics, without global provider state or automatic fallback;
- late-bound Codex authentication resolved only when the SDK client is created;
- an exact-pinned optional `ollama` extra, injected structured Ollama runtime,
  official asynchronous SDK bridge, and bounded real-service contract;
- explicit per-binding Ollama thinking configuration, defaulting off for strict
  structured output so thinking models preserve their content token budget;
- corrected Buck unit and live entry points that propagate failing test status.

These additions do not intentionally remove or rename a public 0.1.0 export.
As a pre-1.0 minor release, 0.2.0 establishes the supported baseline for the
new runtime-catalog and Ollama APIs; consumers must still opt into provider SDK
extras explicitly.

## Required release identity

For version `X.Y.Z`, a candidate release must contain exactly:

- tag `vX.Y.Z`;
- a full, lowercase 40-character Git commit SHA;
- `agentrig-X.Y.Z-py3-none-any.whl`;
- `agentrig-X.Y.Z.tar.gz`;
- `agentrig-X.Y.Z-release.json`, generated by the validator.

The manifest uses schema `agentrig-release-manifest.v1` and records the version,
tag, commit, byte size, and SHA-256 digest of each distribution artifact. The
validator also confirms that wheel and source-distribution metadata agree with
`pyproject.toml`, required package files are present, and repository guidance or
credential-like content is absent.

## Preparing and validating a release

Release preparation starts from a clean committed checkout. Run the normal
offline validation first, then have the release owner create an annotated tag
at that exact commit. Build into an empty directory from the tagged checkout
and validate the candidate artifacts:

```sh
uv sync --locked --extra codex --extra ollama --extra openai
uv lock --check
uv run python tools/generate_buck_python_deps.py --check
uv run python tools/validate_agent_context.py
uv run python tools/typecheck.py
uv run python -m unittest discover -s tests/unit -t .
./buck2 test //... --exclude live --always-exclude

release_version="0.2.2"
release_commit="$(git rev-parse HEAD)"
git tag -a "v$release_version" -m "AgentRig $release_version"
release_dir="$(mktemp -d)"
uv build --out-dir "$release_dir"
uv run python tools/validate_release.py \
  --dist-dir "$release_dir" \
  --tag "v$release_version" \
  --commit "$release_commit" \
  --write
```

The validator checks that the checkout is clean and that the annotated tag and
`HEAD` both resolve to the recorded commit. The release owner may also verify
that relationship directly:

```sh
test "$(git rev-parse "v$release_version^{commit}")" = "$release_commit"
```

Tag creation is shown to define the required sequence; it remains a release-owner
Git operation and is never performed by the validator. The build directory must
start empty. Do not mix artifacts from separate builds or rebuild a version
after distribution. Preserve the validated artifacts and manifest as one
release set.

## Distribution boundary

This repository defines and verifies the release identity, but it does not yet
select a public package registry, release hosting channel, artifact-signing
scheme, or automated publishing credentials. Those choices require a separate
decision and must not weaken this contract.

Until a validated tag and artifact set are actually distributed, downstream
path dependencies remain development integrations rather than stable release
dependencies. A downstream consumer should pin either a published immutable
artifact or an immutable tagged source revision and retain enough metadata to
audit what was installed.
