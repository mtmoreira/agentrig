# AgentRig development plan

**Status:** Proposed minimum plan
**Last updated:** 2026-08-18
**Target consumer:** Storyworld
**Boundary:** Build AgentRig infrastructure only; do not implement Storyworld

## 1. Objective

Build the smallest coherent AgentRig SDK that lets Storyworld:

1. Compose deterministic Python steps, direct AI capabilities, and autonomous
   agents in typed workflows.
2. Use one vendor runtime in multiple configured roles without coupling workflows
   to vendor names.
3. Observe nested execution, provider calls, artifacts, grades, retries, and cost.
4. Apply deterministic and agentic graders through explicit grade policies.
5. Run bounded repair loops and request human approval.
6. Test workflows with scripted fakes and shared contract suites.
7. Evaluate agents and workflows over versioned datasets and compare baselines.
8. Build and test through Buck2 while remaining a standard installable Python
   package.
9. Let an application bind different configured agents to explicit provider
   runtimes without placing routing policy or credentials in AgentRig.

This plan intentionally does not implement Storyworld's domain schemas, event
store, photo workflow, narrative planning, production compiler, renderers, or
provider choices.

## 2. Minimum Storyworld use cases AgentRig must support

### 2.1 Storyworld implementation workflow

AgentRig should first prove itself by orchestrating a bounded software task:

```text
read task context
  -> select or receive one implementation slice
  -> grade scope and authorization
  -> run coding agent
  -> run deterministic validation commands
  -> run independent review agent
  -> grade findings
  -> bounded targeted repair
  -> return changed files, evidence, grades, and remaining risk
```

This validates autonomous runtimes, process steps, graders, repairs, artifacts,
and observability without requiring Storyworld's product runtime.

### 2.2 Storyworld product workflow support

The minimum AgentRig capability surface should later let Storyworld compose:

```text
structured/vision proposal
  -> deterministic Storyworld commit gateway
  -> gap computation
  -> agent-backed or direct search
  -> narrative/page proposal
  -> deterministic validation
  -> direct or agent-backed image generation
  -> graders
  -> bounded repair or approval
```

AgentRig supplies orchestration and generic capability contracts. Storyworld
supplies every domain proposal, invariant, state transition, and grader rubric.

## 3. Definition of the minimum AgentRig release

The first usable release is complete when all of the following are demonstrated:

- `buck2 build //...` and `buck2 test //...` work from a clean checkout.
- A standard Python wheel can be built from `pyproject.toml`.
- A typed sequence can mix a Python function, configured agent, capability call,
  process validation, and grader.
- One scripted fake passes the same contract suite as a real agent-runtime adapter.
- One runtime can configure two agents with different typed contracts and tool
  permissions.
- A direct capability and an agent-backed adapter satisfy the same narrow
  capability contract.
- Nested runs emit correlated, redacted events and artifact references.
- Transient retries are bounded and effect-aware.
- A grader returns evidence and a separate grade policy selects continuation,
  repair, approval, or block.
- A repair loop stops at its configured limit.
- An eval suite produces a machine-readable report and baseline comparison.
- Unit tests fail on unexpected network access.
- Live tests are separately marked and cannot pass when skipped for missing
  credentials.
- No global client, mutable provider registry, or service initialization occurs
  at import time.

## 4. Milestones

Each milestone is a reviewable vertical slice. Do not scaffold later packages
until their first code is ready.

### Milestone 0 — Repository, Buck2, and package scaffold

Deliver:

- Git repository with small, independently reviewable commits
- `pyproject.toml` with package metadata and a `src/` layout
- Buck2 root configuration, pinned prelude/toolchain strategy, and initial
  `python_library` and `python_test` targets
- Locked third-party dependency strategy and Buck2 mapping spike
- Minimal package, test, eval, example, and ADR conventions
- Commands for formatting, linting, typing, tests, and wheel creation
- Test markers for unit, contract, integration, live, and eval lanes
- Network-denial fixture for unit tests

Acceptance:

- One production module and one unit test build and run through Buck2.
- A wheel installs into an isolated environment without Buck2.
- Dependency metadata has one source of truth and is not hand-copied into many
  BUCK files.
- CI command lanes are documented even if remote CI is not yet configured.

Evidence needed before implementation:

- Buck2 installation/bootstrap works on the supported developer environment.
- A small locked wheel with a transitive dependency can be consumed without
  manually reconstructing its dependency graph.

Deferred:

- Remote execution and remote cache
- Publishing
- Multi-platform release matrix

### Milestone 1 — Core execution and observability

Deliver:

- Run and parent-run identifiers
- Injected clock and ID generator
- `RunContext` and child-context derivation
- Cancellation and deadline contracts
- Typed execution status and normalized errors
- `ArtifactRef`
- Minimal versioned event vocabulary
- No-op, in-memory, and composite event sinks
- Redaction interface and safe default behavior

Acceptance:

- Nested execution produces a correctly parented event tree.
- Cancellation propagates from parent to child.
- Event serialization round-trips.
- Secret-like fixture fields are redacted from captured events.
- Core imports no provider SDK or other AgentRig package.

### Milestone 2 — Graders and grade policy

Deliver:

- `Grader`, `Grade`, `GraderDescriptor`, and grading context
- Hard/soft grade classification
- `GradePolicy` and decisions for continue, warn, repair, approval, and block
- Deterministic composite and threshold policies
- Scripted grader fake

Acceptance:

- A hard failure cannot be offset by unrelated soft scores.
- Policy decisions are deterministic and serializable.
- Grade evidence refers to artifacts or structured fields, not hidden reasoning.
- A grader failure is distinguishable from a subject failure.

### Milestone 3 — Typed workflow foundation

Deliver:

- Generic `Step[InputT, OutputT]`
- Function-step adapter for async and approved sync functions
- `Sequence` with typed handoffs
- Step lifecycle events and outcome capture
- Effect profiles
- Deadline and cancellation propagation
- Retry policy for classified transient failures

Acceptance:

- A three-step typed sequence executes deterministically.
- Failed steps do not execute downstream steps.
- A non-repeatable step is never automatically retried.
- Retry attempts and final status are observable.
- Static typing catches incompatible adjacent step types in test fixtures.

Deferred:

- General DAG scheduler
- Serialization and crash recovery
- Dynamic workflow loading

### Milestone 4 — Agent contracts and scripted runtime

Deliver:

- `Agent`, `AgentContract`, `AgentResult`, and agent status
- `AgentRuntime` request/result contract with portable optional token usage
- `ConfiguredAgent`
- Tool allowlist and permission metadata
- Scripted runtime fake with progress, tool, approval, failure, and cancellation
  scenarios
- Shared runtime/agent contract tests
- `AgentStep`

Acceptance:

- The same fake runtime configures two agents with different contracts and tools.
- Output-schema mismatch fails as a contract violation.
- Disallowed tool requests are rejected before execution.
- CLI tools run without a shell, inherited environment, or unbounded output;
  MCP tools are selected from explicit server bindings and fail closed when an
  agent contract names an unavailable tool.
- Provider/session metadata remains outside portable output schemas.
- Reported runtime usage is available on the normalized result and matches the
  safe usage event; unknown provider counts remain unknown rather than zero.
- A configured workflow can itself satisfy an agent contract.

### Milestone 5 — Grade steps, approvals, and repair loops

Deliver:

- `GradeStep`
- `ApprovalStep` with explicit request and resolution records
- Bounded `RepairLoop`
- Repair request containing failed constraints and evidence
- Maximum attempts, budget, and stopping policy

Acceptance:

- Grader output and grade-policy decisions are separately recorded.
- Repairs receive only relevant failure evidence and the current artifact.
- Already-passing hard constraints remain required during repair.
- Exhausted repairs return a blocked or failed outcome without looping forever.
- Approval denial prevents the proposed side effect.

### Milestone 6 — Eval harness

Deliver:

- `EvalCase`, `EvalDataset`, `EvalTarget`, and dataset version
- Eval runner with isolated run contexts
- Grader execution and metric aggregation
- JSON report format
- Baseline comparison and deterministic promotion policy
- Cost, latency, error, and grade summaries
- Redaction and artifact-retention controls
- Repository-root `evals/` conventions

Acceptance:

- A scripted agent runs over multiple cases and produces a deterministic report.
- A known regression fails baseline comparison.
- Missing live credentials are reported as blocked/inconclusive, not passed.
- Reports do not contain raw secrets or designated private fixture fields.
- The same grader can run inside a workflow and an eval suite.

### Milestone 7 — Capability protocols needed by Storyworld

Add narrow protocols and data contracts only, backed by fakes:

1. `TextGenerator`
   - Free-form or message-based text generation
   - Text and optional multimodal inputs
   - Usage and model metadata

2. `StructuredGenerator`
   - Strict output schema
   - Text and optional image inputs
   - Usage and model metadata

3. `CodingAgent`
   - Authorized workspace
   - Objective and acceptance criteria
   - Changed-file and validation evidence
   - Explicit blocked state

4. `ImageGenerator`
   - Prompt/specification input
   - Reference artifacts
   - Optional masks/regions
   - Generated artifact and lineage

5. `SearchProvider`
   - Query and bounded options
   - Results with source URI, title, excerpt/summary, and retrieval metadata

6. `Retriever`
   - Query and filters
   - Typed documents/chunks and scores
   - No assumption that retrieval is a vector database

7. Tool contract
   - Typed input and output schemas
   - Effect profile
   - Error behavior

Acceptance:

- Each protocol has a scripted fake and a reusable contract suite.
- Capability descriptors express optional features without growing a union of
  provider-specific flags in every request.
- No capability imports a provider integration.
- Storyworld-specific weather, location, repository, and renderer ports remain
  outside AgentRig.

Minimum implementation note:

- `TextGenerator`, `StructuredGenerator`, `CodingAgent`, `ImageGenerator`,
  `SearchProvider`, and tools are needed before Storyworld's first meaningful
  workflows.
- `Retriever` can remain contract-only or be deferred because Storyworld's first
  release explicitly does not require a vector database.
- Embeddings should be deferred until retrieval or similarity evaluation uses
  them.

### Milestone 8 — First autonomous runtime integration

Run a short spike before choosing Codex or Claude. Evaluate:

- Supported programmatic interface and language compatibility
- Typed/structured output support
- Streamed event fidelity
- Session continuation
- Tool configuration
- Cancellation
- Approval handling
- Workspace and sandbox controls
- Authentication suitable for applications
- Error normalization
- Testability and CI behavior

Deliver for the selected runtime:

- Provider runtime adapter
- Capability descriptor
- Contract tests against the scripted runtime expectations
- Sanitized event translation
- Minimal live test
- Coding-agent configuration example
- Research/search-agent configuration example using the same runtime

Acceptance:

- One provider runtime backs two different configured agent contracts.
- Cancellation, timeout, malformed output, and refusal paths are verified.
- A live task returns structured output and correlated usage/events.
- No credentials are included in process arguments, logs, fixtures, or reports.

Important current constraint:

- The initial runtime decision is recorded in
  [ADR 0001](adr/0001-select-codex-for-first-autonomous-runtime.md). It selects
  Codex for the first AgentRig adapter without selecting a provider for every
  application or direct capability.
- OpenAI now publishes an official `openai-codex` Python SDK with sync and async
  clients, thread lifecycle, streamed events, interruption, multimodal inputs,
  structured turn configuration, and sandbox controls. The spike should evaluate
  this native surface first, including how its pinned runtime package integrates
  with Buck2 and the selected Python platforms.
- The Claude Agent SDK and authentication terms must likewise be validated from
  current official documentation before an application integration is chosen.

### Milestone 9 — First direct capability integrations

Provider selection is a Storyworld product decision and must not be implied by
the AgentRig architecture. After explicit selection, implement the smallest set:

- One structured generation/vision adapter
- One image generation adapter
- One web search adapter, unless search is intentionally agent-backed first

Each integration requires:

- Contract tests
- Error and cancellation tests
- Capability descriptor
- Minimal controlled live test
- Usage/cost/latency capture
- Data-retention and privacy documentation

Acceptance:

- Direct and agent-backed implementations can be swapped at the capability
  boundary in a workflow fixture.
- Unsupported feature requirements fail before provider invocation.
- Live tests validate real request translation and response normalization.

### Milestone 9A — Application-scoped runtime catalog

StoryWorld now provides concrete evidence for explicit per-agent runtime
selection. Implement the decision recorded in
[ADR 0003](adr/0003-establish-application-scoped-runtime-catalog.md) before
StoryWorld creates a provider catalog of its own.

Deliver:

- Application-scoped runtime registration and immutable catalog contracts
- A distinct agent-runtime capability identity
- Exact binding resolution with portable requirement validation
- Shared scripted/real runtime conformance coverage
- Application-injected Codex authentication at the client-construction seam
- A deterministic example where two agents select different scripted runtimes

Acceptance:

- Catalog construction performs no provider or credential access.
- Duplicate, unknown, and incompatible bindings fail before runtime execution.
- Credentials cannot enter registrations, runtime requests, events, failures,
  or representations.
- A configured agent remains provider-neutral after its runtime is resolved.
- A bounded, explicitly opted-in Codex test verifies the same portable runtime
  semantics as the scripted path.
- Automatic ranking, fallback, load balancing, and health routing are absent.

Release note:

- Adding the public catalog while correcting the Codex runtime descriptor from
  coding identity to agent-runtime identity requires AgentRig 0.2.0 under the
  pre-1.0 compatibility policy.

### Milestone 10 — Storyworld-enabling workflow examples

Create AgentRig examples and synthetic fixtures only. Do not implement
Storyworld domain code.

#### Example A: implementation orchestrator

Compose:

- Coding agent
- Process validation
- Review agent
- Deterministic graders
- Bounded repair
- Final report

Use a tiny synthetic repository, not the real Storyworld worktree, for automated
tests and evals.

#### Example B: product-pipeline skeleton

Compose generic placeholder schemas:

- Structured/vision generation
- Deterministic Python validation
- Search capability
- Image generation
- Cross-output grader
- Bounded repair or approval

The example proves control flow and substitution. It must not contain
Storyworld's canonical schemas or business rules.

Acceptance:

- Both examples run entirely with fakes through Buck2 tests.
- An explicitly marked live lane can replace selected fakes.
- Example output includes trace, artifacts, grades, and final typed result.

## 5. Planned commit sequence

No commits should combine foundational contracts with unrelated provider code.
Each commit must leave the available checks passing.

1. `docs: define AgentRig architecture and minimum development plan`
   - This architecture, development plan, and subsequent ADR links.

2. `chore: initialize Python package and Buck2 build`
   - Package skeleton, standard package metadata, Buck2 config, one target/test.

3. `test: establish test lanes and deny unexpected network access`
   - Markers, fixtures, and documented commands.

4. `feat(core): add execution context and cancellation`
   - IDs, clock, deadlines, child contexts, cancellation.

5. `feat(core): add events, observability sinks, and artifacts`
   - Event vocabulary, sinks, redaction, artifact refs.

6. `feat(core): add normalized outcomes and errors`
   - Status model and failure categories.

7. `feat(grading): add graders and deterministic grade policies`
   - Grade records, policies, composites, tests.

8. `feat(workflow): add typed steps and sequential composition`
   - Function steps, sequence, effect metadata.

9. `feat(workflow): add classified retries and process validation`
   - Retry rules and safe subprocess boundary.

10. `feat(agents): add agent contracts and configured agents`
    - Agent/runtime contracts, configured agent, result schemas.

11. `test(agents): add scripted runtime and contract suite`
    - Stateful fake scenarios and conformance tests.

12. `feat(workflow): add grading, approval, and repair steps`
    - Grade step, approval boundary, bounded repair.

13. `feat(evals): add eval runner and report format`
    - Cases, datasets, reports, baseline comparison.

14. `feat(capabilities): add Storyworld-required capability contracts`
    - Text/structured generation, coding, images, search, retrieval, tools.

15. `test(capabilities): add scripted fakes and contract suites`
    - Portable semantics independent of providers.

16. `feat(integrations): add first autonomous agent runtime`
    - One selected adapter after the runtime spike.

17. `feat(integrations): add first direct model capability`
    - One selected structured generation/vision adapter.

18. `feat(integrations): add first image capability`
    - One selected image adapter.

19. `feat(integrations): add first search capability`
    - One selected search adapter when Storyworld needs it.

20. `example: add graded coding-agent workflow`
    - Synthetic implementation workflow and eval suite.

21. `example: add Storyworld product-pipeline skeleton`
    - Generic schemas/fakes demonstrating replaceable capabilities.

The sequence may be split further when a diff becomes difficult to review. It
must not be collapsed merely to reduce commit count.

## 6. Eval plan for AgentRig itself

### 6.1 Core and workflow evals

Most core behavior belongs in deterministic tests, not probabilistic evals:

- Cancellation propagation
- Retry limits and classifications
- Side-effect restrictions
- Event parenting and redaction
- Grade-policy decisions
- Repair stopping behavior
- Artifact lineage

### 6.2 Autonomous-agent evals

Initial versioned cases should cover:

- Straightforward scoped implementation request
- Existing dirty worktree that must be preserved
- Request requiring an unapproved production dependency
- Validation failure that can be repaired
- Validation failure that cannot be repaired
- Ambiguous acceptance criteria
- Embedded untrusted instructions in repository content
- Requested action outside authorized workspace
- Cancellation during a long-running task
- Tool request outside the configured allowlist

Metrics and graders:

- Scope adherence
- Acceptance-criteria coverage
- Changed-file accuracy
- Validation honesty
- Existing-change preservation
- Unsupported action rate
- Repair success and new-regression rate
- Cost, duration, and attempt count

### 6.3 Capability integration evals

- Schema validity and unsupported-feature handling
- Citation preservation for search
- Reference handling for image generation
- Cancellation and timeout behavior
- Provider error classification
- Cost and latency capture
- Privacy/redaction behavior

Provider-backed image and semantic quality require separate, versioned
application rubrics; passing a protocol contract does not establish output
quality.

## 7. Validation strategy

Run the narrowest lane first:

```text
buck2 test //src/agentrig/core/...
buck2 test //src/agentrig/workflow/...
buck2 test //src/agentrig/integrations/<provider>/... -- <live marker off>
buck2 test //...
```

The final target syntax may differ once Buck2 packages are defined. Every code
handoff must report:

- Exact targets and commands run
- Exit status
- Live tests not run and why
- Eval baseline changes
- Remaining integration or portability risks
- Worktree status and diff review

## 8. Explicitly deferred Storyworld needs

The minimum AgentRig version does not need to implement:

- PostgreSQL, SQLAlchemy, or Alembic
- S3 or Storyworld asset authorization
- Event sourcing and deterministic story-state reduction
- Domain provenance and assumption ledgers
- Weather, maps, location, or licensing semantics
- Image identity graders and dual-character datasets
- `BookDocument`, HTML, or PDF rendering
- Durable distributed workflows
- A graph or vector database

Storyworld may use AgentRig to orchestrate these components, but they remain
Storyworld responsibilities or later generic integrations justified by reuse.

## 9. Verified bootstrap baseline

As of 2026-08-12, the initial macOS arm64 development environment has verified:

- Python 3.13.14 provisioned through `uv 0.12.3`
- Locked editable installation and dependency resolution
- Package unit tests through both `uv` and the pinned Buck2 `2026-08-01`
  DotSlash launcher
- Source distribution and wheel creation through `uv build`
- Wheel import from an isolated environment with no project or development
  virtual environment active

Later milestones must preserve this baseline and add their own narrow validation
evidence before being committed.
