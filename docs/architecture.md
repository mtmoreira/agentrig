# AgentRig foundational architecture

**Status:** Proposed foundation
**Last updated:** 2026-08-17
**Scope:** AgentRig SDK, not Storyworld domain implementation

**Visual companion:** [Architecture field guide](architecture.html)

## 1. Purpose

AgentRig is a typed Python SDK for composing heterogeneous AI capabilities into
observable, testable workflows and agents. A workflow may combine autonomous
agents such as Codex or Claude, direct model calls such as Gemini, image
generation, web search, retrieval, deterministic Python functions, and tools.

AgentRig exists to reuse execution infrastructure across agents. It does not try
to make every AI system look semantically identical, nor does it own the domain
model of applications such as Storyworld.

The central design statement is:

> Workflows depend on typed capabilities. Integrations bind providers and agent
> runtimes to those capabilities. Core supplies execution, observation, grading,
> and lifecycle semantics.

## 2. Scope and non-goals

AgentRig should provide:

- Typed inputs and outputs at orchestration boundaries.
- Python-first workflow composition.
- Adapters for direct services and autonomous agent runtimes.
- Shared execution context, cancellation, timeouts, artifacts, and errors.
- Structured observability without mandatory telemetry vendors.
- Graders and policies for validation, repair, blocking, and approval.
- Eval infrastructure for agents, workflows, capabilities, and integrations.
- Fakes and contract suites for deterministic development.
- Explicit handling of side effects, replay safety, and provider capabilities.

AgentRig should not initially provide:

- A universal lowest-common-denominator API for all AI systems.
- Storyworld entities, events, repositories, renderers, or policies.
- A declarative YAML workflow language.
- A global provider or dependency registry.
- Distributed execution or a durable workflow service.
- Automatic provider selection without explicit requirements and policy.
- A general-purpose vector database abstraction before a consumer requires it.
- A plugin marketplace or dynamic import system.

## 3. Design principles

### 3.1 Capability, not vendor, is the dependency boundary

`Codex`, `Claude`, `Gemini`, and `OpenAI` identify products, providers, or
runtimes. They do not identify what a caller needs.

A caller should request a narrow capability such as:

- Structured generation
- Coding task execution
- Image generation
- Web search
- Retrieval
- Embedding
- Tool execution

The same provider runtime may satisfy several capabilities. Several providers
may satisfy the same capability. Neither fact requires the workflow to know the
vendor name.

### 3.2 Preserve semantic differences

An LLM call, a vector search, an image generation, and an autonomous coding task
have different requests, results, failure modes, costs, and side effects. They
must not be forced behind an untyped `run(Any) -> Any` API.

AgentRig standardizes their execution envelope and composition, not their domain
semantics.

### 3.3 Composition is the default

Provider runtime, instructions, tools, schemas, policies, graders, and workflow
structure should be assembled as objects. Subclassing is reserved for cases
where an implementation truly needs to replace or specialize lifecycle logic.

### 3.4 Normalize the boundary, preserve the escape hatch

Capability protocols define the portable boundary a workflow can rely upon.
Integrations may expose explicitly named provider options for features that do
not belong in the portable contract. Provider-specific objects must not leak
into core result types.

### 3.5 Observation is part of execution

Every run should have a stable identity and parent relationship. Steps, agents,
provider calls, tool calls, artifacts, grades, retries, and usage records should
be correlated through core events.

Core defines observability contracts. Concrete exporters remain integrations.

### 3.6 Grading and enforcement are separate

A `Grader` measures or validates and returns evidence. A `GradePolicy` decides
whether grades permit continuation, require repair, require approval, or block
the workflow.

This separation prevents a numeric score or an LLM judge from silently becoming
an authorization decision.

### 3.7 Explicit side-effect semantics

Retries and replay are safe only when the operation's effects are understood.
Every step should declare whether it is read-only, idempotent, compensatable, or
non-repeatable. AgentRig must never blindly retry a coding agent after it may
have partially changed a workspace.

### 3.8 Build abstractions from demonstrated reuse

An integration may directly implement one capability. A shared client, runtime,
or transport layer should be extracted only when multiple implementations use
it. AgentRig does not require a wrapper around every vendor SDK.

### 3.9 Image roles and routing are explicit

An image edit distinguishes its edit base, edit mask, identity references,
style references, and composition references in the portable request. Output
lineage binds every input artifact ID. The bounded image executor invokes only
the route ID selected by the caller and may retry only declared failure kinds
on that same route; it does not rank providers or fall through to another one.

Image usage fields remain `None` when an implementation cannot report them.
Cost ceilings therefore require a route that advertises cost reporting and a
result that supplies matching cost and currency. Provider SDK imports remain in
optional integration bridge modules, never the base capability or workflow.

## 4. Vocabulary

### Capability

A narrow, typed behavioral contract such as `ImageGenerator` or `Retriever`.
Capabilities describe what a caller needs, not which provider supplies it.

### Integration

Provider-specific code that translates between an external SDK, CLI, protocol,
or service and an AgentRig capability or runtime contract.

### Agent runtime

An execution substrate that can run configured autonomous agents. It manages
vendor sessions, requests, streamed events, cancellation, approvals, and result
translation. Examples may include Codex and the Claude Agent SDK.

An agent runtime is not itself a coding agent, search agent, or image agent.

### Agent

A configured, goal-directed executable with a typed input and output contract.
It may reason, use tools, maintain a provider session, and decide intermediate
actions within declared limits.

An agent is assembled from a runtime plus configuration such as instructions,
tools, output schema, permissions, and policy.

### Tool

A callable capability made available to an agent for intermediate use. A tool
has a typed schema, a bounded purpose, and declared side effects. Tools are not
implicitly available to every agent.

Local executables use a fixed absolute executable, a fixed working directory,
no shell, an explicit environment allowlist, and bounded time and output. MCP
servers are immutable bindings with an explicit transport and tool allowlist.
Agent contracts refer to MCP tools through stable `mcp.<server>.<tool>` IDs;
runtime adapters must reject unknown IDs before provider initialization and
translate provider events back to those same IDs.

### Step

A typed node in a workflow. A step may call an agent, invoke a capability, run a
Python function, execute a subprocess, ask for approval, or apply a grader.

### Workflow

A Python-composed graph of steps with explicit control flow. A workflow can be
exposed as an `Agent` when it has a stable contract, allowing recursive
composition.

### Artifact

A durable or addressable output such as a file, generated image, report, patch,
or serialized record. An artifact reference carries identity, media type, hash
when known, lineage, and privacy metadata; it does not require core to own the
underlying storage implementation.

### Grader

A deterministic or agentic evaluator that returns one or more structured
grades, including evidence and grader version.

### Grade policy

Deterministic policy that reduces grades into a workflow decision: continue,
warn, repair, request approval, or block.

### Eval

A repeatable experiment that executes an agent, workflow, or capability over a
versioned dataset, runs graders, aggregates results, and compares them with a
baseline or promotion policy.

## 5. Package boundaries

The intended package structure is:

```text
src/agentrig/
  core/
    context.py
    events.py
    observability.py
    artifacts.py
    errors.py
    cancellation.py
    execution.py
    grading.py
    policy.py

  capabilities/
    text_generation.py
    structured_generation.py
    coding.py
    images.py
    search.py
    retrieval.py
    embeddings.py
    tools.py

  agents/
    base.py
    contract.py
    runtime.py
    configured.py
    adapters.py

  workflow/
    step.py
    sequence.py
    parallel.py
    branch.py
    loop.py
    retry.py
    approval.py

  integrations/
    openai/
    anthropic/
    google/
    vectorstores/
    process/
    observability/

  graders/
    deterministic.py
    agentic.py
    process.py

  evals/
    case.py
    dataset.py
    runner.py
    aggregation.py
    report.py
    baseline.py

  testing/
    fakes.py
    scripted.py
    contracts.py

evals/
  datasets/
  suites/
  graders/
  baselines/
  reports/

examples/
  storyworld_implementation/
  storyworld_product_pipeline/
```

Directories should be created only when their first implementation is added.
This tree describes ownership, not a requirement to scaffold empty packages.

### 5.1 Dependency direction

Using `consumer -> dependency` notation:

```text
capabilities -> core
agents       -> capabilities, core
graders      -> core
workflow     -> agents, capabilities, graders, core
integrations -> agents and/or capabilities, core
evals        -> public AgentRig APIs
```

More precisely:

- `core` imports no other AgentRig package and no provider SDK.
- `capabilities` imports only `core` and Python standard-library types.
- `agents` imports `core` and capability/tool contracts, never integrations.
- `workflow` imports public core, capability, agent, and grader contracts.
- `integrations` implement contracts and may use external dependencies.
- `evals` may execute public AgentRig APIs but production packages never import
  `evals`.
- Composition roots in applications import concrete integrations and inject
  them into workflows.

Import-boundary checks should enforce these rules.

## 6. Core execution model

### 6.1 Run context

Every execution receives an explicit `RunContext`. It should contain or provide:

- Run ID and optional parent run ID
- Clock and ID generator
- Cancellation token
- Deadline and budget view
- Event sink
- Artifact recorder
- Redaction policy
- Approval channel when enabled
- Immutable labels and correlation metadata

It must not contain provider credentials, mutable global registries, or hidden
network clients.

Child steps derive child contexts so the execution tree is observable without
sharing unrelated mutable state.

### 6.2 Events and observability

The initial event vocabulary should remain small:

- Run started, completed, failed, blocked, and cancelled
- Step started and completed
- Progress reported
- Provider call started and completed
- Tool call started and completed
- Artifact produced
- Grade produced
- Retry scheduled
- Approval requested and resolved
- Usage reported

Events should be versioned, timestamped, correlated, and redacted. Event payloads
should store references or summaries by default rather than private raw prompts,
uploaded media, credentials, or entire provider responses.

Core supplies no-op, in-memory, and composite sinks. Console, JSONL,
OpenTelemetry, and vendor exporters live under integrations.

### 6.3 Errors and outcomes

AgentRig should distinguish:

- Invalid input or contract violation
- Transient provider failure
- Permanent provider failure
- Policy refusal
- Approval required or denied
- Budget exhaustion
- Cancellation or deadline expiration
- Grader failure
- Workflow blocked state
- Unexpected implementation failure

Provider adapters normalize external failures into these categories while
preserving sanitized provider codes in metadata.

Workflow boundaries return an explicit execution outcome containing status,
typed output when available, artifacts, grades, and sanitized failure details.
Ergonomic helpers may unwrap successful outcomes or raise typed exceptions, but
the full outcome remains available to orchestrators and evals.

### 6.4 Artifacts

Core should define an `ArtifactRef`, not a storage service implementation. The
reference should support:

- Stable artifact ID
- Kind and media type
- URI or workspace-relative path
- Content hash when available
- Producer run and parent inputs
- Privacy or retention labels
- Provider lineage

Storyworld can bind these references to its own asset store and richer domain
lineage without AgentRig owning `BookDocument`, character assets, or project
authorization.

### 6.5 Side-effect classification

Every workflow step declares an effect profile:

```text
read_only       no externally visible writes
idempotent      repeated execution with the same key has the same effect
compensatable   writes can be reversed by a defined compensation
non_repeatable  may create irreversible or ambiguous effects
```

The profile informs retry, parallelism, checkpointing, and human-approval
policy. It is an assertion that contract tests should verify where practical;
it is not magic enforcement.

## 7. Capability contracts

Capability contracts should use domain-shaped methods and typed request/result
objects. Illustrative protocols follow; exact field definitions belong to their
implementation milestones.

```python
class TextGenerator(Protocol):
    async def generate(
        self,
        request: TextGenerationRequest,
        context: RunContext,
    ) -> TextGenerationResult: ...


class StructuredGenerator(Protocol):
    async def generate(
        self,
        request: StructuredGenerationRequest[OutputT],
        context: RunContext,
    ) -> StructuredGenerationResult[OutputT]: ...


class CodingAgent(Protocol):
    async def execute(
        self,
        task: CodingTask,
        context: RunContext,
    ) -> CodingResult: ...


class ImageGenerator(Protocol):
    async def generate(
        self,
        request: ImageGenerationRequest,
        context: RunContext,
    ) -> ImageGenerationResult: ...


class SearchProvider(Protocol):
    async def search(
        self,
        request: SearchRequest,
        context: RunContext,
    ) -> SearchResult: ...


class Retriever(Protocol):
    async def retrieve(
        self,
        request: RetrievalRequest,
        context: RunContext,
    ) -> RetrievalResult: ...
```

These contracts intentionally differ. A coding result may contain a worktree
change and validations; an image result contains image artifacts and generation
metadata; a search result contains citations and retrieval metadata.

### 7.1 Capability descriptors

Portable APIs must not imply that all implementations support every feature.
Each implementation exposes a descriptor covering relevant facts such as:

- Streaming and cancellation
- Structured output
- Session continuation
- Approval requests
- Tool support
- Reference-image count
- Mask or region support
- Citation support
- Idempotency-key support
- Data-retention characteristics
- Known limits

Callers express `CapabilityRequirements`; composition validates them before an
expensive run starts. Applications inject implementations explicitly. When an
application needs several selectable runtimes, it may construct an immutable,
application-scoped runtime catalog and resolve an exact binding through those
requirements. Automatic policy-based selection and fallback remain deferred.

## 8. Agents and runtimes

### 8.1 Agent contract

An agent has a stable, versioned contract:

```python
class Agent(Protocol[InputT, OutputT]):
    @property
    def contract(self) -> AgentContract[InputT, OutputT]: ...

    async def run(
        self,
        input: InputT,
        context: RunContext,
    ) -> AgentResult[OutputT]: ...
```

`AgentContract` should include:

- Agent ID and version
- Purpose
- Input and output schema identities
- Allowed tools and capabilities
- Permission and side-effect profile
- Prompt/configuration version
- Declared limits and stopping policy

### 8.2 Agent runtime contract

An `AgentRuntime` handles provider-specific autonomous execution:

```python
class AgentRuntime(Protocol):
    async def execute(
        self,
        request: AgentExecutionRequest,
        context: RunContext,
    ) -> AgentExecutionResult: ...
```

The request contains provider-neutral configuration and an explicit
provider-options field owned by the integration. The runtime translates
provider events into AgentRig core events and provider output into a normalized
execution result. That result contains an `AgentRuntimeUsage` value with
optional input, cached-input, and output token counts. Unknown counts remain
`None`; applications must not infer zero usage from missing provider data.

A runtime that advertises `usage_reporting` returns the same normalized usage
value it projects into its safe usage event. Applications can therefore enforce
portable limits from the result without depending on provider event schemas or
retaining raw provider responses.

### 8.3 Configured agents

A configured agent composes:

```text
agent runtime
  + typed agent contract
  + instructions
  + allowed tools
  + output codec/schema
  + permissions
  + execution budgets
  + provider options
```

For example, one Codex runtime may back multiple independent agents:

```python
runtime = CodexRuntime(...)

implementer = ConfiguredAgent[CodingTask, CodingResult](
    runtime=runtime,
    contract=coding_contract,
    instructions=implementation_instructions,
    tools=coding_tools,
)

researcher = ConfiguredAgent[ResearchTask, ResearchReport](
    runtime=runtime,
    contract=research_contract,
    instructions=research_instructions,
    tools=search_tools,
)
```

These are distinct configured agents even though they share a vendor runtime.
They can have different permissions, schemas, graders, and eval baselines.

### 8.4 Application-scoped runtime catalog

An application that uses several runtime bindings may construct an explicit
catalog at its composition root:

```text
agent route
  -> exact binding ID
  -> capability requirement check
  -> AgentRuntime
```

The catalog standardizes registration identity, duplicate rejection, and
capability validation. It does not choose a vendor, read application
configuration, resolve credentials, or initialize provider services. It is an
injected value, never a mutable global registry or import-time service locator.

Provider integrations may accept a late-bound authentication source for client
construction. The application owns credential references and secret resolution;
AgentRig must not put resolved credentials in registrations, portable requests,
events, failures, or representations. See
[ADR 0003](adr/0003-establish-application-scoped-runtime-catalog.md).

### 8.5 Agent-backed capabilities

Sometimes a workflow needs an `ImageGenerator`, but the implementation is an
agent that invokes an image tool and verifies the output. An adapter should bind
that agent to the narrow capability:

```python
image_generator: ImageGenerator = AgentBackedImageGenerator(
    agent=visual_agent,
)
```

The workflow can replace it with a provider-native integration:

```python
image_generator = OpenAIImageGenerator(...)
```

Both satisfy the image-generation contract, but their descriptors, traces,
costs, latency, and failure modes remain visible.

This pattern answers the multi-role provider problem:

- Do not create one giant `Codex` object implementing every possible method.
- Do not classify Codex permanently as a coding platform.
- Configure separate agents for separate contracts.
- Adapt those agents to narrow capabilities where substitution is useful.
- Keep direct provider integrations available when autonomy adds no value.

### 8.6 Workflow as agent

A workflow may expose an `AgentContract` and the `Agent.run` method. That makes a
multi-step process substitutable anywhere an agent with the same input/output
contract is accepted.

Internally, it remains a workflow with explicit deterministic control flow. The
outer caller does not need to know whether one model call, one autonomous agent,
or twenty composed steps produced the result.

## 9. Workflow composition

### 9.1 Python is the workflow language

The initial SDK uses ordinary Python construction:

```python
storyworld_implementation = Sequence(
    SelectTaskStep(planner),
    GradeStep(scope_graders, scope_policy),
    AgentStep(implementer),
    ProcessValidationStep(),
    AgentStep(reviewer),
    RepairLoop(
        repair_agent=implementer,
        graders=implementation_graders,
        policy=repair_policy,
        max_attempts=2,
    ),
)
```

Python supplies types, branching, local debugging, dependency injection, and
refactoring without requiring a second configuration language. Serialization is
deferred until durable execution is justified.

### 9.2 Initial step types

- `FunctionStep`: adapt a sync or async Python callable.
- `AgentStep`: invoke a typed agent.
- `CapabilityStep`: invoke a narrow capability.
- `Sequence`: pass typed output from one step to the next.
- `Parallel`: execute independent, side-effect-compatible branches.
- `Branch`: select a branch using deterministic policy.
- `GradeStep`: run graders and apply a grade policy.
- `RepairLoop`: bounded generation, grading, and targeted repair.
- `ApprovalStep`: pause for an explicit external decision.
- `ProcessStep`: run an approved local command with captured output.

Only `FunctionStep`, `AgentStep`, `Sequence`, `GradeStep`, and a bounded
`RepairLoop` are required for the first vertical slice.

### 9.3 Concurrency

Parallel execution requires independent inputs and compatible effect profiles.
Two coding agents must not modify the same worktree concurrently. A workflow may
parallelize read-only reviews or calls over independent artifacts, then join
their typed results.

### 9.4 Retry and repair are different

- A retry repeats the same operation after a classified transient technical
  failure.
- A repair creates a new request informed by grader evidence after a valid but
  unacceptable result.

Retry policy belongs to operation execution. Repair policy belongs to workflow
logic. Neither should be hidden inside a provider adapter.

### 9.5 State and durability

The initial runtime is in-process. It records events and outcomes but does not
promise crash recovery. Durable checkpoints require serializable inputs,
versioned workflow definitions, replay-safe effects, and a persistence model.
Those should be added only after Storyworld demonstrates a concrete
failure/resume requirement.

## 10. Graders and grade policy

### 10.1 Grader contract

```python
class Grader(Protocol[SubjectT]):
    @property
    def descriptor(self) -> GraderDescriptor: ...

    async def grade(
        self,
        subject: SubjectT,
        context: GradingContext,
    ) -> Grade: ...
```

A grade should carry:

- Grader ID and version
- Metric or rule name
- Pass, warning, or failure status
- Optional score and calibrated range
- Evidence references
- Explanation
- Hard/soft classification
- Cost and latency when agentic

### 10.2 Grade policy contract

```python
class GradePolicy(Protocol):
    def decide(self, grades: Sequence[Grade]) -> GradeDecision: ...
```

`GradeDecision` is one of:

```text
continue
continue_with_warning
repair
request_approval
block
```

Hard deterministic failures cannot be averaged away by aesthetic or model-based
scores. Policies are deterministic and versioned.

### 10.3 Production grading and eval grading

The same grader may be used during a workflow and during an eval. Ownership
remains distinct:

- A workflow uses grades to decide what happens to one run.
- An eval runner aggregates grades over a dataset and compares versions.

An agent must not be its sole grader. Agentic graders should be checked against
human labels where they control promotion or consequential behavior.

## 11. Eval architecture

AgentRig provides reusable eval infrastructure under `src/agentrig/evals/` and
keeps AgentRig's own datasets and suites in the repository root `evals/`.

Minimum concepts:

- `EvalCase`: versioned input, expected constraints, allowed variability,
  prohibited behavior, metadata, and fixture references.
- `EvalDataset`: immutable selection of cases.
- `EvalTarget`: an agent, workflow, capability, or integration under test.
- `EvalRunner`: execution, trace capture, and grader invocation.
- `EvalReport`: per-case outcomes, aggregates, failures, usage, and environment.
- `Baseline`: approved report or metric thresholds for comparison.
- `PromotionPolicy`: deterministic release decision over report evidence.

Evals must support deterministic fakes and explicitly marked live runs. Missing
credentials or unavailable providers are inconclusive/blocked, never passing.
Reports must redact secrets and private input by default.

## 12. Extension strategy

### 12.1 Compose when

Prefer composition when changing:

- Provider or model
- Agent instructions or output schema
- Tool set
- Graders or grade policy
- Retry, budget, or approval policy
- Workflow order or branching
- Observability sinks
- Storage or provider routing
- One capability implementation for another

Composition should cover most application development.

### 12.2 Implement a protocol when

Implement a capability or runtime protocol when adding:

- A new provider-native image generator
- A new search backend
- A new autonomous agent runtime
- A vector-store retriever
- A new artifact or telemetry adapter
- A domain-specific capability owned by a consuming application

Protocol implementations should be usable without inheriting an AgentRig base
class.

### 12.3 Subclass when

Subclass only when closely related implementations must share invariant
lifecycle mechanics and template hooks. Possible examples include:

- Several HTTP integrations sharing one thoroughly tested request lifecycle.
- Several agent runtimes sharing session and streamed-event translation.
- A family of graders sharing parsing and evidence normalization.

Even then, prefer a small helper object when it can be injected. Do not subclass
`CodexAgent` to create `CodexImageAgent`; configure a new agent contract and
compose or adapt it to `ImageGenerator`.

### 12.4 Do not extend framework classes to define workflows

Application workflows should normally be values assembled from steps, not a
subclass per workflow. A custom step class is justified when it encapsulates
reusable control flow or an external boundary with meaningful behavior.

## 13. Integration layout and provider reuse

Provider packages may contain several independent adapters and an optional
shared runtime/client:

```text
integrations/openai/
  responses.py
  images.py
  search.py
  codex_runtime.py
  error_mapping.py

integrations/anthropic/
  messages.py
  claude_runtime.py
  error_mapping.py

integrations/google/
  structured_generation.py
  embeddings.py
  error_mapping.py
```

A shared `OpenAIClient` or `AnthropicService` layer is not required. Extract
shared transport only when it eliminates real duplication across adapters.

Every supported integration requires:

- A documented capability descriptor
- A shared contract suite where semantics are portable
- Error normalization tests
- Cancellation and timeout behavior
- Sanitized observability
- A minimal live integration test
- An eval suite for agentic behavior

## 14. Security, privacy, and authorization

- Credentials are configured outside workflow inputs and never appear in events,
  artifacts, eval fixtures, or reports.
- Tool and capability access is allowlisted per configured agent.
- External content is untrusted data and cannot redefine workflow policy.
- Raw prompts, user media, repository content, and provider responses are not
  logged by default.
- Workspaces, asset stores, and external actions remain inside caller-granted
  authorization.
- Graders measure output; they do not grant permissions.
- Human approval is explicit and scoped to a proposed action.
- AgentRig does not weaken provider or operating-system sandboxing.

## 15. Build and packaging strategy

### 15.1 Use Buck2, not legacy Buck

Buck2 can build Python libraries, binaries, and tests and provides an explicit
dependency graph, caching, and future remote-execution options. It is suitable
as AgentRig's build and test front door.

However, Buck2 is additional operational complexity for a small Python SDK. Its
prebuilt Python package support does not infer pip dependencies, so hand-writing
transitive third-party dependencies would be fragile. AgentRig therefore uses a
hybrid responsibility model:

- `buck2` is the authoritative local/CI command surface for build and test.
- `pyproject.toml` is the authoritative Python package and wheel metadata.
- A checked-in Python lockfile is the authoritative external dependency set.
- A reproducible bridge generates or maps locked wheels to Buck2 targets.
- Publishing builds standard wheels and source distributions; consumers do not
  need Buck2 to install AgentRig.
- AgentRig core should minimize third-party dependencies.

The first build milestone must prove this setup with one library target and one
test target before the repository accumulates packages.

### 15.2 Expected command surface

```text
buck2 build //...
buck2 test //...
buck2 run //tools:lint
buck2 run //tools:typecheck
buck2 run //tools:build_wheel
```

Exact target names will be established by the repository scaffold. No formatter,
type checker, test framework, or dependency resolver is selected by this
architecture document; those choices require an explicit dependency/tooling
decision.

### 15.3 Release identity and compatibility

AgentRig releases use Semantic Versioning and matching `vX.Y.Z` Git tags. A
release is not identified by package metadata alone: it binds the version and
tag to one full source commit and to the SHA-256 digests of exactly one wheel
and one source distribution. The repository release validator enforces this
channel-neutral contract and rejects repository-only or credential-like
content before distribution.

During `0.x`, minor versions may contain documented public-API compatibility
breaks; patch versions remain backward-compatible within a minor line except
for an unavoidable, explicitly documented security or correctness fix. Public
symbols are those exported by their owning package's `__init__.py`. Versions,
tags, artifacts, and manifests are never reused.

The distribution channel, artifact signing, and publication credentials remain
separate decisions. See [the release contract](releases.md) and
[ADR 0002](adr/0002-establish-immutable-release-contract.md).

## 16. Theory of operation

A typical AgentRig run proceeds as follows:

1. The application explicitly constructs provider integrations, configured
   agents, graders, policies, and a workflow.
2. The application supplies typed input and creates a root `RunContext`.
3. The workflow derives a child context for each step and emits lifecycle events.
4. A step invokes a capability, configured agent, Python function, or approved
   process boundary.
5. Integrations translate external requests, events, results, usage, and errors.
6. Outputs and artifacts are recorded with lineage while private content is
   redacted according to policy.
7. Graders evaluate the result and emit versioned grades with evidence.
8. Grade policy deterministically chooses continue, repair, approval, or block.
9. Repair loops are bounded and receive only the failed constraints and relevant
   evidence.
10. The workflow returns a typed outcome and a trace reference.
11. An eval runner may repeat the same target over a versioned dataset and compare
    aggregate results to a baseline.

## 17. Storyworld boundary

AgentRig should enable Storyworld without absorbing it.

AgentRig may supply generic capabilities for structured generation, vision input,
image generation, search, retrieval, agent execution, tools, artifacts,
workflows, graders, and evals.

Storyworld must own:

- Story entities, facts, events, snapshots, and provenance
- Intent, identity, narrative, production, and quality schemas
- Commit gateways and deterministic domain invariants
- Weather, location, licensing, and other domain-specific ports
- Book compilation and rendering
- Project authorization and asset retention
- Storyworld-specific agents, prompts, graders, and eval datasets

Storyworld can expose its deterministic functions as AgentRig tools and wrap its
configured workflows as AgentRig agents.

## 18. Deferred decisions

- Whether the first real autonomous runtime adapter is Codex or Claude
- Which direct model, image, search, and embedding providers to support first
- The Python dependency resolver and lockfile bridge used with Buck2
- Exact schema library and minimum Python version
- Durable execution backend
- Automatic provider ranking, fallback, and health-based routing policy
- Persistent trace and artifact backends

These decisions should be made through small spikes and ADRs, not implied by
folder names.

## 19. Non-normative references

- [Buck2 introduction](https://buck2.build/docs/)
- [Buck2 Python rules](https://buck2.build/docs/prelude/rules/python/)
- [Buck2 prebuilt Python libraries](https://buck2.build/docs/prelude/rules/python/prebuilt_python_library/)
- [OpenAI Codex Python SDK](https://github.com/openai/codex/blob/main/sdk/python/README.md)
- [OpenAI Codex TypeScript SDK](https://github.com/openai/codex/blob/main/sdk/typescript/README.md)
- [OpenAI model and tool capabilities](https://developers.openai.com/api/docs/models)
- [Claude Agent SDK overview](https://code.claude.com/docs/en/agent-sdk/overview)
