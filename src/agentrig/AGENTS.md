# AgentRig framework guidance

## Package ownership

- Put universal execution vocabulary and provider-free primitives in `core`.
- Put portable agent/runtime contracts in `agents`; do not import integrations.
- Put narrow direct capability protocols in `capabilities`; express portable
  features and limits rather than provider-specific request flags.
- Put typed orchestration in `workflow`; preserve child failure identity,
  artifacts, cancellation, deadlines, effects, and event lineage.
- Put reusable deterministic fakes and conformance suites in `testing`.
- Put provider SDK translation in `integrations/<provider>` and keep it optional.

## Contract changes

- Inspect the owning module, its public `__init__.py`, BUCK target, unit tests,
  typing fixtures, scripted fake, and contract suite before changing a public
  contract.
- Validate inputs at construction and again at provider or workflow boundaries
  where implementations could return invalid values.
- Treat schema validation as structural. Enforce domain invariants in typed
  decoders or constructors rather than relying on provider compliance.
- Use stable string values for serialized enums and stable versioned identities
  for contracts, descriptors, schemas, reports, and datasets.
- Preserve JSON safety, immutable snapshots, detached serialization results,
  and deterministic ordering.
- Keep expected domain or tool failures distinct from cancellation,
  implementation faults, and grader failures.

## Provider integrations

- Translate provider types behind an injected seam so unit tests need no SDK,
  network, subprocess, or credentials.
- Enforce the portable contract's tools, permissions, workspace, network,
  approval, and execution bounds before client construction and on every turn.
- Emit only allowlisted lifecycle, safe identity, and aggregate usage fields.
  Never forward raw notifications, prompts, reasoning, arguments, or transport
  error messages.
- Close provider processes, streams, and background readers on success, failure,
  cancellation, and timeout.
- Cover the real dependency bridge with an offline integration test and keep
  authenticated execution in a separate live target.

## Required evidence

- Add focused unit tests for valid states, invalid states, cancellation, and raw
  exception normalization.
- Add or update typing fixtures when generic composition or public protocols
  change.
- Extend scripted fakes and reusable contract suites when portable behavior
  changes; do not encode provider behavior into shared suites.
- Update package BUCK dependencies explicitly and run the owning package target
  before broader lanes.
