# Contract design checklist

## Construction and immutability

- Reject empty, padded, noncanonical, nonfinite, duplicate, ambiguous, or mutually inconsistent values during construction.
- Copy caller-owned collections and recursively freeze values retained by an immutable contract.
- Return detached mutable data from serialization methods so callers cannot mutate internal state.
- Keep deterministic input order when order is semantic; reject duplicates rather than silently deduplicating.
- Represent impossible success, failure, blocked, cancellation, approval, and artifact combinations as constructor errors.

## Stable public meaning

- Use stable explicit string values for serialized enums.
- Give contracts, descriptors, schemas, datasets, and reports stable identities and versions when compatibility depends on them.
- Keep request requirements distinct from implementation capability descriptors. Derive unmet requirements before execution.
- Treat JSON Schema as structural validation. Re-check cross-field and domain rules in typed constructors or decoders.
- Preserve typed identity through protocols, adapters, outcomes, and artifacts; do not replace it with provider labels or unstructured dictionaries.

## Failure and privacy boundaries

- Represent expected domain or tool failure as a normalized result when callers are meant to branch on it.
- Normalize cancellation, deadlines, grader failures, provider failures, contract violations, and unexpected implementation faults into their correct categories.
- Never retain an unexpected exception message. Preserve only deliberately safe codes and structured metadata.
- Make `repr`, event attributes, reports, baselines, and serialized forms safe by design. Test absence of private content as behavior.

## Execution semantics

- Check cancellation and deadlines before recording a call or consuming a scripted outcome.
- Check authority, feature support, limits, filters, schemas, and effect constraints before crossing the implementation boundary.
- Classify effects conservatively. Only intrinsically repeatable work may be retried automatically.
- Preserve child failure identity, artifacts, lineage, and safe event ordering through workflow and agent adapters.
