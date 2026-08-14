# Sourced digest capability pipeline

This example composes two provider-independent capability protocols into a
typed workflow:

```text
DigestRequest
  -> SearchProvider.search(SearchRequest)
  -> SearchEvidence
  -> StructuredGenerator.generate(StructuredGenerationRequest)
  -> SourcedDigest
```

The search request carries a hard source-count bound and requires citation
support. The generation request carries an immutable JSON Schema and a hard
output-token bound. Each request is checked against the injected capability's
portable descriptor before provider execution.

The output schema proves that provider JSON has the expected shape, but shape
validation alone cannot establish citation provenance. The second workflow
step therefore accepts a generated `source_uri` only when the preceding search
result returned that exact URI. Invented citations fail with a sanitized
workflow error.

`workflow.py` owns the provider-neutral composition and domain types.
`scripted.py` injects deterministic implementations from `agentrig.testing`;
it does not contact a provider or require credentials. The scripted path also
shows that search metadata, generation usage, and model identity remain
portable output data while topics, evidence text, prompts, and generated prose
do not enter workflow lifecycle events.

Run the deterministic example:

```console
uv run python -m examples.capabilities.sourced_digest.scripted
```

Or with Buck2:

```console
./buck2 run //examples/capabilities/sourced_digest:scripted
./buck2 test //examples/capabilities/sourced_digest:test
```

Replace `ScriptedSearchProvider` and `ScriptedStructuredGenerator` at the
composition root with live implementations of the same protocols. The
workflow and its trust-boundary checks do not change.
