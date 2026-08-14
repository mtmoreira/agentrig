"""Provider-neutral sourced-digest capability pipeline."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field

from agentrig.capabilities import (
    GenerationUsage,
    ModelMetadata,
    SearchCitation,
    SearchProvider,
    SearchRequest,
    SearchResult,
    SearchRetrievalMetadata,
    StructuredGenerationRequest,
    StructuredGenerator,
    StructuredOutputSchema,
    TextGenerationRequest,
)
from agentrig.core import (
    AgentRigError,
    EffectProfile,
    Failure,
    FailureKind,
    JsonValue,
    RunContext,
)
from agentrig.workflow import (
    FunctionStep,
    Sequence,
    Step,
    StepDescriptor,
    Workflow,
)

DIGEST_SCHEMA = "example.sourced-digest.v1"


@dataclass(frozen=True, slots=True, kw_only=True)
class DigestRequest:
    """Private topic and hard source bound supplied by the caller."""

    topic: str = field(repr=False)
    max_sources: int = 2

    def __post_init__(self) -> None:
        _require_content("digest topic", self.topic)
        if (
            isinstance(self.max_sources, bool)
            or not isinstance(self.max_sources, int)
            or self.max_sources <= 0
        ):
            raise ValueError("digest max_sources must be a positive integer")


@dataclass(frozen=True, slots=True, kw_only=True)
class GeneratedDigest:
    """Strict intermediate value decoded from provider JSON."""

    headline: str = field(repr=False)
    summary: str = field(repr=False)
    source_uris: tuple[str, ...] = field(repr=False)


@dataclass(frozen=True, slots=True, kw_only=True)
class SourcedDigest:
    """Typed digest with citations and portable execution metadata."""

    headline: str = field(repr=False)
    summary: str = field(repr=False)
    citations: tuple[SearchCitation, ...]
    search_metadata: SearchRetrievalMetadata
    generation_usage: GenerationUsage
    generation_model: ModelMetadata

    def __post_init__(self) -> None:
        _require_content("digest headline", self.headline)
        _require_content("digest summary", self.summary)
        if not self.citations or any(
            not isinstance(citation, SearchCitation)
            for citation in self.citations
        ):
            raise ValueError("sourced digest requires search citations")
        if not isinstance(self.search_metadata, SearchRetrievalMetadata):
            raise TypeError(
                "sourced digest search_metadata must be search metadata"
            )
        if not isinstance(self.generation_usage, GenerationUsage):
            raise TypeError(
                "sourced digest generation_usage must be GenerationUsage"
            )
        if not isinstance(self.generation_model, ModelMetadata):
            raise TypeError(
                "sourced digest generation_model must be ModelMetadata"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class SearchEvidence:
    """Internal handoff from bounded search to structured generation."""

    request: DigestRequest = field(repr=False)
    result: SearchResult = field(repr=False)


def digest_output_schema() -> StructuredOutputSchema[GeneratedDigest]:
    """Return the strict portable schema requested from any generator."""
    json_schema: Mapping[str, JsonValue] = {
        "type": "object",
        "properties": {
            "headline": {"type": "string"},
            "summary": {"type": "string"},
            "source_uris": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "uniqueItems": True,
            },
        },
        "required": ["headline", "summary", "source_uris"],
        "additionalProperties": False,
    }
    return StructuredOutputSchema(
        schema_id=DIGEST_SCHEMA,
        json_schema=json_schema,
        decoder=_decode_digest,
    )


def build_sourced_digest_workflow(
    *,
    search_provider: SearchProvider,
    generator: StructuredGenerator[GeneratedDigest],
) -> Workflow[DigestRequest, SourcedDigest]:
    """Compose replaceable search and structured-generation capabilities."""
    if not isinstance(search_provider, SearchProvider):
        raise TypeError("sourced digest search_provider must satisfy SearchProvider")
    if not isinstance(generator, StructuredGenerator):
        raise TypeError(
            "sourced digest generator must satisfy StructuredGenerator"
        )

    async def search_sources(
        request: DigestRequest,
        context: RunContext,
    ) -> SearchEvidence:
        search_request = SearchRequest(
            query=request.topic,
            max_results=request.max_sources,
        )
        _require_capability_support(
            search_request.requirements.unmet_by(search_provider.descriptor),
            code="example.search_requirements_unmet",
        )
        result = await search_provider.search(search_request, context)
        if not result.hits:
            raise AgentRigError(
                Failure(
                    kind=FailureKind.WORKFLOW_BLOCKED,
                    message="no source evidence was available for the digest",
                    code="example.no_source_evidence",
                )
            )
        return SearchEvidence(request=request, result=result)

    async def generate_digest(
        evidence: SearchEvidence,
        context: RunContext,
    ) -> SourcedDigest:
        generation_request = StructuredGenerationRequest(
            input=TextGenerationRequest(
                prompt=_generation_prompt(evidence),
                max_output_tokens=256,
            ),
            output_schema=digest_output_schema(),
        )
        _require_capability_support(
            generation_request.requirements.unmet_by(generator.descriptor),
            code="example.generation_requirements_unmet",
        )
        try:
            generated = await generator.generate(generation_request, context)
        except ValueError as error:
            raise AgentRigError(
                Failure(
                    kind=FailureKind.INVALID_INPUT,
                    message="generated digest did not match its output schema",
                    code="example.digest_schema_mismatch",
                )
            ) from error

        citations_by_uri = {
            hit.source_uri: hit.citation for hit in evidence.result.hits
        }
        if any(
            source_uri not in citations_by_uri
            for source_uri in generated.output.source_uris
        ):
            raise AgentRigError(
                Failure(
                    kind=FailureKind.INVALID_INPUT,
                    message="generated digest referenced an untrusted source",
                    code="example.citation_not_in_search_results",
                )
            )

        return SourcedDigest(
            headline=generated.output.headline,
            summary=generated.output.summary,
            citations=tuple(
                citations_by_uri[source_uri]
                for source_uri in generated.output.source_uris
            ),
            search_metadata=evidence.result.metadata,
            generation_usage=generated.usage,
            generation_model=generated.model,
        )

    search_step: Step[DigestRequest, SearchEvidence] = FunctionStep(
        descriptor=StepDescriptor(
            step_id="digest.search-sources",
            version="1",
            effect_profile=EffectProfile.READ_ONLY,
        ),
        function=search_sources,
    )
    generation_step: Step[SearchEvidence, SourcedDigest] = FunctionStep(
        descriptor=StepDescriptor(
            step_id="digest.generate",
            version="1",
            effect_profile=EffectProfile.READ_ONLY,
        ),
        function=generate_digest,
    )
    workflow: Workflow[DigestRequest, SourcedDigest] = Sequence(
        search_step,
        generation_step,
    )
    return workflow


def _decode_digest(value: JsonValue) -> GeneratedDigest:
    if not isinstance(value, Mapping) or set(value) != {
        "headline",
        "summary",
        "source_uris",
    }:
        raise ValueError("generated digest must match the declared object")
    headline = _require_content("generated digest headline", value["headline"])
    summary = _require_content("generated digest summary", value["summary"])
    encoded_source_uris = value["source_uris"]
    if not isinstance(encoded_source_uris, (list, tuple)):
        raise ValueError("generated digest source_uris must be an array")
    source_uris = tuple(
        _require_content("generated digest source URI", source_uri)
        for source_uri in encoded_source_uris
    )
    if not source_uris:
        raise ValueError("generated digest requires at least one source URI")
    if len(source_uris) != len(set(source_uris)):
        raise ValueError("generated digest source URIs must be unique")
    return GeneratedDigest(
        headline=headline,
        summary=summary,
        source_uris=source_uris,
    )


def _generation_prompt(evidence: SearchEvidence) -> str:
    sources = [
        {
            "source_uri": hit.source_uri,
            "title": hit.title,
            "evidence": hit.summary if hit.summary is not None else hit.excerpt,
        }
        for hit in evidence.result.hits
    ]
    return (
        "Create a concise sourced digest for the supplied topic. Cite only "
        "source_uri values from this evidence object: "
        + json.dumps(
            {"topic": evidence.request.topic, "sources": sources},
            sort_keys=True,
        )
    )


def _require_capability_support(
    unmet_requirements: tuple[str, ...],
    *,
    code: str,
) -> None:
    if unmet_requirements:
        raise AgentRigError(
            Failure(
                kind=FailureKind.INVALID_INPUT,
                message="configured capability cannot satisfy the request",
                code=code,
                metadata={
                    "unmet_requirement_count": str(len(unmet_requirements))
                },
            )
        )


def _require_content(field_name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must contain non-whitespace text")
    return value
