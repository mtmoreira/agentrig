"""Provider-neutral runtime-backed SearchProvider composition."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from agentrig.agents import (
    Agent,
    AgentContract,
    AgentLimits,
    AgentRuntime,
    ConfiguredAgent,
)
from agentrig.capabilities import (
    CapabilityDescriptor,
    CapabilityFeature,
    CapabilityKind,
    CapabilityLimit,
    DataRetention,
    SearchHit,
    SearchProvider,
    SearchRequest,
    SearchResult,
    SearchRetrievalMetadata,
)
from agentrig.core import AgentRigError, EffectProfile, Failure, FailureKind
from agentrig.core._json import JsonValue
from agentrig.core.context import RunContext

REQUEST_SCHEMA = "example.web-research-request.v1"
REPORT_SCHEMA = "example.web-research-report.v1"
MAX_RESULTS = 3

REPORT_JSON_SCHEMA: Mapping[str, JsonValue] = {
    "type": "object",
    "properties": {
        "hits": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_uri": {"type": "string"},
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                },
                "required": ["source_uri", "title", "summary"],
                "additionalProperties": False,
            },
            "minItems": 1,
            "maxItems": MAX_RESULTS,
        },
    },
    "required": ["hits"],
    "additionalProperties": False,
}

INSTRUCTIONS = (
    "Research the encoded query with web search only. Return no more than the "
    "requested number of distinct primary sources. Each hit must contain its "
    "exact HTTPS source URL, title, and a concise evidence summary. Return only "
    "the structured report and never invent a citation."
)


@dataclass(frozen=True, slots=True, kw_only=True)
class ResearchReport:
    """Strict runtime report before request-relative result construction."""

    hits: tuple[SearchHit, ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class SearchRequestCodec:
    schema_id: str = REQUEST_SCHEMA

    def encode(self, value: SearchRequest) -> JsonValue:
        if not isinstance(value, SearchRequest):
            raise ValueError("search request required")
        return {
            "query": value.query,
            "max_results": value.max_results,
        }


@dataclass(frozen=True, slots=True)
class ResearchReportCodec:
    schema_id: str = REPORT_SCHEMA

    def decode(self, value: JsonValue) -> ResearchReport:
        if not isinstance(value, Mapping) or set(value) != {"hits"}:
            raise ValueError("research report object required")
        encoded_hits = value["hits"]
        if not isinstance(encoded_hits, (list, tuple)):
            raise ValueError("research report hits must be an array")
        if not encoded_hits or len(encoded_hits) > MAX_RESULTS:
            raise ValueError("research report hit count is invalid")
        hits = tuple(_decode_hit(item) for item in encoded_hits)
        source_uris = tuple(hit.source_uri for hit in hits)
        if len(source_uris) != len(set(source_uris)):
            raise ValueError("research report source URIs must be unique")
        return ResearchReport(hits=hits)


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeSearchProvider:
    """Expose one structured autonomous runtime as a SearchProvider."""

    descriptor: CapabilityDescriptor
    _agent: Agent[SearchRequest, ResearchReport] = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.descriptor, CapabilityDescriptor):
            raise TypeError("search descriptor must be a CapabilityDescriptor")
        if self.descriptor.kind is not CapabilityKind.SEARCH:
            raise ValueError("search descriptor must use the search kind")
        if not isinstance(self._agent, Agent):
            raise TypeError("runtime search provider requires an Agent")

    async def search(
        self,
        request: SearchRequest,
        context: RunContext,
    ) -> SearchResult:
        if not isinstance(request, SearchRequest):
            raise TypeError("runtime search request must be a SearchRequest")
        if not isinstance(context, RunContext):
            raise TypeError("runtime search context must be a RunContext")
        request.require_supported_by(self.descriptor)
        started = context.clock.monotonic()
        report = (await self._agent.run(request, context)).unwrap()
        duration = max(0.0, context.clock.monotonic() - started)
        try:
            return SearchResult(
                request=request,
                hits=report.hits,
                metadata=SearchRetrievalMetadata(
                    retrieved_at=context.clock.now(),
                    duration_seconds=duration,
                    total_available=len(report.hits),
                ),
            )
        except (TypeError, ValueError) as error:
            raise AgentRigError(
                Failure(
                    kind=FailureKind.INVALID_INPUT,
                    message="research report violated the bounded search contract",
                    code="example.research_report_invalid",
                )
            ) from error


def configure_runtime_search_provider(
    runtime: AgentRuntime,
    *,
    runtime_capability_id: str,
    web_search_tool_id: str,
    search_capability_id: str,
    search_capability_version: str,
    data_retention: DataRetention,
) -> SearchProvider:
    """Bind a portable search capability to an injected autonomous runtime."""
    for label, value in (
        ("runtime capability ID", runtime_capability_id),
        ("web-search tool ID", web_search_tool_id),
        ("search capability ID", search_capability_id),
        ("search capability version", search_capability_version),
    ):
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError(f"{label} must be nonempty and trimmed")
    if not isinstance(data_retention, DataRetention):
        raise TypeError("search data retention must be a DataRetention")
    descriptor = CapabilityDescriptor(
        capability_id=search_capability_id,
        version=search_capability_version,
        kind=CapabilityKind.SEARCH,
        features=frozenset({CapabilityFeature.CITATIONS}),
        limits={CapabilityLimit.MAX_RESULTS: MAX_RESULTS},
        data_retention=data_retention,
    )
    contract = AgentContract[SearchRequest, ResearchReport](
        agent_id="web-researcher",
        version="1",
        purpose="Return a bounded set of citation-ready web sources",
        input_schema=REQUEST_SCHEMA,
        output_schema=REPORT_SCHEMA,
        prompt_version="web-researcher-prompt-1",
        effect_profile=EffectProfile.READ_ONLY,
        limits=AgentLimits(max_turns=1, max_tool_calls=6),
        stopping_policy="bounded_cited_sources",
        allowed_tools=(web_search_tool_id,),
        allowed_capabilities=(runtime_capability_id,),
        permissions={"workspace": "read_only", "network": "allowed"},
    )
    return RuntimeSearchProvider(
        descriptor=descriptor,
        _agent=ConfiguredAgent(
            runtime=runtime,
            contract=contract,
            instructions=INSTRUCTIONS,
            input_codec=SearchRequestCodec(),
            output_codec=ResearchReportCodec(),
        ),
    )


def _decode_hit(value: JsonValue) -> SearchHit:
    if not isinstance(value, Mapping) or set(value) != {
        "source_uri",
        "title",
        "summary",
    }:
        raise ValueError("research hit object required")
    source_uri = value["source_uri"]
    title = value["title"]
    summary = value["summary"]
    if not isinstance(source_uri, str):
        raise ValueError("research source URI must be a string")
    parsed = urlsplit(source_uri)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("research source URI must be an absolute HTTPS URL")
    if not isinstance(title, str):
        raise ValueError("research title must be a string")
    if not isinstance(summary, str):
        raise ValueError("research summary must be a string")
    return SearchHit(
        source_uri=source_uri,
        title=title,
        summary=summary,
    )
