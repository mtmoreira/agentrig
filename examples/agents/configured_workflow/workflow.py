"""Provider-neutral configured-agent and workflow composition."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from agentrig.agents import (
    Agent,
    AgentContract,
    AgentLimits,
    AgentRuntime,
    ConfiguredAgent,
)
from agentrig.core import EffectProfile, JsonValue, RunContext
from agentrig.workflow import (
    AgentStep,
    FunctionStep,
    Sequence,
    Step,
    StepDescriptor,
    Workflow,
    WorkflowAgent,
)

REQUEST_SCHEMA = "example.research-request.v1"
BRIEF_SCHEMA = "example.research-brief.v1"
DELIVERY_SCHEMA = "example.delivered-brief.v1"


@dataclass(frozen=True, slots=True, kw_only=True)
class ResearchRequest:
    """Typed input for the configured research agent."""

    topic: str = field(repr=False)


@dataclass(frozen=True, slots=True, kw_only=True)
class ResearchBrief:
    """Strictly decoded autonomous-runtime output."""

    answer: str = field(repr=False)
    sources: tuple[str, ...] = field(repr=False)


@dataclass(frozen=True, slots=True, kw_only=True)
class DeliveredBrief:
    """Final workflow output exposed through a second agent contract."""

    brief: ResearchBrief = field(repr=False)
    word_count: int


@dataclass(frozen=True, slots=True)
class ResearchRequestCodec:
    schema_id: str = REQUEST_SCHEMA

    def encode(self, value: ResearchRequest) -> JsonValue:
        if not isinstance(value, ResearchRequest):
            raise ValueError("research request required")
        topic = value.topic.strip()
        if not topic:
            raise ValueError("research topic must not be empty")
        return {"topic": topic}


@dataclass(frozen=True, slots=True)
class ResearchBriefCodec:
    schema_id: str = BRIEF_SCHEMA

    def decode(self, value: JsonValue) -> ResearchBrief:
        if not isinstance(value, Mapping) or set(value) != {"answer", "sources"}:
            raise ValueError("research brief object required")
        answer = value["answer"]
        encoded_sources = value["sources"]
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("research answer required")
        if not isinstance(encoded_sources, (list, tuple)) or not encoded_sources:
            raise ValueError("at least one research source required")
        sources: list[str] = []
        for source in encoded_sources:
            if not isinstance(source, str) or not source.strip():
                raise ValueError("research sources must be strings")
            sources.append(source)
        if len(sources) != len(set(sources)):
            raise ValueError("research sources must be unique")
        return ResearchBrief(answer=answer, sources=tuple(sources))


def research_contract() -> AgentContract[ResearchRequest, ResearchBrief]:
    return AgentContract(
        agent_id="researcher",
        version="1",
        purpose="Produce one bounded sourced research brief",
        input_schema=REQUEST_SCHEMA,
        output_schema=BRIEF_SCHEMA,
        prompt_version="research-prompt-1",
        effect_profile=EffectProfile.READ_ONLY,
        limits=AgentLimits(max_turns=3, max_tool_calls=1),
        stopping_policy="output_schema_satisfied",
        allowed_tools=("search",),
        permissions={"network": "search_only"},
    )


def delivery_contract() -> AgentContract[ResearchRequest, DeliveredBrief]:
    return AgentContract(
        agent_id="research-delivery",
        version="1",
        purpose="Run the sourced research delivery workflow",
        input_schema=REQUEST_SCHEMA,
        output_schema=DELIVERY_SCHEMA,
        prompt_version="workflow-1",
        effect_profile=EffectProfile.READ_ONLY,
        limits=AgentLimits(max_turns=3, max_tool_calls=1),
        stopping_policy="workflow_completed",
        allowed_tools=("search",),
        permissions={"network": "search_only"},
    )


def configure_researcher(
    runtime: AgentRuntime,
) -> Agent[ResearchRequest, ResearchBrief]:
    """Bind typed schemas and authority to an injected autonomous runtime."""
    return ConfiguredAgent(
        runtime=runtime,
        contract=research_contract(),
        instructions="Return a concise answer and an ordered source URI list.",
        input_codec=ResearchRequestCodec(),
        output_codec=ResearchBriefCodec(),
        provider_options={"response_mode": "structured"},
    )


def build_delivery_workflow(
    researcher: Agent[ResearchRequest, ResearchBrief],
) -> Workflow[ResearchRequest, DeliveredBrief]:
    """Adapt the agent into a typed workflow and add deterministic processing."""

    async def summarize(
        brief: ResearchBrief,
        context: RunContext,
    ) -> DeliveredBrief:
        del context
        return DeliveredBrief(
            brief=brief,
            word_count=len(brief.answer.split()),
        )

    researcher_step: Step[ResearchRequest, ResearchBrief] = AgentStep(researcher)
    summarize_step: Step[ResearchBrief, DeliveredBrief] = FunctionStep(
        descriptor=StepDescriptor(
            step_id="research.count-words",
            version="1",
            effect_profile=EffectProfile.READ_ONLY,
        ),
        function=summarize,
    )
    workflow: Workflow[ResearchRequest, DeliveredBrief] = Sequence(
        researcher_step,
        summarize_step,
    )
    return workflow


def expose_delivery_agent(
    workflow: Workflow[ResearchRequest, DeliveredBrief],
) -> Agent[ResearchRequest, DeliveredBrief]:
    """Expose the composed workflow through the portable agent protocol."""
    return WorkflowAgent(
        workflow=workflow,
        contract=delivery_contract(),
    )
