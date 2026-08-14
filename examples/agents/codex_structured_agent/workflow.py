"""Provider-neutral typed composition for the structured-agent example."""

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
from agentrig.core import EffectProfile, JsonValue

REQUEST_SCHEMA = "example.decision-request.v1"
BRIEF_SCHEMA = "example.decision-brief.v1"

BRIEF_JSON_SCHEMA: Mapping[str, JsonValue] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "risks": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        },
        "recommendation": {
            "type": "string",
            "enum": ["proceed", "revise", "stop"],
        },
    },
    "required": ["summary", "risks", "recommendation"],
    "additionalProperties": False,
}

INSTRUCTIONS = (
    "Analyze the encoded decision request without using tools. Return one "
    "concise JSON object matching the configured output schema."
)


@dataclass(frozen=True, slots=True, kw_only=True)
class DecisionRequest:
    """Private typed input supplied to either runtime."""

    question: str = field(repr=False)
    constraints: tuple[str, ...] = field(repr=False)


@dataclass(frozen=True, slots=True, kw_only=True)
class DecisionBrief:
    """Strictly decoded decision output."""

    summary: str = field(repr=False)
    risks: tuple[str, ...] = field(repr=False)
    recommendation: str


@dataclass(frozen=True, slots=True)
class DecisionRequestCodec:
    schema_id: str = REQUEST_SCHEMA

    def encode(self, value: DecisionRequest) -> JsonValue:
        if not isinstance(value, DecisionRequest):
            raise ValueError("decision request required")
        question = value.question.strip()
        if not question:
            raise ValueError("decision question must not be empty")
        constraints = tuple(constraint.strip() for constraint in value.constraints)
        if not constraints or any(not constraint for constraint in constraints):
            raise ValueError("decision constraints must be nonempty")
        if len(constraints) != len(set(constraints)):
            raise ValueError("decision constraints must be unique")
        return {
            "question": question,
            "constraints": constraints,
        }


@dataclass(frozen=True, slots=True)
class DecisionBriefCodec:
    schema_id: str = BRIEF_SCHEMA

    def decode(self, value: JsonValue) -> DecisionBrief:
        if not isinstance(value, Mapping) or set(value) != {
            "summary",
            "risks",
            "recommendation",
        }:
            raise ValueError("decision brief object required")
        summary = value["summary"]
        encoded_risks = value["risks"]
        recommendation = value["recommendation"]
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("decision summary required")
        if not isinstance(encoded_risks, (list, tuple)) or not encoded_risks:
            raise ValueError("at least one decision risk required")
        risks: list[str] = []
        for risk in encoded_risks:
            if not isinstance(risk, str) or not risk.strip():
                raise ValueError("decision risks must be nonempty strings")
            risks.append(risk)
        if len(risks) != len(set(risks)):
            raise ValueError("decision risks must be unique")
        if recommendation not in {"proceed", "revise", "stop"}:
            raise ValueError("decision recommendation is invalid")
        return DecisionBrief(
            summary=summary,
            risks=tuple(risks),
            recommendation=recommendation,
        )


def decision_contract(
    *,
    runtime_capability_id: str,
) -> AgentContract[DecisionRequest, DecisionBrief]:
    """Declare least authority while leaving provider selection injectable."""
    if (
        not isinstance(runtime_capability_id, str)
        or not runtime_capability_id
        or runtime_capability_id != runtime_capability_id.strip()
    ):
        raise ValueError("runtime capability ID must be nonempty and trimmed")
    return AgentContract(
        agent_id="decision-brief",
        version="1",
        purpose="Produce one bounded structured decision brief",
        input_schema=REQUEST_SCHEMA,
        output_schema=BRIEF_SCHEMA,
        prompt_version="decision-prompt-1",
        effect_profile=EffectProfile.READ_ONLY,
        limits=AgentLimits(max_turns=1, max_tool_calls=0),
        stopping_policy="output_schema_satisfied",
        allowed_capabilities=(runtime_capability_id,),
        permissions={
            "workspace": "read_only",
            "network": "denied",
        },
    )


def configure_decision_agent(
    runtime: AgentRuntime,
    *,
    runtime_capability_id: str,
) -> Agent[DecisionRequest, DecisionBrief]:
    """Bind the portable agent contract and codecs to an injected runtime."""
    return ConfiguredAgent(
        runtime=runtime,
        contract=decision_contract(
            runtime_capability_id=runtime_capability_id,
        ),
        instructions=INSTRUCTIONS,
        input_codec=DecisionRequestCodec(),
        output_codec=DecisionBriefCodec(),
    )
