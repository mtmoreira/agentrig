"""Configured typed agents backed by a provider-neutral runtime."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Generic, Protocol, TypeVar, runtime_checkable

from agentrig.agents.contract import AgentContract, AgentResult
from agentrig.agents.runtime import (
    AgentExecutionRequest,
    AgentExecutionResult,
    AgentRuntime,
)
from agentrig.core._json import JsonValue, freeze_json_object
from agentrig.core._validation import require_trimmed_string
from agentrig.core.context import RunContext
from agentrig.core.errors import Failure, FailureKind, normalize_exception

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")
EncodedT = TypeVar("EncodedT", contravariant=True)
DecodedT = TypeVar("DecodedT", covariant=True)


@runtime_checkable
class AgentInputCodec(Protocol[EncodedT]):
    """Encode one typed agent input into its declared portable schema."""

    @property
    def schema_id(self) -> str:
        """Return the stable schema identity implemented by this codec."""
        ...

    def encode(self, value: EncodedT) -> JsonValue:
        """Encode and validate one typed input value."""
        ...


@runtime_checkable
class AgentOutputCodec(Protocol[DecodedT]):
    """Decode one portable runtime output into its declared typed schema."""

    @property
    def schema_id(self) -> str:
        """Return the stable schema identity implemented by this codec."""
        ...

    def decode(self, value: JsonValue) -> DecodedT:
        """Validate and decode one runtime output value."""
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class ConfiguredAgent(Generic[InputT, OutputT]):
    """Bind a typed contract and codecs to one autonomous runtime."""

    runtime: AgentRuntime
    contract: AgentContract[InputT, OutputT]
    instructions: str
    input_codec: AgentInputCodec[InputT]
    output_codec: AgentOutputCodec[OutputT]
    provider_options: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.runtime, AgentRuntime):
            raise TypeError("configured agent runtime must be an AgentRuntime")
        if not isinstance(self.contract, AgentContract):
            raise TypeError("configured agent contract must be an AgentContract")
        require_trimmed_string("agent instructions", self.instructions)
        if not isinstance(self.input_codec, AgentInputCodec):
            raise TypeError("configured agent input_codec must be an AgentInputCodec")
        if not isinstance(self.output_codec, AgentOutputCodec):
            raise TypeError(
                "configured agent output_codec must be an AgentOutputCodec"
            )

        input_schema = require_trimmed_string(
            "agent input codec schema",
            self.input_codec.schema_id,
        )
        if input_schema != self.contract.input_schema:
            raise ValueError(
                "input codec schema must match the agent contract input schema"
            )
        output_schema = require_trimmed_string(
            "agent output codec schema",
            self.output_codec.schema_id,
        )
        if output_schema != self.contract.output_schema:
            raise ValueError(
                "output codec schema must match the agent contract output schema"
            )
        object.__setattr__(
            self,
            "provider_options",
            freeze_json_object(
                "configured agent provider options",
                self.provider_options,
            ),
        )

    async def run(
        self,
        input: InputT,
        context: RunContext,
    ) -> AgentResult[OutputT]:
        """Encode, execute, validate, and return one portable typed result."""
        if not isinstance(context, RunContext):
            raise TypeError("agent context must be a RunContext")

        constraint_failure = _check_constraints(context)
        if constraint_failure is not None:
            return AgentResult.from_failure(constraint_failure)

        try:
            encoded_input = self.input_codec.encode(input)
            request = AgentExecutionRequest(
                contract=self.contract,
                instructions=self.instructions,
                input=encoded_input,
                provider_options=self.provider_options,
            )
        except Exception:
            return AgentResult.from_failure(
                _schema_failure("input", self.contract.input_schema)
            )

        try:
            execution = await self.runtime.execute(request, context)
            if not isinstance(execution, AgentExecutionResult):
                raise TypeError("agent runtime returned an invalid result")
            constraint_failure = _check_constraints(context)
            if constraint_failure is not None:
                return AgentResult.from_failure(
                    constraint_failure,
                    artifacts=execution.result.artifacts,
                )
        except asyncio.CancelledError as error:
            return AgentResult.from_failure(normalize_exception(error))
        except Exception as error:
            return AgentResult.from_failure(normalize_exception(error))

        runtime_result = execution.result
        if not runtime_result.is_success:
            if runtime_result.failure is None:
                raise AssertionError("validated runtime failure has no details")
            return AgentResult.from_failure(
                runtime_result.failure,
                artifacts=runtime_result.artifacts,
            )

        try:
            output = self.output_codec.decode(runtime_result.unwrap())
        except Exception:
            return AgentResult.from_failure(
                _schema_failure("output", self.contract.output_schema),
                artifacts=runtime_result.artifacts,
            )
        return AgentResult.succeeded(
            output,
            artifacts=runtime_result.artifacts,
        )


def _check_constraints(context: RunContext) -> Failure | None:
    try:
        context.cancellation.raise_if_cancelled()
        if context.deadline is not None:
            context.deadline.raise_if_expired(context.clock)
    except asyncio.CancelledError as error:
        return normalize_exception(error)
    except Exception as error:
        return normalize_exception(error)
    return None


def _schema_failure(direction: str, schema_id: str) -> Failure:
    return Failure(
        kind=FailureKind.INVALID_INPUT,
        message=f"agent {direction} does not satisfy its declared schema",
        code=f"agent.{direction}_schema_mismatch",
        metadata={"schema_id": schema_id},
    )
