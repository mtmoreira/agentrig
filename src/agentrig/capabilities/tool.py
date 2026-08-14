"""Typed tool contracts, invocations, and normalized results."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

from agentrig.capabilities.base import CapabilityDescriptor, CapabilityKind
from agentrig.core._json import JsonValue, freeze_json_object, freeze_json_value
from agentrig.core._validation import require_trimmed_string
from agentrig.core.artifacts import ArtifactRef
from agentrig.core.context import RunContext
from agentrig.core.effects import EffectProfile
from agentrig.core.errors import Failure
from agentrig.core.outcomes import ExecutionOutcome, ExecutionStatus

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class _MissingEncodedOutput:
    __slots__ = ()


_MISSING_ENCODED_OUTPUT = _MissingEncodedOutput()


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolSchema(Generic[InputT]):
    """Immutable JSON schema plus typed encoder and decoder."""

    schema_id: str
    json_schema: Mapping[str, JsonValue]
    encoder: Callable[[InputT], JsonValue] = field(
        repr=False,
        compare=False,
    )
    decoder: Callable[[JsonValue], InputT] = field(
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        require_trimmed_string("tool schema ID", self.schema_id)
        frozen_schema = freeze_json_object(
            "tool JSON schema",
            self.json_schema,
        )
        if not frozen_schema:
            raise ValueError("tool JSON schema must not be empty")
        object.__setattr__(self, "json_schema", frozen_schema)
        if not callable(self.encoder):
            raise TypeError("tool schema encoder must be callable")
        if not callable(self.decoder):
            raise TypeError("tool schema decoder must be callable")

    def encode(self, value: InputT) -> JsonValue:
        """Encode and freeze one typed value for a tool boundary."""
        return freeze_json_value("tool encoded value", self.encoder(value))

    def decode(self, value: JsonValue) -> InputT:
        """Freeze and decode one value received from a tool boundary."""
        return self.decoder(freeze_json_value("tool decoded value", value))


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolContract(Generic[InputT, OutputT]):
    """Stable tool identity, bounded purpose, schemas, and side effects."""

    descriptor: CapabilityDescriptor
    purpose: str
    effect_profile: EffectProfile
    input_schema: ToolSchema[InputT] = field(repr=False)
    output_schema: ToolSchema[OutputT] = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.descriptor, CapabilityDescriptor):
            raise TypeError(
                "tool contract descriptor must be a CapabilityDescriptor"
            )
        if self.descriptor.kind is not CapabilityKind.TOOL:
            raise ValueError("tool contract descriptor must use the tool kind")
        _require_content_text("tool purpose", self.purpose)
        if not isinstance(self.effect_profile, EffectProfile):
            raise TypeError(
                "tool contract effect_profile must be an EffectProfile"
            )
        if not isinstance(self.input_schema, ToolSchema):
            raise TypeError("tool contract input_schema must be a ToolSchema")
        if not isinstance(self.output_schema, ToolSchema):
            raise TypeError("tool contract output_schema must be a ToolSchema")

    @property
    def tool_id(self) -> str:
        """Return the stable capability identity used by agent allowlists."""
        return self.descriptor.capability_id

    @property
    def version(self) -> str:
        """Return the stable tool implementation version."""
        return self.descriptor.version


@dataclass(frozen=True, slots=True, init=False)
class ToolInvocation(Generic[InputT, OutputT]):
    """One typed input encoded against the exact invoked tool contract."""

    invocation_id: str
    contract: ToolContract[InputT, OutputT]
    input: InputT = field(repr=False)
    encoded_input: JsonValue = field(repr=False)

    def __init__(
        self,
        *,
        invocation_id: str,
        contract: ToolContract[InputT, OutputT],
        input: InputT,
    ) -> None:
        require_trimmed_string("tool invocation ID", invocation_id)
        if not isinstance(contract, ToolContract):
            raise TypeError(
                "tool invocation contract must be a ToolContract"
            )
        encoded_input = contract.input_schema.encode(input)
        object.__setattr__(self, "invocation_id", invocation_id)
        object.__setattr__(self, "contract", contract)
        object.__setattr__(self, "input", input)
        object.__setattr__(self, "encoded_input", encoded_input)


@dataclass(frozen=True, slots=True, init=False)
class ToolResult(Generic[OutputT]):
    """One schema-validated success or normalized expected tool failure."""

    invocation_id: str
    tool_id: str
    tool_version: str
    output_schema_id: str
    outcome: ExecutionOutcome[OutputT] = field(repr=False)
    encoded_output: JsonValue | None = field(default=None, repr=False)

    def __init__(
        self,
        *,
        invocation: ToolInvocation[Any, OutputT],
        encoded_output: JsonValue | _MissingEncodedOutput = (
            _MISSING_ENCODED_OUTPUT
        ),
        failure: Failure | None = None,
        artifacts: Iterable[ArtifactRef] = (),
    ) -> None:
        if not isinstance(invocation, ToolInvocation):
            raise TypeError("tool result invocation must be a ToolInvocation")
        if isinstance(encoded_output, _MissingEncodedOutput):
            if not isinstance(failure, Failure):
                raise ValueError(
                    "tool result requires encoded output or normalized failure"
                )
            frozen_output: JsonValue | None = None
            outcome: ExecutionOutcome[OutputT] = ExecutionOutcome.from_failure(
                failure,
                artifacts=artifacts,
            )
        else:
            if failure is not None:
                raise ValueError(
                    "successful tool result must not contain a failure"
                )
            frozen_output = freeze_json_value(
                "tool result encoded output",
                encoded_output,
            )
            output = invocation.contract.output_schema.decode(frozen_output)
            outcome = ExecutionOutcome.succeeded(
                output,
                artifacts=artifacts,
            )
        object.__setattr__(self, "invocation_id", invocation.invocation_id)
        object.__setattr__(self, "tool_id", invocation.contract.tool_id)
        object.__setattr__(self, "tool_version", invocation.contract.version)
        object.__setattr__(
            self,
            "output_schema_id",
            invocation.contract.output_schema.schema_id,
        )
        object.__setattr__(self, "encoded_output", frozen_output)
        object.__setattr__(self, "outcome", outcome)

    @classmethod
    def succeeded(
        cls,
        *,
        invocation: ToolInvocation[InputT, OutputT],
        encoded_output: JsonValue,
        artifacts: Iterable[ArtifactRef] = (),
    ) -> ToolResult[OutputT]:
        """Decode a successful output through the invoked contract schema."""
        if not isinstance(invocation, ToolInvocation):
            raise TypeError(
                "successful tool result invocation must be a ToolInvocation"
            )
        return cls(
            invocation=invocation,
            encoded_output=encoded_output,
            artifacts=artifacts,
        )

    @classmethod
    def from_failure(
        cls,
        *,
        invocation: ToolInvocation[InputT, OutputT],
        failure: Failure,
        artifacts: Iterable[ArtifactRef] = (),
    ) -> ToolResult[OutputT]:
        """Return one expected normalized failure bound to the invocation."""
        if not isinstance(invocation, ToolInvocation):
            raise TypeError(
                "failed tool result invocation must be a ToolInvocation"
            )
        if not isinstance(failure, Failure):
            raise TypeError("tool result failure must be a Failure")
        return cls(
            invocation=invocation,
            failure=failure,
            artifacts=artifacts,
        )

    @property
    def status(self) -> ExecutionStatus:
        """Return the stable normalized execution status."""
        return self.outcome.status

    @property
    def failure(self) -> Failure | None:
        """Return normalized failure details when unsuccessful."""
        return self.outcome.failure

    @property
    def artifacts(self) -> tuple[ArtifactRef, ...]:
        """Return durable artifacts produced before termination."""
        return self.outcome.artifacts

    def unwrap(self) -> OutputT:
        """Return successful output or raise its normalized error."""
        return self.outcome.unwrap()


@runtime_checkable
class Tool(Protocol[InputT, OutputT]):
    """Invoke a typed bounded capability exposed to an agent."""

    @property
    def contract(self) -> ToolContract[InputT, OutputT]:
        """Return stable schemas, identity, purpose, and effect profile."""
        ...

    async def invoke(
        self,
        invocation: ToolInvocation[InputT, OutputT],
        context: RunContext,
    ) -> ToolResult[OutputT]:
        """Return a typed result or normalized expected failure."""
        ...


def _require_content_text(field_name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must contain non-whitespace text")
    return value
