from __future__ import annotations

import asyncio
import unittest
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime

from agentrig.capabilities import (
    CapabilityDescriptor,
    CapabilityKind,
    DataRetention,
    Tool,
    ToolContract,
    ToolInvocation,
    ToolResult,
    ToolSchema,
)
from agentrig.core import (
    AgentRigError,
    ArtifactId,
    ArtifactRef,
    CancellationSource,
    EffectProfile,
    ExecutionStatus,
    Failure,
    FailureKind,
    JsonValue,
    RunCancelled,
    RunContext,
    RunId,
)


@dataclass(frozen=True)
class AddInput:
    left: int
    right: int


@dataclass(frozen=True)
class AddOutput:
    total: int


@dataclass(frozen=True)
class FixedClock:
    def now(self) -> datetime:
        return datetime(2026, 8, 14, 8, 0, tzinfo=UTC)

    def monotonic(self) -> float:
        return 100.0


@dataclass
class SequentialRunIdGenerator:
    next_value: int = 1

    def generate(self) -> RunId:
        run_id = RunId(f"run-{self.next_value}")
        self.next_value += 1
        return run_id


def create_context(
    source: CancellationSource | None = None,
) -> RunContext:
    owned_source = source if source is not None else CancellationSource()
    return RunContext.create_root(
        clock=FixedClock(),
        id_generator=SequentialRunIdGenerator(),
        cancellation=owned_source.token,
    )


def encode_input(value: AddInput) -> JsonValue:
    return {"left": value.left, "right": value.right}


def decode_input(value: JsonValue) -> AddInput:
    if not isinstance(value, Mapping):
        raise ValueError("tool input must be an object")
    left = value.get("left")
    right = value.get("right")
    if (
        isinstance(left, bool)
        or not isinstance(left, int)
        or isinstance(right, bool)
        or not isinstance(right, int)
    ):
        raise ValueError("tool input fields must be integers")
    return AddInput(left=left, right=right)


def encode_output(value: AddOutput) -> JsonValue:
    return {"total": value.total}


def decode_output(value: JsonValue) -> AddOutput:
    if not isinstance(value, Mapping):
        raise ValueError("tool output must be an object")
    total = value.get("total")
    if isinstance(total, bool) or not isinstance(total, int):
        raise ValueError("tool output total must be an integer")
    return AddOutput(total=total)


def create_input_schema() -> ToolSchema[AddInput]:
    return ToolSchema(
        schema_id="example.math.add.input.v1",
        json_schema={
            "type": "object",
            "required": ["left", "right"],
        },
        encoder=encode_input,
        decoder=decode_input,
    )


def create_output_schema() -> ToolSchema[AddOutput]:
    return ToolSchema(
        schema_id="example.math.add.output.v1",
        json_schema={
            "type": "object",
            "required": ["total"],
        },
        encoder=encode_output,
        decoder=decode_output,
    )


def create_contract(
    *,
    version: str = "2",
    effect_profile: EffectProfile = EffectProfile.READ_ONLY,
) -> ToolContract[AddInput, AddOutput]:
    return ToolContract(
        descriptor=CapabilityDescriptor(
            capability_id="math.add",
            version=version,
            kind=CapabilityKind.TOOL,
            data_retention=DataRetention.NOT_RETAINED,
        ),
        purpose="  Add two bounded integers.\n",
        effect_profile=effect_profile,
        input_schema=create_input_schema(),
        output_schema=create_output_schema(),
    )


def create_invocation(
    contract: ToolContract[AddInput, AddOutput] | None = None,
) -> ToolInvocation[AddInput, AddOutput]:
    return ToolInvocation(
        invocation_id="invocation-1",
        contract=contract if contract is not None else create_contract(),
        input=AddInput(left=2, right=3),
    )


def create_artifact() -> ArtifactRef:
    return ArtifactRef(
        artifact_id=ArtifactId("artifact-tool-output"),
        kind="tool-output",
        media_type="application/json",
        producer_run_id=RunId("run-producer"),
        workspace_path="outputs/tool.json",
    )


@dataclass
class ScriptedAddTool:
    contract: ToolContract[AddInput, AddOutput]
    outcomes: tuple[JsonValue | Failure, ...]
    calls: list[tuple[ToolInvocation[AddInput, AddOutput], RunContext]] = field(
        default_factory=list
    )

    async def invoke(
        self,
        invocation: ToolInvocation[AddInput, AddOutput],
        context: RunContext,
    ) -> ToolResult[AddOutput]:
        if invocation.contract != self.contract:
            raise ValueError("tool invocation contract does not match runtime")
        context.cancellation.raise_if_cancelled()
        if context.deadline is not None:
            context.deadline.raise_if_expired(context.clock)
        index = len(self.calls)
        self.calls.append((invocation, context))
        scripted = self.outcomes[min(index, len(self.outcomes) - 1)]
        if isinstance(scripted, Failure):
            return ToolResult.from_failure(
                invocation=invocation,
                failure=scripted,
            )
        return ToolResult.succeeded(
            invocation=invocation,
            encoded_output=scripted,
        )


async def invoke_typed(
    tool: Tool[AddInput, AddOutput],
    invocation: ToolInvocation[AddInput, AddOutput],
    context: RunContext,
) -> ToolResult[AddOutput]:
    return await tool.invoke(invocation, context)


class ToolSchemaTest(unittest.TestCase):
    def test_freezes_schema_and_values_across_typed_codecs(self) -> None:
        json_schema: dict[str, JsonValue] = {
            "type": "object",
            "required": ["left", "right"],
        }
        schema = ToolSchema(
            schema_id="example.math.add.input.v1",
            json_schema=json_schema,
            encoder=encode_input,
            decoder=decode_input,
        )
        json_schema["type"] = "changed"
        required = json_schema["required"]
        if not isinstance(required, list):
            raise AssertionError("test schema required field must be a list")
        required.append("changed")

        encoded = schema.encode(AddInput(left=2, right=3))

        self.assertEqual(schema.json_schema["type"], "object")
        self.assertEqual(schema.json_schema["required"], ("left", "right"))
        self.assertEqual(encoded, {"left": 2, "right": 3})
        self.assertEqual(schema.decode(encoded), AddInput(left=2, right=3))
        self.assertNotIn("encoder", repr(schema))
        self.assertNotIn("decoder", repr(schema))
        with self.assertRaises(TypeError):
            schema.json_schema["type"] = "changed"  # type: ignore[index]
        if not isinstance(encoded, Mapping):
            raise AssertionError("encoded tool input must be an object")
        with self.assertRaises(TypeError):
            encoded["left"] = 10  # type: ignore[index]

    def test_rejects_invalid_schema_configuration_and_encoded_json(self) -> None:
        with self.assertRaises(ValueError):
            ToolSchema(
                schema_id=" padded ",
                json_schema={"type": "object"},
                encoder=encode_input,
                decoder=decode_input,
            )
        with self.assertRaises(ValueError):
            ToolSchema(
                schema_id="schema",
                json_schema={},
                encoder=encode_input,
                decoder=decode_input,
            )
        with self.assertRaises(TypeError):
            ToolSchema(
                schema_id="schema",
                json_schema={"type": "object"},
                encoder="invalid",  # type: ignore[arg-type]
                decoder=decode_input,
            )
        with self.assertRaises(TypeError):
            ToolSchema(
                schema_id="schema",
                json_schema={"type": "object"},
                encoder=encode_input,
                decoder="invalid",  # type: ignore[arg-type]
            )

        def encode_invalid(value: AddInput) -> JsonValue:
            del value
            return object()  # type: ignore[return-value]

        invalid_encoder = ToolSchema(
            schema_id="schema",
            json_schema={"type": "object"},
            encoder=encode_invalid,
            decoder=decode_input,
        )
        with self.assertRaises(ValueError):
            invalid_encoder.encode(AddInput(left=1, right=2))


class ToolContractTest(unittest.TestCase):
    def test_preserves_stable_identity_purpose_schemas_and_effects(self) -> None:
        contract = create_contract(effect_profile=EffectProfile.IDEMPOTENT)

        self.assertEqual(contract.tool_id, "math.add")
        self.assertEqual(contract.version, "2")
        self.assertEqual(contract.purpose, "  Add two bounded integers.\n")
        self.assertEqual(contract.descriptor.kind, CapabilityKind.TOOL)
        self.assertEqual(contract.effect_profile, EffectProfile.IDEMPOTENT)
        self.assertEqual(
            contract.input_schema.schema_id,
            "example.math.add.input.v1",
        )
        self.assertEqual(
            contract.output_schema.schema_id,
            "example.math.add.output.v1",
        )

    def test_rejects_invalid_contract_configuration(self) -> None:
        descriptor = CapabilityDescriptor(
            capability_id="math.add",
            version="1",
            kind=CapabilityKind.TOOL,
        )
        values = {
            "descriptor": descriptor,
            "purpose": "Add values.",
            "effect_profile": EffectProfile.READ_ONLY,
            "input_schema": create_input_schema(),
            "output_schema": create_output_schema(),
        }
        for name, value, error_type in (
            ("descriptor", "invalid", TypeError),
            ("purpose", " \n", ValueError),
            ("effect_profile", "invalid", TypeError),
            ("input_schema", "invalid", TypeError),
            ("output_schema", "invalid", TypeError),
        ):
            with self.subTest(field=name):
                invalid = {**values, name: value}
                with self.assertRaises(error_type):
                    ToolContract(**invalid)  # type: ignore[arg-type]

        with self.assertRaises(ValueError):
            ToolContract(
                descriptor=CapabilityDescriptor(
                    capability_id="math.add",
                    version="1",
                    kind=CapabilityKind.SEARCH,
                ),
                purpose="Add values.",
                effect_profile=EffectProfile.READ_ONLY,
                input_schema=create_input_schema(),
                output_schema=create_output_schema(),
            )


class ToolInvocationTest(unittest.TestCase):
    def test_encodes_private_input_against_the_exact_contract(self) -> None:
        contract = create_contract()
        invocation = create_invocation(contract)

        self.assertEqual(invocation.invocation_id, "invocation-1")
        self.assertIs(invocation.contract, contract)
        self.assertEqual(invocation.input, AddInput(left=2, right=3))
        self.assertEqual(invocation.encoded_input, {"left": 2, "right": 3})
        self.assertNotIn("AddInput", repr(invocation))
        self.assertNotIn("left", repr(invocation))
        if not isinstance(invocation.encoded_input, Mapping):
            raise AssertionError("encoded invocation input must be an object")
        with self.assertRaises(TypeError):
            invocation.encoded_input["left"] = 10  # type: ignore[index]

    def test_rejects_invalid_identity_contract_or_input(self) -> None:
        contract = create_contract()
        with self.assertRaises(ValueError):
            ToolInvocation(
                invocation_id=" padded ",
                contract=contract,
                input=AddInput(left=1, right=2),
            )
        with self.assertRaises(TypeError):
            ToolInvocation(
                invocation_id="invocation",
                contract="invalid",  # type: ignore[arg-type]
                input=AddInput(left=1, right=2),
            )
        with self.assertRaises(AttributeError):
            ToolInvocation(
                invocation_id="invocation",
                contract=contract,
                input="invalid",  # type: ignore[arg-type]
            )


class ToolResultTest(unittest.TestCase):
    def test_success_derives_identity_decodes_output_and_preserves_artifacts(
        self,
    ) -> None:
        invocation = create_invocation()
        artifact = create_artifact()
        encoded: dict[str, JsonValue] = {"total": 5, "trace": ["safe"]}

        result = ToolResult.succeeded(
            invocation=invocation,
            encoded_output=encoded,
            artifacts=(artifact,),
        )
        encoded["total"] = 100
        trace = encoded["trace"]
        if not isinstance(trace, list):
            raise AssertionError("test trace must be a list")
        trace.append("changed")

        self.assertEqual(result.status, ExecutionStatus.SUCCEEDED)
        self.assertEqual(result.unwrap(), AddOutput(total=5))
        self.assertEqual(result.invocation_id, "invocation-1")
        self.assertEqual(result.tool_id, "math.add")
        self.assertEqual(result.tool_version, "2")
        self.assertEqual(
            result.output_schema_id,
            "example.math.add.output.v1",
        )
        self.assertEqual(
            result.encoded_output,
            {"total": 5, "trace": ("safe",)},
        )
        self.assertEqual(result.artifacts, (artifact,))
        self.assertNotIn("total", repr(result))

    def test_json_null_is_a_valid_successful_output(self) -> None:
        def encode_none(value: None) -> JsonValue:
            return value

        def decode_none(value: JsonValue) -> None:
            if value is not None:
                raise ValueError("output must be null")
            return None

        contract = ToolContract(
            descriptor=CapabilityDescriptor(
                capability_id="example.noop",
                version="1",
                kind=CapabilityKind.TOOL,
            ),
            purpose="Complete one no-op call.",
            effect_profile=EffectProfile.READ_ONLY,
            input_schema=create_input_schema(),
            output_schema=ToolSchema(
                schema_id="example.noop.output.v1",
                json_schema={"type": "null"},
                encoder=encode_none,
                decoder=decode_none,
            ),
        )
        invocation = ToolInvocation(
            invocation_id="noop-1",
            contract=contract,
            input=AddInput(left=0, right=0),
        )

        result = ToolResult.succeeded(
            invocation=invocation,
            encoded_output=None,
        )

        self.assertEqual(result.status, ExecutionStatus.SUCCEEDED)
        self.assertIsNone(result.encoded_output)
        self.assertIsNone(result.unwrap())

    def test_failure_preserves_normalized_details_artifacts_and_unwrap(self) -> None:
        invocation = create_invocation()
        artifact = create_artifact()
        failure = Failure(
            kind=FailureKind.INVALID_INPUT,
            message="tool input violates its domain constraints",
            code="tool.invalid_domain_input",
            metadata={"tool_id": "math.add"},
        )

        result = ToolResult.from_failure(
            invocation=invocation,
            failure=failure,
            artifacts=(artifact,),
        )

        self.assertEqual(result.status, ExecutionStatus.FAILED)
        self.assertIs(result.failure, failure)
        self.assertIsNone(result.encoded_output)
        self.assertEqual(result.artifacts, (artifact,))
        with self.assertRaises(AgentRigError) as raised:
            result.unwrap()
        self.assertIs(raised.exception.failure, failure)

    def test_rejects_ambiguous_or_invalid_results(self) -> None:
        invocation = create_invocation()
        failure = Failure(
            kind=FailureKind.UNEXPECTED,
            message="tool failed",
        )
        with self.assertRaises(ValueError):
            ToolResult(invocation=invocation)
        with self.assertRaises(ValueError):
            ToolResult(
                invocation=invocation,
                encoded_output={"total": 5},
                failure=failure,
            )
        with self.assertRaises(TypeError):
            ToolResult.succeeded(
                invocation="invalid",  # type: ignore[arg-type]
                encoded_output={"total": 5},
            )
        with self.assertRaises(TypeError):
            ToolResult.from_failure(
                invocation=invocation,
                failure="invalid",  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            ToolResult.succeeded(
                invocation=invocation,
                encoded_output={"total": "invalid"},
            )


class ToolProtocolTest(unittest.TestCase):
    def test_fake_tool_runs_typed_success_and_expected_failure_in_order(
        self,
    ) -> None:
        failure = Failure(
            kind=FailureKind.TRANSIENT_PROVIDER,
            message="tool dependency temporarily unavailable",
            code="tool.dependency_unavailable",
        )
        contract = create_contract(effect_profile=EffectProfile.IDEMPOTENT)
        tool = ScriptedAddTool(
            contract=contract,
            outcomes=({"total": 5}, failure),
        )
        context = create_context()

        first = asyncio.run(
            invoke_typed(tool, create_invocation(contract), context)
        )
        second = asyncio.run(
            invoke_typed(tool, create_invocation(contract), context)
        )

        self.assertIsInstance(tool, Tool)
        self.assertEqual(first.unwrap(), AddOutput(total=5))
        self.assertIs(second.failure, failure)
        self.assertEqual(tool.contract.effect_profile, EffectProfile.IDEMPOTENT)
        self.assertEqual(len(tool.calls), 2)

    def test_fake_rejects_mismatched_contract_and_cancellation_before_call(
        self,
    ) -> None:
        contract = create_contract()
        tool = ScriptedAddTool(contract=contract, outcomes=({"total": 5},))
        with self.assertRaises(ValueError):
            asyncio.run(
                tool.invoke(
                    create_invocation(create_contract(version="3")),
                    create_context(),
                )
            )
        self.assertEqual(tool.calls, [])

        source = CancellationSource()
        source.cancel("caller stopped")
        with self.assertRaises(RunCancelled):
            asyncio.run(
                tool.invoke(
                    create_invocation(contract),
                    create_context(source),
                )
            )
        self.assertEqual(tool.calls, [])


if __name__ == "__main__":
    unittest.main()
