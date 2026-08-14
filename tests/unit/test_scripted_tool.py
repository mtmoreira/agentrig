from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
import unittest

from agentrig.capabilities import (
    CapabilityDescriptor,
    CapabilityKind,
    DataRetention,
    Tool,
    ToolContract,
    ToolInvocation,
    ToolSchema,
)
from agentrig.core import (
    AgentRigError,
    ArtifactId,
    ArtifactRef,
    CancellationSource,
    Deadline,
    DeadlineExceeded,
    EffectProfile,
    ExecutionStatus,
    Failure,
    FailureKind,
    JsonValue,
    RunCancelled,
    RunContext,
    RunId,
)
from agentrig.testing import (
    ScriptedTool,
    ScriptedToolFailure,
    ScriptedToolSuccess,
    ToolContractSuite,
)


@dataclass(frozen=True, slots=True)
class AddInput:
    left: int
    right: int


@dataclass(frozen=True, slots=True)
class AddOutput:
    total: int


@dataclass(frozen=True)
class FixedClock:
    def now(self) -> datetime:
        return datetime(2026, 8, 15, 9, 0, tzinfo=UTC)

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
    *,
    deadline: Deadline | None = None,
) -> RunContext:
    owned_source = source if source is not None else CancellationSource()
    return RunContext.create_root(
        clock=FixedClock(),
        id_generator=SequentialRunIdGenerator(),
        cancellation=owned_source.token,
        deadline=deadline,
    )


def cancelled_context() -> RunContext:
    source = CancellationSource()
    source.cancel("contract cancellation")
    return create_context(source)


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


def create_contract(version: str = "1") -> ToolContract[AddInput, AddOutput]:
    return ToolContract(
        descriptor=CapabilityDescriptor(
            capability_id="math.add",
            version=version,
            kind=CapabilityKind.TOOL,
            data_retention=DataRetention.NOT_RETAINED,
        ),
        purpose="Add two bounded integers.",
        effect_profile=EffectProfile.READ_ONLY,
        input_schema=ToolSchema(
            schema_id="math.add.input.v1",
            json_schema={"type": "object"},
            encoder=encode_input,
            decoder=decode_input,
        ),
        output_schema=ToolSchema(
            schema_id="math.add.output.v1",
            json_schema={"type": "object"},
            encoder=encode_output,
            decoder=decode_output,
        ),
    )


def create_invocation(
    contract: ToolContract[AddInput, AddOutput],
    *,
    invocation_id: str = "invocation-1",
) -> ToolInvocation[AddInput, AddOutput]:
    return ToolInvocation(
        invocation_id=invocation_id,
        contract=contract,
        input=AddInput(left=2, right=3),
    )


def output_artifact() -> ArtifactRef:
    return ArtifactRef(
        artifact_id=ArtifactId("tool-output"),
        kind="tool-output",
        media_type="application/json",
        producer_run_id=RunId("run-provider"),
        workspace_path="outputs/tool.json",
    )


def expected_failure() -> Failure:
    return Failure(
        kind=FailureKind.INVALID_INPUT,
        message="tool input violates its domain constraints",
        code="tool.invalid_domain_input",
    )


class ScriptedToolTest(unittest.TestCase):
    def test_returns_typed_success_and_expected_failure_in_order(self) -> None:
        contract = create_contract()
        artifact = output_artifact()
        failure = expected_failure()
        tool = ScriptedTool[AddInput, AddOutput](
            contract=contract,
            outcomes=(
                ScriptedToolSuccess(
                    encoded_output={"total": 5},
                    artifacts=(artifact,),
                ),
                ScriptedToolFailure(
                    failure=failure,
                    artifacts=(artifact,),
                ),
            ),
        )
        context = create_context()

        success = asyncio.run(
            tool.invoke(create_invocation(contract), context)
        )
        snapshot = tool.calls
        failed = asyncio.run(
            tool.invoke(
                create_invocation(contract, invocation_id="invocation-2"),
                context,
            )
        )

        self.assertIsInstance(tool, Tool)
        self.assertEqual(success.unwrap(), AddOutput(total=5))
        self.assertEqual(success.artifacts, (artifact,))
        self.assertEqual(failed.status, ExecutionStatus.FAILED)
        self.assertIs(failed.failure, failure)
        self.assertEqual(failed.artifacts, (artifact,))
        self.assertEqual(tuple(call.index for call in snapshot), (0,))
        self.assertEqual(tuple(call.index for call in tool.calls), (0, 1))
        self.assertTrue(tool.is_exhausted)

    def test_preserves_json_null_as_a_successful_output(self) -> None:
        def encode_none(value: None) -> JsonValue:
            return value

        def decode_none(value: JsonValue) -> None:
            if value is not None:
                raise ValueError("tool output must be null")
            return None

        contract = ToolContract[AddInput, None](
            descriptor=CapabilityDescriptor(
                capability_id="example.noop",
                version="1",
                kind=CapabilityKind.TOOL,
            ),
            purpose="Complete one no-op call.",
            effect_profile=EffectProfile.READ_ONLY,
            input_schema=ToolSchema(
                schema_id="math.add.input.v1",
                json_schema={"type": "object"},
                encoder=encode_input,
                decoder=decode_input,
            ),
            output_schema=ToolSchema(
                schema_id="example.noop.output.v1",
                json_schema={"type": "null"},
                encoder=encode_none,
                decoder=decode_none,
            ),
        )
        tool = ScriptedTool[AddInput, None](
            contract=contract,
            outcomes=(ScriptedToolSuccess(encoded_output=None),),
        )
        invocation = ToolInvocation(
            invocation_id="noop-1",
            contract=contract,
            input=AddInput(left=0, right=0),
        )

        result = asyncio.run(tool.invoke(invocation, create_context()))

        self.assertEqual(result.status, ExecutionStatus.SUCCEEDED)
        self.assertIsNone(result.encoded_output)
        self.assertIsNone(result.unwrap())

    def test_exact_contract_and_constraints_do_not_consume_outcomes(self) -> None:
        contract = create_contract()
        tool = ScriptedTool[AddInput, AddOutput](
            contract=contract,
            outcomes=(ScriptedToolSuccess(encoded_output={"total": 5}),),
        )

        with self.assertRaisesRegex(ValueError, "does not match"):
            asyncio.run(
                tool.invoke(
                    create_invocation(create_contract(version="2")),
                    create_context(),
                )
            )
        with self.assertRaises(RunCancelled):
            asyncio.run(
                tool.invoke(
                    create_invocation(contract),
                    cancelled_context(),
                )
            )
        expired = Deadline(
            expires_at=datetime(2026, 8, 15, 9, 0, tzinfo=UTC),
            monotonic_deadline=100.0,
        )
        with self.assertRaises(DeadlineExceeded):
            asyncio.run(
                tool.invoke(
                    create_invocation(contract),
                    create_context(deadline=expired),
                )
            )

        self.assertEqual(tool.calls, ())
        self.assertFalse(tool.is_exhausted)

    def test_output_is_decoded_by_each_invoked_contract(self) -> None:
        contract = create_contract()
        tool = ScriptedTool[AddInput, AddOutput](
            contract=contract,
            outcomes=(
                ScriptedToolSuccess(encoded_output={"total": "invalid"}),
            ),
        )

        with self.assertRaisesRegex(ValueError, "must be an integer"):
            asyncio.run(
                tool.invoke(create_invocation(contract), create_context())
            )
        self.assertEqual(len(tool.calls), 1)
        self.assertTrue(tool.is_exhausted)

    def test_exhaustion_is_sanitized_and_repeat_last_is_unbounded(self) -> None:
        contract = create_contract()
        exhausted = ScriptedTool[AddInput, AddOutput](
            contract=contract,
            outcomes=(ScriptedToolSuccess(encoded_output={"total": 5}),),
        )
        asyncio.run(
            exhausted.invoke(create_invocation(contract), create_context())
        )

        with self.assertRaises(AgentRigError) as raised:
            asyncio.run(
                exhausted.invoke(create_invocation(contract), create_context())
            )
        self.assertEqual(
            raised.exception.failure.code,
            "scripted_tool.exhausted",
        )

        repeating = ScriptedTool[AddInput, AddOutput](
            contract=contract,
            outcomes=(ScriptedToolSuccess(encoded_output={"total": 5}),),
            repeat_last=True,
        )
        for index in range(3):
            result = asyncio.run(
                repeating.invoke(
                    create_invocation(
                        contract,
                        invocation_id=f"repeat-{index}",
                    ),
                    create_context(),
                )
            )
            self.assertEqual(result.unwrap(), AddOutput(total=5))
        self.assertFalse(repeating.is_exhausted)


class ToolContractSuiteTest(unittest.TestCase):
    def test_suite_verifies_shared_portable_semantics(self) -> None:
        contract = create_contract()
        tool = ScriptedTool[AddInput, AddOutput](
            contract=contract,
            outcomes=(ScriptedToolSuccess(encoded_output={"total": 5}),),
        )
        suite = ToolContractSuite[AddInput, AddOutput](
            tool=tool,
            supported_invocation=create_invocation(contract),
            unsupported_invocation=create_invocation(
                create_contract(version="2")
            ),
            context=create_context(),
            cancelled_context=cancelled_context(),
            invocation_count=lambda: len(tool.calls),
        )

        result = asyncio.run(suite.verify())

        self.assertEqual(result.unwrap(), AddOutput(total=5))
        self.assertEqual(len(tool.calls), 1)

    def test_suite_accepts_an_expected_failed_tool_result(self) -> None:
        contract = create_contract()
        failure = expected_failure()
        tool = ScriptedTool[AddInput, AddOutput](
            contract=contract,
            outcomes=(ScriptedToolFailure(failure=failure),),
        )
        suite = ToolContractSuite[AddInput, AddOutput](
            tool=tool,
            supported_invocation=create_invocation(contract),
            unsupported_invocation=create_invocation(
                create_contract(version="2")
            ),
            context=create_context(),
            cancelled_context=cancelled_context(),
            invocation_count=lambda: len(tool.calls),
        )

        result = asyncio.run(suite.verify())

        self.assertIs(result.failure, failure)
        self.assertEqual(len(tool.calls), 1)

    def test_rejects_invalid_suite_fixture_configuration(self) -> None:
        contract = create_contract()
        tool = ScriptedTool[AddInput, AddOutput](
            contract=contract,
            outcomes=(ScriptedToolSuccess(encoded_output={"total": 5}),),
        )
        with self.assertRaisesRegex(ValueError, "must already be cancelled"):
            ToolContractSuite(
                tool=tool,
                supported_invocation=create_invocation(contract),
                unsupported_invocation=create_invocation(
                    create_contract(version="2")
                ),
                context=create_context(),
                cancelled_context=create_context(),
                invocation_count=lambda: len(tool.calls),
            )
        with self.assertRaisesRegex(ValueError, "another contract"):
            ToolContractSuite(
                tool=tool,
                supported_invocation=create_invocation(contract),
                unsupported_invocation=create_invocation(contract),
                context=create_context(),
                cancelled_context=cancelled_context(),
                invocation_count=lambda: len(tool.calls),
            )


class ScriptedToolValidationTest(unittest.TestCase):
    def test_rejects_invalid_contract_outcomes_and_scenarios(self) -> None:
        contract = create_contract()
        with self.assertRaises(TypeError):
            ScriptedTool(
                contract="invalid",  # type: ignore[arg-type]
                outcomes=(ScriptedToolSuccess(encoded_output={"total": 5}),),
            )
        with self.assertRaises(ValueError):
            ScriptedTool[AddInput, AddOutput](contract=contract, outcomes=())
        with self.assertRaises(TypeError):
            ScriptedTool[AddInput, AddOutput](
                contract=contract,
                outcomes=("invalid",),  # type: ignore[arg-type]
            )
        with self.assertRaises(TypeError):
            ScriptedToolFailure(
                failure="invalid",  # type: ignore[arg-type]
            )
        artifact = output_artifact()
        with self.assertRaises(ValueError):
            ScriptedToolSuccess(
                encoded_output={"total": 5},
                artifacts=(artifact, artifact),
            )
        with self.assertRaises(TypeError):
            ScriptedTool[AddInput, AddOutput](
                contract=contract,
                outcomes=(ScriptedToolSuccess(encoded_output={"total": 5}),),
                repeat_last=1,  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
