from __future__ import annotations

import asyncio
import unittest
from dataclasses import dataclass
from datetime import UTC, datetime

from agentrig.agents import (
    Agent,
    AgentContract,
    AgentLimits,
    AgentResult,
    AgentStatus,
)
from agentrig.core import (
    AgentRigError,
    ArtifactId,
    ArtifactRef,
    CancellationSource,
    EffectProfile,
    Failure,
    FailureKind,
    RunContext,
    RunId,
)


@dataclass(frozen=True)
class FixedClock:
    def now(self) -> datetime:
        return datetime(2026, 8, 13, 19, 0, tzinfo=UTC)

    def monotonic(self) -> float:
        return 100.0


@dataclass
class SequentialRunIdGenerator:
    next_value: int = 1

    def generate(self) -> RunId:
        run_id = RunId(f"run-{self.next_value}")
        self.next_value += 1
        return run_id


def create_context() -> RunContext:
    source = CancellationSource()
    return RunContext.create_root(
        clock=FixedClock(),
        id_generator=SequentialRunIdGenerator(),
        cancellation=source.token,
    )


def create_contract() -> AgentContract[str, int]:
    return AgentContract(
        agent_id="text.length",
        version="1",
        purpose="Measure text length",
        input_schema="example.text.v1",
        output_schema="example.length.v1",
        prompt_version="prompt-3",
        effect_profile=EffectProfile.READ_ONLY,
        limits=AgentLimits(max_turns=2, max_tool_calls=1),
        stopping_policy="output_schema_satisfied",
        allowed_tools=("text.inspect",),
        allowed_capabilities=("structured_generation",),
        permissions={"workspace": "read_only"},
    )


def create_artifact(artifact_id: str = "artifact-1") -> ArtifactRef:
    return ArtifactRef(
        artifact_id=ArtifactId(artifact_id),
        kind="report",
        media_type="text/plain",
        producer_run_id=RunId("run-1"),
        workspace_path=f"outputs/{artifact_id}.txt",
    )


@dataclass(frozen=True)
class TextLengthAgent:
    contract: AgentContract[str, int]

    async def run(
        self,
        input: str,
        context: RunContext,
    ) -> AgentResult[int]:
        context.cancellation.raise_if_cancelled()
        return AgentResult.succeeded(len(input))


async def run_typed_agent(
    agent: Agent[str, int],
    input: str,
    context: RunContext,
) -> int:
    return (await agent.run(input, context)).unwrap()


class AgentStatusTest(unittest.TestCase):
    def test_vocabulary_has_stable_wire_values(self) -> None:
        self.assertEqual(
            {status.value for status in AgentStatus},
            {"blocked", "cancelled", "failed", "succeeded"},
        )


class AgentLimitsTest(unittest.TestCase):
    def test_requires_explicit_valid_bounds(self) -> None:
        for max_turns in (True, 0, -1, 1.5):
            with self.subTest(max_turns=max_turns):
                with self.assertRaises(ValueError):
                    AgentLimits(  # type: ignore[arg-type]
                        max_turns=max_turns,
                        max_tool_calls=0,
                    )

        for max_tool_calls in (True, -1, 1.5):
            with self.subTest(max_tool_calls=max_tool_calls):
                with self.assertRaises(ValueError):
                    AgentLimits(  # type: ignore[arg-type]
                        max_turns=1,
                        max_tool_calls=max_tool_calls,
                    )


class AgentContractTest(unittest.TestCase):
    def test_preserves_typed_identity_authority_and_limits(self) -> None:
        permissions = {"workspace": "read_only"}
        contract = AgentContract[str, int](
            agent_id="text.length",
            version="1",
            purpose="Measure text length",
            input_schema="example.text.v1",
            output_schema="example.length.v1",
            prompt_version="prompt-3",
            effect_profile=EffectProfile.READ_ONLY,
            limits=AgentLimits(max_turns=2, max_tool_calls=1),
            stopping_policy="output_schema_satisfied",
            allowed_tools=("text.inspect",),
            allowed_capabilities=("structured_generation",),
            permissions=permissions,
        )
        permissions["workspace"] = "read_write"

        self.assertEqual(contract.agent_id, "text.length")
        self.assertEqual(contract.allowed_tools, ("text.inspect",))
        self.assertEqual(
            contract.allowed_capabilities,
            ("structured_generation",),
        )
        self.assertEqual(contract.permissions["workspace"], "read_only")
        self.assertEqual(contract.limits.max_turns, 2)
        with self.assertRaises(TypeError):
            contract.permissions["network"] = "none"  # type: ignore[index]

    def test_rejects_invalid_identity_authority_and_policy(self) -> None:
        values: tuple[dict[str, object], ...] = (
            {"agent_id": " padded"},
            {"version": ""},
            {"purpose": "purpose "},
            {"input_schema": ""},
            {"output_schema": " output"},
            {"prompt_version": ""},
            {"stopping_policy": " padded"},
            {"allowed_tools": ("duplicate", "duplicate")},
            {"allowed_capabilities": ("",)},
            {"permissions": {"": "value"}},
        )
        for overrides in values:
            with self.subTest(overrides=overrides):
                base: dict[str, object] = {
                    "agent_id": "agent",
                    "version": "1",
                    "purpose": "Purpose",
                    "input_schema": "input.v1",
                    "output_schema": "output.v1",
                    "prompt_version": "prompt-1",
                    "effect_profile": EffectProfile.READ_ONLY,
                    "limits": AgentLimits(max_turns=1, max_tool_calls=0),
                    "stopping_policy": "complete",
                }
                with self.assertRaises(ValueError):
                    AgentContract(**(base | overrides))  # type: ignore[arg-type]

        with self.assertRaises(TypeError):
            AgentContract(
                agent_id="agent",
                version="1",
                purpose="Purpose",
                input_schema="input.v1",
                output_schema="output.v1",
                prompt_version="prompt-1",
                effect_profile="read_only",  # type: ignore[arg-type]
                limits=AgentLimits(max_turns=1, max_tool_calls=0),
                stopping_policy="complete",
            )


class AgentResultTest(unittest.TestCase):
    def test_success_preserves_output_and_artifacts(self) -> None:
        artifacts = [create_artifact()]

        result = AgentResult.succeeded("complete", artifacts=artifacts)
        artifacts.append(create_artifact("artifact-2"))

        self.assertEqual(result.status, AgentStatus.SUCCEEDED)
        self.assertTrue(result.is_success)
        self.assertEqual(result.unwrap(), "complete")
        self.assertEqual(result.artifacts, (create_artifact(),))

    def test_failures_map_to_terminal_status_and_unwrap(self) -> None:
        for kind in FailureKind:
            with self.subTest(kind=kind):
                failure = Failure(kind=kind, message=f"safe {kind.value}")
                result: AgentResult[str] = AgentResult.from_failure(failure)
                if kind is FailureKind.CANCELLED:
                    expected = AgentStatus.CANCELLED
                elif kind in (
                    FailureKind.APPROVAL_REQUIRED,
                    FailureKind.WORKFLOW_BLOCKED,
                ):
                    expected = AgentStatus.BLOCKED
                else:
                    expected = AgentStatus.FAILED

                self.assertEqual(result.status, expected)
                with self.assertRaises(AgentRigError) as raised:
                    result.unwrap()
                self.assertIs(raised.exception.failure, failure)

    def test_rejects_impossible_states_and_invalid_artifacts(self) -> None:
        failure = Failure(
            kind=FailureKind.UNEXPECTED,
            message="unexpected implementation failure",
        )
        invalid_values = (
            {
                "status": AgentStatus.SUCCEEDED,
                "output": "value",
                "failure": failure,
            },
            {"status": AgentStatus.FAILED},
            {
                "status": AgentStatus.FAILED,
                "output": "partial",
                "failure": failure,
            },
            {"status": AgentStatus.BLOCKED, "failure": failure},
        )
        for values in invalid_values:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    AgentResult(**values)  # type: ignore[arg-type]

        artifact = create_artifact()
        with self.assertRaises(ValueError):
            AgentResult.succeeded(
                "value",
                artifacts=(artifact, artifact),
            )
        with self.assertRaises(TypeError):
            AgentResult.succeeded(
                "value",
                artifacts=("not-an-artifact",),  # type: ignore[arg-type]
            )


class AgentProtocolTest(unittest.TestCase):
    def test_protocol_supports_a_typed_async_agent(self) -> None:
        agent = TextLengthAgent(create_contract())

        output = asyncio.run(run_typed_agent(agent, "draft", create_context()))

        self.assertIsInstance(agent, Agent)
        self.assertEqual(output, 5)


if __name__ == "__main__":
    unittest.main()
