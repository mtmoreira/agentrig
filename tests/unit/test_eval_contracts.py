from __future__ import annotations

import asyncio
import unittest
from dataclasses import dataclass

from agentrig.agents import AgentContract, AgentLimits, AgentResult
from agentrig.core import (
    ArtifactId,
    ArtifactRef,
    CancellationSource,
    EffectProfile,
    ExecutionOutcome,
    Failure,
    FailureKind,
    RunContext,
    RunId,
    SystemClock,
    Uuid4IdGenerator,
)
from agentrig.evals import (
    AgentEvalTarget,
    EvalCase,
    EvalDataset,
    EvalTarget,
    EvalTargetDescriptor,
    EvalTargetKind,
)


class EvalCaseTest(unittest.TestCase):
    def test_preserves_versioned_expectations_and_fixture_references(self) -> None:
        metadata = {
            "category": "safety",
            "labels": ["scope", "authorization"],
        }
        expected_constraints = ["Remain inside the authorized workspace."]
        fixture_refs = ["fixtures/dirty-worktree.json"]

        case = EvalCase[str](
            case_id="agent.dirty-worktree",
            version="2",
            input="Implement the requested change.",
            expected_constraints=expected_constraints,  # type: ignore[arg-type]
            allowed_variability=("Equivalent file-local implementations.",),
            prohibited_behaviors=("Overwrite existing user changes.",),
            fixture_refs=fixture_refs,  # type: ignore[arg-type]
            metadata=metadata,
        )
        expected_constraints.append("late mutation")
        fixture_refs.append("fixtures/late.json")
        metadata["category"] = "changed"
        labels = metadata["labels"]
        if not isinstance(labels, list):
            raise AssertionError("test metadata labels are not mutable")
        labels.append("late")

        self.assertEqual(case.case_id, "agent.dirty-worktree")
        self.assertEqual(case.version, "2")
        self.assertEqual(case.input, "Implement the requested change.")
        self.assertEqual(
            case.expected_constraints,
            ("Remain inside the authorized workspace.",),
        )
        self.assertEqual(
            case.allowed_variability,
            ("Equivalent file-local implementations.",),
        )
        self.assertEqual(
            case.prohibited_behaviors,
            ("Overwrite existing user changes.",),
        )
        self.assertEqual(case.fixture_refs, ("fixtures/dirty-worktree.json",))
        self.assertEqual(case.metadata["category"], "safety")
        self.assertEqual(case.metadata["labels"], ("scope", "authorization"))
        self.assertNotIn("Implement the requested change", repr(case))
        self.assertNotIn("safety", repr(case))

    def test_rejects_invalid_identity_expectations_and_metadata(self) -> None:
        with self.assertRaises(ValueError):
            EvalCase(case_id=" padded ", version="1", input=None)
        with self.assertRaises(ValueError):
            EvalCase(case_id="case", version="", input=None)
        for field_name in (
            "expected_constraints",
            "allowed_variability",
            "prohibited_behaviors",
            "fixture_refs",
        ):
            with self.subTest(field_name=field_name):
                with self.assertRaises(ValueError):
                    EvalCase(
                        case_id="case",
                        version="1",
                        input=None,
                        **{field_name: ("duplicate", "duplicate")},
                    )
        with self.assertRaises(ValueError):
            EvalCase(
                case_id="case",
                version="1",
                input=None,
                expected_constraints=(" padded ",),
            )
        with self.assertRaises(ValueError):
            EvalCase(
                case_id="case",
                version="1",
                input=None,
                metadata={"invalid": object()},  # type: ignore[dict-item]
            )


class EvalDatasetTest(unittest.TestCase):
    def test_copies_ordered_cases_and_freezes_metadata(self) -> None:
        first = EvalCase(case_id="first", version="1", input="a")
        second = EvalCase(case_id="second", version="3", input="b")
        cases = [first, second]
        metadata = {"suite": {"owner": "quality"}}

        dataset = EvalDataset[str](
            dataset_id="agent-contract",
            version="2026.08.13",
            cases=cases,  # type: ignore[arg-type]
            metadata=metadata,
        )
        cases.reverse()
        metadata["suite"] = {"owner": "changed"}

        self.assertEqual(dataset.dataset_id, "agent-contract")
        self.assertEqual(dataset.version, "2026.08.13")
        self.assertEqual(dataset.cases, (first, second))
        self.assertEqual(dataset.metadata["suite"], {"owner": "quality"})
        self.assertNotIn("quality", repr(dataset))

    def test_rejects_empty_invalid_or_duplicate_case_selections(self) -> None:
        first = EvalCase(case_id="same", version="1", input="a")
        second = EvalCase(case_id="same", version="2", input="b")

        with self.assertRaises(ValueError):
            EvalDataset(dataset_id="dataset", version="1", cases=())
        with self.assertRaises(TypeError):
            EvalDataset(
                dataset_id="dataset",
                version="1",
                cases=("invalid",),  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            EvalDataset(
                dataset_id="dataset",
                version="1",
                cases=(first, second),
            )
        with self.assertRaises(ValueError):
            EvalDataset(dataset_id=" padded ", version="1", cases=(first,))
        with self.assertRaises(ValueError):
            EvalDataset(dataset_id="dataset", version="", cases=(first,))


@dataclass(frozen=True)
class EchoTarget:
    descriptor: EvalTargetDescriptor

    async def run(
        self,
        input: str,
        context: RunContext,
    ) -> ExecutionOutcome[str]:
        del context
        return ExecutionOutcome.succeeded(input)


class EvalTargetTest(unittest.TestCase):
    def test_protocol_preserves_typed_target_identity(self) -> None:
        descriptor = EvalTargetDescriptor(
            target_id="echo",
            version="4",
            kind=EvalTargetKind.INTEGRATION,
        )
        target = EchoTarget(descriptor=descriptor)

        self.assertIsInstance(target, EvalTarget)
        self.assertIs(target.descriptor, descriptor)

    def test_vocabulary_and_descriptor_validation_are_stable(self) -> None:
        self.assertEqual(
            tuple(kind.value for kind in EvalTargetKind),
            ("agent", "workflow", "capability", "integration"),
        )
        with self.assertRaises(ValueError):
            EvalTargetDescriptor(
                target_id="",
                version="1",
                kind=EvalTargetKind.AGENT,
            )
        with self.assertRaises(ValueError):
            EvalTargetDescriptor(
                target_id="target",
                version=" padded ",
                kind=EvalTargetKind.WORKFLOW,
            )
        with self.assertRaises(TypeError):
            EvalTargetDescriptor(
                target_id="target",
                version="1",
                kind="agent",  # type: ignore[arg-type]
            )


def create_agent_contract() -> AgentContract[str, str]:
    return AgentContract(
        agent_id="echo-agent",
        version="2",
        purpose="Echo one string for evaluation",
        input_schema="example.text.v1",
        output_schema="example.text.v1",
        prompt_version="prompt-2",
        effect_profile=EffectProfile.READ_ONLY,
        limits=AgentLimits(max_turns=1, max_tool_calls=0),
        stopping_policy="output_schema_satisfied",
    )


def create_context() -> RunContext:
    return RunContext.create_root(
        clock=SystemClock(),
        id_generator=Uuid4IdGenerator(RunId),
        cancellation=CancellationSource().token,
    )


@dataclass(frozen=True)
class ResultAgent:
    contract: AgentContract[str, str]
    result: AgentResult[str]

    async def run(self, input: str, context: RunContext) -> AgentResult[str]:
        del input, context
        return self.result


class AgentEvalTargetTest(unittest.TestCase):
    def test_derives_identity_and_preserves_success_or_failure(self) -> None:
        artifact = ArtifactRef(
            artifact_id=ArtifactId("artifact-1"),
            kind="report",
            media_type="text/plain",
            producer_run_id=RunId("producer-run"),
            workspace_path="outputs/report.txt",
        )
        success = AgentEvalTarget(
            ResultAgent(
                contract=create_agent_contract(),
                result=AgentResult.succeeded(
                    "complete",
                    artifacts=(artifact,),
                ),
            )
        )

        outcome = asyncio.run(success.run("draft", create_context()))

        self.assertEqual(outcome.unwrap(), "complete")
        self.assertEqual(success.descriptor.target_id, "echo-agent")
        self.assertEqual(success.descriptor.version, "2")
        self.assertEqual(success.descriptor.kind, EvalTargetKind.AGENT)
        self.assertEqual(outcome.artifacts, (artifact,))
        self.assertIsInstance(success, EvalTarget)

        failure = Failure(
            kind=FailureKind.WORKFLOW_BLOCKED,
            message="provider credentials are unavailable",
            code="provider.credentials_unavailable",
        )
        blocked = AgentEvalTarget(
            ResultAgent(
                contract=create_agent_contract(),
                result=AgentResult.from_failure(
                    failure,
                    artifacts=(artifact,),
                ),
            )
        )

        blocked_outcome = asyncio.run(
            blocked.run("draft", create_context())
        )

        self.assertIs(blocked_outcome.failure, failure)
        self.assertEqual(blocked_outcome.artifacts, (artifact,))

    def test_rejects_invalid_agent_contract_context_and_result(self) -> None:
        with self.assertRaises(TypeError):
            AgentEvalTarget("invalid")  # type: ignore[arg-type]

        @dataclass(frozen=True)
        class InvalidContractAgent:
            contract: object = "invalid"

            async def run(self, input: str, context: RunContext) -> object:
                del input, context
                return AgentResult.succeeded("unreachable")

        with self.assertRaises(TypeError):
            AgentEvalTarget(InvalidContractAgent())  # type: ignore[arg-type]

        valid = AgentEvalTarget(
            ResultAgent(
                contract=create_agent_contract(),
                result=AgentResult.succeeded("complete"),
            )
        )
        with self.assertRaises(TypeError):
            asyncio.run(valid.run("draft", "invalid"))  # type: ignore[arg-type]

        @dataclass(frozen=True)
        class InvalidResultAgent:
            contract: AgentContract[str, str]

            async def run(self, input: str, context: RunContext) -> object:
                del input, context
                return "invalid"

        invalid_result = AgentEvalTarget(
            InvalidResultAgent(create_agent_contract())  # type: ignore[arg-type]
        )
        with self.assertRaises(TypeError):
            asyncio.run(invalid_result.run("draft", create_context()))


if __name__ == "__main__":
    unittest.main()
