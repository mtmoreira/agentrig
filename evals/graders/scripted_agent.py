"""Deterministic quality grader for scripted agent outputs."""

from dataclasses import dataclass

from agentrig.core import (
    Grade,
    GradeClassification,
    GradeStatus,
    GraderDescriptor,
    GradingContext,
    ScoreRange,
)
from agentrig.evals import EvalSubject

SCRIPTED_AGENT_GRADER = GraderDescriptor(
    grader_id="scripted-agent.exact-output",
    version="1",
)


@dataclass(frozen=True, slots=True)
class ScriptedAgentExactOutputGrader:
    """Require the case's declared output without exposing either value."""

    descriptor: GraderDescriptor = SCRIPTED_AGENT_GRADER

    async def grade(
        self,
        subject: EvalSubject[str, str],
        context: GradingContext,
    ) -> Grade:
        del context
        expected = subject.case.metadata.get("expected_output")
        if not isinstance(expected, str):
            raise ValueError("scripted agent case requires expected_output metadata")
        passed = subject.output == expected
        return Grade(
            grader=self.descriptor,
            metric="exact_output",
            status=GradeStatus.PASS if passed else GradeStatus.FAILURE,
            classification=GradeClassification.HARD,
            explanation=(
                "Output matched the declared expectation."
                if passed
                else "Output did not match the declared expectation."
            ),
            score=1.0 if passed else 0.0,
            score_range=ScoreRange(minimum=0.0, maximum=1.0),
        )
