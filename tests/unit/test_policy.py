from __future__ import annotations

import json
import math
import unittest
from collections.abc import Sequence
from dataclasses import dataclass

from agentrig.core import (
    CompositeGradePolicy,
    Grade,
    GradeClassification,
    GradeDecision,
    GradePolicy,
    GradePolicyDescriptor,
    GradeReference,
    GradeStatus,
    GradeThreshold,
    GraderDescriptor,
    ScoreRange,
    ThresholdGradePolicy,
)


STRUCTURE = GraderDescriptor(grader_id="story.structure", version="1")
QUALITY = GraderDescriptor(grader_id="story.quality", version="2")


def create_grade(
    *,
    grader: GraderDescriptor = STRUCTURE,
    metric: str = "required_sections",
    status: GradeStatus = GradeStatus.PASS,
    classification: GradeClassification = GradeClassification.HARD,
    score: float | None = None,
) -> Grade:
    return Grade(
        grader=grader,
        metric=metric,
        status=status,
        classification=classification,
        explanation="Stable test explanation.",
        score=score,
        score_range=(
            ScoreRange(minimum=0, maximum=1) if score is not None else None
        ),
    )


def quality_reference() -> GradeReference:
    return GradeReference(
        grader_id=QUALITY.grader_id,
        grader_version=QUALITY.version,
        metric="coherence",
    )


def create_threshold_policy(**overrides: object) -> ThresholdGradePolicy:
    values: dict[str, object] = {
        "descriptor": GradePolicyDescriptor(
            policy_id="story.release",
            version="1",
        ),
        "thresholds": (
            GradeThreshold(
                grade=quality_reference(),
                minimum_score=0.8,
            ),
        ),
    }
    values.update(overrides)
    return ThresholdGradePolicy(**values)  # type: ignore[arg-type]


@dataclass(frozen=True)
class ConstantPolicy:
    descriptor: GradePolicyDescriptor
    decision: GradeDecision

    def decide(self, grades: Sequence[Grade]) -> GradeDecision:
        del grades
        return self.decision


class InvalidPolicy:
    @property
    def descriptor(self) -> GradePolicyDescriptor:
        return GradePolicyDescriptor(policy_id="invalid", version="1")

    def decide(self, grades: Sequence[Grade]) -> str:
        del grades
        return "block"


def decide_through_protocol(
    policy: GradePolicy,
    grades: tuple[Grade, ...],
) -> GradeDecision:
    return policy.decide(grades)


class GradeDecisionTest(unittest.TestCase):
    def test_vocabulary_is_stable_and_json_serializable(self) -> None:
        self.assertEqual(
            {decision.value for decision in GradeDecision},
            {
                "block",
                "continue",
                "continue_with_warning",
                "repair",
                "request_approval",
            },
        )
        serialized = json.dumps(GradeDecision.REQUEST_APPROVAL)

        self.assertEqual(
            GradeDecision(json.loads(serialized)),
            GradeDecision.REQUEST_APPROVAL,
        )


class ThresholdGradePolicyTest(unittest.TestCase):
    def test_high_soft_score_continues(self) -> None:
        policy = create_threshold_policy()
        grade = create_grade(
            grader=QUALITY,
            metric="coherence",
            classification=GradeClassification.SOFT,
            score=0.9,
        )

        decision = decide_through_protocol(policy, (grade,))

        self.assertIsInstance(policy, GradePolicy)
        self.assertEqual(decision, GradeDecision.CONTINUE)

    def test_low_soft_score_requires_repair(self) -> None:
        policy = create_threshold_policy()
        grade = create_grade(
            grader=QUALITY,
            metric="coherence",
            classification=GradeClassification.SOFT,
            score=0.7,
        )

        self.assertEqual(policy.decide((grade,)), GradeDecision.REPAIR)

    def test_hard_failure_cannot_be_offset_by_a_high_soft_score(self) -> None:
        policy = create_threshold_policy()
        hard_failure = create_grade(status=GradeStatus.FAILURE)
        high_soft_score = create_grade(
            grader=QUALITY,
            metric="coherence",
            classification=GradeClassification.SOFT,
            score=1.0,
        )

        first = policy.decide((hard_failure, high_soft_score))
        reversed_order = policy.decide((high_soft_score, hard_failure))

        self.assertEqual(first, GradeDecision.BLOCK)
        self.assertEqual(reversed_order, first)

    def test_statuses_map_to_warning_repair_and_block(self) -> None:
        policy = create_threshold_policy()
        passing_score = create_grade(
            grader=QUALITY,
            metric="coherence",
            classification=GradeClassification.SOFT,
            score=0.9,
        )

        warning = create_grade(
            grader=STRUCTURE,
            status=GradeStatus.WARNING,
        )
        soft_failure = create_grade(
            grader=STRUCTURE,
            status=GradeStatus.FAILURE,
            classification=GradeClassification.SOFT,
        )
        hard_failure = create_grade(
            grader=STRUCTURE,
            status=GradeStatus.FAILURE,
        )

        self.assertEqual(
            policy.decide((passing_score, warning)),
            GradeDecision.CONTINUE_WITH_WARNING,
        )
        self.assertEqual(
            policy.decide((passing_score, soft_failure)),
            GradeDecision.REPAIR,
        )
        self.assertEqual(
            policy.decide((passing_score, hard_failure)),
            GradeDecision.BLOCK,
        )

    def test_rule_and_policy_actions_are_configurable(self) -> None:
        approval_policy = create_threshold_policy(
            thresholds=(
                GradeThreshold(
                    grade=quality_reference(),
                    minimum_score=0.8,
                    failure_decision=GradeDecision.REQUEST_APPROVAL,
                ),
            )
        )
        low_score = create_grade(
            grader=QUALITY,
            metric="coherence",
            classification=GradeClassification.SOFT,
            score=0.5,
        )

        self.assertEqual(
            approval_policy.decide((low_score,)),
            GradeDecision.REQUEST_APPROVAL,
        )

        warning_policy = create_threshold_policy(
            soft_failure_decision=GradeDecision.CONTINUE_WITH_WARNING
        )
        soft_failure = create_grade(
            grader=STRUCTURE,
            status=GradeStatus.FAILURE,
            classification=GradeClassification.SOFT,
        )
        self.assertEqual(
            warning_policy.decide((low_score, soft_failure)),
            GradeDecision.REPAIR,
        )

    def test_missing_or_unscored_required_grade_blocks(self) -> None:
        policy = create_threshold_policy()

        self.assertEqual(policy.decide(()), GradeDecision.BLOCK)
        self.assertEqual(
            policy.decide(
                (
                    create_grade(
                        grader=QUALITY,
                        metric="coherence",
                        classification=GradeClassification.SOFT,
                    ),
                )
            ),
            GradeDecision.BLOCK,
        )

    def test_invalid_configurations_and_grade_sets_are_rejected(self) -> None:
        threshold = GradeThreshold(
            grade=quality_reference(),
            minimum_score=0.8,
        )
        with self.assertRaises(ValueError):
            create_threshold_policy(thresholds=())
        with self.assertRaises(ValueError):
            create_threshold_policy(thresholds=(threshold, threshold))
        with self.assertRaises(ValueError):
            GradeThreshold(
                grade=quality_reference(),
                minimum_score=math.nan,
            )
        with self.assertRaises(ValueError):
            create_threshold_policy(
                hard_failure_decision=GradeDecision.CONTINUE
            )

        grade = create_grade(
            grader=QUALITY,
            metric="coherence",
            classification=GradeClassification.SOFT,
            score=0.9,
        )
        with self.assertRaises(ValueError):
            create_threshold_policy().decide((grade, grade))
        with self.assertRaises(TypeError):
            create_threshold_policy().decide(("not-a-grade",))  # type: ignore[arg-type]

    def test_threshold_must_fit_the_grade_calibration(self) -> None:
        policy = create_threshold_policy(
            thresholds=(
                GradeThreshold(
                    grade=quality_reference(),
                    minimum_score=2,
                ),
            )
        )
        grade = create_grade(
            grader=QUALITY,
            metric="coherence",
            classification=GradeClassification.SOFT,
            score=0.9,
        )

        with self.assertRaises(ValueError):
            policy.decide((grade,))


class CompositeGradePolicyTest(unittest.TestCase):
    def create_constant(
        self,
        policy_id: str,
        decision: GradeDecision,
    ) -> ConstantPolicy:
        return ConstantPolicy(
            descriptor=GradePolicyDescriptor(policy_id=policy_id, version="1"),
            decision=decision,
        )

    def test_selects_the_strongest_decision_independent_of_policy_order(self) -> None:
        warning = self.create_constant(
            "warning",
            GradeDecision.CONTINUE_WITH_WARNING,
        )
        approval = self.create_constant(
            "approval",
            GradeDecision.REQUEST_APPROVAL,
        )
        repair = self.create_constant("repair", GradeDecision.REPAIR)
        descriptor = GradePolicyDescriptor(policy_id="composite", version="1")

        first = CompositeGradePolicy(
            warning,
            approval,
            repair,
            descriptor=descriptor,
        )
        second = CompositeGradePolicy(
            repair,
            warning,
            approval,
            descriptor=descriptor,
        )

        self.assertEqual(first.decide(()), GradeDecision.REQUEST_APPROVAL)
        self.assertEqual(second.decide(()), first.decide(()))

    def test_enforces_hard_failure_against_permissive_children(self) -> None:
        permissive = self.create_constant("permissive", GradeDecision.CONTINUE)
        policy = CompositeGradePolicy(
            permissive,
            descriptor=GradePolicyDescriptor(
                policy_id="hard-protection",
                version="1",
            ),
        )
        hard_failure = create_grade(status=GradeStatus.FAILURE)
        unrelated_high_score = create_grade(
            grader=QUALITY,
            metric="coherence",
            classification=GradeClassification.SOFT,
            score=1.0,
        )

        self.assertEqual(
            policy.decide((unrelated_high_score, hard_failure)),
            GradeDecision.BLOCK,
        )

    def test_rejects_invalid_children_and_return_values(self) -> None:
        descriptor = GradePolicyDescriptor(policy_id="composite", version="1")
        with self.assertRaises(ValueError):
            CompositeGradePolicy(descriptor=descriptor)

        repeated = self.create_constant("same", GradeDecision.CONTINUE)
        with self.assertRaises(ValueError):
            CompositeGradePolicy(repeated, repeated, descriptor=descriptor)

        invalid = InvalidPolicy()
        policy = CompositeGradePolicy(invalid, descriptor=descriptor)
        with self.assertRaises(TypeError):
            policy.decide(())


class PolicyDescriptorTest(unittest.TestCase):
    def test_requires_stable_trimmed_identity(self) -> None:
        with self.assertRaises(ValueError):
            GradePolicyDescriptor(policy_id="", version="1")
        with self.assertRaises(ValueError):
            GradePolicyDescriptor(policy_id="policy", version=" 1")


if __name__ == "__main__":
    unittest.main()
