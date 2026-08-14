"""Provider-neutral release-note evaluation and promotion suite."""

from __future__ import annotations

from dataclasses import dataclass, field

from agentrig.core import (
    Grade,
    GradeClassification,
    GradeEvidence,
    GradeStatus,
    GraderDescriptor,
    GradingContext,
    ScoreRange,
)
from agentrig.evals import (
    DeterministicPromotionPolicy,
    EvalBaseline,
    EvalCase,
    EvalComparison,
    EvalDataset,
    EvalReport,
    EvalRunResult,
    EvalRunner,
    EvalSubject,
    EvalTarget,
    PromotionDecision,
    PromotionPolicyDescriptor,
    compare_to_baseline,
)

DATASET_VERSION = "2026-08-14"


def _require_content(field_name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must contain non-whitespace text")
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class ReleaseRequest:
    """One stable case identity with private draft notes."""

    release_id: str
    draft_notes: str = field(repr=False)

    def __post_init__(self) -> None:
        _require_content("release ID", self.release_id)
        _require_content("release draft notes", self.draft_notes)


@dataclass(frozen=True, slots=True, kw_only=True)
class ReleaseSummary:
    """Private target output evaluated by the deterministic grader."""

    text: str = field(repr=False)

    def __post_init__(self) -> None:
        _require_content("release summary text", self.text)


RELEASE_DATASET = EvalDataset[ReleaseRequest](
    dataset_id="release-note-quality",
    version=DATASET_VERSION,
    cases=(
        EvalCase(
            case_id="release.alpha",
            version="1",
            input=ReleaseRequest(
                release_id="alpha",
                draft_notes="private roadmap notes for the alpha release",
            ),
            expected_constraints=(
                "Explain typed workflow composition.",
                "Mention portable contracts.",
            ),
            prohibited_behaviors=("Expose private draft notes.",),
            metadata={"required_terms": ["typed", "workflow", "portable"]},
        ),
        EvalCase(
            case_id="release.beta",
            version="1",
            input=ReleaseRequest(
                release_id="beta",
                draft_notes="private launch notes for the beta release",
            ),
            expected_constraints=(
                "Explain baseline comparison.",
                "Mention regression gates.",
            ),
            prohibited_behaviors=("Expose private draft notes.",),
            metadata={"required_terms": ["baseline", "regression", "gate"]},
        ),
    ),
    metadata={"execution": "offline", "purpose": "release-gate"},
)

RELEASE_GRADER = GraderDescriptor(
    grader_id="release-note.required-terms",
    version="1",
)

RELEASE_PROMOTION_POLICY = DeterministicPromotionPolicy(
    descriptor=PromotionPolicyDescriptor(
        policy_id="release-note.promotion",
        version="1",
    )
)


@dataclass(frozen=True, slots=True)
class RequiredTermsGrader:
    """Score only the case's declared required terms."""

    descriptor: GraderDescriptor = RELEASE_GRADER

    async def grade(
        self,
        subject: EvalSubject[ReleaseRequest, ReleaseSummary],
        context: GradingContext,
    ) -> Grade:
        del context
        required_terms = _required_terms(subject)
        normalized_output = subject.output.text.casefold()
        matched = sum(
            term.casefold() in normalized_output for term in required_terms
        )
        passed = matched == len(required_terms)
        return Grade(
            grader=self.descriptor,
            metric="required_terms",
            status=GradeStatus.PASS if passed else GradeStatus.FAILURE,
            classification=GradeClassification.HARD,
            explanation=(
                "All declared terms were present."
                if passed
                else "One or more declared terms were absent."
            ),
            evidence=(GradeEvidence(field_path=("text",)),),
            score=matched / len(required_terms),
            score_range=ScoreRange(minimum=0.0, maximum=1.0),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class RegressionGateAssessment:
    """Structured comparison evidence and its deterministic release decision."""

    comparison: EvalComparison
    decision: PromotionDecision

    def __post_init__(self) -> None:
        if not isinstance(self.comparison, EvalComparison):
            raise TypeError("gate assessment comparison must be an EvalComparison")
        if not isinstance(self.decision, PromotionDecision):
            raise TypeError("gate assessment decision must be a PromotionDecision")


def create_eval_runner(
    target: EvalTarget[ReleaseRequest, ReleaseSummary],
) -> EvalRunner[ReleaseRequest, ReleaseSummary]:
    """Bind the versioned dataset's grader to an injected target."""
    return EvalRunner(
        target=target,
        graders=(RequiredTermsGrader(),),
    )


def create_private_report(
    run: EvalRunResult[ReleaseSummary],
) -> EvalReport:
    """Create a durable report with the default payload-retention policy."""
    return EvalReport.from_run(
        run,
        environment={
            "mode": "offline",
            "api_key": "example-private-environment-value",
        },
    )


def approve_baseline(report: EvalReport) -> EvalBaseline:
    """Create the payload-free snapshot reviewed for future comparisons."""
    return EvalBaseline.from_report(
        baseline_id="release-note.release",
        version=DATASET_VERSION,
        report=report,
    )


def assess_candidate(
    *,
    baseline: EvalBaseline,
    candidate: EvalReport,
) -> RegressionGateAssessment:
    """Compare the candidate and reduce the evidence to one decision."""
    comparison = compare_to_baseline(baseline, candidate)
    return RegressionGateAssessment(
        comparison=comparison,
        decision=RELEASE_PROMOTION_POLICY.decide(comparison),
    )


def _required_terms(
    subject: EvalSubject[ReleaseRequest, ReleaseSummary],
) -> tuple[str, ...]:
    encoded_terms = subject.case.metadata.get("required_terms")
    if not isinstance(encoded_terms, (list, tuple)) or not encoded_terms:
        raise ValueError("release eval case requires declared terms")
    terms: list[str] = []
    for term in encoded_terms:
        terms.append(_require_content("release required term", term))
    if len(terms) != len(set(terms)):
        raise ValueError("release required terms must be unique")
    return tuple(terms)
