"""Versioned evaluation contracts and execution infrastructure."""

from agentrig.evals.baseline import (
    EVAL_BASELINE_SCHEMA_VERSION,
    DeterministicPromotionPolicy,
    EvalBaseline,
    EvalChangeKind,
    EvalComparison,
    EvalComparisonTolerance,
    EvalInconclusive,
    EvalInconclusiveReason,
    EvalMetric,
    EvalMetricChange,
    PromotionDecision,
    PromotionPolicy,
    PromotionPolicyDescriptor,
    compare_to_baseline,
)
from agentrig.evals.case import EvalCase
from agentrig.evals.dataset import EvalDataset
from agentrig.evals.report import (
    EVAL_REPORT_SCHEMA_VERSION,
    EvalReport,
    EvalReportArtifact,
    EvalReportCase,
    EvalReportGrade,
    EvalReportRetention,
)
from agentrig.evals.runner import (
    EvalCaseResult,
    EvalCost,
    EvalGraderFailure,
    EvalRunResult,
    EvalRunner,
    EvalSubject,
    EvalSummary,
)
from agentrig.evals.target import EvalTarget, EvalTargetDescriptor, EvalTargetKind

__all__ = (
    "DeterministicPromotionPolicy",
    "EVAL_BASELINE_SCHEMA_VERSION",
    "EvalCase",
    "EvalCaseResult",
    "EvalCost",
    "EvalBaseline",
    "EvalChangeKind",
    "EvalComparison",
    "EvalComparisonTolerance",
    "EvalDataset",
    "EVAL_REPORT_SCHEMA_VERSION",
    "EvalGraderFailure",
    "EvalInconclusive",
    "EvalInconclusiveReason",
    "EvalMetric",
    "EvalMetricChange",
    "EvalRunResult",
    "EvalRunner",
    "EvalReport",
    "EvalReportArtifact",
    "EvalReportCase",
    "EvalReportGrade",
    "EvalReportRetention",
    "EvalSubject",
    "EvalSummary",
    "EvalTarget",
    "EvalTargetDescriptor",
    "EvalTargetKind",
    "PromotionDecision",
    "PromotionPolicy",
    "PromotionPolicyDescriptor",
    "compare_to_baseline",
)
