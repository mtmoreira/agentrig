"""Versioned evaluation contracts and execution infrastructure."""

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
    "EvalCase",
    "EvalCaseResult",
    "EvalCost",
    "EvalDataset",
    "EVAL_REPORT_SCHEMA_VERSION",
    "EvalGraderFailure",
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
)
