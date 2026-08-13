"""Versioned evaluation contracts and execution infrastructure."""

from agentrig.evals.case import EvalCase
from agentrig.evals.dataset import EvalDataset
from agentrig.evals.target import EvalTarget, EvalTargetDescriptor, EvalTargetKind

__all__ = (
    "EvalCase",
    "EvalDataset",
    "EvalTarget",
    "EvalTargetDescriptor",
    "EvalTargetKind",
)
