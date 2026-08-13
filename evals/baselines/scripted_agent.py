"""Baseline and promotion policy for the scripted-agent eval suite."""

from agentrig.evals import (
    DeterministicPromotionPolicy,
    EvalBaseline,
    EvalReport,
    PromotionPolicyDescriptor,
)

SCRIPTED_AGENT_PROMOTION_POLICY = DeterministicPromotionPolicy(
    descriptor=PromotionPolicyDescriptor(
        policy_id="scripted-agent.promotion",
        version="1",
    )
)


def create_scripted_agent_baseline(report: EvalReport) -> EvalBaseline:
    """Create the reviewed baseline snapshot for this dataset version."""
    return EvalBaseline.from_report(
        baseline_id="scripted-agent.release",
        version="2026-08-13",
        report=report,
    )
