"""Deterministic cases for the provider-neutral scripted agent runtime."""

from agentrig.evals import EvalCase, EvalDataset

SCRIPTED_AGENT_DATASET = EvalDataset[str](
    dataset_id="scripted-agent-quality",
    version="2026-08-13",
    cases=(
        EvalCase(
            case_id="scripted.alpha",
            version="1",
            input="private prompt alpha",
            expected_constraints=("Return the approved alpha response.",),
            prohibited_behaviors=("Expose the private input in reports.",),
            metadata={
                "expected_output": "ALPHA",
                "private_fields": ["input"],
            },
        ),
        EvalCase(
            case_id="scripted.beta",
            version="1",
            input="private prompt beta",
            expected_constraints=("Return the approved beta response.",),
            prohibited_behaviors=("Expose the private input in reports.",),
            metadata={
                "expected_output": "BETA",
                "private_fields": ["input"],
            },
        ),
    ),
    metadata={"execution": "offline", "owner": "agentrig"},
)
