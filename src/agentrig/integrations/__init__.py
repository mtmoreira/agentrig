"""Optional provider and executable integrations."""

from agentrig.integrations.bounded_process import ProcessFailure, ProcessOutput, ProcessSpec, run_bounded_process
from agentrig.integrations.backend_journal import LocalBackendJournal
from agentrig.integrations.backend_staging import StagedWorkspace, WorkspaceStagePlan, stage_workspace

from agentrig.integrations.local_backends import ExistingRootlessDockerBackend, LimaBackend
from agentrig.integrations.lima_lifecycle import LimaArtifactVerifier, LimaLifecycleBackend

from agentrig.integrations.process import (
    CommandInput,
    CommandOutput,
    CommandTool,
    DetachedCommandOutput,
    DetachedCommandTool,
)

__all__ = (
    "StagedWorkspace",
    "WorkspaceStagePlan",
    "stage_workspace",
    "LimaArtifactVerifier",
    "LimaLifecycleBackend",
    "LocalBackendJournal",
    "ProcessFailure",
    "ProcessOutput",
    "ProcessSpec",
    "run_bounded_process",
    "ExistingRootlessDockerBackend",
    "LimaBackend",
    "CommandInput",
    "CommandOutput",
    "CommandTool",
    "DetachedCommandOutput",
    "DetachedCommandTool",
)
