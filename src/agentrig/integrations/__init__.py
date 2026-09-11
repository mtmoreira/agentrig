"""Optional provider and executable integrations."""

from agentrig.integrations.backend_artifacts import audit_backend_artifacts
from agentrig.integrations.bounded_process import ProcessFailure, ProcessOutput, ProcessSpec, run_bounded_process
from agentrig.integrations.backend_journal import LocalBackendJournal
from agentrig.integrations.backend_staging import StagedWorkspace, WorkspaceStagePlan, stage_workspace
from agentrig.integrations.guest_workspace import (
    GuestWorkspaceEndpoint, GuestWorkspaceReceipt, copy_to_guest,
    receive_guest_workspace, transfer_guest_workspace,
)

from agentrig.integrations.local_backends import ExistingRootlessDockerBackend, LimaBackend
from agentrig.integrations.lima_lifecycle import LimaArtifactVerifier, LimaLifecycleBackend
from agentrig.integrations.lima_retirement import (
    LimaGuestServiceGeneration, LimaGuestServiceRetirementAdapter,
    LimaQuiescenceObservation, LimaRetirementObservation,
    bind_lima_guest_service_generation, decode_lima_service_observation,
)
from agentrig.integrations.lima_configuration import LimaConfiguration, LimaConfigurationAudit, audit_lima_configuration
from agentrig.integrations.guest_operations import JournaledGuestWorkspaces, guest_operation_digest, inspect_guest_workspace
from agentrig.integrations.guest_retirement import (
    GuestServiceRetirementAdapter, JournaledGuestServiceRetirement,
    guest_retirement_operation_digest,
)
from agentrig.integrations.guest_service import (
    copy_guest_service, handle_guest_request, inspect_guest_service,
    request_guest_service, serve_guest_once, verify_linux_guest_peer,
    guest_service_retirement_digest, retire_guest_service, serve_guest_requests,
)

from agentrig.integrations.process import (
    CommandInput,
    CommandOutput,
    CommandTool,
    DetachedCommandOutput,
    DetachedCommandTool,
)

__all__ = (
    "GuestServiceRetirementAdapter",
    "JournaledGuestServiceRetirement",
    "guest_retirement_operation_digest",
    "guest_service_retirement_digest",
    "retire_guest_service",
    "serve_guest_requests",
    "copy_guest_service",
    "handle_guest_request",
    "inspect_guest_service",
    "request_guest_service",
    "serve_guest_once",
    "verify_linux_guest_peer",
    "JournaledGuestWorkspaces",
    "guest_operation_digest",
    "inspect_guest_workspace",
    "LimaConfiguration",
    "LimaConfigurationAudit",
    "audit_lima_configuration",
    "GuestWorkspaceEndpoint",
    "copy_to_guest",
    "GuestWorkspaceReceipt",
    "receive_guest_workspace",
    "transfer_guest_workspace",
    "audit_backend_artifacts",
    "StagedWorkspace",
    "WorkspaceStagePlan",
    "stage_workspace",
    "LimaArtifactVerifier",
    "LimaLifecycleBackend",
    "LimaGuestServiceGeneration",
    "LimaGuestServiceRetirementAdapter",
    "LimaQuiescenceObservation",
    "LimaRetirementObservation",
    "bind_lima_guest_service_generation",
    "decode_lima_service_observation",
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
