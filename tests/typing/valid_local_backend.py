"""A replacement backend composes without a Lima import."""

from agentrig.capabilities import (
    BackendAuthority, BackendPlan, BackendResult, LocalBackend, LocalBackendCatalog,
)
from agentrig.core import RunContext
from agentrig.testing import ScriptedLocalBackend
from agentrig.capabilities import BackendOperationJournal, JournaledLocalBackends
from agentrig.testing import MemoryBackendJournal
from agentrig.capabilities import BackendArtifact, BackendArtifactManifest
from agentrig.integrations import (
    GuestWorkspaceEndpoint, GuestWorkspaceReceipt, WorkspaceStagePlan, copy_to_guest,
    transfer_guest_workspace,
)
import socket
from agentrig.integrations import LimaConfiguration, LimaConfigurationAudit, audit_lima_configuration
from agentrig.integrations import JournaledGuestWorkspaces, guest_operation_digest, inspect_guest_workspace
from pathlib import Path
from agentrig.integrations import (
    copy_guest_service, inspect_guest_service, handle_guest_request,
    request_guest_service, serve_guest_once, verify_linux_guest_peer,
)

backend: LocalBackend = ScriptedLocalBackend(["running"])
catalog = LocalBackendCatalog([backend])
journal: BackendOperationJournal = MemoryBackendJournal()
lifecycle = JournaledLocalBackends(catalog=catalog, journal=journal)
artifacts = BackendArtifactManifest(artifacts=(BackendArtifact(artifact_id='guest', path='guest.raw',
    sha256='sha256:' + '1' * 64, size_bytes=123),))
service = JournaledGuestWorkspaces(journal=journal, copy=copy_guest_service, inspect=inspect_guest_service)


def guest_service(connection: socket.socket, root: Path, sockets: Path, instance: str,
                  plan: WorkspaceStagePlan, context: RunContext) -> GuestWorkspaceReceipt:
    verify_linux_guest_peer(connection, context)
    handle_guest_request(connection, root, instance, context)
    request_guest_service(connection, plan, instance, plan.digest, context, action='inspect')
    return serve_guest_once(root, sockets, instance, context, service_authorized=True)


def transfer(connection: socket.socket, plan: WorkspaceStagePlan, instance_id: str,
             approved_digest: str, context: RunContext) -> GuestWorkspaceReceipt:
    return transfer_guest_workspace(connection, plan, instance_id, approved_digest, context)


def copy(endpoint: GuestWorkspaceEndpoint, plan: WorkspaceStagePlan,
         approved_digest: str, context: RunContext) -> GuestWorkspaceReceipt:
    return copy_to_guest(endpoint, plan, approved_digest, context)


def audit_configuration(configuration: LimaConfiguration, context: RunContext) -> LimaConfigurationAudit:
    return audit_lima_configuration(configuration, context)


def recover_copy(driver: JournaledGuestWorkspaces, endpoint: GuestWorkspaceEndpoint,
                 plan: WorkspaceStagePlan, context: RunContext) -> GuestWorkspaceReceipt:
    return driver.apply(endpoint, plan, guest_operation_digest(endpoint, plan, 'inspect'),
        context, journal_write_authorized=True, action='inspect')


def inspect_copy(root: Path, plan: WorkspaceStagePlan, instance: str, context: RunContext) -> GuestWorkspaceReceipt:
    return inspect_guest_workspace(root, plan, instance, context)


async def inspect(plan: BackendPlan, authority: BackendAuthority,
                  context: RunContext) -> BackendResult:
    return await catalog.apply(plan, authority, context)
