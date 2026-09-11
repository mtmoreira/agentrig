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

backend: LocalBackend = ScriptedLocalBackend(["running"])
catalog = LocalBackendCatalog([backend])
journal: BackendOperationJournal = MemoryBackendJournal()
lifecycle = JournaledLocalBackends(catalog=catalog, journal=journal)
artifacts = BackendArtifactManifest(artifacts=(BackendArtifact(artifact_id='guest', path='guest.raw',
    sha256='sha256:' + '1' * 64, size_bytes=123),))


def transfer(connection: socket.socket, plan: WorkspaceStagePlan, instance_id: str,
             approved_digest: str, context: RunContext) -> GuestWorkspaceReceipt:
    return transfer_guest_workspace(connection, plan, instance_id, approved_digest, context)


def copy(endpoint: GuestWorkspaceEndpoint, plan: WorkspaceStagePlan,
         approved_digest: str, context: RunContext) -> GuestWorkspaceReceipt:
    return copy_to_guest(endpoint, plan, approved_digest, context)


def audit_configuration(configuration: LimaConfiguration, context: RunContext) -> LimaConfigurationAudit:
    return audit_lima_configuration(configuration, context)


async def inspect(plan: BackendPlan, authority: BackendAuthority,
                  context: RunContext) -> BackendResult:
    return await catalog.apply(plan, authority, context)
