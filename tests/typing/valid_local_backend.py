"""A replacement backend composes without a Lima import."""

from agentrig.capabilities import (
    BackendAuthority, BackendPlan, BackendResult, LocalBackend, LocalBackendCatalog,
)
from agentrig.core import RunContext
from agentrig.testing import ScriptedLocalBackend
from agentrig.capabilities import BackendOperationJournal, JournaledLocalBackends
from agentrig.testing import MemoryBackendJournal
from agentrig.capabilities import BackendArtifact, BackendArtifactManifest

backend: LocalBackend = ScriptedLocalBackend(["running"])
catalog = LocalBackendCatalog([backend])
journal: BackendOperationJournal = MemoryBackendJournal()
lifecycle = JournaledLocalBackends(catalog=catalog, journal=journal)
artifacts = BackendArtifactManifest(artifacts=(BackendArtifact(artifact_id='guest', path='guest.raw',
    sha256='sha256:' + '1' * 64, size_bytes=123),))


async def inspect(plan: BackendPlan, authority: BackendAuthority,
                  context: RunContext) -> BackendResult:
    return await catalog.apply(plan, authority, context)
