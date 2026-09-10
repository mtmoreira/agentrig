"""A replacement backend composes without a Lima import."""

from agentrig.capabilities import (
    BackendAuthority, BackendPlan, BackendResult, LocalBackend, LocalBackendCatalog,
)
from agentrig.core import RunContext
from agentrig.testing import ScriptedLocalBackend
from agentrig.capabilities import BackendOperationJournal, JournaledLocalBackends
from agentrig.testing import MemoryBackendJournal

backend: LocalBackend = ScriptedLocalBackend(["running"])
catalog = LocalBackendCatalog([backend])
journal: BackendOperationJournal = MemoryBackendJournal()
lifecycle = JournaledLocalBackends(catalog=catalog, journal=journal)


async def inspect(plan: BackendPlan, authority: BackendAuthority,
                  context: RunContext) -> BackendResult:
    return await catalog.apply(plan, authority, context)
