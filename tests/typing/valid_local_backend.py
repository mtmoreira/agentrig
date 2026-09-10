"""A replacement backend composes without a Lima import."""

from agentrig.capabilities import (
    BackendAuthority, BackendPlan, BackendResult, LocalBackend, LocalBackendCatalog,
)
from agentrig.core import RunContext
from agentrig.testing import ScriptedLocalBackend

backend: LocalBackend = ScriptedLocalBackend(["running"])
catalog = LocalBackendCatalog([backend])


async def inspect(plan: BackendPlan, authority: BackendAuthority,
                  context: RunContext) -> BackendResult:
    return await catalog.apply(plan, authority, context)
