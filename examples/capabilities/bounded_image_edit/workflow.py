"""Reusable composition for one explicitly routed bounded image edit."""

from __future__ import annotations

from agentrig.capabilities import ImageGenerationRequest
from agentrig.core import RunContext
from agentrig.workflow import ImageExecution, ImageGenerationExecutor


async def execute_bounded_edit(
    executor: ImageGenerationExecutor,
    *,
    route_id: str,
    request: ImageGenerationRequest,
    context: RunContext,
) -> ImageExecution:
    """Execute the exact selected route without provider fallback."""
    return await executor.execute(
        route_id=route_id,
        request=request,
        context=context,
    )
