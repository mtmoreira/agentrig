"""Typed execution boundary for systems under evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, TypeVar, runtime_checkable

from agentrig.core._validation import require_trimmed_string
from agentrig.core.context import RunContext
from agentrig.core.outcomes import ExecutionOutcome

InputT = TypeVar("InputT", contravariant=True)
OutputT = TypeVar("OutputT")


class EvalTargetKind(StrEnum):
    """Stable categories of systems that can be evaluated."""

    AGENT = "agent"
    WORKFLOW = "workflow"
    CAPABILITY = "capability"
    INTEGRATION = "integration"


@dataclass(frozen=True, slots=True, kw_only=True)
class EvalTargetDescriptor:
    """Stable identity and category for one configured evaluation target."""

    target_id: str
    version: str
    kind: EvalTargetKind

    def __post_init__(self) -> None:
        require_trimmed_string("eval target ID", self.target_id)
        require_trimmed_string("eval target version", self.version)
        if not isinstance(self.kind, EvalTargetKind):
            raise TypeError("eval target kind must be an EvalTargetKind")


@runtime_checkable
class EvalTarget(Protocol[InputT, OutputT]):
    """Execute one typed case through a normalized evaluation boundary."""

    @property
    def descriptor(self) -> EvalTargetDescriptor:
        """Return the stable identity of this configured target."""
        ...

    async def run(
        self,
        input: InputT,
        context: RunContext,
    ) -> ExecutionOutcome[OutputT]:
        """Run one case input inside the eval runner's isolated context."""
        ...
