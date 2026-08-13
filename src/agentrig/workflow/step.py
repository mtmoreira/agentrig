"""Typed workflow step contracts and explicit side-effect semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeVar, runtime_checkable

from agentrig.core._validation import require_trimmed_string
from agentrig.core.context import RunContext
from agentrig.core.effects import EffectProfile as EffectProfile

InputT = TypeVar("InputT", contravariant=True)
OutputT = TypeVar("OutputT", covariant=True)


@dataclass(frozen=True, order=True, slots=True, kw_only=True)
class StepDescriptor:
    """Stable identity and effect declaration for one configured step."""

    step_id: str
    version: str
    effect_profile: EffectProfile

    def __post_init__(self) -> None:
        require_trimmed_string("step ID", self.step_id)
        require_trimmed_string("step version", self.version)
        if not isinstance(self.effect_profile, EffectProfile):
            raise TypeError("step effect_profile must be an EffectProfile")


@runtime_checkable
class Step(Protocol[InputT, OutputT]):
    """Transform one typed input inside an explicit execution context."""

    @property
    def descriptor(self) -> StepDescriptor:
        """Return the stable identity and effect profile for this step."""
        ...

    async def run(self, input: InputT, context: RunContext) -> OutputT:
        """Return typed output or raise for the executor to normalize."""
        ...
