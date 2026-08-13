"""Typed workflow step contracts and explicit side-effect semantics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, TypeVar, runtime_checkable

from agentrig.core._validation import require_trimmed_string
from agentrig.core.context import RunContext

InputT = TypeVar("InputT", contravariant=True)
OutputT = TypeVar("OutputT", covariant=True)


class EffectProfile(StrEnum):
    """Stable side-effect classification used by execution policy."""

    READ_ONLY = "read_only"
    IDEMPOTENT = "idempotent"
    COMPENSATABLE = "compensatable"
    NON_REPEATABLE = "non_repeatable"

    @property
    def allows_automatic_retry(self) -> bool:
        """Whether repeating after a transient failure is safe by declaration."""
        return self in (EffectProfile.READ_ONLY, EffectProfile.IDEMPOTENT)


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
