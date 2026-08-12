"""Immutable execution context and deterministic child derivation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from agentrig.core.cancellation import CancellationToken
from agentrig.core.clock import Clock
from agentrig.core.deadline import Deadline
from agentrig.core.identity import IdGenerator, RunId


def _freeze_string_map(
    field_name: str,
    values: Mapping[str, str],
) -> Mapping[str, str]:
    copied: dict[str, str] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key or key != key.strip():
            raise ValueError(
                f"{field_name} keys must be non-empty strings without "
                "surrounding whitespace"
            )
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError(
                f"{field_name} values must be non-empty strings without "
                "surrounding whitespace"
            )
        copied[key] = value
    return MappingProxyType(copied)


@dataclass(frozen=True, slots=True, kw_only=True)
class RunContext:
    """Provider-independent, read-only dependencies for one execution."""

    run_id: RunId
    parent_run_id: RunId | None
    clock: Clock
    id_generator: IdGenerator[RunId]
    cancellation: CancellationToken
    deadline: Deadline | None = None
    labels: Mapping[str, str] = field(default_factory=dict)
    correlation: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "labels",
            _freeze_string_map("labels", self.labels),
        )
        object.__setattr__(
            self,
            "correlation",
            _freeze_string_map("correlation", self.correlation),
        )

    @classmethod
    def create_root(
        cls,
        *,
        clock: Clock,
        id_generator: IdGenerator[RunId],
        cancellation: CancellationToken,
        deadline: Deadline | None = None,
        labels: Mapping[str, str] | None = None,
        correlation: Mapping[str, str] | None = None,
    ) -> RunContext:
        """Create a root context from explicitly owned runtime dependencies."""
        return cls(
            run_id=id_generator.generate(),
            parent_run_id=None,
            clock=clock,
            id_generator=id_generator,
            cancellation=cancellation,
            deadline=deadline,
            labels=labels if labels is not None else {},
            correlation=correlation if correlation is not None else {},
        )

    def derive_child(
        self,
        *,
        cancellation: CancellationToken | None = None,
        deadline: Deadline | None = None,
        timeout_seconds: float | None = None,
        labels: Mapping[str, str] | None = None,
        correlation: Mapping[str, str] | None = None,
    ) -> RunContext:
        """Derive a child that cannot outlive this context's deadline.

        Cancellation is inherited unless the caller supplies a separately
        owned token, such as one from a linked child ``CancellationSource``.
        """
        relative_deadline = (
            Deadline.after(timeout_seconds, self.clock)
            if timeout_seconds is not None
            else None
        )
        child_deadline = Deadline.earliest(
            self.deadline,
            deadline,
            relative_deadline,
        )

        child_labels = dict(self.labels)
        if labels is not None:
            child_labels.update(labels)

        child_correlation = dict(self.correlation)
        if correlation is not None:
            child_correlation.update(correlation)

        return RunContext(
            run_id=self.id_generator.generate(),
            parent_run_id=self.run_id,
            clock=self.clock,
            id_generator=self.id_generator,
            cancellation=(
                cancellation if cancellation is not None else self.cancellation
            ),
            deadline=child_deadline,
            labels=child_labels,
            correlation=child_correlation,
        )
