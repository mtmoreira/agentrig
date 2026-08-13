"""Immutable case definitions for versioned evaluation datasets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from agentrig.core._json import JsonValue, freeze_json_object
from agentrig.core._validation import require_trimmed_string

InputT = TypeVar("InputT")


@dataclass(frozen=True, slots=True, kw_only=True)
class EvalCase(Generic[InputT]):
    """One versioned input and its declarative evaluation expectations."""

    case_id: str
    version: str
    input: InputT = field(repr=False)
    expected_constraints: tuple[str, ...] = ()
    allowed_variability: tuple[str, ...] = ()
    prohibited_behaviors: tuple[str, ...] = ()
    fixture_refs: tuple[str, ...] = ()
    metadata: Mapping[str, JsonValue] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        require_trimmed_string("eval case ID", self.case_id)
        require_trimmed_string("eval case version", self.version)
        object.__setattr__(
            self,
            "expected_constraints",
            _copy_unique_strings(
                "eval case expected constraint",
                self.expected_constraints,
            ),
        )
        object.__setattr__(
            self,
            "allowed_variability",
            _copy_unique_strings(
                "eval case allowed variability",
                self.allowed_variability,
            ),
        )
        object.__setattr__(
            self,
            "prohibited_behaviors",
            _copy_unique_strings(
                "eval case prohibited behavior",
                self.prohibited_behaviors,
            ),
        )
        object.__setattr__(
            self,
            "fixture_refs",
            _copy_unique_strings(
                "eval case fixture reference",
                self.fixture_refs,
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            freeze_json_object("eval case metadata", self.metadata),
        )


def _copy_unique_strings(
    field_name: str,
    values: tuple[str, ...],
) -> tuple[str, ...]:
    copied = tuple(values)
    for value in copied:
        require_trimmed_string(field_name, value)
    if len(copied) != len(set(copied)):
        raise ValueError(f"{field_name} values must be unique")
    return copied
