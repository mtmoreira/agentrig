"""Immutable, versioned selections of evaluation cases."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from agentrig.core._json import JsonValue, freeze_json_object
from agentrig.core._validation import require_trimmed_string
from agentrig.evals.case import EvalCase

InputT = TypeVar("InputT")


@dataclass(frozen=True, slots=True, kw_only=True)
class EvalDataset(Generic[InputT]):
    """A stable dataset identity and ordered selection of versioned cases."""

    dataset_id: str
    version: str
    cases: tuple[EvalCase[InputT], ...]
    metadata: Mapping[str, JsonValue] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        require_trimmed_string("eval dataset ID", self.dataset_id)
        require_trimmed_string("eval dataset version", self.version)
        copied_cases = tuple(self.cases)
        if not copied_cases:
            raise ValueError("eval dataset requires at least one case")
        for case in copied_cases:
            if not isinstance(case, EvalCase):
                raise TypeError("eval dataset cases must contain EvalCase values")
        case_ids = tuple(case.case_id for case in copied_cases)
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("eval dataset case IDs must be unique")
        object.__setattr__(self, "cases", copied_cases)
        object.__setattr__(
            self,
            "metadata",
            freeze_json_object("eval dataset metadata", self.metadata),
        )
