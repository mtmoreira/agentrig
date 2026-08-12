"""Typed execution identities and their generation contract."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar
from uuid import UUID, uuid4

IdT_co = TypeVar("IdT_co", covariant=True)
IdT = TypeVar("IdT")


class IdGenerator(Protocol[IdT_co]):
    """Generate identifiers without binding callers to an ID scheme."""

    def generate(self) -> IdT_co:
        """Return a new identifier."""
        ...


@dataclass(frozen=True, order=True, slots=True)
class RunId:
    """Opaque, serializable identity for one execution."""

    value: str

    def __post_init__(self) -> None:
        if not self.value:
            raise ValueError("run ID must not be empty")
        if self.value != self.value.strip():
            raise ValueError("run ID must not have surrounding whitespace")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class Uuid4IdGenerator(Generic[IdT]):
    """Construct typed identifiers from random UUID4 values."""

    id_type: Callable[[str], IdT]
    uuid_factory: Callable[[], UUID] = uuid4

    def generate(self) -> IdT:
        return self.id_type(str(self.uuid_factory()))
