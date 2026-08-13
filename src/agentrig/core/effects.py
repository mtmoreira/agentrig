"""Portable side-effect semantics shared by executable boundaries."""

from __future__ import annotations

from enum import StrEnum


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
