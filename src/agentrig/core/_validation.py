"""Internal validation helpers shared by core value objects."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType


def require_trimmed_string(field_name: str, value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(
            f"{field_name} must be a non-empty string without surrounding "
            "whitespace"
        )
    return value


def freeze_string_map(
    field_name: str,
    values: Mapping[str, str],
) -> Mapping[str, str]:
    copied: dict[str, str] = {}
    for key, value in values.items():
        validated_key = require_trimmed_string(f"{field_name} key", key)
        validated_value = require_trimmed_string(f"{field_name} value", value)
        copied[validated_key] = validated_value
    return MappingProxyType(copied)
