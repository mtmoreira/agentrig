"""Internal immutable JSON-value helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import TypeAlias, cast

from agentrig.core._validation import require_trimmed_string

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = (
    JsonScalar
    | Mapping[str, "JsonValue"]
    | list["JsonValue"]
    | tuple["JsonValue", ...]
)


def freeze_json_object(
    field_name: str,
    value: Mapping[str, JsonValue],
) -> Mapping[str, JsonValue]:
    frozen = freeze_json_value(field_name, value)
    return cast(Mapping[str, JsonValue], frozen)


def freeze_json_value(field_name: str, value: JsonValue) -> JsonValue:
    return _freeze_json_value(value, path=field_name, active_ids=set())


def thaw_json_value(value: JsonValue) -> JsonValue:
    if isinstance(value, Mapping):
        return {key: thaw_json_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [thaw_json_value(child) for child in value]
    return value


def _freeze_json_value(
    value: JsonValue,
    *,
    path: str,
    active_ids: set[int],
) -> JsonValue:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain non-finite numbers")
        return value
    if isinstance(value, Mapping):
        return _freeze_json_mapping(value, path=path, active_ids=active_ids)
    if isinstance(value, (list, tuple)):
        return _freeze_json_sequence(value, path=path, active_ids=active_ids)
    raise ValueError(f"{path} contains a non-JSON value: {type(value).__name__}")


def _freeze_json_mapping(
    value: Mapping[str, JsonValue],
    *,
    path: str,
    active_ids: set[int],
) -> Mapping[str, JsonValue]:
    identity = id(value)
    if identity in active_ids:
        raise ValueError(f"{path} contains a reference cycle")
    active_ids.add(identity)
    try:
        frozen: dict[str, JsonValue] = {}
        for key, child in value.items():
            validated_key = require_trimmed_string(f"{path} key", key)
            frozen[validated_key] = _freeze_json_value(
                child,
                path=f"{path}.{validated_key}",
                active_ids=active_ids,
            )
        return MappingProxyType(frozen)
    finally:
        active_ids.remove(identity)


def _freeze_json_sequence(
    value: list[JsonValue] | tuple[JsonValue, ...],
    *,
    path: str,
    active_ids: set[int],
) -> tuple[JsonValue, ...]:
    identity = id(value)
    if identity in active_ids:
        raise ValueError(f"{path} contains a reference cycle")
    active_ids.add(identity)
    try:
        return tuple(
            _freeze_json_value(
                child,
                path=f"{path}[{index}]",
                active_ids=active_ids,
            )
            for index, child in enumerate(value)
        )
    finally:
        active_ids.remove(identity)
