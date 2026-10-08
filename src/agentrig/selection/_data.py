"""Strict, payload-safe JSON decoding for selection and review records."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import cast


def object_data(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(k, str) for k in value):
        raise ValueError("record must be a JSON object")
    return cast(Mapping[str, object], value)


def fields(value: Mapping[str, object], names: str) -> None:
    if set(value) != set(names.split()):
        raise ValueError("record has missing or unknown fields")


def sequence(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError("record collection must be a JSON array")
    return value


def string(value: object) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError("record value must be a non-empty trimmed string")
    return value


def digest(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("record digest must be a lowercase SHA-256 hex string")
    return value


def version(value: object) -> None:
    if type(value) is not int or value != 1:
        raise ValueError("record schema version is unsupported")


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("record contains duplicate object keys")
        result[key] = value
    return result


def _constant(value: str) -> object:
    raise ValueError("record contains a non-finite number")


def loads_record(value: str) -> Mapping[str, object]:
    try:
        return object_data(json.loads(value, object_pairs_hook=_pairs, parse_constant=_constant))
    except (ValueError, TypeError, RecursionError):
        raise ValueError("invalid JSON record") from None
