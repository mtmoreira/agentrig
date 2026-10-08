"""Runtime-safe application-scoped model set selection; no provider activation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from agentrig.core._json import JsonValue, freeze_json_object
from agentrig.core._validation import freeze_string_map
from agentrig.selection._data import digest, fields, object_data, string, version


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelSet:
    """An application-owned set of role-to-model identifiers."""

    application_id: str
    model_set_id: str
    version: str
    models: Mapping[str, str] = field(repr=False)
    settings: Mapping[str, JsonValue] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        for item in (self.application_id, self.model_set_id, self.version):
            string(item)
        if not self.models:
            raise ValueError("model set must contain at least one model role")
        object.__setattr__(self, "models", freeze_string_map("model set", self.models))
        frozen = freeze_json_object("model set settings", self.settings)
        if any(isinstance(value, (Mapping, list, tuple)) for value in frozen.values()):
            raise ValueError("model set settings must contain JSON scalar values")
        object.__setattr__(self, "settings", frozen)

    def to_data(self) -> dict[str, JsonValue]:
        return {"application_id": self.application_id, "model_set_id": self.model_set_id,
                "version": self.version, "models": dict(self.models), "settings": dict(self.settings)}

    @classmethod
    def from_data(cls, data: Mapping[str, object]) -> ModelSet:
        fields(data, "application_id model_set_id version models settings")
        settings: dict[str, JsonValue] = {}
        for key, value in object_data(data["settings"]).items():
            if value is not None and not isinstance(value, (bool, int, float, str)):
                raise ValueError("model set settings must contain JSON scalar values")
            settings[key] = value
        return cls(application_id=string(data["application_id"]),
                   model_set_id=string(data["model_set_id"]), version=string(data["version"]),
                   models={string(k): string(v) for k, v in object_data(data["models"]).items()},
                   settings=settings)


@dataclass(frozen=True, slots=True, kw_only=True)
class PreferredModelSet:
    """Portable winner provenance; deliberately excludes raw human votes."""

    application_id: str
    model_set: ModelSet
    comparison_id: str
    manifest_digest: str
    scoring_version: int = 1
    schema_version: int = 1

    def __post_init__(self) -> None:
        version(self.schema_version)
        version(self.scoring_version)
        string(self.application_id)
        string(self.comparison_id)
        digest(self.manifest_digest)
        if not isinstance(self.model_set, ModelSet):
            raise TypeError("preferred model set must contain a ModelSet")
        if self.model_set.application_id != self.application_id:
            raise ValueError("preferred model set belongs to a different application")

    def to_data(self) -> dict[str, JsonValue]:
        return {"schema_version": self.schema_version, "application_id": self.application_id,
                "model_set": self.model_set.to_data(), "comparison_id": self.comparison_id,
                "manifest_digest": self.manifest_digest, "scoring_version": self.scoring_version}

    @classmethod
    def from_data(cls, data: Mapping[str, object]) -> PreferredModelSet:
        fields(data, "schema_version application_id model_set comparison_id manifest_digest scoring_version")
        version(data["schema_version"])
        version(data["scoring_version"])
        return cls(application_id=string(data["application_id"]),
                   model_set=ModelSet.from_data(object_data(data["model_set"])),
                   comparison_id=string(data["comparison_id"]),
                   manifest_digest=digest(data["manifest_digest"]))


def select_model_set(
    *, application_id: str, default: ModelSet, preference: PreferredModelSet | None = None,
    explicit: ModelSet | None = None, persisted: ModelSet | None = None,
) -> ModelSet:
    """Resolve whole sets: resumed configuration, explicit set, preference, default.

    Consumers validate availability and apply explicit field overrides on new runs.
    Resumed runs must keep their persisted configuration. No providers are called.
    """
    string(application_id)
    candidates = (persisted, explicit, preference.model_set if preference else None, default)
    for candidate in candidates:
        if candidate is not None:
            if not isinstance(candidate, ModelSet):
                raise TypeError("selection requires ModelSet values")
            if candidate.application_id != application_id:
                raise ValueError("model set belongs to a different application")
    if preference is not None and preference.application_id != application_id:
        raise ValueError("preference belongs to a different application")
    return next(candidate for candidate in candidates if candidate is not None)
