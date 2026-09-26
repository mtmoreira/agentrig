"""Replaceable local execution backends; plans and saved state are not authority."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
import hashlib
import json
import re
from types import MappingProxyType
from typing import Protocol, cast, runtime_checkable

from agentrig.core import JsonValue, RunContext
from agentrig.core._json import freeze_json_object, thaw_json_value
from agentrig.core.deadline import DeadlineExceeded


class BackendAction(StrEnum):
    INSPECT = "inspect"
    PREPARE = "prepare"
    START = "start"
    STOP = "stop"


class BackendEffect(StrEnum):
    LOCAL_WRITE = "local_write"
    RUNTIME_CONTACT = "runtime_contact"
    RUNTIME_START = "runtime_start"
    PRIVATE_KEY_CREATION = "private_key_creation"


def backend_identifier(value: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,95}", value) is None:
        raise ValueError("backend identifier invalid")
    return value


def backend_digest(value: JsonValue) -> str:
    payload = json.dumps(thaw_json_value(value), sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _digest(value: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"sha256:[a-f0-9]{64}", value) is None:
        raise ValueError("backend digest invalid")


def _configuration(value: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    try:
        frozen = freeze_json_object("backend configuration", value)
        size = len(json.dumps(thaw_json_value(frozen), ensure_ascii=True).encode())
        if size > 65536:
            raise ValueError("too large")
        return frozen
    except (ValueError, TypeError, RecursionError):
        raise ValueError("backend configuration invalid") from None


@dataclass(frozen=True, slots=True, kw_only=True)
class BackendDescriptor:
    backend_id: str
    version: str
    capabilities: frozenset[str]
    actions: frozenset[BackendAction]

    def __post_init__(self) -> None:
        backend_identifier(self.backend_id)
        backend_identifier(self.version)
        capabilities = frozenset(self.capabilities)
        actions = frozenset(self.actions)
        if not capabilities or len(capabilities) > 32 or not actions:
            raise ValueError("backend descriptor invalid")
        for name in capabilities:
            backend_identifier(name)
        if any(type(action) is not BackendAction for action in actions):
            raise ValueError("backend action invalid")
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(self, "actions", actions)


@dataclass(frozen=True, slots=True, kw_only=True)
class BackendRequest:
    action: BackendAction
    operation_id: str
    configuration: Mapping[str, JsonValue] = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.action) is not BackendAction:
            raise ValueError("backend action invalid")
        if not isinstance(self.operation_id, str) or re.fullmatch(r"[a-f0-9]{32}", self.operation_id) is None:
            raise ValueError("backend operation invalid")
        object.__setattr__(self, "configuration", _configuration(self.configuration))


@dataclass(frozen=True, slots=True, kw_only=True)
class BackendPlan:
    descriptor: BackendDescriptor
    request: BackendRequest = field(repr=False)
    effects: frozenset[BackendEffect]
    material: Mapping[str, JsonValue] = field(repr=False)
    blockers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.descriptor, BackendDescriptor) or not isinstance(self.request, BackendRequest):
            raise TypeError("backend plan invalid")
        if self.request.action not in self.descriptor.actions:
            raise ValueError("backend action unsupported")
        effects = frozenset(self.effects)
        if any(type(effect) is not BackendEffect for effect in effects):
            raise ValueError("backend effect invalid")
        blockers = tuple(self.blockers)
        if len(blockers) > 32 or len(set(blockers)) != len(blockers):
            raise ValueError("backend blockers invalid")
        for blocker in blockers:
            backend_identifier(blocker)
        object.__setattr__(self, "effects", effects)
        object.__setattr__(self, "blockers", blockers)
        object.__setattr__(self, "material", _configuration(self.material))

    def to_json(self) -> Mapping[str, JsonValue]:
        """Explicit review export, including private paths; never an event payload."""
        value = _configuration({"schema": "agentrig.local-backend-plan.v1",
            "backend_id": self.descriptor.backend_id, "version": self.descriptor.version,
            "capabilities": tuple(sorted(self.descriptor.capabilities)),
            "actions": tuple(sorted(action.value for action in self.descriptor.actions)),
            "action": self.request.action.value, "operation_id": self.request.operation_id,
            "configuration": self.request.configuration, "material": self.material,
            "effects": tuple(sorted(effect.value for effect in self.effects)), "blockers": self.blockers})
        return cast(Mapping[str, JsonValue], thaw_json_value(value))

    @property
    def digest(self) -> str:
        return backend_digest(self.to_json())


@dataclass(frozen=True, slots=True, kw_only=True)
class BackendAuthority:
    """An invocation-local grant supplied by the application, never loaded from state."""

    plan_digest: str
    operation_id: str
    effects: frozenset[BackendEffect]

    def __post_init__(self) -> None:
        _digest(self.plan_digest)
        if not isinstance(self.operation_id, str) or re.fullmatch(r"[a-f0-9]{32}", self.operation_id) is None:
            raise ValueError("backend operation invalid")
        effects = frozenset(self.effects)
        if any(type(effect) is not BackendEffect for effect in effects):
            raise ValueError("backend effect invalid")
        object.__setattr__(self, "effects", effects)


@dataclass(frozen=True, slots=True, kw_only=True)
class BackendResult:
    plan_digest: str
    operation_id: str
    status: str
    binding: Mapping[str, JsonValue] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        _digest(self.plan_digest)
        if self.status not in ("prepared", "running", "stopped", "absent"):
            raise ValueError("backend result status invalid")
        if not isinstance(self.operation_id, str) or re.fullmatch(r"[a-f0-9]{32}", self.operation_id) is None:
            raise ValueError("backend operation invalid")
        object.__setattr__(self, "binding", _configuration(self.binding))


class BackendFailure(RuntimeError):
    """Fixed safe code only; adapter exception details are never retained."""

    def __init__(self, code: str) -> None:
        allowed = {"backend_unavailable", "backend_incompatible", "backend_plan_changed",
                   "backend_plan_blocked", "backend_authority_required", "backend_invalid_result",
                   "backend_execution_failed", "backend_reconciliation_required", "backend_journal_invalid",
                   "backend_journal_busy", "backend_operation_reused", "backend_staging_failed",
                   "backend_artifacts_invalid"}
        self.code = code if code in allowed else "backend_execution_failed"
        super().__init__(self.code)


@runtime_checkable
class LocalBackend(Protocol):
    @property
    def descriptor(self) -> BackendDescriptor: ...

    def plan(self, request: BackendRequest) -> BackendPlan:
        """Pure deterministic planning. Must not inspect files, a daemon or a VM."""
        ...

    async def apply(self, plan: BackendPlan, authority: BackendAuthority,
                    context: RunContext) -> BackendResult: ...


def check_backend_call(backend: LocalBackend, plan: BackendPlan, authority: BackendAuthority,
                       context: RunContext) -> None:
    """Shared preflight, repeated by adapters immediately before their effects."""
    context.cancellation.raise_if_cancelled()
    if context.deadline is not None:
        context.deadline.raise_if_expired(context.clock)
    if not isinstance(plan, BackendPlan) or plan.descriptor != backend.descriptor:
        raise BackendFailure("backend_plan_changed")
    try:
        current = backend.plan(plan.request)
    except Exception:
        raise BackendFailure("backend_plan_changed") from None
    if current != plan:
        raise BackendFailure("backend_plan_changed")
    if plan.blockers:
        raise BackendFailure("backend_plan_blocked")
    if (not isinstance(authority, BackendAuthority) or authority.plan_digest != plan.digest or
            authority.operation_id != plan.request.operation_id or authority.effects != plan.effects):
        raise BackendFailure("backend_authority_required")


@dataclass(frozen=True, slots=True, init=False)
class LocalBackendCatalog:
    _backends: Mapping[str, LocalBackend] = field(repr=False)
    _descriptors: Mapping[str, BackendDescriptor] = field(repr=False)

    def __init__(self, backends: Iterable[LocalBackend]) -> None:
        copied: dict[str, LocalBackend] = {}
        for backend in backends:
            if not isinstance(backend, LocalBackend) or not isinstance(backend.descriptor, BackendDescriptor):
                raise TypeError("local backend invalid")
            key = backend.descriptor.backend_id
            if key in copied:
                raise ValueError("duplicate local backend")
            copied[key] = backend
        object.__setattr__(self, "_backends", MappingProxyType(copied))
        object.__setattr__(self, "_descriptors", MappingProxyType(
            {key: backend.descriptor for key, backend in copied.items()}))

    @property
    def descriptors(self) -> tuple[BackendDescriptor, ...]:
        return tuple(self._descriptors.values())

    def resolve(self, backend_id: str, required: frozenset[str] = frozenset()) -> LocalBackend:
        backend = self._backends.get(backend_id)
        if backend is None:
            raise BackendFailure("backend_unavailable")
        if backend.descriptor != self._descriptors[backend_id]:
            raise BackendFailure("backend_plan_changed")
        if not required <= backend.descriptor.capabilities:
            raise BackendFailure("backend_incompatible")
        return backend

    async def apply(self, plan: BackendPlan, authority: BackendAuthority,
                    context: RunContext) -> BackendResult:
        try:
            if not isinstance(plan, BackendPlan):
                raise BackendFailure("backend_plan_changed")
            backend = self.resolve(plan.descriptor.backend_id)
            check_backend_call(backend, plan, authority, context)
            result = await backend.apply(plan, authority, context)
        except (BackendFailure, DeadlineExceeded):
            raise
        except Exception:
            raise BackendFailure("backend_execution_failed") from None
        if (not isinstance(result, BackendResult) or result.plan_digest != plan.digest or
                result.operation_id != plan.request.operation_id):
            raise BackendFailure("backend_invalid_result")
        expected = {BackendAction.INSPECT: {"running", "stopped", "absent"},
                    BackendAction.PREPARE: {"prepared"}, BackendAction.START: {"running"},
                    BackendAction.STOP: {"stopped", "absent"}}
        if result.status not in expected[plan.request.action]:
            raise BackendFailure("backend_invalid_result")
        return result
