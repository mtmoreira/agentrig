"""Offline backend substitution and portable preflight checks."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace

from agentrig.capabilities.local_backend import (
    BackendAction, BackendAuthority, BackendDescriptor, BackendEffect,
    BackendFailure, BackendPlan, BackendRequest, BackendResult,
    LocalBackend, LocalBackendCatalog, check_backend_call,
)
from agentrig.core import DeadlineExceeded, RunCancelled, RunContext


class ScriptedLocalBackend:
    """An independent backend with no process, filesystem or provider adapter."""

    descriptor = BackendDescriptor(backend_id="scripted-local", version="1",
        capabilities=frozenset({"docker-unix-endpoint.v1", "current-user-rootless.v1"}),
        actions=frozenset({BackendAction.INSPECT}))

    def __init__(self, outcomes: Iterable[str]) -> None:
        self._outcomes = tuple(outcomes)
        if any(status not in {"running", "stopped", "absent"} for status in self._outcomes):
            raise ValueError("scripted backend status invalid")
        self._calls: list[str] = []

    @property
    def calls(self) -> tuple[str, ...]:
        return tuple(self._calls)

    def plan(self, request: BackendRequest) -> BackendPlan:
        return BackendPlan(descriptor=self.descriptor, request=request,
            effects=frozenset({BackendEffect.RUNTIME_CONTACT}), material={"scripted": True})

    async def apply(self, plan: BackendPlan, authority: BackendAuthority,
                    context: RunContext) -> BackendResult:
        check_backend_call(self, plan, authority, context)
        if len(self._calls) >= len(self._outcomes):
            raise BackendFailure("backend_unavailable")
        status = self._outcomes[len(self._calls)]
        self._calls.append(plan.request.operation_id)
        return BackendResult(plan_digest=plan.digest, operation_id=plan.request.operation_id,
                             status=status, binding={"scripted": True})


@dataclass(frozen=True, slots=True, kw_only=True)
class LocalBackendContractSuite:
    """Reject changed plans, missing grants and cancellation before effects.

    The caller supplies a fresh fixture and invocation counter. Successful
    adapter execution and blocked plans are both valid fixture outcomes.
    """

    backend: LocalBackend = field(repr=False)
    request: BackendRequest = field(repr=False)
    context: RunContext = field(repr=False)
    cancelled_context: RunContext = field(repr=False)
    expired_context: RunContext = field(repr=False)
    invocation_count: Callable[[], int] = field(repr=False)

    async def verify(self) -> BackendResult | None:
        if not self.cancelled_context.cancellation.is_cancelled:
            raise ValueError("contract requires cancelled context")
        if (self.expired_context.deadline is None or
                not self.expired_context.deadline.is_expired(self.expired_context.clock)):
            raise ValueError("contract requires expired context")
        catalog = LocalBackendCatalog([self.backend])
        plan = self.backend.plan(self.request)
        grant = BackendAuthority(plan_digest=plan.digest,
            operation_id=plan.request.operation_id, effects=plan.effects)
        initial = self.invocation_count()
        attempts = ((replace(plan, material={"changed": True}), grant, self.context, "backend_plan_changed"),
                    (plan, replace(grant, plan_digest="sha256:" + "0" * 64), self.context,
                     "backend_plan_blocked" if plan.blockers else "backend_authority_required"),
                    (plan, grant, self.cancelled_context, "cancelled"),
                    (plan, grant, self.expired_context, "expired"))
        for changed, authority, context, expected in attempts:
            try:
                await catalog.apply(changed, authority, context)
            except BackendFailure as error:
                if error.code != expected:
                    raise AssertionError("backend preflight failure identity changed") from None
            except RunCancelled:
                if expected != "cancelled":
                    raise AssertionError("unexpected backend cancellation") from None
            except DeadlineExceeded:
                if expected != "expired":
                    raise AssertionError("unexpected backend deadline") from None
            else:
                raise AssertionError("backend accepted invalid preflight")
            if self.invocation_count() != initial:
                raise AssertionError("backend invoked after failed preflight")
        if plan.blockers:
            try:
                await catalog.apply(plan, grant, self.context)
            except BackendFailure as error:
                if error.code != "backend_plan_blocked":
                    raise AssertionError("backend lost blocked-plan identity") from None
            else:
                raise AssertionError("backend accepted blocked plan")
            if self.invocation_count() != initial:
                raise AssertionError("blocked backend invoked")
            return None
        result = await catalog.apply(plan, grant, self.context)
        if self.invocation_count() != initial + 1:
            raise AssertionError("backend did not invoke exactly once")
        return result
