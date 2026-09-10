"""Durable intent before backend effects, with explicit observation-only recovery."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from typing import Protocol

from agentrig.capabilities.local_backend import (
    BackendAction, BackendAuthority, BackendFailure, BackendPlan, BackendResult,
    LocalBackendCatalog, backend_digest, backend_identifier, check_backend_call,
)
from agentrig.core import RunContext


@dataclass(frozen=True, slots=True, kw_only=True)
class BackendOperation:
    """Safe identity and state only. No permission, private configuration or PID."""

    operation_id: str
    backend_id: str
    backend_version: str
    configuration_digest: str
    plan_digest: str
    action: BackendAction
    status: str = 'uncertain'
    observation: str | None = None

    def __post_init__(self) -> None:
        from agentrig.capabilities.local_backend import BackendRequest, _digest
        BackendRequest(action=self.action, operation_id=self.operation_id, configuration={})
        backend_identifier(self.backend_id)
        backend_identifier(self.backend_version)
        _digest(self.configuration_digest)
        _digest(self.plan_digest)
        if self.status not in {'uncertain', 'completed', 'reconciled'}:
            raise ValueError('backend operation status invalid')
        if self.status == 'uncertain':
            if self.observation is not None:
                raise ValueError('uncertain backend observation invalid')
        elif self.observation not in {'prepared', 'running', 'stopped', 'absent'}:
            raise ValueError('backend observation invalid')
        if self.status == 'reconciled' and self.observation == 'prepared':
            raise ValueError('backend reconciliation observation invalid')

    @classmethod
    def intent(cls, plan: BackendPlan) -> BackendOperation:
        return cls(operation_id=plan.request.operation_id, backend_id=plan.descriptor.backend_id,
            backend_version=plan.descriptor.version, configuration_digest=backend_digest(plan.request.configuration),
            plan_digest=plan.digest, action=plan.request.action)


class BackendOperationWriter(Protocol):
    def current(self) -> BackendOperation | None: ...
    def create(self, operation: BackendOperation) -> None:
        """Durably store fresh intent; reject every previously used operation ID."""
        ...
    def settle(self, expected: BackendOperation, result: BackendOperation) -> None:
        """Durable compare-and-set of the current uncertain operation."""
        ...


class BackendOperationJournal(Protocol):
    def locked(self) -> AbstractContextManager[BackendOperationWriter]:
        """Exclusive nonblocking writer, bound to one local instance directory."""
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class JournaledLocalBackends:
    catalog: LocalBackendCatalog
    journal: BackendOperationJournal

    async def apply(self, plan: BackendPlan, authority: BackendAuthority, context: RunContext,
                    *, journal_write_authorized: bool = False) -> BackendResult:
        if journal_write_authorized is not True:
            raise BackendFailure('backend_authority_required')
        backend = self.catalog.resolve(plan.descriptor.backend_id)
        check_backend_call(backend, plan, authority, context)
        with self.journal.locked() as writer:
            current = writer.current()
            if current is not None and current.status == 'uncertain':
                raise BackendFailure('backend_reconciliation_required')
            operation = BackendOperation.intent(plan)
            writer.create(operation)
            # Every failure (including cancellation and failed result publication)
            # leaves durable uncertainty. No automatic retry or failure-to-absent.
            result = await self.catalog.apply(plan, authority, context)
            writer.settle(operation, replace(operation, status='completed', observation=result.status))
            return result

    async def reconcile(self, inspection: BackendPlan, authority: BackendAuthority,
                        context: RunContext, *, journal_write_authorized: bool = False) -> BackendResult:
        if journal_write_authorized is not True:
            raise BackendFailure('backend_authority_required')
        backend = self.catalog.resolve(inspection.descriptor.backend_id)
        check_backend_call(backend, inspection, authority, context)
        if inspection.request.action is not BackendAction.INSPECT:
            raise BackendFailure('backend_plan_changed')
        with self.journal.locked() as writer:
            current = writer.current()
            if current is None or current.status != 'uncertain':
                raise BackendFailure('backend_reconciliation_required')
            intent = BackendOperation.intent(inspection)
            if any(getattr(current, field) != getattr(intent, field) for field in
                   ('operation_id', 'backend_id', 'backend_version', 'configuration_digest')):
                raise BackendFailure('backend_plan_changed')
            # The adapter must inspect the exact instance identified by the
            # unchanged configuration and operation. Saved state grants nothing.
            result = await self.catalog.apply(inspection, authority, context)
            writer.settle(current, replace(current, status='reconciled', observation=result.status))
            return result
