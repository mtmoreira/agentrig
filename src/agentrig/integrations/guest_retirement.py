"""Durable service retirement and observation-only recovery, without VM control."""

from contextlib import AbstractContextManager
from dataclasses import dataclass, field, replace
from typing import Literal, Protocol

from agentrig.capabilities import BackendAction, BackendFailure, BackendOperation
from agentrig.capabilities.backend_operations import BackendOperationJournal
from agentrig.capabilities.local_backend import backend_digest
from agentrig.core import DeadlineExceeded, RunCancelled, RunContext
from agentrig.integrations.guest_operations import _Inspection
from agentrig.integrations.guest_service import guest_service_retirement_digest
from agentrig.integrations.guest_workspace import GuestWorkspaceEndpoint, _instance


class GuestServiceRetirementAdapter(Protocol):
    """Trusted backend-specific verification and transport, never a saved grant.

    quiescent authenticates the exact service generation and its forwarding,
    proves all clients finished endpoint checks, and holds a backend lifecycle
    fence against restart/replacement throughout the context. Enter/exit failures
    must fail closed. A PID, free local lease or missing socket is insufficient.

    retire removes only the reviewed socket using the low-level guarded port (or
    its authenticated guest equivalent). inspect_retired makes a fresh read-only
    observation in the owned namespace, binding retirement to this exact endpoint
    and retained backend evidence. Missing/unavailable/replaced state alone must
    fail; never recreate a socket, repair records or replay retirement to inspect.
    Successful methods/context entry return exactly None, checked at runtime.
    All methods honor the supplied cancellation/deadline. Nothing is registered
    by default; implementing this protocol is not proof of runtime qualification.
    """

    def quiescent(self, endpoint: GuestWorkspaceEndpoint,
                  context: RunContext) -> AbstractContextManager[object]: ...

    def retire(self, endpoint: GuestWorkspaceEndpoint, approved_digest: str,
               context: RunContext) -> object: ...

    def inspect_retired(self, endpoint: GuestWorkspaceEndpoint,
                        context: RunContext) -> object: ...


def guest_retirement_operation_digest(endpoint: GuestWorkspaceEndpoint, operation_id: str,
                                      action: Literal['retire', 'inspect']) -> str:
    """Distinct fresh approvals for removal and reconciliation of one operation."""
    _instance(operation_id)
    if action not in ('retire', 'inspect'):
        raise ValueError('guest retirement action invalid')
    return backend_digest({'schema': 'agentrig.journaled-guest-retirement.v1',
        'endpoint_digest': guest_service_retirement_digest(endpoint),
        'operation_id': operation_id, 'action': action})


@dataclass(frozen=True, slots=True, kw_only=True)
class JournaledGuestServiceRetirement:
    """Dedicated per-instance retirement journal, separate from copy/VM state.

    Persist uncertainty before invoking the adapter; hold the journal lease until
    verification and adapter context exit succeed. Every ambiguous result remains
    uncertain. Recovery only observes the exact original intent, under a distinct
    approval, and never calls retire. Completed/reconciled records are historical
    observations and confer no new authority to start or stop a service.
    """

    journal: BackendOperationJournal = field(repr=False)
    adapter: GuestServiceRetirementAdapter | None = field(default=None, repr=False)

    def apply(self, endpoint: GuestWorkspaceEndpoint, operation_id: str,
              approved_digest: str, context: RunContext, *,
              journal_write_authorized: bool = False,
              action: Literal['retire', 'inspect'] = 'retire') -> BackendOperation:
        try:
            check = _Inspection(context)
            check.check()
            if (journal_write_authorized is not True or approved_digest !=
                    guest_retirement_operation_digest(endpoint, operation_id, action)):
                raise BackendFailure('backend_authority_required')
            if self.adapter is None:
                raise BackendFailure('backend_plan_blocked')
            intent = BackendOperation(operation_id=operation_id,
                backend_id='guest-service-retirement', backend_version='1',
                configuration_digest=guest_service_retirement_digest(endpoint),
                plan_digest=guest_retirement_operation_digest(endpoint, operation_id, 'retire'),
                action=BackendAction.STOP)
            with self.journal.locked() as writer:
                current = writer.current()
                if action == 'retire':
                    if current is not None and current.status == 'uncertain':
                        raise BackendFailure('backend_reconciliation_required')
                    writer.create(intent)
                elif current != intent:
                    raise BackendFailure('backend_reconciliation_required')
                try:
                    check.check()
                    observed = False
                    with self.adapter.quiescent(endpoint, context) as binding:
                        if binding is not None:
                            raise BackendFailure('backend_invalid_result')
                        check.check()
                        if action == 'retire':
                            if self.adapter.retire(endpoint, intent.configuration_digest, context) is not None:
                                raise BackendFailure('backend_invalid_result')
                            check.check()
                        if self.adapter.inspect_retired(endpoint, context) is not None:
                            raise BackendFailure('backend_invalid_result')
                        check.check()
                        observed = True
                    # A context manager that suppresses a failed verification
                    # must not turn it into a successful retirement observation.
                    if not observed:
                        raise BackendFailure('backend_invalid_result')
                    check.check()
                except (BackendFailure, DeadlineExceeded, RunCancelled):
                    raise
                except Exception:
                    # Normalize adapter OSError before the journal can classify
                    # it as a local storage failure. Preserve no private text.
                    raise BackendFailure('backend_staging_failed') from None
                result = replace(intent, status='completed' if action == 'retire' else 'reconciled',
                    observation='stopped')
                writer.settle(intent, result)
                return result
        except (BackendFailure, DeadlineExceeded, RunCancelled):
            raise
        except Exception:
            raise BackendFailure('backend_staging_failed') from None
