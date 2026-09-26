"""Independent in-memory journal and shared durable operation contract checks."""

from contextlib import contextmanager
from dataclasses import replace
from typing import Iterator

from agentrig.capabilities.backend_operations import BackendOperation, BackendOperationJournal, BackendOperationWriter
from agentrig.capabilities.local_backend import BackendFailure


class MemoryBackendJournal:
    def __init__(self) -> None:
        self._records: dict[str, BackendOperation] = {}
        self._current: str | None = None
        self._locked = False

    @contextmanager
    def locked(self) -> Iterator[BackendOperationWriter]:
        if self._locked:
            raise BackendFailure('backend_journal_busy')
        self._locked = True
        try:
            yield self
        finally:
            self._locked = False

    def current(self) -> BackendOperation | None:
        return self._records[self._current] if self._current is not None else None

    def create(self, operation: BackendOperation) -> None:
        current = self.current()
        if operation.status != 'uncertain' or current is not None and current.status == 'uncertain':
            raise BackendFailure('backend_reconciliation_required')
        if operation.operation_id in self._records:
            raise BackendFailure('backend_operation_reused')
        self._records[operation.operation_id] = operation
        self._current = operation.operation_id

    def settle(self, expected: BackendOperation, result: BackendOperation) -> None:
        if (self.current() != expected or expected.status != 'uncertain' or
                result.status not in {'completed', 'reconciled'} or
                replace(result, status='uncertain', observation=None) != expected):
            raise BackendFailure('backend_journal_invalid')
        self._records[result.operation_id] = result


class BackendJournalContractSuite:
    def verify(self, journal: BackendOperationJournal, intent: BackendOperation) -> None:
        with journal.locked() as writer:
            if writer.current() is not None:
                raise AssertionError('journal fixture must be empty')
            writer.create(intent)
        with journal.locked() as writer:
            if writer.current() != intent:
                raise AssertionError('journal lost durable intent')
            try:
                writer.create(replace(intent, operation_id='b' * 32))
            except BackendFailure as error:
                if error.code != 'backend_reconciliation_required':
                    raise AssertionError('journal lost uncertainty') from None
            else:
                raise AssertionError('journal overwrote uncertainty')
            try:
                with journal.locked():
                    raise AssertionError('journal allowed concurrent writer')
            except BackendFailure as error:
                if error.code != 'backend_journal_busy':
                    raise AssertionError('journal lost writer exclusion') from None
            completed = replace(intent, status='completed', observation='running')
            writer.settle(intent, completed)
        with journal.locked() as writer:
            if writer.current() != completed:
                raise AssertionError('journal lost completion')
            try:
                writer.create(intent)
            except BackendFailure as error:
                if error.code != 'backend_operation_reused':
                    raise AssertionError('journal lost operation identity') from None
            else:
                raise AssertionError('journal replayed completed operation')
