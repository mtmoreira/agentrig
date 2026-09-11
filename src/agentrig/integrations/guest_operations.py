"""Journaled guest copies and read-only recovery; no service registration.

The application must supply authenticated backend verification and inspection.
A local journal or matching copy receipt cannot authenticate a VM or authorize it.
"""

from collections.abc import Callable
from dataclasses import dataclass, field, replace
import os
from pathlib import Path
import stat
from typing import Literal, cast

from agentrig.capabilities import BackendAction, BackendFailure, BackendOperation
from agentrig.capabilities.backend_operations import BackendOperationJournal
from agentrig.capabilities.local_backend import backend_digest
from agentrig.core import DeadlineExceeded, JsonValue, RunCancelled, RunContext
from agentrig.integrations.backend_artifacts import _identity
from agentrig.integrations.backend_journal import _private
from agentrig.integrations.backend_staging import WorkspaceStagePlan
from agentrig.integrations.guest_workspace import (
    GuestWorkspaceEndpoint, GuestWorkspaceReceipt, _encode, _manifest, _receipt,
    _request, _verify_copy, copy_to_guest,
)


@dataclass(frozen=True, slots=True)
class _Inspection:
    context: RunContext

    def check(self) -> None:
        self.context.cancellation.raise_if_cancelled()
        if self.context.deadline is None:
            raise BackendFailure('backend_staging_failed')
        self.context.deadline.raise_if_expired(self.context.clock)


def _expected(plan: WorkspaceStagePlan, instance_id: str) -> GuestWorkspaceReceipt:
    header = _request(plan, instance_id)
    value = _receipt(header, _manifest(header))
    return GuestWorkspaceReceipt(request_digest=cast(str, value['request_digest']),
        plan_digest=plan.digest, operation_id=plan.operation_id,
        file_count=len(plan.files), total_bytes=sum(map(len, plan.files.values())))


def inspect_guest_workspace(root: Path, plan: WorkspaceStagePlan, instance_id: str,
                            context: RunContext) -> GuestWorkspaceReceipt:
    """Reverify an existing complete guest operation, never repair or replay it.

    Run in the authenticated guest service with an ancestor-verified private root.
    Both receipts, exact inventory, current file bytes and local directory binding
    must match. Missing/partial state is failure, never proof of safe absence.
    No socket is opened and no file, permission or receipt is changed.
    """
    _Inspection(context).check()
    if str(root) != plan.guest_root:
        raise BackendFailure('backend_plan_changed')
    try:
        header = _request(plan, instance_id)
    except (ValueError, TypeError, RecursionError):
        raise BackendFailure('backend_staging_failed') from None
    return _inspect_guest_request(root, header, context)


def _inspect_guest_request(root: Path, header: dict[str, JsonValue],
                           context: RunContext) -> GuestWorkspaceReceipt:
    """Inspect using only a validated manifest; no source payload is required."""
    opened: list[int] = []
    check = _Inspection(context)
    try:
        check.check()
        manifest = _manifest(header)
        if str(root) != header['guest_root']:
            raise BackendFailure('backend_plan_changed')
        operation_id = cast(str, header['operation_id'])
        root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        opened.append(root_fd)
        root_info = os.fstat(root_fd)
        _private(root_info, directory=True)
        operation = os.open(operation_id, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd)
        opened.append(operation)
        info = os.fstat(operation)
        _private(info, directory=True)
        record = _encode(dict(_receipt(header, manifest), device=info.st_dev, inode=info.st_ino, uid=info.st_uid))
        expected = {'.intent.json', '.complete.json'}
        for artifact in manifest.artifacts:
            name = artifact.path
            expected.add(name)
            expected.update(str(p) for p in Path(name).parents if str(p) != '.')

        def inventory(directory: int, prefix: str = '') -> dict[str, tuple[int, ...]]:
            found: dict[str, tuple[int, ...]] = {}
            with os.scandir(directory) as entries:
                for entry in entries:
                    check.check()
                    name = prefix + entry.name
                    if name not in expected:
                        raise BackendFailure('backend_staging_failed')
                    item = os.stat(entry.name, dir_fd=directory, follow_symlinks=False)
                    found[name] = _identity(item)
                    if stat.S_ISDIR(item.st_mode):
                        if item.st_uid != os.getuid() or stat.S_IMODE(item.st_mode) != 0o755:
                            raise BackendFailure('backend_staging_failed')
                        child = os.open(entry.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
                        try:
                            if _identity(os.fstat(child)) != _identity(item):
                                raise BackendFailure('backend_staging_failed')
                            found.update(inventory(child, name + '/'))
                        finally:
                            os.close(child)
                    elif not stat.S_ISREG(item.st_mode) or item.st_nlink != 1:
                        raise BackendFailure('backend_staging_failed')
            return found

        before = inventory(operation)
        if set(before) != expected:
            raise BackendFailure('backend_staging_failed')
        for name in ('.intent.json', '.complete.json'):
            check.check()
            fd = os.open(name, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=operation)
            with os.fdopen(fd, 'rb') as stream:
                value = os.fstat(stream.fileno())
                _private(value, directory=False)
                if (stat.S_IMODE(value.st_mode) != 0o600 or value.st_size != len(record) or
                        stream.read(len(record) + 1) != record):
                    raise BackendFailure('backend_staging_failed')
        _verify_copy(operation, manifest, check)
        check.check()
        if (inventory(operation) != before or _identity(root.lstat()) != _identity(root_info) or
                _identity(os.stat(operation_id, dir_fd=root_fd, follow_symlinks=False)) != _identity(info)):
            raise BackendFailure('backend_staging_failed')
        return GuestWorkspaceReceipt(request_digest=backend_digest(header),
            plan_digest=cast(str, header['plan_digest']), operation_id=operation_id,
            file_count=len(manifest.artifacts), total_bytes=sum(a.size_bytes for a in manifest.artifacts))
    except (BackendFailure, DeadlineExceeded, RunCancelled):
        raise
    except Exception:
        raise BackendFailure('backend_staging_failed') from None
    finally:
        for fd in reversed(opened):
            os.close(fd)


def guest_operation_digest(endpoint: GuestWorkspaceEndpoint, plan: WorkspaceStagePlan,
                           action: Literal['copy', 'inspect']) -> str:
    if action not in ('copy', 'inspect'):
        raise ValueError('guest operation action invalid')
    return backend_digest({'schema': 'agentrig.journaled-guest-operation.v1',
        'transfer_digest': endpoint.transfer_digest(plan), 'action': action})


GuestEndpointVerifier = Callable[[GuestWorkspaceEndpoint, RunContext], object]
GuestCopier = Callable[[GuestWorkspaceEndpoint, WorkspaceStagePlan, str, RunContext], GuestWorkspaceReceipt]
GuestInspector = Callable[[GuestWorkspaceEndpoint, WorkspaceStagePlan, RunContext], GuestWorkspaceReceipt]


@dataclass(frozen=True, slots=True, kw_only=True)
class JournaledGuestWorkspaces:
    """A dedicated per-instance transfer journal, distinct from lifecycle state.

    Trusted application adapters authenticate the exact instance/forwarding before
    any request and after its reply. They must honor the context and normalize
    errors. No verifier is installed by default. Recovery requires a separately
    supplied authenticated, read-only guest inspector; it never resends payloads.
    This synchronous coordinator holds the writer lease through the whole call.
    """

    journal: BackendOperationJournal = field(repr=False)
    verify_endpoint: GuestEndpointVerifier | None = field(default=None, repr=False)
    copy: GuestCopier = field(default=copy_to_guest, repr=False)
    inspect: GuestInspector | None = field(default=None, repr=False)

    def apply(self, endpoint: GuestWorkspaceEndpoint, plan: WorkspaceStagePlan,
              approved_digest: str, context: RunContext, *, journal_write_authorized: bool = False,
              action: Literal['copy', 'inspect'] = 'copy') -> GuestWorkspaceReceipt:
        try:
            check = _Inspection(context)
            check.check()
            if (journal_write_authorized is not True or
                    approved_digest != guest_operation_digest(endpoint, plan, action)):
                raise BackendFailure('backend_authority_required')
            if self.verify_endpoint is None or action == 'inspect' and self.inspect is None:
                raise BackendFailure('backend_plan_blocked')
            expected = _expected(plan, endpoint.instance_id)
            # PREPARE here means workspace preparation, never VM preparation.
            intent = BackendOperation(operation_id=plan.operation_id, backend_id='guest-workspace-copy',
                backend_version='1', configuration_digest=endpoint.transfer_digest(plan),
                plan_digest=guest_operation_digest(endpoint, plan, 'copy'), action=BackendAction.PREPARE)
            with self.journal.locked() as writer:
                current = writer.current()
                if action == 'copy':
                    if current is not None and current.status == 'uncertain':
                        raise BackendFailure('backend_reconciliation_required')
                    writer.create(intent)
                elif current != intent:
                    raise BackendFailure('backend_reconciliation_required')
                try:
                    check.check()
                    if self.verify_endpoint(endpoint, context) is not None:
                        raise BackendFailure('backend_invalid_result')
                    check.check()
                    if action == 'copy':
                        result = self.copy(endpoint, plan, endpoint.transfer_digest(plan), context)
                    else:
                        assert self.inspect is not None
                        result = self.inspect(endpoint, plan, context)
                    check.check()
                    if self.verify_endpoint(endpoint, context) is not None:
                        raise BackendFailure('backend_invalid_result')
                except (BackendFailure, DeadlineExceeded, RunCancelled):
                    raise
                except Exception:
                    # Normalize transport errors before the journal context can
                    # classify an OSError as a local journal-storage failure.
                    raise BackendFailure('backend_staging_failed') from None
                if type(result) is not GuestWorkspaceReceipt or result != expected:
                    raise BackendFailure('backend_invalid_result')
                check.check()
                writer.settle(intent, replace(intent, status='completed', observation='prepared'))
                return result
        except (BackendFailure, DeadlineExceeded, RunCancelled):
            raise
        except Exception:
            raise BackendFailure('backend_staging_failed') from None
