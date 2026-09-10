"""Private, durable backend operation journal. Importing performs no filesystem I/O."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import stat
from typing import Iterator

from agentrig.capabilities.backend_operations import BackendOperation, BackendOperationWriter
from agentrig.capabilities.local_backend import BackendAction, BackendFailure


def _private(info: os.stat_result, *, directory: bool) -> None:
    if (info.st_uid != os.getuid() or info.st_mode & 0o077 or
            (not stat.S_ISDIR(info.st_mode) if directory else
             not stat.S_ISREG(info.st_mode) or info.st_nlink != 1)):
        raise BackendFailure('backend_journal_invalid')


class _Writer:
    def __init__(self, directory: int) -> None:
        self.directory = directory
        info = os.fstat(directory)
        self.binding = {'device': info.st_dev, 'inode': info.st_ino, 'uid': info.st_uid}

    def _read(self, name: str) -> dict[str, object] | None:
        try:
            return self._read_value(name)
        except (OSError, ValueError, TypeError, KeyError):
            raise BackendFailure('backend_journal_invalid') from None

    def _read_value(self, name: str) -> dict[str, object] | None:
        try:
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=self.directory)
        except FileNotFoundError:
            return None
        with os.fdopen(fd, 'rb') as stream:
            info = os.fstat(stream.fileno())
            _private(info, directory=False)
            if info.st_size > 8192:
                raise BackendFailure('backend_journal_invalid')
            data = stream.read(8193)
        if len(data) > 8192:
            raise BackendFailure('backend_journal_invalid')
        def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, value in pairs:
                if key in result:
                    raise BackendFailure('backend_journal_invalid')
                result[key] = value
            return result
        value = json.loads(data, object_pairs_hook=unique)
        if not isinstance(value, dict):
            raise BackendFailure('backend_journal_invalid')
        return value

    def current(self) -> BackendOperation | None:
        pointer = self._read('current.json')
        if pointer is None:
            return None
        if set(pointer) != {'binding', 'operation_id'} or pointer['binding'] != self.binding:
            raise BackendFailure('backend_journal_invalid')
        operation_id = pointer['operation_id']
        if not isinstance(operation_id, str) or len(operation_id) != 32 or any(c not in '0123456789abcdef' for c in operation_id):
            raise BackendFailure('backend_journal_invalid')
        value = self._read('operation-' + operation_id + '.json')
        if value is None or set(value) != {'schema', 'binding', 'operation'} or value['binding'] != self.binding or value['schema'] != 'agentrig.backend-operation.v1':
            raise BackendFailure('backend_journal_invalid')
        operation = value['operation']
        if not isinstance(operation, dict) or set(operation) != set(BackendOperation.__dataclass_fields__):
            raise BackendFailure('backend_journal_invalid')
        operation['action'] = BackendAction(operation['action'])
        result = BackendOperation(**operation)
        if result.operation_id != operation_id:
            raise BackendFailure('backend_journal_invalid')
        return result

    def _write(self, name: str, value: dict[str, object], *, fresh: bool = False) -> None:
        payload = (json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n').encode()
        if len(payload) > 8192:
            raise BackendFailure('backend_journal_invalid')
        # Fixed staging name under the exclusive lock. Crash leftovers are not
        # overwritten or removed automatically; they require owner inspection.
        temporary = '.pending.json'
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=self.directory)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if fresh:
            # link is atomic and refuses an existing operation ID. Never replace
            # archived operations when beginning a new lifecycle action.
            os.link(temporary, name, src_dir_fd=self.directory, dst_dir_fd=self.directory,
                    follow_symlinks=False)
            os.unlink(temporary, dir_fd=self.directory)
        else:
            os.replace(temporary, name, src_dir_fd=self.directory, dst_dir_fd=self.directory)
        os.fsync(self.directory)

    def _envelope(self, operation: BackendOperation) -> dict[str, object]:
        return {'schema': 'agentrig.backend-operation.v1', 'binding': self.binding,
                'operation': asdict(operation)}

    def create(self, operation: BackendOperation) -> None:
        current = self.current()
        if operation.status != 'uncertain' or current is not None and current.status == 'uncertain':
            raise BackendFailure('backend_reconciliation_required')
        # Reject reuse before creating a staging file, retaining earlier records.
        if self._read('operation-' + operation.operation_id + '.json') is not None:
            raise BackendFailure('backend_operation_reused')
        self._write('operation-' + operation.operation_id + '.json', self._envelope(operation), fresh=True)
        self._write('current.json', {'binding': self.binding, 'operation_id': operation.operation_id})

    def settle(self, expected: BackendOperation, result: BackendOperation) -> None:
        if (self.current() != expected or expected.status != 'uncertain' or
                result.status not in {'completed', 'reconciled'} or
                replace(result, status='uncertain', observation=None) != expected):
            raise BackendFailure('backend_journal_invalid')
        self._write('operation-' + result.operation_id + '.json', self._envelope(result))


class LocalBackendJournal:
    """One existing private directory per instance, never created implicitly.

    The application supplies a trusted ancestor-verified directory. This adapter
    rejects links at the directory entry, binds records to its local filesystem
    identity, and uses descriptor-relative files throughout the locked operation.
    Copied records and stored PIDs cannot confer runtime authority.
    """

    def __init__(self, directory: Path) -> None:
        if not directory.is_absolute() or '..' in directory.parts or directory == Path('/'):
            raise ValueError('backend journal directory invalid')
        self._directory = directory

    def inspect(self) -> BackendOperation | None:
        """Read a local snapshot without a lock file, directory creation or effects.

        This diagnostic snapshot is not an execution gate. Mutating callers use
        locked() and recheck current state while holding its exclusive lease.
        """
        directory: int | None = None
        try:
            directory = os.open(self._directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            _private(os.fstat(directory), directory=True)
            writer = _Writer(directory)
            before = writer._read('current.json')
            result = writer.current()
            if writer._read('current.json') != before:
                raise BackendFailure('backend_journal_busy')
            return result
        except BackendFailure:
            raise
        except (OSError, ValueError, TypeError, KeyError):
            raise BackendFailure('backend_journal_invalid') from None
        finally:
            if directory is not None:
                os.close(directory)

    @contextmanager
    def locked(self) -> Iterator[BackendOperationWriter]:
        directory: int | None = None
        lock: int | None = None
        writer: _Writer | None = None
        try:
            if os.name != 'posix':
                raise BackendFailure('backend_journal_invalid')
            import fcntl
            directory = os.open(self._directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            _private(os.fstat(directory), directory=True)
            lock = os.open('.operation.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK,
                           0o600, dir_fd=directory)
            _private(os.fstat(lock), directory=False)
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise BackendFailure('backend_journal_busy') from None
            writer = _Writer(directory)
            yield writer
        except BackendFailure:
            raise
        except (OSError, ValueError, TypeError, KeyError):
            raise BackendFailure('backend_journal_invalid') from None
        finally:
            if writer is not None:
                writer.directory = -1
            if lock is not None:
                os.close(lock)
            if directory is not None:
                os.close(directory)
