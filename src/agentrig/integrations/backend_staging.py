"""Digest-bound local staging for a separately verified guest mount.

Copying files does not prove that a VM sees the guest path. Mount verification
and workload isolation remain the selected backend's responsibilities.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
from types import MappingProxyType

from agentrig.capabilities.local_backend import BackendFailure, backend_digest
from agentrig.core import RunContext
from agentrig.integrations.backend_journal import _private


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkspaceStagePlan:
    host_root: Path = field(repr=False)
    guest_root: str = field(repr=False)
    operation_id: str
    files: Mapping[str, bytes] = field(repr=False)

    def __post_init__(self) -> None:
        for root_value in (str(self.host_root), self.guest_root):
            if (not root_value.startswith('/') or root_value.startswith('//') or root_value == '/' or
                    str(PurePosixPath(root_value)) != root_value or '..' in PurePosixPath(root_value).parts or
                    any(ord(char) < 32 for char in root_value)):
                raise ValueError('workspace root invalid')
        if re.fullmatch('[a-f0-9]{32}', self.operation_id) is None:
            raise ValueError('workspace operation invalid')
        copied = dict(self.files)
        if not 1 <= len(copied) <= 128:
            raise ValueError('workspace file count invalid')
        for name, value in copied.items():
            if (not isinstance(name, str) or len(name) > 256 or
                    re.fullmatch(r'[a-zA-Z0-9_-][a-zA-Z0-9_./-]*', name) is None or
                    any(part in ('', '.', '..') or part.startswith('.') for part in name.split('/')) or
                    len(name.split('/')) > 8 or not isinstance(value, bytes) or len(value) > 1024 * 1024):
                raise ValueError('workspace file invalid')
            if any(str(parent) in copied for parent in PurePosixPath(name).parents if str(parent) != '.'):
                raise ValueError('workspace file path conflict')
        if sum(map(len, copied.values())) > 16 * 1024 * 1024:
            raise ValueError('workspace size invalid')
        object.__setattr__(self, 'files', MappingProxyType(copied))

    @property
    def digest(self) -> str:
        return backend_digest({'schema': 'agentrig.workspace-stage.v1',
            'host_root': str(self.host_root), 'guest_root': self.guest_root,
            'operation_id': self.operation_id,
            'files': {name: {'sha256': hashlib.sha256(value).hexdigest(), 'size': len(value)}
                      for name, value in sorted(self.files.items())}})


@dataclass(frozen=True, slots=True, kw_only=True)
class StagedWorkspace:
    plan_digest: str
    host_directory: Path = field(repr=False)
    guest_directory: str = field(repr=False)
    file_count: int
    guest_visibility_verified: bool = field(default=False, init=False)


def _check(context: RunContext) -> None:
    context.cancellation.raise_if_cancelled()
    if context.deadline is not None:
        context.deadline.raise_if_expired(context.clock)


def _write(directory: int, name: str, payload: bytes) -> None:
    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.fsync(directory)


def stage_workspace(plan: WorkspaceStagePlan, approved_digest: str, context: RunContext) -> StagedWorkspace:
    """Publish one fresh operation directory; retain partial work on interruption.

    The application supplies an ancestor-verified private host mount directory.
    Existing operation directories are never adopted, overwritten or deleted.
    """
    _check(context)
    if approved_digest != plan.digest:
        raise BackendFailure('backend_authority_required')
    root: int | None = None
    operation: int | None = None
    try:
        root = os.open(plan.host_root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        _private(os.fstat(root), directory=True)
        os.mkdir(plan.operation_id, mode=0o700, dir_fd=root)
        os.fsync(root)
        operation = os.open(plan.operation_id, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root)
        info = os.fstat(operation)
        _private(info, directory=True)
        identity = {'schema': 'agentrig.workspace-stage-receipt.v1', 'plan_digest': plan.digest,
                    'device': info.st_dev, 'inode': info.st_ino, 'uid': info.st_uid,
                    'guest_visibility_verified': False}
        receipt = (json.dumps(identity, sort_keys=True) + '\n').encode()
        _write(operation, '.intent.json', receipt)
        for name, value in sorted(plan.files.items()):
            _check(context)
            current = operation
            opened: list[int] = []
            try:
                for component in name.split('/')[:-1]:
                    try:
                        os.mkdir(component, mode=0o700, dir_fd=current)
                        os.fsync(current)
                    except FileExistsError:
                        pass
                    current = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=current)
                    opened.append(current)
                    _private(os.fstat(current), directory=True)
                _write(current, name.split('/')[-1], value)
            finally:
                for fd in reversed(opened):
                    os.close(fd)
        _check(context)
        _write(operation, '.complete.json', receipt)
        return StagedWorkspace(plan_digest=plan.digest, host_directory=plan.host_root / plan.operation_id,
            guest_directory=plan.guest_root + '/' + plan.operation_id, file_count=len(plan.files))
    except BackendFailure:
        raise
    except (OSError, ValueError, TypeError):
        raise BackendFailure('backend_staging_failed') from None
    finally:
        if operation is not None:
            os.close(operation)
        if root is not None:
            os.close(root)
