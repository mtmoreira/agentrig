"""One-shot workspace transfer over an already authenticated local stream.

The backend supplies and authenticates the connected Unix stream (including any
VM forwarding) and the guest service's private root. This module never discovers
endpoints, launches a service, executes collateral, or grants runtime readiness.
"""

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat
import struct
import time
from typing import Protocol, cast

from agentrig.capabilities import BackendArtifact, BackendArtifactManifest, BackendFailure
from agentrig.capabilities.local_backend import backend_digest, _digest
from agentrig.core import DeadlineExceeded, JsonValue, RunContext
from agentrig.integrations.backend_journal import _private
from agentrig.integrations.backend_staging import WorkspaceStagePlan, _write

_SCHEMA = 'agentrig.guest-workspace.v1'
_HEADER_LIMIT = 65536


@dataclass(frozen=True, slots=True, kw_only=True)
class GuestWorkspaceEndpoint:
    """Fresh backend observation of its private workspace.sock forwarding path.

    This is an identity binding, not an authentication secret or persisted
    authority. The backend must validate all ancestors and the VM forwarding.
    """

    root: Path = field(repr=False)
    instance_id: str
    device: int
    inode: int
    uid: int

    def __post_init__(self) -> None:
        _instance(self.instance_id)
        value = str(self.root)
        if (not self.root.is_absolute() or '..' in self.root.parts or value.startswith('//') or
                self.root == Path('/') or any(ord(c) < 32 or ord(c) == 127 for c in value)):
            raise ValueError('guest workspace endpoint invalid')
        for number in (self.device, self.inode, self.uid):
            if type(number) is not int or number < 0:
                raise ValueError('guest workspace endpoint identity invalid')

    def transfer_digest(self, plan: WorkspaceStagePlan) -> str:
        return backend_digest({'schema': 'agentrig.guest-workspace-authority.v1',
            'root': str(self.root), 'instance_id': self.instance_id, 'device': self.device,
            'inode': self.inode, 'uid': self.uid, 'plan_digest': plan.digest})


@dataclass(frozen=True, slots=True, kw_only=True)
class GuestWorkspaceReceipt:
    request_digest: str
    plan_digest: str
    operation_id: str
    file_count: int
    total_bytes: int
    # This is a copy observation. It cannot prove container mount/UID access.
    workload_access_verified: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        _digest(self.request_digest)
        _digest(self.plan_digest)
        _instance(self.operation_id)
        if (type(self.file_count) is not int or not 1 <= self.file_count <= 128 or
                type(self.total_bytes) is not int or not 0 <= self.total_bytes <= 16 * 1024**2):
            raise ValueError('guest workspace receipt invalid')


def _instance(value: str) -> None:
    if not isinstance(value, str) or re.fullmatch('[a-f0-9]{32}', value) is None:
        raise ValueError('guest workspace identity invalid')


def _encode(value: dict[str, JsonValue]) -> bytes:
    result = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()
    if len(result) > _HEADER_LIMIT:
        raise ValueError('guest workspace header invalid')
    return result


def _object(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('guest workspace header invalid')
        result[key] = value
    return result


class _Stream:
    def __init__(self, connection: socket.socket, context: RunContext) -> None:
        self.connection = connection
        self.context = context
        self.expires = time.monotonic() + 60
        self.check()
        if context.deadline is None or connection.family != socket.AF_UNIX or connection.type != socket.SOCK_STREAM:
            raise ValueError('guest workspace stream invalid')
        connection.settimeout(0.05)

    def check(self) -> None:
        self.context.cancellation.raise_if_cancelled()
        if self.context.deadline is not None:
            self.context.deadline.raise_if_expired(self.context.clock)
        if time.monotonic() >= self.expires:
            raise BackendFailure('backend_staging_failed')

    def read(self, count: int) -> bytes:
        result = bytearray()
        while len(result) < count:
            self.check()
            try:
                chunk = self.connection.recv(min(65536, count - len(result)))
            except socket.timeout:
                continue
            if not chunk:
                raise ValueError('guest workspace stream incomplete')
            result.extend(chunk)
        return bytes(result)

    def write(self, data: bytes) -> None:
        remaining = memoryview(data)
        while remaining:
            self.check()
            try:
                count = self.connection.send(remaining[:65536])
            except socket.timeout:
                continue
            if not count:
                raise ValueError('guest workspace stream incomplete')
            remaining = remaining[count:]

    def header(self) -> dict[str, JsonValue]:
        size, = struct.unpack('!I', self.read(4))
        if not 1 <= size <= _HEADER_LIMIT:
            raise ValueError('guest workspace header invalid')
        value = json.loads(self.read(size), object_pairs_hook=_object)
        if not isinstance(value, dict):
            raise ValueError('guest workspace header invalid')
        return cast(dict[str, JsonValue], value)

    def send_header(self, value: dict[str, JsonValue]) -> None:
        data = _encode(value)
        self.write(struct.pack('!I', len(data)) + data)

    def eof(self) -> None:
        while True:
            self.check()
            try:
                if self.connection.recv(1):
                    raise ValueError('guest workspace trailing data')
                return
            except socket.timeout:
                continue


def _manifest(header: dict[str, JsonValue]) -> BackendArtifactManifest:
    if set(header) != {'schema', 'instance_id', 'plan_digest', 'operation_id', 'guest_root', 'files'}:
        raise ValueError('guest workspace header invalid')
    if header['schema'] != _SCHEMA:
        raise ValueError('guest workspace schema invalid')
    for name in ('instance_id', 'operation_id'):
        value = header[name]
        if not isinstance(value, str):
            raise ValueError('guest workspace identity invalid')
        _instance(value)
    digest = header['plan_digest']
    if not isinstance(digest, str):
        raise ValueError('guest workspace digest invalid')
    _digest(digest)
    rows = header['files']
    if not isinstance(rows, list) or not 1 <= len(rows) <= 128:
        raise ValueError('guest workspace files invalid')
    artifacts: list[BackendArtifact] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {'path', 'size', 'sha256'}:
            raise ValueError('guest workspace file invalid')
        path, size, sha = row['path'], row['size'], row['sha256']
        if (not isinstance(path, str) or '/' not in path or type(size) is not int or
                not 0 <= size <= 1024**2 or not isinstance(sha, str)):
            raise ValueError('guest workspace file invalid')
        artifacts.append(BackendArtifact(artifact_id='file-' + str(index), path=path,
                                        size_bytes=size, sha256=sha))
    manifest = BackendArtifactManifest(artifacts=tuple(artifacts))
    if (sum(a.size_bytes for a in manifest.artifacts) > 16 * 1024**2 or
            [a.path for a in artifacts] != [a.path for a in manifest.artifacts]):
        raise ValueError('guest workspace files invalid')
    return manifest


def _request(plan: WorkspaceStagePlan, instance_id: str) -> dict[str, JsonValue]:
    header: dict[str, JsonValue] = {'schema': _SCHEMA, 'instance_id': instance_id,
        'plan_digest': plan.digest, 'operation_id': plan.operation_id, 'guest_root': plan.guest_root,
        'files': [{'path': name, 'size': len(value), 'sha256': 'sha256:' + hashlib.sha256(value).hexdigest()}
                  for name, value in sorted(plan.files.items())]}
    _manifest(header)
    _encode(header)
    return header


def _receipt(header: dict[str, JsonValue], manifest: BackendArtifactManifest) -> dict[str, JsonValue]:
    return {'schema': _SCHEMA, 'request_digest': backend_digest(header),
            'plan_digest': header['plan_digest'], 'operation_id': header['operation_id'],
            'file_count': len(manifest.artifacts), 'total_bytes': sum(a.size_bytes for a in manifest.artifacts),
            'workload_access_verified': False}


class _Check(Protocol):
    def check(self) -> None: ...


def _verify_copy(operation: int, manifest: BackendArtifactManifest, stream: _Check) -> None:
    """Reopen the exact copied paths before acknowledging; never follow links."""
    for artifact in manifest.artifacts:
        current = operation
        opened: list[int] = []
        try:
            for component in artifact.path.split('/')[:-1]:
                current = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=current)
                opened.append(current)
                info = os.fstat(current)
                if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o755:
                    raise ValueError('guest workspace directory changed')
            name = artifact.path.split('/')[-1]
            descriptor = os.open(name, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=current)
            opened.append(descriptor)
            before = os.fstat(descriptor)
            if (not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode) != 0o644 or
                    before.st_uid != os.getuid() or before.st_nlink != 1 or before.st_size != artifact.size_bytes):
                raise ValueError('guest workspace file changed')
            digest = hashlib.sha256()
            remaining = artifact.size_bytes
            while remaining:
                stream.check()
                chunk = os.read(descriptor, min(65536, remaining))
                if not chunk:
                    raise ValueError('guest workspace file changed')
                digest.update(chunk)
                remaining -= len(chunk)
            after = os.fstat(descriptor)
            current_info = os.stat(name, dir_fd=current, follow_symlinks=False)
            def identity(info: os.stat_result) -> tuple[int, ...]:
                return (info.st_dev, info.st_ino, info.st_uid, info.st_mode, info.st_size,
                        info.st_nlink, info.st_mtime_ns, info.st_ctime_ns)
            if (os.read(descriptor, 1) or 'sha256:' + digest.hexdigest() != artifact.sha256 or
                    identity(before) != identity(after) or identity(after) != identity(current_info)):
                raise ValueError('guest workspace file changed')
        finally:
            for descriptor in reversed(opened):
                os.close(descriptor)


def transfer_guest_workspace(connection: socket.socket, plan: WorkspaceStagePlan,
                             instance_id: str, approved_digest: str,
                             context: RunContext) -> GuestWorkspaceReceipt:
    """Consume an owned, authenticated stream once; close it on every outcome.

    Caller must journal fresh authority and verify backend/endpoint identity
    before supplying this stream. No automatic reconnect or replay is permitted.
    A failure after sending may have copied files: preserve it as uncertainty.
    No path on the host is opened, shared, chmodded or sent as executable argv.
    """
    try:
        stream = _Stream(connection, context)
        if approved_digest != plan.digest:
            raise BackendFailure('backend_authority_required')
        header = _request(plan, instance_id)
        manifest = _manifest(header)
        stream.send_header(header)
        for name in sorted(plan.files):
            stream.write(plan.files[name])
        connection.shutdown(socket.SHUT_WR)
        response = stream.header()
        stream.eof()
        if _encode(response) != _encode(_receipt(header, manifest)):
            raise ValueError('guest workspace receipt invalid')
        stream.check()
        return GuestWorkspaceReceipt(request_digest=backend_digest(header), plan_digest=plan.digest,
            operation_id=plan.operation_id, file_count=len(manifest.artifacts),
            total_bytes=sum(a.size_bytes for a in manifest.artifacts))
    except DeadlineExceeded:
        raise
    except (OSError, ValueError, TypeError, RecursionError):
        raise BackendFailure('backend_staging_failed') from None
    finally:
        connection.close()


def copy_to_guest(endpoint: GuestWorkspaceEndpoint, plan: WorkspaceStagePlan,
                  approved_digest: str, context: RunContext) -> GuestWorkspaceReceipt:
    """Connect once to the exact private socket; no discovery, retries or TCP.

    Fresh authority covers both the reviewed plan and the backend-observed local
    socket identity. The backend still owns VM peer/forwarding authentication and
    the durable operation journal. Any failure after sending is uncertain.
    """
    descriptor: int | None = None
    connection: socket.socket | None = None
    try:
        context.cancellation.raise_if_cancelled()
        if context.deadline is None:
            raise ValueError('guest workspace deadline required')
        context.deadline.raise_if_expired(context.clock)
        if approved_digest != endpoint.transfer_digest(plan):
            raise BackendFailure('backend_authority_required')
        _request(plan, endpoint.instance_id)
        descriptor = os.open(endpoint.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        bound = os.fstat(descriptor)
        _private(bound, directory=True)

        def check_endpoint() -> None:
            assert descriptor is not None
            context.cancellation.raise_if_cancelled()
            assert context.deadline is not None
            context.deadline.raise_if_expired(context.clock)
            root_info = endpoint.root.lstat()
            _private(root_info, directory=True)
            info = os.stat('workspace.sock', dir_fd=descriptor, follow_symlinks=False)
            if ((root_info.st_dev, root_info.st_ino) != (bound.st_dev, bound.st_ino) or
                    not stat.S_ISSOCK(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600 or
                    info.st_nlink != 1 or info.st_uid != os.getuid() or
                    (info.st_dev, info.st_ino, info.st_uid) != (endpoint.device, endpoint.inode, endpoint.uid)):
                raise ValueError('guest workspace endpoint changed')

        check_endpoint()
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.settimeout(min(.05, context.deadline.remaining_seconds(context.clock)))
        connection.connect(str(endpoint.root / 'workspace.sock'))
        check_endpoint()
        result = transfer_guest_workspace(connection, plan, endpoint.instance_id, plan.digest, context)
        check_endpoint()
        return result
    except DeadlineExceeded:
        raise
    except (OSError, ValueError, TypeError, RecursionError):
        raise BackendFailure('backend_staging_failed') from None
    finally:
        if connection is not None:
            connection.close()
        if descriptor is not None:
            os.close(descriptor)


def receive_guest_workspace(connection: socket.socket, root: Path, instance_id: str,
                            context: RunContext) -> GuestWorkspaceReceipt:
    """Guest service handler for one authenticated connection, never a listener.

    Backend supplies an ancestor-verified private root owned by its guest user.
    Each fresh operation has a 0700 parent with private intent/completion records.
    Only descendants intended as individual read-only container bind roots are
    0755/0644. Container UID mapping and mount access still require a real test.
    Incomplete operations remain, and every reuse fails without overwriting.
    """
    root_fd: int | None = None
    operation: int | None = None
    try:
        stream = _Stream(connection, context)
        _instance(instance_id)
        if not root.is_absolute() or '..' in root.parts or root == Path('/'):
            raise ValueError('guest workspace root invalid')
        header = stream.header()
        manifest = _manifest(header)
        if header['instance_id'] != instance_id or header['guest_root'] != str(root):
            raise ValueError('guest workspace binding invalid')
        operation_id = cast(str, header['operation_id'])
        root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        _private(os.fstat(root_fd), directory=True)
        stream.check()
        os.mkdir(operation_id, mode=0o700, dir_fd=root_fd)
        os.fsync(root_fd)
        operation = os.open(operation_id, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd)
        _private(os.fstat(operation), directory=True)
        result = _receipt(header, manifest)
        info = os.fstat(operation)
        record = dict(result, device=info.st_dev, inode=info.st_ino, uid=info.st_uid)
        _write(operation, '.intent.json', _encode(record))
        for artifact in manifest.artifacts:
            stream.check()
            payload = stream.read(artifact.size_bytes)
            if 'sha256:' + hashlib.sha256(payload).hexdigest() != artifact.sha256:
                raise ValueError('guest workspace content invalid')
            current = operation
            opened: list[int] = []
            try:
                for component in artifact.path.split('/')[:-1]:
                    try:
                        os.mkdir(component, mode=0o700, dir_fd=current)
                    except FileExistsError:
                        pass
                    current = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=current)
                    opened.append(current)
                    directory_info = os.fstat(current)
                    if directory_info.st_uid != os.getuid() or directory_info.st_mode & 0o7022:
                        raise ValueError('guest workspace directory invalid')
                    os.fchmod(current, 0o755)
                _write(current, artifact.path.split('/')[-1], payload)
                descriptor = os.open(artifact.path.split('/')[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=current)
                try:
                    _private(os.fstat(descriptor), directory=False)
                    os.fchmod(descriptor, 0o644)
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                for descriptor in reversed(opened):
                    os.fsync(descriptor)
                os.fsync(operation)
            finally:
                for descriptor in reversed(opened):
                    os.close(descriptor)
        stream.eof()
        stream.check()
        _verify_copy(operation, manifest, stream)
        current_root = root.lstat()
        bound_root = os.fstat(root_fd)
        current_operation = os.stat(operation_id, dir_fd=root_fd, follow_symlinks=False)
        _private(current_root, directory=True)
        _private(current_operation, directory=True)
        if ((current_root.st_dev, current_root.st_ino) != (bound_root.st_dev, bound_root.st_ino) or
                (current_operation.st_dev, current_operation.st_ino) != (info.st_dev, info.st_ino)):
            raise ValueError('guest workspace root changed')
        stream.check()
        _write(operation, '.complete.json', _encode(record))
        stream.send_header(result)
        connection.shutdown(socket.SHUT_WR)
        return GuestWorkspaceReceipt(request_digest=backend_digest(header), plan_digest=cast(str, header['plan_digest']),
            operation_id=operation_id, file_count=len(manifest.artifacts),
            total_bytes=sum(a.size_bytes for a in manifest.artifacts))
    except DeadlineExceeded:
        raise
    except (OSError, ValueError, TypeError, RecursionError):
        raise BackendFailure('backend_staging_failed') from None
    finally:
        connection.close()
        if operation is not None:
            os.close(operation)
        if root_fd is not None:
            os.close(root_fd)
