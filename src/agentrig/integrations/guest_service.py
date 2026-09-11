"""Bounded one-request guest service and replaceable host protocol client.

Importing never starts a listener. Activation and VM forwarding qualification
belong to the selected backend. No key, token, shell or provider is used here.
"""

from collections.abc import Callable
from dataclasses import asdict
import os
from pathlib import Path
import socket
import stat
import struct
import sys
import time
from typing import Literal

from agentrig.capabilities import BackendFailure
from agentrig.core import DeadlineExceeded, RunCancelled, RunContext
from agentrig.integrations.backend_journal import _private
from agentrig.integrations.backend_staging import WorkspaceStagePlan
from agentrig.integrations.guest_operations import _Inspection, _expected, _inspect_guest_request
from agentrig.integrations.guest_workspace import (
    GuestWorkspaceEndpoint, GuestWorkspaceReceipt, _Stream, _encode, _guest_connection,
    _instance, _request, receive_guest_workspace, transfer_guest_workspace,
)

_SCHEMA = 'agentrig.guest-service.v1'
GuestPeerVerifier = Callable[[socket.socket, RunContext], object]


def _linux_peer_available() -> int:
    option = getattr(socket, 'SO_PEERCRED', None)
    if not sys.platform.startswith('linux') or type(option) is not int:
        raise BackendFailure('backend_unavailable')
    if os.getuid() in (0, 65534) or os.geteuid() != os.getuid():
        raise BackendFailure('backend_authority_required')
    return option


def verify_linux_guest_peer(connection: socket.socket, context: RunContext) -> None:
    """Require the service account's UID from the Linux connected Unix socket.

    This authenticates the guest-local forwarder process at connection time, not
    the original host caller or VM. The backend must authenticate that outer hop.
    Kernel PID/UID/GID metadata is not retained or returned to callers.
    """
    _Inspection(context).check()
    option = _linux_peer_available()
    try:
        if connection.family != socket.AF_UNIX or connection.type != socket.SOCK_STREAM:
            raise BackendFailure('backend_authority_required')
        raw = connection.getsockopt(socket.SOL_SOCKET, option, struct.calcsize('3i'))
        pid, uid, _ = struct.unpack('3i', raw)
        if pid <= 0 or uid != os.getuid():
            raise BackendFailure('backend_authority_required')
    except BackendFailure:
        raise
    except Exception:
        raise BackendFailure('backend_authority_required') from None


def handle_guest_request(connection: socket.socket, root: Path, instance_id: str,
                         context: RunContext, *, verify_peer: GuestPeerVerifier = verify_linux_guest_peer) -> GuestWorkspaceReceipt:
    """Consume one connected stream, authenticate before reads, close on all exits.

    The caller supplies an ancestor-verified private root. A replacement verifier
    is trusted application code, never selected by a request or saved receipt.
    """
    try:
        _Inspection(context).check()
        _instance(instance_id)
        if verify_peer(connection, context) is not None:
            raise BackendFailure('backend_authority_required')
        stream = _Stream(connection, context)
        command = stream.header()
        if (set(command) != {'schema', 'action', 'instance_id'} or command['schema'] != _SCHEMA or
                command['instance_id'] != instance_id or command['action'] not in ('copy', 'inspect')):
            raise BackendFailure('backend_staging_failed')
        stream.check()
        if command['action'] == 'copy':
            return receive_guest_workspace(connection, root, instance_id, context)
        request = stream.header()
        if request.get('instance_id') != instance_id:
            raise BackendFailure('backend_staging_failed')
        # Inspection never receives collateral bytes. Require EOF before opening
        # guest state so trailing data cannot be interpreted as a second command.
        stream.eof()
        result = _inspect_guest_request(root, request, context)
        stream.check()
        stream.send_header({'schema': _SCHEMA, **asdict(result)})
        connection.shutdown(socket.SHUT_WR)
        return result
    except (BackendFailure, DeadlineExceeded, RunCancelled):
        raise
    except Exception:
        raise BackendFailure('backend_staging_failed') from None
    finally:
        connection.close()


def request_guest_service(connection: socket.socket, plan: WorkspaceStagePlan, instance_id: str,
                          approved_digest: str, context: RunContext, *,
                          action: Literal['copy', 'inspect']) -> GuestWorkspaceReceipt:
    """One request on an owned authenticated stream; no retry or reconnect.

    This lower-level grant binds the plan. The backend/client must also bind the
    action and endpoint before opening the connection; journaled operations do so.
    """
    try:
        _Inspection(context).check()
        if approved_digest != plan.digest:
            raise BackendFailure('backend_authority_required')
        if action not in ('copy', 'inspect'):
            raise BackendFailure('backend_plan_changed')
        request = _request(plan, instance_id)
        stream = _Stream(connection, context)
        stream.send_header({'schema': _SCHEMA, 'action': action, 'instance_id': instance_id})
        if action == 'copy':
            return transfer_guest_workspace(connection, plan, instance_id, approved_digest, context)
        stream.send_header(request)
        connection.shutdown(socket.SHUT_WR)
        response = stream.header()
        stream.eof()
        expected = _expected(plan, instance_id)
        if _encode(response) != _encode({'schema': _SCHEMA, **asdict(expected)}):
            raise BackendFailure('backend_invalid_result')
        stream.check()
        return expected
    except (BackendFailure, DeadlineExceeded, RunCancelled):
        raise
    except Exception:
        raise BackendFailure('backend_staging_failed') from None
    finally:
        connection.close()


def copy_guest_service(endpoint: GuestWorkspaceEndpoint, plan: WorkspaceStagePlan,
                       approved_digest: str, context: RunContext) -> GuestWorkspaceReceipt:
    """GuestCopier adapter for JournaledGuestWorkspaces after backend verification."""
    _Inspection(context).check()
    if approved_digest != endpoint.transfer_digest(plan):
        raise BackendFailure('backend_authority_required')
    _expected(plan, endpoint.instance_id)
    with _guest_connection(endpoint, context) as connection:
        return request_guest_service(connection, plan, endpoint.instance_id, plan.digest, context, action='copy')


def inspect_guest_service(endpoint: GuestWorkspaceEndpoint, plan: WorkspaceStagePlan,
                          context: RunContext) -> GuestWorkspaceReceipt:
    """Trusted GuestInspector adapter, called only after fresh journal authority.

    The application supplies a freshly authenticated endpoint. The inspector does
    not reconstruct execution authority from the stored journal or receipt.
    """
    _Inspection(context).check()
    _expected(plan, endpoint.instance_id)
    with _guest_connection(endpoint, context) as connection:
        return request_guest_service(connection, plan, endpoint.instance_id, plan.digest, context, action='inspect')


def _service_directory(path: Path) -> int:
    """Anchor ancestors without following links; final directory must be private."""
    if not path.is_absolute() or str(path).startswith('//') or '..' in path.parts or path == Path('/'):
        raise BackendFailure('backend_staging_failed')
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for component in path.parts[1:]:
            prior = os.fstat(fd)
            if (prior.st_uid not in (0, os.getuid()) or
                    prior.st_mode & 0o022 and not (prior.st_uid == 0 and prior.st_mode & stat.S_ISVTX)):
                raise BackendFailure('backend_staging_failed')
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        _private(os.fstat(fd), directory=True)
        return fd
    except BaseException:
        os.close(fd)
        raise


def serve_guest_once(root: Path, socket_directory: Path, instance_id: str, context: RunContext, *,
                      service_authorized: bool = False,
                      verify_peer: GuestPeerVerifier = verify_linux_guest_peer) -> GuestWorkspaceReceipt:
    """Bind workspace.sock, accept exactly one request, then close the listener.

    Explicit application activation only. Both existing directories are anchored
    before binding. Existing sockets/files are never adopted or removed. Waiting
    has a 60-second wall-clock ceiling and honors the context deadline. A narrow
    temporary process umask makes the socket 0600 at creation, before listening.
    Retain the socket entry after closing, allowing the host's final endpoint
    identity check and preserving evidence. Backend-owned cleanup/reactivation
    requires separate review; an existing entry always blocks a new listener.
    The caller owns process isolation; this function does not daemonize or install.
    """
    opened: list[int] = []
    listener: socket.socket | None = None
    bound: os.stat_result | None = None
    socket_fd: int | None = None
    try:
        check = _Inspection(context)
        check.check()
        if service_authorized is not True:
            raise BackendFailure('backend_authority_required')
        _instance(instance_id)
        if verify_peer is verify_linux_guest_peer:
            _linux_peer_available()
        name = socket_directory / 'workspace.sock'
        if len(str(name).encode()) > 103:
            raise BackendFailure('backend_staging_failed')
        root_fd = _service_directory(root)
        opened.append(root_fd)
        socket_fd = _service_directory(socket_directory)
        opened.append(socket_fd)
        root_identity = os.fstat(root_fd)
        socket_identity = os.fstat(socket_fd)

        def directories_unchanged() -> None:
            for path, original in [(root, root_identity), (socket_directory, socket_identity)]:
                current = path.lstat()
                _private(current, directory=True)
                if (current.st_dev, current.st_ino) != (original.st_dev, original.st_ino):
                    raise BackendFailure('backend_staging_failed')

        directories_unchanged()
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        previous_umask = os.umask(0o177)
        try:
            listener.bind(str(name))
        finally:
            os.umask(previous_umask)
        bound = os.stat('workspace.sock', dir_fd=socket_fd, follow_symlinks=False)
        if not stat.S_ISSOCK(bound.st_mode) or stat.S_IMODE(bound.st_mode) != 0o600 or bound.st_uid != os.getuid():
            raise BackendFailure('backend_staging_failed')
        directories_unchanged()
        listener.listen(1)
        listener.settimeout(.05)
        expires = time.monotonic() + 60
        while True:
            check.check()
            if time.monotonic() >= expires:
                raise BackendFailure('backend_staging_failed')
            try:
                connection, _ = listener.accept()
                break
            except socket.timeout:
                continue
        try:
            directories_unchanged()
            result = handle_guest_request(connection, root, instance_id, context, verify_peer=verify_peer)
            directories_unchanged()
            return result
        finally:
            connection.close()
    except (BackendFailure, DeadlineExceeded, RunCancelled):
        raise
    except Exception:
        raise BackendFailure('backend_staging_failed') from None
    finally:
        if listener is not None:
            listener.close()
        for fd in reversed(opened):
            os.close(fd)
