"""Real Unix protocol/files with synthetic peer verification; no VM or provider."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import os
from pathlib import Path
import socket
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from agentrig.capabilities import BackendFailure
from agentrig.core import DeadlineExceeded
from agentrig.integrations import (
    GuestWorkspaceEndpoint, JournaledGuestWorkspaces, LocalBackendJournal,
    copy_guest_service, guest_operation_digest, handle_guest_request, inspect_guest_service,
    request_guest_service, serve_guest_once,
)
from agentrig.integrations.guest_workspace import _Stream, _request
from tests.integration import test_guest_workspace as fixture


class GuestServiceTests(unittest.TestCase):
    def setUp(self):
        self.f = fixture.GuestWorkspaceTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.calls = []
        self.verify = lambda connection, context: self.calls.append('peer')

    def exchange(self, action, plan=None, verify=None):
        plan = self.f.plan if plan is None else plan
        client, server = socket.socketpair()
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(handle_guest_request, server, self.f.guest, self.f.instance,
                self.f.context, verify_peer=self.verify if verify is None else verify)
            try:
                result = request_guest_service(client, plan, self.f.instance, plan.digest,
                    self.f.context, action=action)
            except BaseException:
                try:
                    future.result(timeout=4)
                except Exception:
                    pass
                raise
            self.assertEqual(result, future.result(timeout=4))
        self.assertEqual(client.fileno(), -1)
        self.assertEqual(server.fileno(), -1)
        return result

    def test_copy_then_manifest_only_inspection_preserves_guest_bytes(self):
        result = self.exchange('copy')
        before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in self.f.guest.rglob('*') if p.is_file()}
        self.assertEqual(self.exchange('inspect'), result)
        self.assertEqual(before, {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in self.f.guest.rglob('*') if p.is_file()})
        self.assertEqual(self.calls, ['peer', 'peer'])
        self.assertFalse(result.workload_access_verified)

    def test_authentication_failure_precedes_request_read_and_guest_write(self):
        client, server = socket.socketpair()
        self.addCleanup(client.close)
        def denied(*args):
            raise ValueError('private peer failure')
        with patch('agentrig.integrations.guest_service._Stream.header') as header:
            with self.assertRaisesRegex(BackendFailure, '^backend_staging_failed$'):
                handle_guest_request(server, self.f.guest, self.f.instance, self.f.context, verify_peer=denied)
            header.assert_not_called()
        self.assertEqual(client.recv(1), b'')
        self.assertEqual(list(self.f.guest.iterdir()), [])

    def test_missing_operation_and_corrupt_bytes_never_return_success(self):
        with self.assertRaises(BackendFailure):
            self.exchange('inspect')
        self.exchange('copy')
        (self.f.guest / self.f.plan.operation_id / 'input/design.sv').write_bytes(b'corrupt')
        with self.assertRaises(BackendFailure):
            self.exchange('inspect')

    def test_malformed_dispatch_and_wrong_instance_fail_closed(self):
        for command in [{'schema': 'wrong'},
                {'schema': 'agentrig.guest-service.v1', 'action': 'inspect', 'instance_id': 'c' * 32},
                {'schema': 'agentrig.guest-service.v1', 'action': 'execute', 'instance_id': self.f.instance}]:
            client, server = socket.socketpair()
            with client:
                _Stream(client, self.f.context).send_header(command)
                client.shutdown(socket.SHUT_WR)
                with self.assertRaises(BackendFailure):
                    handle_guest_request(server, self.f.guest, self.f.instance, self.f.context, verify_peer=self.verify)
        self.assertEqual(list(self.f.guest.iterdir()), [])

    def test_inspection_trailing_payload_rejected_before_guest_read(self):
        client, server = socket.socketpair()
        with client:
            stream = _Stream(client, self.f.context)
            stream.send_header({'schema': 'agentrig.guest-service.v1', 'action': 'inspect', 'instance_id': self.f.instance})
            stream.send_header(_request(self.f.plan, self.f.instance))
            stream.write(b'unexpected payload')
            client.shutdown(socket.SHUT_WR)
            with patch('agentrig.integrations.guest_service._inspect_guest_request') as inspect:
                with self.assertRaises(BackendFailure):
                    handle_guest_request(server, self.f.guest, self.f.instance, self.f.context, verify_peer=self.verify)
                inspect.assert_not_called()

    def test_no_authority_or_deadline_sends_nothing(self):
        for ctx, grant in [(self.f.context, 'sha256:' + '0' * 64),
                           (replace(self.f.context, deadline=None), self.f.plan.digest)]:
            client, server = socket.socketpair()
            with server:
                with self.assertRaises(BackendFailure):
                    request_guest_service(client, self.f.plan, self.f.instance, grant, ctx, action='copy')
                self.assertEqual(server.recv(1), b'')

    def test_waiting_for_dispatch_honors_deadline(self):
        client, server = socket.socketpair()
        with client:
            with self.assertRaises(DeadlineExceeded):
                handle_guest_request(server, self.f.guest, self.f.instance,
                    self.f.make_context(.02), verify_peer=self.verify)
        self.assertEqual(server.fileno(), -1)

    def test_journal_recovery_uses_inspection_protocol_without_resending(self):
        journal_root = self.f.root / 'journal'
        journal_root.mkdir(mode=0o700)
        journal = LocalBackendJournal(journal_root)
        endpoint = GuestWorkspaceEndpoint(root=self.f.host, instance_id=self.f.instance, device=0, inode=0, uid=os.getuid())
        def copy(*args):
            self.exchange('copy')
            raise OSError('lost reply')
        driver = JournaledGuestWorkspaces(journal=journal, verify_endpoint=lambda *args: None,
            copy=copy, inspect=lambda *args: self.exchange('inspect'))
        with self.assertRaises(BackendFailure):
            driver.apply(endpoint, self.f.plan, guest_operation_digest(endpoint, self.f.plan, 'copy'),
                self.f.context, journal_write_authorized=True)
        driver.apply(endpoint, self.f.plan, guest_operation_digest(endpoint, self.f.plan, 'inspect'),
            self.f.context, journal_write_authorized=True, action='inspect')
        self.assertEqual(journal.inspect().status, 'completed')
        self.assertEqual(self.calls, ['peer', 'peer'])

    def socket_root(self):
        temporary = tempfile.TemporaryDirectory(prefix='agentrig-svc-', dir='/tmp')
        self.addCleanup(temporary.cleanup)
        return Path(temporary.name).resolve()

    def test_listener_copy_and_inspect_adapters_keep_owned_endpoint_evidence(self):
        for action in ['copy', 'inspect']:
            socket_root = self.socket_root()
            ready = threading.Event()
            original_listen = socket.socket.listen
            def listening(listener, backlog):
                original_listen(listener, backlog)
                ready.set()
            with patch.object(socket.socket, 'listen', listening), ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(serve_guest_once, self.f.guest, socket_root, self.f.instance,
                    self.f.context, service_authorized=True, verify_peer=self.verify)
                until = time.monotonic() + 1
                while not ready.wait(.005):
                    if future.done():
                        future.result()
                    self.assertLess(time.monotonic(), until)
                info = (socket_root / 'workspace.sock').lstat()
                endpoint = GuestWorkspaceEndpoint(root=socket_root, instance_id=self.f.instance,
                    device=info.st_dev, inode=info.st_ino, uid=info.st_uid)
                if action == 'copy':
                    result = copy_guest_service(endpoint, self.f.plan, endpoint.transfer_digest(self.f.plan), self.f.context)
                else:
                    result = inspect_guest_service(endpoint, self.f.plan, self.f.context)
                self.assertEqual(result, future.result(timeout=4))
            self.assertTrue((socket_root / 'workspace.sock').is_socket())
            with self.assertRaises(BackendFailure):
                serve_guest_once(self.f.guest, socket_root, self.f.instance, self.f.context,
                    service_authorized=True, verify_peer=self.verify)
            self.assertEqual((socket_root / 'workspace.sock').lstat().st_ino, info.st_ino)

    def test_existing_entry_and_missing_authority_are_preserved(self):
        root = self.socket_root()
        target = root / 'workspace.sock'
        target.write_bytes(b'unrelated entry')
        for allowed in [False, True]:
            with self.assertRaises(BackendFailure):
                serve_guest_once(self.f.guest, root, self.f.instance, self.f.context,
                    service_authorized=allowed, verify_peer=self.verify)
            self.assertEqual(target.read_bytes(), b'unrelated entry')


if __name__ == '__main__':
    unittest.main()
