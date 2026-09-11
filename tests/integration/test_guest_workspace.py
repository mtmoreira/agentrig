"""Real local socket transfers only; no VM, service, credential or container."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import socket
import stat
import struct
import tempfile
import unittest
from unittest.mock import patch

from agentrig.capabilities import BackendFailure
from agentrig.core import CancellationSource, Deadline, DeadlineExceeded, RunCancelled, RunContext, RunId
from agentrig.core.clock import SystemClock
from agentrig.core.identity import Uuid4IdGenerator
from agentrig.integrations import (
    GuestWorkspaceEndpoint, WorkspaceStagePlan, copy_to_guest,
    receive_guest_workspace, transfer_guest_workspace,
)
from agentrig.integrations.guest_workspace import _encode, _request, _receipt, _manifest, _Stream, _verify_copy


class GuestWorkspaceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='agentrig-guest-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.host = self.root / 'host'
        self.guest = self.root / 'guest'
        self.host.mkdir(mode=0o700)
        self.guest.mkdir(mode=0o700)
        self.plan = WorkspaceStagePlan(host_root=self.host, guest_root=str(self.guest),
            operation_id='a' * 32, files={'input/design.sv': b'synthetic rtl', 'control/run.py': b'not executed'})
        self.instance = 'b' * 32
        self.source = CancellationSource()
        self.context = self.make_context()

    def make_context(self, seconds=3):
        clock = SystemClock()
        return RunContext.create_root(clock=clock, id_generator=Uuid4IdGenerator(RunId),
            cancellation=self.source.token, deadline=Deadline.after(seconds, clock))

    def exchange(self, plan=None, root=None, approved=None):
        plan = self.plan if plan is None else plan
        root = self.guest if root is None else root
        client, server = socket.socketpair()
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(receive_guest_workspace, server, root, self.instance, self.context)
            try:
                result = transfer_guest_workspace(client, plan, self.instance,
                    plan.digest if approved is None else approved, self.context)
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

    def send_raw(self, data, root=None):
        client, server = socket.socketpair()
        self.addCleanup(client.close)
        # These bounded malformed fixtures fit in the socket buffer. Finish the
        # write side before starting rejection, so the receiver may close early
        # without racing this test's shutdown call (ENOTCONN on macOS).
        client.sendall(data)
        client.shutdown(socket.SHUT_WR)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(receive_guest_workspace, server, self.guest if root is None else root,
                                 self.instance, self.context)
            with self.assertRaisesRegex(BackendFailure, '^backend_staging_failed$'):
                future.result(timeout=4)
        self.assertEqual(server.fileno(), -1)

    def raw(self, header=None, payload=None):
        header = _encode(_request(self.plan, self.instance)) if header is None else header
        payload = b''.join(self.plan.files[name] for name in sorted(self.plan.files)) if payload is None else payload
        return struct.pack('!I', len(header)) + header + payload

    def test_exact_copy_modes_receipts_and_no_host_writes(self):
        result = self.exchange()
        self.assertFalse(result.workload_access_verified)
        self.assertEqual(list(self.host.iterdir()), [])
        operation = self.guest / self.plan.operation_id
        self.assertEqual(stat.S_IMODE(operation.stat().st_mode), 0o700)
        for name, payload in self.plan.files.items():
            target = operation / name
            self.assertEqual(target.read_bytes(), payload)
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o644)
            self.assertEqual(stat.S_IMODE(target.parent.stat().st_mode), 0o755)
        intent = (operation / '.intent.json').read_bytes()
        self.assertEqual(intent, (operation / '.complete.json').read_bytes())
        self.assertEqual(stat.S_IMODE((operation / '.complete.json').stat().st_mode), 0o600)
        self.assertNotIn(b'synthetic rtl', intent)
        self.assertNotIn(str(self.host).encode(), intent)

    def test_replay_preserves_completed_tree(self):
        self.exchange()
        operation = self.guest / self.plan.operation_id
        before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in operation.rglob('*') if p.is_file()}
        with self.assertRaises(BackendFailure):
            self.exchange()
        self.assertEqual(before, {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in operation.rglob('*') if p.is_file()})

    def test_changed_authority_no_files_or_request_bytes(self):
        client, server = socket.socketpair()
        self.addCleanup(server.close)
        with self.assertRaisesRegex(BackendFailure, 'backend_authority_required'):
            transfer_guest_workspace(client, self.plan, self.instance, 'sha256:' + '0' * 64, self.context)
        self.assertEqual(server.recv(1), b'')
        self.assertEqual(list(self.guest.iterdir()), [])

    def test_binding_links_and_untrusted_root_rejected(self):
        for plan in [replace(self.plan, guest_root='/wrong/root')]:
            with self.assertRaises(BackendFailure):
                self.exchange(plan)
        link = self.root / 'linked'
        link.symlink_to(self.guest, target_is_directory=True)
        with self.assertRaises(BackendFailure):
            self.exchange(replace(self.plan, guest_root=str(link)), root=link)
        self.guest.chmod(0o755)
        with self.assertRaises(BackendFailure):
            self.exchange()
        self.guest.chmod(0o700)
        self.assertEqual(list(self.guest.iterdir()), [])

    def test_malformed_oversized_duplicate_header_and_traversal(self):
        bad = _request(self.plan, self.instance)
        bad['files'][0]['path'] = '../escape'
        for data in [struct.pack('!I', 65537), self.raw(b'{"schema":1,"schema":2}', b''),
                     self.raw(_encode(bad)), self.raw(b'[]', b''), self.raw(b'{' * 100, b'')]:
            self.send_raw(data)
        self.assertEqual(list(self.guest.iterdir()), [])

    def test_wrong_content_retains_intent_and_rejects_retry(self):
        self.send_raw(self.raw(payload=b'x' * sum(map(len, self.plan.files.values()))))
        operation = self.guest / self.plan.operation_id
        self.assertTrue((operation / '.intent.json').exists())
        self.assertFalse((operation / '.complete.json').exists())
        with self.assertRaises(BackendFailure):
            self.exchange()

    def test_trailing_data_never_completes(self):
        self.send_raw(self.raw() + b'extra')
        self.assertFalse((self.guest / self.plan.operation_id / '.complete.json').exists())

    def test_truncated_payload_never_completes(self):
        self.send_raw(self.raw(payload=b'partial'))
        self.assertFalse((self.guest / self.plan.operation_id / '.complete.json').exists())

    def test_cancellation_and_deadline_keep_identity_and_close_stream(self):
        for cancel in [False, True]:
            client, server = socket.socketpair()
            self.addCleanup(server.close)
            if cancel:
                self.source.cancel('cancelled')
            context = self.make_context(0 if not cancel else 3)
            with self.assertRaises(RunCancelled if cancel else DeadlineExceeded):
                transfer_guest_workspace(client, self.plan, self.instance, self.plan.digest, context)
            self.assertEqual(client.fileno(), -1)
            self.assertEqual(server.recv(1), b'')

    def test_waiting_peer_deadline_is_bounded(self):
        client, server = socket.socketpair()
        self.addCleanup(server.close)
        with self.assertRaises(DeadlineExceeded):
            receive_guest_workspace(client, self.guest, self.instance, self.make_context(.05))
        self.assertEqual(client.fileno(), -1)

    def test_partial_copy_failure_retains_private_intent(self):
        with patch('agentrig.integrations.guest_workspace.os.fchmod', side_effect=OSError('private diagnostic')):
            with self.assertRaisesRegex(BackendFailure, '^backend_staging_failed$'):
                self.exchange()
        operation = self.guest / self.plan.operation_id
        self.assertTrue((operation / '.intent.json').is_file())
        self.assertFalse((operation / '.complete.json').exists())

    def test_maximum_file_and_empty_file_use_framed_binary_payload(self):
        plan = replace(self.plan, files={'input/full.bin': b'\0' * 1024**2, 'input/empty': b''})
        self.exchange(plan)
        self.assertEqual((self.guest / plan.operation_id / 'input/full.bin').stat().st_size, 1024**2)

    def test_replaced_or_hardlinked_file_cannot_receive_completion(self):
        def tamper(operation, manifest, stream):
            target = self.guest / self.plan.operation_id / manifest.artifacts[0].path
            target.write_bytes(b'changed')
            return _verify_copy(operation, manifest, stream)
        with patch('agentrig.integrations.guest_workspace._verify_copy', side_effect=tamper):
            with self.assertRaises(BackendFailure):
                self.exchange()
        self.assertFalse((self.guest / self.plan.operation_id / '.complete.json').exists())
        self.plan = replace(self.plan, operation_id='c' * 32)
        def link(operation, manifest, stream):
            target = self.guest / self.plan.operation_id / manifest.artifacts[0].path
            os.link(target, self.guest / 'unexpected-link')
            return _verify_copy(operation, manifest, stream)
        with patch('agentrig.integrations.guest_workspace._verify_copy', side_effect=link):
            with self.assertRaises(BackendFailure):
                self.exchange()
        self.assertFalse((self.guest / self.plan.operation_id / '.complete.json').exists())

    def test_root_replacement_cannot_receive_completion(self):
        def replace_root(operation, manifest, stream):
            self.guest.rename(self.root / 'retained')
            self.guest.mkdir(mode=0o700)
            return _verify_copy(operation, manifest, stream)
        with patch('agentrig.integrations.guest_workspace._verify_copy', side_effect=replace_root):
            with self.assertRaises(BackendFailure):
                self.exchange()
        self.assertFalse((self.root / 'retained' / self.plan.operation_id / '.complete.json').exists())

    def test_wrong_receipt_and_extra_response_bytes_are_rejected(self):
        for mode in ['count', 'numeric-bool', 'extra']:
            client, server = socket.socketpair()
            def fake():
                with server:
                    stream = _Stream(server, self.context)
                    header = stream.header()
                    manifest = _manifest(header)
                    stream.read(sum(a.size_bytes for a in manifest.artifacts))
                    stream.eof()
                    receipt = _receipt(header, manifest)
                    if mode == 'count':
                        receipt['file_count'] = 99
                    if mode == 'numeric-bool':
                        receipt['workload_access_verified'] = 0
                    stream.send_header(receipt)
                    if mode == 'extra':
                        stream.write(b'extra')
                    server.shutdown(socket.SHUT_WR)
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(fake)
                with self.assertRaises(BackendFailure):
                    transfer_guest_workspace(client, self.plan, self.instance, self.plan.digest, self.context)
                future.result(timeout=4)

    def test_no_deadline_is_rejected_without_consuming_request(self):
        client, server = socket.socketpair()
        self.addCleanup(server.close)
        with self.assertRaises(BackendFailure):
            transfer_guest_workspace(client, self.plan, self.instance, self.plan.digest,
                replace(self.context, deadline=None))
        self.assertEqual(server.recv(1), b'')

    def endpoint(self):
        # Darwin Unix socket names have a short platform path limit.
        temporary = tempfile.TemporaryDirectory(prefix='agentrig-sock-', dir='/tmp')
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.addCleanup(listener.close)
        listener.settimeout(.2)
        listener.bind(str(root / 'workspace.sock'))
        (root / 'workspace.sock').chmod(0o600)
        listener.listen(1)
        info = (root / 'workspace.sock').lstat()
        endpoint = GuestWorkspaceEndpoint(root=root, instance_id=self.instance, device=info.st_dev,
                                          inode=info.st_ino, uid=info.st_uid)
        return endpoint, listener

    def test_real_private_endpoint_connects_once_and_copies(self):
        endpoint, listener = self.endpoint()
        def serve():
            connection, _ = listener.accept()
            return receive_guest_workspace(connection, self.guest, self.instance, self.context)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(serve)
            result = copy_to_guest(endpoint, self.plan, endpoint.transfer_digest(self.plan), self.context)
            self.assertEqual(result, future.result(timeout=4))

    def test_endpoint_change_and_insufficient_authority_never_connect(self):
        endpoint, listener = self.endpoint()
        with self.assertRaisesRegex(BackendFailure, 'backend_authority_required'):
            copy_to_guest(endpoint, self.plan, self.plan.digest, self.context)
        changed = replace(endpoint, inode=endpoint.inode + 1)
        with self.assertRaises(BackendFailure):
            copy_to_guest(changed, self.plan, changed.transfer_digest(self.plan), self.context)
        (endpoint.root / 'workspace.sock').chmod(0o666)
        with self.assertRaises(BackendFailure):
            copy_to_guest(endpoint, self.plan, endpoint.transfer_digest(self.plan), self.context)
        with self.assertRaises(socket.timeout):
            listener.accept()
        self.assertEqual(list(self.guest.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
