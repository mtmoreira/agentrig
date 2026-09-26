"""Owned local files/listeners; peer and backend quiescence checks are synthetic."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import socket
import tempfile
import threading
import unittest
from unittest.mock import patch

from agentrig.capabilities import BackendFailure
from agentrig.core import RunCancelled
from agentrig.integrations import (
    GuestWorkspaceEndpoint, copy_guest_service, guest_service_retirement_digest,
    inspect_guest_service, retire_guest_service, serve_guest_requests,
)
from agentrig.integrations.guest_service import _service_lock, _check_service_lock
from tests.integration import test_guest_workspace as fixture


class ServiceLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.f = fixture.GuestWorkspaceTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        temp = tempfile.TemporaryDirectory(prefix='agentrig-life-', dir='/tmp')
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.calls = []

    def endpoint(self):
        value = (self.root / 'workspace.sock').lstat()
        return GuestWorkspaceEndpoint(root=self.root, instance_id=self.f.instance,
            device=value.st_dev, inode=value.st_ino, uid=value.st_uid)

    def verify(self, *args):
        self.calls.append('peer')

    @contextmanager
    def listener(self, limit=2, verifier=None):
        ready = threading.Event()
        original = socket.socket.listen
        def listen(sock, backlog):
            original(sock, backlog)
            ready.set()
        with patch.object(socket.socket, 'listen', listen), ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(serve_guest_requests, self.f.guest, self.root, self.f.instance,
                self.f.context, request_limit=limit, service_authorized=True,
                verify_peer=self.verify if verifier is None else verifier)
            try:
                for _ in range(200):
                    if ready.wait(.005):
                        break
                    if future.done():
                        future.result()
                self.assertTrue(ready.is_set())
                yield future
            finally:
                if not future.done():
                    self.f.source.cancel()

    def test_filesystem_two_requests_same_endpoint_retire_and_restart(self):
        with self.listener() as future:
            endpoint = self.endpoint()
            receipt = copy_guest_service(endpoint, self.f.plan, endpoint.transfer_digest(self.f.plan), self.f.context)
            self.assertEqual(inspect_guest_service(endpoint, self.f.plan, self.f.context), receipt)
            self.assertEqual(future.result(timeout=4), (receipt, receipt))
        before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in self.f.guest.rglob('*') if p.is_file()}
        lock_inode = (self.root / '.service.lock').stat().st_ino
        verified = []
        retire_guest_service(endpoint, guest_service_retirement_digest(endpoint), self.f.context,
            verify_quiescent=lambda *args: verified.append('stopped-and-drained'))
        self.assertEqual(verified, ['stopped-and-drained'])
        self.assertFalse((self.root / 'workspace.sock').exists())
        self.assertEqual(lock_inode, (self.root / '.service.lock').stat().st_ino)
        self.assertEqual(before, {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in self.f.guest.rglob('*') if p.is_file()})
        with self.listener(limit=1) as future:
            next_endpoint = self.endpoint()
            self.assertEqual(inspect_guest_service(next_endpoint, self.f.plan, self.f.context), receipt)
            self.assertEqual(future.result(timeout=4), (receipt,))
        self.assertEqual(self.calls, ['peer'] * 3)

    def test_filesystem_active_listener_blocks_retirement_before_verifier(self):
        with self.listener() as future:
            endpoint = self.endpoint()
            with self.assertRaisesRegex(BackendFailure, 'backend_journal_busy'):
                retire_guest_service(endpoint, guest_service_retirement_digest(endpoint), self.f.context,
                    verify_quiescent=lambda *args: self.fail('active lease must block verifier'))
            self.assertEqual(self.endpoint(), endpoint)
            self.f.source.cancel()
            with self.assertRaises(RunCancelled):
                future.result(timeout=4)

    def test_filesystem_peer_failure_stops_without_processing_or_cleanup(self):
        with self.listener(verifier=lambda *args: True) as future:
            endpoint = self.endpoint()
            with self.assertRaises(BackendFailure):
                inspect_guest_service(endpoint, self.f.plan, self.f.context)
            with self.assertRaises(BackendFailure):
                future.result(timeout=4)
        self.assertEqual(list(self.f.guest.iterdir()), [])
        self.assertEqual(self.endpoint(), endpoint)

    def test_filesystem_changed_entry_during_verification_is_preserved(self):
        with self.listener(limit=1) as future:
            endpoint = self.endpoint()
            copy_guest_service(endpoint, self.f.plan, endpoint.transfer_digest(self.f.plan), self.f.context)
            future.result(timeout=4)
        def replace_entry(*args):
            (self.root / 'workspace.sock').rename(self.root / 'retained.sock')
            (self.root / 'workspace.sock').write_bytes(b'unrelated replacement')
        with self.assertRaises(BackendFailure):
            retire_guest_service(endpoint, guest_service_retirement_digest(endpoint), self.f.context,
                verify_quiescent=replace_entry)
        self.assertEqual((self.root / 'workspace.sock').read_bytes(), b'unrelated replacement')
        self.assertTrue((self.root / 'retained.sock').is_socket())

    def test_real_exclusive_lease_rejects_concurrency_and_releases_on_close(self):
        directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        self.addCleanup(os.close, directory)
        first = _service_lock(directory)
        try:
            with self.assertRaisesRegex(BackendFailure, 'backend_journal_busy'):
                _service_lock(directory)
        finally:
            os.close(first)
        second = _service_lock(directory)
        os.close(second)
        self.assertEqual((self.root / '.service.lock').read_bytes(), b'')

    def test_replaced_lock_inode_is_detected_without_removal(self):
        directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        self.addCleanup(os.close, directory)
        lease = _service_lock(directory)
        self.addCleanup(os.close, lease)
        (self.root / '.service.lock').rename(self.root / 'retained.lock')
        (self.root / '.service.lock').write_bytes(b'foreign')
        with self.assertRaises(BackendFailure):
            _check_service_lock(directory, lease)
        self.assertEqual((self.root / '.service.lock').read_bytes(), b'foreign')

    def test_invalid_lock_content_is_not_truncated(self):
        directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        self.addCleanup(os.close, directory)
        (self.root / '.service.lock').write_bytes(b'foreign record')
        with self.assertRaises(BackendFailure):
            _service_lock(directory)
        self.assertEqual((self.root / '.service.lock').read_bytes(), b'foreign record')

    def test_retirement_lock_acquisition_never_creates_missing_state(self):
        directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        self.addCleanup(os.close, directory)
        with self.assertRaises(FileNotFoundError):
            _service_lock(directory, create=False)
        self.assertEqual(list(self.root.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
