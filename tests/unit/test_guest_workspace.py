"""Pure transfer contract checks; real streams belong to integration tests."""

from dataclasses import replace
from pathlib import Path
import unittest

from agentrig.integrations import GuestWorkspaceEndpoint, GuestWorkspaceReceipt, WorkspaceStagePlan
from agentrig.integrations.guest_workspace import _manifest, _request


class GuestWorkspaceContractTests(unittest.TestCase):
    def setUp(self):
        self.plan = WorkspaceStagePlan(host_root=Path('/private/host'), guest_root='/guest/workspaces',
            operation_id='a' * 32, files={'input/design.sv': b'synthetic rtl'})

    def test_request_is_sorted_bound_and_no_host_root_is_transmitted(self):
        header = _request(self.plan, 'b' * 32)
        self.assertEqual(header['plan_digest'], self.plan.digest)
        self.assertNotIn('/private/host', str(header))
        self.assertEqual(len(_manifest(header).artifacts), 1)
        changed = replace(self.plan, files={'input/design.sv': b'changed'})
        self.assertNotEqual(_request(changed, 'b' * 32), header)

    def test_no_root_files_auth_paths_or_malformed_instance(self):
        for name in ['single', 'input/id_rsa', 'input/secret.key', 'input/credentials.json']:
            with self.assertRaises(ValueError):
                _request(replace(self.plan, files={name: b'synthetic'}), 'b' * 32)
        with self.assertRaises(ValueError):
            _request(self.plan, 'not-an-instance')

    def test_strict_header_and_immutable_receipt(self):
        header = _request(self.plan, 'b' * 32)
        for field, value in [('extra', True), ('schema', 'unknown'), ('files', []), ('plan_digest', 'bad')]:
            changed = dict(header, **{field: value})
            with self.assertRaises(ValueError):
                _manifest(changed)
        receipt = GuestWorkspaceReceipt(request_digest='sha256:' + '1' * 64,
            plan_digest=self.plan.digest, operation_id=self.plan.operation_id, file_count=1, total_bytes=13)
        self.assertFalse(receipt.workload_access_verified)
        for field, value in [('file_count', True), ('total_bytes', -1), ('operation_id', 'bad')]:
            with self.assertRaises(ValueError):
                replace(receipt, **{field: value})

    def test_endpoint_authority_binds_plan_instance_path_and_socket_identity(self):
        endpoint = GuestWorkspaceEndpoint(root=Path('/private/endpoint'), instance_id='b' * 32,
            device=1, inode=2, uid=1000)
        digest = endpoint.transfer_digest(self.plan)
        for changed in [replace(endpoint, inode=3), replace(endpoint, instance_id='c' * 32),
                        replace(endpoint, root=Path('/another/endpoint'))]:
            self.assertNotEqual(digest, changed.transfer_digest(self.plan))
        self.assertNotEqual(digest, endpoint.transfer_digest(replace(self.plan, operation_id='c' * 32)))
        self.assertNotIn('/private/endpoint', repr(endpoint))
        with self.assertRaises(ValueError):
            replace(endpoint, inode=True)


if __name__ == '__main__':
    unittest.main()
