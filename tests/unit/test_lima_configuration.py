"""Input policy only; no claim about installed Lima or a bootable guest."""

from dataclasses import replace
import json
from pathlib import Path
import unittest

from agentrig.integrations import LimaConfiguration


class LimaConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.policy = LimaConfiguration(state_root=Path('/private/owned'),
            guest_image_sha256='sha256:' + '1' * 64, guest_image_size=16)

    def test_detached_profile_and_exact_roundtrip(self):
        self.policy.validate(self.policy.json_bytes())
        self.policy.validate(json.dumps(self.policy.document()).encode())
        document = self.policy.document()
        document['mounts'].append({'location': '~'})
        self.assertEqual(self.policy.document()['mounts'], [])
        self.assertNotIn('/private/owned', repr(self.policy))

    def test_resources_and_templated_roots_fail(self):
        for options in [{'cpus': True}, {'cpus': 0}, {'guest_uid': 65534}, {'guest_image_size': 0},
                        {'guest_image_sha256': 'none'}, {'state_root': Path('/tmp/{{.Home}}')},
                        {'state_root': Path('/' + 'x' * 120)},
                        {'state_root': Path('/private/tmp/openrtl-m42b-lima-live-v2/state')}]:
            with self.assertRaises(ValueError):
                replace(self.policy, **options)

    def test_unreviewed_fields_and_implicit_acquisition_fail(self):
        for key, value in [('base', ['template:docker']), ('upgradePackages', True), ('provision', []),
                           ('containerd', {'user': True}), ('mounts', [{'location': '/home'}]),
                           ('copyToHost', [{'guest': '/tmp/x', 'host': '/tmp/y'}]),
                           ('hostResolver', {'enabled': True}), ('unknown', False), ('plain', True),
                           ('cpus', 4.0), ('upgradePackages', 0)]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.policy.validate(json.dumps(dict(self.policy.document(), **{key: value})).encode())

    def test_nested_forward_and_image_mutations_fail(self):
        for mutate in [lambda d: d['images'][0].update(location='https://example.invalid/image'),
                       lambda d: d['images'].append(d['images'][0]),
                       lambda d: d['portForwards'][0].update(hostSocket='/tmp/foreign.sock'),
                       lambda d: d['portForwards'][1].update(reverse=True),
                       lambda d: d['portForwards'][2].update(ignore=False),
                       lambda d: d['ssh'].update(forwardAgent=True),
                       lambda d: d['provision'][0].update(script='unreviewed installer'),
                       lambda d: d['user'].update(uid=65534)]:
            value = self.policy.document()
            mutate(value)
            with self.assertRaises(ValueError):
                self.policy.validate(json.dumps(value).encode())

    def test_duplicate_yaml_malformed_and_oversized_inputs_fail(self):
        for data in [b'{"cpus":1,"cpus":4}', b'vmType: vz\n', b'[]', b'NaN', b'{' * 2000,
                     self.policy.json_bytes() + b' x', b' ' * 65537]:
            with self.assertRaises(ValueError):
                self.policy.validate(data)


if __name__ == '__main__':
    unittest.main()
