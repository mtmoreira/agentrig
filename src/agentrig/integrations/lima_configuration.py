"""Restricted Lima 2.2.0 input configuration; no activation or YAML execution.

Only the application's complete JSON profile is accepted. This is an input-file
and image-byte audit, not Lima's effective-config parser or a guest qualification.
"""

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
from typing import cast

from agentrig.capabilities import BackendArtifact, BackendArtifactManifest, BackendFailure
from agentrig.capabilities.local_backend import backend_digest
from agentrig.core import DeadlineExceeded, JsonValue, RunContext
from agentrig.integrations.backend_artifacts import audit_backend_artifacts, _identity
from agentrig.integrations.backend_journal import _private
from agentrig.integrations.local_backends import _integer, _path, _pin

_SKIP_INSTALL = '#!/bin/sh\nexit 0\n'
_LIMA_INSTANCE = 'managed-' + '0' * 32
_LIMA_SOCKET_SUFFIX = '/ssh.sock.' + '0' * 16


def _lima_state_root(value: str, instance: str = _LIMA_INSTANCE) -> str:
    """Validate every fixed Unix-socket path before Lima sees the profile."""
    root = _path(value)
    paths = (
        root + '/transport/workspace.sock',
        root + '/transport/docker.sock',
        root + '/lima/' + instance + _LIMA_SOCKET_SUFFIX,
    )
    if (root == '/' or any(c in root for c in ('{', '}', '$', '~')) or
            any(len(path.encode()) > 103 for path in paths)):
        raise ValueError('lima configuration root invalid')
    return root


@dataclass(frozen=True, slots=True, kw_only=True)
class LimaConfiguration:
    state_root: Path = field(repr=False)
    guest_image_sha256: str
    guest_image_size: int
    cpus: int = 4
    memory_mib: int = 4096
    disk_gib: int = 20
    guest_uid: int = 1000

    def __post_init__(self) -> None:
        # HostSocket is a Lima Go-template field. Accept literal paths only and
        # reserve Lima's longest observed per-instance control-socket suffix.
        _lima_state_root(str(self.state_root))
        _pin(self.guest_image_sha256)
        _integer(self.guest_image_size, 1, 4 * 1024**3)
        _integer(self.cpus, 1, 8)
        _integer(self.memory_mib, 1024, 8192)
        _integer(self.disk_gib, 8, 128)
        _integer(self.guest_uid, 1000, 60000)

    @property
    def image_manifest(self) -> BackendArtifactManifest:
        return BackendArtifactManifest(artifacts=(BackendArtifact(artifact_id='guest-image', path='guest.raw',
            sha256=self.guest_image_sha256, size_bytes=self.guest_image_size),))

    def document(self) -> dict[str, JsonValue]:
        root = str(self.state_root)
        user_run = '/run/user/' + str(self.guest_uid)
        return {'minimumLimaVersion': '2.2.0', 'vmType': 'vz', 'arch': 'aarch64', 'os': 'Linux',
            'plain': False, 'cpus': self.cpus, 'memory': str(self.memory_mib) + 'MiB',
            'disk': str(self.disk_gib) + 'GiB', 'base': [],
            'images': [{'location': root + '/artifacts/guest.raw', 'arch': 'aarch64',
                        'digest': self.guest_image_sha256}],
            'user': {'name': 'agentrig', 'comment': 'AgentRig guest', 'uid': self.guest_uid,
                     'home': '/home/agentrig', 'shell': '/bin/bash', 'passwordlessSudo': True},
            'mounts': [], 'mountType': 'virtiofs', 'copyToHost': [],
            'containerd': {'system': False, 'user': False},
            'ssh': {'loadDotSSHPubKeys': False, 'forwardAgent': False, 'forwardX11': False,
                    'forwardX11Trusted': False, 'overVsock': True},
            'caCerts': {'files': [], 'certs': [], 'removeDefaults': False},
            'upgradePackages': False, 'propagateProxyEnv': False,
            'hostResolver': {'enabled': False, 'ipv6': False, 'hosts': {}},
            'env': {}, 'param': {}, 'networks': [], 'timezone': 'UTC',
            'provision': [{'mode': 'dependency', 'skipDefaultDependencyResolution': True,
                           'script': _SKIP_INSTALL}], 'probes': [],
            'vmOpts': {'vz': {'diskImageFormat': 'raw', 'rosetta': {'enabled': False, 'binfmt': False}}},
            'audio': {'device': ''}, 'video': {'display': 'none'},
            'portForwards': [
                {'guestSocket': user_run + '/agentrig/workspace.sock',
                 'hostSocket': root + '/transport/workspace.sock', 'static': True, 'reverse': False},
                {'guestSocket': user_run + '/docker.sock',
                 'hostSocket': root + '/transport/docker.sock', 'static': True, 'reverse': False},
                {'guestIP': '0.0.0.0', 'guestIPMustBeZero': False, 'guestPortRange': [1, 65535],
                 'proto': 'any', 'ignore': True}]}

    @property
    def digest(self) -> str:
        return backend_digest({'schema': 'agentrig.lima-configuration-policy.v1',
            'configuration': self.document(), 'image_manifest': self.image_manifest.digest})

    def json_bytes(self) -> bytes:
        return (json.dumps(self.document(), sort_keys=True, indent=2) + '\n').encode()

    def validate(self, data: bytes) -> None:
        """Reject extras, duplicate keys, numeric booleans, templates and YAML."""
        if not isinstance(data, bytes) or not 1 <= len(data) <= 65536:
            raise ValueError('lima configuration invalid')
        def unique(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
            value: dict[str, JsonValue] = {}
            for key, item in pairs:
                if key in value:
                    raise ValueError('lima configuration invalid')
                value[key] = item
            return value
        try:
            value = json.loads(data, object_pairs_hook=unique)
            if not isinstance(value, dict) or backend_digest(cast(dict[str, JsonValue], value)) != backend_digest(self.document()):
                raise ValueError('lima configuration invalid')
        except (ValueError, TypeError, RecursionError):
            raise ValueError('lima configuration invalid') from None


@dataclass(frozen=True, slots=True, kw_only=True)
class LimaConfigurationAudit:
    policy_digest: str
    configuration_sha256: str
    image_manifest_digest: str
    dependency_closure_qualified: bool = field(default=False, init=False)
    effective_configuration_verified: bool = field(default=False, init=False)
    execution_authorized: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        for value in (self.policy_digest, self.configuration_sha256, self.image_manifest_digest):
            _pin(value)


def audit_lima_configuration(policy: LimaConfiguration, context: RunContext) -> LimaConfigurationAudit:
    """Audit one existing private configuration.json and artifacts/guest.raw.

    The application verifies state-root ancestors first. Global Lima configuration
    must be absent or an empty private directory; its contents are never read.
    No other instance directory, key, executable, socket or daemon is inspected.
    This deliberately cannot serve as a complete LimaArtifactVerifier.
    """
    opened: list[int] = []
    identities: dict[str, tuple[int, ...] | None] = {}
    def check() -> None:
        context.cancellation.raise_if_cancelled()
        if context.deadline is None:
            raise BackendFailure('backend_artifacts_invalid')
        context.deadline.raise_if_expired(context.clock)
    try:
        check()
        root = os.open(policy.state_root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        opened.append(root)
        _private(os.fstat(root), directory=True)
        identities['root'] = _identity(os.fstat(root))
        # Check only dedicated global configuration paths, not instance contents.
        try:
            lima = os.open('lima', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root)
        except FileNotFoundError:
            identities['lima'] = None
        else:
            opened.append(lima)
            _private(os.fstat(lima), directory=True)
            identities['lima'] = _identity(os.fstat(lima))
            try:
                config = os.open('_config', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=lima)
            except FileNotFoundError:
                identities['global'] = None
            else:
                opened.append(config)
                _private(os.fstat(config), directory=True)
                identities['global'] = _identity(os.fstat(config))
                with os.scandir(config) as entries:
                    if next(entries, None) is not None:
                        raise BackendFailure('backend_artifacts_invalid')
        descriptor = os.open('configuration.json', os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=root)
        with os.fdopen(descriptor, 'rb') as stream:
            info = os.fstat(stream.fileno())
            _private(info, directory=False)
            if info.st_mode & 0o7777 != 0o600 or not 1 <= info.st_size <= 65536:
                raise BackendFailure('backend_artifacts_invalid')
            data = stream.read(65537)
            policy.validate(data)
            if _identity(info) != _identity(os.fstat(stream.fileno())):
                raise BackendFailure('backend_artifacts_invalid')
        image = audit_backend_artifacts(policy.state_root / 'artifacts', policy.image_manifest, context)
        check()
        if _identity(os.stat('configuration.json', dir_fd=root, follow_symlinks=False)) != _identity(info):
            raise BackendFailure('backend_artifacts_invalid')
        paths = {'root': policy.state_root, 'lima': policy.state_root / 'lima',
                 'global': policy.state_root / 'lima/_config'}
        for name, expected in identities.items():
            try:
                current = _identity(paths[name].lstat())
            except FileNotFoundError:
                current = None
            if current != expected:
                raise BackendFailure('backend_artifacts_invalid')
        return LimaConfigurationAudit(policy_digest=policy.digest,
            configuration_sha256='sha256:' + hashlib.sha256(data).hexdigest(), image_manifest_digest=image.manifest_digest)
    except (BackendFailure, DeadlineExceeded):
        raise
    except (OSError, ValueError, TypeError):
        raise BackendFailure('backend_artifacts_invalid') from None
    finally:
        for descriptor in reversed(opened):
            os.close(descriptor)
