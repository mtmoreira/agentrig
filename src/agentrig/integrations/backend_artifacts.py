"""Read-only bounded audit of an explicitly declared private artifact directory."""

import hashlib
import os
from pathlib import Path
import stat

from agentrig.capabilities.backend_artifacts import BackendArtifactAudit, BackendArtifactManifest
from agentrig.capabilities.local_backend import BackendFailure
from agentrig.core import RunContext
from agentrig.integrations.backend_journal import _private


def _check(context: RunContext) -> None:
    context.cancellation.raise_if_cancelled()
    if context.deadline is not None:
        context.deadline.raise_if_expired(context.clock)


def _identity(info: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_uid, info.st_mode, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def audit_backend_artifacts(root: Path, manifest: BackendArtifactManifest,
                            context: RunContext) -> BackendArtifactAudit:
    """Verify exactly the declared files, never follow links or execute content.

    The application must supply an ancestor-verified root. Directory descriptors
    bind this read operation, not later execution. Results are point-in-time byte
    observations, not proof of an offline dependency closure or an authority token.
    A finite context deadline is mandatory for streaming potentially large images.
    """
    _check(context)
    if (context.deadline is None or not root.is_absolute() or '..' in root.parts or
            root == Path('/') or not isinstance(manifest, BackendArtifactManifest)):
        raise BackendFailure('backend_artifacts_invalid')
    opened: dict[str, int] = {}
    identities: dict[str, tuple[int, int, int, int, int, int, int]] = {}
    files: dict[str, tuple[int, int, int, int, int, int, int]] = {}
    try:
        opened[''] = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        directories: dict[str, set[str]] = {'': set()}
        for artifact in manifest.artifacts:
            components = artifact.path.split('/')
            for index, component in enumerate(components):
                parent = '/'.join(components[:index])
                directories.setdefault(parent, set()).add(component)
        # Inventory before reading payloads: unexpected files (including auth
        # material) are rejected without opening them or reporting their names.
        for name in sorted(directories, key=lambda value: (value.count('/'), value)):
            _check(context)
            if name:
                parent, _, leaf = name.rpartition('/')
                opened[name] = os.open(leaf, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                       dir_fd=opened[parent])
            descriptor = opened[name]
            info = os.fstat(descriptor)
            _private(info, directory=True)
            identities[name] = _identity(info)
            actual: set[str] = set()
            with os.scandir(descriptor) as entries:
                for entry in entries:
                    _check(context)
                    if entry.name not in directories[name] or entry.name in actual:
                        raise BackendFailure('backend_artifacts_invalid')
                    actual.add(entry.name)
            if actual != directories[name]:
                raise BackendFailure('backend_artifacts_invalid')
        for artifact in manifest.artifacts:
            _check(context)
            parent, _, name = artifact.path.rpartition('/')
            descriptor = os.open(name, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=opened[parent])
            with os.fdopen(descriptor, 'rb') as stream:
                before = os.fstat(stream.fileno())
                _private(before, directory=False)
                if (before.st_size != artifact.size_bytes or
                        stat.S_IMODE(before.st_mode) != (0o700 if artifact.executable else 0o600)):
                    raise BackendFailure('backend_artifacts_invalid')
                digest = hashlib.sha256()
                remaining = artifact.size_bytes
                while remaining:
                    _check(context)
                    chunk = stream.read(min(65536, remaining))
                    if not chunk:
                        raise BackendFailure('backend_artifacts_invalid')
                    digest.update(chunk)
                    remaining -= len(chunk)
                if (stream.read(1) or _identity(os.fstat(stream.fileno())) != _identity(before) or
                        'sha256:' + digest.hexdigest() != artifact.sha256):
                    raise BackendFailure('backend_artifacts_invalid')
                current = os.stat(name, dir_fd=opened[parent], follow_symlinks=False)
                if _identity(current) != _identity(before):
                    raise BackendFailure('backend_artifacts_invalid')
                files[artifact.path] = _identity(before)
        for path, identity in files.items():
            parent, _, name = path.rpartition('/')
            if _identity(os.stat(name, dir_fd=opened[parent], follow_symlinks=False)) != identity:
                raise BackendFailure('backend_artifacts_invalid')
        for name, descriptor in opened.items():
            if _identity(os.fstat(descriptor)) != identities[name]:
                raise BackendFailure('backend_artifacts_invalid')
        if _identity(root.lstat()) != identities['']:
            raise BackendFailure('backend_artifacts_invalid')
        _check(context)
        return BackendArtifactAudit(manifest_digest=manifest.digest, file_count=len(manifest.artifacts),
                                    total_bytes=sum(a.size_bytes for a in manifest.artifacts))
    except BackendFailure:
        raise
    except (OSError, ValueError, TypeError):
        raise BackendFailure('backend_artifacts_invalid') from None
    finally:
        for descriptor in reversed(tuple(opened.values())):
            os.close(descriptor)
