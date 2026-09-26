"""Declared artifact bytes, separate from dependency completeness or authority."""

from dataclasses import dataclass, field
from pathlib import PurePosixPath
import re

from agentrig.capabilities.local_backend import backend_digest, backend_identifier, _digest


@dataclass(frozen=True, slots=True, kw_only=True)
class BackendArtifact:
    artifact_id: str
    path: str = field(repr=False)
    sha256: str
    size_bytes: int
    executable: bool = False

    def __post_init__(self) -> None:
        backend_identifier(self.artifact_id)
        _digest(self.sha256)
        if (not isinstance(self.path, str) or len(self.path) > 256 or
                re.fullmatch(r'[A-Za-z0-9_-][A-Za-z0-9_./-]*', self.path) is None or
                len(self.path.split('/')) > 8 or any(part in ('', '.', '..') or part.startswith('.')
                    for part in self.path.split('/')) or
                any(part.lower() in {'id_rsa', 'id_ed25519', 'id_ecdsa', 'id_dsa', 'credentials',
                                     'credentials.json', 'tokens.json'} for part in self.path.split('/')) or
                PurePosixPath(self.path).suffix.lower() in {'.key', '.pem', '.p12', '.pfx', '.crt', '.cer', '.der'}):
            raise ValueError('backend artifact path invalid')
        if type(self.size_bytes) is not int or not 0 <= self.size_bytes <= 4 * 1024**3:
            raise ValueError('backend artifact size invalid')
        if type(self.executable) is not bool:
            raise ValueError('backend artifact executable flag invalid')


@dataclass(frozen=True, slots=True, kw_only=True)
class BackendArtifactManifest:
    artifacts: tuple[BackendArtifact, ...] = field(repr=False)

    def __post_init__(self) -> None:
        values = tuple(self.artifacts)
        if not 1 <= len(values) <= 128 or any(not isinstance(a, BackendArtifact) for a in values):
            raise ValueError('backend artifact manifest invalid')
        paths = {a.path for a in values}
        if len(paths) != len(values) or len({a.artifact_id for a in values}) != len(values):
            raise ValueError('backend artifact manifest duplicate')
        if any(str(parent) in paths for path in paths for parent in PurePosixPath(path).parents):
            raise ValueError('backend artifact path conflict')
        if sum(a.size_bytes for a in values) > 16 * 1024**3:
            raise ValueError('backend artifact manifest size invalid')
        object.__setattr__(self, 'artifacts', tuple(sorted(values, key=lambda a: a.path)))

    @property
    def digest(self) -> str:
        return backend_digest({'schema': 'agentrig.backend-artifacts.v1', 'artifacts': [
            {'id': a.artifact_id, 'path': a.path, 'sha256': a.sha256, 'size_bytes': a.size_bytes,
             'executable': a.executable} for a in self.artifacts]})


@dataclass(frozen=True, slots=True, kw_only=True)
class BackendArtifactAudit:
    manifest_digest: str
    file_count: int
    total_bytes: int
    # A content audit can never establish either of these properties.
    dependency_closure_qualified: bool = field(default=False, init=False)
    execution_authorized: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        _digest(self.manifest_digest)
        if (type(self.file_count) is not int or not 1 <= self.file_count <= 128 or
                type(self.total_bytes) is not int or not 0 <= self.total_bytes <= 16 * 1024**3):
            raise ValueError('backend artifact audit invalid')
