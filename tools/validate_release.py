"""Validate immutable AgentRig release artifacts and emit their manifest."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from email.message import Message
from email.parser import BytesParser
from email.policy import default
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tarfile
import tomllib
from typing import Protocol, cast
from zipfile import BadZipFile, ZipFile


RELEASE_MANIFEST_SCHEMA = "agentrig-release-manifest.v1"
_DISTRIBUTION_NAME = "agentrig"
_STABLE_VERSION = re.compile(
    r"^(?:0|[1-9][0-9]*)\."
    r"(?:0|[1-9][0-9]*)\."
    r"(?:0|[1-9][0-9]*)$"
)
_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_FORBIDDEN_PARTS = frozenset(
    {
        ".agents",
        ".env",
        ".git",
        ".git-credentials",
        ".netrc",
        ".npmrc",
        ".pypirc",
        "AGENTS.md",
        "SKILL.md",
    }
)
_FORBIDDEN_SUFFIXES = frozenset({".key", ".p12", ".pem", ".pfx"})


class ReleaseValidationError(ValueError):
    """Raised when a candidate release violates the repository contract."""


class GitRunner(Protocol):
    """Run one read-only Git query in a repository checkout."""

    def __call__(self, arguments: Sequence[str], cwd: Path) -> str: ...


@dataclass(frozen=True, slots=True, kw_only=True)
class ReleaseArtifact:
    """Immutable identity and digest for one distribution artifact."""

    filename: str
    size_bytes: int
    sha256: str

    def to_dict(self) -> dict[str, str | int]:
        return {
            "filename": self.filename,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class ReleaseManifest:
    """Version, source revision, and artifacts for one immutable release."""

    distribution: str
    version: str
    tag: str
    commit: str
    artifacts: tuple[ReleaseArtifact, ...]

    @property
    def filename(self) -> str:
        return f"{self.distribution}-{self.version}-release.json"

    def to_json(self) -> str:
        payload: dict[str, object] = {
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "commit": self.commit,
            "distribution": self.distribution,
            "schema_version": RELEASE_MANIFEST_SCHEMA,
            "tag": self.tag,
            "version": self.version,
        }
        return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def validate_release(
    repository_root: Path,
    dist_directory: Path,
    *,
    tag: str,
    commit: str,
) -> ReleaseManifest:
    """Validate a built wheel and sdist against repository release metadata."""
    root = repository_root.resolve()
    dist = dist_directory.resolve()
    project = _load_project(root / "pyproject.toml")
    name = _require_string(project.get("name"), "project name")
    version = _require_string(project.get("version"), "project version")
    if name != _DISTRIBUTION_NAME:
        raise ReleaseValidationError(
            f"project name must be {_DISTRIBUTION_NAME!r}"
        )
    if _STABLE_VERSION.fullmatch(version) is None:
        raise ReleaseValidationError(
            "project version must be a stable MAJOR.MINOR.PATCH value"
        )
    expected_tag = f"v{version}"
    if tag != expected_tag:
        raise ReleaseValidationError(
            f"release tag must be {expected_tag!r}"
        )
    if _COMMIT_SHA.fullmatch(commit) is None:
        raise ReleaseValidationError(
            "release commit must be a full lowercase Git SHA"
        )
    if not dist.is_dir():
        raise ReleaseValidationError("release distribution directory is missing")

    expected_wheel = f"{name}-{version}-py3-none-any.whl"
    expected_sdist = f"{name}-{version}.tar.gz"
    wheel_paths = tuple(sorted(dist.glob(f"{name}-*.whl")))
    sdist_paths = tuple(sorted(dist.glob(f"{name}-*.tar.gz")))
    _require_exact_artifact(wheel_paths, expected_wheel, "wheel")
    _require_exact_artifact(sdist_paths, expected_sdist, "source distribution")
    wheel_path = wheel_paths[0]
    sdist_path = sdist_paths[0]

    _validate_wheel(wheel_path, project)
    _validate_sdist(sdist_path, project)
    artifacts = tuple(
        _artifact_record(path)
        for path in sorted((wheel_path, sdist_path), key=lambda item: item.name)
    )
    return ReleaseManifest(
        distribution=name,
        version=version,
        tag=tag,
        commit=commit,
        artifacts=artifacts,
    )


def validate_repository_identity(
    repository_root: Path,
    *,
    tag: str,
    commit: str,
    run_git: GitRunner | None = None,
) -> None:
    """Prove that a clean checkout and annotated tag identify the commit."""
    runner = _run_git if run_git is None else run_git
    root = repository_root.resolve()
    head = runner(("rev-parse", "HEAD"), root)
    if head != commit:
        raise ReleaseValidationError(
            "release commit does not match the checkout HEAD"
        )
    status = runner(("status", "--porcelain=v1", "--untracked-files=all"), root)
    if status:
        raise ReleaseValidationError("release checkout must be clean")
    tag_reference = f"refs/tags/{tag}"
    if runner(("cat-file", "-t", tag_reference), root) != "tag":
        raise ReleaseValidationError("release tag must be annotated")
    tagged_commit = runner(("rev-parse", f"{tag}^{{commit}}"), root)
    if tagged_commit != commit:
        raise ReleaseValidationError(
            "release tag does not identify the release commit"
        )


def write_release_manifest(
    manifest: ReleaseManifest,
    dist_directory: Path,
) -> Path:
    """Write the deterministic manifest without replacing different content."""
    dist = dist_directory.resolve()
    output = dist / manifest.filename
    content = manifest.to_json()
    if output.exists():
        if output.read_text(encoding="utf-8") != content:
            raise ReleaseValidationError(
                "release manifest already exists with different content"
            )
        return output
    output.write_text(content, encoding="utf-8")
    return output


def _load_project(project_path: Path) -> Mapping[str, object]:
    if not project_path.is_file():
        raise ReleaseValidationError("pyproject.toml is missing")
    with project_path.open("rb") as project_file:
        document = tomllib.load(project_file)
    project = document.get("project")
    if not isinstance(project, Mapping):
        raise ReleaseValidationError("pyproject.toml has no project table")
    return cast(Mapping[str, object], project)


def _validate_wheel(
    wheel_path: Path,
    project: Mapping[str, object],
) -> None:
    with ZipFile(wheel_path) as archive:
        names = tuple(item.filename for item in archive.infolist())
        _validate_archive_names(names, archive_kind="wheel")
        metadata_names = tuple(
            name for name in names if name.endswith(".dist-info/METADATA")
        )
        if len(metadata_names) != 1:
            raise ReleaseValidationError(
                "wheel must contain exactly one METADATA record"
            )
        metadata = BytesParser(policy=default).parsebytes(
            archive.read(metadata_names[0])
        )

    required = {"agentrig/__init__.py", "agentrig/py.typed"}
    if not required.issubset(names):
        raise ReleaseValidationError("wheel is missing required package files")
    for repository_root in ("docs/", "evals/", "examples/", "tests/", "tools/"):
        if any(name.startswith(repository_root) for name in names):
            raise ReleaseValidationError(
                "wheel contains repository-only top-level content"
            )
    _validate_metadata(metadata, project, source="wheel")


def _validate_sdist(
    sdist_path: Path,
    project: Mapping[str, object],
) -> None:
    version = _require_string(project.get("version"), "project version")
    root_name = f"{_DISTRIBUTION_NAME}-{version}"
    prefix = f"{root_name}/"
    with tarfile.open(sdist_path, mode="r:gz") as archive:
        members = tuple(archive.getmembers())
        names = tuple(member.name for member in members)
        _validate_archive_names(names, archive_kind="source distribution")
        if any(member.issym() or member.islnk() for member in members):
            raise ReleaseValidationError(
                "source distribution must not contain links"
            )
        if any(name != root_name and not name.startswith(prefix) for name in names):
            raise ReleaseValidationError(
                "source distribution contains an invalid root path"
            )
        required = {
            f"{prefix}PKG-INFO",
            f"{prefix}README.md",
            f"{prefix}pyproject.toml",
            f"{prefix}src/agentrig/__init__.py",
            f"{prefix}src/agentrig/py.typed",
        }
        if not required.issubset(names):
            raise ReleaseValidationError(
                "source distribution is missing required files"
            )
        metadata_file = archive.extractfile(f"{prefix}PKG-INFO")
        project_file = archive.extractfile(f"{prefix}pyproject.toml")
        if metadata_file is None or project_file is None:
            raise ReleaseValidationError(
                "source distribution metadata cannot be read"
            )
        metadata = BytesParser(policy=default).parsebytes(metadata_file.read())
        embedded_document = tomllib.loads(project_file.read().decode("utf-8"))

    embedded_project = embedded_document.get("project")
    if not isinstance(embedded_project, Mapping):
        raise ReleaseValidationError(
            "source distribution has no project metadata"
        )
    embedded_name = _require_string(embedded_project.get("name"), "sdist name")
    embedded_version = _require_string(
        embedded_project.get("version"),
        "sdist version",
    )
    expected_version = _require_string(project.get("version"), "project version")
    if embedded_name != _DISTRIBUTION_NAME or embedded_version != expected_version:
        raise ReleaseValidationError(
            "source distribution project metadata does not match the release"
        )
    _validate_metadata(metadata, project, source="source distribution")


def _validate_metadata(
    metadata: Message,
    project: Mapping[str, object],
    *,
    source: str,
) -> None:
    expected_name = _require_string(project.get("name"), "project name")
    expected_version = _require_string(project.get("version"), "project version")
    expected_python = _require_string(
        project.get("requires-python"),
        "project requires-python",
    )
    if metadata.get("Name") != expected_name:
        raise ReleaseValidationError(f"{source} name metadata does not match")
    if metadata.get("Version") != expected_version:
        raise ReleaseValidationError(f"{source} version metadata does not match")
    if metadata.get("Requires-Python") != expected_python:
        raise ReleaseValidationError(
            f"{source} Python requirement metadata does not match"
        )

    actual_requirements = tuple(metadata.get_all("Requires-Dist", []))
    expected_requirements = _expected_requirements(project)
    if actual_requirements != expected_requirements:
        raise ReleaseValidationError(
            f"{source} dependency metadata does not match pyproject.toml"
        )
    actual_extras = tuple(metadata.get_all("Provides-Extra", []))
    expected_extras = _expected_extras(project)
    if actual_extras != expected_extras:
        raise ReleaseValidationError(
            f"{source} extra metadata does not match pyproject.toml"
        )


def _expected_requirements(project: Mapping[str, object]) -> tuple[str, ...]:
    requirements = list(_require_string_sequence(project.get("dependencies", [])))
    optional = project.get("optional-dependencies", {})
    if not isinstance(optional, Mapping):
        raise ReleaseValidationError(
            "project optional-dependencies must be a table"
        )
    for extra in sorted(optional):
        if not isinstance(extra, str):
            raise ReleaseValidationError("project extra name must be text")
        values = _require_string_sequence(optional[extra])
        requirements.extend(
            f"{requirement} ; extra == '{extra}'" for requirement in values
        )
    return tuple(requirements)


def _expected_extras(project: Mapping[str, object]) -> tuple[str, ...]:
    optional = project.get("optional-dependencies", {})
    if not isinstance(optional, Mapping):
        raise ReleaseValidationError(
            "project optional-dependencies must be a table"
        )
    extras = tuple(sorted(optional))
    if any(not isinstance(extra, str) for extra in extras):
        raise ReleaseValidationError("project extra name must be text")
    return cast(tuple[str, ...], extras)


def _validate_archive_names(
    names: Sequence[str],
    *,
    archive_kind: str,
) -> None:
    if len(names) != len(set(names)):
        raise ReleaseValidationError(f"{archive_kind} contains duplicate paths")
    for name in names:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise ReleaseValidationError(
                f"{archive_kind} contains an unsafe path"
            )
        if any(_is_forbidden_part(part) for part in path.parts):
            raise ReleaseValidationError(
                f"{archive_kind} contains repository-only or private content"
            )
        if path.suffix.lower() in _FORBIDDEN_SUFFIXES:
            raise ReleaseValidationError(
                f"{archive_kind} contains credential-like content"
            )


def _is_forbidden_part(part: str) -> bool:
    return part in _FORBIDDEN_PARTS or part.startswith(".env.")


def _require_exact_artifact(
    paths: Sequence[Path],
    expected_name: str,
    artifact_kind: str,
) -> None:
    actual_names = tuple(path.name for path in paths)
    if actual_names != (expected_name,):
        raise ReleaseValidationError(
            f"release must contain exactly one {artifact_kind}: {expected_name}"
        )


def _artifact_record(path: Path) -> ReleaseArtifact:
    digest = hashlib.sha256()
    with path.open("rb") as artifact_file:
        for chunk in iter(lambda: artifact_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return ReleaseArtifact(
        filename=path.name,
        size_bytes=path.stat().st_size,
        sha256=digest.hexdigest(),
    )


def _run_git(arguments: Sequence[str], cwd: Path) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=cwd,
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        command = arguments[0] if arguments else "query"
        raise ReleaseValidationError(f"Git {command} failed")
    return completed.stdout.strip()


def _require_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ReleaseValidationError(f"{field_name} must be non-empty text")
    return value


def _require_string_sequence(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ReleaseValidationError("project dependency list must be an array")
    copied = tuple(value)
    if any(not isinstance(item, str) or not item for item in copied):
        raise ReleaseValidationError("project dependencies must be text")
    return cast(tuple[str, ...], copied)


def main(argv: Sequence[str] | None = None) -> int:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=repository_root)
    parser.add_argument("--dist-dir", type=Path, default=repository_root / "dist")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--write", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        validate_repository_identity(
            arguments.repository_root,
            tag=arguments.tag,
            commit=arguments.commit,
        )
        manifest = validate_release(
            arguments.repository_root,
            arguments.dist_dir,
            tag=arguments.tag,
            commit=arguments.commit,
        )
        if arguments.write:
            output = write_release_manifest(manifest, arguments.dist_dir)
            print(f"release manifest written: {output.name}")
        else:
            print(manifest.to_json(), end="")
    except (
        BadZipFile,
        OSError,
        ReleaseValidationError,
        tarfile.TarError,
        tomllib.TOMLDecodeError,
    ) as error:
        print(f"release validation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
