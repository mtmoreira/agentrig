from __future__ import annotations

from collections.abc import Sequence
from email.message import EmailMessage
from io import BytesIO
from pathlib import Path
import tarfile
import tempfile
import unittest
from zipfile import ZIP_DEFLATED, ZipFile

from tools.validate_release import (
    RELEASE_MANIFEST_SCHEMA,
    ReleaseValidationError,
    validate_repository_identity,
    validate_release,
    write_release_manifest,
)


_COMMIT = "0123456789abcdef0123456789abcdef01234567"


class ReleaseValidationTest(unittest.TestCase):
    def test_repository_identity_requires_clean_matching_annotated_tag(self) -> None:
        calls: list[tuple[str, ...]] = []
        outputs = {
            ("rev-parse", "HEAD"): _COMMIT,
            ("status", "--porcelain=v1", "--untracked-files=all"): "",
            ("cat-file", "-t", "refs/tags/v0.1.0"): "tag",
            ("rev-parse", "v0.1.0^{commit}"): _COMMIT,
        }

        def run_git(arguments: Sequence[str], cwd: Path) -> str:
            normalized = tuple(arguments)
            calls.append(normalized)
            self.assertEqual(cwd, Path.cwd().resolve())
            return outputs[normalized]

        validate_repository_identity(
            Path.cwd(),
            tag="v0.1.0",
            commit=_COMMIT,
            run_git=run_git,
        )

        self.assertEqual(tuple(calls), tuple(outputs))

    def test_repository_identity_rejects_dirty_checkout(self) -> None:
        outputs = {
            ("rev-parse", "HEAD"): _COMMIT,
            ("status", "--porcelain=v1", "--untracked-files=all"): " M file",
        }

        with self.assertRaisesRegex(ReleaseValidationError, "must be clean"):
            validate_repository_identity(
                Path.cwd(),
                tag="v0.1.0",
                commit=_COMMIT,
                run_git=lambda arguments, cwd: outputs[tuple(arguments)],
            )

    def test_repository_identity_rejects_lightweight_tag(self) -> None:
        outputs = {
            ("rev-parse", "HEAD"): _COMMIT,
            ("status", "--porcelain=v1", "--untracked-files=all"): "",
            ("cat-file", "-t", "refs/tags/v0.1.0"): "commit",
        }

        with self.assertRaisesRegex(ReleaseValidationError, "must be annotated"):
            validate_repository_identity(
                Path.cwd(),
                tag="v0.1.0",
                commit=_COMMIT,
                run_git=lambda arguments, cwd: outputs[tuple(arguments)],
            )

    def test_valid_release_binds_tag_commit_and_artifact_digests(self) -> None:
        with self._candidate_release() as candidate:
            root, dist = candidate

            manifest = validate_release(
                root,
                dist,
                tag="v0.1.0",
                commit=_COMMIT,
            )
            repeated = validate_release(
                root,
                dist,
                tag="v0.1.0",
                commit=_COMMIT,
            )

        self.assertEqual(manifest.distribution, "agentrig")
        self.assertEqual(manifest.version, "0.1.0")
        self.assertEqual(manifest.tag, "v0.1.0")
        self.assertEqual(manifest.commit, _COMMIT)
        self.assertEqual(repeated.to_json(), manifest.to_json())
        self.assertEqual(
            tuple(artifact.filename for artifact in manifest.artifacts),
            (
                "agentrig-0.1.0-py3-none-any.whl",
                "agentrig-0.1.0.tar.gz",
            ),
        )
        self.assertIn(RELEASE_MANIFEST_SCHEMA, manifest.to_json())
        for artifact in manifest.artifacts:
            self.assertGreater(artifact.size_bytes, 0)
            self.assertEqual(len(artifact.sha256), 64)

    def test_tag_must_match_project_version(self) -> None:
        with self._candidate_release() as candidate:
            root, dist = candidate

            with self.assertRaisesRegex(ReleaseValidationError, "release tag"):
                validate_release(root, dist, tag="v0.2.0", commit=_COMMIT)

    def test_commit_must_be_a_full_lowercase_sha(self) -> None:
        with self._candidate_release() as candidate:
            root, dist = candidate

            with self.assertRaisesRegex(ReleaseValidationError, "full lowercase"):
                validate_release(root, dist, tag="v0.1.0", commit="abc123")

    def test_missing_distribution_artifact_is_rejected(self) -> None:
        with self._candidate_release() as candidate:
            root, dist = candidate
            (dist / "agentrig-0.1.0.tar.gz").unlink()

            with self.assertRaisesRegex(
                ReleaseValidationError,
                "source distribution",
            ):
                validate_release(root, dist, tag="v0.1.0", commit=_COMMIT)

    def test_additional_versioned_wheel_is_rejected(self) -> None:
        with self._candidate_release() as candidate:
            root, dist = candidate
            (dist / "agentrig-0.1.1-py3-none-any.whl").write_bytes(b"other")

            with self.assertRaisesRegex(ReleaseValidationError, "exactly one wheel"):
                validate_release(root, dist, tag="v0.1.0", commit=_COMMIT)

    def test_wheel_metadata_must_match_project(self) -> None:
        with self._candidate_release(wheel_version="0.2.0") as candidate:
            root, dist = candidate

            with self.assertRaisesRegex(ReleaseValidationError, "version metadata"):
                validate_release(root, dist, tag="v0.1.0", commit=_COMMIT)

    def test_repository_context_in_wheel_is_rejected(self) -> None:
        with self._candidate_release(extra_wheel_path="agentrig/AGENTS.md") as candidate:
            root, dist = candidate

            with self.assertRaisesRegex(
                ReleaseValidationError,
                "repository-only or private content",
            ):
                validate_release(root, dist, tag="v0.1.0", commit=_COMMIT)

    def test_repository_context_in_source_distribution_is_rejected(self) -> None:
        with self._candidate_release(
            extra_sdist_path="src/agentrig/AGENTS.md"
        ) as candidate:
            root, dist = candidate

            with self.assertRaisesRegex(
                ReleaseValidationError,
                "repository-only or private content",
            ):
                validate_release(root, dist, tag="v0.1.0", commit=_COMMIT)

    def test_manifest_write_is_idempotent_but_not_replaceable(self) -> None:
        with self._candidate_release() as candidate:
            root, dist = candidate
            manifest = validate_release(
                root,
                dist,
                tag="v0.1.0",
                commit=_COMMIT,
            )

            output = write_release_manifest(manifest, dist)
            self.assertEqual(output.read_text(encoding="utf-8"), manifest.to_json())
            self.assertEqual(write_release_manifest(manifest, dist), output)
            output.write_text("different\n", encoding="utf-8")

            with self.assertRaisesRegex(ReleaseValidationError, "different content"):
                write_release_manifest(manifest, dist)

    def _candidate_release(
        self,
        *,
        wheel_version: str = "0.1.0",
        extra_wheel_path: str | None = None,
        extra_sdist_path: str | None = None,
    ) -> _CandidateRelease:
        temporary_directory = tempfile.TemporaryDirectory()
        root = Path(temporary_directory.name)
        dist = root / "dist"
        dist.mkdir()
        self._write_pyproject(root)
        self._write_wheel(
            dist,
            version=wheel_version,
            extra_path=extra_wheel_path,
        )
        self._write_sdist(dist, extra_path=extra_sdist_path)
        return _CandidateRelease(temporary_directory, root, dist)

    def _write_pyproject(self, root: Path) -> None:
        (root / "pyproject.toml").write_text(
            """[project]
name = "agentrig"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = []

[project.optional-dependencies]
codex = ["openai-codex==0.144.4"]
""",
            encoding="utf-8",
        )

    def _write_wheel(
        self,
        dist: Path,
        *,
        version: str,
        extra_path: str | None,
    ) -> None:
        wheel_path = dist / "agentrig-0.1.0-py3-none-any.whl"
        with ZipFile(wheel_path, mode="w", compression=ZIP_DEFLATED) as archive:
            archive.writestr("agentrig/__init__.py", "")
            archive.writestr("agentrig/py.typed", "")
            archive.writestr(
                "agentrig-0.1.0.dist-info/METADATA",
                self._metadata(version).as_bytes(),
            )
            archive.writestr("agentrig-0.1.0.dist-info/WHEEL", "Wheel-Version: 1.0\n")
            archive.writestr("agentrig-0.1.0.dist-info/RECORD", "")
            if extra_path is not None:
                archive.writestr(extra_path, "repository context")

    def _write_sdist(self, dist: Path, *, extra_path: str | None) -> None:
        sdist_path = dist / "agentrig-0.1.0.tar.gz"
        prefix = "agentrig-0.1.0/"
        files = {
            "PKG-INFO": self._metadata("0.1.0").as_bytes(),
            "README.md": b"# AgentRig\n",
            "pyproject.toml": (
                b"[project]\n"
                b'name = "agentrig"\n'
                b'version = "0.1.0"\n'
            ),
            "src/agentrig/__init__.py": b"",
            "src/agentrig/py.typed": b"",
        }
        if extra_path is not None:
            files[extra_path] = b"repository context"
        with tarfile.open(sdist_path, mode="w:gz") as archive:
            root_info = tarfile.TarInfo(prefix.removesuffix("/"))
            root_info.type = tarfile.DIRTYPE
            root_info.mtime = 0
            archive.addfile(root_info)
            for relative_name, content in files.items():
                info = tarfile.TarInfo(f"{prefix}{relative_name}")
                info.size = len(content)
                info.mtime = 0
                archive.addfile(info, fileobj=BytesIO(content))

    def _metadata(self, version: str) -> EmailMessage:
        metadata = EmailMessage()
        metadata["Metadata-Version"] = "2.3"
        metadata["Name"] = "agentrig"
        metadata["Version"] = version
        metadata["Requires-Python"] = ">=3.12"
        metadata["Requires-Dist"] = "openai-codex==0.144.4 ; extra == 'codex'"
        metadata["Provides-Extra"] = "codex"
        return metadata


class _CandidateRelease:
    def __init__(
        self,
        owned: tempfile.TemporaryDirectory[str],
        root: Path,
        dist: Path,
    ) -> None:
        self._owned = owned
        self.root = root
        self.dist = dist

    def __enter__(self) -> tuple[Path, Path]:
        return self.root, self.dist

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> None:
        self._owned.cleanup()


if __name__ == "__main__":
    unittest.main()
