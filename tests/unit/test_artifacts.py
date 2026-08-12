from __future__ import annotations

import unittest

from agentrig.core import ArtifactId, ArtifactRef, ContentDigest, RunId


def create_artifact(**overrides: object) -> ArtifactRef:
    values: dict[str, object] = {
        "artifact_id": ArtifactId("artifact-1"),
        "kind": "image",
        "media_type": "image/png",
        "producer_run_id": RunId("run-1"),
        "workspace_path": "outputs/cover.png",
    }
    values.update(overrides)
    return ArtifactRef(**values)  # type: ignore[arg-type]


class ArtifactIdTest(unittest.TestCase):
    def test_is_an_opaque_string_value(self) -> None:
        artifact_id = ArtifactId("artifact-1")

        self.assertEqual(artifact_id.value, "artifact-1")
        self.assertEqual(str(artifact_id), "artifact-1")

    def test_rejects_empty_or_padded_values(self) -> None:
        for invalid_value in ("", " artifact-1", "artifact-1 "):
            with self.subTest(invalid_value=invalid_value):
                with self.assertRaises(ValueError):
                    ArtifactId(invalid_value)


class ContentDigestTest(unittest.TestCase):
    def test_preserves_algorithm_and_value(self) -> None:
        digest = ContentDigest("sha256", "abc123")

        self.assertEqual(str(digest), "sha256:abc123")

    def test_rejects_invalid_algorithm_or_value(self) -> None:
        invalid_digests = (
            ("SHA256", "abc123"),
            ("sha 256", "abc123"),
            ("sha256", "abc 123"),
            ("sha256", ""),
        )
        for algorithm, value in invalid_digests:
            with self.subTest(algorithm=algorithm, value=value):
                with self.assertRaises(ValueError):
                    ContentDigest(algorithm, value)


class ArtifactRefTest(unittest.TestCase):
    def test_workspace_reference_copies_and_freezes_lineage(self) -> None:
        labels = {"privacy": "private", "retention": "project"}
        lineage = {"provider": "example", "model": "image-v1"}
        inputs = [ArtifactId("artifact-source")]

        artifact = create_artifact(
            content_digest=ContentDigest("sha256", "abc123"),
            input_artifact_ids=inputs,
            labels=labels,
            provider_lineage=lineage,
        )
        labels["privacy"] = "public"
        lineage["model"] = "mutated"
        inputs.append(ArtifactId("artifact-later"))

        self.assertEqual(
            artifact.input_artifact_ids,
            (ArtifactId("artifact-source"),),
        )
        self.assertEqual(artifact.labels["privacy"], "private")
        self.assertEqual(artifact.provider_lineage["model"], "image-v1")
        with self.assertRaises(TypeError):
            artifact.labels["new"] = "value"  # type: ignore[index]

    def test_uri_reference_requires_a_scheme(self) -> None:
        artifact = create_artifact(
            workspace_path=None,
            uri="s3://example-bucket/outputs/cover.png",
        )

        self.assertEqual(
            artifact.uri,
            "s3://example-bucket/outputs/cover.png",
        )
        with self.assertRaises(ValueError):
            create_artifact(workspace_path=None, uri="outputs/cover.png")

    def test_exactly_one_location_is_required(self) -> None:
        with self.assertRaises(ValueError):
            create_artifact(workspace_path=None)
        with self.assertRaises(ValueError):
            create_artifact(uri="s3://bucket/cover.png")

    def test_workspace_path_must_be_canonical_and_relative(self) -> None:
        for invalid_path in (
            "/tmp/cover.png",
            "../cover.png",
            "outputs/../cover.png",
            "./outputs/cover.png",
            "outputs//cover.png",
            "outputs\\cover.png",
        ):
            with self.subTest(invalid_path=invalid_path):
                with self.assertRaises(ValueError):
                    create_artifact(workspace_path=invalid_path)

    def test_kind_and_media_type_are_validated(self) -> None:
        for overrides in (
            {"kind": "Generated Image"},
            {"kind": "Image"},
            {"media_type": "image"},
            {"media_type": "image/ png"},
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    create_artifact(**overrides)

    def test_input_lineage_rejects_duplicates_and_self_reference(self) -> None:
        repeated_id = ArtifactId("artifact-source")
        with self.assertRaises(ValueError):
            create_artifact(input_artifact_ids=(repeated_id, repeated_id))
        with self.assertRaises(ValueError):
            create_artifact(input_artifact_ids=(ArtifactId("artifact-1"),))


if __name__ == "__main__":
    unittest.main()
