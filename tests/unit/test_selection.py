from __future__ import annotations

import json
import os
import stat
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from agentrig.selection import JsonPreferenceStore, ModelSet, PreferredModelSet, select_model_set


def model_set(name: str) -> ModelSet:
    return ModelSet(application_id="sample.books", model_set_id=name, version="1",
                    models={"writer": name, "image": f"image-{name}"},
                    settings={"quality": "standard", "repair_budget": 2, "enabled": True})


def preference() -> PreferredModelSet:
    return PreferredModelSet(application_id="sample.books", model_set=model_set("preferred"),
                             comparison_id="comparison-1", manifest_digest="a" * 64)


class SelectionTests(unittest.TestCase):
    def test_whole_model_set_precedence_and_resume(self) -> None:
        default, preferred, explicit, persisted = model_set("default"), preference(), model_set("explicit"), model_set("persisted")
        self.assertEqual(select_model_set(application_id="sample.books", default=default), default)
        self.assertEqual(select_model_set(application_id="sample.books", default=default, preference=preferred), preferred.model_set)
        self.assertEqual(select_model_set(application_id="sample.books", default=default, preference=preferred,
                                          explicit=explicit), explicit)
        self.assertEqual(select_model_set(application_id="sample.books", default=default, preference=preferred,
                                          explicit=explicit, persisted=persisted), persisted)
        self.assertEqual(preferred.model_set.settings["repair_budget"], 2)

    def test_scope_and_invalid_configuration_fail_instead_of_fallback(self) -> None:
        with self.assertRaises(ValueError):
            select_model_set(application_id="another.app", default=model_set("default"))
        with self.assertRaises(ValueError):
            replace(preference(), application_id="another.app")
        with self.assertRaises(ValueError):
            replace(model_set("default"), models={})
        with self.assertRaises(ValueError):
            replace(model_set("default"), settings={"secret": {"nested": "payload"}})
        with self.assertRaises(ValueError):
            replace(model_set("default"), settings={"cost": float("nan")})

    def test_roundtrip_frozen_settings_and_no_private_model_values_in_repr(self) -> None:
        models = {"writer": "PRIVATE MODEL"}
        settings = {"profile": "PRIVATE PROFILE"}
        original = ModelSet(application_id="sample.books", model_set_id="set", version="1",
                            models=models, settings=settings)
        models["writer"] = "mutated"
        settings["profile"] = "mutated"
        self.assertEqual(original.models["writer"], "PRIVATE MODEL")
        self.assertEqual(original.settings["profile"], "PRIVATE PROFILE")
        self.assertNotIn("PRIVATE", repr(original))
        self.assertEqual(original, ModelSet.from_data(json.loads(json.dumps(original.to_data()))))
        self.assertEqual(preference(), PreferredModelSet.from_data(preference().to_data()))
        for value in (True, 2):
            data = preference().to_data()
            data["schema_version"] = value
            with self.assertRaises(ValueError):
                PreferredModelSet.from_data(data)

    def test_explicit_private_file_store_roundtrip_scope_and_invalid_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preferred.json"
            store = JsonPreferenceStore(path=path)
            self.assertIsNone(store.read("sample.books"))
            store.write(preference())
            self.assertEqual(store.read("sample.books"), preference())
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            with self.assertRaises(ValueError):
                store.read("another.app")
            other = replace(preference(), application_id="another.app",
                            model_set=replace(preference().model_set, application_id="another.app"))
            with self.assertRaises(ValueError):
                store.write(other)
            self.assertEqual(store.read("sample.books"), preference())
            path.write_text('{"private":"SENSITIVE CONTENT",', encoding="utf-8")
            with self.assertRaises(ValueError) as captured:
                store.read("sample.books")
            self.assertNotIn("SENSITIVE", str(captured.exception))
            self.assertNotIn(str(path), repr(store))

    def test_duplicate_json_keys_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preferred.json"
            data = json.dumps(preference().to_data())
            path.write_text(data[:-1] + ', "schema_version": 1}', encoding="utf-8")
            with self.assertRaises(ValueError):
                JsonPreferenceStore(path=path).read("sample.books")

    def test_symbolic_links_and_oversized_files_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.json"
            link = root / "link.json"
            link.symlink_to(target)
            with self.assertRaises(ValueError):
                JsonPreferenceStore(path=link).read("sample.books")
            with self.assertRaises(ValueError):
                JsonPreferenceStore(path=link).write(preference())
            self.assertFalse(target.exists())
            target.write_bytes(b" " * 1_048_577)
            with self.assertRaises(ValueError):
                JsonPreferenceStore(path=target).read("sample.books")

    def test_deep_json_and_non_regular_files_fail_without_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "deep.json"
            path.write_text("[" * 2000 + "0" + "]" * 2000, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid"):
                JsonPreferenceStore(path=path).read("sample.books")
            with self.assertRaises(ValueError):
                JsonPreferenceStore(path=root).read("sample.books")
            if hasattr(os, "mkfifo"):
                fifo = root / "pipe"
                os.mkfifo(fifo)
                with self.assertRaisesRegex(ValueError, "regular file"):
                    JsonPreferenceStore(path=fifo).read("sample.books")


if __name__ == "__main__":
    unittest.main()
