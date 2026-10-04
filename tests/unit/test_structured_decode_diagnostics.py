"""Opt-in decoder diagnostics expose only schema-owned constants."""

from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError, replace
import unittest

from agentrig.capabilities import (
    StructuredDecodeDiagnostic,
    StructuredGenerationRequest,
    StructuredOutputDecodeError,
    StructuredOutputSchema,
    TextGenerationRequest,
)
from agentrig.core import JsonValue
from agentrig.testing import ScriptedStructuredGenerator
from tests.unit.test_scripted_generation import (
    create_context,
    structured_descriptor,
    structured_outcome,
)


def diagnostic() -> StructuredDecodeDiagnostic:
    return StructuredDecodeDiagnostic(
        code="required_field", location="/items/*/title"
    )


def decode(value: JsonValue) -> JsonValue:
    return value


def schema() -> StructuredOutputSchema[JsonValue]:
    return StructuredOutputSchema(
        schema_id="synthetic.diagnostics.v1",
        json_schema={"type": "object"},
        decoder=decode,
        decode_diagnostics={"missing_title": diagnostic()},
    )


class StructuredDecodeDiagnosticTest(unittest.TestCase):
    def test_diagnostic_and_registry_are_frozen_copies(self) -> None:
        original = {"missing_title": diagnostic()}
        selected = replace(schema(), decode_diagnostics=original)
        original["missing_title"] = StructuredDecodeDiagnostic(
            code="other", location="/other"
        )
        self.assertEqual(selected.decode_diagnostics["missing_title"], diagnostic())
        with self.assertRaises(TypeError):
            selected.decode_diagnostics["missing_title"] = diagnostic()  # type: ignore[index]
        with self.assertRaises(FrozenInstanceError):
            selected.decode_diagnostics["missing_title"].code = "other"  # type: ignore[misc]
        self.assertNotIn("missing_title", repr(selected))
        self.assertEqual(dict(selected.json_schema), {"type": "object"})

    def test_bounded_code_location_and_key_reject_unsafe_shapes(self) -> None:
        for value in ("", " padded", "a\nb", "a" * 97, "private value", 3, True):
            with self.subTest(token=value):
                with self.assertRaisesRegex(ValueError, "token is invalid"):
                    StructuredDecodeDiagnostic(code=value, location="/")  # type: ignore[arg-type]
                with self.assertRaisesRegex(ValueError, "token is invalid"):
                    StructuredOutputDecodeError(value)  # type: ignore[arg-type]
                with self.assertRaisesRegex(ValueError, "token is invalid"):
                    replace(schema(), decode_diagnostics={value: diagnostic()})  # type: ignore[dict-item]
        for value in ("", "title", "/private value", "/a\nb", "/" + "a" * 256, True):
            with self.subTest(location=value):
                with self.assertRaisesRegex(ValueError, "location is invalid"):
                    StructuredDecodeDiagnostic(code="field", location=value)  # type: ignore[arg-type]
        self.assertEqual(StructuredDecodeDiagnostic(code="root", location="/").location, "/")

    def test_registry_rejects_untyped_entries_and_bounds_inventory(self) -> None:
        with self.assertRaises(TypeError):
            replace(schema(), decode_diagnostics=[])  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            replace(schema(), decode_diagnostics={"key": "private"})  # type: ignore[dict-item]
        with self.assertRaisesRegex(ValueError, "registry is too large"):
            replace(schema(), decode_diagnostics={f"key{index}": diagnostic() for index in range(257)})

    def test_exception_text_is_fixed_and_only_registered_pairs_resolve(self) -> None:
        error = StructuredOutputDecodeError("missing_title")
        selected = schema()
        self.assertEqual(str(error), "structured output decoder rejected value")
        self.assertEqual(error.args, ("structured output decoder rejected value",))
        self.assertNotIn("missing_title", repr(error))
        self.assertIs(selected.resolve_decode_diagnostic(error), selected.decode_diagnostics["missing_title"])
        self.assertIsNone(selected.resolve_decode_diagnostic(StructuredOutputDecodeError("unregistered_private")))
        self.assertIsNone(replace(selected, decode_diagnostics={}).resolve_decode_diagnostic(error))
        for ordinary in (ValueError("private"), TypeError("private"), RuntimeError("private")):
            self.assertIsNone(selected.resolve_decode_diagnostic(ordinary))

    def test_cannot_mix_diagnostic_fields_from_unregistered_exception_attributes(self) -> None:
        error = StructuredOutputDecodeError("missing_title")
        error.code = "private_code"  # type: ignore[attr-defined]
        error.location = "/private"  # type: ignore[attr-defined]
        self.assertEqual(schema().resolve_decode_diagnostic(error), diagnostic())
        error.diagnostic_key = {"private": "value"}  # type: ignore[assignment]
        self.assertIsNone(schema().resolve_decode_diagnostic(error))
        error.diagnostic_key = "private" * 100
        self.assertIsNone(schema().resolve_decode_diagnostic(error))
        del error.diagnostic_key
        self.assertIsNone(schema().resolve_decode_diagnostic(error))

    def test_typed_decoder_runs_once_and_preserves_provider_free_diagnostic(self) -> None:
        seen: list[JsonValue] = []

        def reject(value: JsonValue) -> JsonValue:
            seen.append(value)
            raise StructuredOutputDecodeError("missing_title")

        selected = replace(schema(), decoder=reject)
        with self.assertRaises(StructuredOutputDecodeError) as raised:
            selected.decode({"items": [{"unexpected": "private"}]})
        self.assertEqual(len(seen), 1)
        self.assertEqual(selected.resolve_decode_diagnostic(raised.exception), diagnostic())
        with self.assertRaises(TypeError):
            seen[0]["changed"] = True  # type: ignore[index]

    def test_scripted_generator_uses_same_opt_in_schema_without_transport_policy(self) -> None:
        def reject(value: JsonValue) -> JsonValue:
            raise StructuredOutputDecodeError("missing_title")

        selected = replace(schema(), decoder=reject)
        generator = ScriptedStructuredGenerator[JsonValue](
            descriptor=structured_descriptor(), outcomes=(structured_outcome(),)
        )
        request = StructuredGenerationRequest(
            input=TextGenerationRequest(prompt="synthetic", max_output_tokens=32),
            output_schema=selected,
        )
        with self.assertRaises(StructuredOutputDecodeError) as raised:
            asyncio.run(generator.generate(request, create_context()))
        self.assertEqual(selected.resolve_decode_diagnostic(raised.exception), diagnostic())
        self.assertEqual(len(generator.calls), 1)
        self.assertTrue(generator.is_exhausted)


if __name__ == "__main__":
    unittest.main()
