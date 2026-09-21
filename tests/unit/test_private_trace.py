from __future__ import annotations

import json
import unittest
from dataclasses import replace

from agentrig.core import (AgentRigError, PrivateTraceCapture, PrivateTraceRecord,
                           DEFAULT_REDACTION_POLICY, redact_private_trace_content)
from agentrig.agents import AgentRuntimeUsage
from agentrig.capabilities import GenerationUsage
from agentrig.integrations.bounded_process import _trace_argv
from tests.unit.test_ollama_runtime import create_context


class Recorder:
    def __init__(self) -> None:
        self.records: list[PrivateTraceRecord] = []

    def record(self, record: PrivateTraceRecord) -> None:
        self.records.append(record)


class PrivateTraceTest(unittest.TestCase):
    def test_opt_in_is_inherited_but_separate_from_safe_events_and_reprs(self) -> None:
        recorder = Recorder()
        context, events = create_context()
        self.assertIsNone(context.private_trace_capture)
        capture = PrivateTraceCapture(recorder=recorder)
        root = replace(context, private_trace_capture=capture, correlation={"operation_id": "operation-1"})
        child = root.derive_child()
        self.assertIs(child.private_trace_capture, capture)
        capture.record(child, kind="provider.request", metadata={"provider": "synthetic"},
                       content={"prompt": "synthetic private content", "api_key": "discard me"})
        record = recorder.records[0]
        self.assertEqual(record.parent_run_id, root.run_id)
        self.assertEqual(record.correlation["operation_id"], "operation-1")
        self.assertEqual(json.loads(record.content)["prompt"], "synthetic private content")
        self.assertEqual(json.loads(record.content)["api_key"], "[REDACTED]")
        self.assertNotIn("synthetic private content", repr(record))
        self.assertNotIn("discard me", record.content)
        self.assertEqual(events.events, ())

    def test_utf8_truncation_is_bounded_and_explicit(self) -> None:
        recorder = Recorder()
        capture = PrivateTraceCapture(recorder=recorder, max_content_bytes=12)
        context, _ = create_context()
        capture.record(context, kind="provider.response", metadata={}, content={"text": "é" * 32})
        record = recorder.records[0]
        self.assertTrue(record.truncated)
        self.assertGreater(record.content_bytes, len(record.content.encode()))
        self.assertLessEqual(len(record.content.encode()), 12)
        with self.assertRaises(ValueError):
            replace(record, truncated=False)

    def test_recorder_failure_does_not_expose_its_message(self) -> None:
        class FailingRecorder:
            def record(self, record: PrivateTraceRecord) -> None:
                raise RuntimeError("synthetic private storage failure")

        capture = PrivateTraceCapture(recorder=FailingRecorder())
        context, _ = create_context()
        with self.assertRaises(AgentRigError) as raised:
            capture.record(context, kind="provider.request", metadata={}, content={"text": "example"})
        self.assertEqual(raised.exception.failure.code, "private_trace.recording_failed")
        self.assertNotIn("synthetic private", str(raised.exception))

    def test_invalid_policy_and_unsafe_metadata_are_rejected(self) -> None:
        recorder = Recorder()
        for value in (0, -1, True, 4 * 1024 * 1024 + 1):
            with self.assertRaises(ValueError):
                PrivateTraceCapture(recorder=recorder, max_content_bytes=value)
        with self.assertRaises(TypeError):
            PrivateTraceCapture(recorder=recorder, include_process=1)  # type: ignore[arg-type]
        context, _ = create_context()
        with self.assertRaises(ValueError):
            PrivateTraceCapture(recorder=recorder).record(
                context, kind="provider.request", metadata={"api_key": "private"}, content={})
        self.assertEqual(recorder.records, [])

    def test_process_secret_flags_are_redacted(self) -> None:
        self.assertEqual(_trace_argv(("/bin/tool", "--password", "secret", "--api-key=value", "normal")),
                         ["/bin/tool", "--password", "[REDACTED]", "--api-key=[REDACTED]", "normal"])

    def test_token_details_are_subsets_without_double_counting(self) -> None:
        for cls in (AgentRuntimeUsage, GenerationUsage):
            usage = cls(input_tokens=12, cached_input_tokens=4, output_tokens=8, reasoning_tokens=3)
            self.assertEqual(usage.total_tokens, 20)
            with self.assertRaises(ValueError):
                cls(input_tokens=2, cached_input_tokens=3)
            with self.assertRaises(ValueError):
                cls(output_tokens=2, reasoning_tokens=3)
            with self.assertRaises(ValueError):
                cls(reasoning_tokens=1)

    def test_embedded_secrets_are_scrubbed_in_provider_text_and_process_output(self) -> None:
        cases = (
            'use password=hunter_fixture and continue',
            'OPENAI_API_KEY=fixture_key_value',
            'text "api_key": "fixture_key_value" more text',
            "pass client_secret='fixture secret with spaces' to nobody",
            'Authorization: Bearer fixture_short',
            'Cookie: sid=fixture_cookie; other=fixture_other_cookie',
            '--password fixture_password --normal keep',
            'before\n-----BEGIN RSA PRIVATE KEY-----\nfixture_key_body\n-----END RSA PRIVATE KEY-----\nafter',
            'before\n-----BEGIN PRIVATE KEY-----\nfixture_partial_key_body',
        )
        for value in cases:
            with self.subTest(value=value):
                content = redact_private_trace_content({'messages': [{'content': value}], 'stdout': value})
                rendered = repr(content)
                self.assertNotIn('fixture', rendered)
                self.assertIn('[REDACTED]', rendered)
        ordinary = {'content': 'password=fixture_password'}
        self.assertEqual(DEFAULT_REDACTION_POLICY.redact_json_object(ordinary), ordinary)

    def test_private_argv_scrub_handles_aliases_assignments_and_split_values(self) -> None:
        argv = ('/bin/tool', '--apiKey', 'fixture_key', '--apikey', 'fixture_key2', '--password=fixture_password',
                '-u', 'fixture_user:fixture_pass', '-pfixture_attached',
                'API_KEY=fixture_environment_arg', '-H', 'Authorization: Basic fixture_basic', '--count', '2')
        content = redact_private_trace_content({'argv': argv})
        self.assertNotIn('fixture', repr(content))
        self.assertEqual(content['argv'][-2:], ('--count', '2'))  # type: ignore[index]

    def test_credential_mapping_aliases_are_redacted_before_private_storage(self) -> None:
        recorder = Recorder()
        context, events = create_context()
        capture = PrivateTraceCapture(recorder=recorder)
        for name in ('passphrase', 'pwd', 'apikey', 'apiKey', 'OPENAI_API_KEY', 'client-secret'):
            with self.subTest(name=name):
                capture.record(context, kind='provider.response', metadata={}, content={
                    name: 'fixture_root_credential',
                    'nested': [{name: {'value': 'fixture_nested_credential'}, 'public': 'keep'}],
                    'count': 7,
                })
                saved = json.loads(recorder.records[-1].content)
                self.assertEqual(saved[name], '[REDACTED]')
                self.assertEqual(saved['nested'][0][name], '[REDACTED]')
                self.assertEqual(saved['nested'][0]['public'], 'keep')
                self.assertEqual(saved['count'], 7)
                self.assertNotIn('fixture_', recorder.records[-1].content)
        self.assertEqual(events.events, ())

    def test_private_scrub_is_detached_idempotent_and_bounded(self) -> None:
        source = {'text': 'before password=fixture_password after'}
        once = redact_private_trace_content(source)
        twice = redact_private_trace_content(once)
        self.assertEqual(once, twice)
        self.assertEqual(source['text'], 'before password=fixture_password after')
        oversized = redact_private_trace_content({'text': 'a' * (4 * 1024 * 1024 + 1)})
        self.assertEqual(oversized['text'], '[REDACTED: private text exceeds inspection bound]')


if __name__ == "__main__":
    unittest.main()
