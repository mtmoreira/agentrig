from __future__ import annotations

import asyncio
import gc
import unittest
import warnings
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from openai_codex.client import CodexConfig
from openai_codex.generated.v2_all import (
    AgentMessageThreadItem,
    CommandExecutionStatus,
    CommandExecutionThreadItem,
    ItemCompletedNotification,
    ItemStartedNotification,
    LegacyAppPathString,
    MessagePhase,
    ThreadItem,
    ThreadStartParams,
    ThreadTokenUsage,
    ThreadTokenUsageUpdatedNotification,
    TokenUsageBreakdown,
    Turn,
    TurnCompletedNotification,
    TurnStartParams,
    TurnStatus,
)
from openai_codex.models import JsonObject, Notification

from agentrig.agents import (
    AgentRuntimeCatalog,
    AgentRuntimeRegistration,
)
from agentrig.capabilities import (
    CapabilityFeature,
    CapabilityKind,
    CapabilityRequirements,
)
from agentrig.core import AgentRigError
from agentrig.integrations.openai import (
    CODEX_AGENT_RUNTIME_CAPABILITY,
    CODEX_SHELL_TOOL,
    CodexAgentRuntime,
    CodexApprovalMode,
    CodexApprovalRequested,
    CodexClient,
    CodexSandboxMode,
    CodexSandboxPolicy,
    CodexThreadRequest,
    CodexToolCallCompleted,
    CodexToolCallStarted,
    CodexTurnCompleted,
    CodexTurnRequest,
    CodexTurnStatus,
    CodexUsageReported,
)
from agentrig.integrations.openai.sdk import CodexSdkClientFactory


class FakeRawClient:
    def __init__(
        self,
        config: CodexConfig,
        approval_handler: Callable[[str, JsonObject | None], JsonObject],
    ) -> None:
        self.config = config
        self.approval_handler = approval_handler
        self.started = False
        self.initialized = False
        self.closed = False
        self.thread_params: list[ThreadStartParams] = []
        self.turn_params: list[TurnStartParams] = []
        self.notifications: list[Notification] = []
        self.interrupts: list[tuple[str, str]] = []
        self.unregistered: list[str] = []

    def start(self) -> None:
        self.started = True

    def initialize(self) -> object:
        self.initialized = True
        return object()

    def close(self) -> None:
        self.closed = True

    def thread_start(self, params: ThreadStartParams) -> object:
        self.thread_params.append(params)
        return SimpleNamespace(thread=SimpleNamespace(id="thread-1"))

    def thread_resume(self, thread_id: str, params: object) -> object:
        del params
        return SimpleNamespace(thread=SimpleNamespace(id=thread_id))

    def turn_start(
        self,
        thread_id: str,
        input_items: str,
        params: TurnStartParams,
    ) -> object:
        self.assert_turn_values = (thread_id, input_items)
        self.turn_params.append(params)
        return SimpleNamespace(turn=SimpleNamespace(id="turn-1"))

    def turn_interrupt(self, thread_id: str, turn_id: str) -> object:
        self.interrupts.append((thread_id, turn_id))
        return object()

    def next_turn_notification(self, turn_id: str) -> Notification:
        if turn_id != "turn-1":
            raise AssertionError("unexpected turn")
        return self.notifications.pop(0)

    def unregister_turn_notifications(self, turn_id: str) -> None:
        self.unregistered.append(turn_id)


def command_item(status: CommandExecutionStatus) -> ThreadItem:
    return ThreadItem(
        root=CommandExecutionThreadItem(
            id="call-1",
            type="commandExecution",
            command="private command",
            command_actions=[],
            cwd=LegacyAppPathString(root="/workspace"),
            status=status,
        )
    )


def completed_turn() -> Turn:
    return Turn(
        id="turn-1",
        items=[],
        status=TurnStatus.completed,
    )


class CodexSdkBridgeTest(unittest.TestCase):
    def test_catalog_preflight_does_not_resolve_authentication(self) -> None:
        builder_calls = 0

        class AuthenticationSource:
            calls = 0

            def resolve_environment(self) -> dict[str, str]:
                self.calls += 1
                return {"EXAMPLE_AUTH": "private"}

        def build(
            config: CodexConfig,
            approval_handler: Callable[[str, JsonObject | None], JsonObject],
        ) -> Any:
            nonlocal builder_calls
            del config, approval_handler
            builder_calls += 1
            raise AssertionError("builder must not be called")

        source = AuthenticationSource()
        runtime = CodexAgentRuntime(
            client_factory=CodexSdkClientFactory(
                authentication_source=source,
                raw_client_builder=build,
            ),
            model="gpt-5.6-terra",
            sandbox=CodexSandboxPolicy(
                mode=CodexSandboxMode.READ_ONLY,
                cwd="/workspace",
            ),
            output_schemas={"example.output.v1": {"type": "object"}},
        )
        catalog = AgentRuntimeCatalog(
            (
                AgentRuntimeRegistration(
                    binding_id="codex-primary",
                    descriptor=CODEX_AGENT_RUNTIME_CAPABILITY,
                    runtime=runtime,
                ),
            )
        )

        with self.assertRaises(AgentRigError) as raised:
            catalog.resolve(
                "codex-primary",
                CapabilityRequirements(
                    kind=CapabilityKind.AGENT_RUNTIME,
                    features=frozenset({CapabilityFeature.CITATIONS}),
                ),
            )

        self.assertEqual(
            raised.exception.failure.code,
            "agent_runtime.binding_incompatible",
        )
        self.assertEqual(source.calls, 0)
        self.assertEqual(builder_calls, 0)

    def test_resolves_and_copies_authentication_only_at_client_creation(self) -> None:
        private_value = "private-authentication-value"
        environment = {"EXAMPLE_AUTH": private_value}
        captured: list[FakeRawClient] = []

        class AuthenticationSource:
            calls = 0

            def resolve_environment(self) -> dict[str, str]:
                self.calls += 1
                return environment

        def build(
            config: CodexConfig,
            approval_handler: Callable[[str, JsonObject | None], JsonObject],
        ) -> Any:
            raw = FakeRawClient(config, approval_handler)
            captured.append(raw)
            return raw

        source = AuthenticationSource()
        factory = CodexSdkClientFactory(
            authentication_source=source,
            raw_client_builder=build,
        )

        self.assertEqual(source.calls, 0)
        client = factory.create()
        environment["EXAMPLE_AUTH"] = "changed"

        self.assertEqual(source.calls, 1)
        self.assertEqual(
            captured[0].config.env,
            {"EXAMPLE_AUTH": private_value},
        )
        self.assertNotIn(private_value, repr(factory))
        self.assertNotIn(private_value, repr(client))

    def test_authentication_resolution_failure_is_safe_and_skips_builder(self) -> None:
        private_value = "private-resolution-failure"
        builder_calls = 0

        class FailingAuthenticationSource:
            def resolve_environment(self) -> dict[str, str]:
                raise RuntimeError(private_value)

        def build(
            config: CodexConfig,
            approval_handler: Callable[[str, JsonObject | None], JsonObject],
        ) -> Any:
            nonlocal builder_calls
            del config, approval_handler
            builder_calls += 1
            raise AssertionError("builder must not be called")

        factory = CodexSdkClientFactory(
            authentication_source=FailingAuthenticationSource(),
            raw_client_builder=build,
        )

        with self.assertRaises(AgentRigError) as raised:
            factory.create()

        self.assertEqual(builder_calls, 0)
        self.assertEqual(
            raised.exception.failure.code,
            "codex.authentication_resolution_failed",
        )
        self.assertNotIn(private_value, repr(raised.exception))
        self.assertNotIn(private_value, repr(raised.exception.failure))

    def test_maps_stable_sdk_lifecycle_without_raw_payloads(self) -> None:
        captured: list[FakeRawClient] = []

        def build(
            config: CodexConfig,
            approval_handler: Callable[[str, JsonObject | None], JsonObject],
        ) -> Any:
            raw = FakeRawClient(config, approval_handler)
            captured.append(raw)
            return raw

        client = CodexSdkClientFactory(raw_client_builder=build).create()
        self.assertIsInstance(client, CodexClient)
        sandbox = CodexSandboxPolicy(
            mode=CodexSandboxMode.WORKSPACE_WRITE,
            cwd="/workspace",
            writable_roots=("/workspace",),
        )

        async def exercise() -> tuple[object, ...]:
            thread = await client.start_thread(
                CodexThreadRequest(
                    model="gpt-5.3-codex",
                    instructions="Return JSON.",
                    sandbox=sandbox,
                    approval_mode=CodexApprovalMode.MANUAL,
                    allowed_tools=(CODEX_SHELL_TOOL,),
                )
            )
            turn = await thread.start_turn(
                CodexTurnRequest(
                    prompt='{"value":"draft"}',
                    output_schema={"type": "object"},
                    sandbox=sandbox,
                    approval_mode=CodexApprovalMode.MANUAL,
                )
            )
            raw = captured[0]
            usage = TokenUsageBreakdown(
                input_tokens=10,
                cached_input_tokens=2,
                output_tokens=4,
                reasoning_output_tokens=1,
                total_tokens=14,
            )
            raw.notifications.extend(
                [
                    Notification(
                        method="item/started",
                        payload=ItemStartedNotification(
                            item=command_item(CommandExecutionStatus.in_progress),
                            started_at_ms=1,
                            thread_id="thread-1",
                            turn_id="turn-1",
                        ),
                    ),
                    Notification(
                        method="item/completed",
                        payload=ItemCompletedNotification(
                            item=command_item(CommandExecutionStatus.completed),
                            completed_at_ms=2,
                            thread_id="thread-1",
                            turn_id="turn-1",
                        ),
                    ),
                    Notification(
                        method="thread/tokenUsage/updated",
                        payload=ThreadTokenUsageUpdatedNotification(
                            thread_id="thread-1",
                            turn_id="turn-1",
                            token_usage=ThreadTokenUsage(last=usage, total=usage),
                        ),
                    ),
                    Notification(
                        method="item/completed",
                        payload=ItemCompletedNotification(
                            item=ThreadItem(
                                root=AgentMessageThreadItem(
                                    id="message-1",
                                    type="agentMessage",
                                    phase=MessagePhase.final_answer,
                                    text='{"result":"complete"}',
                                )
                            ),
                            completed_at_ms=3,
                            thread_id="thread-1",
                            turn_id="turn-1",
                        ),
                    ),
                    Notification(
                        method="turn/completed",
                        payload=TurnCompletedNotification(
                            thread_id="thread-1",
                            turn=completed_turn(),
                        ),
                    ),
                ]
            )
            events = tuple([event async for event in turn.events()])
            await client.close()
            return events

        events = asyncio.run(exercise())
        raw = captured[0]

        self.assertFalse(raw.config.experimental_api)
        self.assertIsNone(raw.config.env)
        self.assertEqual(
            raw.config.config_overrides,
            (
                "features.shell_tool=false",
                'web_search="disabled"',
                "tools.view_image=false",
                "features.multi_agent=false",
                "features.skill_mcp_dependency_install=false",
                'history.persistence="none"',
            ),
        )
        self.assertTrue(raw.started)
        self.assertTrue(raw.initialized)
        self.assertTrue(raw.closed)
        thread_config = raw.thread_params[0].config
        if thread_config is None:
            raise AssertionError("thread config is required")
        self.assertIs(thread_config["features"]["multi_agent"], False)
        self.assertEqual(thread_config["features"]["shell_tool"], True)
        self.assertNotIn("agents", thread_config)
        self.assertEqual(thread_config["web_search"], "disabled")
        self.assertEqual(thread_config["mcp_servers"], {})
        turn_sandbox = raw.turn_params[0].sandbox_policy
        if turn_sandbox is None:
            raise AssertionError("turn sandbox policy is required")
        self.assertEqual(turn_sandbox.root.type, "workspaceWrite")
        self.assertEqual(raw.unregistered, ["turn-1"])
        self.assertIsInstance(events[0], CodexToolCallStarted)
        self.assertIsInstance(events[1], CodexToolCallCompleted)
        self.assertIsInstance(events[2], CodexUsageReported)
        self.assertEqual(
            events[-1],
            CodexTurnCompleted(
                turn_id="turn-1",
                status=CodexTurnStatus.SUCCEEDED,
                output={"result": "complete"},
            ),
        )
        self.assertNotIn("private command", repr(events))

    def test_declines_approvals_and_emits_only_safe_identity(self) -> None:
        captured: list[FakeRawClient] = []

        def build(
            config: CodexConfig,
            approval_handler: Callable[[str, JsonObject | None], JsonObject],
        ) -> Any:
            raw = FakeRawClient(config, approval_handler)
            captured.append(raw)
            return raw

        client = CodexSdkClientFactory(raw_client_builder=build).create()

        async def exercise() -> tuple[object, ...]:
            sandbox = CodexSandboxPolicy(
                mode=CodexSandboxMode.READ_ONLY,
                cwd="/workspace",
            )
            thread = await client.start_thread(
                CodexThreadRequest(
                    model="gpt-5.3-codex",
                    instructions="Return JSON.",
                    sandbox=sandbox,
                    approval_mode=CodexApprovalMode.MANUAL,
                    allowed_tools=(CODEX_SHELL_TOOL,),
                )
            )
            turn = await thread.start_turn(
                CodexTurnRequest(
                    prompt='{"value":"draft"}',
                    output_schema={"type": "object"},
                    sandbox=sandbox,
                    approval_mode=CodexApprovalMode.MANUAL,
                )
            )
            raw = captured[0]
            decision = raw.approval_handler(
                "item/commandExecution/requestApproval",
                {
                    "threadId": "thread-1",
                    "turnId": "turn-1",
                    "itemId": "call-1",
                    "command": "password=private",
                },
            )
            self.assertEqual(decision, {"decision": "decline"})
            raw.notifications.append(
                Notification(
                    method="turn/completed",
                    payload=TurnCompletedNotification(
                        thread_id="thread-1",
                        turn=completed_turn(),
                    ),
                )
            )
            events = tuple([event async for event in turn.events()])
            await client.close()
            return events

        events = asyncio.run(exercise())

        self.assertEqual(
            events[0],
            CodexApprovalRequested(
                turn_id="turn-1",
                approval_id="call-1",
                tool_name=CODEX_SHELL_TOOL,
            ),
        )
        self.assertNotIn("private", repr(events))

    def test_default_factory_initializes_the_pinned_runtime_offline(self) -> None:
        client = CodexSdkClientFactory().create()

        async def exercise() -> None:
            try:
                thread = await client.start_thread(
                    CodexThreadRequest(
                        model="gpt-5.6-terra",
                        instructions="Initialize one offline test thread.",
                        sandbox=CodexSandboxPolicy(
                            mode=CodexSandboxMode.READ_ONLY,
                            cwd=str(Path.cwd().resolve()),
                        ),
                        approval_mode=CodexApprovalMode.DENY_ALL,
                    )
                )
                self.assertTrue(thread.thread_id)
            finally:
                await client.close()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ResourceWarning)
            asyncio.run(exercise())
            gc.collect()

        resource_warnings = [
            warning
            for warning in caught
            if issubclass(warning.category, ResourceWarning)
        ]
        self.assertEqual(resource_warnings, [])


if __name__ == "__main__":
    unittest.main()
