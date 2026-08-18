from __future__ import annotations

import asyncio
import unittest
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime

from agentrig.agents import (
    AgentContract,
    AgentExecutionResult,
    AgentLimits,
    AgentRuntimeCatalog,
    AgentRuntimeRegistration,
    ConfiguredAgent,
)
from agentrig.capabilities import (
    CapabilityDescriptor,
    CapabilityFeature,
    CapabilityKind,
    CapabilityRequirements,
    DataRetention,
)
from agentrig.core import (
    AgentRigError,
    CancellationSource,
    EffectProfile,
    JsonValue,
    RunContext,
    RunId,
)
from agentrig.testing import ScriptedAgentRuntime, ScriptedAgentScenario


@dataclass(frozen=True)
class FixedClock:
    def now(self) -> datetime:
        return datetime(2026, 8, 17, 18, 0, tzinfo=UTC)

    def monotonic(self) -> float:
        return 100.0


@dataclass
class SequentialRunIdGenerator:
    next_value: int = 1

    def generate(self) -> RunId:
        run_id = RunId(f"run-{self.next_value}")
        self.next_value += 1
        return run_id


@dataclass(frozen=True)
class InputCodec:
    schema_id: str = "example.input.v1"

    def encode(self, value: str) -> JsonValue:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("input required")
        return {"value": value}


@dataclass(frozen=True)
class OutputCodec:
    schema_id: str = "example.output.v1"

    def decode(self, value: JsonValue) -> str:
        if not isinstance(value, Mapping) or set(value) != {"value"}:
            raise ValueError("output object required")
        output = value["value"]
        if not isinstance(output, str) or not output.strip():
            raise ValueError("output value required")
        return output


@dataclass(frozen=True)
class CountingRuntime:
    calls: list[str] = field(default_factory=list, compare=False, repr=False)

    async def execute(self, request: object, context: object) -> object:
        del request, context
        self.calls.append("execute")
        raise AssertionError("runtime must not execute during resolution")


def descriptor(
    capability_id: str,
    *,
    features: frozenset[CapabilityFeature] = frozenset(),
) -> CapabilityDescriptor:
    return CapabilityDescriptor(
        capability_id=capability_id,
        version="1",
        kind=CapabilityKind.AGENT_RUNTIME,
        features=features,
        data_retention=DataRetention.NOT_RETAINED,
    )


def requirements(
    *,
    features: frozenset[CapabilityFeature] = frozenset(),
) -> CapabilityRequirements:
    return CapabilityRequirements(
        kind=CapabilityKind.AGENT_RUNTIME,
        features=features,
        allowed_data_retention=frozenset({DataRetention.NOT_RETAINED}),
    )


def context() -> RunContext:
    return RunContext.create_root(
        clock=FixedClock(),
        id_generator=SequentialRunIdGenerator(),
        cancellation=CancellationSource().token,
    )


def contract(agent_id: str) -> AgentContract[str, str]:
    return AgentContract(
        agent_id=agent_id,
        version="1",
        purpose="Return one selected runtime identity",
        input_schema="example.input.v1",
        output_schema="example.output.v1",
        prompt_version="prompt-1",
        effect_profile=EffectProfile.READ_ONLY,
        limits=AgentLimits(max_turns=1, max_tool_calls=0),
        stopping_policy="output_schema_satisfied",
    )


class AgentRuntimeRegistrationTest(unittest.TestCase):
    def test_requires_agent_runtime_descriptor_and_protocol(self) -> None:
        runtime = CountingRuntime()
        registration = AgentRuntimeRegistration(
            binding_id="codex-primary",
            descriptor=descriptor("openai.codex.agent_runtime"),
            runtime=runtime,  # type: ignore[arg-type]
        )

        self.assertEqual(registration.binding_id, "codex-primary")
        self.assertNotIn("calls=", repr(registration))
        self.assertEqual(runtime.calls, [])

        coding = CapabilityDescriptor(
            capability_id="example.coding",
            version="1",
            kind=CapabilityKind.CODING,
        )
        with self.assertRaises(ValueError):
            AgentRuntimeRegistration(
                binding_id="wrong-kind",
                descriptor=coding,
                runtime=runtime,  # type: ignore[arg-type]
            )
        with self.assertRaises(TypeError):
            AgentRuntimeRegistration(
                binding_id="wrong-runtime",
                descriptor=descriptor("example.runtime"),
                runtime=object(),  # type: ignore[arg-type]
            )


class AgentRuntimeCatalogTest(unittest.TestCase):
    def test_resolves_exact_compatible_binding_without_execution(self) -> None:
        runtime = CountingRuntime()
        registration = AgentRuntimeRegistration(
            binding_id="codex-primary",
            descriptor=descriptor(
                "openai.codex.agent_runtime",
                features=frozenset({CapabilityFeature.STRUCTURED_OUTPUT}),
            ),
            runtime=runtime,  # type: ignore[arg-type]
        )
        catalog = AgentRuntimeCatalog((registration,))

        resolved = catalog.resolve(
            "codex-primary",
            requirements(
                features=frozenset({CapabilityFeature.STRUCTURED_OUTPUT})
            ),
        )

        self.assertIs(resolved, runtime)
        self.assertEqual(catalog.binding_ids, ("codex-primary",))
        self.assertEqual(catalog.registrations, (registration,))
        self.assertEqual(runtime.calls, [])
        self.assertEqual(repr(catalog), "AgentRuntimeCatalog()")

    def test_duplicate_unknown_disabled_and_incompatible_fail_closed(self) -> None:
        runtime = CountingRuntime()
        registration = AgentRuntimeRegistration(
            binding_id="local-runtime",
            descriptor=descriptor("example.local.agent_runtime"),
            runtime=runtime,  # type: ignore[arg-type]
        )
        with self.assertRaisesRegex(ValueError, "must be unique"):
            AgentRuntimeCatalog((registration, registration))

        disabled = AgentRuntimeRegistration(
            binding_id="disabled-runtime",
            descriptor=descriptor("example.disabled.agent_runtime"),
            runtime=runtime,  # type: ignore[arg-type]
            enabled=False,
        )
        catalog = AgentRuntimeCatalog((registration, disabled))

        cases = (
            ("unknown-runtime", requirements(), "agent_runtime.binding_unknown"),
            (
                "disabled-runtime",
                requirements(),
                "agent_runtime.binding_disabled",
            ),
            (
                "local-runtime",
                requirements(
                    features=frozenset({CapabilityFeature.TOOL_USE})
                ),
                "agent_runtime.binding_incompatible",
            ),
        )
        for binding_id, required, code in cases:
            with self.subTest(code=code):
                with self.assertRaises(AgentRigError) as raised:
                    catalog.resolve(binding_id, required)
                self.assertEqual(raised.exception.failure.code, code)
        self.assertEqual(runtime.calls, [])

    def test_failures_and_representations_omit_private_runtime_state(self) -> None:
        private_value = "credential-value-must-not-appear"

        @dataclass(frozen=True)
        class PrivateRuntime:
            secret: str = field(repr=False)

            async def execute(self, request: object, context: object) -> object:
                del request, context
                raise AssertionError("not called")

        registration = AgentRuntimeRegistration(
            binding_id="private-runtime",
            descriptor=descriptor("example.private.agent_runtime"),
            runtime=PrivateRuntime(private_value),  # type: ignore[arg-type]
        )
        catalog = AgentRuntimeCatalog((registration,))

        with self.assertRaises(AgentRigError) as raised:
            catalog.resolve(
                "private-runtime",
                requirements(
                    features=frozenset({CapabilityFeature.TOOL_USE})
                ),
            )

        rendered = " ".join(
            (
                repr(registration),
                repr(catalog),
                repr(raised.exception),
                repr(raised.exception.failure),
            )
        )
        self.assertNotIn(private_value, rendered)

    def test_two_configured_agents_use_distinct_resolved_runtimes(self) -> None:
        codex_runtime = ScriptedAgentRuntime(
            scenarios=(
                ScriptedAgentScenario(
                    result=AgentExecutionResult.succeeded({"value": "codex"})
                ),
            )
        )
        ollama_runtime = ScriptedAgentRuntime(
            scenarios=(
                ScriptedAgentScenario(
                    result=AgentExecutionResult.succeeded({"value": "ollama"})
                ),
            )
        )
        catalog = AgentRuntimeCatalog(
            (
                AgentRuntimeRegistration(
                    binding_id="codex-primary",
                    descriptor=descriptor("openai.codex.agent_runtime"),
                    runtime=codex_runtime,
                ),
                AgentRuntimeRegistration(
                    binding_id="ollama-local",
                    descriptor=descriptor("ollama.agent_runtime"),
                    runtime=ollama_runtime,
                ),
            )
        )
        configured = tuple(
            ConfiguredAgent(
                runtime=catalog.resolve(binding_id, requirements()),
                contract=contract(agent_id),
                instructions="Return one selected runtime identity.",
                input_codec=InputCodec(),
                output_codec=OutputCodec(),
            )
            for binding_id, agent_id in (
                ("codex-primary", "researcher"),
                ("ollama-local", "writer"),
            )
        )

        results = tuple(
            asyncio.run(agent.run("input", context())) for agent in configured
        )

        self.assertEqual(
            tuple(result.unwrap() for result in results),
            ("codex", "ollama"),
        )
        self.assertEqual(len(codex_runtime.calls), 1)
        self.assertEqual(len(ollama_runtime.calls), 1)


if __name__ == "__main__":
    unittest.main()
