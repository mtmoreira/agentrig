"""Positive fixture for the injected Ollama runtime boundary."""

from __future__ import annotations

from dataclasses import dataclass

from agentrig.agents import AgentContract, AgentLimits, AgentRuntime
from agentrig.capabilities import (
    CapabilityFeature,
    CapabilityKind,
    CapabilityRequirements,
    TextGenerationRequest,
    TextGenerator,
    TextMessage,
    TextMessageRole,
)
from agentrig.core import EffectProfile, PrivateTraceCapture, PrivateTraceRecord, PrivateTraceRecorder, RunContext
from agentrig.integrations import PersistentProcessSession, ProcessSpec
from agentrig.integrations.ollama import (
    OLLAMA_AGENT_RUNTIME_CAPABILITY,
    OllamaAgentRuntime,
    OllamaAuthenticationSource,
    OllamaChatRequest,
    OllamaChatResponse,
    OllamaClient,
    OllamaClientFactory,
    OllamaFinishReason,
    OllamaRuntimeOptions,
    OllamaTextGenerator,
)
from agentrig.integrations.ollama.sdk import OllamaSdkClientFactory


class FakeClient:
    async def chat(self, request: OllamaChatRequest) -> OllamaChatResponse:
        return OllamaChatResponse(
            content='{"answer":"complete"}',
            model=request.model,
            finish_reason=OllamaFinishReason.STOP,
        )

    async def close(self) -> None:
        return None


@dataclass(frozen=True)
class FakeFactory:
    def create(self) -> OllamaClient:
        return FakeClient()


@dataclass(frozen=True)
class ExampleAuthenticationSource:
    def resolve_headers(self) -> dict[str, str]:
        return {"X-Application-Auth": "resolved-at-runtime"}


class ExamplePrivateRecorder:
    def record(self, record: PrivateTraceRecord) -> None:
        # Application-owned private storage is deliberately outside this fixture.
        del record


private_recorder: PrivateTraceRecorder = ExamplePrivateRecorder()
private_capture = PrivateTraceCapture(recorder=private_recorder, include_reasoning=True)


factory: OllamaClientFactory = FakeFactory()
authentication_source: OllamaAuthenticationSource = ExampleAuthenticationSource()
sdk_factory: OllamaClientFactory = OllamaSdkClientFactory(
    host="http://127.0.0.1:11434",
    authentication_source=authentication_source,
)
options = OllamaRuntimeOptions(
    temperature=0.2,
    seed=7,
    max_output_tokens=128,
    think=False,
)
contract = AgentContract[str, dict[str, str]](
    agent_id="ollama-runtime",
    version="1",
    purpose="Return one structured result",
    input_schema="example.input.v1",
    output_schema="example.output.v1",
    prompt_version="prompt-1",
    effect_profile=EffectProfile.READ_ONLY,
    limits=AgentLimits(max_turns=1, max_tool_calls=0),
    stopping_policy="structured_output",
    allowed_capabilities=(OLLAMA_AGENT_RUNTIME_CAPABILITY.capability_id,),
    permissions={"workspace": "denied", "network": "allowed"},
)
runtime: AgentRuntime = OllamaAgentRuntime(
    client_factory=factory,
    model="gemma-test",
    output_schemas={"example.output.v1": {"type": "object"}},
    options=options,
)
requirements = CapabilityRequirements(
    kind=CapabilityKind.AGENT_RUNTIME,
    features=frozenset({CapabilityFeature.USAGE_REPORTING}),
)
requirements.require(OLLAMA_AGENT_RUNTIME_CAPABILITY)

text_generator: TextGenerator = OllamaTextGenerator(
    client_factory=factory,
    model="gemma-test",
    options=OllamaRuntimeOptions(temperature=0, think=False),
)
raw_schema_request = TextGenerationRequest(
    messages=(
        TextMessage(role=TextMessageRole.USER, text="Return JSON."),
        TextMessage(role=TextMessageRole.ASSISTANT, text="{broken"),
        TextMessage(role=TextMessageRole.USER, text="Repair it."),
    ),
    output_schema={"type": "object"},
    max_output_tokens=1536,
)


async def generate_raw(context: RunContext) -> str:
    return (await text_generator.generate(raw_schema_request, context)).text


session = PersistentProcessSession(ProcessSpec(
    argv=("/usr/bin/python3", "-I", "-S", "worker.py"),
    cwd="/private/tmp/agentrig-example",
))


async def use_session(context: RunContext) -> object:
    await session.start(context)
    result = await session.request({"value": 1}, context)
    await session.shutdown(context)
    return result
