"""Positive fixture for the portable OpenAI Responses boundary."""

from __future__ import annotations

from dataclasses import dataclass

from agentrig.capabilities import StructuredGenerator
from agentrig.core import ArtifactRef, ArtifactResolver, ResolvedArtifact
from agentrig.integrations.openai import (
    OpenAIResponsesAuthenticationSource,
    OpenAIResponsesClient,
    OpenAIResponsesClientFactory,
    OpenAIResponsesStructuredGenerator,
)


@dataclass(frozen=True)
class Authentication:
    def resolve_api_key(self) -> str:
        return "private"


@dataclass(frozen=True)
class Resolver:
    async def resolve(self, artifact: ArtifactRef) -> ResolvedArtifact:
        return ResolvedArtifact(artifact=artifact, content=b"image")


class Client:
    async def create(self, request: object) -> object:
        raise NotImplementedError

    async def close(self) -> None:
        return None


@dataclass(frozen=True)
class Factory:
    client: OpenAIResponsesClient

    def create(self) -> OpenAIResponsesClient:
        return self.client


authentication: OpenAIResponsesAuthenticationSource = Authentication()
resolver: ArtifactResolver = Resolver()
factory: OpenAIResponsesClientFactory
generator: StructuredGenerator[str] = OpenAIResponsesStructuredGenerator(
    client_factory=factory,
    artifact_resolver=resolver,
    model="vision-model",
)
