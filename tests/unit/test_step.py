from __future__ import annotations

import asyncio
import unittest
from dataclasses import dataclass
from datetime import UTC, datetime

from agentrig.core import CancellationSource, RunContext, RunId
from agentrig.workflow import EffectProfile, Step, StepDescriptor


@dataclass(frozen=True)
class FixedClock:
    def now(self) -> datetime:
        return datetime(2026, 8, 13, 8, 30, tzinfo=UTC)

    def monotonic(self) -> float:
        return 100.0


@dataclass
class SequentialRunIdGenerator:
    next_value: int = 1

    def generate(self) -> RunId:
        run_id = RunId(f"run-{self.next_value}")
        self.next_value += 1
        return run_id


def create_context() -> RunContext:
    source = CancellationSource()
    return RunContext.create_root(
        clock=FixedClock(),
        id_generator=SequentialRunIdGenerator(),
        cancellation=source.token,
    )


@dataclass(frozen=True)
class TextLengthStep:
    descriptor: StepDescriptor = StepDescriptor(
        step_id="text.length",
        version="1",
        effect_profile=EffectProfile.READ_ONLY,
    )

    async def run(self, input: str, context: RunContext) -> int:
        context.cancellation.raise_if_cancelled()
        return len(input)


async def run_typed_step(
    step: Step[str, int],
    input: str,
    context: RunContext,
) -> int:
    return await step.run(input, context)


class EffectProfileTest(unittest.TestCase):
    def test_vocabulary_has_stable_wire_values(self) -> None:
        self.assertEqual(
            {profile.value for profile in EffectProfile},
            {
                "compensatable",
                "idempotent",
                "non_repeatable",
                "read_only",
            },
        )

    def test_only_intrinsically_repeatable_profiles_allow_automatic_retry(
        self,
    ) -> None:
        self.assertTrue(EffectProfile.READ_ONLY.allows_automatic_retry)
        self.assertTrue(EffectProfile.IDEMPOTENT.allows_automatic_retry)
        self.assertFalse(EffectProfile.COMPENSATABLE.allows_automatic_retry)
        self.assertFalse(EffectProfile.NON_REPEATABLE.allows_automatic_retry)


class StepDescriptorTest(unittest.TestCase):
    def test_preserves_stable_identity_and_effect_profile(self) -> None:
        descriptor = StepDescriptor(
            step_id="story.compile",
            version="2",
            effect_profile=EffectProfile.IDEMPOTENT,
        )

        self.assertEqual(descriptor.step_id, "story.compile")
        self.assertEqual(descriptor.version, "2")
        self.assertEqual(descriptor.effect_profile, EffectProfile.IDEMPOTENT)

    def test_rejects_invalid_identity_or_effect_profile(self) -> None:
        with self.assertRaises(ValueError):
            StepDescriptor(
                step_id=" padded",
                version="1",
                effect_profile=EffectProfile.READ_ONLY,
            )
        with self.assertRaises(ValueError):
            StepDescriptor(
                step_id="step",
                version="",
                effect_profile=EffectProfile.READ_ONLY,
            )
        with self.assertRaises(TypeError):
            StepDescriptor(
                step_id="step",
                version="1",
                effect_profile="read_only",  # type: ignore[arg-type]
            )


class StepContractTest(unittest.TestCase):
    def test_protocol_supports_a_typed_async_step(self) -> None:
        step = TextLengthStep()

        output = asyncio.run(run_typed_step(step, "draft", create_context()))

        self.assertIsInstance(step, Step)
        self.assertEqual(output, 5)
        self.assertEqual(
            step.descriptor.effect_profile,
            EffectProfile.READ_ONLY,
        )


if __name__ == "__main__":
    unittest.main()
