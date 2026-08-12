from __future__ import annotations

import unittest
from dataclasses import dataclass
from datetime import UTC, datetime

from agentrig.core import (
    CancellationSource,
    Clock,
    Deadline,
    IdGenerator,
    RunContext,
    RunId,
)


@dataclass
class SequentialRunIdGenerator:
    next_value: int = 1

    def generate(self) -> RunId:
        run_id = RunId(f"run-{self.next_value}")
        self.next_value += 1
        return run_id


@dataclass(frozen=True)
class FixedClock:
    monotonic_time: float

    def now(self) -> datetime:
        return datetime(2026, 8, 12, 12, 0, tzinfo=UTC)

    def monotonic(self) -> float:
        return self.monotonic_time


def create_root(
    *,
    clock: Clock | None = None,
    id_generator: IdGenerator[RunId] | None = None,
    cancellation_source: CancellationSource | None = None,
    deadline: Deadline | None = None,
    labels: dict[str, str] | None = None,
    correlation: dict[str, str] | None = None,
) -> RunContext:
    effective_clock = clock if clock is not None else FixedClock(100.0)
    effective_generator = (
        id_generator if id_generator is not None else SequentialRunIdGenerator()
    )
    effective_source = (
        cancellation_source
        if cancellation_source is not None
        else CancellationSource()
    )
    return RunContext.create_root(
        clock=effective_clock,
        id_generator=effective_generator,
        cancellation=effective_source.token,
        deadline=deadline,
        labels=labels,
        correlation=correlation,
    )


class RunContextTest(unittest.TestCase):
    def test_root_uses_injected_dependencies_and_generated_identity(self) -> None:
        clock = FixedClock(100.0)
        generator = SequentialRunIdGenerator()
        source = CancellationSource()

        context = create_root(
            clock=clock,
            id_generator=generator,
            cancellation_source=source,
        )

        self.assertEqual(context.run_id, RunId("run-1"))
        self.assertIsNone(context.parent_run_id)
        self.assertIs(context.clock, clock)
        self.assertIs(context.id_generator, generator)
        self.assertIs(context.cancellation, source.token)

    def test_child_has_new_identity_and_parent_lineage(self) -> None:
        generator = SequentialRunIdGenerator()
        root = create_root(id_generator=generator)

        child = root.derive_child()
        grandchild = child.derive_child()

        self.assertEqual(child.run_id, RunId("run-2"))
        self.assertEqual(child.parent_run_id, root.run_id)
        self.assertEqual(grandchild.run_id, RunId("run-3"))
        self.assertEqual(grandchild.parent_run_id, child.run_id)
        self.assertIs(child.clock, root.clock)
        self.assertIs(child.id_generator, root.id_generator)
        self.assertIs(child.cancellation, root.cancellation)

    def test_child_deadline_can_narrow_but_not_extend_parent(self) -> None:
        clock = FixedClock(100.0)
        parent_deadline = Deadline.after(20.0, clock)
        root = create_root(clock=clock, deadline=parent_deadline)

        narrower = root.derive_child(timeout_seconds=5.0)
        attempted_extension = root.derive_child(
            deadline=Deadline.after(30.0, clock)
        )

        self.assertEqual(narrower.deadline, Deadline.after(5.0, clock))
        self.assertIs(attempted_extension.deadline, parent_deadline)

    def test_child_can_use_a_separately_owned_linked_token(self) -> None:
        parent_source = CancellationSource()
        root = create_root(cancellation_source=parent_source)
        child_source = parent_source.create_child()
        child = root.derive_child(cancellation=child_source.token)

        child_source.cancel("only this child stopped")

        self.assertTrue(child.cancellation.is_cancelled)
        self.assertFalse(root.cancellation.is_cancelled)

    def test_metadata_is_copied_frozen_inherited_and_overlaid(self) -> None:
        labels = {"environment": "test"}
        root = create_root(
            labels=labels,
            correlation={"request_id": "request-1"},
        )
        labels["environment"] = "mutated"

        child = root.derive_child(
            labels={"environment": "child", "component": "writer"},
            correlation={"step_id": "step-1"},
        )

        self.assertEqual(root.labels, {"environment": "test"})
        self.assertEqual(
            child.labels,
            {"environment": "child", "component": "writer"},
        )
        self.assertEqual(
            child.correlation,
            {"request_id": "request-1", "step_id": "step-1"},
        )
        with self.assertRaises(TypeError):
            child.labels["new"] = "value"  # type: ignore[index]

    def test_metadata_requires_trimmed_nonempty_strings(self) -> None:
        invalid_maps = (
            {"": "value"},
            {" padded": "value"},
            {"key": ""},
            {"key": "value "},
        )

        for values in invalid_maps:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    create_root(labels=values)


if __name__ == "__main__":
    unittest.main()
