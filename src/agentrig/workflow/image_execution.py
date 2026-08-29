"""Bounded, provider-neutral execution for one explicitly selected image route."""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Iterable

from agentrig.capabilities import (
    CapabilityFeature,
    ImageGenerationRequest,
    ImageGenerationResult,
    ImageGenerator,
)
from agentrig.core._validation import require_trimmed_string
from agentrig.core.context import RunContext
from agentrig.core.errors import Failure, FailureKind, normalize_exception
from agentrig.core.events import Event, EventKind
from agentrig.core.outcomes import ExecutionOutcome


@dataclass(frozen=True, slots=True, kw_only=True)
class ImageRoute:
    """Stable route identity bound to exactly one generator instance."""

    route_id: str
    generator: ImageGenerator

    def __post_init__(self) -> None:
        require_trimmed_string("image route ID", self.route_id)
        if not isinstance(self.generator, ImageGenerator):
            raise TypeError("image route generator must satisfy ImageGenerator")


@dataclass(frozen=True, slots=True, kw_only=True)
class ImageExecutionPolicy:
    """Finite attempts, retry classes, concurrency, and optional cost ceiling."""

    max_attempts: int = 1
    retryable_failure_kinds: frozenset[FailureKind] = frozenset(
        {FailureKind.TRANSIENT_PROVIDER}
    )
    max_concurrency: int = 1
    max_total_cost: float | None = None
    currency: str | None = None

    def __post_init__(self) -> None:
        _positive_integer("image max attempts", self.max_attempts)
        _positive_integer("image max concurrency", self.max_concurrency)
        kinds = tuple(self.retryable_failure_kinds)
        if any(not isinstance(item, FailureKind) for item in kinds):
            raise TypeError("image retry kinds must contain FailureKind values")
        if len(kinds) != len(set(kinds)):
            raise ValueError("image retry kinds must not contain duplicates")
        object.__setattr__(self, "retryable_failure_kinds", frozenset(kinds))
        if any(item is not FailureKind.TRANSIENT_PROVIDER for item in kinds):
            raise ValueError(
                "image retries may declare only transient provider failures"
            )
        if self.max_total_cost is not None:
            if (
                isinstance(self.max_total_cost, bool)
                or not isinstance(self.max_total_cost, (int, float))
                or not math.isfinite(self.max_total_cost)
                or self.max_total_cost < 0
            ):
                raise ValueError("image cost ceiling must be finite and non-negative")
        if (self.max_total_cost is None) != (self.currency is None):
            raise ValueError(
                "image cost ceiling and currency must be configured together"
            )
        if self.currency is not None:
            require_trimmed_string("image budget currency", self.currency)
            if len(self.currency) != 3 or self.currency.upper() != self.currency:
                raise ValueError(
                    "image budget currency must be three uppercase letters"
                )


@dataclass(frozen=True, slots=True, kw_only=True)
class ImageAttemptEvidence:
    """Sanitized evidence for one selected-route invocation."""

    route_id: str
    attempt: int
    capability_id: str
    capability_version: str
    failure_kind: FailureKind | None = None
    failure_code: str | None = None

    def __post_init__(self) -> None:
        require_trimmed_string("image attempt route ID", self.route_id)
        _positive_integer("image attempt number", self.attempt)
        require_trimmed_string("image attempt capability ID", self.capability_id)
        require_trimmed_string(
            "image attempt capability version", self.capability_version
        )
        if self.failure_kind is not None and not isinstance(
            self.failure_kind, FailureKind
        ):
            raise TypeError("image attempt failure kind must be FailureKind or None")
        if self.failure_code is not None:
            require_trimmed_string("image attempt failure code", self.failure_code)
        if self.failure_kind is None and self.failure_code is not None:
            raise ValueError("successful image attempt cannot have a failure code")


@dataclass(frozen=True, slots=True, kw_only=True)
class ImageExecution:
    """Terminal outcome plus complete bounded-attempt evidence."""

    route_id: str
    outcome: ExecutionOutcome[ImageGenerationResult]
    attempts: tuple[ImageAttemptEvidence, ...]

    def __post_init__(self) -> None:
        require_trimmed_string("image execution route ID", self.route_id)
        if not isinstance(self.outcome, ExecutionOutcome):
            raise TypeError("image execution outcome must be ExecutionOutcome")
        attempts = tuple(self.attempts)
        if any(not isinstance(item, ImageAttemptEvidence) for item in attempts):
            raise TypeError("image execution attempts must contain evidence values")
        if tuple(item.attempt for item in attempts) != tuple(
            range(1, len(attempts) + 1)
        ):
            raise ValueError("image execution attempts must be contiguous")
        if any(item.route_id != self.route_id for item in attempts):
            raise ValueError("image execution attempts must use the selected route")
        object.__setattr__(self, "attempts", attempts)


class ImageGenerationExecutor:
    """Execute only a caller-selected route under one shared concurrency gate."""

    def __init__(
        self,
        *,
        routes: Iterable[ImageRoute],
        policy: ImageExecutionPolicy,
    ) -> None:
        copied_routes = tuple(routes)
        if not copied_routes:
            raise ValueError("image executor requires at least one route")
        if any(not isinstance(item, ImageRoute) for item in copied_routes):
            raise TypeError("image executor routes must contain ImageRoute values")
        route_ids = tuple(item.route_id for item in copied_routes)
        if len(route_ids) != len(set(route_ids)):
            raise ValueError("image executor route IDs must be unique")
        if not isinstance(policy, ImageExecutionPolicy):
            raise TypeError("image executor policy must be ImageExecutionPolicy")
        self._routes = {item.route_id: item for item in copied_routes}
        self._policy = policy
        self._semaphore = asyncio.Semaphore(policy.max_concurrency)

    async def execute(
        self,
        *,
        route_id: str,
        request: ImageGenerationRequest,
        context: RunContext,
    ) -> ImageExecution:
        """Invoke and retry only the exact selected route; never fall through."""
        require_trimmed_string("selected image route ID", route_id)
        if not isinstance(request, ImageGenerationRequest):
            raise TypeError("image executor request must be ImageGenerationRequest")
        if not isinstance(context, RunContext):
            raise TypeError("image executor context must be RunContext")
        route = self._routes.get(route_id)
        if route is None:
            return _configuration_failure(route_id, "image.route_not_configured")
        if self._policy.max_attempts > 1 and request.idempotency_key is None:
            failure = Failure(
                kind=FailureKind.INVALID_INPUT,
                message="image retries require an idempotency key",
                code="image.retry_idempotency_required",
            )
            return ImageExecution(
                route_id=route.route_id,
                outcome=ExecutionOutcome.from_failure(failure),
                attempts=(),
            )
        try:
            _check_constraints(context)
            request.require_supported_by(route.generator.descriptor)
        except (asyncio.CancelledError, Exception) as error:
            return _preflight_failure(route, error)
        if (
            self._policy.max_total_cost is not None
            and CapabilityFeature.COST_REPORTING
            not in route.generator.descriptor.features
        ):
            return _failure_execution(
                route,
                Failure(
                    kind=FailureKind.BUDGET_EXHAUSTED,
                    message="image route cannot prove usage under its cost ceiling",
                    code="image.usage_required",
                ),
            )

        async with self._semaphore:
            try:
                _check_constraints(context)
            except (asyncio.CancelledError, Exception) as error:
                return _preflight_failure(route, error)
            return await self._attempt(route, request, context)

    async def _attempt(
        self,
        route: ImageRoute,
        request: ImageGenerationRequest,
        context: RunContext,
    ) -> ImageExecution:
        attempts: list[ImageAttemptEvidence] = []
        for attempt in range(1, self._policy.max_attempts + 1):
            _emit(context, EventKind.PROVIDER_CALL_STARTED, route, attempt)
            try:
                _check_constraints(context)
                result = await route.generator.generate(request, context)
                budget_failure = self._budget_failure(result)
                outcome: ExecutionOutcome[ImageGenerationResult]
                if budget_failure is not None:
                    attempts.append(_attempt_evidence(route, attempt, budget_failure))
                    outcome = ExecutionOutcome.from_failure(budget_failure)
                else:
                    attempts.append(_attempt_evidence(route, attempt, None))
                    outcome = ExecutionOutcome.succeeded(
                        result, artifacts=(result.image,)
                    )
                _emit_completed(context, route, attempt, outcome.failure)
                return ImageExecution(
                    route_id=route.route_id,
                    outcome=outcome,
                    attempts=tuple(attempts),
                )
            except (asyncio.CancelledError, Exception) as error:
                failure = normalize_exception(error)
                attempts.append(_attempt_evidence(route, attempt, failure))
                _emit_completed(context, route, attempt, failure)
                if (
                    attempt == self._policy.max_attempts
                    or failure.kind not in self._policy.retryable_failure_kinds
                ):
                    return ImageExecution(
                        route_id=route.route_id,
                        outcome=ExecutionOutcome.from_failure(failure),
                        attempts=tuple(attempts),
                    )
                _emit_retry(context, route, attempt, failure)
        raise AssertionError("validated image policy produced no attempts")

    def _budget_failure(
        self, result: ImageGenerationResult
    ) -> Failure | None:
        ceiling = self._policy.max_total_cost
        if ceiling is None:
            return None
        usage = result.usage
        if usage.cost is None or usage.currency is None:
            return Failure(
                kind=FailureKind.BUDGET_EXHAUSTED,
                message="image result did not report required cost usage",
                code="image.usage_unknown",
            )
        if usage.currency != self._policy.currency:
            return Failure(
                kind=FailureKind.BUDGET_EXHAUSTED,
                message="image usage currency differs from its cost ceiling",
                code="image.usage_currency_mismatch",
            )
        if usage.cost > ceiling:
            return Failure(
                kind=FailureKind.BUDGET_EXHAUSTED,
                message="image result exceeded its cost ceiling",
                code="image.cost_exhausted",
            )
        return None


def _check_constraints(context: RunContext) -> None:
    context.cancellation.raise_if_cancelled()
    if context.deadline is not None:
        context.deadline.raise_if_expired(context.clock)


def _positive_integer(label: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")


def _attempt_evidence(
    route: ImageRoute, attempt: int, failure: Failure | None
) -> ImageAttemptEvidence:
    descriptor = route.generator.descriptor
    return ImageAttemptEvidence(
        route_id=route.route_id,
        attempt=attempt,
        capability_id=descriptor.capability_id,
        capability_version=descriptor.version,
        failure_kind=failure.kind if failure is not None else None,
        failure_code=failure.code if failure is not None else None,
    )


def _preflight_failure(route: ImageRoute, error: BaseException) -> ImageExecution:
    return ImageExecution(
        route_id=route.route_id,
        outcome=ExecutionOutcome.from_failure(normalize_exception(error)),
        attempts=(),
    )


def _failure_execution(route: ImageRoute, failure: Failure) -> ImageExecution:
    return ImageExecution(
        route_id=route.route_id,
        outcome=ExecutionOutcome.from_failure(failure),
        attempts=(),
    )


def _configuration_failure(route_id: str, code: str) -> ImageExecution:
    failure = Failure(
        kind=FailureKind.INVALID_INPUT,
        message="selected image route is not configured",
        code=code,
    )
    return ImageExecution(
        route_id=route_id,
        outcome=ExecutionOutcome.from_failure(failure),
        attempts=(),
    )


def _emit(
    context: RunContext, kind: EventKind, route: ImageRoute, attempt: int
) -> None:
    context.event_sink.emit(
        Event.from_context(
            event_id=context.event_id_generator.generate(),
            kind=kind,
            context=context,
            attributes={"route_id": route.route_id, "attempt": attempt},
        )
    )


def _emit_completed(
    context: RunContext,
    route: ImageRoute,
    attempt: int,
    failure: Failure | None,
) -> None:
    attributes: dict[str, str | int] = {
        "route_id": route.route_id,
        "attempt": attempt,
        "status": "succeeded" if failure is None else "failed",
    }
    if failure is not None:
        attributes["failure_kind"] = failure.kind.value
        if failure.code is not None:
            attributes["failure_code"] = failure.code
    context.event_sink.emit(
        Event.from_context(
            event_id=context.event_id_generator.generate(),
            kind=EventKind.PROVIDER_CALL_COMPLETED,
            context=context,
            attributes=attributes,
        )
    )


def _emit_retry(
    context: RunContext,
    route: ImageRoute,
    attempt: int,
    failure: Failure,
) -> None:
    context.event_sink.emit(
        Event.from_context(
            event_id=context.event_id_generator.generate(),
            kind=EventKind.RETRY_SCHEDULED,
            context=context,
            attributes={
                "route_id": route.route_id,
                "attempt": attempt,
                "next_attempt": attempt + 1,
                "failure_kind": failure.kind.value,
            },
        )
    )
