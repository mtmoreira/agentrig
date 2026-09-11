"""Lima service generation fencing and retirement observation; no activation."""

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Literal, cast

from agentrig.capabilities import BackendAction, BackendFailure, BackendPlan
from agentrig.capabilities.local_backend import _digest, backend_digest, backend_identifier
from agentrig.core import DeadlineExceeded, JsonValue, RunCancelled, RunContext
from agentrig.integrations.guest_operations import _Inspection
from agentrig.integrations.guest_service import guest_service_retirement_digest
from agentrig.integrations.guest_workspace import GuestWorkspaceEndpoint, _instance
from agentrig.integrations.lima_lifecycle import LimaLifecycleBackend


@dataclass(frozen=True, slots=True, kw_only=True)
class LimaGuestServiceGeneration:
    """Reviewed identity plus a fresh authenticated generation observation.

    The generation digest is a safe opaque identity, never a credential or saved
    runtime authority. The backend fence must freshly authenticate it on every
    operation. Construct through bind_lima_guest_service_generation.
    """

    endpoint: GuestWorkspaceEndpoint = field(repr=False)
    backend_id: str
    backend_version: str
    configuration_digest: str
    instance: str
    generation_digest: str

    def __post_init__(self) -> None:
        _instance(self.endpoint.instance_id)
        if self.instance != 'managed-' + self.endpoint.instance_id:
            raise ValueError('lima service instance invalid')
        for value in (self.configuration_digest, self.generation_digest):
            _digest(value)
        backend_identifier(self.backend_id)
        backend_identifier(self.backend_version)

    @property
    def digest(self) -> str:
        return backend_digest({'schema': 'agentrig.lima-guest-service-generation.v1',
            'backend_id': self.backend_id, 'backend_version': self.backend_version,
            'configuration_digest': self.configuration_digest, 'instance': self.instance,
            'generation_digest': self.generation_digest,
            'endpoint_digest': guest_service_retirement_digest(self.endpoint)})


@dataclass(frozen=True, slots=True, kw_only=True)
class LimaQuiescenceObservation:
    binding_digest: str
    vm_state: Literal['stopped'] = field(default='stopped', init=False)
    forwarding_state: Literal['stopped'] = field(default='stopped', init=False)
    active_clients: Literal[0] = field(default=0, init=False)
    generation_fenced: Literal[True] = field(default=True, init=False)

    def __post_init__(self) -> None:
        _digest(self.binding_digest)


@dataclass(frozen=True, slots=True, kw_only=True)
class LimaRetirementObservation:
    binding_digest: str
    socket_state: Literal['absent'] = field(default='absent', init=False)
    service_lock_retained: Literal[True] = field(default=True, init=False)
    workspace_retained: Literal[True] = field(default=True, init=False)

    def __post_init__(self) -> None:
        _digest(self.binding_digest)


LimaGenerationFence = Callable[[LimaGuestServiceGeneration, RunContext],
                               AbstractContextManager[LimaQuiescenceObservation]]
LimaSocketRetirer = Callable[[GuestWorkspaceEndpoint, str, RunContext], object]
LimaRetirementObserver = Callable[[LimaGuestServiceGeneration, RunContext],
                                  LimaRetirementObservation]


def bind_lima_guest_service_generation(backend: LimaLifecycleBackend, plan: BackendPlan,
                                       endpoint: GuestWorkspaceEndpoint,
                                       generation_digest: str) -> LimaGuestServiceGeneration:
    """Bind an inspect plan and endpoint to a fresh backend generation identity."""
    _digest(generation_digest)
    try:
        config = plan.request.configuration
        if (plan.request.action is not BackendAction.INSPECT or backend.plan(plan.request) != plan or
                config['instance_id'] != endpoint.instance_id or
                plan.material['instance'] != 'managed-' + endpoint.instance_id or
                endpoint.root != Path(cast(str, config['state_root'])) / 'transport'):
            raise BackendFailure('backend_plan_changed')
        return LimaGuestServiceGeneration(endpoint=endpoint,
            backend_id=plan.descriptor.backend_id, backend_version=plan.descriptor.version,
            configuration_digest=backend_digest(config),
            instance=plan.material['instance'], generation_digest=generation_digest)
    except BackendFailure:
        raise
    except (KeyError, TypeError, ValueError):
        raise BackendFailure('backend_plan_changed') from None


def decode_lima_service_observation(data: bytes, generation: LimaGuestServiceGeneration, *,
                                    phase: Literal['quiescent', 'retired']) -> (
                                        LimaQuiescenceObservation | LimaRetirementObservation):
    """Decode a bounded exact observation from an authenticated backend channel."""
    if not isinstance(data, bytes) or not 1 <= len(data) <= 4096 or phase not in ('quiescent', 'retired'):
        raise BackendFailure('backend_invalid_result')
    def unique(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
        value: dict[str, JsonValue] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('duplicate key')
            value[key] = item
        return value
    try:
        value = json.loads(data, object_pairs_hook=unique)
        expected: dict[str, JsonValue] = {'schema': 'agentrig.lima-service-observation.v1',
            'phase': phase, 'binding_digest': generation.digest,
            'instance': generation.instance, 'generation_digest': generation.generation_digest}
        if phase == 'quiescent':
            expected.update(vm_state='stopped', forwarding_state='stopped',
                            active_clients=0, generation_fenced=True)
        else:
            expected.update(socket_state='absent', service_lock_retained=True,
                            workspace_retained=True)
        if (type(value) is not dict or set(value) != set(expected) or
                any(type(value[key]) is not type(item) or value[key] != item
                    for key, item in expected.items())):
            raise ValueError('observation changed')
        if phase == 'quiescent':
            return LimaQuiescenceObservation(binding_digest=generation.digest)
        return LimaRetirementObservation(binding_digest=generation.digest)
    except (UnicodeError, ValueError, TypeError, RecursionError):
        raise BackendFailure('backend_invalid_result') from None


@dataclass(frozen=True, slots=True, kw_only=True)
class LimaGuestServiceRetirementAdapter:
    """Backend-specific adapter; all runtime contact stays behind injected seams.

    The fence owns authentication and restart exclusion. This class validates its
    bounded observations and exact generation binding. It is not registered by
    default and importing/constructing it performs no I/O.
    """

    generation: LimaGuestServiceGeneration
    fence_generation: LimaGenerationFence | None = field(default=None, repr=False)
    retire_socket: LimaSocketRetirer | None = field(default=None, repr=False)
    observe_retired: LimaRetirementObserver | None = field(default=None, repr=False)

    def _check(self, endpoint: GuestWorkspaceEndpoint, context: RunContext) -> _Inspection:
        check = _Inspection(context)
        check.check()
        if endpoint != self.generation.endpoint:
            raise BackendFailure('backend_plan_changed')
        return check

    @contextmanager
    def quiescent(self, endpoint: GuestWorkspaceEndpoint,
                  context: RunContext) -> Iterator[None]:
        check = self._check(endpoint, context)
        if self.fence_generation is None:
            raise BackendFailure('backend_plan_blocked')
        try:
            with self.fence_generation(self.generation, context) as observed:
                check.check()
                expected = LimaQuiescenceObservation(binding_digest=self.generation.digest)
                if type(observed) is not LimaQuiescenceObservation or observed != expected:
                    raise BackendFailure('backend_invalid_result')
                yield None
                check.check()
        except (BackendFailure, DeadlineExceeded, RunCancelled):
            raise
        except Exception:
            raise BackendFailure('backend_execution_failed') from None

    def retire(self, endpoint: GuestWorkspaceEndpoint, approved_digest: str,
               context: RunContext) -> None:
        check = self._check(endpoint, context)
        if approved_digest != guest_service_retirement_digest(endpoint):
            raise BackendFailure('backend_authority_required')
        if self.retire_socket is None:
            raise BackendFailure('backend_plan_blocked')
        try:
            if self.retire_socket(endpoint, approved_digest, context) is not None:
                raise BackendFailure('backend_invalid_result')
            check.check()
        except (BackendFailure, DeadlineExceeded, RunCancelled):
            raise
        except Exception:
            raise BackendFailure('backend_execution_failed') from None

    def inspect_retired(self, endpoint: GuestWorkspaceEndpoint,
                        context: RunContext) -> None:
        check = self._check(endpoint, context)
        if self.observe_retired is None:
            raise BackendFailure('backend_plan_blocked')
        try:
            observed = self.observe_retired(self.generation, context)
            expected = LimaRetirementObservation(binding_digest=self.generation.digest)
            if type(observed) is not LimaRetirementObservation or observed != expected:
                raise BackendFailure('backend_invalid_result')
            check.check()
        except (BackendFailure, DeadlineExceeded, RunCancelled):
            raise
        except Exception:
            raise BackendFailure('backend_execution_failed') from None
