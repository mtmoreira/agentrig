"""Provider-independent coding-agent task and evidence contracts."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Protocol, runtime_checkable

from agentrig.capabilities.base import (
    CapabilityDescriptor,
    CapabilityKind,
    CapabilityLimit,
    CapabilityRequirements,
)
from agentrig.core._validation import require_trimmed_string
from agentrig.core.artifacts import ArtifactRef
from agentrig.core.context import RunContext
from agentrig.core.errors import Failure, FailureKind

_BLOCKING_FAILURE_KINDS = frozenset(
    {
        FailureKind.APPROVAL_REQUIRED,
        FailureKind.WORKFLOW_BLOCKED,
    }
)


class CodingChangeKind(StrEnum):
    """Portable changed-file classifications."""

    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"


class CodingValidationStatus(StrEnum):
    """Stable outcomes for one reported validation operation."""

    PASSED = "passed"
    FAILED = "failed"
    NOT_RUN = "not_run"


class CodingStatus(StrEnum):
    """Terminal coding outcomes distinct from technical execution failures."""

    SUCCEEDED = "succeeded"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkspaceAuthorization:
    """One canonical workspace and its explicitly writable relative roots."""

    workspace_id: str
    root_path: str = field(repr=False)
    writable_roots: tuple[str, ...] = field(repr=False)

    def __post_init__(self) -> None:
        require_trimmed_string("workspace authorization ID", self.workspace_id)
        _require_workspace_root(self.root_path)
        copied_roots = tuple(self.writable_roots)
        if not copied_roots:
            raise ValueError(
                "workspace authorization requires at least one writable root"
            )
        for root in copied_roots:
            _require_relative_path(
                "workspace writable root",
                root,
                allow_workspace_root=True,
            )
        if len(copied_roots) != len(set(copied_roots)):
            raise ValueError("workspace writable roots must not contain duplicates")
        if "." in copied_roots and len(copied_roots) != 1:
            raise ValueError(
                "workspace root authorization must not include redundant roots"
            )
        object.__setattr__(self, "writable_roots", copied_roots)

    def permits(self, path: str) -> bool:
        """Whether one canonical workspace-relative file is writable."""
        _require_relative_path(
            "workspace file path",
            path,
            allow_workspace_root=False,
        )
        candidate = PurePosixPath(path)
        for root in self.writable_roots:
            if root == ".":
                return True
            scope = PurePosixPath(root)
            if candidate == scope or scope in candidate.parents:
                return True
        return False


@dataclass(frozen=True, slots=True, kw_only=True)
class CodingTask:
    """Bounded objective and acceptance contract for one authorized workspace."""

    task_id: str
    workspace: WorkspaceAuthorization
    objective: str = field(repr=False)
    acceptance_criteria: tuple[str, ...] = field(repr=False)
    max_changed_files: int
    requirements: CapabilityRequirements = field(
        default_factory=lambda: CapabilityRequirements(
            kind=CapabilityKind.CODING,
        )
    )

    def __post_init__(self) -> None:
        require_trimmed_string("coding task ID", self.task_id)
        if not isinstance(self.workspace, WorkspaceAuthorization):
            raise TypeError(
                "coding task workspace must be a WorkspaceAuthorization"
            )
        _require_content_text("coding task objective", self.objective)
        copied_criteria = tuple(self.acceptance_criteria)
        if not copied_criteria:
            raise ValueError(
                "coding task requires at least one acceptance criterion"
            )
        for criterion in copied_criteria:
            _require_content_text("coding acceptance criterion", criterion)
        if len(copied_criteria) != len(set(copied_criteria)):
            raise ValueError(
                "coding acceptance criteria must not contain duplicates"
            )
        object.__setattr__(self, "acceptance_criteria", copied_criteria)
        if (
            isinstance(self.max_changed_files, bool)
            or not isinstance(self.max_changed_files, int)
            or self.max_changed_files <= 0
        ):
            raise ValueError(
                "coding task max_changed_files must be a positive integer"
            )
        if not isinstance(self.requirements, CapabilityRequirements):
            raise TypeError(
                "coding task requirements must be CapabilityRequirements"
            )
        if self.requirements.kind is not CapabilityKind.CODING:
            raise ValueError("coding task requirements must use the coding kind")
        if CapabilityLimit.MAX_CHANGED_FILES in self.requirements.minimum_limits:
            raise ValueError(
                "coding task changed-file capacity is derived from "
                "max_changed_files"
            )

    @property
    def capability_requirements(self) -> CapabilityRequirements:
        """Combine optional features with this task's changed-file capacity."""
        return CapabilityRequirements(
            kind=CapabilityKind.CODING,
            features=self.requirements.features,
            minimum_limits={
                **self.requirements.minimum_limits,
                CapabilityLimit.MAX_CHANGED_FILES: self.max_changed_files,
            },
            allowed_data_retention=self.requirements.allowed_data_retention,
        )

    def require_supported_by(self, descriptor: CapabilityDescriptor) -> None:
        """Fail before agent execution if its portable support is insufficient."""
        self.capability_requirements.require(descriptor)


@dataclass(frozen=True, slots=True, kw_only=True)
class ChangedFileEvidence:
    """One authorized changed path and optional durable evidence artifact."""

    path: str
    change_kind: CodingChangeKind
    evidence_artifact: ArtifactRef | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        _require_relative_path(
            "changed file path",
            self.path,
            allow_workspace_root=False,
        )
        if not isinstance(self.change_kind, CodingChangeKind):
            raise TypeError("changed file kind must be a CodingChangeKind")
        if self.evidence_artifact is not None and not isinstance(
            self.evidence_artifact,
            ArtifactRef,
        ):
            raise TypeError(
                "changed file evidence_artifact must be an ArtifactRef or None"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class ValidationEvidence:
    """Sanitized outcome of one validation without retaining raw command output."""

    validation_id: str
    status: CodingValidationStatus
    summary: str = field(repr=False)
    exit_code: int | None = None
    output_artifact: ArtifactRef | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        require_trimmed_string("coding validation ID", self.validation_id)
        if not isinstance(self.status, CodingValidationStatus):
            raise TypeError(
                "coding validation status must be a CodingValidationStatus"
            )
        _require_content_text("coding validation summary", self.summary)
        if self.exit_code is not None and (
            isinstance(self.exit_code, bool)
            or not isinstance(self.exit_code, int)
            or self.exit_code < 0
        ):
            raise ValueError(
                "coding validation exit_code must be a non-negative integer"
            )
        if self.status is CodingValidationStatus.NOT_RUN:
            if self.exit_code is not None:
                raise ValueError("not-run validation must not have an exit code")
        elif self.status is CodingValidationStatus.PASSED:
            if self.exit_code not in (None, 0):
                raise ValueError(
                    "passed validation must not have a nonzero exit code"
                )
        elif self.exit_code == 0:
            raise ValueError("failed validation must not have a zero exit code")
        if self.output_artifact is not None and not isinstance(
            self.output_artifact,
            ArtifactRef,
        ):
            raise TypeError(
                "validation output_artifact must be an ArtifactRef or None"
            )


@dataclass(frozen=True, slots=True, init=False)
class CodingResult:
    """Authorized coding evidence in a succeeded or explicit blocked state."""

    task_id: str
    workspace_id: str
    status: CodingStatus
    changed_files: tuple[ChangedFileEvidence, ...]
    validations: tuple[ValidationEvidence, ...]
    blocker: Failure | None = field(default=None, repr=False)

    def __init__(
        self,
        *,
        task: CodingTask,
        status: CodingStatus,
        changed_files: Iterable[ChangedFileEvidence] = (),
        validations: Iterable[ValidationEvidence] = (),
        blocker: Failure | None = None,
    ) -> None:
        if not isinstance(task, CodingTask):
            raise TypeError("coding result task must be a CodingTask")
        if not isinstance(status, CodingStatus):
            raise TypeError("coding result status must be a CodingStatus")
        copied_changes = tuple(changed_files)
        for change in copied_changes:
            if not isinstance(change, ChangedFileEvidence):
                raise TypeError(
                    "coding result changes must contain ChangedFileEvidence"
                )
            if not task.workspace.permits(change.path):
                raise ValueError(
                    "coding result contains a change outside its authorization"
                )
        paths = tuple(change.path for change in copied_changes)
        if len(paths) != len(set(paths)):
            raise ValueError("coding result changed paths must be unique")

        copied_validations = tuple(validations)
        if any(
            not isinstance(item, ValidationEvidence)
            for item in copied_validations
        ):
            raise TypeError(
                "coding result validations must contain ValidationEvidence"
            )
        validation_ids = tuple(
            item.validation_id for item in copied_validations
        )
        if len(validation_ids) != len(set(validation_ids)):
            raise ValueError("coding result validation IDs must be unique")

        if status is CodingStatus.SUCCEEDED:
            if blocker is not None:
                raise ValueError("successful coding result must not have a blocker")
            if not copied_validations:
                raise ValueError(
                    "successful coding result requires validation evidence"
                )
            if any(
                item.status is not CodingValidationStatus.PASSED
                for item in copied_validations
            ):
                raise ValueError(
                    "successful coding result requires all validations to pass"
                )
        else:
            if not isinstance(blocker, Failure):
                raise ValueError("blocked coding result requires a blocker")
            if blocker.kind not in _BLOCKING_FAILURE_KINDS:
                raise ValueError(
                    "coding result blocker must be a blocking failure kind"
                )

        if len(copied_changes) > task.max_changed_files:
            raise ValueError(
                "coding result exceeds the task's changed-file bound"
            )

        object.__setattr__(self, "task_id", task.task_id)
        object.__setattr__(self, "workspace_id", task.workspace.workspace_id)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "changed_files", copied_changes)
        object.__setattr__(self, "validations", copied_validations)
        object.__setattr__(self, "blocker", blocker)

    @classmethod
    def succeeded(
        cls,
        *,
        task: CodingTask,
        changed_files: Iterable[ChangedFileEvidence] = (),
        validations: Iterable[ValidationEvidence],
    ) -> CodingResult:
        """Create a completed result with wholly passing validation evidence."""
        return cls(
            task=task,
            status=CodingStatus.SUCCEEDED,
            changed_files=changed_files,
            validations=validations,
        )

    @classmethod
    def blocked(
        cls,
        *,
        task: CodingTask,
        blocker: Failure,
        changed_files: Iterable[ChangedFileEvidence] = (),
        validations: Iterable[ValidationEvidence] = (),
    ) -> CodingResult:
        """Create an explicit blocked result while retaining partial evidence."""
        return cls(
            task=task,
            status=CodingStatus.BLOCKED,
            changed_files=changed_files,
            validations=validations,
            blocker=blocker,
        )

    @property
    def is_success(self) -> bool:
        return self.status is CodingStatus.SUCCEEDED


@runtime_checkable
class CodingAgent(Protocol):
    """Execute bounded coding tasks inside caller-authorized workspaces."""

    @property
    def descriptor(self) -> CapabilityDescriptor:
        """Return stable identity and supported optional features."""
        ...

    async def execute(
        self,
        task: CodingTask,
        context: RunContext,
    ) -> CodingResult:
        """Return completed evidence or an explicit blocked result."""
        ...


def _require_workspace_root(root_path: str) -> None:
    require_trimmed_string("workspace root path", root_path)
    if "\\" in root_path or "\x00" in root_path:
        raise ValueError("workspace root path must use safe POSIX syntax")
    parsed = PurePosixPath(root_path)
    if (
        not parsed.is_absolute()
        or root_path == "/"
        or root_path.startswith("//")
        or ".." in parsed.parts
        or parsed.as_posix() != root_path
    ):
        raise ValueError(
            "workspace root path must be canonical, absolute, and bounded"
        )


def _require_relative_path(
    field_name: str,
    path: str,
    *,
    allow_workspace_root: bool,
) -> None:
    require_trimmed_string(field_name, path)
    if "\\" in path or "\x00" in path:
        raise ValueError(f"{field_name} must use safe POSIX syntax")
    parsed = PurePosixPath(path)
    if parsed.is_absolute() or ".." in parsed.parts:
        raise ValueError(f"{field_name} must remain inside the workspace")
    if parsed.as_posix() != path:
        raise ValueError(f"{field_name} must be canonical")
    if path == "." and not allow_workspace_root:
        raise ValueError(f"{field_name} must identify a file or directory")


def _require_content_text(field_name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must contain non-whitespace text")
    return value
