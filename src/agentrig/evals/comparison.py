"""Portable immutable comparison matrices and private, manifest-bound rankings.

Applications execute their EvalRunner targets and export review files. This
module retains only explicit relative references, never opens review content,
and never invokes or enables a provider. Serialization is an explicit private
retention boundary, not an event/log representation.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import TypeVar

from agentrig.core._json import JsonValue
from agentrig.core.grading import GradeClassification, GradeStatus
from agentrig.evals.case import EvalCase
from agentrig.evals.runner import EvalCaseResult
from agentrig.selection import ModelSet, PreferredModelSet
from agentrig.selection._data import digest, fields, object_data, sequence, string, version

COMPARISON_SCHEMA_VERSION = 1
RANKING_SCHEMA_VERSION = 1
RANKING_SCORING_VERSION = 1
T = TypeVar("T")


def _unique(values: tuple[str, ...]) -> tuple[str, ...]:
    copied = tuple(values)
    for value in copied:
        string(value)
    if len(set(copied)) != len(copied):
        raise ValueError("record identifiers must be unique")
    return copied


@dataclass(frozen=True, slots=True, kw_only=True)
class ReviewAsset:
    """One explicitly retained review file under the application export root."""

    asset_id: str
    kind: str
    label: str = field(repr=False)
    path: str = field(repr=False)
    sha256: str = field(repr=False)
    media_type: str

    def __post_init__(self) -> None:
        string(self.asset_id)
        string(self.label)
        string(self.media_type)
        digest(self.sha256)
        string(self.path)
        path = PurePosixPath(self.path)
        if (path.is_absolute() or path.as_posix() != self.path or
                any(part in (".", "..") for part in path.parts) or
                "\\" in self.path or ":" in self.path or "\x00" in self.path):
            raise ValueError("review asset path must be a normalized relative POSIX path")
        if self.path == ".":
            raise ValueError("review asset path must name a file")
        if self.kind not in ("text", "image", "artifact"):
            raise ValueError("review asset kind is unsupported")

    def to_data(self) -> dict[str, JsonValue]:
        return {"asset_id": self.asset_id, "kind": self.kind, "label": self.label,
                "path": self.path, "sha256": self.sha256, "media_type": self.media_type}

    @classmethod
    def from_data(cls, data: Mapping[str, object]) -> ReviewAsset:
        fields(data, "asset_id kind label path sha256 media_type")
        return cls(asset_id=string(data["asset_id"]), kind=string(data["kind"]),
                   label=string(data["label"]), path=string(data["path"]),
                   sha256=digest(data["sha256"]), media_type=string(data["media_type"]))


@dataclass(frozen=True, slots=True, kw_only=True)
class ReviewProjection:
    """An ordered collection: inputs, photos, prompts, pages, or other artifacts."""

    assets: tuple[ReviewAsset, ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        copied = tuple(self.assets)
        if any(not isinstance(item, ReviewAsset) for item in copied):
            raise TypeError("review projection requires ReviewAsset values")
        _unique(tuple(item.asset_id for item in copied))
        object.__setattr__(self, "assets", copied)

    def to_data(self) -> dict[str, JsonValue]:
        return {"assets": [asset.to_data() for asset in self.assets]}

    @classmethod
    def from_data(cls, data: Mapping[str, object]) -> ReviewProjection:
        fields(data, "assets")
        return cls(assets=tuple(ReviewAsset.from_data(object_data(item)) for item in sequence(data["assets"])))


@dataclass(frozen=True, slots=True, kw_only=True)
class ComparisonCost:
    """Unknown cost is missing evidence, never a free or observed run."""

    status: str = "unknown"
    amount: float | None = None
    currency: str | None = None

    def __post_init__(self) -> None:
        if self.status not in ("unknown", "estimated", "observed"):
            raise ValueError("comparison cost status is unsupported")
        if self.status == "unknown":
            if self.amount is not None or self.currency is not None:
                raise ValueError("unknown cost cannot specify an amount or currency")
            return
        if (isinstance(self.amount, bool) or not isinstance(self.amount, (int, float))
                or not math.isfinite(self.amount) or self.amount < 0):
            raise ValueError("known cost must have a finite non-negative amount")
        if (not isinstance(self.currency, str) or len(self.currency) != 3
                or not self.currency.isascii() or not self.currency.isalpha()
                or not self.currency.isupper()):
            raise ValueError("known cost must have an uppercase three-letter currency")
        object.__setattr__(self, "amount", float(self.amount))

    def to_data(self) -> dict[str, JsonValue]:
        return {"status": self.status, "amount": self.amount, "currency": self.currency}

    @classmethod
    def from_data(cls, data: Mapping[str, object]) -> ComparisonCost:
        fields(data, "status amount currency")
        amount = data["amount"]
        currency = data["currency"]
        if amount is not None and (isinstance(amount, bool) or not isinstance(amount, (int, float))):
            raise ValueError("comparison cost amount is invalid")
        if currency is not None and not isinstance(currency, str):
            raise ValueError("comparison cost currency is invalid")
        return cls(status=string(data["status"]), amount=amount, currency=currency)


@dataclass(frozen=True, slots=True, kw_only=True)
class ComparisonCandidate:
    """One complete application-scoped configuration, evaluated as a set."""

    candidate_id: str
    version: str
    model_set: ModelSet
    label: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        string(self.candidate_id)
        string(self.version)
        if not isinstance(self.model_set, ModelSet):
            raise TypeError("comparison candidate requires a ModelSet")
        if not isinstance(self.label, str):
            raise TypeError("comparison candidate label must be a string")

    def to_data(self) -> dict[str, JsonValue]:
        return {"candidate_id": self.candidate_id, "version": self.version,
                "model_set": self.model_set.to_data(), "label": self.label}

    @classmethod
    def from_data(cls, data: Mapping[str, object]) -> ComparisonCandidate:
        fields(data, "candidate_id version model_set label")
        label = data["label"]
        if not isinstance(label, str):
            raise ValueError("comparison label must be a string")
        return cls(candidate_id=string(data["candidate_id"]), version=string(data["version"]),
                   model_set=ModelSet.from_data(object_data(data["model_set"])), label=label)


@dataclass(frozen=True, slots=True, kw_only=True)
class ComparisonCase:
    """Versioned case identity plus original inputs selected for private review."""

    case_id: str
    version: str
    review: ReviewProjection = field(default_factory=ReviewProjection, repr=False)

    def __post_init__(self) -> None:
        string(self.case_id)
        string(self.version)
        if not isinstance(self.review, ReviewProjection):
            raise TypeError("comparison case review requires ReviewProjection")

    @classmethod
    def from_eval_case(cls, case: EvalCase[T], *, review: ReviewProjection) -> ComparisonCase:
        return cls(case_id=case.case_id, version=case.version, review=review)

    def to_data(self) -> dict[str, JsonValue]:
        return {"case_id": self.case_id, "version": self.version, "review": self.review.to_data()}

    @classmethod
    def from_data(cls, data: Mapping[str, object]) -> ComparisonCase:
        fields(data, "case_id version review")
        return cls(case_id=string(data["case_id"]), version=string(data["version"]),
                   review=ReviewProjection.from_data(object_data(data["review"])))


@dataclass(frozen=True, slots=True, kw_only=True)
class ComparisonCell:
    """One candidate's outcome on one case, without raw failure details."""

    case_id: str
    candidate_id: str
    status: str
    review: ReviewProjection = field(default_factory=ReviewProjection, repr=False)
    cost: ComparisonCost = field(default_factory=ComparisonCost)
    run_id: str | None = None

    def __post_init__(self) -> None:
        string(self.case_id)
        string(self.candidate_id)
        if self.status not in ("succeeded", "failed", "pending"):
            raise ValueError("comparison cell status is unsupported")
        if self.run_id is not None:
            string(self.run_id)
        if not isinstance(self.review, ReviewProjection) or not isinstance(self.cost, ComparisonCost):
            raise TypeError("comparison cell review or cost has invalid type")

    @classmethod
    def from_eval_result(
        cls, result: EvalCaseResult[T], *, candidate_id: str,
        review: ReviewProjection, cost: ComparisonCost | None = None,
    ) -> ComparisonCell:
        eligible = (
            result.outcome.is_success
            and not result.grader_failures
            and not any(grade.classification is GradeClassification.HARD and
                        grade.status is GradeStatus.FAILURE for grade in result.grades)
        )
        return cls(case_id=result.case_id, candidate_id=candidate_id,
                   status="succeeded" if eligible else "failed",
                   review=review, cost=cost or ComparisonCost(), run_id=result.run_id.value)

    def to_data(self) -> dict[str, JsonValue]:
        return {"case_id": self.case_id, "candidate_id": self.candidate_id, "status": self.status,
                "review": self.review.to_data(), "cost": self.cost.to_data(), "run_id": self.run_id}

    @classmethod
    def from_data(cls, data: Mapping[str, object]) -> ComparisonCell:
        fields(data, "case_id candidate_id status review cost run_id")
        return cls(case_id=string(data["case_id"]), candidate_id=string(data["candidate_id"]),
                   status=string(data["status"]), review=ReviewProjection.from_data(object_data(data["review"])),
                   cost=ComparisonCost.from_data(object_data(data["cost"])),
                   run_id=None if data["run_id"] is None else string(data["run_id"]))


@dataclass(frozen=True, slots=True, kw_only=True)
class ComparisonManifest:
    """Immutable identity of an explicit case-by-candidate comparison matrix."""

    comparison_id: str
    version: str
    application_id: str
    cases: tuple[ComparisonCase, ...] = field(repr=False)
    candidates: tuple[ComparisonCandidate, ...] = field(repr=False)
    cells: tuple[ComparisonCell, ...] = field(repr=False)
    schema_version: int = COMPARISON_SCHEMA_VERSION

    def __post_init__(self) -> None:
        version(self.schema_version)
        for value in (self.comparison_id, self.version, self.application_id):
            string(value)
        cases, candidates, cells = tuple(self.cases), tuple(self.candidates), tuple(self.cells)
        if any(not isinstance(item, ComparisonCase) for item in cases):
            raise TypeError("comparison cases require ComparisonCase values")
        if any(not isinstance(item, ComparisonCandidate) for item in candidates):
            raise TypeError("comparison candidates require ComparisonCandidate values")
        if any(not isinstance(item, ComparisonCell) for item in cells):
            raise TypeError("comparison cells require ComparisonCell values")
        case_ids = _unique(tuple(item.case_id for item in cases))
        candidate_ids = _unique(tuple(item.candidate_id for item in candidates))
        if not cases or len(candidates) < 2:
            raise ValueError("comparison requires at least one case and two candidates")
        if any(item.model_set.application_id != self.application_id for item in candidates):
            raise ValueError("comparison candidate belongs to a different application")
        keys = tuple((item.case_id, item.candidate_id) for item in cells)
        if len(set(keys)) != len(keys):
            raise ValueError("comparison contains duplicate matrix cells")
        if any(a not in case_ids or b not in candidate_ids for a, b in keys):
            raise ValueError("comparison cell has an unknown case or candidate")
        referenced_files: dict[str, tuple[str, str]] = {}
        reviews = (*(item.review for item in cases), *(item.review for item in cells))
        for review in reviews:
            for asset in review.assets:
                identity = (asset.sha256, asset.media_type)
                if asset.path in referenced_files and referenced_files[asset.path] != identity:
                    raise ValueError("comparison references conflicting review file identities")
                referenced_files[asset.path] = identity
        # Canonical cell order prevents incidental execution order changing identity.
        ordered = {(item.case_id, item.candidate_id): item for item in cells}
        object.__setattr__(self, "cases", cases)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "cells", tuple(ordered[(a, b)] for a in case_ids for b in candidate_ids if (a, b) in ordered))

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(candidate.candidate_id for candidate in self.candidates)

    @property
    def complete(self) -> bool:
        return len(self.cells) == len(self.cases) * len(self.candidates) and all(cell.status == "succeeded" for cell in self.cells)

    @property
    def digest(self) -> str:
        canonical = json.dumps(self.to_data(), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_data(self) -> dict[str, JsonValue]:
        return {"schema_version": self.schema_version, "comparison_id": self.comparison_id,
                "version": self.version, "application_id": self.application_id,
                "cases": [case.to_data() for case in self.cases],
                "candidates": [candidate.to_data() for candidate in self.candidates],
                "cells": [cell.to_data() for cell in self.cells]}

    @classmethod
    def from_data(cls, data: Mapping[str, object]) -> ComparisonManifest:
        fields(data, "schema_version comparison_id version application_id cases candidates cells")
        version(data["schema_version"])
        return cls(comparison_id=string(data["comparison_id"]), version=string(data["version"]),
                   application_id=string(data["application_id"]),
                   cases=tuple(ComparisonCase.from_data(object_data(item)) for item in sequence(data["cases"])),
                   candidates=tuple(ComparisonCandidate.from_data(object_data(item)) for item in sequence(data["candidates"])),
                   cells=tuple(ComparisonCell.from_data(object_data(item)) for item in sequence(data["cells"])))


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class HumanRankingBallot:
    """One private complete ranking; equal ranks express ties, best group first."""

    ballot_id: str
    manifest_digest: str
    case_id: str
    case_version: str
    candidate_ids: tuple[str, ...]
    ranks: tuple[tuple[str, ...], ...]
    schema_version: int = RANKING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        version(self.schema_version)
        string(self.ballot_id)
        digest(self.manifest_digest)
        string(self.case_id)
        string(self.case_version)
        candidates = _unique(self.candidate_ids)
        ranks = tuple(tuple(group) for group in self.ranks)
        if len(candidates) < 2 or not ranks or any(not group for group in ranks):
            raise ValueError("ballot requires a complete ranking of at least two candidates")
        ranked = _unique(tuple(candidate for group in ranks for candidate in group))
        if set(ranked) != set(candidates):
            raise ValueError("ballot ranks must include every candidate exactly once")
        # Tied group member order has no semantic meaning.
        positions = {candidate: index for index, candidate in enumerate(candidates)}
        object.__setattr__(self, "candidate_ids", candidates)
        object.__setattr__(self, "ranks", tuple(tuple(sorted(group, key=positions.__getitem__)) for group in ranks))

    def __repr__(self) -> str:
        return "HumanRankingBallot(<private>)"

    def validate_for(self, manifest: ComparisonManifest) -> None:
        if self.manifest_digest != manifest.digest:
            raise ValueError("ballot belongs to a stale or different manifest")
        case = next((item for item in manifest.cases if item.case_id == self.case_id), None)
        if case is None or case.version != self.case_version:
            raise ValueError("ballot case identity is stale or unknown")
        if self.candidate_ids != manifest.candidate_ids:
            raise ValueError("ballot candidate roster differs from the manifest")
        cells = tuple(item for item in manifest.cells if item.case_id == self.case_id)
        if len(cells) != len(manifest.candidates) or any(item.status != "succeeded" for item in cells):
            raise ValueError("ballot case has incomplete or failed matrix cells")

    def to_data(self) -> dict[str, JsonValue]:
        return {"schema_version": self.schema_version, "ballot_id": self.ballot_id,
                "manifest_digest": self.manifest_digest, "case_id": self.case_id,
                "case_version": self.case_version, "candidate_ids": list(self.candidate_ids),
                "ranks": [list(group) for group in self.ranks]}

    @classmethod
    def from_data(cls, data: Mapping[str, object]) -> HumanRankingBallot:
        fields(data, "schema_version ballot_id manifest_digest case_id case_version candidate_ids ranks")
        version(data["schema_version"])
        return cls(ballot_id=string(data["ballot_id"]), manifest_digest=digest(data["manifest_digest"]),
                   case_id=string(data["case_id"]), case_version=string(data["case_version"]),
                   candidate_ids=tuple(string(item) for item in sequence(data["candidate_ids"])),
                   ranks=tuple(tuple(string(candidate) for candidate in sequence(group)) for group in sequence(data["ranks"])))


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class RankingLedger:
    """Private append-only ballots for an exact immutable manifest."""

    manifest_digest: str
    ballots: tuple[HumanRankingBallot, ...] = ()
    schema_version: int = RANKING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        version(self.schema_version)
        digest(self.manifest_digest)
        copied = tuple(self.ballots)
        if any(not isinstance(item, HumanRankingBallot) for item in copied):
            raise TypeError("ranking ledger requires HumanRankingBallot values")
        _unique(tuple(item.ballot_id for item in copied))
        _unique(tuple(item.case_id for item in copied))
        if any(item.manifest_digest != self.manifest_digest for item in copied):
            raise ValueError("ranking ledger mixes different manifests")
        object.__setattr__(self, "ballots", copied)

    def __repr__(self) -> str:
        return "RankingLedger(<private>)"

    def record(self, manifest: ComparisonManifest, ballot: HumanRankingBallot) -> RankingLedger:
        if self.manifest_digest != manifest.digest:
            raise ValueError("ranking ledger belongs to a stale or different manifest")
        for existing in self.ballots:
            existing.validate_for(manifest)
        ballot.validate_for(manifest)
        for existing in self.ballots:
            if existing == ballot:
                return self
            if existing.ballot_id == ballot.ballot_id or existing.case_id == ballot.case_id:
                raise ValueError("ranking ledger already contains this ballot ID or case")
        return RankingLedger(manifest_digest=self.manifest_digest, ballots=(*self.ballots, ballot))

    def to_data(self) -> dict[str, JsonValue]:
        return {"schema_version": self.schema_version, "manifest_digest": self.manifest_digest,
                "ballots": [ballot.to_data() for ballot in self.ballots]}

    @classmethod
    def from_data(cls, data: Mapping[str, object]) -> RankingLedger:
        fields(data, "schema_version manifest_digest ballots")
        version(data["schema_version"])
        return cls(manifest_digest=digest(data["manifest_digest"]),
                   ballots=tuple(HumanRankingBallot.from_data(object_data(item)) for item in sequence(data["ballots"])))


@dataclass(frozen=True, slots=True, kw_only=True)
class RankingSummary:
    """Normalized pairwise scores, equally weighted across the same reviewed cases."""

    manifest_digest: str
    scores: Mapping[str, float]
    cases_ranked: int
    cases_total: int
    winner_candidate_id: str | None
    reason: str
    scoring_version: int = RANKING_SCORING_VERSION

    def __post_init__(self) -> None:
        digest(self.manifest_digest)
        version(self.scoring_version)
        if (type(self.cases_ranked) is not int or type(self.cases_total) is not int
                or not 0 <= self.cases_ranked <= self.cases_total or self.cases_total < 1):
            raise ValueError("ranking summary case counts are invalid")
        copied = dict(self.scores)
        for candidate, score in copied.items():
            string(candidate)
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("ranking summary scores must be finite and normalized")
        if self.reason not in ("complete", "incomplete_matrix", "incomplete_rankings", "tie"):
            raise ValueError("ranking summary reason is invalid")
        if self.winner_candidate_id is not None:
            if self.reason != "complete" or self.cases_ranked != self.cases_total or self.winner_candidate_id not in copied:
                raise ValueError("ranking summary winner requires complete evidence")
            best = max(copied.values())
            if tuple(key for key, value in copied.items() if value == best) != (self.winner_candidate_id,):
                raise ValueError("ranking summary winner must have a unique highest score")
        elif self.reason == "complete":
            raise ValueError("complete ranking summary requires a winner")
        object.__setattr__(self, "scores", MappingProxyType(copied))

    def to_preference(self, manifest: ComparisonManifest) -> PreferredModelSet:
        if self.manifest_digest != manifest.digest or not manifest.complete:
            raise ValueError("preference requires the exact complete comparison manifest")
        if self.winner_candidate_id is None or self.reason != "complete":
            raise ValueError("preference requires a unique completely ranked winner")
        candidate = next((item for item in manifest.candidates if item.candidate_id == self.winner_candidate_id), None)
        if candidate is None:
            raise ValueError("preference candidate is absent from the manifest")
        return PreferredModelSet(application_id=manifest.application_id, model_set=candidate.model_set,
                                 comparison_id=manifest.comparison_id, manifest_digest=manifest.digest,
                                 scoring_version=self.scoring_version)

    def to_data(self) -> dict[str, JsonValue]:
        return {"manifest_digest": self.manifest_digest, "scores": dict(self.scores),
                "cases_ranked": self.cases_ranked, "cases_total": self.cases_total,
                "winner_candidate_id": self.winner_candidate_id, "reason": self.reason,
                "scoring_version": self.scoring_version}

    @classmethod
    def from_data(cls, data: Mapping[str, object]) -> RankingSummary:
        fields(data, "manifest_digest scores cases_ranked cases_total winner_candidate_id reason scoring_version")
        version(data["scoring_version"])
        ranked, total = data["cases_ranked"], data["cases_total"]
        if isinstance(ranked, bool) or not isinstance(ranked, int) or isinstance(total, bool) or not isinstance(total, int):
            raise ValueError("ranking summary case counts are invalid")
        scores: dict[str, float] = {}
        for candidate, score in object_data(data["scores"]).items():
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ValueError("ranking summary score is invalid")
            scores[string(candidate)] = float(score)
        return cls(manifest_digest=digest(data["manifest_digest"]), scores=scores,
                   cases_ranked=ranked, cases_total=total, reason=string(data["reason"]),
                   winner_candidate_id=None if data["winner_candidate_id"] is None else string(data["winner_candidate_id"]))


def aggregate_rankings(manifest: ComparisonManifest, ledger: RankingLedger) -> RankingSummary:
    """Award a pairwise win 1, tie 1/2, loss 0; average cases equally.

    Integer half-win totals determine exact ties before float presentation. Costs
    are evidence for the reviewer, never a hidden score or tie-breaker.
    """
    if ledger.manifest_digest != manifest.digest:
        raise ValueError("ranking ledger belongs to a stale or different manifest")
    points = dict.fromkeys(manifest.candidate_ids, 0)
    for ballot in ledger.ballots:
        ballot.validate_for(manifest)
        positions = {candidate: rank for rank, group in enumerate(ballot.ranks) for candidate in group}
        for candidate in manifest.candidate_ids:
            points[candidate] += sum(2 if positions[candidate] < positions[other] else
                                     1 if positions[candidate] == positions[other] else 0
                                     for other in manifest.candidate_ids if other != candidate)
    count = len(ledger.ballots)
    denominator = 2 * (len(manifest.candidates) - 1) * count
    scores = {candidate: value / denominator for candidate, value in points.items()} if count else {}
    winner: str | None = None
    if not manifest.complete:
        reason = "incomplete_matrix"
    elif count != len(manifest.cases):
        reason = "incomplete_rankings"
    else:
        highest = max(points.values())
        winners = tuple(candidate for candidate, value in points.items() if value == highest)
        reason = "complete" if len(winners) == 1 else "tie"
        if len(winners) == 1:
            winner = winners[0]
    return RankingSummary(manifest_digest=manifest.digest, scores=scores, cases_ranked=count,
                          cases_total=len(manifest.cases), winner_candidate_id=winner, reason=reason)
