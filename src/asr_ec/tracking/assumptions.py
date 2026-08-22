"""Explicit, append-only scientific assumptions for a run."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path


class AssumptionsError(ValueError):
    """Raised when an assumptions registry would become ambiguous or mutable."""


class EvidenceLabel(str, Enum):
    PAPER = "PAPER"
    REFERENCE_PAPER = "REFERENCE_PAPER"
    INFERENCE = "INFERENCE"
    RECOMMENDATION = "RECOMMENDATION"
    OPEN_GAP = "OPEN_GAP"


@dataclass(frozen=True, slots=True)
class Assumption:
    assumption_id: str
    choice: str
    reason: str
    evidence_label: EvidenceLabel
    alternatives_tested: tuple[str, ...] = ()
    selected_on: str | None = None

    def __post_init__(self) -> None:
        if not self.assumption_id.strip() or not self.choice.strip() or not self.reason.strip():
            raise AssumptionsError("assumption id, choice, and reason must be non-empty")

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["evidence_label"] = self.evidence_label.value
        result["alternatives_tested"] = list(self.alternatives_tested)
        return result


class AssumptionsRegistry:
    """Collect unique assumptions and write one immutable registry per run."""

    def __init__(self) -> None:
        self._items: list[Assumption] = []

    def add(self, assumption: Assumption) -> None:
        if any(item.assumption_id == assumption.assumption_id for item in self._items):
            raise AssumptionsError(f"duplicate assumption id: {assumption.assumption_id}")
        self._items.append(assumption)

    def to_json(self) -> str:
        return (
            json.dumps(
                [item.to_dict() for item in self._items],
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    def write_once(self, destination: Path) -> None:
        if destination.exists():
            raise AssumptionsError(f"refusing to overwrite assumptions registry: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.to_json(), encoding="utf-8")
