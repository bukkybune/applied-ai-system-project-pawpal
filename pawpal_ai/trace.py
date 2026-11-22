"""Step-by-step record of what the setup agent did, for the UI and for logs."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class TraceStep:
    step: str  # "parse" | "research" | "draft" | "schedule" | "check" | "revise" | "result"
    summary: str
    data: dict = field(default_factory=dict)
    seconds: float = 0.0


@dataclass
class Trace:
    goal: str
    mode: str
    steps: list[TraceStep] = field(default_factory=list)
    _started: float = field(default_factory=time.perf_counter, repr=False)

    def add(self, step: str, summary: str, **data) -> None:
        self.steps.append(
            TraceStep(step, summary, data, round(time.perf_counter() - self._started, 3))
        )

    def to_dict(self) -> dict:
        return {
            "time": datetime.now().isoformat(timespec="seconds"),
            "goal": self.goal,
            "mode": self.mode,
            "steps": [
                {"step": s.step, "summary": s.summary, "data": s.data, "t": s.seconds}
                for s in self.steps
            ],
        }

    def save(self, path: Path) -> None:
        """Append this trace as one JSON line."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(self.to_dict(), default=str) + "\n")
