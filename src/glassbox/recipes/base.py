"""Shared recipe types.

Every recipe turns one Question into one RecipeResult. Single-call recipes leave
`stages` empty; the pipeline fills it, one StageRecord per call, which is what
makes stage-level error attribution possible later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from glassbox.dataset import Question
from glassbox.usage import Usage


@dataclass(frozen=True)
class StageRecord:
    name: str
    prompt: str
    output: str
    usage: Usage
    seconds: float


@dataclass(frozen=True)
class RecipeResult:
    recipe: str
    question_id: str
    final_answer: str
    sections: dict[str, str] | None
    stages: list[StageRecord]
    usage: Usage
    seconds: float
    metadata: dict = field(default_factory=dict)


class Recipe(Protocol):
    name: str

    def run(self, question: Question, client) -> RecipeResult: ...
