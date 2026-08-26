"""Shared recipe types.

Every recipe turns one Question into one RecipeResult. Single-call recipes leave
`stages` empty; the pipeline fills it, one StageRecord per call, which is what
makes stage-level error attribution possible later.
"""

from __future__ import annotations

import hashlib
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


def prompt_fingerprint(*parts: str) -> str:
    """A stable hash identifying a recipe's prompt *construction*, not any one
    call's literal text.

    Each recipe hashes its fixed system/prompt templates and (where relevant)
    the fully-rendered shared instructions text (e.g.
    ``case_file_json_instructions()``) -- never the per-question substitutions
    (``question``, ``course``), which differ on every call by design and would
    otherwise make every result's hash unique instead of comparable.

    This is what makes a prompt change auditable from the data: every result
    produced under one prompt version shares one hash, and a recipe whose
    instructions text changes (e.g. schema.py's tightening mid-experiment)
    gets a different hash from that point on. Plain sha256 over the parts
    joined with a unit separator (`\\x1f`) so that, e.g., ("ab", "c") and
    ("a", "bc") never collide.
    """
    joined = "\x1f".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()
