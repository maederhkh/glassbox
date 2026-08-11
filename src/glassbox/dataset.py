"""Load a frozen question sample.

Ids and metadata come from the versioned manifest; question and reference-answer
text comes from the gitignored cache, so the repository never redistributes LEXam
content. Order always follows the manifest, so runs are comparable across sessions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from glassbox.config import CACHE_DIR, DATA_DIR


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    answer: str
    course: str
    area: str
    jurisdiction: str
    year: str
    question_words: int
    answer_words: int


def _read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Regenerate it with:\n"
            f"  python scripts/select_questions.py --split dev --n 20 "
            f"--seed 20260810 --name dev_20"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_manifest(name: str) -> dict:
    return _read_json(DATA_DIR / f"{name}.json")


def load_sample(name: str) -> list[Question]:
    manifest = load_manifest(name)
    rows = _read_json(CACHE_DIR / f"{name}_full.json")
    by_id = {row["id"]: row for row in rows}

    questions = []
    for entry in manifest["selected"]:
        row = by_id[entry["id"]]
        questions.append(
            Question(
                id=row["id"],
                question=row["question"],
                answer=row["answer"],
                course=row["course"],
                area=row["area"],
                jurisdiction=row["jurisdiction"],
                year=str(row["year"]),
                question_words=int(row["question_words"]),
                answer_words=int(row["answer_words"]),
            )
        )
    return questions
