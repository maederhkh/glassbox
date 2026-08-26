"""Persist and reload recipe results as JSON, one file per result."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from glassbox.recipes.base import RecipeResult, StageRecord
from glassbox.usage import Usage


def save_result(result: RecipeResult, run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"{result.recipe}__{result.question_id}.json"
    path.write_text(json.dumps(asdict(result), indent=2, ensure_ascii=False),
                    encoding="utf-8")
    return path


def result_path(recipe_name: str, question_id: str, run_dir: Path) -> Path:
    return Path(run_dir) / f"{recipe_name}__{question_id}.json"


def _to_result(payload: dict) -> RecipeResult:
    return RecipeResult(
        recipe=payload["recipe"],
        question_id=payload["question_id"],
        final_answer=payload["final_answer"],
        sections=payload["sections"],
        stages=[
            StageRecord(
                name=s["name"], prompt=s["prompt"], output=s["output"],
                usage=Usage(**s["usage"]), seconds=s["seconds"],
            )
            for s in payload["stages"]
        ],
        usage=Usage(**payload["usage"]),
        seconds=payload["seconds"],
        metadata=payload.get("metadata", {}),
    )


# A saved RecipeResult is a single JSON object with exactly these top-level keys
# (see save_result/asdict above). Grading scripts (scripts/grade_runs.py) write
# their own *.json output -- a list of row dicts, e.g. lexam_scores.json,
# scores_provisional.json -- into this same run directory, because that is
# where a human looking for "the scores for this run" will look. glob("*.json")
# then picks those files up too. Recognising the result shape (rather than
# hard-coding score filenames) means any future grader output is skipped the
# same way without storage.py needing to know its name in advance.
_RESULT_KEYS = frozenset(
    {"recipe", "question_id", "final_answer", "sections", "stages", "usage",
     "seconds", "metadata"}
)


def _is_result_payload(payload: object) -> bool:
    return isinstance(payload, dict) and _RESULT_KEYS.issubset(payload.keys())


def load_result(path: Path) -> RecipeResult:
    """A single previously-saved result, e.g. to check what a resumed run
    already has on disk without re-scanning the whole run directory."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return _to_result(payload)


def load_results(run_dir: Path) -> list[RecipeResult]:
    results = []
    for p in sorted(Path(run_dir).glob("*.json")):
        payload = json.loads(p.read_text(encoding="utf-8"))
        if not _is_result_payload(payload):
            continue
        results.append(_to_result(payload))
    return sorted(results, key=lambda r: (r.recipe, r.question_id))
