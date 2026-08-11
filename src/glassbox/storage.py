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


def load_results(run_dir: Path) -> list[RecipeResult]:
    results = [
        _to_result(json.loads(p.read_text(encoding="utf-8")))
        for p in sorted(Path(run_dir).glob("*.json"))
    ]
    return sorted(results, key=lambda r: (r.recipe, r.question_id))
