"""Run one recipe over a question set, saving each result as it completes."""

from __future__ import annotations

from pathlib import Path

from glassbox.dataset import Question
from glassbox.recipes.base import RecipeResult
from glassbox.storage import save_result
from glassbox.usage import Usage


def run_recipe(recipe, questions: list[Question], client, run_dir: Path,
               verbose: bool = True) -> list[RecipeResult]:
    results: list[RecipeResult] = []
    total = Usage.zero()

    for i, question in enumerate(questions, 1):
        result = recipe.run(question, client)
        save_result(result, run_dir)
        results.append(result)
        total = total + result.usage
        if verbose:
            print(f"[{i}/{len(questions)}] {question.id[:8]} "
                  f"{result.usage.total_tokens:>7} tok  {result.seconds:>5.1f}s")

    return results
