"""Run one recipe over a question set, saving each result as it completes.

Resumable, the same pattern as build_checklists.py and grade_runs.py: a question
whose result file already exists in ``run_dir`` is skipped rather than re-spent,
unless ``rerun`` is True. A run that dies partway through -- or is deliberately
split into chunks to stay under a single command's time budget -- can simply be
re-run to pick up where it left off.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from glassbox.dataset import Question
from glassbox.recipes.base import RecipeResult
from glassbox.storage import load_result, result_path, save_result
from glassbox.usage import Usage


def run_recipe(recipe, questions: list[Question], client, run_dir: Path,
               verbose: bool = True, rerun: bool = False,
               dataset_revision: str | None = None,
               seed: int | None = None) -> list[RecipeResult]:
    """``dataset_revision``/``seed`` identify the frozen sample manifest a run
    was drawn from (see ``data/<name>.json``). They are recipe-independent, so
    rather than plumbing them through every ``Recipe.run()`` signature, they
    are merged into a freshly-run result's metadata here, once, uniformly
    across every recipe. Skipped (already-run) results are loaded as-is and
    never rewritten, so old results on disk keep whatever metadata they were
    saved with.
    """
    run_dir = Path(run_dir)
    results: list[RecipeResult] = []
    total = Usage.zero()
    extra_metadata = {
        k: v for k, v in {"dataset_revision": dataset_revision, "seed": seed}.items()
        if v is not None
    }

    for i, question in enumerate(questions, 1):
        path = result_path(recipe.name, question.id, run_dir)
        if path.exists() and not rerun:
            result = load_result(path)
            if verbose:
                print(f"[{i}/{len(questions)}] {question.id[:8]} "
                      f"{result.usage.total_tokens:>7} tok  [already run]")
        else:
            result = recipe.run(question, client)
            if extra_metadata:
                result = dataclasses.replace(
                    result, metadata={**result.metadata, **extra_metadata}
                )
            save_result(result, run_dir)
            if verbose:
                print(f"[{i}/{len(questions)}] {question.id[:8]} "
                      f"{result.usage.total_tokens:>7} tok  {result.seconds:>5.1f}s")
        results.append(result)
        total = total + result.usage

    return results
