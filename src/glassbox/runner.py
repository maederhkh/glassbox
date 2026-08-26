"""Run one recipe over a question set, saving each result as it completes.

Resumable, the same pattern as build_checklists.py and grade_runs.py: a question
whose result file already exists in ``run_dir`` is skipped rather than re-spent,
unless ``rerun`` is True. A run that dies partway through -- or is deliberately
split into chunks to stay under a single command's time budget -- can simply be
re-run to pick up where it left off.
"""

from __future__ import annotations

import dataclasses
import json
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
        result = None
        if path.exists() and not rerun:
            try:
                result = load_result(path)
            except json.JSONDecodeError:
                # The project's documented failure mode: a run killed
                # mid-write leaves a truncated JSON file behind. Treat it as
                # absent rather than crashing or refusing the whole
                # directory -- re-running one question is cheap, and a
                # bare JSONDecodeError doesn't even name the file.
                print(f"[warning] {path} is corrupt (invalid JSON) -- "
                      f"treating it as absent and re-running "
                      f"{question.id[:8]}.")
                result = None
            else:
                _check_prompt_hash(recipe, question, result, path, run_dir)

        if result is not None:
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


def _check_prompt_hash(recipe, question: Question, result: RecipeResult,
                        path: Path, run_dir: Path) -> None:
    """Refuse to resume a result recorded under a different prompt version.

    Skips the comparison (rather than crashing) when either side has no
    recorded hash: a recipe with no ``PROMPT_HASH`` class attribute, or a
    stored result predating the field (e.g. the 11 August results, which have
    since been regenerated). Anything else -- a genuine mismatch -- is the
    exact contamination a prior incident on this project produced (20 results
    on an old prompt, 2 on a new one, discovered only via file mtimes), so it
    raises rather than warns: a warning in a long scrolling run is a warning
    nobody reads.
    """
    current_hash = getattr(recipe, "PROMPT_HASH", None)
    stored_hash = result.metadata.get("prompt_hash")
    if current_hash is None or stored_hash is None:
        return
    if current_hash != stored_hash:
        raise RuntimeError(
            f"refusing to resume {question.id!r}: {path} was produced under "
            f"prompt_hash {stored_hash!r}, but {recipe.name!r}'s current "
            f"prompt_hash is {current_hash!r}. Resuming would silently mix "
            f"prompt versions in {run_dir}. Delete the run directory and "
            f"re-run, or pass --rerun."
        )
