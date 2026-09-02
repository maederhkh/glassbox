"""Run a recipe over a frozen question set."""

from __future__ import annotations

import argparse

from glassbox.config import (
    EFFORT_BASELINE, EFFORT_RAISED, RUNS_DIR, SYSTEM_MODEL, SYSTEM_TEMPERATURE,
)
from glassbox.dataset import load_manifest, load_sample
from glassbox.llm import LLMClient
from glassbox.recipes.pipeline import PipelineRecipe
from glassbox.recipes.plain import PlainRecipe
from glassbox.recipes.structured import StructuredRecipe, ThinkLongerRecipe
from glassbox.runner import run_recipe
from glassbox.usage import Usage, cost_usd

RECIPES = {
    "plain": PlainRecipe,
    "structured": StructuredRecipe,
    "think_longer": ThinkLongerRecipe,
    "pipeline": PipelineRecipe,
}

# Recipe 3's only difference from Recipe 2 is the raised reasoning effort, so
# the right effort is the recipe's default rather than something an operator
# must remember to pass. Running think_longer at the baseline would silently
# produce Recipe 2 wearing Recipe 3's label.
DEFAULT_EFFORT = {
    "plain": EFFORT_BASELINE,
    "structured": EFFORT_BASELINE,
    "think_longer": EFFORT_RAISED,
    # Four calls, not harder thinking per call. Raised effort is Recipe 3's
    # manipulation and only Recipe 3's.
    "pipeline": EFFORT_BASELINE,
}
VALID_EFFORTS = (EFFORT_BASELINE, EFFORT_RAISED)


def resolve_effort(recipe_name: str, explicit: str | None) -> str:
    """The effort a run should use: the recipe's default, or an explicit override.

    Rejects an unrecognised value here, before ``main()`` constructs a client,
    so a typo cannot reach the API and burn the client's retries against it.
    """
    if recipe_name not in DEFAULT_EFFORT:
        raise SystemExit(
            f"no default reasoning effort recorded for recipe {recipe_name!r}; "
            f"add one to DEFAULT_EFFORT rather than letting it fall back silently"
        )
    effort = DEFAULT_EFFORT[recipe_name] if explicit is None else explicit
    if effort not in VALID_EFFORTS:
        raise SystemExit(
            f"unrecognised --effort {effort!r}; expected one of "
            f"{', '.join(repr(e) for e in VALID_EFFORTS)}"
        )
    return effort


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recipe", required=True, choices=sorted(RECIPES))
    p.add_argument("--questions", default="dev_20")
    p.add_argument(
        "--effort", default=None,
        help="override the recipe's default reasoning effort "
             f"(plain/structured: {EFFORT_BASELINE}, think_longer: {EFFORT_RAISED})",
    )
    p.add_argument("--tag", default="", help="suffix for the output directory")
    p.add_argument(
        "--only",
        help="run a single question id from the sample, writing into the same "
             "run directory as a full run. For backfilling one result (e.g. "
             "after a manifest replacement) without re-running the rest.",
    )
    p.add_argument(
        "--rerun", action="store_true",
        help="re-run and overwrite a question's result even if one already "
             "exists in the run directory. Without this flag, a question "
             "whose result file already exists is skipped rather than "
             "re-spent, so a killed run costs nothing to resume.",
    )
    a = p.parse_args()

    effort = resolve_effort(a.recipe, a.effort)
    questions = load_sample(a.questions)
    manifest = load_manifest(a.questions)
    if a.only:
        questions = [q for q in questions if q.id == a.only]
        if not questions:
            raise SystemExit(f"{a.only!r} not found in sample {a.questions!r}")
    client = LLMClient(model=SYSTEM_MODEL, temperature=SYSTEM_TEMPERATURE,
                       reasoning_effort=effort)
    run_dir = RUNS_DIR / f"{a.questions}__{a.recipe}{a.tag}"

    print(f"{a.recipe} over {len(questions)} questions -> {run_dir}\n")
    results = run_recipe(
        RECIPES[a.recipe](), questions, client, run_dir, rerun=a.rerun,
        dataset_revision=manifest.get("dataset_revision"), seed=manifest.get("seed"),
    )

    total = Usage.zero()
    for r in results:
        total = total + r.usage
    cost = cost_usd(total, SYSTEM_MODEL)

    print(f"\ncalls          {total.calls}")
    print(f"input tokens   {total.input_tokens:,}")
    print(f"output tokens  {total.output_tokens:,}")
    print(f"reasoning tok  {total.reasoning_tokens:,}")
    print(f"total tokens   {total.total_tokens:,}")
    print(f"cost           {'unknown' if cost is None else f'${cost:.4f}'}")
    print(f"time           {sum(r.seconds for r in results):.1f}s")


if __name__ == "__main__":
    main()
