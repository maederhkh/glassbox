"""Run a recipe over a frozen question set."""

from __future__ import annotations

import argparse

from glassbox.config import EFFORT_BASELINE, RUNS_DIR, SYSTEM_MODEL, SYSTEM_TEMPERATURE
from glassbox.dataset import load_manifest, load_sample
from glassbox.llm import LLMClient
from glassbox.recipes.plain import PlainRecipe
from glassbox.recipes.structured import StructuredRecipe, ThinkLongerRecipe
from glassbox.runner import run_recipe
from glassbox.usage import Usage, cost_usd

RECIPES = {
    "plain": PlainRecipe,
    "structured": StructuredRecipe,
    "think_longer": ThinkLongerRecipe,
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recipe", required=True, choices=sorted(RECIPES))
    p.add_argument("--questions", default="dev_20")
    p.add_argument("--effort", default=EFFORT_BASELINE)
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

    questions = load_sample(a.questions)
    manifest = load_manifest(a.questions)
    if a.only:
        questions = [q for q in questions if q.id == a.only]
        if not questions:
            raise SystemExit(f"{a.only!r} not found in sample {a.questions!r}")
    client = LLMClient(model=SYSTEM_MODEL, temperature=SYSTEM_TEMPERATURE,
                       reasoning_effort=a.effort)
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
