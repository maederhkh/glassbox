"""Score persisted runs with LEXam's official judge."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from glassbox.config import JUDGE_MODELS, JUDGE_TEMPERATURE, RUNS_DIR
from glassbox.dataset import load_sample
from glassbox.grading.lexam_judge import judge_answer, unparseable_counts
from glassbox.llm import LLMClient
from glassbox.storage import load_results


def find_orphaned_results(results, questions: dict) -> list[str]:
    """Result question ids that are not in the current sample.

    This happens when a result was saved before the manifest was later
    changed by ``select_questions.py --replace`` and that id was dropped
    from the sample -- exactly what happened to ``0f6dd9e7`` during this
    study. Scoring a saved answer against a question that no longer exists
    in the frozen sample is meaningless, and a bare ``KeyError`` on
    ``questions[r.question_id]`` gives no hint why.
    """
    return [r.question_id for r in results if r.question_id not in questions]


def find_missing_questions(results, questions: dict) -> list[str]:
    """Sample question ids with no corresponding result in this run directory.

    Without this check, a partial run directory silently reports a mean over
    fewer questions than the full sample, with only "scored N/N" (N = result
    count, not sample size) as the only hint that anything was missing.
    """
    present = {r.question_id for r in results}
    return [qid for qid in questions if qid not in present]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--questions", default="dev_20")
    a = p.parse_args()

    run_dir = Path(a.run_dir) if Path(a.run_dir).is_absolute() else RUNS_DIR / a.run_dir
    results = load_results(run_dir)
    questions = {q.id: q for q in load_sample(a.questions)}

    orphaned = find_orphaned_results(results, questions)
    if orphaned:
        raise SystemExit(
            f"{run_dir} has {len(orphaned)} result(s) for question id(s) not in "
            f"sample {a.questions!r}: {', '.join(orphaned)}.\n"
            f"The sample was likely changed by select_questions.py --replace since "
            f"these results were saved -- check data/{a.questions}.json's "
            f"\"replacements\" list, delete the stale result file(s), and backfill a "
            f"result for the replacement id with `run_recipe.py ... --only <id>` if "
            f"one is missing. Refusing to score before any judge calls are made."
        )

    missing = find_missing_questions(results, questions)
    if missing:
        print(f"WARNING: {len(missing)}/{len(questions)} sample questions have no "
              f"result in {run_dir} and will not be scored: {', '.join(missing)}\n")

    clients = [LLMClient(model=m, temperature=JUDGE_TEMPERATURE) for m in JUDGE_MODELS]

    rows = []
    for i, r in enumerate(results, 1):
        q = questions[r.question_id]
        verdict = judge_answer(clients, q.question, q.answer, r.final_answer)
        rows.append({
            "question_id": r.question_id, "recipe": r.recipe,
            "lexam_score": verdict.score, "per_judge": verdict.per_judge,
            "judges": list(JUDGE_MODELS),
        })
        print(f"[{i}/{len(results)}] {r.question_id[:8]} {verdict.score}")

    out = run_dir / "lexam_scores.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    scored = [r["lexam_score"] for r in rows if r["lexam_score"] is not None]
    print(f"\nscored {len(scored)}/{len(rows)} results "
          f"({len(rows)}/{len(questions)} of sample {a.questions!r})")
    if scored:
        print(f"mean {statistics.mean(scored) * 100:.1f} / 100")

    counts = unparseable_counts(rows, list(JUDGE_MODELS))
    print("\nunparseable (None) counts per judge:")
    for name, c in counts.items():
        print(f"  {name}: {c}/{len(rows)}")

    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
