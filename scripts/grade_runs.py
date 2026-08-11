"""Score persisted runs with LEXam's official judge."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from glassbox.config import JUDGE_MODELS, JUDGE_TEMPERATURE, RUNS_DIR
from glassbox.dataset import load_sample
from glassbox.grading.lexam_judge import judge_answer
from glassbox.llm import LLMClient
from glassbox.storage import load_results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--questions", default="dev_20")
    a = p.parse_args()

    run_dir = Path(a.run_dir) if Path(a.run_dir).is_absolute() else RUNS_DIR / a.run_dir
    results = load_results(run_dir)
    questions = {q.id: q for q in load_sample(a.questions)}
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
    print(f"\nscored {len(scored)}/{len(rows)}")
    if scored:
        print(f"mean {statistics.mean(scored) * 100:.1f} / 100")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
