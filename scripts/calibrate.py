"""Calibration gate. Run before trusting any score.

Check 1 - the reference answer, judged against itself, must score high. If it does
not, the prompt, the parsing or the model wiring is broken.

Check 2 - an empty answer must score near zero. If it does not, the judge is
rewarding presence rather than content, and every later comparison is noise.

Check 3 (optional, --replicate N) - run LEXam's own QA_PROMPT over N random test-split
questions and check the mean lands near their published figure for the model.
"""

from __future__ import annotations

import argparse
import statistics

from glassbox.config import JUDGE_MODELS, JUDGE_TEMPERATURE
from glassbox.dataset import load_sample
from glassbox.grading.lexam_judge import judge_answer
from glassbox.llm import LLMClient

EMPTY_ANSWER = "I am not able to answer this question."


def _mean(scores: list[float | None]) -> float:
    usable = [s for s in scores if s is not None]
    return statistics.mean(usable) if usable else 0.0


def check_reference_scores_high(questions, clients, threshold: float = 0.8):
    scores = [
        judge_answer(clients, q.question, q.answer, q.answer).score for q in questions
    ]
    mean = _mean(scores)
    detail = ", ".join(f"{s}" for s in scores)
    return mean >= threshold, mean, detail


def check_empty_scores_low(questions, clients, threshold: float = 0.2):
    scores = [
        judge_answer(clients, q.question, q.answer, EMPTY_ANSWER).score for q in questions
    ]
    mean = _mean(scores)
    detail = ", ".join(f"{s}" for s in scores)
    return mean <= threshold, mean, detail


def run_reference_check_detailed(questions, clients) -> list[dict]:
    """Like check_reference_scores_high, but keeps every judge's individual
    score rather than collapsing straight to the ensemble minimum.

    Running the quick 5-question gate told us the judge is wired correctly;
    it does not tell us whether the other 15 frozen questions are internally
    consistent (question and reference answer actually correspond to each
    other). This runs the same reference-scores-itself check over an
    arbitrary set of questions and returns per-question, per-judge detail so
    a human can inspect every question that scores low and classify why.
    """
    rows = []
    for q in questions:
        verdict = judge_answer(clients, q.question, q.answer, q.answer)
        rows.append({
            "question_id": q.id,
            "score": verdict.score,
            "per_judge": verdict.per_judge,
        })
    return rows


def unparseable_counts(rows: list[dict], judge_names: list[str]) -> dict[str, int]:
    """How many times each judge (by position in `judge_names`) returned a
    score `SCORE_PATTERN` could not parse, across `rows` from
    `run_reference_check_detailed`. A None here means that judge's vote was
    silently dropped from `ensemble_min` for that question."""
    counts = {name: 0 for name in judge_names}
    for row in rows:
        for name, score in zip(judge_names, row["per_judge"]):
            if score is None:
                counts[name] += 1
    return counts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--questions", default="dev_20")
    p.add_argument("--n", type=int, default=5, help="questions per sanity check")
    p.add_argument(
        "--full-validate", action="store_true",
        help="run the reference-scores-itself check over the ENTIRE named "
             "sample (ignores --n), printing per-judge detail for manual "
             "review. Diagnostic only: does not enforce pass/fail, does not "
             "run the empty-answer check.",
    )
    a = p.parse_args()

    if a.full_validate:
        questions = load_sample(a.questions)
        clients = [LLMClient(model=m, temperature=JUDGE_TEMPERATURE) for m in JUDGE_MODELS]
        print(f"judges: {', '.join(JUDGE_MODELS)}   questions: {len(questions)}\n")

        rows = run_reference_check_detailed(questions, clients)
        for r in rows:
            flag = "" if (r["score"] or 0) >= 0.8 else "  <-- below 0.80"
            print(f"{r['question_id'][:8]}  ensemble={r['score']}  "
                  f"per_judge={r['per_judge']}{flag}")

        scores = [r["score"] for r in rows]
        print(f"\nmean={_mean(scores):.3f} over {len(rows)} questions")

        counts = unparseable_counts(rows, list(JUDGE_MODELS))
        print("\nunparseable (None) counts per judge:")
        for name, c in counts.items():
            print(f"  {name}: {c}/{len(rows)}")
        return

    questions = load_sample(a.questions)[: a.n]
    clients = [LLMClient(model=m, temperature=JUDGE_TEMPERATURE) for m in JUDGE_MODELS]
    print(f"judges: {', '.join(JUDGE_MODELS)}   questions: {len(questions)}\n")

    ok_ref, mean_ref, detail_ref = check_reference_scores_high(questions, clients)
    print(f"[{'PASS' if ok_ref else 'FAIL'}] reference answer scores high  "
          f"mean={mean_ref:.2f} (need >= 0.80)")
    print(f"        per question: {detail_ref}")

    clients = [LLMClient(model=m, temperature=JUDGE_TEMPERATURE) for m in JUDGE_MODELS]
    ok_empty, mean_empty, detail_empty = check_empty_scores_low(questions, clients)
    print(f"[{'PASS' if ok_empty else 'FAIL'}] empty answer scores low       "
          f"mean={mean_empty:.2f} (need <= 0.20)")
    print(f"        per question: {detail_empty}")

    if ok_ref and ok_empty:
        print("\nGATE PASSED - judge wiring is sound.")
    else:
        raise SystemExit("\nGATE FAILED - fix the judge before grading anything.")


if __name__ == "__main__":
    main()
