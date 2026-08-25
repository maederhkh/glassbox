"""Score persisted runs on all three grading layers: LEXam's official judge
ensemble, per-criterion sub-scores, and checklist coverage.

Runs are chunkable and resumable, the same pattern as build_checklists.py: each
row is written to disk as soon as it is scored, and a (question_id, recipe)
pair already present in the output file is skipped rather than re-spent. A run
that dies partway through -- or is deliberately split into --limit-sized
chunks to stay under a single command's time budget -- can simply be re-run to
pick up where it left off.

Output is written as scores_provisional.json, not scores.json: spec section
7.4 requires checklists to be frozen by hand-verification before any answer is
scored, and that verification has not happened yet for the checklists in
data/checklists/. The "_provisional" name is a guardrail so nobody later
mistakes these numbers for the study's numbers.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

from glassbox.config import CHECKLIST_DIR, JUDGE_MODELS, JUDGE_TEMPERATURE, RUNS_DIR
from glassbox.dataset import load_sample
from glassbox.grading.checklist import coverage_fraction, load_checklist, score_checklist
from glassbox.grading.lexam_judge import judge_answer, unparseable_counts
from glassbox.grading.normalise import normalise
from glassbox.grading.subscores import score_subscores
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


def load_existing_scores(path: Path) -> dict[tuple[str, str], dict]:
    """Rows already scored in a prior invocation, keyed by (question_id, recipe).

    Empty if ``path`` does not exist yet -- the first invocation for a run
    directory. Reading these back is what makes re-running this script after
    a partial (or --limit-capped) run cheap: a result whose key is already
    present here is reported, not re-scored.
    """
    if not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {(row["question_id"], row["recipe"]): row for row in rows}


def score_one(result, question, judges, grader, checklist_dir: Path) -> dict:
    """Grade a single result on all three layers, blind to which recipe made it."""
    blind = normalise(result)

    verdict = judge_answer(judges, question.question, question.answer, blind)
    subs = score_subscores(question.question, question.answer, blind, grader)
    checklist_verdict = score_checklist(
        load_checklist(question.id, checklist_dir), blind, grader
    )
    coverage = coverage_fraction(checklist_verdict)

    return {
        "question_id": result.question_id,
        "recipe": result.recipe,
        "lexam_score": verdict.score,
        "per_judge": verdict.per_judge,
        "judges": list(JUDGE_MODELS),
        "issue_spotting": subs.issue_spotting,
        "rule_recall": subs.rule_recall,
        "rule_application": subs.rule_application,
        "conclusion": subs.conclusion,
        "coverage": coverage,
        "points_total": len(checklist_verdict.verdicts),
        "points_covered": sum(
            1 for v in checklist_verdict.verdicts if v.coverage == "covered"),
        "contradictions": len(checklist_verdict.contradictions),
        "inventions": len(checklist_verdict.inventions),
        "total_tokens": result.usage.total_tokens,
    }


def _sorted_rows(existing: dict[tuple[str, str], dict]) -> list[dict]:
    """The on-disk row order: sorted by (recipe, question_id), matching
    ``load_results``' own sort -- independent of whatever order a
    ``run_scoring`` loop happened to visit ``results`` in.
    """
    return sorted(existing.values(), key=lambda row: (row["recipe"], row["question_id"]))


def _write_scores(path: Path, rows: list[dict]) -> None:
    """Write ``rows`` to ``path`` as JSON, atomically.

    A plain ``write_text`` leaves a truncated, unparseable file on disk if the
    process is killed mid-write; ``load_existing_scores`` would then crash on
    every row, not just the one in flight, on the very next invocation.
    Writing to a temp file in the same directory first and swapping it in
    with ``os.replace`` (atomic on both POSIX and Windows) means a kill
    leaves either the previous complete file or the new complete file in
    place -- never something in between.
    """
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


def run_scoring(
    results, questions: dict, judges, grader, checklist_dir: Path, out_path: Path,
    limit: int | None = None,
) -> tuple[list[dict], int]:
    """Score every result in ``results`` not already present at ``out_path``,
    persisting after every new row. Returns ``(rows, scored_this_run)``:
    ``rows`` is every row on disk by the time this returns (existing rows
    plus whatever was newly scored this call), ``scored_this_run`` is how
    many were newly scored.

    Each persist writes the full current union of existing and newly scored
    rows (``_sorted_rows(existing)``), never just the rows ``results``
    happened to visit so far. Writing a partial prefix would transiently
    drop any already-scored row that sorts after whatever is being scored
    right now -- for as long as the loop takes to reach it again, or until
    the run ends -- and a kill in that window would silently lose a row
    that real money already paid for.
    """
    existing = load_existing_scores(out_path)
    scored_this_run = 0
    for i, r in enumerate(results, 1):
        key = (r.question_id, r.recipe)
        if key in existing:
            print(f"[{i}/{len(results)}] {r.question_id[:8]} [already scored]")
            continue
        if limit is not None and scored_this_run >= limit:
            print(f"[{i}/{len(results)}] {r.question_id[:8]} -- not scored yet, "
                  f"--limit {limit} reached this run; re-run to continue")
            continue

        q = questions[r.question_id]
        row = score_one(r, q, judges, grader, checklist_dir)
        existing[key] = row
        scored_this_run += 1
        _write_scores(out_path, _sorted_rows(existing))
        print(f"[{i}/{len(results)}] {r.question_id[:8]} "
              f"lexam={row['lexam_score']} coverage={row['coverage']:.2f}")

    rows = _sorted_rows(existing)
    _write_scores(out_path, rows)
    return rows, scored_this_run


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--questions", default="dev_20")
    p.add_argument("--limit", type=int, default=None,
                   help="Score at most this many new results in this run, then "
                        "stop; already-scored results are still reported and do "
                        "not count against the limit. Re-run (same --run-dir) "
                        "to continue where this run left off.")
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

    out = run_dir / "scores_provisional.json"
    judges = [LLMClient(model=m, temperature=JUDGE_TEMPERATURE) for m in JUDGE_MODELS]
    grader = LLMClient(model=JUDGE_MODELS[0], temperature=JUDGE_TEMPERATURE)

    rows, scored_this_run = run_scoring(
        results, questions, judges, grader, CHECKLIST_DIR, out, limit=a.limit
    )

    print(f"\nscored {len(rows)}/{len(results)} results "
          f"({len(rows)}/{len(questions)} of sample {a.questions!r}), "
          f"{scored_this_run} newly scored this run")
    if len(rows) < len(results):
        print(f"{len(results) - len(rows)} result(s) still unscored -- re-run "
              f"(same --run-dir) to continue")

    def mean_of(key):
        vals = [row[key] for row in rows if row[key] is not None]
        return statistics.mean(vals) if vals else float("nan")

    print(f"\n{'lexam score':<18}{mean_of('lexam_score') * 100:>8.1f} / 100")
    print(f"{'coverage':<18}{mean_of('coverage') * 100:>8.1f} %")
    print(f"{'issue spotting':<18}{mean_of('issue_spotting'):>8.2f}")
    print(f"{'rule recall':<18}{mean_of('rule_recall'):>8.2f}")
    print(f"{'rule application':<18}{mean_of('rule_application'):>8.2f}")
    print(f"{'conclusion':<18}{mean_of('conclusion'):>8.2f}")
    print(f"{'contradictions':<18}{sum(r['contradictions'] for r in rows):>8}")
    print(f"{'inventions':<18}{sum(r['inventions'] for r in rows):>8}")

    counts = unparseable_counts(rows, list(JUDGE_MODELS))
    print("\nunparseable (None) counts per judge:")
    for name, c in counts.items():
        print(f"  {name}: {c}/{len(rows)}")

    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
