"""Draft a checklist for every question in a sample, for human verification.

Runs are chunkable and resumable: each checklist is written to disk as soon as it
is built, and by default a question already on disk is skipped rather than
re-spent. That means a run that dies partway through -- or one deliberately split
into --limit-sized chunks to stay under a single command's time budget -- can
simply be re-run to pick up where it left off. Pass --rebuild to discard and
regenerate checklists that already exist (e.g. after tightening BUILD_PROMPT).
"""

from __future__ import annotations

import argparse
import statistics

from glassbox.config import CHECKLIST_DIR, SYSTEM_MODEL
from glassbox.dataset import load_sample
from glassbox.grading.checklist import (
    build_checklist, checklist_to_markdown, load_checklist, save_checklist,
)
from glassbox.llm import LLMClient


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--questions", default="dev_20")
    p.add_argument("--rebuild", action="store_true",
                    help="Rebuild checklists that already exist on disk instead "
                         "of skipping them")
    p.add_argument("--limit", type=int, default=None,
                    help="Build at most this many new checklists in this run, "
                         "then stop; already-built questions are still reported "
                         "and do not count against the limit. Re-run (without "
                         "--rebuild) to continue where this run left off")
    a = p.parse_args()

    questions = load_sample(a.questions)
    client = LLMClient(model=SYSTEM_MODEL, temperature=0.0)
    review_dir = CHECKLIST_DIR / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    counts, sources, built = [], [], 0
    for i, q in enumerate(questions, 1):
        path = CHECKLIST_DIR / f"{q.id}.json"
        if path.exists() and not a.rebuild:
            checklist = load_checklist(q.id, CHECKLIST_DIR)
            print(f"[{i}/{len(questions)}] {q.id[:8]} {len(checklist.points):>2} points "
                  f"({checklist.source}) [already built]")
        elif a.limit is not None and built >= a.limit:
            print(f"[{i}/{len(questions)}] {q.id[:8]} -- not built yet, --limit "
                  f"{a.limit} reached this run; re-run to continue")
            continue
        else:
            checklist = build_checklist(q, client)
            save_checklist(checklist, CHECKLIST_DIR)
            (review_dir / f"{q.id}.md").write_text(
                checklist_to_markdown(checklist, q), encoding="utf-8"
            )
            built += 1
            print(f"[{i}/{len(questions)}] {q.id[:8]} {len(checklist.points):>2} points "
                  f"({checklist.source})")
        counts.append(len(checklist.points))
        sources.append(checklist.source)

    print(f"\nbuilt {built} this run; {len(counts)}/{len(questions)} checklists on disk")
    if counts:
        print(f"total points   {sum(counts)}")
        print(f"median/question {statistics.median(counts):.1f}  "
              f"range {min(counts)}-{max(counts)}")
        print(f"reference types {dict((s, sources.count(s)) for s in set(sources))}")
    if len(counts) < len(questions):
        print(f"\n{len(questions) - len(counts)} question(s) still unbuilt -- re-run "
              f"(without --rebuild) to continue")
    else:
        print(f"\nreview the markdown in {review_dir} and correct it by hand")


if __name__ == "__main__":
    main()
