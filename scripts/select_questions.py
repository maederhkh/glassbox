"""Draw a frozen, reproducible sample of LEXam English open questions.

Used twice in the project:

  * development set - drawn from the ``dev`` split, for building and debugging
    the pipeline
  * evaluation set  - drawn from the ``test`` split, for the actual experiment

The two splits are disjoint by construction, so the development set can never
contaminate the evaluation set.

Only ids, metadata and provenance are written to ``data/``. The question and
answer text is LEXam's, and is cached outside version control so the repository
never redistributes it.

Usage::

    python scripts/select_questions.py --split dev  --n 20 --seed 20260810 --name dev_20
    python scripts/select_questions.py --split test --n 30 --seed 20260810 --name eval_30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

DATASET = "LEXam-Benchmark/LEXam"
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"

MIN_ANSWER_WORDS = 150
MIN_QUESTION_WORDS = 50
EXCLUDE_AREAS = ("Interdisciplinary",)


def load_split(split: str, revision: str | None = None) -> tuple[pd.DataFrame, str]:
    """Return the English open questions of ``split`` plus the dataset revision.

    Pass the revision recorded in an existing manifest to pin the fetch to
    that exact historical snapshot (used by ``replace_question`` below) --
    otherwise the current HEAD revision is resolved and returned.
    """
    if revision is None:
        revision = HfApi().dataset_info(DATASET).sha
    path = hf_hub_download(
        DATASET,
        f"open_question/{split}-00000-of-00001.parquet",
        repo_type="dataset",
        revision=revision,
    )
    df = pd.read_parquet(path)
    df = df[df["language"] == "en"].copy()
    df["question_words"] = df["question"].str.split().str.len()
    df["answer_words"] = df["answer"].str.split().str.len()
    return df, revision


def _fetch_rows(split: str, ids: list[str], revision: str) -> pd.DataFrame:
    """Rows for exactly ``ids``, fetched from ``split`` at ``revision``."""
    df, _ = load_split(split, revision=revision)
    matched = df[df["id"].isin(ids)]
    missing = set(ids) - set(matched["id"])
    if missing:
        raise SystemExit(f"id(s) not found in '{split}' at revision {revision}: {missing}")
    return matched


META_COLS = [
    "id", "course", "area", "jurisdiction", "year", "question_words", "answer_words",
]


def _render_markdown(name: str, split: str, seed: int, revision: str,
                      eligible_n: int, english_n: int, selected: list[dict]) -> str:
    lines = [
        f"# {name}",
        "",
        f"{len(selected)} questions drawn from LEXam `open_question` / `{split}` / English, "
        f"seed `{seed}`.",
        f"Eligible population: {eligible_n} of {english_n} English questions "
        f"(answer >= {MIN_ANSWER_WORDS} words, question >= {MIN_QUESTION_WORDS} words, "
        f"excluding {', '.join(EXCLUDE_AREAS)}).",
        f"Dataset revision `{revision}`.",
        "",
        "| # | course | area | jurisdiction | year | q words | a words | id |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(selected, 1):
        lines.append(
            f"| {i} | {r['course']} | {r['area']} | {r['jurisdiction']} | {r['year']} | "
            f"{r['question_words']} | {r['answer_words']} | `{r['id']}` |"
        )
    return "\n".join(lines) + "\n"


def _write_full_cache(name: str, split: str, revision: str, selected: list[dict]) -> None:
    """(Re)write the gitignored full-text cache, in ``selected``'s order."""
    ids = [r["id"] for r in selected]
    rows = _fetch_rows(split, ids, revision)
    text_cols = META_COLS + ["question", "answer"]
    position = {qid: i for i, qid in enumerate(ids)}
    ordered = rows.assign(_position=rows["id"].map(position)).sort_values("_position")
    ordered[text_cols].to_json(
        CACHE_DIR / f"{name}_full.json", orient="records", indent=2, force_ascii=False
    )


def replace_question(name: str, old_id: str, new_id: str, reason: str) -> None:
    """Substitute one selected question with a pre-recorded reserve.

    This is a single, explicit substitution -- not a redraw. The other 19
    selected questions keep their ids, order and metadata exactly; only the
    entry matching ``old_id`` is replaced in place with the reserve matching
    ``new_id``, which is then removed from the reserve list. The
    substitution (old id, new id, position, reason) is appended to a
    ``replacements`` list inside the manifest itself, so a future reader can
    see what happened and why without reconstructing it from git history.

    The reserve's metadata is re-fetched from the dataset at the manifest's
    own recorded revision (not copied from the reserve list) so the
    manifest, the markdown view and the gitignored full-text cache are
    regenerated from one source of truth and cannot drift apart.
    """
    manifest_path = DATA_DIR / f"{name}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    selected = manifest["selected"]
    reserves = manifest["reserves"]

    old_index = next((i for i, r in enumerate(selected) if r["id"] == old_id), None)
    if old_index is None:
        raise SystemExit(f"{old_id!r} is not in {name}'s selected list")

    reserve_index = next((i for i, r in enumerate(reserves) if r["id"] == new_id), None)
    if reserve_index is None:
        raise SystemExit(f"{new_id!r} is not in {name}'s reserve list")

    fetched = _fetch_rows(manifest["split"], [new_id], manifest["dataset_revision"])
    new_row = fetched[META_COLS].to_dict(orient="records")[0]

    reserves.pop(reserve_index)
    selected[old_index] = new_row
    manifest.setdefault("replacements", []).append({
        "removed": old_id,
        "substituted": new_id,
        "position": old_index,
        "reason": reason,
    })

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    md = _render_markdown(
        name, manifest["split"], manifest["seed"], manifest["dataset_revision"],
        manifest["population"]["eligible_after_criterion"],
        manifest["population"]["english_in_split"], selected,
    )
    (DATA_DIR / f"{name}.md").write_text(md, encoding="utf-8")

    _write_full_cache(name, manifest["split"], manifest["dataset_revision"], selected)

    print(f"{name}: replaced {old_id} with {new_id} at position {old_index + 1}")
    print(f"reason: {reason}")
    print(f"reserves remaining: {len(reserves)}")
    print(f"wrote {manifest_path}")
    print(f"wrote {DATA_DIR / f'{name}.md'}")
    print(f"wrote {CACHE_DIR / f'{name}_full.json'} (gitignored)")


def apply_criterion(df: pd.DataFrame) -> pd.DataFrame:
    """Keep questions that require applied legal reasoning.

    Two filters, both recorded in the manifest so the draw is reproducible.

    Length is a proxy, not a definition: short answers in LEXam are
    overwhelmingly factual recall ("What are the requirements to take the bar
    exam?" -> "J.D."), which the study is not about.

    ``Interdisciplinary`` is excluded because it holds the Legal Theory and
    Legal Sociology courses, whose questions are moral philosophy and social
    theory essays ("Kelsen argued justice is an empty concept, do you agree?")
    rather than subsumption. They have no issues, rules or elements to
    decompose, so a staged legal-reasoning pipeline is not applicable to them.

    Any drawn question that still turns out to be recall or essay on inspection
    is replaced by the next one in the shuffled order, and the number of
    replacements is reported.
    """
    return df[
        (df["answer_words"] >= MIN_ANSWER_WORDS)
        & (df["question_words"] >= MIN_QUESTION_WORDS)
        & (~df["area"].isin(EXCLUDE_AREAS))
    ].copy()


def select(split: str, n: int, seed: int, name: str) -> None:
    df, revision = load_split(split)
    eligible = apply_criterion(df)

    if len(eligible) < n:
        raise SystemExit(
            f"only {len(eligible)} eligible questions in '{split}', cannot draw {n}"
        )

    # Sort by id first so the draw does not depend on parquet row order.
    ordered = eligible.sort_values("id").reset_index(drop=True)
    shuffled = ordered.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    drawn = shuffled.head(n)
    reserves = shuffled.iloc[n : n + 10]

    selected_records = drawn[META_COLS].to_dict(orient="records")
    manifest = {
        "name": name,
        "dataset": DATASET,
        "dataset_revision": revision,
        "config": "open_question",
        "split": split,
        "language": "en",
        "criterion": {
            "min_answer_words": MIN_ANSWER_WORDS,
            "min_question_words": MIN_QUESTION_WORDS,
            "excluded_areas": list(EXCLUDE_AREAS),
        },
        "population": {
            "english_in_split": int(len(df)),
            "eligible_after_criterion": int(len(eligible)),
        },
        "seed": seed,
        "n": n,
        "selected": selected_records,
        "reserves": reserves[META_COLS].to_dict(orient="records"),
    }

    DATA_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)

    (DATA_DIR / f"{name}.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    text_cols = META_COLS + ["question", "answer"]
    drawn[text_cols].to_json(
        CACHE_DIR / f"{name}_full.json", orient="records", indent=2, force_ascii=False
    )

    md = _render_markdown(
        name, split, seed, revision, len(eligible), len(df), selected_records,
    )
    (DATA_DIR / f"{name}.md").write_text(md, encoding="utf-8")

    print(f"{name}: drew {n} of {len(eligible)} eligible "
          f"({len(df)} English in '{split}' split)")
    print(f"revision {revision}")
    print()
    print("area:        ", drawn["area"].value_counts().to_dict())
    print("jurisdiction:", drawn["jurisdiction"].value_counts().to_dict())
    print("courses:     ", drawn["course"].nunique(), "distinct")
    print("answer words: median", int(drawn["answer_words"].median()),
          "| range", int(drawn["answer_words"].min()),
          "-", int(drawn["answer_words"].max()))
    print()
    print(f"wrote {DATA_DIR / f'{name}.json'}")
    print(f"wrote {DATA_DIR / f'{name}.md'}")
    print(f"wrote {CACHE_DIR / f'{name}_full.json'} (gitignored)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=["dev", "test"])
    p.add_argument("--n", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--name", required=True)
    p.add_argument(
        "--replace", nargs=2, metavar=("OLD_ID", "NEW_ID"),
        help="substitute OLD_ID (in --name's selected list) with NEW_ID "
             "(in --name's reserve list) in place. Requires --reason. Does "
             "not touch --split/--n/--seed.",
    )
    p.add_argument("--reason", help="required with --replace; recorded in the manifest")
    a = p.parse_args()

    if a.replace:
        if not a.reason:
            p.error("--replace requires --reason")
        replace_question(a.name, a.replace[0], a.replace[1], a.reason)
        return

    if a.split is None or a.n is None or a.seed is None:
        p.error("--split, --n and --seed are required unless using --replace")
    select(a.split, a.n, a.seed, a.name)


if __name__ == "__main__":
    main()
