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


def load_split(split: str) -> tuple[pd.DataFrame, str]:
    """Return the English open questions of ``split`` plus the dataset revision."""
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

    meta_cols = [
        "id",
        "course",
        "area",
        "jurisdiction",
        "year",
        "question_words",
        "answer_words",
    ]
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
        "selected": drawn[meta_cols].to_dict(orient="records"),
        "reserves": reserves[meta_cols].to_dict(orient="records"),
    }

    DATA_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)

    (DATA_DIR / f"{name}.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    text_cols = meta_cols + ["question", "answer"]
    drawn[text_cols].to_json(
        CACHE_DIR / f"{name}_full.json", orient="records", indent=2, force_ascii=False
    )

    lines = [
        f"# {name}",
        "",
        f"{n} questions drawn from LEXam `open_question` / `{split}` / English, "
        f"seed `{seed}`.",
        f"Eligible population: {len(eligible)} of {len(df)} English questions "
        f"(answer >= {MIN_ANSWER_WORDS} words, question >= {MIN_QUESTION_WORDS} words, "
        f"excluding {', '.join(EXCLUDE_AREAS)}).",
        f"Dataset revision `{revision}`.",
        "",
        "| # | course | area | jurisdiction | year | q words | a words | id |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(drawn.itertuples(), 1):
        lines.append(
            f"| {i} | {r.course} | {r.area} | {r.jurisdiction} | {r.year} | "
            f"{r.question_words} | {r.answer_words} | `{r.id}` |"
        )
    (DATA_DIR / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

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
    p.add_argument("--split", required=True, choices=["dev", "test"])
    p.add_argument("--n", required=True, type=int)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--name", required=True)
    a = p.parse_args()
    select(a.split, a.n, a.seed, a.name)


if __name__ == "__main__":
    main()
