"""Tests for scripts/select_questions.py's replace_question mechanism.

replace_question is the one function that mutates the study's frozen
sample, so both the happy path and every error path are covered here.
_fetch_rows (the only thing in this module that touches the network) is
always stubbed or, where an error path must be reached without ever
calling it, replaced with a poison pill that fails the test if invoked.
"""

from __future__ import annotations

import json
import sys

import pandas as pd
import pytest

from scripts import select_questions as sq


def _row(id_, **overrides):
    row = {
        "id": id_, "course": "Contracts", "area": "Private",
        "jurisdiction": "Domestic", "year": "2020",
        "question_words": 60, "answer_words": 160,
    }
    row.update(overrides)
    return row


def _text_row(id_, **overrides):
    row = _row(id_, **overrides)
    row["question"] = f"question text for {id_} " * 10
    row["answer"] = f"answer text for {id_} " * 20
    return row


def _write_manifest(data_dir, name, selected, reserves, **extra):
    manifest = {
        "name": name,
        "dataset": "LEXam-Benchmark/LEXam",
        "dataset_revision": "rev-pinned",
        "config": "open_question",
        "split": "dev",
        "language": "en",
        "criterion": {"min_answer_words": 150, "min_question_words": 50, "excluded_areas": []},
        "population": {"english_in_split": 80, "eligible_after_criterion": 22},
        "seed": 42,
        "n": 2,
        "selected": selected,
        "reserves": reserves,
    }
    manifest.update(extra)
    (data_dir / f"{name}.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def _stub_fetch_rows(text_rows_by_id):
    """Network-free stand-in for _fetch_rows: looks up ids in a fixed table."""
    def fake(split, ids, revision):
        missing = [i for i in ids if i not in text_rows_by_id]
        if missing:
            raise SystemExit(f"id(s) not found in {split!r} at revision {revision}: {missing}")
        return pd.DataFrame([text_rows_by_id[i] for i in ids])
    return fake


def _poison_fetch(split, ids, revision):
    raise AssertionError(
        "_fetch_rows must not be called on this path -- it is the only network access "
        "in this module, and validation should fail before reaching it"
    )


@pytest.fixture
def sample(tmp_path, monkeypatch):
    """A tiny 2-selected/2-reserve manifest + matching gitignored cache, in tmp_path."""
    monkeypatch.setattr(sq, "DATA_DIR", tmp_path)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(sq, "CACHE_DIR", cache_dir)

    selected = [_row("old-1"), _row("keep-2", course="Torts")]
    reserves = [_row("reserve-1", course="Property"), _row("reserve-2")]
    _write_manifest(tmp_path, "tiny", selected, reserves)

    cache_rows = [_text_row("old-1"), _text_row("keep-2", course="Torts")]
    (cache_dir / "tiny_full.json").write_text(json.dumps(cache_rows), encoding="utf-8")

    return tmp_path, cache_dir


def test_replace_swaps_in_place_and_leaves_the_survivor_untouched(sample, monkeypatch):
    data_dir, cache_dir = sample
    # _write_full_cache re-fetches every surviving *and* new selected id (not
    # just the new one), so the survivor must be in the fetch table too.
    fetch_table = {
        "reserve-1": _text_row("reserve-1", course="Property"),
        "keep-2": _text_row("keep-2", course="Torts"),
    }
    monkeypatch.setattr(sq, "_fetch_rows", _stub_fetch_rows(fetch_table))

    sq.replace_question("tiny", "old-1", "reserve-1", "mismatched reference answer")

    manifest = json.loads((data_dir / "tiny.json").read_text(encoding="utf-8"))

    # Position preserved: reserve-1 takes old-1's old slot (index 0), keep-2 untouched at index 1.
    ids = [r["id"] for r in manifest["selected"]]
    assert ids == ["reserve-1", "keep-2"]
    assert manifest["selected"][1] == _row("keep-2", course="Torts")
    assert manifest["selected"][0]["course"] == "Property"

    # Reserve consumed, in order -- only reserve-1 removed, reserve-2 remains.
    assert [r["id"] for r in manifest["reserves"]] == ["reserve-2"]

    # Provenance recorded inside the manifest itself.
    assert manifest["replacements"] == [{
        "removed": "old-1", "substituted": "reserve-1",
        "position": 0, "reason": "mismatched reference answer",
    }]


def test_replace_regenerates_markdown_and_full_text_cache_together(sample, monkeypatch):
    data_dir, cache_dir = sample
    fetch_table = {
        "reserve-1": _text_row("reserve-1", course="Property"),
        "keep-2": _text_row("keep-2", course="Torts"),
    }
    monkeypatch.setattr(sq, "_fetch_rows", _stub_fetch_rows(fetch_table))

    sq.replace_question("tiny", "old-1", "reserve-1", "reason")

    md = (data_dir / "tiny.md").read_text(encoding="utf-8")
    assert "reserve-1" in md
    assert "old-1" not in md

    cache = json.loads((cache_dir / "tiny_full.json").read_text(encoding="utf-8"))
    assert [r["id"] for r in cache] == ["reserve-1", "keep-2"]  # manifest order, not fetch order
    assert all("question" in r and "answer" in r for r in cache)
    assert cache[0]["question"] == fetch_table["reserve-1"]["question"]


def test_replace_rejects_an_old_id_not_in_selected(sample, monkeypatch):
    monkeypatch.setattr(sq, "_fetch_rows", _poison_fetch)
    with pytest.raises(SystemExit, match="not in tiny's selected list"):
        sq.replace_question("tiny", "never-selected", "reserve-1", "reason")


def test_replace_rejects_a_new_id_not_in_reserves(sample, monkeypatch):
    monkeypatch.setattr(sq, "_fetch_rows", _poison_fetch)
    with pytest.raises(SystemExit, match="not in tiny's reserve list"):
        sq.replace_question("tiny", "old-1", "not-a-reserve", "reason")


def test_replace_propagates_failure_when_reserve_id_missing_at_pinned_revision(sample, monkeypatch):
    # Both ids are valid manifest entries, but the pinned-revision fetch itself
    # reports the reserve id as absent from the dataset (e.g. a corrupted or
    # stale manifest) -- _fetch_rows's own SystemExit must propagate unmodified.
    monkeypatch.setattr(sq, "_fetch_rows", _stub_fetch_rows({}))
    with pytest.raises(SystemExit, match="not found in"):
        sq.replace_question("tiny", "old-1", "reserve-1", "reason")


def test_replace_cannot_be_silently_double_applied(sample, monkeypatch):
    """old_id is removed from `selected` by the first call, so a second call
    with the same old_id has nothing to find -- accidental double-apply is a
    loud error, not a silent no-op or a second substitution."""
    fetch_table = {
        "reserve-1": _text_row("reserve-1", course="Property"),
        "reserve-2": _text_row("reserve-2"),
        "keep-2": _text_row("keep-2", course="Torts"),
    }
    monkeypatch.setattr(sq, "_fetch_rows", _stub_fetch_rows(fetch_table))

    sq.replace_question("tiny", "old-1", "reserve-1", "first reason")
    with pytest.raises(SystemExit, match="not in tiny's selected list"):
        sq.replace_question("tiny", "old-1", "reserve-2", "second reason")


def test_main_requires_reason_when_replace_is_given(monkeypatch):
    monkeypatch.setattr(
        sys, "argv",
        ["select_questions.py", "--name", "tiny", "--replace", "old-1", "reserve-1"],
    )
    monkeypatch.setattr(sq, "_fetch_rows", _poison_fetch)
    monkeypatch.setattr(sq, "replace_question", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("replace_question must not run without --reason")
    ))
    with pytest.raises(SystemExit):
        sq.main()
