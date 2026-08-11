import json

import pytest

from glassbox.dataset import Question, load_sample


def test_loads_twenty_dev_questions():
    questions = load_sample("dev_20")
    assert len(questions) == 20
    assert all(isinstance(q, Question) for q in questions)


def test_questions_carry_text_and_metadata():
    q = load_sample("dev_20")[0]
    assert len(q.question) > 100
    assert len(q.answer) > 100
    assert q.area in {"Public", "Private", "Criminal"}
    assert q.answer_words >= 150
    assert q.question_words >= 50


def test_order_is_stable_across_calls():
    assert [q.id for q in load_sample("dev_20")] == [q.id for q in load_sample("dev_20")]


def test_ids_match_the_committed_manifest():
    from glassbox.config import DATA_DIR

    manifest = json.loads((DATA_DIR / "dev_20.json").read_text(encoding="utf-8"))
    expected = [row["id"] for row in manifest["selected"]]
    assert [q.id for q in load_sample("dev_20")] == expected


def test_unknown_sample_raises():
    with pytest.raises(FileNotFoundError):
        load_sample("does_not_exist")


def test_load_sample_follows_manifest_order_even_when_cache_is_shuffled(tmp_path, monkeypatch):
    """Guards the load-bearing guarantee: order must come from the manifest.

    ``scripts/select_questions.py`` happens to write the manifest and cache in
    the same order, so a regression that switched ``load_sample`` to read the
    cache's row order instead of the manifest's ``selected`` order would pass
    every other test here. This test builds a synthetic manifest and a cache
    whose row order genuinely differs from it, so only a correct
    manifest-order implementation can pass.
    """
    from glassbox import dataset

    manifest_selected = [
        {
            "id": f"q{i}",
            "course": "Contracts",
            "area": "Private",
            "jurisdiction": "Domestic",
            "year": "2020",
            "question_words": 60,
            "answer_words": 160,
        }
        for i in range(4)
    ]
    manifest = {"name": "tiny", "selected": manifest_selected}

    def make_row(entry):
        return {**entry, "question": "word " * 60, "answer": "word " * 160}

    cache_rows = [make_row(entry) for entry in manifest_selected]
    shuffled_cache = [cache_rows[2], cache_rows[0], cache_rows[3], cache_rows[1]]

    manifest_ids = [entry["id"] for entry in manifest_selected]
    shuffled_ids = [row["id"] for row in shuffled_cache]
    assert manifest_ids != shuffled_ids, (
        "fixture is broken: cache order must genuinely differ from manifest "
        "order, otherwise this test cannot detect an order regression"
    )

    (tmp_path / "tiny.json").write_text(json.dumps(manifest), encoding="utf-8")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "tiny_full.json").write_text(json.dumps(shuffled_cache), encoding="utf-8")

    monkeypatch.setattr(dataset, "DATA_DIR", tmp_path)
    monkeypatch.setattr(dataset, "CACHE_DIR", cache_dir)

    loaded_ids = [q.id for q in dataset.load_sample("tiny")]
    assert loaded_ids == manifest_ids
    assert loaded_ids != shuffled_ids
