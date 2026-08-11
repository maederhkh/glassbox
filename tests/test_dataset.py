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
