import json

from scripts.grade_runs import (
    find_missing_questions, find_orphaned_results, load_existing_scores,
)


class _Result:
    def __init__(self, question_id):
        self.question_id = question_id


def test_find_orphaned_results_flags_ids_not_in_the_sample():
    results = [_Result("in-sample"), _Result("stale-id")]
    questions = {"in-sample": object()}
    assert find_orphaned_results(results, questions) == ["stale-id"]


def test_find_orphaned_results_is_empty_when_every_result_is_in_the_sample():
    results = [_Result("a"), _Result("b")]
    questions = {"a": object(), "b": object()}
    assert find_orphaned_results(results, questions) == []


def test_find_missing_questions_flags_sample_ids_with_no_result():
    results = [_Result("a")]
    questions = {"a": object(), "b": object(), "c": object()}
    assert find_missing_questions(results, questions) == ["b", "c"]


def test_find_missing_questions_is_empty_when_every_question_has_a_result():
    results = [_Result("a"), _Result("b")]
    questions = {"a": object(), "b": object()}
    assert find_missing_questions(results, questions) == []


def test_load_existing_scores_is_empty_when_the_file_does_not_exist(tmp_path):
    assert load_existing_scores(tmp_path / "scores_provisional.json") == {}


def test_load_existing_scores_keys_rows_by_question_id_and_recipe(tmp_path):
    path = tmp_path / "scores_provisional.json"
    rows = [{"question_id": "q1", "recipe": "plain", "lexam_score": 0.5}]
    path.write_text(json.dumps(rows), encoding="utf-8")

    existing = load_existing_scores(path)

    assert existing == {("q1", "plain"): rows[0]}
