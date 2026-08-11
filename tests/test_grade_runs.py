from scripts.grade_runs import find_missing_questions, find_orphaned_results


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
