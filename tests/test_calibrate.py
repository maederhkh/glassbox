from glassbox.llm import FakeLLMClient

from scripts.calibrate import (
    check_empty_scores_low, check_reference_scores_high, run_reference_check_detailed,
)


class _Q:
    def __init__(self, qid, question, answer):
        self.id, self.question, self.answer = qid, question, answer


QUESTIONS = [_Q("q1", "question one", "reference one"),
             _Q("q2", "question two", "reference two")]


def test_reference_check_passes_when_judge_scores_high():
    clients = [FakeLLMClient(["[[1.0]]", "[[0.9]]"])]
    passed, mean, _ = check_reference_scores_high(QUESTIONS, clients, threshold=0.8)
    assert passed is True
    assert mean == 0.95


def test_reference_check_fails_when_judge_scores_low():
    clients = [FakeLLMClient(["[[0.2]]", "[[0.3]]"])]
    passed, mean, _ = check_reference_scores_high(QUESTIONS, clients, threshold=0.8)
    assert passed is False
    assert mean == 0.25


def test_empty_check_passes_when_judge_scores_low():
    clients = [FakeLLMClient(["[[0.0]]", "[[0.1]]"])]
    passed, mean, _ = check_empty_scores_low(QUESTIONS, clients, threshold=0.2)
    assert passed is True


def test_empty_check_fails_when_judge_is_too_generous():
    clients = [FakeLLMClient(["[[0.6]]", "[[0.7]]"])]
    passed, mean, _ = check_empty_scores_low(QUESTIONS, clients, threshold=0.2)
    assert passed is False


def test_detailed_check_keeps_per_judge_scores_alongside_the_ensemble():
    clients = [FakeLLMClient(["[[0.9]]", "[[0.2]]"]), FakeLLMClient(["[[0.4]]", "[[0.3]]"])]
    rows = run_reference_check_detailed(QUESTIONS, clients)
    assert rows == [
        {"question_id": "q1", "score": 0.4, "per_judge": [0.9, 0.4]},
        {"question_id": "q2", "score": 0.2, "per_judge": [0.2, 0.3]},
    ]


def test_detailed_check_preserves_none_for_an_unparseable_judge():
    clients = [FakeLLMClient(["not a score", "[[0.5]]"])]
    rows = run_reference_check_detailed(QUESTIONS[:1], clients)
    assert rows[0]["per_judge"] == [None]
    assert rows[0]["score"] is None
