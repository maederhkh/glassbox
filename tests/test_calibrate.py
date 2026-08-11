from glassbox.llm import FakeLLMClient

from scripts.calibrate import check_empty_scores_low, check_reference_scores_high


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
