from glassbox.grading.lexam_judge import (
    JudgeVerdict, ensemble_min, judge_answer, judge_once, parse_score,
)
from glassbox.llm import FakeLLMClient


def test_parses_a_well_formed_score():
    assert parse_score("Good reasoning. The correctness score: [[0.7]]") == 0.7


def test_parses_the_last_score_when_several_appear():
    assert parse_score("maybe [[0.3]] but on reflection [[0.8]]") == 0.8


def test_returns_none_when_no_score_present():
    assert parse_score("This answer is quite good overall.") is None


def test_clamps_out_of_range_scores_to_zero():
    assert parse_score("score: [[9.9]]") == 0.0


def test_ensemble_takes_the_minimum():
    assert ensemble_min([0.8, 0.4, 0.6]) == 0.4


def test_ensemble_ignores_unparseable_judges():
    assert ensemble_min([0.8, None, 0.6]) == 0.6


def test_ensemble_is_none_when_every_judge_failed():
    assert ensemble_min([None, None]) is None


def test_judge_once_returns_score_and_full_text():
    client = FakeLLMClient(["Reasoning here. The correctness score: [[0.6]]"])
    score, text = judge_once(client, "the question", "the reference", "the candidate")
    assert score == 0.6
    assert "correctness score" in text


def test_judge_prompt_contains_all_three_inputs():
    client = FakeLLMClient(["[[0.5]]"])
    judge_once(client, "QQQ", "RRR", "CCC")
    prompt = client.prompts[0]
    assert "QQQ" in prompt and "RRR" in prompt and "CCC" in prompt


def test_judge_answer_across_multiple_judges_takes_the_minimum():
    clients = [FakeLLMClient(["[[0.9]]"]), FakeLLMClient(["[[0.4]]"])]
    verdict = judge_answer(clients, "q", "r", "c")
    assert isinstance(verdict, JudgeVerdict)
    assert verdict.score == 0.4
    assert verdict.per_judge == [0.9, 0.4]
