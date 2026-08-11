from glassbox.grading.lexam_judge import (
    JudgeVerdict, ensemble_min, judge_answer, judge_once, parse_score, unparseable_counts,
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


def test_unparseable_counts_tallies_none_per_judge_by_position():
    rows = [
        {"question_id": "q1", "score": 0.4, "per_judge": [0.9, None]},
        {"question_id": "q2", "score": None, "per_judge": [None, None]},
    ]
    assert unparseable_counts(rows, ["judge-a", "judge-b"]) == {"judge-a": 1, "judge-b": 2}


def test_unparseable_counts_is_zero_when_every_judge_parses():
    rows = [{"question_id": "q1", "score": 0.5, "per_judge": [0.5, 0.5]}]
    assert unparseable_counts(rows, ["judge-a", "judge-b"]) == {"judge-a": 0, "judge-b": 0}


# --- Deliberate contract boundary, not a bug --------------------------------
#
# SCORE_PATTERN (`\[\[(\d\.\d)\]\]`) matches only a single digit, a literal
# dot, and a single digit. That means well-intentioned scores a judge model
# might plausibly emit -- a bare "[[1]]" for a perfect score, an unclamped
# two-digit "[[10]]", or a finer-grained "[[0.75]]" -- do NOT match and parse
# as None, exactly like genuinely unparseable text.
#
# This is LEXam's own published regex, reproduced verbatim for leaderboard
# comparability (see the module docstring and JUDGE_TEMPERATURE commentary in
# config.py). Loosening it to accept "[[1]]" or "[[0.75]]" would silently
# diverge from the protocol this study claims to follow -- a worse outcome
# than the narrow parsing it causes. So the fix here is not to the regex; it
# is to pin this boundary with tests, so a future reader who trips over
# `parse_score("[[1]]") is None` finds a deliberate, documented contract
# instead of mistaking it for an oversight and "fixing" it.
#
# The operational consequence -- a judge dropped this way vanishes from
# `ensemble_min`'s input entirely, which can bias the ensemble score upward
# if the dropped judge happened to be the strict one -- is not silent at the
# JudgeVerdict level: it is always visible per-judge in
# `JudgeVerdict.per_judge` (a None in that list). Surfacing per-judge None
# rates in run-level reporting is a follow-up for the task that builds the
# score summary, not this module.


def test_bare_integer_score_is_not_parsed():
    assert parse_score("The correctness score: [[1]]") is None


def test_finer_grained_decimal_score_is_not_parsed():
    assert parse_score("The correctness score: [[0.75]]") is None


def test_two_digit_score_is_not_parsed():
    assert parse_score("The correctness score: [[10]]") is None
