from glassbox.grading.subscores import SubScores, score_subscores
from glassbox.llm import FakeLLMClient

RESPONSE = """{"issue_spotting": 0.8, "rule_recall": 0.6,
"rule_application": 0.4, "conclusion": 0.7}"""


def test_returns_all_four_criteria():
    s = score_subscores("the question", "the reference", "the answer",
                        FakeLLMClient([RESPONSE]))
    assert s == SubScores(issue_spotting=0.8, rule_recall=0.6,
                          rule_application=0.4, conclusion=0.7)


def test_clamps_out_of_range_values():
    client = FakeLLMClient(['{"issue_spotting": 1.9, "rule_recall": -0.3, '
                            '"rule_application": 0.5, "conclusion": 0.5}'])
    s = score_subscores("q", "r", "a", client)
    assert s.issue_spotting == 1.0
    assert s.rule_recall == 0.0
