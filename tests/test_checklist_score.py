from glassbox.grading.checklist import (
    Checklist, ChecklistPoint, coverage_fraction, score_checklist,
)
from glassbox.llm import FakeLLMClient

CHECKLIST = Checklist(question_id="q1", source="model_answer", points=[
    ChecklistPoint(id="q1-p1", text="Shops owe a duty of care."),
    ChecklistPoint(id="q1-p2", text="No warning sign breaches that duty."),
    ChecklistPoint(id="q1-p3", text="The breach caused the injury."),
])

RESPONSE = """{"verdicts": [
  {"point_id": "q1-p1", "coverage": "covered", "evidence": "shops owe a duty"},
  {"point_id": "q1-p2", "coverage": "partial", "evidence": "mentions the sign"},
  {"point_id": "q1-p3", "coverage": "missed", "evidence": ""}],
 "contradictions": ["says duty is owed only to employees"],
 "inventions": ["cites a non-existent Occupiers Act 1998"]}"""


def test_returns_one_verdict_per_point():
    v = score_checklist(CHECKLIST, "an answer", FakeLLMClient([RESPONSE]))
    assert [x.point_id for x in v.verdicts] == ["q1-p1", "q1-p2", "q1-p3"]
    assert [x.coverage for x in v.verdicts] == ["covered", "partial", "missed"]


def test_captures_contradictions_and_inventions():
    v = score_checklist(CHECKLIST, "an answer", FakeLLMClient([RESPONSE]))
    assert len(v.contradictions) == 1
    assert len(v.inventions) == 1


def test_coverage_fraction_counts_partial_as_half():
    v = score_checklist(CHECKLIST, "an answer", FakeLLMClient([RESPONSE]))
    assert coverage_fraction(v) == (1.0 + 0.5 + 0.0) / 3


def test_missing_points_in_judge_output_default_to_missed():
    sparse = '{"verdicts": [{"point_id": "q1-p1", "coverage": "covered", "evidence": "x"}], \
"contradictions": [], "inventions": []}'
    v = score_checklist(CHECKLIST, "an answer", FakeLLMClient([sparse]))
    assert len(v.verdicts) == 3
    assert [x.coverage for x in v.verdicts] == ["covered", "missed", "missed"]


def test_prompt_lists_every_point_and_the_answer():
    client = FakeLLMClient([RESPONSE])
    score_checklist(CHECKLIST, "THE ANSWER TEXT", client)
    prompt = client.prompts[0]
    assert "q1-p1" in prompt and "q1-p3" in prompt
    assert "THE ANSWER TEXT" in prompt
