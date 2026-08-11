from glassbox.dataset import Question
from glassbox.grading.checklist import (
    Checklist, build_checklist, load_checklist, save_checklist,
)
from glassbox.llm import FakeLLMClient

QUESTION = Question(
    id="q1", question="Is the shop liable?", answer="Shops owe a duty of care. "
    "Failing to display a warning sign breaches it. The breach caused the injury.",
    course="Tort Law", area="Private", jurisdiction="Generic", year="2022",
    question_words=60, answer_words=200,
)

RESPONSE = """{"source": "model_answer", "points": [
  {"text": "Shops owe customers a duty of care."},
  {"text": "Failing to display a warning sign breaches that duty."},
  {"text": "The breach caused the injury."}]}"""


def test_builds_one_point_per_proposition():
    checklist = build_checklist(QUESTION, FakeLLMClient([RESPONSE]))
    assert isinstance(checklist, Checklist)
    assert len(checklist.points) == 3
    assert checklist.points[0].text == "Shops owe customers a duty of care."


def test_point_ids_are_stable_and_prefixed_by_question():
    checklist = build_checklist(QUESTION, FakeLLMClient([RESPONSE]))
    assert [p.id for p in checklist.points] == ["q1-p1", "q1-p2", "q1-p3"]


def test_records_whether_reference_was_model_answer_or_marking_scheme():
    checklist = build_checklist(QUESTION, FakeLLMClient([RESPONSE]))
    assert checklist.source == "model_answer"


def test_tolerates_json_wrapped_in_a_code_fence():
    fenced = f"Here you go:\n```json\n{RESPONSE}\n```"
    assert len(build_checklist(QUESTION, FakeLLMClient([fenced])).points) == 3


def test_prompt_contains_reference_answer():
    client = FakeLLMClient([RESPONSE])
    build_checklist(QUESTION, client)
    assert "Shops owe a duty of care" in client.prompts[0]


def test_round_trips_through_disk(tmp_path):
    checklist = build_checklist(QUESTION, FakeLLMClient([RESPONSE]))
    save_checklist(checklist, tmp_path)
    assert load_checklist("q1", tmp_path) == checklist
