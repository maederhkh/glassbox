from glassbox.dataset import Question
from glassbox.grading.checklist import (
    Checklist, ChecklistPoint, build_checklist, load_checklist, save_checklist,
)
from glassbox.llm import FakeLLMClient
from scripts.build_checklists import run_build

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


DIGIT_POLLUTED_RESPONSE = """{"source": "mark_annotated", "points": [
  {"text": "Shops owe customers a duty of care."},
  {"text": "1 1"},
  {"text": "The breach caused the injury."}]}"""


def test_drops_stray_digit_points_and_keeps_surviving_ids_contiguous():
    checklist = build_checklist(QUESTION, FakeLLMClient([DIGIT_POLLUTED_RESPONSE]))
    assert [p.text for p in checklist.points] == [
        "Shops owe customers a duty of care.",
        "The breach caused the injury.",
    ]
    assert [p.id for p in checklist.points] == ["q1-p1", "q1-p2"]


def test_mark_annotated_source_round_trips_through_disk(tmp_path):
    checklist = build_checklist(QUESTION, FakeLLMClient([DIGIT_POLLUTED_RESPONSE]))
    assert checklist.source == "mark_annotated"
    save_checklist(checklist, tmp_path)
    assert load_checklist("q1", tmp_path) == checklist


EXISTING_CHECKLIST = Checklist(
    question_id="q1", source="marking_scheme",
    points=[ChecklistPoint(id="q1-p1", text="Pre-existing point from a prior run.")],
)


def test_run_build_skips_a_question_that_already_has_a_checklist_on_disk(tmp_path):
    save_checklist(EXISTING_CHECKLIST, tmp_path)

    # No responses queued: a call to the client would raise AssertionError,
    # so getting a result at all proves the question was skipped, not rebuilt.
    counts, sources, built = run_build([QUESTION], FakeLLMClient([]), tmp_path)

    assert built == 0
    assert counts == [1]
    assert sources == ["marking_scheme"]
    assert load_checklist("q1", tmp_path) == EXISTING_CHECKLIST


def test_run_build_builds_a_question_with_no_checklist_on_disk(tmp_path):
    client = FakeLLMClient([RESPONSE])

    counts, sources, built = run_build([QUESTION], client, tmp_path)

    assert built == 1
    assert counts == [3]
    assert sources == ["model_answer"]
    assert load_checklist("q1", tmp_path).points[0].text == (
        "Shops owe customers a duty of care."
    )


def test_run_build_rebuild_true_overwrites_rather_than_skips(tmp_path):
    save_checklist(EXISTING_CHECKLIST, tmp_path)
    client = FakeLLMClient([RESPONSE])

    counts, sources, built = run_build([QUESTION], client, tmp_path, rebuild=True)

    assert built == 1
    assert counts == [3]
    assert sources == ["model_answer"]
    rebuilt = load_checklist("q1", tmp_path)
    assert rebuilt != EXISTING_CHECKLIST
    assert len(rebuilt.points) == 3
