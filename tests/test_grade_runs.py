import json

from glassbox.grading.checklist import Checklist, ChecklistPoint, save_checklist
from glassbox.llm import FakeLLMClient
from glassbox.usage import Usage
from scripts.grade_runs import (
    find_missing_questions, find_orphaned_results, load_existing_scores, run_scoring,
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


class _FakeResult:
    def __init__(self, question_id):
        self.question_id = question_id
        self.recipe = "plain"
        self.final_answer = "an answer"
        self.sections = None
        self.usage = Usage(input_tokens=1, output_tokens=1, reasoning_tokens=0, calls=1)


class _FakeQuestion:
    def __init__(self, id_):
        self.id = id_
        self.question = "Q?"
        self.answer = "A."


def _checklist_response(point_id: str) -> str:
    return (f'{{"verdicts": [{{"point_id": "{point_id}", "coverage": "covered", '
            f'"evidence": "x"}}], "contradictions": [], "inventions": []}}')


def test_run_scoring_never_drops_an_already_scored_row_that_sorts_later(tmp_path):
    """Reproduces the exact scenario find_orphaned_results' own error message
    walks a user into: an already-scored row on disk, plus a new, unscored
    result whose key sorts earlier (e.g. a backfilled replacement id). The
    old code persisted only "rows so far" (a prefix of iteration order)
    after scoring the early item, which overwrote the file down to just that
    one row -- dropping the already-scored, later-sorting row from disk
    until a further write happened to include it again. A kill in that
    window loses a row that real money already paid for.

    This test has no natural place to "pause" the loop from the outside, so
    it hooks the moment right after that: the third item's (z-third) first
    client call happens only after a-early has been scored and written AND
    m-existing's skip step has run, so reading the file from inside that
    call captures exactly the window the old code lost m-existing's row in.
    """
    out_path = tmp_path / "scores_provisional.json"
    existing_row = {
        "question_id": "m-existing", "recipe": "plain", "lexam_score": 0.6,
        "per_judge": [0.6, 0.6, 0.6], "judges": ["j1", "j2", "j3"],
        "issue_spotting": 0.5, "rule_recall": 0.5, "rule_application": 0.5,
        "conclusion": 0.5, "coverage": 0.5, "points_total": 1,
        "points_covered": 1, "contradictions": 0, "inventions": 0,
        "total_tokens": 42,
    }
    out_path.write_text(json.dumps([existing_row]), encoding="utf-8")

    checklist_dir = tmp_path / "checklists"
    for qid in ["a-early", "z-third"]:
        save_checklist(
            Checklist(question_id=qid, source="model_answer",
                      points=[ChecklistPoint(id=f"{qid}-p1", text="A point.")]),
            checklist_dir,
        )

    results = [_FakeResult("a-early"), _FakeResult("m-existing"), _FakeResult("z-third")]
    questions = {"a-early": _FakeQuestion("a-early"), "z-third": _FakeQuestion("z-third")}

    judge_resp = "The correctness score: [[0.7]]"
    subs_resp = ('{"issue_spotting": 0.6, "rule_recall": 0.6, '
                 '"rule_application": 0.6, "conclusion": 0.6}')

    # Each client is reused across BOTH new items (a-early, then z-third), in
    # that order -- matching how main() builds one client per role and passes
    # the same instances into every score_one call.
    judges = [FakeLLMClient([judge_resp, judge_resp]) for _ in range(3)]
    grader = FakeLLMClient([
        subs_resp, _checklist_response("a-early-p1"),
        subs_resp, _checklist_response("z-third-p1"),
    ])

    disk_state_at_third_items_start = {}

    class _DiskCheckingJudge:
        def __init__(self, inner):
            self._inner = inner
            self._n = 0

        def complete(self, prompt, system=None):
            self._n += 1
            if self._n == 2:  # z-third's turn; a-early is already scored+written
                on_disk = json.loads(out_path.read_text(encoding="utf-8"))
                disk_state_at_third_items_start["ids"] = {
                    (row["question_id"], row["recipe"]) for row in on_disk
                }
            return self._inner.complete(prompt, system=system)

    judges[0] = _DiskCheckingJudge(judges[0])

    run_scoring(results, questions, judges, grader, checklist_dir, out_path)

    assert ("m-existing", "plain") in disk_state_at_third_items_start["ids"], (
        "the already-scored row was missing from disk while the next result "
        "was being scored -- a kill in that window would have lost it for good"
    )
