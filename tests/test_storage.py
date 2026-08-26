import json
import os
from pathlib import Path

from glassbox.recipes.base import RecipeResult
from glassbox.storage import load_results, save_result
from glassbox.usage import Usage

RESULT = RecipeResult(
    recipe="plain", question_id="q1", final_answer="an answer", sections=None,
    stages=[], usage=Usage(input_tokens=10, output_tokens=20, reasoning_tokens=5, calls=1),
    seconds=1.5, metadata={"model": "gpt-5-mini", "temperature": 0.7},
)


def test_round_trips_a_result(tmp_path):
    save_result(RESULT, tmp_path)
    loaded = load_results(tmp_path)
    assert len(loaded) == 1
    assert loaded[0] == RESULT


def test_save_result_publishes_via_a_temp_file_and_os_replace(tmp_path, monkeypatch):
    # A plain write_text leaves a truncated, unparseable file in place if the
    # process is killed mid-write -- the project's documented failure mode
    # (six runs lost to it). Writing to a temp file first and swapping it in
    # with os.replace (atomic on both POSIX and Windows) means a kill before
    # the replace leaves either the previous complete file or nothing at the
    # destination path -- never a partial one. Spy on os.replace (rather than
    # blocking it) so the real call still runs and the file actually lands.
    calls = []
    real_replace = os.replace

    def spy_replace(src, dst):
        calls.append((Path(src), Path(dst)))
        real_replace(src, dst)

    monkeypatch.setattr("glassbox.storage.os.replace", spy_replace)

    path = save_result(RESULT, tmp_path)

    assert calls, "save_result must publish via os.replace, not a direct write_text"
    tmp_src, dst = calls[0]
    assert dst == path
    assert tmp_src != path, "must write to a different temp path first"
    assert not tmp_src.exists(), "os.replace should have moved the temp file away"
    assert json.loads(path.read_text(encoding="utf-8"))["recipe"] == "plain"


def test_filename_includes_recipe_and_question_id(tmp_path):
    path = save_result(RESULT, tmp_path)
    assert "plain" in path.name and "q1" in path.name


def test_results_load_in_a_stable_order(tmp_path):
    for qid in ["q3", "q1", "q2"]:
        save_result(
            RecipeResult(recipe="plain", question_id=qid, final_answer="a", sections=None,
                         stages=[], usage=Usage.zero(), seconds=0.0, metadata={}),
            tmp_path,
        )
    assert [r.question_id for r in load_results(tmp_path)] == ["q1", "q2", "q3"]


def test_load_results_skips_grader_output_written_into_the_same_directory(tmp_path):
    # grade_runs.py globs the run directory with its own *.json output (e.g.
    # scores_provisional.json, lexam_scores.json) -- a list of row dicts, not a
    # RecipeResult. A second invocation of grade_runs.py must not crash trying
    # to parse its own prior output as a result.
    save_result(RESULT, tmp_path)
    (tmp_path / "scores_provisional.json").write_text(
        json.dumps([{"question_id": "q1", "recipe": "plain", "lexam_score": 0.5}]),
        encoding="utf-8",
    )
    loaded = load_results(tmp_path)
    assert len(loaded) == 1
    assert loaded[0] == RESULT
