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
