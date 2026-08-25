from glassbox.grading.normalise import normalise
from glassbox.recipes.base import RecipeResult
from glassbox.usage import Usage


def _result(recipe, final_answer, sections=None):
    return RecipeResult(recipe=recipe, question_id="q1", final_answer=final_answer,
                        sections=sections, stages=[], usage=Usage.zero(), seconds=0.0,
                        metadata={})


def test_plain_answer_passes_through_with_whitespace_collapsed():
    assert normalise(_result("plain", "The shop  is\n\n\nliable.")) == "The shop is\n\nliable."


def test_sectioned_answer_is_flattened_to_prose_without_section_headings():
    out = normalise(_result("pipeline", "final text", sections={
        "issues": "Duty of care.", "conclusion": "Liable."}))
    assert "issues" not in out.lower()
    assert "Duty of care." in out and "Liable." in out


def test_recipe_name_never_leaks_into_normalised_text():
    for recipe in ["plain", "structured", "think_longer", "pipeline"]:
        assert recipe not in normalise(_result(recipe, "an answer")).lower()


def test_stage_and_json_artefacts_are_stripped():
    text = '```json\n{"issues": ["a"]}\n```\nStage 1 output: The shop is liable.'
    out = normalise(_result("pipeline", text))
    assert "```" not in out
    assert "stage 1" not in out.lower()


def test_identical_content_normalises_identically_across_recipes():
    a = normalise(_result("plain", "The shop is liable."))
    b = normalise(_result("pipeline", "The shop is liable."))
    assert a == b
