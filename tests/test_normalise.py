from glassbox.grading.normalise import normalise
from glassbox.recipes.base import RecipeResult
from glassbox.usage import Usage


def _result(recipe, final_answer, sections=None):
    return RecipeResult(recipe=recipe, question_id="q1", final_answer=final_answer,
                        sections=sections, stages=[], usage=Usage.zero(), seconds=0.0,
                        metadata={})


def test_plain_answer_passes_through_with_whitespace_collapsed():
    assert normalise(_result("plain", "The shop  is\n\n\nliable.")) == "The shop is\n\nliable."


def test_final_answer_wins_over_sections_so_content_is_never_doubled():
    """A recipe carrying both is graded on its final answer, not both.

    CaseFile.to_sections() puts the full prose answer inside the sections dict, so
    joining the values would give schema-following recipes roughly double Recipe 1's
    graded content. See the comment in normalise().
    """
    out = normalise(_result("pipeline", "The shop is liable.", sections={
        "issues": "Duty of care.", "conclusion": "Liable.",
        "final_answer": "The shop is liable."}))
    assert out == "The shop is liable."
    assert "Duty of care." not in out


def test_sections_are_flattened_when_there_is_no_final_answer():
    out = normalise(_result("pipeline", "", sections={
        "issues": "Duty of care.", "conclusion": "Liable."}))
    assert "issues" not in out.lower()
    assert "Duty of care." in out and "Liable." in out


def test_identical_content_normalises_identically_even_when_one_carries_sections():
    """The comparison that the doubling bug would have broken."""
    plain = normalise(_result("plain", "The shop is liable."))
    pipeline = normalise(_result("pipeline", "The shop is liable.", sections={
        "issues": "Duty of care.", "rules": "Occupiers owe a duty.",
        "final_answer": "The shop is liable."}))
    assert plain == pipeline


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
