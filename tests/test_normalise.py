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


# --- C1 regression fixtures: real on-disk output, not imagined output. ---
#
# Reproducing normalise() over all 60 dev_20 results on disk showed 15/20
# think_longer answers carry "Step N — Heading" scaffolding straight through
# to the grader (0/20 in plain and structured) because GPT-5-mini writes the
# separator as an em dash (U+2014), which the original ASCII-only [:.-] class
# never matched. Checking _SECTION_HEADING against the same 60 answers found it
# never fired at all: the model never writes markdown '#' headings, it writes
# bare/bulleted/numbered section names instead.


def test_stage_marker_strips_em_dash_step_headings_gpt5_mini_actually_writes():
    """Real shape from think_longer, question 205b9732."""
    text = (
        "Step 1 — Issues identified\n"
        "I identify the following principal strategic legal issues the facts raise:\n"
        "1. Choice of legal vehicle (foundation v alternatives): HJS asks whether to "
        "create a foundation but must weigh alternatives.\n"
        "Step 2 — Governing rules and elements (summary)\n"
        "2. Legal requirements to establish a Swiss foundation: Swiss law requires a "
        "lawful, determinate purpose."
    )
    out = normalise(_result("think_longer", text))
    assert "step 1" not in out.lower()
    assert "step 2" not in out.lower()
    assert "Choice of legal vehicle" in out
    assert "Legal requirements to establish a Swiss foundation" in out


def test_stage_marker_strips_other_dashes_in_the_same_unicode_block():
    """Em dash is what this model happened to write; the class widens to the whole
    general-punctuation dash block (U+2010-U+2015) so a differently generated
    hyphen/en dash doesn't reopen the same leak."""
    for dash in ("‐", "‑", "‒", "–", "—", "―"):
        text = f"Step 1 {dash} Issues identified\nThe shop is liable."
        out = normalise(_result("think_longer", text))
        assert "step 1" not in out.lower(), f"leaked for dash {dash!r}"
        assert "The shop is liable." in out


def test_stage_marker_does_not_strip_prose_that_merely_begins_with_the_word_step():
    """Boundary: real legal writing can start a sentence with 'Step N' as prose,
    with no separator punctuation at all. That must survive untouched."""
    text = "Step 1 of the analysis is straightforward and does not require further comment."
    assert normalise(_result("plain", text)) == text


def test_section_heading_strips_bare_word_heading_with_no_markdown_hash():
    """Real shape from plain, question 104d4012: a bare 'Issue' heading line, no
    markdown '#' -- all the original regex recognised."""
    text = (
        "Issue\n"
        "1. Whether the contractual provisions in the Joint Production Agreement "
        "(JPA) amount to an agreement prohibited by Article 101(1) TFEU."
    )
    out = normalise(_result("plain", text))
    assert out.splitlines()[0] != "Issue"
    assert "Whether the contractual provisions" in out


def test_section_heading_strips_bulleted_colon_form_but_not_inline_content():
    """Real shape from plain, question 205b9732: '- Application:' stands alone as
    a heading; '- Rule/principle: ...' on the line before is an inline
    heading+content line and must survive, since it is not a bare heading."""
    text = (
        "- Rule/principle: ZGB requires a defined lawful purpose.\n"
        "- Application:\n"
        "  - HJS's stated aim: fund basic research."
    )
    out = normalise(_result("plain", text))
    assert "- Application:" not in out
    assert "- Rule/principle: ZGB requires a defined lawful purpose." in out
    assert "HJS's stated aim: fund basic research." in out


def test_section_heading_strips_numbered_bare_headings_but_not_numbered_prose():
    """Real shape from plain, question 74b136af: a 'Structure' outline numbers bare
    headings ('1. Issues', '4. Conclusion') alongside numbered items that carry
    real descriptive text ('2. Applicable rules and theories (...)'), which must
    NOT be stripped -- they are not bare heading-only lines."""
    text = (
        "Structure\n"
        "1. Issues\n"
        "2. Applicable rules and theories (corporate theories; fiduciary duties)\n"
        "3. Application to facts (who the BoD owes duties to)\n"
        "4. Conclusion"
    )
    out = normalise(_result("plain", text))
    assert "1. Issues" not in out
    assert "4. Conclusion" not in out
    assert "2. Applicable rules and theories (corporate theories; fiduciary duties)" in out
    assert "3. Application to facts (who the BoD owes duties to)" in out


def test_stage_marker_stripping_does_not_unmask_a_bare_heading_underneath():
    """Real shape from think_longer, questions a9c68c6b and ba486dee: when the
    heading word is the *entire* remainder of a 'Step N — ' line (no extra
    descriptive words), stripping the stage prefix must not leave a now-bare
    'Issues' / 'Conclusion' / 'Overall conclusion' line behind for the grader."""
    text = (
        "Step 1 — Issues\n"
        "1. Whether the agreement is a contract of sale of goods within the CISG.\n"
        "Step 4 — Overall conclusion\n"
        "On the facts provided the CISG governs the contract."
    )
    out = normalise(_result("think_longer", text))
    assert out.splitlines()[0] != "Issues"
    assert "Overall conclusion" not in out
    assert "Whether the agreement is a contract of sale of goods" in out
    assert "On the facts provided the CISG governs the contract." in out


def test_section_heading_strips_all_caps_overall_conclusion():
    """Real shape from think_longer, question 104d4012: 'OVERALL CONCLUSION' in
    caps, no markdown hash, no colon."""
    text = (
        "This does not change the result.\n\n"
        "OVERALL CONCLUSION\n"
        "W cannot lawfully demand payment of the contractual penalty from S."
    )
    out = normalise(_result("think_longer", text))
    assert "OVERALL CONCLUSION" not in out
    assert "W cannot lawfully demand payment of the contractual penalty from S." in out
