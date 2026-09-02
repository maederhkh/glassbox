"""Recipe 4's stage-runner and Stage 1.

Only Stage 1 exists yet, so the recipe is deliberately not registered in
scripts/run_recipe.py: a one-stage result written into a run directory would
look like a pipeline answer and would be graded as one.

The case-file relay is the mechanism the whole recipe rests on, so it is tested
here with a synthetic second stage rather than waiting for the real Stage 2.
"""

import pytest

from glassbox.dataset import Question
from glassbox.llm import FakeLLMClient
from glassbox.recipes import steps
from glassbox.recipes.pipeline import STAGE_ISSUES, Stage, PipelineRecipe

QUESTION = Question(
    id="q1",
    question="Cinestar signed a five-year exclusive supply contract. Is it valid?",
    answer="REFERENCE-ANSWER-SENTINEL: the contract is void.",
    course="European Economic Law", area="Public", jurisdiction="International",
    year="2022", question_words=60, answer_words=200,
)

ISSUES_JSON = (
    '{"issues": [{"id": "i1", "statement": "Whether the exclusivity clause '
    'restricts competition", "why_it_arises": "The contract binds one supplier '
    'for five years."}]}'
)


def test_stage_one_makes_exactly_one_call():
    client = FakeLLMClient([ISSUES_JSON])
    PipelineRecipe(stages=(STAGE_ISSUES,)).run(QUESTION, client)
    assert len(client.prompts) == 1


def test_stage_one_asks_only_for_issues():
    client = FakeLLMClient([ISSUES_JSON])
    PipelineRecipe(stages=(STAGE_ISSUES,)).run(QUESTION, client)
    prompt = client.prompts[0]
    assert steps.STEP_ISSUES in prompt
    # The point of decomposing: this call must not also ask for the other three.
    assert steps.STEP_RULES not in prompt
    assert steps.STEP_APPLICATION not in prompt
    assert steps.STEP_CONCLUSION not in prompt


def test_the_reference_answer_never_reaches_a_stage_prompt():
    client = FakeLLMClient([ISSUES_JSON])
    PipelineRecipe(stages=(STAGE_ISSUES,)).run(QUESTION, client)
    assert QUESTION.question in client.prompts[0]
    assert "REFERENCE-ANSWER-SENTINEL" not in client.prompts[0]


def test_each_call_is_recorded_as_its_own_stage():
    result = PipelineRecipe(stages=(STAGE_ISSUES,)).run(QUESTION, FakeLLMClient([ISSUES_JSON]))
    assert [s.name for s in result.stages] == ["issues"]
    assert result.stages[0].usage.calls == 1
    assert ISSUES_JSON in result.stages[0].output


def test_only_the_issues_section_is_filled_so_far():
    result = PipelineRecipe(stages=(STAGE_ISSUES,)).run(QUESTION, FakeLLMClient([ISSUES_JSON]))
    assert set(result.sections) == {"issues"}
    assert "exclusivity clause" in result.sections["issues"]


def test_final_answer_is_empty_until_the_last_stage_writes_one():
    # normalise() prefers a non-empty final_answer and falls back to the
    # sections join, so an empty string here routes a partial run correctly
    # rather than passing off an issues list as an answer.
    result = PipelineRecipe(stages=(STAGE_ISSUES,)).run(QUESTION, FakeLLMClient([ISSUES_JSON]))
    assert result.final_answer == ""


def test_a_later_stage_receives_the_committed_output_of_an_earlier_one():
    # The case-file relay, proven before Stage 2 exists. A synthetic second
    # stage must see Stage 1's parsed issues in its prompt.
    second = Stage(
        name="echo", instruction="Restate the issues.",
        json_instructions='{"conclusion": "..."}', fields=("conclusion",),
    )
    client = FakeLLMClient([ISSUES_JSON, '{"conclusion": "Restated."}'])
    PipelineRecipe(stages=(STAGE_ISSUES, second)).run(QUESTION, client)

    assert len(client.prompts) == 2
    assert "exclusivity clause" in client.prompts[1]


def test_a_later_stage_does_not_receive_the_raw_text_of_an_earlier_one():
    # Spec 6.2: a stage sees committed sections, never another stage's raw
    # model output. Here the raw output carries JSON scaffolding the rendered
    # section does not.
    second = Stage(
        name="echo", instruction="Restate the issues.",
        json_instructions='{"conclusion": "..."}', fields=("conclusion",),
    )
    client = FakeLLMClient([ISSUES_JSON, '{"conclusion": "Restated."}'])
    PipelineRecipe(stages=(STAGE_ISSUES, second)).run(QUESTION, client)

    assert '"why_it_arises"' not in client.prompts[1]


def test_unparseable_stage_output_is_reported_not_swallowed():
    with pytest.raises(ValueError):
        PipelineRecipe(stages=(STAGE_ISSUES,)).run(QUESTION, FakeLLMClient(["not json at all"]))
# --- Day 3: Stages 2 and 3 -------------------------------------------------

RULES_JSON = (
    '{"rules": [{"issue_id": "i1", "rule": "Article 101(1) TFEU prohibits agreements '
    'which restrict competition", "elements": ["an agreement between undertakings", '
    '"an appreciable restriction of competition"]}]}'
)

FINDINGS_JSON = (
    '{"findings": [{"issue_id": "i1", "element": "an agreement between undertakings", '
    '"holds": "yes", "reasoning": "Cinestar and the supplier signed a written contract."}]}'
)

THREE_STAGE_RESPONSES = [ISSUES_JSON, RULES_JSON, FINDINGS_JSON]

# These pin stages 2 and 3, so they name their three stages explicitly rather
# than relying on STAGES holding exactly three -- which stopped being true the
# day Stage 4 landed.
FIRST_THREE = PipelineRecipe.STAGES[:3]



def test_three_stages_make_three_calls_in_order():
    client = FakeLLMClient(THREE_STAGE_RESPONSES)
    result = PipelineRecipe(stages=FIRST_THREE).run(QUESTION, client)
    assert len(client.prompts) == 3
    assert [s.name for s in result.stages] == ["issues", "rules", "application"]


def test_stage_two_asks_only_for_rules():
    client = FakeLLMClient(THREE_STAGE_RESPONSES)
    PipelineRecipe(stages=FIRST_THREE).run(QUESTION, client)
    prompt = client.prompts[1]
    assert steps.STEP_RULES in prompt
    assert steps.STEP_ISSUES not in prompt
    assert steps.STEP_APPLICATION not in prompt
    assert steps.STEP_CONCLUSION not in prompt


def test_stage_three_asks_only_for_application():
    client = FakeLLMClient(THREE_STAGE_RESPONSES)
    PipelineRecipe(stages=FIRST_THREE).run(QUESTION, client)
    prompt = client.prompts[2]
    assert steps.STEP_APPLICATION in prompt
    assert steps.STEP_ISSUES not in prompt
    assert steps.STEP_RULES not in prompt
    assert steps.STEP_CONCLUSION not in prompt


def test_stage_two_receives_stage_ones_issues():
    client = FakeLLMClient(THREE_STAGE_RESPONSES)
    PipelineRecipe(stages=FIRST_THREE).run(QUESTION, client)
    assert "exclusivity clause" in client.prompts[1]


def test_stage_three_receives_everything_committed_before_it():
    # The relay accumulates: Stage 3 cannot apply elements to facts unless it
    # can see both the issues and the elements Stage 2 produced.
    client = FakeLLMClient(THREE_STAGE_RESPONSES)
    PipelineRecipe(stages=FIRST_THREE).run(QUESTION, client)
    prompt = client.prompts[2]
    assert "exclusivity clause" in prompt
    assert "Article 101(1) TFEU" in prompt
    assert "an appreciable restriction of competition" in prompt


def test_earlier_sections_survive_to_the_end_of_the_run():
    # Each stage returns only its own section, so the case file must accumulate
    # across stages rather than being replaced by the latest one.
    result = PipelineRecipe(stages=FIRST_THREE).run(QUESTION, FakeLLMClient(THREE_STAGE_RESPONSES))
    assert set(result.sections) == {"issues", "rules", "application"}
    assert "exclusivity clause" in result.sections["issues"]
    assert "Article 101(1) TFEU" in result.sections["rules"]
    assert "yes" in result.sections["application"]


def test_still_no_final_answer_after_three_stages():
    result = PipelineRecipe(stages=FIRST_THREE).run(QUESTION, FakeLLMClient(THREE_STAGE_RESPONSES))
    assert result.final_answer == ""


def test_usage_sums_across_every_stage():
    result = PipelineRecipe(stages=FIRST_THREE).run(QUESTION, FakeLLMClient(THREE_STAGE_RESPONSES))
    assert result.usage.calls == 3
    assert result.usage.calls == sum(s.usage.calls for s in result.stages)


# --- Day 4: Stage 4, the amendment log, and violation counting -------------

CONCLUSION_JSON = (
    '{"conclusion": "The exclusivity clause infringes Article 101(1) TFEU.", '
    '"final_answer": "The five-year exclusive supply obligation restricts '
    'competition and is void under Article 101(2) TFEU."}'
)

FOUR_STAGE_RESPONSES = [ISSUES_JSON, RULES_JSON, FINDINGS_JSON, CONCLUSION_JSON]


def test_four_stages_make_four_calls_in_order():
    client = FakeLLMClient(FOUR_STAGE_RESPONSES)
    result = PipelineRecipe().run(QUESTION, client)
    assert len(client.prompts) == 4
    assert [s.name for s in result.stages] == [
        "issues", "rules", "application", "conclusion",
    ]


def test_stage_four_asks_only_for_the_conclusion():
    client = FakeLLMClient(FOUR_STAGE_RESPONSES)
    PipelineRecipe().run(QUESTION, client)
    prompt = client.prompts[3]
    assert steps.STEP_CONCLUSION in prompt
    assert steps.STEP_ISSUES not in prompt
    assert steps.STEP_RULES not in prompt
    assert steps.STEP_APPLICATION not in prompt


def test_stage_four_receives_all_three_earlier_sections():
    # It cannot write a legal answer without the rules it is concluding from.
    client = FakeLLMClient(FOUR_STAGE_RESPONSES)
    PipelineRecipe().run(QUESTION, client)
    prompt = client.prompts[3]
    assert "exclusivity clause" in prompt
    assert "Article 101(1) TFEU" in prompt
    assert "an agreement between undertakings" in prompt


def test_the_completed_pipeline_produces_a_final_answer():
    result = PipelineRecipe().run(QUESTION, FakeLLMClient(FOUR_STAGE_RESPONSES))
    assert "void under Article 101(2) TFEU" in result.final_answer
    assert set(result.sections) == {
        "issues", "rules", "application", "conclusion", "final_answer",
    }


# Violations: a stage answering a step that is not its own.

EARLY_CONCLUSION_JSON = (
    '{"issues": [{"id": "i1", "statement": "Whether the clause restricts competition", '
    '"why_it_arises": "Five-year exclusivity."}], '
    '"conclusion": "The clause is void."}'
)


def test_a_stage_reaching_a_conclusion_early_is_counted_as_a_violation():
    # Spec 6.1: without the "must not" constraints there are not four stages,
    # only four chances to answer the whole question. How often the model
    # cannot hold a partial analysis is itself a result.
    result = PipelineRecipe(stages=(STAGE_ISSUES,)).run(
        QUESTION, FakeLLMClient([EARLY_CONCLUSION_JSON])
    )
    assert result.metadata["stage_violations"] == [
        {"stage": "issues", "wrote": ["conclusion"]}
    ]


def test_out_of_turn_content_is_discarded_not_carried_forward():
    # Keeping an early conclusion would let it reach the final stage, which
    # would then echo it -- the decomposition would collapse silently.
    result = PipelineRecipe(stages=(STAGE_ISSUES,)).run(
        QUESTION, FakeLLMClient([EARLY_CONCLUSION_JSON])
    )
    assert "conclusion" not in result.sections
    assert result.final_answer == ""


def test_a_stage_writing_only_its_own_section_records_no_violation():
    result = PipelineRecipe().run(QUESTION, FakeLLMClient(FOUR_STAGE_RESPONSES))
    assert result.metadata["stage_violations"] == []


# Amendments: a stage correcting an earlier section, which is allowed but
# must be declared (spec 6.4).

AMENDING_FINDINGS_JSON = (
    '{"findings": [{"issue_id": "i1", "element": "an agreement between undertakings", '
    '"holds": "yes", "reasoning": "A written contract exists."}], '
    '"rules": [{"issue_id": "i1", "rule": "Article 101(1) TFEU, read with the block '
    'exemption", "elements": ["an agreement between undertakings"]}], '
    '"amendments": [{"section": "rules", "change": "added the block exemption", '
    '"reason": "the original rule statement was incomplete"}]}'
)


def test_a_declared_amendment_to_an_earlier_section_is_not_a_violation():
    client = FakeLLMClient([ISSUES_JSON, RULES_JSON, AMENDING_FINDINGS_JSON])
    result = PipelineRecipe(
        stages=PipelineRecipe.STAGES[:3]
    ).run(QUESTION, client)
    assert result.metadata["stage_violations"] == []


def test_a_declared_amendment_actually_updates_the_earlier_section():
    client = FakeLLMClient([ISSUES_JSON, RULES_JSON, AMENDING_FINDINGS_JSON])
    result = PipelineRecipe(stages=PipelineRecipe.STAGES[:3]).run(QUESTION, client)
    assert "block exemption" in result.sections["rules"]


def test_amendments_are_recorded_so_self_correction_can_be_counted():
    # The pipeline's rescue rate is a measured quantity, so an amendment that
    # happened but was not recorded would shrink it invisibly.
    client = FakeLLMClient([ISSUES_JSON, RULES_JSON, AMENDING_FINDINGS_JSON])
    result = PipelineRecipe(stages=PipelineRecipe.STAGES[:3]).run(QUESTION, client)
    assert result.metadata["amendments"] == [
        {"stage": "application", "section": "rules",
         "change": "added the block exemption",
         "reason": "the original rule statement was incomplete"}
    ]


# --- Day 5: registration and readiness ------------------------------------

def test_the_pipeline_is_registered_as_a_runnable_recipe():
    # All four stages exist, so a result is a complete pipeline answer rather
    # than a partial one that would be graded as though it were complete.
    from scripts.run_recipe import RECIPES
    assert RECIPES["pipeline"] is PipelineRecipe


def test_the_pipeline_defaults_to_baseline_effort():
    # Its extra compute comes from making four calls, not from thinking harder
    # per call. Raised effort is Recipe 3's manipulation and only Recipe 3's.
    from glassbox.config import EFFORT_BASELINE
    from scripts.run_recipe import resolve_effort
    assert resolve_effort("pipeline", None) == EFFORT_BASELINE


def test_a_pipeline_answer_is_graded_on_its_final_answer_only():
    # The doubling guard, on the recipe it was written for. sections carries
    # the analysis AND the final answer, so joining them would hand the grader
    # roughly twice what Recipe 1 gets.
    from glassbox.grading.normalise import normalise
    result = PipelineRecipe().run(QUESTION, FakeLLMClient(FOUR_STAGE_RESPONSES))
    blinded = normalise(result)
    assert "void under Article 101(2) TFEU" in blinded
    assert "an agreement between undertakings" not in blinded
    assert "pipeline" not in blinded.lower()


def test_a_pipeline_run_is_flagged_when_any_stage_truncates():
    # One truncated stage corrupts every stage after it, since they read its
    # output through the relay.
    client = FakeLLMClient(FOUR_STAGE_RESPONSES, finish_reason="length")
    result = PipelineRecipe().run(QUESTION, client)
    assert result.metadata["truncated"] is True


def test_a_clean_pipeline_run_is_not_flagged():
    result = PipelineRecipe().run(QUESTION, FakeLLMClient(FOUR_STAGE_RESPONSES))
    assert result.metadata["truncated"] is False
