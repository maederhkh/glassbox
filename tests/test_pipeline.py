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
        json_instructions='{"conclusion": "..."}', section="conclusion",
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
        json_instructions='{"conclusion": "..."}', section="conclusion",
    )
    client = FakeLLMClient([ISSUES_JSON, '{"conclusion": "Restated."}'])
    PipelineRecipe(stages=(STAGE_ISSUES, second)).run(QUESTION, client)

    assert '"why_it_arises"' not in client.prompts[1]


def test_unparseable_stage_output_is_reported_not_swallowed():
    with pytest.raises(ValueError):
        PipelineRecipe(stages=(STAGE_ISSUES,)).run(QUESTION, FakeLLMClient(["not json at all"]))


def test_the_pipeline_is_not_registered_as_a_runnable_recipe_yet():
    # Stages 2-4 do not exist. A one-stage result in a run directory would be
    # graded as though it were a pipeline answer.
    from scripts.run_recipe import RECIPES
    assert "pipeline" not in RECIPES


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


def test_three_stages_make_three_calls_in_order():
    client = FakeLLMClient(THREE_STAGE_RESPONSES)
    result = PipelineRecipe().run(QUESTION, client)
    assert len(client.prompts) == 3
    assert [s.name for s in result.stages] == ["issues", "rules", "application"]


def test_stage_two_asks_only_for_rules():
    client = FakeLLMClient(THREE_STAGE_RESPONSES)
    PipelineRecipe().run(QUESTION, client)
    prompt = client.prompts[1]
    assert steps.STEP_RULES in prompt
    assert steps.STEP_ISSUES not in prompt
    assert steps.STEP_APPLICATION not in prompt
    assert steps.STEP_CONCLUSION not in prompt


def test_stage_three_asks_only_for_application():
    client = FakeLLMClient(THREE_STAGE_RESPONSES)
    PipelineRecipe().run(QUESTION, client)
    prompt = client.prompts[2]
    assert steps.STEP_APPLICATION in prompt
    assert steps.STEP_ISSUES not in prompt
    assert steps.STEP_RULES not in prompt
    assert steps.STEP_CONCLUSION not in prompt


def test_stage_two_receives_stage_ones_issues():
    client = FakeLLMClient(THREE_STAGE_RESPONSES)
    PipelineRecipe().run(QUESTION, client)
    assert "exclusivity clause" in client.prompts[1]


def test_stage_three_receives_everything_committed_before_it():
    # The relay accumulates: Stage 3 cannot apply elements to facts unless it
    # can see both the issues and the elements Stage 2 produced.
    client = FakeLLMClient(THREE_STAGE_RESPONSES)
    PipelineRecipe().run(QUESTION, client)
    prompt = client.prompts[2]
    assert "exclusivity clause" in prompt
    assert "Article 101(1) TFEU" in prompt
    assert "an appreciable restriction of competition" in prompt


def test_earlier_sections_survive_to_the_end_of_the_run():
    # Each stage returns only its own section, so the case file must accumulate
    # across stages rather than being replaced by the latest one.
    result = PipelineRecipe().run(QUESTION, FakeLLMClient(THREE_STAGE_RESPONSES))
    assert set(result.sections) == {"issues", "rules", "application"}
    assert "exclusivity clause" in result.sections["issues"]
    assert "Article 101(1) TFEU" in result.sections["rules"]
    assert "yes" in result.sections["application"]


def test_still_no_final_answer_after_three_stages():
    result = PipelineRecipe().run(QUESTION, FakeLLMClient(THREE_STAGE_RESPONSES))
    assert result.final_answer == ""


def test_usage_sums_across_every_stage():
    result = PipelineRecipe().run(QUESTION, FakeLLMClient(THREE_STAGE_RESPONSES))
    assert result.usage.calls == 3
    assert result.usage.calls == sum(s.usage.calls for s in result.stages)
