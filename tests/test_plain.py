from glassbox.dataset import Question
from glassbox.llm import FakeLLMClient
from glassbox.recipes.plain import PlainRecipe

QUESTION = Question(
    id="q1", question="Does the shop owe a duty of care?", answer="Yes, shops owe...",
    course="Tort Law", area="Private", jurisdiction="Generic", year="2022",
    question_words=60, answer_words=200,
)


def test_plain_makes_exactly_one_call():
    client = FakeLLMClient(["The shop owes a duty of care because..."])
    PlainRecipe().run(QUESTION, client)
    assert len(client.prompts) == 1


def test_plain_returns_the_model_text_as_final_answer():
    client = FakeLLMClient(["The shop owes a duty of care because..."])
    result = PlainRecipe().run(QUESTION, client)
    assert result.final_answer == "The shop owes a duty of care because..."
    assert result.recipe == "plain"
    assert result.question_id == "q1"


def test_plain_has_no_sections_and_no_stages():
    result = PlainRecipe().run(QUESTION, FakeLLMClient(["answer"]))
    assert result.sections is None
    assert result.stages == []


def test_prompt_contains_question_and_course_but_not_reference_answer():
    # NOTE: deviation from the task-4 brief. As literally transcribed, this test
    # checked `QUESTION.course in client.prompts[0]`, but PlainRecipe (also per
    # the brief, verbatim) puts the course into the *system* message via
    # PLAIN_SYSTEM, not into PLAIN_PROMPT's user message - PLAIN_PROMPT only
    # interpolates `question`. The two pieces of brief-supplied code were
    # mutually inconsistent. Fixed here by checking course against
    # client.systems[0], where it actually lives, rather than altering the
    # protected PLAIN_PROMPT/PLAIN_SYSTEM content. The reference-answer
    # exclusion check - the load-bearing part of this test - is unchanged.
    client = FakeLLMClient(["answer"])
    PlainRecipe().run(QUESTION, client)
    prompt = client.prompts[0]
    assert QUESTION.question in prompt
    assert QUESTION.course in client.systems[0]
    assert "Yes, shops owe" not in prompt


def test_usage_is_recorded():
    result = PlainRecipe().run(QUESTION, FakeLLMClient(["answer"]))
    assert result.usage.calls == 1


def test_metadata_carries_a_prompt_hash_and_timestamp():
    a = PlainRecipe().run(QUESTION, FakeLLMClient(["answer"]))
    b = PlainRecipe().run(QUESTION, FakeLLMClient(["a different answer"]))
    assert a.metadata["prompt_hash"] == b.metadata["prompt_hash"]
    assert isinstance(a.metadata["timestamp"], str) and a.metadata["timestamp"]


def test_a_truncated_answer_is_flagged_in_the_metadata():
    # Hitting the output cap leaves a cut-off answer that would score badly
    # for the wrong reason. It must be visible in the result, not inferred.
    from glassbox.recipes.plain import PlainRecipe
    result = PlainRecipe().run(
        QUESTION, FakeLLMClient(["cut off mid-sen"], finish_reason="length")
    )
    assert result.metadata["truncated"] is True


def test_a_complete_answer_is_not_flagged():
    from glassbox.recipes.plain import PlainRecipe
    result = PlainRecipe().run(QUESTION, FakeLLMClient(["a complete answer"]))
    assert result.metadata["truncated"] is False
