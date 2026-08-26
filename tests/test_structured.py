from glassbox.dataset import Question
from glassbox.llm import FakeLLMClient
from glassbox.recipes.plain import PlainRecipe
from glassbox.recipes.structured import StructuredRecipe, ThinkLongerRecipe

QUESTION = Question(
    id="q1", question="Is the shop liable?", answer="reference text here",
    course="Tort Law", area="Private", jurisdiction="Generic", year="2022",
    question_words=60, answer_words=200,
)

PAYLOAD = """{"issues": [{"id": "i1", "statement": "Duty owed?", "why_it_arises": "customer"}],
 "rules": [{"issue_id": "i1", "rule": "Occupiers owe a duty.", "elements": ["occupier"]}],
 "findings": [{"issue_id": "i1", "element": "occupier", "holds": "yes", "reasoning": "controls"}],
 "conclusion": "Liable.", "final_answer": "The shop is liable because it controls the premises.",
 "amendments": []}"""


def test_structured_makes_exactly_one_call():
    client = FakeLLMClient([PAYLOAD])
    StructuredRecipe().run(QUESTION, client)
    assert len(client.prompts) == 1


def test_structured_populates_sections_and_final_answer():
    result = StructuredRecipe().run(QUESTION, FakeLLMClient([PAYLOAD]))
    assert result.recipe == "structured"
    assert result.final_answer.startswith("The shop is liable")
    assert set(result.sections) == {
        "issues", "rules", "application", "conclusion", "final_answer"}


def test_structured_records_no_stages():
    result = StructuredRecipe().run(QUESTION, FakeLLMClient([PAYLOAD]))
    assert result.stages == []


def test_prompt_asks_for_all_four_sections_and_omits_the_reference():
    client = FakeLLMClient([PAYLOAD])
    StructuredRecipe().run(QUESTION, client)
    prompt = client.prompts[0].lower()
    for word in ["issue", "rule", "element", "appl", "conclusion"]:
        assert word in prompt
    assert "reference text here" not in client.prompts[0]


def test_think_longer_uses_the_same_prompt_as_structured():
    a, b = FakeLLMClient([PAYLOAD]), FakeLLMClient([PAYLOAD])
    StructuredRecipe().run(QUESTION, a)
    ThinkLongerRecipe().run(QUESTION, b)
    assert a.prompts[0] == b.prompts[0]


def test_think_longer_is_named_distinctly():
    result = ThinkLongerRecipe().run(QUESTION, FakeLLMClient([PAYLOAD]))
    assert result.recipe == "think_longer"


def test_unparseable_output_falls_back_to_raw_text():
    result = StructuredRecipe().run(QUESTION, FakeLLMClient(["not json at all"]))
    assert result.final_answer == "not json at all"
    assert result.sections is None
    assert result.metadata["parse_failed"] is True


def test_prompt_hash_is_stable_across_separate_calls():
    # Same prompt (same recipe, no per-question substitution in the hash) must
    # reproduce the same hash -- this is what lets every result from one run
    # be checked for a single shared value rather than compared pairwise.
    a = StructuredRecipe().run(QUESTION, FakeLLMClient([PAYLOAD]))
    b = StructuredRecipe().run(QUESTION, FakeLLMClient([PAYLOAD]))
    assert a.metadata["prompt_hash"] == b.metadata["prompt_hash"]


def test_think_longer_shares_structureds_prompt_hash():
    # Recipes 2 and 3 differ only in reasoning_effort (set on the client by the
    # caller, never touched here) -- their prompt_hash must be identical, or
    # something besides effort varies between them.
    a = StructuredRecipe().run(QUESTION, FakeLLMClient([PAYLOAD]))
    b = ThinkLongerRecipe().run(QUESTION, FakeLLMClient([PAYLOAD]))
    assert a.metadata["prompt_hash"] == b.metadata["prompt_hash"]


def test_prompt_hash_differs_from_a_recipe_with_a_different_prompt():
    structured = StructuredRecipe().run(QUESTION, FakeLLMClient([PAYLOAD]))
    plain = PlainRecipe().run(QUESTION, FakeLLMClient(["a plain answer"]))
    assert structured.metadata["prompt_hash"] != plain.metadata["prompt_hash"]


def test_metadata_carries_a_non_empty_timestamp():
    result = StructuredRecipe().run(QUESTION, FakeLLMClient([PAYLOAD]))
    assert isinstance(result.metadata["timestamp"], str)
    assert result.metadata["timestamp"]
