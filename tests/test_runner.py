from glassbox.dataset import Question
from glassbox.llm import FakeLLMClient
from glassbox.recipes.plain import PlainRecipe
from glassbox.runner import run_recipe
from glassbox.storage import load_result, result_path, save_result

QUESTION_A = Question(
    id="qa", question="Is the shop liable?", answer="reference text here",
    course="Tort Law", area="Private", jurisdiction="Generic", year="2022",
    question_words=60, answer_words=200,
)
QUESTION_B = Question(
    id="qb", question="Did the tenant breach the lease?", answer="reference text",
    course="Property Law", area="Private", jurisdiction="Generic", year="2022",
    question_words=60, answer_words=200,
)


def test_skips_a_question_whose_result_already_exists(tmp_path):
    existing = PlainRecipe().run(QUESTION_A, FakeLLMClient(["first answer"]))
    save_result(existing, tmp_path)

    # Only one response queued, for QUESTION_B. A wrongly re-run QUESTION_A
    # would exhaust the queue and blow up with an AssertionError, so getting
    # a result at all proves it was skipped, not re-spent.
    client = FakeLLMClient(["second answer"])
    results = run_recipe(PlainRecipe(), [QUESTION_A, QUESTION_B], client, tmp_path)

    assert [r.question_id for r in results] == ["qa", "qb"]
    assert results[0].final_answer == "first answer"
    assert results[1].final_answer == "second answer"
    assert len(client.prompts) == 1


def test_runs_and_saves_a_question_with_no_result_on_disk(tmp_path):
    client = FakeLLMClient(["an answer"])
    results = run_recipe(PlainRecipe(), [QUESTION_A], client, tmp_path)

    assert len(client.prompts) == 1
    assert load_result(result_path("plain", "qa", tmp_path)).final_answer == "an answer"


def test_rerun_true_overwrites_rather_than_skips(tmp_path):
    existing = PlainRecipe().run(QUESTION_A, FakeLLMClient(["first answer"]))
    save_result(existing, tmp_path)

    client = FakeLLMClient(["replaced answer"])
    results = run_recipe(PlainRecipe(), [QUESTION_A], client, tmp_path, rerun=True)

    assert len(client.prompts) == 1
    assert results[0].final_answer == "replaced answer"
    assert load_result(result_path("plain", "qa", tmp_path)).final_answer == (
        "replaced answer"
    )


def test_dataset_revision_and_seed_are_merged_into_a_fresh_result(tmp_path):
    client = FakeLLMClient(["an answer"])
    results = run_recipe(PlainRecipe(), [QUESTION_A], client, tmp_path,
                          dataset_revision="abc123", seed=20260810)

    assert results[0].metadata["dataset_revision"] == "abc123"
    assert results[0].metadata["seed"] == 20260810
    # And persisted, not just held in the in-memory result.
    loaded = load_result(result_path("plain", "qa", tmp_path))
    assert loaded.metadata["dataset_revision"] == "abc123"
    assert loaded.metadata["seed"] == 20260810


def test_dataset_revision_and_seed_are_omitted_when_not_passed(tmp_path):
    results = run_recipe(PlainRecipe(), [QUESTION_A], FakeLLMClient(["an answer"]), tmp_path)

    assert "dataset_revision" not in results[0].metadata
    assert "seed" not in results[0].metadata


def test_dataset_revision_and_seed_are_not_backfilled_onto_a_skipped_result(tmp_path):
    existing = PlainRecipe().run(QUESTION_A, FakeLLMClient(["first answer"]))
    save_result(existing, tmp_path)

    # No responses queued: if the skip-existing path made a call, this would
    # blow up with an AssertionError instead of quietly reusing the old file.
    results = run_recipe(PlainRecipe(), [QUESTION_A], FakeLLMClient([]), tmp_path,
                          dataset_revision="abc123", seed=1)

    assert "dataset_revision" not in results[0].metadata
    assert "seed" not in results[0].metadata
