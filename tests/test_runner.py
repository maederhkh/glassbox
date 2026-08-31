import dataclasses

import pytest

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


def test_resumes_past_a_corrupt_result_file_with_a_warning_naming_the_path(tmp_path, capsys):
    # The project's documented failure mode: a run killed mid-write leaves a
    # truncated JSON file behind. Resuming must not crash with a bare
    # JSONDecodeError -- it should treat the file as absent, warn (naming the
    # path), and re-run that one question.
    path = result_path("plain", QUESTION_A.id, tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"recipe": "plain", "question_id": "qa", "final_a', encoding="utf-8")

    client = FakeLLMClient(["recovered answer"])
    results = run_recipe(PlainRecipe(), [QUESTION_A], client, tmp_path)

    assert len(client.prompts) == 1
    assert results[0].final_answer == "recovered answer"

    captured = capsys.readouterr().out
    assert str(path) in captured

    # The corrupt file has been overwritten with a valid, loadable one.
    assert load_result(path).final_answer == "recovered answer"


def test_refuses_to_resume_when_stored_prompt_hash_differs_from_the_recipes_current_one(
    tmp_path,
):
    # The exact contamination a prior incident on this project produced: the
    # prompt changed mid-experiment, and a partial run resumed silently onto
    # the new prompt, mixing hashes in one directory. Only file mtimes caught
    # it that time; this must refuse outright instead.
    stale = PlainRecipe().run(QUESTION_A, FakeLLMClient(["old answer"]))
    stale = dataclasses.replace(
        stale, metadata={**stale.metadata, "prompt_hash": "stale-hash-does-not-match"}
    )
    save_result(stale, tmp_path)

    with pytest.raises(RuntimeError) as exc_info:
        run_recipe(PlainRecipe(), [QUESTION_A], FakeLLMClient([]), tmp_path)

    message = str(exc_info.value)
    assert QUESTION_A.id in message
    assert "stale-hash-does-not-match" in message
    assert PlainRecipe.PROMPT_HASH in message


def test_skips_hash_check_when_the_recipe_has_no_prompt_hash_attribute(tmp_path):
    # A recipe with no PROMPT_HASH class attribute (e.g. not yet wired up for
    # this) must not crash -- it should skip the comparison and resume
    # normally.
    stale = PlainRecipe().run(QUESTION_A, FakeLLMClient(["old answer"]))
    stale = dataclasses.replace(
        stale, metadata={**stale.metadata, "prompt_hash": "totally-different"}
    )
    save_result(stale, tmp_path)

    class NoHashRecipe:
        name = "plain"

        def run(self, question, client):
            return PlainRecipe().run(question, client)

    results = run_recipe(NoHashRecipe(), [QUESTION_A], FakeLLMClient([]), tmp_path)
    assert results[0].final_answer == "old answer"


def test_skips_hash_check_when_the_stored_result_predates_the_prompt_hash_field(tmp_path):
    # The 11 August results predate the prompt_hash field (though they have
    # since been regenerated) -- a stored result with no recorded hash must
    # not crash the comparison either.
    legacy = PlainRecipe().run(QUESTION_A, FakeLLMClient(["legacy answer"]))
    legacy_metadata = {k: v for k, v in legacy.metadata.items() if k != "prompt_hash"}
    legacy = dataclasses.replace(legacy, metadata=legacy_metadata)
    save_result(legacy, tmp_path)

    results = run_recipe(PlainRecipe(), [QUESTION_A], FakeLLMClient([]), tmp_path)
    assert results[0].final_answer == "legacy answer"


def test_refuses_to_resume_when_stored_reasoning_effort_differs_from_the_current_run(
    tmp_path,
):
    # The same contamination class the prompt_hash check guards against, but on
    # the other manipulated variable: Recipe 3 (think_longer) is identical to
    # Recipe 2 except for a raised reasoning effort, so resuming a partial
    # think_longer run at the wrong effort would silently mix effort levels
    # within one arm. Must refuse outright, not warn.
    stale = PlainRecipe().run(QUESTION_A, FakeLLMClient(["old answer"], reasoning_effort="low"))
    save_result(stale, tmp_path)

    with pytest.raises(RuntimeError) as exc_info:
        run_recipe(PlainRecipe(), [QUESTION_A], FakeLLMClient([], reasoning_effort="high"), tmp_path)

    message = str(exc_info.value)
    assert QUESTION_A.id in message
    assert "low" in message
    assert "high" in message


def test_resumes_normally_when_stored_and_current_reasoning_effort_match(tmp_path):
    stale = PlainRecipe().run(QUESTION_A, FakeLLMClient(["old answer"], reasoning_effort="high"))
    save_result(stale, tmp_path)

    results = run_recipe(
        PlainRecipe(), [QUESTION_A], FakeLLMClient([], reasoning_effort="high"), tmp_path
    )
    assert results[0].final_answer == "old answer"


def test_skips_reasoning_effort_check_when_the_current_client_reports_no_effort(tmp_path):
    # FakeLLMClient defaults reasoning_effort to None, mirroring a client that
    # doesn't report effort at all -- must not crash, and must not refuse.
    stale = PlainRecipe().run(QUESTION_A, FakeLLMClient(["old answer"], reasoning_effort="high"))
    save_result(stale, tmp_path)

    results = run_recipe(PlainRecipe(), [QUESTION_A], FakeLLMClient([]), tmp_path)
    assert results[0].final_answer == "old answer"


def test_skips_reasoning_effort_check_when_the_stored_result_predates_the_field(tmp_path):
    # Older results may predate the reasoning_effort field entirely (the same
    # tolerance prompt_hash's check already gives pre-field results).
    legacy = PlainRecipe().run(QUESTION_A, FakeLLMClient(["legacy answer"], reasoning_effort="high"))
    legacy_metadata = {k: v for k, v in legacy.metadata.items() if k != "reasoning_effort"}
    legacy = dataclasses.replace(legacy, metadata=legacy_metadata)
    save_result(legacy, tmp_path)

    results = run_recipe(
        PlainRecipe(), [QUESTION_A], FakeLLMClient([], reasoning_effort="low"), tmp_path
    )
    assert results[0].final_answer == "legacy answer"


def test_dataset_revision_and_seed_are_not_backfilled_onto_a_skipped_result(tmp_path):
    existing = PlainRecipe().run(QUESTION_A, FakeLLMClient(["first answer"]))
    save_result(existing, tmp_path)

    # No responses queued: if the skip-existing path made a call, this would
    # blow up with an AssertionError instead of quietly reusing the old file.
    results = run_recipe(PlainRecipe(), [QUESTION_A], FakeLLMClient([]), tmp_path,
                          dataset_revision="abc123", seed=1)

    assert "dataset_revision" not in results[0].metadata
    assert "seed" not in results[0].metadata
