"""The four step instructions must have exactly one home.

Recipe 2 delivers all four in a single call; the Phase 2 pipeline will deliver
them one per call. Recipe 2 is only a valid control for the pipeline if the two
use identical wording, so the wording cannot live in two files where it can
drift. These tests pin that single source, and pin the resulting prompt hash so
a refactor cannot silently change what the model is asked.
"""

from glassbox.recipes import steps
from glassbox.recipes.structured import PROMPT_HASH, STRUCTURED_PROMPT

# The hash recorded in all 40 structured/think_longer results already on disk.
# It must not change: a different hash means the model would be asked something
# different, those results would no longer be resumable, and Recipe 2 would stop
# being a like-for-like control. A deliberate future prompt change updates this
# line, which makes the change visible in the diff rather than silent.
FROZEN_PROMPT_HASH = "b00409307a467fb79aba7c34ff757c3bc0d10658556c3b5c013b7ce51f88ccff"


def test_the_refactor_did_not_change_what_the_model_is_asked():
    assert PROMPT_HASH == FROZEN_PROMPT_HASH


def test_there_are_exactly_four_steps_in_pipeline_order():
    assert steps.ORDERED_STEPS == (
        steps.STEP_ISSUES,
        steps.STEP_RULES,
        steps.STEP_APPLICATION,
        steps.STEP_CONCLUSION,
    )


def test_every_step_appears_verbatim_in_the_structured_prompt():
    # Recipe 2's single call must ask for the same four things the pipeline
    # will ask for one at a time.
    for step in steps.ORDERED_STEPS:
        assert step in STRUCTURED_PROMPT


def test_the_structured_prompt_numbers_the_steps_from_the_shared_source():
    # Not just "the text appears somewhere" -- the numbered list Recipe 2 sends
    # is built from ORDERED_STEPS, so adding or reordering a step there changes
    # the prompt rather than leaving the two out of step.
    expected = "\n".join(f"{i}. {s}" for i, s in enumerate(steps.ORDERED_STEPS, 1))
    assert expected in STRUCTURED_PROMPT


def test_steps_are_single_instructions_not_a_numbered_block():
    # Each constant is one instruction with no numbering of its own, so the
    # pipeline can hand one to a stage without stripping a "3. " prefix.
    for step in steps.ORDERED_STEPS:
        assert not step.lstrip().startswith(("1.", "2.", "3.", "4."))
        assert "\n" not in step
