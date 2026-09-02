import pytest

from glassbox.config import EFFORT_BASELINE, EFFORT_RAISED
from scripts.run_recipe import resolve_effort


def test_think_longer_defaults_to_raised_effort():
    # Recipe 3's only distinguishing feature from Recipe 2 is effort -- this is
    # what makes the raised effort the default instead of something an
    # operator has to remember to pass.
    assert resolve_effort("think_longer", None) == EFFORT_RAISED


def test_plain_defaults_to_baseline_effort():
    assert resolve_effort("plain", None) == EFFORT_BASELINE


def test_structured_defaults_to_baseline_effort():
    assert resolve_effort("structured", None) == EFFORT_BASELINE


def test_explicit_effort_overrides_the_recipes_default():
    assert resolve_effort("plain", EFFORT_RAISED) == EFFORT_RAISED
    assert resolve_effort("think_longer", EFFORT_BASELINE) == EFFORT_BASELINE


def test_unrecognised_effort_is_rejected_before_any_client_is_constructed():
    # resolve_effort constructs nothing itself -- main() calls it before
    # LLMClient(...), so a typo here never reaches the API.
    with pytest.raises(SystemExit):
        resolve_effort("plain", "medium")


def test_baseline_effort_gets_the_smaller_output_cap():
    # Derived from the effort rather than the recipe name: output volume tracks
    # how hard the model thinks, and this stays correct if another recipe ever
    # uses raised effort. Observed maxima on 60 real answers were 9,775 output
    # plus reasoning tokens at baseline and 30,792 at raised effort.
    from glassbox.config import EFFORT_BASELINE, MAX_OUTPUT_TOKENS_BASELINE
    from scripts.run_recipe import resolve_max_output_tokens
    assert resolve_max_output_tokens(EFFORT_BASELINE) == MAX_OUTPUT_TOKENS_BASELINE


def test_raised_effort_gets_the_larger_output_cap():
    from glassbox.config import EFFORT_RAISED, MAX_OUTPUT_TOKENS_RAISED
    from scripts.run_recipe import resolve_max_output_tokens
    assert resolve_max_output_tokens(EFFORT_RAISED) == MAX_OUTPUT_TOKENS_RAISED


def test_both_caps_clear_the_output_actually_observed():
    # Truncation wastes a paid call and corrupts the answer, so the caps keep
    # real headroom over what the model has actually produced.
    from glassbox.config import MAX_OUTPUT_TOKENS_BASELINE, MAX_OUTPUT_TOKENS_RAISED
    assert MAX_OUTPUT_TOKENS_BASELINE > 9_775
    assert MAX_OUTPUT_TOKENS_RAISED > 30_792


def test_an_unrecognised_effort_has_no_cap_of_its_own():
    from scripts.run_recipe import resolve_max_output_tokens
    with pytest.raises(SystemExit):
        resolve_max_output_tokens("medium")
