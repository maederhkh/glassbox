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
