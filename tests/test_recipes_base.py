from glassbox.recipes.base import prompt_fingerprint


def test_prompt_fingerprint_is_stable_for_the_same_input():
    assert prompt_fingerprint("system", "user") == prompt_fingerprint("system", "user")


def test_prompt_fingerprint_differs_when_a_part_differs():
    assert prompt_fingerprint("system", "user") != prompt_fingerprint("system", "different")


def test_prompt_fingerprint_does_not_collide_across_a_part_boundary():
    # Without a separator, ("ab", "c") and ("a", "bc") would hash identically
    # once joined -- the unit separator is what prevents that.
    assert prompt_fingerprint("ab", "c") != prompt_fingerprint("a", "bc")
