from glassbox.usage import Usage, cost_usd


def test_usage_adds_componentwise():
    a = Usage(input_tokens=10, output_tokens=5, reasoning_tokens=2, calls=1)
    b = Usage(input_tokens=3, output_tokens=7, reasoning_tokens=0, calls=1)
    total = a + b
    assert total == Usage(input_tokens=13, output_tokens=12, reasoning_tokens=2, calls=2)


def test_usage_zero_is_additive_identity():
    a = Usage(input_tokens=10, output_tokens=5, reasoning_tokens=2, calls=1)
    assert Usage.zero() + a == a


def test_cost_is_none_for_unpriced_model():
    usage = Usage(input_tokens=1000, output_tokens=1000, reasoning_tokens=0, calls=1)
    assert cost_usd(usage, "some-unpriced-model") is None


def test_cost_counts_reasoning_tokens_as_output():
    usage = Usage(input_tokens=0, output_tokens=1_000_000, reasoning_tokens=1_000_000, calls=1)
    priced = cost_usd(usage, "openai/gpt-5-mini")
    unpriced_half = cost_usd(
        Usage(input_tokens=0, output_tokens=1_000_000, reasoning_tokens=0, calls=1), "openai/gpt-5-mini")
    assert priced is not None and unpriced_half is not None
    assert priced == 2 * unpriced_half
