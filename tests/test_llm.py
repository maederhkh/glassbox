import pytest

from glassbox.llm import FakeLLMClient
from glassbox.usage import Usage


def test_fake_client_returns_queued_responses_in_order():
    client = FakeLLMClient(["first", "second"])
    assert client.complete("a").text == "first"
    assert client.complete("b").text == "second"


def test_fake_client_records_prompts_and_systems():
    client = FakeLLMClient(["x"])
    client.complete("the prompt", system="the system")
    assert client.prompts == ["the prompt"]
    assert client.systems == ["the system"]


def test_fake_client_reports_usage():
    client = FakeLLMClient(["hello"])
    completion = client.complete("a")
    assert isinstance(completion.usage, Usage)
    assert completion.usage.calls == 1


def test_fake_client_raises_when_exhausted():
    client = FakeLLMClient(["only one"])
    client.complete("a")
    with pytest.raises(AssertionError, match="exhausted"):
        client.complete("b")
