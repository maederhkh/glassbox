from types import SimpleNamespace

import httpx
import openai
import pytest
from tenacity import wait_fixed

from glassbox.llm import FakeLLMClient, LLMClient, _is_retryable
from glassbox.usage import Usage


def _status_error(status_code: int) -> openai.APIStatusError:
    request = httpx.Request("POST", "https://openrouter.test/chat/completions")
    response = httpx.Response(status_code, request=request)
    return openai.APIStatusError("boom", response=response, body=None)


def _stubbed_client(monkeypatch, create) -> LLMClient:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    client = LLMClient()
    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    return client


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 422])
def test_client_errors_are_not_retryable(status):
    assert _is_retryable(_status_error(status)) is False


@pytest.mark.parametrize("status", [500, 502, 503])
def test_server_errors_are_retryable(status):
    assert _is_retryable(_status_error(status)) is True


def test_rate_limit_is_retryable():
    request = httpx.Request("POST", "https://openrouter.test/chat/completions")
    response = httpx.Response(429, request=request)
    error = openai.RateLimitError("slow down", response=response, body=None)
    assert _is_retryable(error) is True


def test_connection_and_timeout_errors_are_retryable():
    request = httpx.Request("POST", "https://openrouter.test/chat/completions")
    assert _is_retryable(openai.APIConnectionError(request=request)) is True
    assert _is_retryable(openai.APITimeoutError(request)) is True


def test_unrelated_exceptions_are_not_retryable():
    assert _is_retryable(ValueError("not an API failure")) is False


def test_out_of_credit_fails_after_one_attempt(monkeypatch):
    attempts = []

    def create(**kwargs):
        attempts.append(kwargs)
        raise _status_error(402)

    client = _stubbed_client(monkeypatch, create)
    with pytest.raises(openai.APIStatusError):
        client._call([{"role": "user", "content": "q"}])
    assert len(attempts) == 1


def test_transient_errors_are_retried_until_success(monkeypatch):
    monkeypatch.setattr(LLMClient._call.retry, "wait", wait_fixed(0))
    attempts = []

    def create(**kwargs):
        attempts.append(kwargs)
        if len(attempts) < 3:
            raise _status_error(503)
        return "the-response"

    client = _stubbed_client(monkeypatch, create)
    result = client._call([{"role": "user", "content": "q"}])
    assert result == "the-response"
    assert len(attempts) == 3


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
