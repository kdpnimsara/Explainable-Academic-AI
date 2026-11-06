
import httpx
import pytest

from src.generation.openai_compatible_model import (
    OpenAICompatibleChatModel,
)


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self.payload = payload or {}
        self.text = text

    def json(self):
        return self.payload


def make_model(**kwargs):
    defaults = {
        "base_url": "https://example.test/v1",
        "model_name": "test-model",
        "api_key": "test-secret",
        "timeout": 10,
        "temperature": 0.2,
    }
    defaults.update(kwargs)
    return OpenAICompatibleChatModel(**defaults)


def test_model_generates_text(monkeypatch):
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs

        return FakeResponse(
            payload={
                "choices": [
                    {
                        "message": {
                            "content": "The answer is supported by evidence."
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr(httpx, "post", fake_post)

    model = make_model()
    result = model.generate("Explain the examination rules.")

    assert result == "The answer is supported by evidence."
    assert captured["url"] == (
        "https://example.test/v1/chat/completions"
    )
    assert captured["kwargs"]["json"]["model"] == "test-model"
    assert captured["kwargs"]["headers"]["Authorization"] == (
        "Bearer test-secret"
    )


def test_model_works_without_api_key_for_local_endpoints(monkeypatch):
    captured = {}

    def fake_post(url, **kwargs):
        captured.update(kwargs)

        return FakeResponse(
            payload={
                "choices": [
                    {
                        "message": {
                            "content": "Local model response."
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr(httpx, "post", fake_post)

    model = make_model(
        base_url="http://localhost:11434/v1",
        model_name="local-model",
        api_key="",
    )

    assert model.generate("Test") == "Local model response."
    assert "Authorization" not in captured["headers"]


def test_model_supports_content_blocks(monkeypatch):
    def fake_post(url, **kwargs):
        return FakeResponse(
            payload={
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"type": "text", "text": "Evidence "},
                                {"type": "text", "text": "validated."},
                            ]
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr(httpx, "post", fake_post)

    model = make_model()

    assert model.generate("Test") == "Evidence validated."


def test_model_rejects_empty_prompt():
    model = make_model()

    with pytest.raises(ValueError):
        model.generate("   ")


def test_model_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        make_model(base_url="invalid-url")


def test_model_reports_api_error(monkeypatch):
    def fake_post(url, **kwargs):
        return FakeResponse(
            status_code=401,
            text="Unauthorized",
        )

    monkeypatch.setattr(httpx, "post", fake_post)

    model = make_model()

    with pytest.raises(RuntimeError, match="HTTP 401"):
        model.generate("Test question")


def test_model_reports_invalid_response(monkeypatch):
    def fake_post(url, **kwargs):
        return FakeResponse(
            payload={"unexpected": "response"}
        )

    monkeypatch.setattr(httpx, "post", fake_post)

    model = make_model()

    with pytest.raises(RuntimeError, match="no choices"):
        model.generate("Test question")


def test_model_reports_timeout(monkeypatch):
    def fake_post(url, **kwargs):
        request = httpx.Request("POST", url)

        raise httpx.TimeoutException(
            "Request timed out",
            request=request,
        )

    monkeypatch.setattr(httpx, "post", fake_post)

    model = make_model()

    with pytest.raises(RuntimeError, match="timed out"):
        model.generate("Test question")
