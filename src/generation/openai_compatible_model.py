
"""OpenAI-compatible chat completion model adapter.

Supports configurable OpenAI-compatible endpoints without requiring
a provider-specific SDK. Credentials are loaded from environment
variables or a local .env file.
"""

import os
from typing import Any

import httpx
from dotenv import load_dotenv


class OpenAICompatibleChatModel:
    """Generate text through an OpenAI-compatible chat API."""

    def __init__(
        self,
        base_url: str | None = None,
        model_name: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        temperature: float | None = None,
    ) -> None:

        load_dotenv(override=False)

        self.base_url = (
            base_url or os.getenv("LLM_BASE_URL", "")
        ).strip().rstrip("/")

        self.model_name = (
            model_name or os.getenv("LLM_MODEL", "")
        ).strip()

        configured_key = (
            api_key
            if api_key is not None
            else os.getenv("LLM_API_KEY", "")
        )

        self.api_key = configured_key.strip()

        try:
            self.timeout = float(
                timeout
                if timeout is not None
                else os.getenv("LLM_TIMEOUT", "60")
            )

            self.temperature = float(
                temperature
                if temperature is not None
                else os.getenv("LLM_TEMPERATURE", "0.2")
            )

        except (TypeError, ValueError) as exc:
            raise ValueError(
                "LLM_TIMEOUT and LLM_TEMPERATURE must be valid numbers."
            ) from exc

        if not self.base_url:
            raise ValueError(
                "LLM_BASE_URL is missing. Configure it in your local .env file."
            )

        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError(
                "LLM_BASE_URL must be a valid HTTP or HTTPS URL."
            )

        if not self.model_name:
            raise ValueError(
                "LLM_MODEL is missing. Configure it in your local .env file."
            )

        if self.timeout <= 0:
            raise ValueError("timeout must be greater than zero.")

        if not 0 <= self.temperature <= 2:
            raise ValueError(
                "temperature must be between 0 and 2."
            )

        if self.base_url.endswith("/chat/completions"):
            self.endpoint_url = self.base_url
        else:
            self.endpoint_url = (
                self.base_url + "/chat/completions"
            )

    def generate(self, prompt: str) -> str:
        """Generate a response from the configured language model."""

        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string.")

        prompt = prompt.strip()

        if not prompt:
            raise ValueError("prompt cannot be empty.")

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        # Authentication is optional for local endpoints.
        if self.api_key:
            headers["Authorization"] = (
                f"Bearer {self.api_key}"
            )

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": self.temperature,
            "stream": False,
        }

        try:
            response = httpx.post(
                self.endpoint_url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )

        except httpx.TimeoutException as exc:
            raise RuntimeError(
                "The language model request timed out. "
                "Check the model service and increase LLM_TIMEOUT if necessary."
            ) from exc

        except httpx.RequestError as exc:
            raise RuntimeError(
                "Could not connect to the configured language model. "
                "Check LLM_BASE_URL, network access, and model availability."
            ) from exc

        if response.status_code >= 400:
            detail = response.text[:1000]

            raise RuntimeError(
                f"Language model API returned HTTP "
                f"{response.status_code}: {detail}"
            )

        try:
            data: Any = response.json()

        except ValueError as exc:
            raise RuntimeError(
                "The language model returned invalid JSON."
            ) from exc

        if not isinstance(data, dict):
            raise RuntimeError(
                "The language model response must be a JSON object."
            )

        choices = data.get("choices")

        if not isinstance(choices, list) or not choices:
            raise RuntimeError(
                "The language model response contains no choices."
            )

        choice = choices[0]

        if not isinstance(choice, dict):
            raise RuntimeError(
                "The language model returned an invalid choice."
            )

        message = choice.get("message")

        if not isinstance(message, dict):
            raise RuntimeError(
                "The language model response contains no message."
            )

        content = message.get("content")

        # Some compatible providers return content as text blocks.
        if isinstance(content, list):
            text_parts = []

            for item in content:
                if (
                    isinstance(item, dict)
                    and isinstance(item.get("text"), str)
                ):
                    text_parts.append(item["text"])

            content = "".join(text_parts)

        if not isinstance(content, str):
            raise RuntimeError(
                "The language model did not return text content."
            )

        answer = content.strip()

        if not answer:
            raise RuntimeError(
                "The language model returned an empty answer."
            )

        return answer
