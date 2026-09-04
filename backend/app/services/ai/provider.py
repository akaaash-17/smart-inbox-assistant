from abc import ABC, abstractmethod

import requests


class AIProviderError(Exception):
    """Raised when an AI provider cannot complete a request."""


class AIProvider(ABC):
    """
    Abstract interface for AI model providers.

    The rest of the application depends on this interface rather
    than a specific model vendor.
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate a response from the configured AI model.
        """
        raise NotImplementedError


class OllamaProvider(AIProvider):
    """
    Local AI provider using Ollama's HTTP API.
    """

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        """
        Send a prompt to Ollama and return the generated text.
        """

        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise AIProviderError(
                f"Unable to connect to Ollama: {exc}"
            ) from exc

        if response.status_code != 200:
            raise AIProviderError(
                "Ollama returned HTTP "
                f"{response.status_code}: {response.text}"
            )

        try:
            response_data = response.json()
        except ValueError as exc:
            raise AIProviderError(
                "Ollama returned an invalid JSON response."
            ) from exc

        generated_text = response_data.get("response")

        if not isinstance(generated_text, str):
            raise AIProviderError(
                "Ollama response did not contain generated text."
            )

        return generated_text.strip()