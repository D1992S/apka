"""
LLM PROVIDER ABSTRACTION
=========================
Wspólny interfejs dla różnych dostawców LLM (OpenAI, Google AI Studio, etc.)
Ułatwia dodawanie nowych providerów i ujednolica obsługę błędów.
"""

import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple


class LLMProvider(ABC):
    """Abstrakcyjny interfejs dla dostawców LLM"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Nazwa providera (np. 'openai', 'google')"""
        pass

    @abstractmethod
    def chat_complete(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict] = None,
        **kwargs
    ) -> str:
        """
        Wysyła zapytanie do LLM i zwraca odpowiedź.

        Args:
            messages: Lista wiadomości [{"role": "user/system/assistant", "content": "..."}]
            model: Opcjonalny model (jeśli None, użyje domyślnego)
            temperature: Temperatura generowania (0.0-1.0)
            max_tokens: Maksymalna liczba tokenów odpowiedzi
            response_format: Format odpowiedzi (np. {"type": "json_object"})

        Returns:
            Tekst odpowiedzi od modelu
        """
        pass

    @abstractmethod
    def test_connection(self) -> Tuple[bool, str]:
        """
        Testuje połączenie z API.

        Returns:
            (success, message) - czy udało się połączyć
        """
        pass

    def safe_chat_complete(
        self,
        messages: List[Dict[str, str]],
        max_retries: int = 2,
        **kwargs
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Bezpieczne wywołanie z retry i obsługą błędów.

        Returns:
            (response, error_message) - jeśli error_message is None, sukces
        """
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                result = self.chat_complete(messages, **kwargs)
                return result, None
            except Exception as e:
                last_error = e
                error_msg = str(e).lower()

                # Błędy które nie mają sensu retryować
                if any(x in error_msg for x in ["invalid_api_key", "incorrect api key", "authentication"]):
                    return None, "Nieprawidłowy klucz API"
                if "insufficient_quota" in error_msg:
                    return None, "Brak środków na koncie API"

                # Błędy które warto retryować
                if any(x in error_msg for x in ["rate limit", "rate_limit", "too many requests"]):
                    if attempt < max_retries:
                        time.sleep(2 ** attempt)
                        continue
                    return None, "Przekroczono limit zapytań. Spróbuj ponownie za chwilę."

                if any(x in error_msg for x in ["timeout", "connection", "network"]):
                    if attempt < max_retries:
                        time.sleep(1)
                        continue
                    return None, "Problem z połączeniem. Sprawdź internet."

                if attempt < max_retries:
                    time.sleep(1)
                    continue

        return None, f"Błąd API: {str(last_error)[:100]}"


class OpenAIProvider(LLMProvider):
    """Provider dla OpenAI API (GPT-4, GPT-4o, etc.)"""

    def __init__(self, api_key: str, default_model: str = "gpt-4o"):
        self.api_key = api_key
        self.default_model = default_model
        self._client = None

    @property
    def name(self) -> str:
        return "openai"

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, timeout=60.0)
        return self._client

    def chat_complete(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict] = None,
        **kwargs
    ) -> str:
        model = model or self.default_model

        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }

        if max_tokens:
            params["max_tokens"] = max_tokens
        if response_format:
            params["response_format"] = response_format

        params.update(kwargs)

        response = self.client.chat.completions.create(**params)
        return response.choices[0].message.content

    def test_connection(self) -> Tuple[bool, str]:
        try:
            response = self.client.chat.completions.create(
                model=self.default_model,
                messages=[{"role": "user", "content": "Test. Reply: OK"}],
                max_tokens=5,
                temperature=0
            )
            if response.choices and response.choices[0].message.content:
                return True, f"Połączono z OpenAI ({self.default_model})"
            return False, "Brak odpowiedzi z API"
        except Exception as e:
            error_msg = str(e)
            if "Incorrect API key" in error_msg or "invalid_api_key" in error_msg:
                return False, "Nieprawidłowy klucz API"
            elif "Rate limit" in error_msg:
                return False, "Przekroczono limit zapytań"
            elif "insufficient_quota" in error_msg:
                return False, "Brak środków na koncie"
            return False, f"Błąd: {error_msg[:50]}"


class GoogleAIProvider(LLMProvider):
    """Provider dla Google AI Studio (Gemini)"""

    def __init__(self, api_key: str, default_model: str = "gemini-1.5-pro"):
        self.api_key = api_key
        self.default_model = default_model
        self._configured = False

    @property
    def name(self) -> str:
        return "google"

    def _ensure_configured(self):
        if not self._configured:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._configured = True

    def chat_complete(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict] = None,
        **kwargs
    ) -> str:
        import google.generativeai as genai

        self._ensure_configured()
        model_name = model or self.default_model
        llm = genai.GenerativeModel(model_name)

        # Konwertuj format messages do Gemini
        # Gemini używa innego formatu - dla prostoty łączymy w jeden prompt
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"[System instruction]: {content}\n")
            elif role == "assistant":
                prompt_parts.append(f"[Previous response]: {content}\n")
            else:
                prompt_parts.append(content)

        full_prompt = "\n".join(prompt_parts)

        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
        )
        if max_tokens:
            generation_config.max_output_tokens = max_tokens

        response = llm.generate_content(full_prompt, generation_config=generation_config)
        return response.text

    def test_connection(self) -> Tuple[bool, str]:
        try:
            import google.generativeai as genai
            self._ensure_configured()
            llm = genai.GenerativeModel(self.default_model)
            response = llm.generate_content(
                "Test. Reply: OK",
                generation_config=genai.types.GenerationConfig(max_output_tokens=5, temperature=0)
            )
            if response.text:
                return True, f"Połączono z Google AI ({self.default_model})"
            return False, "Brak odpowiedzi z API"
        except Exception as e:
            error_msg = str(e)
            if "API_KEY_INVALID" in error_msg or "INVALID_ARGUMENT" in error_msg:
                return False, "Nieprawidłowy klucz API"
            elif "RESOURCE_EXHAUSTED" in error_msg:
                return False, "Przekroczono limit zapytań"
            return False, f"Błąd: {error_msg[:50]}"


def get_provider(provider_name: str, api_key: str, model: str = None) -> Optional[LLMProvider]:
    """
    Factory function - tworzy odpowiedni provider na podstawie nazwy.

    Args:
        provider_name: 'openai' lub 'google'
        api_key: Klucz API
        model: Opcjonalny model (jeśli None, użyje domyślnego)

    Returns:
        Instancja LLMProvider lub None jeśli provider nieznany
    """
    if not api_key:
        return None

    if provider_name == "openai":
        return OpenAIProvider(api_key, model or "gpt-4o")
    elif provider_name == "google":
        return GoogleAIProvider(api_key, model or "gemini-1.5-pro")
    else:
        return None


# Aliasy dla wstecznej kompatybilności
def create_openai_client(api_key: str, model: str = "gpt-4o") -> OpenAIProvider:
    """Tworzy OpenAI provider (alias dla wstecznej kompatybilności)"""
    return OpenAIProvider(api_key, model)


def create_google_client(api_key: str, model: str = "gemini-1.5-pro") -> GoogleAIProvider:
    """Tworzy Google AI provider (alias dla wstecznej kompatybilności)"""
    return GoogleAIProvider(api_key, model)
