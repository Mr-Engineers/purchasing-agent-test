"""OpenAI-compatible LLM client: proxy-server (AGENT_MODE=proxy) or Bedrock directly (only URL + key change)."""

from typing import Any

from aws_bedrock_token_generator import BedrockTokenGenerator
from botocore.session import Session
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from .config import Settings


class Llm:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = AsyncOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key.get_secret_value() if settings.llm_api_key else "unset",
            timeout=settings.llm_timeout_s,
            max_retries=settings.http_retries,
        )
        if settings.llm_api_key is None:
            # Refreshable credentials from the default chain (ECS task role, SSO profile, ...)
            self._credentials = Session().get_credentials()
            if self._credentials is None:
                raise RuntimeError("No AWS credentials found for Bedrock and LLM_API_KEY is not set")
            self._token_generator = BedrockTokenGenerator()

    async def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
                       session_id: str | None = None) -> ChatCompletion:
        options: dict[str, Any] = {}
        if self._settings.llm_api_key is None:
            # Signing is local and cheap; a fresh token per call never outlives the role credentials
            options["api_key"] = self._token_generator.get_token(self._credentials, self._settings.aws_region)
        if session_id:
            options["default_headers"] = {"X-Session-Id": session_id}

        return await self._client.with_options(**options).chat.completions.create(
            model=self._settings.llm_model,
            messages=messages,
            tools=tools,
            temperature=self._settings.llm_temperature,
            stream=False,
        )
