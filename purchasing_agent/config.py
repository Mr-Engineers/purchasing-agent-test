from typing import Literal

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """AGENT_MODE picks how the agent reaches the LLM and the apps.

    proxy  - everything through proxy-server, which authenticates the agent (AGENT_KEY)
             and adds the apps' credentials. Only PROXY_URL is needed; unless set explicitly:
               LLM_BASE_URL=<proxy>/v1
               WAREHOUSE_URL=<proxy>/apps/warehouse  MARKETPLACE_URL=<proxy>/apps/marketplace
    direct - for tests: Bedrock and the apps without the proxy, with their own tokens
             (WAREHOUSE_URL and MARKETPLACE_URL required):
               WAREHOUSE_URL=http://test-backend:8000/api/v1  WAREHOUSE_TOKEN=<test-backend gateway token>
               MARKETPLACE_URL=<backend-2>  MARKETPLACE_TOKEN=<marketplace API token>
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    agent_mode: Literal["proxy", "direct"] = "proxy"

    # OpenAI-compatible Chat Completions endpoint
    llm_base_url: str = "https://bedrock-runtime.eu-north-1.amazonaws.com/openai/v1"
    llm_model: str = "qwen.qwen3-32b-v1:0"
    # None = short-term Bedrock token from the AWS credential chain (task role / SSO profile)
    llm_api_key: SecretStr | None = None
    aws_region: str = "eu-north-1"
    # Qwen3 thinking mode: much slower, rarely needed for this task
    llm_think: bool = False
    llm_temperature: float = 0.2
    llm_timeout_s: float = 180

    warehouse_url: str = "http://localhost:8000/api/v1"
    marketplace_url: str = "http://localhost:8002"

    # proxy mode: proxy-server address and the agent's key for it (also the LLM key)
    proxy_url: str | None = None
    agent_key: SecretStr | None = None
    # Sessions (X-Session-Id) need POST /v1/sessions on the proxy, which it does not have yet
    proxy_sessions: bool = False

    # direct mode: sent as Authorization: Bearer to each app
    warehouse_token: SecretStr | None = None
    marketplace_token: SecretStr | None = None
    # Sent to the warehouse as X-On-Behalf-Of in direct mode (the proxy sets it otherwise)
    agent_id: str = "purchasing-agent"

    max_parallel_sessions: int = 4
    max_llm_steps: int = 10

    http_timeout_s: float = 30
    # Retries on 502/503/504 and connection errors (D14); never on 403
    http_retries: int = 3
    # Long-poll window for GET /v1/approvals/{id}?wait= (D11)
    approval_wait_s: int = 30
    # Safety net above the proxy's own approval timeout (15 min)
    approval_deadline_s: float = 20 * 60

    @model_validator(mode="after")
    def _check_mode(self) -> "Settings":
        explicit = self.model_fields_set
        if self.agent_mode == "proxy":
            if not self.proxy_url:
                raise ValueError("AGENT_MODE=proxy requires PROXY_URL")
            proxy = self.proxy_url.rstrip("/")
            self.proxy_url = proxy
            # Everything goes through the proxy unless a URL is set explicitly
            if "llm_base_url" not in explicit:
                self.llm_base_url = f"{proxy}/v1"
            if "warehouse_url" not in explicit:
                self.warehouse_url = f"{proxy}/apps/warehouse"
            if "marketplace_url" not in explicit:
                self.marketplace_url = f"{proxy}/apps/marketplace"
            # The proxy signs the LLM calls itself; the agent only identifies with its key
            if self.llm_api_key is None and self.llm_base_url.startswith(proxy):
                self.llm_api_key = self.agent_key or SecretStr("none")
        else:
            missing = [name.upper() for name in ("warehouse_url", "marketplace_url") if name not in explicit]
            if missing:
                raise ValueError(f"AGENT_MODE=direct requires {', '.join(missing)}")
            # No proxy: no sessions or approvals, the apps get their own tokens
            self.proxy_url = None
            self.proxy_sessions = False
        return self

    @property
    def uses_proxy(self) -> bool:
        return self.agent_mode == "proxy"
