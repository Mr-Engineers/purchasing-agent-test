"""HTTP client for the apps (warehouse, marketplace), through proxy-server or directly (AGENT_MODE).

Maps responses onto a small set of outcomes the agent reasons about:
ok | error | blocked | rejected | expired. `session_terminated` is raised, because the
session cannot continue. blocked / pending_approval / session_terminated only come from
the proxy (ADR 0004); talking to the apps directly yields ok | error.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Literal

import httpx
from pydantic import SecretStr

from .config import Settings

log = logging.getLogger(__name__)

RETRYABLE_STATUSES = {502, 503, 504}


class SessionTerminated(Exception):
    """Proxy ended the session (too many denials) - stop the task."""


@dataclass
class Outcome:
    status: Literal["ok", "error", "blocked", "rejected", "expired"]
    http_status: int
    body: Any = None
    feedback: str | None = None
    decision_id: str | None = None


class AppsClient:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._http = httpx.AsyncClient(timeout=settings.http_timeout_s)
        self._base_urls = {
            "warehouse": settings.warehouse_url.rstrip("/"),
            "marketplace": settings.marketplace_url.rstrip("/"),
        }
        if settings.uses_proxy:
            # The proxy strips these and adds each app's own credentials
            proxy_headers = _bearer(settings.agent_key)
            self._proxy_headers = proxy_headers
            self._app_headers = {"warehouse": proxy_headers, "marketplace": proxy_headers}
        else:
            self._proxy_headers = {}
            self._app_headers = {
                # The test-backend trusts X-On-Behalf-Of only with its gateway token
                "warehouse": {**_bearer(settings.warehouse_token), "X-On-Behalf-Of": settings.agent_id},
                "marketplace": _bearer(settings.marketplace_token),
            }

    async def aclose(self) -> None:
        await self._http.aclose()

    async def create_session(self, task: str) -> str | None:
        """Proxy-issued session (D7); None in direct mode or while the proxy has no sessions."""
        if not self._settings.proxy_sessions:
            return None
        response = await self._send(
            "POST", f"{self._settings.proxy_url}/v1/sessions", json={"task": task}, headers=self._proxy_headers
        )
        response.raise_for_status()
        return response.json()["session_id"]

    async def call(
        self,
        session_id: str | None,
        app: Literal["warehouse", "marketplace"],
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: Any = None,
    ) -> Outcome:
        headers = dict(self._app_headers[app])
        if session_id:
            headers["X-Session-Id"] = session_id
        if not self._settings.uses_proxy:
            # The proxy generates it otherwise; the warehouse stores it with the purchase order
            headers["X-Request-Id"] = str(uuid.uuid4())
        if method == "POST":
            # One key per logical action, reused across retries
            headers["Idempotency-Key"] = str(uuid.uuid4())

        url = self._base_urls[app] + path
        response = await self._send(method, url, params=params, json=json, headers=headers)
        body = _json_or_text(response)
        status = body.get("status") if isinstance(body, dict) else None

        if response.status_code == 403 and status == "session_terminated":
            raise SessionTerminated(session_id)
        if response.status_code == 403 and status == "blocked":
            return Outcome("blocked", 403, decision_id=body.get("decision_id"))
        if response.status_code == 202 and status == "pending_approval":
            return await self._wait_for_approval(session_id, body)
        if response.is_success:
            return Outcome("ok", response.status_code, body)
        return Outcome("error", response.status_code, body)

    async def _wait_for_approval(self, session_id: str | None, pending: dict[str, Any]) -> Outcome:
        approval_id = pending["approval_id"]
        poll_url = f"{self._settings.proxy_url}{pending.get('poll_url') or f'/v1/approvals/{approval_id}'}"
        decision_id = pending.get("decision_id")
        log.info("approval %s pending (decision %s), waiting for a human", approval_id, decision_id)

        deadline = time.monotonic() + self._settings.approval_deadline_s
        while time.monotonic() < deadline:
            response = await self._send(
                "GET",
                poll_url,
                params={"wait": self._settings.approval_wait_s},
                headers={**self._proxy_headers, **({"X-Session-Id": session_id} if session_id else {})},
                timeout=self._settings.approval_wait_s + self._settings.http_timeout_s,
            )
            body = _json_or_text(response)
            status = body.get("status") if isinstance(body, dict) else None

            if response.status_code == 403 and status == "session_terminated":
                raise SessionTerminated(session_id)
            if not response.is_success:
                return Outcome("error", response.status_code, body, decision_id=decision_id)

            match status:
                case "pending":
                    continue
                case "approved":
                    log.info("approval %s approved", approval_id)
                    return Outcome("ok", 200, body.get("result"), decision_id=decision_id)
                case "rejected":
                    log.info("approval %s rejected: %r", approval_id, body.get("feedback"))
                    return Outcome("rejected", 200, feedback=body.get("feedback") or None, decision_id=decision_id)
                case "expired":
                    log.info("approval %s expired", approval_id)
                    return Outcome("expired", 200, decision_id=decision_id)
                case _:
                    return Outcome("error", response.status_code, body, decision_id=decision_id)

        log.warning("approval %s: gave up waiting", approval_id)
        return Outcome("expired", 200, decision_id=decision_id)

    async def _send(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        attempts = self._settings.http_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                response = await self._http.request(method, url, **kwargs)
            except httpx.TransportError as exc:
                if attempt == attempts:
                    raise
                log.warning("%s %s failed (%s), retry %d/%d", method, url, exc, attempt, attempts - 1)
            else:
                if response.status_code not in RETRYABLE_STATUSES or attempt == attempts:
                    return response
                log.warning("%s %s -> %d, retry %d/%d", method, url, response.status_code, attempt, attempts - 1)
            await asyncio.sleep(2 ** (attempt - 1))
        raise AssertionError("unreachable")


def _bearer(token: SecretStr | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token.get_secret_value()}"} if token else {}


def _json_or_text(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return response.text
