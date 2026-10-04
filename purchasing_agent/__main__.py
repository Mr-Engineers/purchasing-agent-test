"""Main loop: scan the warehouse, run one restock session per SKU in parallel."""

import asyncio
import logging
import time

from .apps import AppsClient
from .config import Settings
from .llm import Llm
from .session import SessionResult, run_restock

log = logging.getLogger("purchasing_agent")


async def scan(apps: AppsClient) -> list[str]:
    session_id = await apps.create_session(task="Scan warehouse for items below reorder threshold")
    outcome = await apps.call(session_id, "warehouse", "GET", "/low-stock")
    if outcome.status != "ok":
        log.error("low-stock scan failed: %s %s", outcome.status, outcome.body)
        return []
    return [item["sku"] for item in outcome.body.get("items", []) if item["qty_needed"] > 0]


async def main() -> None:
    settings = Settings()
    log.info(
        "mode=%s llm=%s warehouse=%s marketplace=%s",
        settings.agent_mode, settings.llm_base_url, settings.warehouse_url, settings.marketplace_url,
    )
    apps = AppsClient(settings)
    llm = Llm(settings)
    semaphore = asyncio.Semaphore(settings.max_parallel_sessions)
    in_flight: dict[str, asyncio.Task[SessionResult]] = {}
    cooldown_until: dict[str, float] = {}

    async def restock(sku: str) -> SessionResult:
        async with semaphore:
            try:
                result = await run_restock(apps, llm, settings, sku)
            except Exception:
                log.exception("[%s] session failed", sku)
                result = SessionResult(sku, None, False, "error")
        if not result.ordered:
            cooldown_until[sku] = time.monotonic() + settings.sku_retry_cooldown_s
        log.info("[%s] session finished: ordered=%s, %s", sku, result.ordered, result.summary)
        return result

    try:
        while True:
            try:
                skus = await scan(apps)
            except Exception:
                log.exception("scan failed")
                skus = []

            now = time.monotonic()
            for sku in skus:
                if sku in in_flight or cooldown_until.get(sku, 0) > now:
                    continue
                task = asyncio.create_task(restock(sku))
                in_flight[sku] = task
                task.add_done_callback(lambda _, sku=sku: in_flight.pop(sku, None))
            log.info("scan: %d low-stock item(s), %d session(s) running", len(skus), len(in_flight))

            if settings.run_once:
                await asyncio.gather(*in_flight.values())
                return
            await asyncio.sleep(settings.poll_interval_s)
    finally:
        await apps.aclose()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.run(main())
