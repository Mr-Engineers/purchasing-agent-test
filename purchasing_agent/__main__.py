"""Single run: scan the warehouse once, run one restock session per SKU in parallel, exit."""

import asyncio
import logging
import sys

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


async def main() -> int:
    """One pass: scan the warehouse once, restock every low-stock SKU, exit.

    Exit code 1 when the scan or any session failed, 0 otherwise.
    """
    settings = Settings()
    log.info(
        "mode=%s llm=%s warehouse=%s marketplace=%s",
        settings.agent_mode, settings.llm_base_url, settings.warehouse_url, settings.marketplace_url,
    )
    apps = AppsClient(settings)
    llm = Llm(settings)
    semaphore = asyncio.Semaphore(settings.max_parallel_sessions)

    async def restock(sku: str) -> SessionResult:
        async with semaphore:
            try:
                result = await run_restock(apps, llm, settings, sku)
            except Exception:
                log.exception("[%s] session failed", sku)
                result = SessionResult(sku, None, False, "error")
        log.info("[%s] session finished: ordered=%s, %s", sku, result.ordered, result.summary)
        return result

    try:
        try:
            skus = await scan(apps)
        except Exception:
            log.exception("scan failed")
            return 1

        # A SKU may appear more than once in the scan; restock it once
        skus = list(dict.fromkeys(skus))
        log.info("scan: %d low-stock item(s)", len(skus))
        results = await asyncio.gather(*(restock(sku) for sku in skus))
    finally:
        await apps.aclose()

    ordered = sum(result.ordered for result in results)
    failed = sum(result.summary == "error" for result in results)
    log.info("done: %d session(s), %d ordered, %d failed", len(results), ordered, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    sys.exit(asyncio.run(main()))
