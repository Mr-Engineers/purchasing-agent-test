"""One session = one task = restocking one SKU (D11)."""

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from openai import PermissionDeniedError

from .apps import AppsClient, SessionTerminated
from .config import Settings
from .llm import Llm
from .tools import TOOLS, Toolbox

log = logging.getLogger(__name__)

THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)

SYSTEM_PROMPT = """\
You are the purchasing agent of an office-supplies warehouse. In this session you restock exactly one product.

Mandate:
- Buy only the product with SKU {sku} ({name}).
- Buy exactly {qty_needed} {unit}. Never order a different quantity.
- Prefer the cheapest offer that has enough available_qty.
- Use the unit price exactly as shown in the chosen offer.

How to work:
1. Call search_offers for the SKU.
2. Pick an offer and call place_order.
3. If an action is blocked, do not repeat it; you may choose another offer.
4. If a human rejects an action, follow their feedback.
5. When the order is placed, or nothing more can be done, reply with a one-sentence summary and no tool calls.
"""


@dataclass
class SessionResult:
    sku: str
    session_id: str | None
    ordered: bool
    summary: str


async def run_restock(apps: AppsClient, llm: Llm, settings: Settings, sku: str) -> SessionResult:
    session_id = await apps.create_session(task=f"Restock warehouse item {sku}")
    log.info("[%s] session %s started", sku, session_id)
    try:
        return await _run(apps, llm, settings, sku, session_id)
    except SessionTerminated:
        log.warning("[%s] session %s terminated by proxy", sku, session_id)
        return SessionResult(sku, session_id, False, "session terminated by proxy")


async def _run(apps: AppsClient, llm: Llm, settings: Settings, sku: str, session_id: str | None) -> SessionResult:
    # Read stock inside the session so the proxy records qty_needed in session state
    stock = await apps.call(session_id, "warehouse", "GET", "/low-stock")
    if stock.status != "ok":
        return SessionResult(sku, session_id, False, f"low-stock read failed: {stock.status}")
    item = next((i for i in stock.body.get("items", []) if i["sku"] == sku), None)
    if item is None or item["qty_needed"] <= 0:
        return SessionResult(sku, session_id, False, "nothing to order")

    system = SYSTEM_PROMPT.format(**item)
    if not settings.llm_think:
        system += "\n/no_think"
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": "Warehouse item to restock:\n" + json.dumps(item, ensure_ascii=False)},
    ]

    toolbox = Toolbox(apps, session_id, item)
    summary = "step limit reached"

    for step in range(1, settings.max_llm_steps + 1):
        try:
            completion = await llm.complete(messages, TOOLS, session_id)
        except PermissionDeniedError as exc:
            if isinstance(exc.body, dict) and exc.body.get("status") == "session_terminated":
                raise SessionTerminated(session_id) from exc
            raise

        message = completion.choices[0].message
        content = THINK_BLOCK.sub("", message.content or "").strip()
        messages.append(_assistant_message(message, content))

        if not message.tool_calls:
            summary = content or "model ended without a summary"
            log.info("[%s] step %d: done: %s", sku, step, summary)
            break

        for call in message.tool_calls:
            log.info("[%s] step %d: %s(%s)", sku, step, call.function.name, call.function.arguments)
            result = await toolbox.execute(call.function.name, call.function.arguments)
            log.info("[%s] step %d: %s -> %s", sku, step, call.function.name, result["status"])
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result, ensure_ascii=False),
            })

        if toolbox.done:
            summary = f"ordered {toolbox.order['quantity']} x {sku} (order {toolbox.order['order_id']})"
            break

    return SessionResult(sku, session_id, toolbox.done, summary)


def _assistant_message(message: Any, content: str) -> dict[str, Any]:
    entry: dict[str, Any] = {"role": "assistant", "content": content}
    if message.tool_calls:
        entry["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.function.name, "arguments": call.function.arguments},
            }
            for call in message.tool_calls
        ]
    return entry
