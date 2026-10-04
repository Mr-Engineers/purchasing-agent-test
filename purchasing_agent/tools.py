"""Tools exposed to the LLM and their execution through the proxy.

Registering the purchase order in the warehouse is not a tool: it follows deterministically
from a successful marketplace order, so the model cannot get its fields wrong.
"""

import json
import logging
from typing import Any

from .apps import AppsClient, Outcome

log = logging.getLogger(__name__)

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_offers",
            "description": "Search the marketplace for offers of a product. Results are sorted by unit price, cheapest first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {"type": "string", "description": "Product SKU, e.g. PAP-A4-80"},
                },
                "required": ["sku"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "place_order",
            "description": (
                "Order a product from a marketplace offer. On success the order is also "
                "registered in the warehouse automatically and the task is complete."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "offer_id": {"type": "string", "description": "offer_id from search_offers results"},
                    "quantity": {"type": "integer", "minimum": 1},
                    "unit_price": {
                        "type": "string",
                        "description": "Unit price amount exactly as shown in the offer, e.g. \"118.00\"",
                    },
                    "currency": {"type": "string", "description": "Currency of the offer, e.g. PLN"},
                },
                "required": ["offer_id", "quantity", "unit_price", "currency"],
            },
        },
    },
]

BLOCKED_MESSAGE = "The action was blocked by the company security policy. Do not repeat it."
REJECTED_MESSAGE = "A human reviewer rejected this action. Take their feedback into account and decide what to do next."
EXPIRED_MESSAGE = "Nobody approved this action in time; treat it as rejected."


class Toolbox:
    def __init__(self, apps: AppsClient, session_id: str | None, item: dict[str, Any]):
        self._apps = apps
        self._session_id = session_id
        self._item = item
        self.order: dict[str, Any] | None = None
        self.purchase_order: dict[str, Any] | None = None

    @property
    def done(self) -> bool:
        return self.order is not None

    async def execute(self, name: str, raw_args: str) -> dict[str, Any]:
        try:
            args = json.loads(raw_args or "{}")
        except json.JSONDecodeError as exc:
            return {"status": "error", "message": f"Arguments are not valid JSON: {exc}"}

        match name:
            case "search_offers":
                return await self._search_offers(**args)
            case "place_order":
                return await self._place_order(**args)
            case _:
                return {"status": "error", "message": f"Unknown tool {name!r}"}

    async def _search_offers(self, sku: str, **_: Any) -> dict[str, Any]:
        outcome = await self._apps.call(self._session_id, "marketplace", "GET", "/search", params={"sku": sku})
        return _for_model(outcome)

    async def _place_order(
        self, offer_id: str, quantity: int, unit_price: str, currency: str = "PLN", **_: Any
    ) -> dict[str, Any]:
        outcome = await self._apps.call(
            self._session_id,
            "marketplace",
            "POST",
            "/orders",
            json={
                "offer_id": offer_id,
                "quantity": quantity,
                "expected_unit_price": {"amount": str(unit_price), "currency": currency},
            },
        )
        if outcome.status != "ok":
            return _for_model(outcome)

        self.order = outcome.body
        log.info("order %s placed: %s x %s", self.order.get("order_id"), self.order.get("quantity"), self.order.get("sku"))
        po = await self._register_purchase_order(self.order)
        return {"status": "ok", "order": self.order, "warehouse_purchase_order": po}

    async def _register_purchase_order(self, order: dict[str, Any]) -> dict[str, Any]:
        outcome = await self._apps.call(
            self._session_id,
            "warehouse",
            "POST",
            "/purchase-orders",
            json={
                "sku": order["sku"],
                "quantity": order["quantity"],
                "unit_price": order["unit_price"],
                "supplier": {
                    "marketplace_order_id": order["order_id"],
                    "merchant_id": order["merchant_id"],
                },
            },
        )
        if outcome.status == "ok":
            self.purchase_order = outcome.body
            log.info("purchase order %s registered", outcome.body.get("id"))
        else:
            log.error("order %s placed but purchase order not registered: %s %s",
                      order["order_id"], outcome.status, outcome.body)
        return _for_model(outcome)


def _for_model(outcome: Outcome) -> dict[str, Any]:
    """What the model sees: generic codes only, no proxy reasoning (D10)."""
    match outcome.status:
        case "ok":
            return {"status": "ok", "result": outcome.body}
        case "blocked":
            return {"status": "blocked", "message": BLOCKED_MESSAGE}
        case "rejected":
            return {"status": "rejected", "message": REJECTED_MESSAGE, "feedback": outcome.feedback}
        case "expired":
            return {"status": "expired", "message": EXPIRED_MESSAGE}
        case _:
            return {"status": "error", "http_status": outcome.http_status, "error": outcome.body}
