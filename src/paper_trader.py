"""Simple paper-trading state helpers shared by UI and worker flows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Any, Mapping


@dataclass(slots=True)
class PaperPosition:
    market: str
    qty: float
    entry: float
    cost: float
    opened_at: float
    strategy: str = "paper"

    @classmethod
    def from_dict(cls, market: str, raw: Mapping[str, Any]) -> "PaperPosition":
        return cls(
            market=market,
            qty=float(raw.get("qty") or 0.0),
            entry=float(raw.get("entry") or 0.0),
            cost=float(raw.get("cost") or 0.0),
            opened_at=float(raw.get("opened_at") or time.time()),
            strategy=str(raw.get("strategy") or "paper"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PaperTrader:
    def __init__(self, positions: Mapping[str, Mapping[str, Any]] | None = None):
        self.positions: dict[str, PaperPosition] = {}
        for market, raw in (positions or {}).items():
            self.positions[market] = PaperPosition.from_dict(market, raw)

    def has_position(self, market: str) -> bool:
        return market in self.positions

    def get_position(self, market: str) -> PaperPosition | None:
        return self.positions.get(market)

    def unrealized_for(self, market: str, price: float) -> float:
        position = self.get_position(market)
        if not position:
            return 0.0
        return (float(price) - position.entry) * position.qty

    def exposure_for(self, market: str) -> float:
        position = self.get_position(market)
        return position.cost if position else 0.0

    def mark_to_market(self, price_map: Mapping[str, float]) -> dict[str, float]:
        return {
            market: self.unrealized_for(market, price)
            for market, price in price_map.items()
            if market in self.positions
        }

    def enter_long(
        self,
        *,
        market: str,
        price: float,
        cost: float,
        strategy: str,
        timestamp: float | None = None,
    ) -> dict[str, Any]:
        qty = (float(cost) / float(price)) if price else 0.0
        position = PaperPosition(
            market=market,
            qty=qty,
            entry=float(price),
            cost=float(cost),
            opened_at=float(timestamp or time.time()),
            strategy=strategy,
        )
        self.positions[market] = position
        return {
            "ts": position.opened_at,
            "market": market,
            "side": "BUY",
            "price": position.entry,
            "qty": position.qty,
            "cost": position.cost,
            "strategy": strategy,
        }

    def exit_long(
        self,
        *,
        market: str,
        price: float,
        reason: str,
        timestamp: float | None = None,
    ) -> dict[str, Any] | None:
        position = self.positions.pop(market, None)
        if not position:
            return None

        trade_ts = float(timestamp or time.time())
        pnl_value = (float(price) - position.entry) * position.qty
        pnl_pct = ((float(price) / position.entry) - 1.0) * 100 if position.entry else 0.0
        return {
            "ts": trade_ts,
            "market": market,
            "side": "SELL",
            "price": float(price),
            "qty": position.qty,
            "entry": position.entry,
            "cost": position.cost,
            "pnl_value": pnl_value,
            "pnl_pct": pnl_pct,
            "reason": reason,
            "strategy": position.strategy,
        }

    def to_state(self) -> dict[str, dict[str, Any]]:
        return {market: position.to_dict() for market, position in self.positions.items()}
