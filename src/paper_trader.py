"""Simple paper-trading state helpers shared by UI and worker flows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Any, Mapping

try:
    from trading_costs import TradingCostModel, cost_model_from_values
except ImportError:
    from src.trading_costs import TradingCostModel, cost_model_from_values


@dataclass(slots=True)
class PaperPosition:
    market: str
    qty: float
    entry: float
    cost: float
    opened_at: float
    strategy: str = "paper"
    entry_order_uuid: str | None = None

    @classmethod
    def from_dict(cls, market: str, raw: Mapping[str, Any]) -> "PaperPosition":
        return cls(
            market=market,
            qty=float(raw.get("qty") or 0.0),
            entry=float(raw.get("entry") or 0.0),
            cost=float(raw.get("cost") or 0.0),
            opened_at=float(raw.get("opened_at") or time.time()),
            strategy=str(raw.get("strategy") or "paper"),
            entry_order_uuid=str(raw.get("entry_order_uuid")) if raw.get("entry_order_uuid") else None,
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
        fee_rate: float = 0.0,
        slippage_bps: float = 0.0,
        qty: float | None = None,
        order_uuid: str | None = None,
        timestamp: float | None = None,
    ) -> dict[str, Any]:
        model: TradingCostModel = cost_model_from_values(fee_rate=fee_rate, slippage_bps=slippage_bps)
        simulated_fill = model.simulate_entry(price=float(price), budget=float(cost))
        effective_price = float(price) if qty is not None else simulated_fill["price"]
        resolved_qty = float(qty) if qty is not None else simulated_fill["qty"]
        position = PaperPosition(
            market=market,
            qty=resolved_qty,
            entry=effective_price,
            cost=float(cost),
            opened_at=float(timestamp or time.time()),
            strategy=strategy,
            entry_order_uuid=order_uuid,
        )
        self.positions[market] = position
        return {
            "ts": position.opened_at,
            "market": market,
            "side": "BUY",
            "price": position.entry,
            "qty": position.qty,
            "cost": position.cost,
            "fee_paid": simulated_fill["fee_paid"] if qty is None else 0.0,
            "strategy": strategy,
            "order_uuid": order_uuid,
        }

    def apply_buy_fill(
        self,
        *,
        market: str,
        qty: float,
        gross_value: float,
        fee_paid: float,
        strategy: str,
        order_uuid: str | None = None,
        timestamp: float | None = None,
    ) -> dict[str, Any] | None:
        resolved_qty = float(qty)
        if resolved_qty <= 0:
            return None

        trade_ts = float(timestamp or time.time())
        total_cost = float(gross_value) + float(fee_paid)
        fill_price = (total_cost / resolved_qty) if resolved_qty > 0 else 0.0
        position = self.positions.get(market)
        if position is None:
            position = PaperPosition(
                market=market,
                qty=resolved_qty,
                entry=fill_price,
                cost=total_cost,
                opened_at=trade_ts,
                strategy=strategy,
                entry_order_uuid=order_uuid,
            )
            self.positions[market] = position
        else:
            position.qty += resolved_qty
            position.cost += total_cost
            position.entry = (position.cost / position.qty) if position.qty > 0 else 0.0
            position.entry_order_uuid = order_uuid or position.entry_order_uuid
            position.strategy = strategy or position.strategy

        return {
            "ts": trade_ts,
            "market": market,
            "side": "BUY",
            "price": fill_price,
            "qty": resolved_qty,
            "cost": total_cost,
            "gross_value": float(gross_value),
            "fee_paid": float(fee_paid),
            "strategy": strategy,
            "order_uuid": order_uuid,
        }

    def exit_long(
        self,
        *,
        market: str,
        price: float,
        reason: str,
        fee_rate: float = 0.0,
        slippage_bps: float = 0.0,
        order_uuid: str | None = None,
        timestamp: float | None = None,
    ) -> dict[str, Any] | None:
        position = self.positions.pop(market, None)
        if not position:
            return None

        trade_ts = float(timestamp or time.time())
        model: TradingCostModel = cost_model_from_values(fee_rate=fee_rate, slippage_bps=slippage_bps)
        exit_fill = model.simulate_exit(price=float(price), qty=position.qty, cost_basis=position.cost)
        return {
            "ts": trade_ts,
            "market": market,
            "side": "SELL",
            "price": exit_fill["price"],
            "qty": position.qty,
            "entry": position.entry,
            "cost": position.cost,
            "fee_paid": exit_fill["fee_paid"],
            "net_proceeds": exit_fill["net_proceeds"],
            "pnl_value": exit_fill["pnl_value"],
            "pnl_pct": exit_fill["pnl_pct"],
            "reason": reason,
            "strategy": position.strategy,
            "order_uuid": order_uuid,
        }

    def apply_sell_fill(
        self,
        *,
        market: str,
        qty: float,
        gross_value: float,
        fee_paid: float,
        reason: str,
        order_uuid: str | None = None,
        timestamp: float | None = None,
    ) -> dict[str, Any] | None:
        position = self.positions.get(market)
        resolved_qty = min(float(qty), position.qty) if position else 0.0
        if position is None or resolved_qty <= 0:
            return None

        trade_ts = float(timestamp or time.time())
        original_entry = position.entry
        allocated_cost = position.cost * (resolved_qty / position.qty) if position.qty > 0 else 0.0
        net_proceeds = max(float(gross_value) - float(fee_paid), 0.0)
        pnl_value = net_proceeds - allocated_cost
        pnl_pct = ((net_proceeds / allocated_cost) - 1.0) * 100 if allocated_cost else 0.0
        trade_price = (net_proceeds / resolved_qty) if resolved_qty > 0 else 0.0

        remaining_qty = max(position.qty - resolved_qty, 0.0)
        remaining_cost = max(position.cost - allocated_cost, 0.0)
        if remaining_qty <= 1e-12:
            self.positions.pop(market, None)
        else:
            position.qty = remaining_qty
            position.cost = remaining_cost
            position.entry = (remaining_cost / remaining_qty) if remaining_qty > 0 else 0.0

        return {
            "ts": trade_ts,
            "market": market,
            "side": "SELL",
            "price": trade_price,
            "qty": resolved_qty,
            "entry": original_entry,
            "cost": allocated_cost,
            "gross_value": float(gross_value),
            "fee_paid": float(fee_paid),
            "net_proceeds": net_proceeds,
            "pnl_value": pnl_value,
            "pnl_pct": pnl_pct,
            "reason": reason,
            "strategy": position.strategy,
            "order_uuid": order_uuid,
        }

    def to_state(self) -> dict[str, dict[str, Any]]:
        return {market: position.to_dict() for market, position in self.positions.items()}
