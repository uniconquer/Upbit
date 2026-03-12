"""Shared trading-cost utilities for backtests and simulated execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


@dataclass(slots=True)
class TradingCostModel:
    fee_rate: float = 0.0005
    slippage_bps: float = 3.0

    @property
    def slippage_rate(self) -> float:
        return max(self.slippage_bps, 0.0) / 10000.0

    def buy_price(self, price: float) -> float:
        return float(price) * (1.0 + self.slippage_rate)

    def sell_price(self, price: float) -> float:
        return float(price) * (1.0 - self.slippage_rate)

    def simulate_entry(self, *, price: float, budget: float) -> dict[str, float]:
        effective_price = self.buy_price(price)
        gross_qty = (float(budget) / effective_price) if effective_price > 0 else 0.0
        fee_paid = float(budget) * self.fee_rate
        net_budget = max(float(budget) - fee_paid, 0.0)
        qty = (net_budget / effective_price) if effective_price > 0 else 0.0
        return {
            "price": effective_price,
            "qty": qty,
            "gross_qty": gross_qty,
            "fee_paid": fee_paid,
            "cost": float(budget),
        }

    def simulate_exit(self, *, price: float, qty: float, cost_basis: float) -> dict[str, float]:
        effective_price = self.sell_price(price)
        gross_proceeds = float(qty) * effective_price
        fee_paid = gross_proceeds * self.fee_rate
        net_proceeds = gross_proceeds - fee_paid
        pnl_value = net_proceeds - float(cost_basis)
        pnl_pct = ((net_proceeds / float(cost_basis)) - 1.0) * 100 if cost_basis else 0.0
        return {
            "price": effective_price,
            "gross_proceeds": gross_proceeds,
            "net_proceeds": net_proceeds,
            "fee_paid": fee_paid,
            "qty": float(qty),
            "pnl_value": pnl_value,
            "pnl_pct": pnl_pct,
        }

    def unrealized_pnl(self, *, entry_cost: float, qty: float, market_price: float) -> float:
        return self.simulate_exit(price=market_price, qty=qty, cost_basis=entry_cost)["pnl_value"]


def cost_model_from_values(*, fee_rate: float = 0.0005, slippage_bps: float = 3.0) -> TradingCostModel:
    return TradingCostModel(fee_rate=max(float(fee_rate), 0.0), slippage_bps=max(float(slippage_bps), 0.0))


def cost_model_from_mapping(raw: Mapping[str, Any] | None) -> TradingCostModel:
    raw = raw or {}
    return cost_model_from_values(
        fee_rate=_to_float(raw.get("fee_rate") if "fee_rate" in raw else raw.get("fee")),
        slippage_bps=_to_float(raw.get("slippage_bps")),
    )
