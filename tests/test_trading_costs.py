from __future__ import annotations

from src.paper_trader import PaperTrader
from src.trading_costs import cost_model_from_values


def test_cost_model_entry_and_exit_apply_fee_and_slippage():
    model = cost_model_from_values(fee_rate=0.001, slippage_bps=10.0)
    entry = model.simulate_entry(price=100.0, budget=10000.0)
    exit_fill = model.simulate_exit(price=110.0, qty=entry["qty"], cost_basis=10000.0)

    assert entry["price"] > 100.0
    assert entry["qty"] < 100.0
    assert exit_fill["price"] < 110.0
    assert exit_fill["net_proceeds"] < entry["qty"] * 110.0


def test_paper_trader_uses_cost_model_for_simulated_round_trip():
    trader = PaperTrader()
    buy_event = trader.enter_long(
        market="KRW-BTC",
        price=100.0,
        cost=10000.0,
        strategy="research_trend",
        fee_rate=0.001,
        slippage_bps=10.0,
    )
    sell_event = trader.exit_long(
        market="KRW-BTC",
        price=110.0,
        reason="signal",
        fee_rate=0.001,
        slippage_bps=10.0,
    )

    assert buy_event["price"] > 100.0
    assert sell_event is not None
    assert sell_event["price"] < 110.0
    assert sell_event["net_proceeds"] < buy_event["qty"] * 110.0
