"""Risk-sized long-only portfolio. All fills are local PaperTrader simulations."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
import pandas as pd

from src.paper_trader import PaperTrader
from src.strategy import average_true_range
from src.trading_costs import cost_model_from_values


@dataclass(frozen=True)
class PortfolioConfig:
    entry_mode: str = 'breakout'
    trend_window: int = 120
    breakout_window: int = 20
    stop_atr: float = 3.0
    macro_window: int = 200
    risk_fraction: float = .0075
    asset_fraction: float = .4
    exposure_fraction: float = .6
    max_positions: int = 2
    drawdown_limit: float = .10
    daily_loss_limit: float = .02
    min_order: float = 5000.
    fee: float = .0005
    slippage_bps: float = 10.
    context_minutes: int = 1
    edge_cost_ratio: float = 0.
    exit_confirm_bars: int = 1
    min_stop_fraction: float = 0.

    def __post_init__(self):
        if self.entry_mode not in {'breakout', 'trend', 'pullback'}:
            raise ValueError('Invalid entry_mode')
        for name, value in asdict(self).items():
            if name == 'entry_mode':
                continue
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {name}")
        for name in ('trend_window', 'breakout_window', 'macro_window', 'max_positions', 'context_minutes', 'exit_confirm_bars'):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"Invalid {name}")
        for name in ('risk_fraction', 'asset_fraction', 'exposure_fraction', 'drawdown_limit', 'daily_loss_limit'):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"Invalid {name}")
        if self.stop_atr <= 0 or self.fee >= 1 or self.slippage_bps >= 10000:
            raise ValueError("Invalid costs or stop distance")
        if self.min_stop_fraction >= 1 or self.context_minutes > 60 or self.exit_confirm_bars > 30:
            raise ValueError('Invalid research feature bounds')


def validate_frames(raw_by_market):
    if not raw_by_market or 'KRW-BTC' not in raw_by_market:
        raise ValueError('KRW-BTC macro reference is required')
    index = raw_by_market['KRW-BTC'].index
    if not isinstance(index, pd.DatetimeIndex) or not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError('Unique chronological DatetimeIndex required')
    for market, raw in raw_by_market.items():
        if not raw.index.equals(index):
            raise ValueError(f'Unaligned candles: {market}')
        values = raw[['open', 'high', 'low', 'close', 'volume']].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values[:, :4] <= 0).any() or (values[:, 4] < 0).any():
            raise ValueError(f'Invalid candles: {market}')
        if ((raw.high < raw[['open', 'close', 'low']].max(axis=1)) |
                (raw.low > raw[['open', 'close', 'high']].min(axis=1))).any():
            raise ValueError(f'Inconsistent OHLC: {market}')
    return index


def build_features(raw_by_market, config: PortfolioConfig):
    validate_frames(raw_by_market)
    btc = raw_by_market['KRW-BTC'].close
    macro = btc.ewm(span=config.macro_window, adjust=False, min_periods=config.macro_window).mean()
    risk_on = (btc > macro) & (macro > macro.shift(5))
    frames = {}
    for market, raw in raw_by_market.items():
        close = raw.close
        trend = close.ewm(span=config.trend_window, adjust=False,
                          min_periods=config.trend_window).mean()
        atr = average_true_range(raw, 14)
        momentum = .5 * close.pct_change(20) + .3 * close.pct_change(60) + .2 * close.pct_change(120)
        breakout = close > raw.high.rolling(config.breakout_window).max().shift(1)
        # Persistent conditions permit re-entry after a missed signal, only once per day.
        if config.entry_mode == 'trend':
            trigger = pd.Series(True, index=raw.index)
        elif config.entry_mode == 'pullback':
            fast = close.ewm(span=10, adjust=False).mean()
            trigger = (close <= fast * 1.01) & (close > close.shift(1))
        else:
            trigger = breakout
        buy = risk_on & (close > trend) & (trend > trend.shift(5)) & trigger & (momentum > 0)
        sell = (~risk_on) | (close < trend) | (momentum < 0)
        score = momentum / (atr / close).replace(0, np.nan)
        frames[market] = pd.DataFrame({'buy': buy.fillna(False), 'sell': sell.fillna(True),
                                      'atr': atr, 'high': raw.high,
                                      'score': score.replace([np.inf, -np.inf], np.nan).fillna(0.)})
    return frames


class PortfolioSimulator:
    """Same decision/risk logic for candle replay and current-price paper quotes.

    A drawdown halt is latched until a new simulation is explicitly created.
    Stops and signal exits continue while new entries are blocked.
    """
    def __init__(self, config: PortfolioConfig, *, initial_cash=100000.):
        if not math.isfinite(initial_cash) or initial_cash <= 0:
            raise ValueError('Positive initial cash required')
        self.config = config
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.trader = PaperTrader()
        self.costs = cost_model_from_values(fee_rate=config.fee, slippage_bps=config.slippage_bps)
        self.stops = {}
        self.events = []
        self.last_signal_day = None
        self.last_signal_key = None
        self.day = None
        self.day_start = float(initial_cash)
        self.last_equity = float(initial_cash)
        self.peak = float(initial_cash)
        self.halted = False
        self.daily_halted = False
        self.blocked_exits = {}

    def equity(self, prices):
        return self.cash + sum(self.costs.simulate_exit(price=prices[m], qty=p.qty, cost_basis=p.cost)
                               ['net_proceeds'] for m, p in self.trader.positions.items())

    def mark(self, prices):
        value = self.equity(prices)
        self.peak = max(self.peak, value)
        self.last_equity = value
        self.halted |= value <= self.peak * (1 - self.config.drawdown_limit)
        self.daily_halted |= value <= self.day_start * (1 - self.config.daily_loss_limit)
        return value

    def _exit(self, market, price, reason, day):
        p = self.trader.get_position(market)
        if p.qty * self.costs.sell_price(price) < self.config.min_order:
            self.blocked_exits[market] = 'below_minimum_order'
            return
        event = self.trader.exit_long(market=market, price=price, reason=reason,
                                     fee_rate=self.config.fee, slippage_bps=self.config.slippage_bps,
                                     timestamp=pd.Timestamp(day, tz='UTC').timestamp())
        self.cash += event['net_proceeds']
        self.stops.pop(market, None)
        self.blocked_exits.pop(market, None)
        self.events.append(event)

    def quote(self, prices, *, day, signals=None, signal_key=None):
        day = str(pd.Timestamp(day).date())
        if self.day is not None and day < self.day:
            raise ValueError('Cannot process an older day')
        if not prices or any(not math.isfinite(p) or p <= 0 for p in prices.values()):
            raise ValueError('Invalid quote')
        if not set(self.trader.positions).issubset(prices):
            raise ValueError('Missing held-market quotes')
        if signals is not None:
            for market, signal in signals.items():
                if market not in prices:
                    raise ValueError('Missing signal-market quote')
                if any(not math.isfinite(float(signal[k])) for k in ('atr', 'high', 'score')):
                    raise ValueError('Invalid signal')
        if day != self.day:
            self.day_start = self.last_equity
            self.daily_halted = False
            self.day = day
        key = signal_key if signal_key is not None else day
        new_signal = signals is not None and key != self.last_signal_key
        if new_signal:
            for market in self.trader.positions:
                signal = signals.get(market)
                if signal and signal['atr'] > 0:
                    self.stops[market] = max(self.stops[market],
                                             signal['high'] - max(self.config.stop_atr * signal['atr'],
                                                                  signal['high'] * self.config.min_stop_fraction))
        exited = set()
        for market in list(self.trader.positions):
            signal = (signals or {}).get(market, {})
            if prices[market] <= self.stops[market]:
                self._exit(market, prices[market], 'stop', day)
                exited.add(market)
            elif new_signal and signal.get('sell', False):
                self._exit(market, prices[market], 'regime_exit', day)
                exited.add(market)
        equity = self.mark(prices)
        if not new_signal:
            return
        self.last_signal_day = day
        self.last_signal_key = key
        if self.halted or self.daily_halted:
            return
        ranked = sorted(signals, key=lambda m: (-signals[m]['score'], m))
        for market in ranked:
            signal = signals[market]
            if (not signal['buy'] or signal['sell'] or signal['atr'] <= 0 or market in exited
                    or self.trader.has_position(market)):
                continue
            if len(self.trader.positions) >= self.config.max_positions:
                break
            price = prices[market]
            stop = price - max(self.config.stop_atr * signal['atr'], price * self.config.min_stop_fraction)
            if stop <= 0:
                continue
            unit = self.costs.simulate_entry(price=price, budget=1.)
            risk_per_krw = 1 - self.costs.simulate_exit(price=stop, qty=unit['qty'], cost_basis=1.)['net_proceeds']
            exposure = sum(p.qty * prices[m] for m, p in self.trader.positions.items())
            budget = min(self.cash, equity * self.config.asset_fraction,
                         max(0., equity * self.config.exposure_fraction - exposure),
                         equity * self.config.risk_fraction / max(risk_per_krw, 1e-9))
            # Minimum order applies to the notional, excluding our fee reserve.
            if budget * (1 - self.config.fee) < self.config.min_order:
                continue
            event = self.trader.enter_long(market=market, price=price, cost=budget,
                                          strategy='adaptive_portfolio', fee_rate=self.config.fee,
                                          slippage_bps=self.config.slippage_bps,
                                          timestamp=pd.Timestamp(day, tz='UTC').timestamp())
            self.cash -= budget
            self.stops[market] = stop
            self.events.append(event)
        self.mark(prices)

    def to_state(self):
        return {'version': 1, 'mode': 'PAPER_ONLY', 'config': asdict(self.config),
                'initial_cash': self.initial_cash, 'cash': self.cash,
                'positions': self.trader.to_state(), 'stops': dict(self.stops), 'events': list(self.events),
                'last_signal_day': self.last_signal_day, 'day': self.day, 'day_start': self.day_start,
                'last_signal_key': self.last_signal_key,
                'last_equity': self.last_equity, 'peak': self.peak, 'halted': self.halted,
                'daily_halted': self.daily_halted, 'blocked_exits': dict(self.blocked_exits)}

    @classmethod
    def from_state(cls, state):
        if state.get('mode') != 'PAPER_ONLY' or state.get('version') != 1:
            raise ValueError('Unsupported paper state')
        obj = cls(PortfolioConfig(**state['config']), initial_cash=state['initial_cash'])
        for key in ('cash', 'day_start', 'last_equity', 'peak'):
            value = float(state[key])
            if not math.isfinite(value) or value < 0:
                raise ValueError('Invalid paper balance')
            setattr(obj, key, value)
        obj.trader = PaperTrader(state['positions'])
        obj.stops = dict(state['stops'])
        if set(obj.stops) != set(obj.trader.positions):
            raise ValueError('Missing position stop')
        for market, p in obj.trader.positions.items():
            if any(not math.isfinite(v) or v <= 0 for v in (p.qty, p.cost, p.entry, obj.stops[market])):
                raise ValueError('Invalid paper position')
        for key in ('events', 'last_signal_day', 'day', 'halted', 'daily_halted', 'blocked_exits'):
            setattr(obj, key, state[key])
        obj.last_signal_key = state.get('last_signal_key', state.get('last_signal_day'))
        return obj


def replay(raw_by_market, config, *, start, end, initial_cash=100000., features=None):
    index = validate_frames(raw_by_market)
    if not 1 <= start < end <= len(index):
        raise ValueError('Invalid replay range')
    features = features if features is not None else build_features(raw_by_market, config)
    sim = PortfolioSimulator(config, initial_cash=initial_cash)
    curve = []
    for i in range(start, end):
        day = str(index[i].date())
        prices = {m: float(raw.open.iloc[i]) for m, raw in raw_by_market.items()}
        signals = {m: frame.iloc[i-1].to_dict() for m, frame in features.items()}
        before = len(sim.events)
        sim.quote(prices, day=day, signals=signals, signal_key=str(index[i]))
        # Today's low can stop an existing/new position, but today's high cannot
        # raise its stop until tomorrow. No optimistic intrabar path assumption.
        for market in list(sim.trader.positions):
            if float(raw_by_market[market].low.iloc[i]) <= sim.stops[market]:
                sim._exit(market, min(prices[market], sim.stops[market]), 'intraday_stop', day)
        for event in sim.events[before:]:
            event['ts'] = index[i].timestamp()
            if event['side'] == 'BUY' and sim.trader.has_position(event['market']):
                sim.trader.get_position(event['market']).opened_at = event['ts']
        closes = {m: float(raw.close.iloc[i]) for m, raw in raw_by_market.items()}
        curve.append(sim.mark(closes))
    equity = pd.Series(curve, index=index[start:end], dtype=float)
    peaks = equity.cummax().clip(lower=initial_cash)
    sells = [e for e in sim.events if e['side'] == 'SELL']
    wins = sum(max(e['pnl_value'], 0.) for e in sells)
    losses = sum(max(-e['pnl_value'], 0.) for e in sells)
    return {'final_equity': float(equity.iloc[-1]),
            'return_pct': float((equity.iloc[-1] / initial_cash - 1) * 100),
            'max_drawdown_pct': float(((equity / peaks - 1) * 100).min()),
            'closed_trades': len(sells), 'buy_trades': len(sim.events) - len(sells),
            'win_rate_pct': sum(e['pnl_value'] > 0 for e in sells) / len(sells) * 100 if sells else 0.,
            'profit_factor': wins / losses if losses else None,
            'halted': sim.halted, 'open_positions': len(sim.trader.positions),
            'equity': equity, 'state': sim.to_state()}
