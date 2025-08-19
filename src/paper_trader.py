from dataclasses import dataclass
from typing import List, Literal, Optional, Dict

Side = Literal['buy', 'sell']

@dataclass
class Order:
    index: int  # price series index where the order executed
    side: Side
    price: float
    size: float  # filled size of asset for buy, or size sold for sell
    cash_change: float  # cash delta after commission/slippage (negative for buy)
    equity: float  # equity after order

@dataclass
class Portfolio:
    cash: float
    asset: float

class PaperTrader:
    def __init__(self, cash: float, slippage_bps: float = 0.0, fee_bps: float = 5.0, position_fraction: float = 1.0):
        """
        slippage_bps: 가정 슬리피지 (1bp = 0.01%)
        fee_bps: 매수/매도 수수료 (왕복 아님, 한 방향)
        position_fraction: 0~1 사이, 시그널 발생 시 현금(or 자산) 사용 비율
        """
        self.port = Portfolio(cash=cash, asset=0.0)
        self.orders: List[Order] = []
        self.slippage_bps = slippage_bps
        self.fee_bps = fee_bps
        self.position_fraction = max(0.0, min(1.0, position_fraction))

    def _apply_slip_fee(self, side: Side, price: float) -> float:
        # 슬리피지: 매수는 +, 매도는 - 방향으로 가격 불리하게
        slip = price * (self.slippage_bps / 10_000.0)
        px = price + slip if side == 'buy' else price - slip
        fee = px * (self.fee_bps / 10_000.0)
        return px + fee if side == 'buy' else px - fee

    def on_signal(self, side: Side, price: float, index: Optional[int] = None):
        exec_price = self._apply_slip_fee(side, price)
        if side == 'buy' and self.port.cash > 0:
            deploy_cash = self.port.cash * self.position_fraction
            if deploy_cash <= 0:
                return
            size = deploy_cash / exec_price
            self.port.asset += size
            self.port.cash -= deploy_cash
            eq = self.port.cash + self.port.asset * price
            self.orders.append(Order(index=index if index is not None else -1, side='buy', price=exec_price, size=size, cash_change=-deploy_cash, equity=eq))
        elif side == 'sell' and self.port.asset > 0:
            sell_size = self.port.asset * self.position_fraction
            if sell_size <= 0:
                return
            proceeds = sell_size * exec_price
            self.port.asset -= sell_size
            self.port.cash += proceeds
            eq = self.port.cash + self.port.asset * price
            self.orders.append(Order(index=index if index is not None else -1, side='sell', price=exec_price, size=sell_size, cash_change=proceeds, equity=eq))

    def equity(self, price: float) -> float:
        return self.port.cash + self.port.asset * price

    def trade_log(self) -> List[Order]:
        return self.orders

    def equity_curve(self, price_series: list[float]) -> List[float]:
        if not price_series:
            return []
        # Reconstruct by replaying orders at their indices.
        cash = 0.0
        asset = 0.0
        # Find initial cash: assume first order equity = cash +/- cash_change + asset*price; simpler: start with sum of negative cash_changes (buys) + remaining cash.
        # Instead take starting cash as sum( -buy cash_changes ).
        start_cash = 0.0
        for o in self.orders:
            if o.side == 'buy':
                start_cash += -o.cash_change
        cash = start_cash
        asset = 0.0
        orders_by_index: Dict[int, List[Order]] = {}
        for o in self.orders:
            orders_by_index.setdefault(o.index, []).append(o)
        curve: List[float] = []
        for i, px in enumerate(price_series):
            if i in orders_by_index:
                for o in orders_by_index[i]:
                    if o.side == 'buy':
                        # consume cash, add asset
                        cash += o.cash_change  # cash_change negative
                        asset += o.size
                    else:  # sell
                        cash += o.cash_change
                        asset -= o.size
            curve.append(cash + asset * px)
        return curve

    def performance(self, price_series: list[float]) -> dict:
        if not price_series:
            return {}
        eq_curve = self.equity_curve(price_series) or []
        if not eq_curve:
            return {}
        import math, statistics
        returns = []
        for i in range(1, len(eq_curve)):
            prev = eq_curve[i-1]
            if prev != 0:
                returns.append(eq_curve[i]/prev - 1)
        total_return = eq_curve[-1]/eq_curve[0]-1 if eq_curve[0] else None
        peak = -math.inf
        mdd = 0.0
        for v in eq_curve:
            if v > peak:
                peak = v
            dd = (peak - v)/peak if peak > 0 else 0
            mdd = max(mdd, dd)
        # Pair trades naively: treat each sell as closing portion; win if cash_change positive relative to cost basis (approx).
        sell_orders = [o for o in self.orders if o.side == 'sell']
        win_rate = None
        if sell_orders:
            wins = sum(1 for o in sell_orders if o.cash_change > 0)
            win_rate = wins/len(sell_orders)
        sharpe = None
        if returns:
            mean_r = statistics.mean(returns)
            stdev_r = statistics.pstdev(returns) or 0.0
            if stdev_r > 0:
                sharpe = (mean_r / stdev_r) * (len(returns) ** 0.5)
        return {
            'total_return': total_return,
            'mdd': mdd,
            'win_rate': win_rate,
            'sharpe_like': sharpe,
            'equity_curve': eq_curve,
        }
