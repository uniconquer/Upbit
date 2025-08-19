"""Strategy utilities & example signal generators.

현재 UI / 간소화 방향에서는 대규모 백테스트 프레임워크를 제거했지만,
추후 확장을 위해 재사용 가능한 최소 전략 함수들을 정리한 모듈입니다.

핵심 아이디어:
 - 모든 전략은 prices(시가/종가 등 float 시퀀스) -> [Signal] 형태 반환
 - 교차(Cross) 로직, 기준선 위/아래 여부 판정 등은 공통 헬퍼로 중복 제거
 - 추후 새로운 지표/전략은 동일한 패턴으로 간단히 추가
"""

from dataclasses import dataclass
from typing import Sequence, Iterable, Callable
import pandas as pd


@dataclass(slots=True)
class Signal:
    index: int
    side: str   # 'buy' | 'sell'
    price: float


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _validate_length(prices: Sequence[float], min_len: int) -> bool:
    return len(prices) >= min_len


def sma(prices: Sequence[float], window: int) -> pd.Series:
    if window <= 0:
        raise ValueError("window > 0 required")
    return pd.Series(prices).rolling(window).mean()


def ema(prices: Sequence[float], window: int) -> pd.Series:
    if window <= 0:
        raise ValueError("window > 0 required")
    return pd.Series(prices).ewm(span=window, adjust=False).mean()


def _crossover_signals(fast: pd.Series, slow: pd.Series, base_prices: Sequence[float], start_index: int) -> list[Signal]:
    """Generic cross (fast > slow) generator.

    start_index: 첫 신호 평가 시작 위치 (충분한 기간 확보 후)
    """
    sigs: list[Signal] = []
    prev_state: bool | None = None
    for i in range(start_index, len(fast)):
        f = fast.iloc[i]; s = slow.iloc[i]
        if pd.isna(f) or pd.isna(s):
            continue
        cur_state = f > s
        if prev_state is None:
            prev_state = cur_state
            continue
        if cur_state and not prev_state:
            sigs.append(Signal(index=i, side='buy', price=float(base_prices[i])))
        elif (not cur_state) and prev_state:
            sigs.append(Signal(index=i, side='sell', price=float(base_prices[i])))
        prev_state = cur_state
    return sigs


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

def sma_cross_signals(prices: Sequence[float], short: int = 5, long: int = 20) -> list[Signal]:
    """단순 이동평균 교차.

    buy : SMA(short) 가 SMA(long) 위로 교차
    sell: SMA(short) 가 SMA(long) 아래로 교차
    """
    if long <= short:
        raise ValueError("long must be > short")
    if not _validate_length(prices, long + 1):
        return []
    fast = sma(prices, short)
    slow = sma(prices, long)
    return _crossover_signals(fast, slow, prices, start_index=long)


def ema_cross_signals(prices: Sequence[float], short: int = 5, long: int = 20) -> list[Signal]:
    """지수 이동평균 교차."""
    if long <= short:
        raise ValueError("long must be > short")
    if not _validate_length(prices, long + 1):
        return []
    fast = ema(prices, short)
    slow = ema(prices, long)
    return _crossover_signals(fast, slow, prices, start_index=long)


def momentum_signals(prices: Sequence[float], long: int = 50) -> list[Signal]:
    """가격이 장기 SMA 위/아래를 기준으로 포지션 전환 (단순 모멘텀)."""
    if not _validate_length(prices, long + 2):
        return []
    baseline = sma(prices, long)
    # baseline 과 price 자체를 비교하므로 fast=price, slow=baseline 으로 크로스 적용
    fast = pd.Series(prices)
    return _crossover_signals(fast, baseline, prices, start_index=long)


def rsi_signals(prices: Sequence[float], period: int = 14, oversold: float = 30.0, overbought: float = 70.0) -> list[Signal]:
    """RSI 기준선 (oversold / overbought) 교차 신호.

    buy : RSI 가 oversold 아래 -> 이상 교차
    sell: RSI 가 overbought 위 -> 이하 교차
    """
    if period <= 1 or oversold >= overbought:
        return []
    if not _validate_length(prices, period + 2):
        return []
    s = pd.Series(prices)
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    sigs: list[Signal] = []
    prev_val: float | None = None
    for i in range(period + 1, len(rsi)):
        cur = rsi.iloc[i]
        if pd.isna(cur):
            continue
        cur_f = float(cur)
        if prev_val is not None:
            if prev_val < oversold <= cur_f:
                sigs.append(Signal(index=i, side='buy', price=float(s.iloc[i])))
            elif prev_val > overbought >= cur_f:
                sigs.append(Signal(index=i, side='sell', price=float(s.iloc[i])))
        prev_val = cur_f
    return sigs


# ---------------------------------------------------------------------------
# Reference help text (optional for UI/Docs)
# ---------------------------------------------------------------------------
SMA_HELP_MD = (
    "### SMA Cross 전략\n\n"
    "단기 / 장기 단순이동평균 교차.\n"
    "- 골든크로스: 단기가 장기 위로 -> buy\n"
    "- 데드크로스: 단기가 장기 아래로 -> sell\n"
    "장점: 단순 & 추세 구간 효율 | 단점: 횡보 구간 노이즈"
)

STRATEGY_HELP = {
    'sma_cross': '단순 이동평균 교차 (short,long)',
    'ema_cross': '지수 이동평균 교차 (short,long)',
    'momentum': '가격 vs SMA(long) 교차 기반 보유/청산',
    'rsi': 'RSI 기준선(oversold/overbought) 역방향 이탈 교차'
}


__all__ = [
    'Signal',
    'sma_cross_signals',
    'ema_cross_signals',
    'momentum_signals',
    'rsi_signals',
    'sma',
    'ema',
    'SMA_HELP_MD',
    'STRATEGY_HELP'
]
