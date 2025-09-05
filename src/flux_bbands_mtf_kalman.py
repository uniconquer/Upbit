"""
Pine Script v6 기반 "Bollinger Bands MTF & Kalman Filter | Flux Charts" 전략의 pandas 포트.

요구 사항:
    pip install pandas numpy

입력 (DataFrame):
    DateTimeIndex (타임존 여부 무관), 컬럼: ['open','high','low','close','volume']

출력 (DataFrame 컬럼):
    'ltf_basis','ltf_upper','ltf_lower',
    'htf_upper','htf_lower',
    'buy_signal','sell_signal'

설명 / 특징:
  - Kalman 필터는 Pine 구현의 단순 1D 버전을 그대로 재현.
  - HTF (상위 타임프레임) Bollinger 밴드를 EMA 로 스무딩 후 LTF 인덱스에 forward-fill.
  - 시그널은 Pine 스크립트의 상태 머신(state machine) 로직을 포트.
  - 모듈은 호출 간 상태를 보존하지 않음. 스트리밍으로 새로운 봉이 들어오면 `state` 딕트를 외부에서 유지하거나, 필요한 경우 별도 상태 관리 함수 사용.

예시:
    from flux_bbands_mtf_kalman import indicator
    out = indicator(df, htf_rule='30min')  # 예: 1분 봉 df 위에 30분 HTF 계산 ('30T' 입력도 허용)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import re

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def stdev(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length).std(ddof=0)

def kalman_filter_1d(series: pd.Series,
                     process_noise: float = 0.2,
                     measurement_error: float = 2.0) -> pd.Series:
    """단순 1차원 Kalman 필터 (Pine 구현을 그대로 재현).

    Pine 로직 메모:
    - 추정치(estimate)가 na이면 직전 봉 종가(close[1])를 초기값으로 사용.
    - 여기서는 루프 i=1 부터 시작하며 이전 추정치가 na 이면 series[i-1] 종가를 대입.
    - 따라서 index 0 위치의 결과는 NaN 으로 남아 Pine 첫 봉 동작과 동일.
    """
    s = series.astype(float)
    n = len(s)
    if n == 0:
        return s.copy() * np.nan

    est = np.full(n, np.nan, dtype=float)
    err = np.full(n, 1.0, dtype=float)  # error_est

    # i=0은 Pine에서도 close[1]가 없으므로 결과가 na가 되는 게 자연스러움
    for i in range(1, n):
        # Pine의 'if na(estimate) estimate := close[1]'에 해당
        prev_est = est[i-1]
        if np.isnan(prev_est):
            prev_est = s.iat[i-1]  # 직전 봉 종가로 초기화

        prediction = prev_est
        k_gain = err[i-1] / (err[i-1] + measurement_error)
        est[i] = prediction + k_gain * (s.iat[i] - prediction)
        err[i] = (1.0 - k_gain) * err[i-1] + process_noise

    return pd.Series(est, index=series.index, name='kf')


def resample_htf(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """DataFrame 을 상위 타임프레임(HTF)으로 리샘플.

    OHLCV 관례에 맞춰:
    open=first, high=max, low=min, close=last, volume=sum
    """
    agg = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    htf = df.resample(rule).agg(agg).dropna(how='all')
    return htf

def bbands(price: pd.Series, length: int, mult: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    basis = price.rolling(length).mean()
    dev = mult * stdev(price, length)
    upper = basis + dev
    lower = basis - dev
    return basis, upper, lower

def align_to_ltf(htf_series: pd.Series, ltf_index: pd.DatetimeIndex) -> pd.Series:
    # HTF 값을 LTF 인덱스에 정렬(fwd-fill) — Pine 의 request.security 유사 동작
    return htf_series.reindex(ltf_index, method='ffill')

def indicator(df: pd.DataFrame,
              ltf_mult: float = 2.0,
              ltf_length: int = 20,
              htf_mult: float = 2.25,
              htf_length: int = 20,
              htf_rule: str = '30m',
              state: dict | None = None) -> pd.DataFrame:
    """Flux Charts Bollinger Bands (LTF Kalman + HTF EMA 스무딩) 및 시그널 계산.

    매개변수
    ----------
    df : ['open','high','low','close','volume'] 컬럼을 가진 DataFrame
    ltf_mult, ltf_length : LTF (Kalman 기반 중심선 + 표준편차) Bollinger Band 파라미터
    htf_mult, htf_length, htf_rule : HTF Bollinger 파라미터 (예: '30m' => 30분; "\\d+T" 형태 입력은 자동 "\\d+m" 변환)
    state : (선택) 상태 머신 플래그 dict
        'crossedAbove','crossedBelow','bearSignaled','bullSignaled'
        None 이면 전체 백필(backfill)용 False 초기화

    반환
    -------
    DataFrame:
        ['ltf_basis','ltf_upper','ltf_lower','htf_upper','htf_lower','buy_signal','sell_signal']
    """
    df = df.copy()

    # LTF (Kalman-smoothed basis, classic stdev)
    ltf_basis = kalman_filter_1d(df['close'])
    dev = ltf_mult * stdev(df['close'], ltf_length)
    ltf_upper = ltf_basis + dev
    ltf_lower = ltf_basis - dev

    # HTF (resample, classic BB, then EMA smooth) with deprecated 'T' alias normalization
    _rule = htf_rule
    if isinstance(_rule, str) and re.fullmatch(r"(\d+)T", _rule):
        _rule = _rule[:-1] + 'min'
    htf = resample_htf(df, _rule)
    htf_basis, htf_upper_raw, htf_lower_raw = bbands(htf['close'], htf_length, htf_mult)
    htf_upper_s = ema(htf_upper_raw, htf_length)
    htf_lower_s = ema(htf_lower_raw, htf_length)

    # Align to LTF index
    htf_upper_full = align_to_ltf(htf_upper_s, df.index)
    htf_lower_full = align_to_ltf(htf_lower_s, df.index)

    # Signals (port of Pine state machine)
    n = len(df)
    buy = np.zeros(n, dtype=bool)
    sell = np.zeros(n, dtype=bool)

    st = dict(crossedAbove=False, crossedBelow=False, bearSignaled=False, bullSignaled=False)
    if state:
        st.update(state)

    for i in range(n):
        c = df['close'].iat[i]
        hu = htf_upper_full.iat[i]
        hl = htf_lower_full.iat[i]

        # update crossed flags
        if not np.isnan(hu) and c > hu:
            st['crossedAbove'] = True
        if not np.isnan(hl) and c < hl:
            st['crossedBelow'] = True

        # sellSignal: crossedAbove and ltfUpper < ltfUpper[1] and not bearSignaled
        if i > 0:
            cond_sell = (st['crossedAbove'] and
                         (ltf_upper.iat[i] < ltf_upper.iat[i-1]) and
                         (not st['bearSignaled']))
            if cond_sell:
                sell[i] = True
                st['bearSignaled'] = True

        # buySignal: crossedBelow and ltfLower > ltfLower[1] and not bullSignaled
        if i > 0:
            cond_buy = (st['crossedBelow'] and
                        (ltf_lower.iat[i] > ltf_lower.iat[i-1]) and
                        (not st['bullSignaled']))
            if cond_buy:
                buy[i] = True
                st['bullSignaled'] = True

        # reset logic
        if not np.isnan(hu) and not np.isnan(ltf_upper.iat[i]):
            if (c < ltf_upper.iat[i]) and (c < hu):
                st['crossedAbove'] = False
                st['bearSignaled'] = False

        if not np.isnan(hl) and not np.isnan(ltf_lower.iat[i]):
            if (c > ltf_lower.iat[i]) and (c > hl):
                st['crossedBelow'] = False
                st['bullSignaled'] = False

    out = pd.DataFrame({
        'ltf_basis': ltf_basis,
        'ltf_upper': ltf_upper,
        'ltf_lower': ltf_lower,
        'htf_upper': htf_upper_full,
        'htf_lower': htf_lower_full,
        'buy_signal': buy,
        'sell_signal': sell,
    }, index=df.index)

    out.attrs['state'] = st  # allow caller to keep flags for streaming
    return out


# ------------------------------
# EMA Signal (ATR 트레일링 스탑) - Pine v5 포트
# ------------------------------
def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def atr(series_high: pd.Series, series_low: pd.Series, series_close: pd.Series, period: int) -> pd.Series:
    tr = true_range(series_high, series_low, series_close)
    return tr.rolling(period).mean()

def ema_signal(df: pd.DataFrame,
               sensitivity: int = 3,
               atr_period: int = 2,
               trend_ema_length: int = 240,
               use_heikin_ashi: bool = False) -> pd.DataFrame:
    """Pine 기반 EMA Signal (ATR 트레일링 스탑) 스크립트 포트.

    반환 컬럼:
    ['price_source','atr_stop','position','cross_above','cross_below',
     'ema_buy','ema_sell','trend_ema','strength']
    (참고: 기존 atr_ema 제거, strength 비율 추가)
    """
    d = df.copy()

    # Heikin Ashi (옵션, 기본 False)
    if use_heikin_ashi:
    # HA close = (o+h+l+c)/4 ; HA open = 이전 (HA open + HA close)/2
        ha_close = (d['open'] + d['high'] + d['low'] + d['close']) / 4.0
        ha_open = d['open'].copy()
        for i in range(1, len(d)):
            ha_open.iat[i] = (ha_open.iat[i-1] + ha_close.iat[i-1]) / 2.0
        price_source = ha_close
                # ATR 계산을 위해 OHLC 필요 → HA 봉으로 근사
        ha_high = d[['high', ha_open, ha_close]].max(axis=1)
        ha_low  = d[['low',  ha_open, ha_close]].min(axis=1)
        high_s, low_s = ha_high, ha_low
        close_s = ha_close
    else:
        price_source = d['close']
        high_s, low_s, close_s = d['high'], d['low'], d['close']

    a = atr(high_s, low_s, close_s, atr_period)  # Wilder 스타일 rolling mean (true_range 구현 참조)
    nLoss = sensitivity * a

    # ATR 트레일링 스탑 개선: 첫 값 명시적 초기화(이전 0.0 사용으로 인한 왜곡 제거)
    atr_stop = pd.Series(index=d.index, dtype=float)
    for i in range(len(d)):
        ps = price_source.iat[i]
        nl = nLoss.iat[i]
        if i == 0:
            atr_stop.iat[i] = ps - nl  # 초기 바: 롱 기준 시작 (필요 시 조건화 가능)
            continue
        prev_stop = atr_stop.iat[i-1]
        prev_price = price_source.iat[i-1]
        if (ps > prev_stop) and (prev_price > prev_stop):
            atr_stop.iat[i] = max(prev_stop, ps - nl)
        elif (ps < prev_stop) and (prev_price < prev_stop):
            atr_stop.iat[i] = min(prev_stop, ps + nl)
        else:
            atr_stop.iat[i] = (ps - nl) if (ps > prev_stop) else (ps + nl)

    # 교차(cross) 정의 (price vs atr_stop)
    cross_above = (price_source > atr_stop) & (price_source.shift(1) <= atr_stop.shift(1))
    cross_below = (price_source < atr_stop) & (price_source.shift(1) >= atr_stop.shift(1))

    # 포지션 상태 결정 (직전 상태 유지)
    position = pd.Series(index=d.index, dtype=int)
    for i in range(len(d)):
        if i == 0:
            position.iat[i] = 0
            continue
        if cross_above.iat[i]:
            position.iat[i] = 1
        elif cross_below.iat[i]:
            position.iat[i] = -1
        else:
            position.iat[i] = position.iat[i-1]

    # 매수/매도 신호 생성
    ema_buy  = (price_source > atr_stop) & cross_above
    ema_sell = (price_source < atr_stop) & cross_below

    trend_ema = d['close'].ewm(span=trend_ema_length, adjust=False).mean()  # 추세 EMA
    strength = ((price_source - atr_stop) / nLoss.replace(0, np.nan)).rename('strength').fillna(-np.inf)  # ATR 대비 괴리율

    return pd.DataFrame({
        'price_source': price_source,
        'atr_stop': atr_stop,
        'position': position,
        'cross_above': cross_above,
        'cross_below': cross_below,
        'ema_buy': ema_buy,
        'ema_sell': ema_sell,
        'trend_ema': trend_ema,
        'strength': strength
    }, index=d.index)

def indicator_with_ema(
    df: pd.DataFrame,
    # Original MTF BB + Kalman params
    ltf_mult: float = 2.0,
    ltf_length: int = 20,
    htf_mult: float = 2.25,
    htf_length: int = 20,
    htf_rule: str = '30min',
    # EMA Signal params
    sensitivity: int = 3,
    atr_period: int = 2,
    trend_ema_length: int = 240,
    use_heikin_ashi: bool = False,
    state: dict | None = None
) -> pd.DataFrame:
    """Flux 기본 지표 + EMA Signal 결합 및 콤보 시그널 산출 래퍼.

    추가 컬럼:
    'ema_buy','ema_sell','atr_stop','trend_ema','strength','combo_buy','combo_sell'
    """
    base = indicator(
        df,
        ltf_mult=ltf_mult, ltf_length=ltf_length,
        htf_mult=htf_mult, htf_length=htf_length,
        htf_rule=htf_rule, state=state
    )
    ema_df = ema_signal(
        df, sensitivity=sensitivity, atr_period=atr_period,
        trend_ema_length=trend_ema_length, use_heikin_ashi=use_heikin_ashi
    )

    join_cols = [c for c in ['atr_stop','ema_buy','ema_sell','trend_ema','strength'] if c in ema_df.columns]
    out = base.join(ema_df[join_cols], how='left')

    # 단순 "두 조건 모두 충족" 콤보 시그널
    out['combo_buy'] = out['buy_signal'].fillna(False) & out.get('ema_buy', pd.Series(False, index=out.index)).fillna(False)
    out['combo_sell'] = out['sell_signal'].fillna(False) & out.get('ema_sell', pd.Series(False, index=out.index)).fillna(False)
    return out
