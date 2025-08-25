"""Mean Reversion 실시간(혹은 주기적) 모니터 & 텔레그램/카카오 알림.

추가 기능 (2025-08):
 - 옵션으로 실제 주문 실행 (UPBIT_LIVE=1 + --live-orders 플래그 모두 필요)
 - 고정 KRW 할당 (--krw-per-trade) 으로 시장가 매수, 비율 기반 시장가 매도
 - 주문/체결(추정) 내역 알림

동작 개념
 1. 대상 마켓 목록 로드 (기본: 거래대금 상위 N개)
 2. 각 마켓 최근 캔들 수집 (인터벌 configurable)
 3. Bollinger Bands (period, k) + RSI(14) 계산
 4. 엔트리 조건: close < lowerBand OR RSI < rsi_buy (포지션 없을 때)
 5. 엑싯 조건: close > upperBand OR RSI > rsi_sell (또는 mid / stop / take)
 6. 상태는 메모리(dict) 로 유지 (필요시 파일 저장 확장 가능)

실제 주문:
 - 매수: 시장가 (ord_type=price) 로 KRW 고정 금액 사용 → 체결 후 수량은 금액/가격(수수료 미반영) 추정
 - 매도: 시장가 (ord_type=market) 로 보유 수량 전량 (추정 수량)
 - create_order 는 UPBIT_LIVE=1 이고 --live-orders 지정 시에만 실제 API 호출, 그 외 simulate
"""
from __future__ import annotations
import os, time, math
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
import argparse
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from upbit_api import UpbitAPI
from notifier import get_notifier

load_dotenv()

def fetch_top_markets(api: UpbitAPI, base: str = "KRW", limit: int = 40, exclude_stables: bool = True) -> List[str]:
    """거래대금 상위 마켓 조회.

    exclude_stables=True 이면 KRW-USDT / KRW-USDC / KRW-USDJ / KRW-DAI / KRW-UST 등
    스테이블 코인(달러 페그) 마켓을 제외한다. (밴드 폭이 매우 좁아 즉시 조건 충족/과도한 진입 방지)
    """
    mkts = api.markets()
    base_prefix = base.upper() + "-"
    filtered = [m for m in mkts if isinstance(m.get('market'), str) and m['market'].startswith(base_prefix)]
    names = [m['market'] for m in filtered]
    tks = api.tickers(names)
    tks_sorted = sorted(tks, key=lambda x: float(x.get('acc_trade_price_24h') or 0), reverse=True)
    markets = [t['market'] for t in tks_sorted[:limit]]
    if exclude_stables:
        stable_keywords = ("USDT", "USDC", "USDJ", "DAI", "UST")
        markets = [m for m in markets if not any(kw in m for kw in stable_keywords)]
    return markets

def compute_indicators(df: pd.DataFrame, bb_period: int, bb_k: float) -> pd.DataFrame:
    if len(df) >= bb_period:
        df['bb_mid'] = df['close'].rolling(bb_period).mean()
        df['bb_std'] = df['close'].rolling(bb_period).std()
        df['bb_upper'] = df['bb_mid'] + bb_k * df['bb_std']
        df['bb_lower'] = df['bb_mid'] - bb_k * df['bb_std']
    else:
        df['bb_mid'] = df['bb_upper'] = df['bb_lower'] = np.nan
    rsi_period = 14
    delta = df['close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    if len(df) > rsi_period:
        roll_up = pd.Series(gain).ewm(alpha=1/rsi_period, adjust=False).mean()
        roll_down = pd.Series(loss).ewm(alpha=1/rsi_period, adjust=False).mean()
        rs = roll_up / roll_down
        df['rsi'] = 100 - (100 / (1 + rs))
    else:
        df['rsi'] = np.nan
    return df

class Position:
    __slots__ = ('market','entry_price','entry_time','volume','krw_alloc')
    def __init__(self, market: str, entry_price: float, entry_time: datetime, volume: float, krw_alloc: float):
        self.market = market
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.volume = volume          # 추정/실제 수량
        self.krw_alloc = krw_alloc    # 사용한 KRW 금액 (매수 기준)

KST = timezone(timedelta(hours=9))

class MRMonitor:
    def __init__(self, api: UpbitAPI, interval: str = 'minute15',
                 bb_period: int = 20, bb_k: float = 2.0,
                 rsi_buy: int = 30, rsi_sell: int = 70,
                 exit_mid: bool = True, stop_pct: float = 0.0, take_pct: float = 0.0,
                 live_orders: bool = False, krw_per_trade: float = 5000.0, max_open: int = 5,
                 min_fetch_seconds: float = 20.0, per_request_sleep: float = 0.12,
                 min_bandwidth_pct: float = 0.5, **_ignored_extra):
        """MR 모니터 초기화.

        min_bandwidth_pct: (%) Bollinger 상단-하단 / 중간 값 비율이 이 값보다 작으면 (너무 좁은 밴드 → 스테이블 가능성) 신호 무시.
        """
        self.api = api
        self.interval = interval
        self.bb_period = bb_period
        self.bb_k = bb_k
        self.rsi_buy = rsi_buy
        self.rsi_sell = rsi_sell
        self.exit_mid = exit_mid
        self.stop_pct = stop_pct
        self.take_pct = take_pct
        self.live_orders = live_orders and (str(os.getenv('UPBIT_LIVE') or '') == '1')
        self.krw_per_trade = krw_per_trade
        self.max_open = max_open
        self.positions: Dict[str, Position] = {}
        self.notifier = get_notifier()
        # --- rate limiting state ---
        self.min_fetch_seconds = min_fetch_seconds  # 최소 재호출 간격 (마켓별)
        self.per_request_sleep = per_request_sleep  # 호출 간 짧은 슬립
        self._last_fetch: Dict[str, float] = {}
        # 안정성 필터: 너무 좁은 밴드폭(스테이블 등) 무시
        self.min_bandwidth_pct = min_bandwidth_pct / 100.0
        self._reason_map = {
            'STOP': '손절',
            'TAKE': '목표가',
            'REVERSAL': '반전',
            'MID': '중심선'
        }
        # 통계
        self.total_realized_pnl = 0.0  # 누적 실현 손익 (KRW 추정)
        self.total_trades = 0
        self.win_trades = 0
        self.loss_trades = 0
        self.current_streak = 0   # +: 연속 승, -: 연속 패

    # ---- helpers ----
    def _short_krw(self, v: float) -> str:
        try:
            if v >= 1e12:  # 조
                return f"{v/1e12:.2f}조"
            if v >= 1e8:   # 억
                return f"{v/1e8:.2f}억"
            if v >= 1e4:   # 만
                return f"{v/1e4:.2f}만"
            if v >= 1000:
                return f"{v:,.0f}"
            return f"{v:.2f}" if v < 1 else f"{v:.0f}"
        except Exception:
            return str(v)

    def _notify(self, msg: str):
        # KST 타임스탬프 프리픽스
        try:
            now_kst = datetime.utcnow().replace(tzinfo=timezone.utc).astimezone(KST)
            ts_txt = now_kst.strftime('%H:%M:%S')
            msg_out = f"[{ts_txt} KST] {msg}"
        except Exception:
            msg_out = msg
        if self.notifier.available():
            try:
                self.notifier.send_text(msg_out)
            except Exception:
                pass
        print(msg_out)

    def _place_entry(self, market: str, close: float, rsi: float) -> Optional[Position]:
        if len(self.positions) >= self.max_open:
            return None
        alloc = self.krw_per_trade
        volume_est = alloc / close if close > 0 else 0.0
        order_info = None
        if self.live_orders:
            try:
                order_info = self.api.create_order(market, side='bid', ord_type='price', price=f"{int(alloc)}", simulate=False)
            except Exception as e:
                self._notify(f"[진입실패][실매매] {market} KRW{alloc:.0f} 오류: {e}")
                return None
        else:
            order_info = self.api.create_order(market, side='bid', ord_type='price', price=f"{int(alloc)}", simulate=True)
        pos = Position(market, close, datetime.utcnow(), volume_est, alloc)
        self.positions[market] = pos
        tag_mode = '실매매' if (self.live_orders and not order_info.get('simulate')) else '모의'
        rsi_txt = f"RSI={rsi:.1f}" if not math.isnan(rsi) else "RSI=NA"
        price_short = self._short_krw(close)
        self._notify(
            f"[진입][{tag_mode}] {market} 진입가 {price_short}원 (≈{close:,.2f}) 매수금액 {self._short_krw(alloc)}원 예상수량 {volume_est:.6f}개 {rsi_txt}"
        )
        return pos

    def _place_exit(self, market: str, close: float, reason: str, pos: Position):
        order_info = None
        if self.live_orders:
            try:
                order_info = self.api.create_order(market, side='ask', ord_type='market', volume=f"{pos.volume:.8f}", simulate=False)
            except Exception as e:
                self._notify(f"[청산실패][실매매] {market} 오류: {e}")
                return
        else:
            order_info = self.api.create_order(market, side='ask', ord_type='market', volume=f"{pos.volume:.8f}", simulate=True)
        pnl_pct = (close / pos.entry_price - 1) * 100
        realized = (close - pos.entry_price) * pos.volume  # 추정 실현 손익 (수수료 제외)
        self.total_realized_pnl += realized
        self.total_trades += 1
        win = realized > 0
        if win:
            self.win_trades += 1
            self.current_streak = self.current_streak + 1 if self.current_streak >= 0 else 1
        elif realized < 0:
            self.loss_trades += 1
            self.current_streak = self.current_streak - 1 if self.current_streak <= 0 else -1
        else:
            # 0 손익은 streak reset
            self.current_streak = 0
        win_rate = (self.win_trades / self.total_trades * 100) if self.total_trades else 0.0
        sign = '+' if pnl_pct >= 0 else ''
        tag_mode = '실매매' if (self.live_orders and not order_info.get('simulate')) else '모의'
        reasons_ko = []
        for r in reason.split('/'):
            reasons_ko.append(self._reason_map.get(r, r))
        reasons_ko_txt = '/'.join(reasons_ko) if reasons_ko else '-'
        if pnl_pct > 0:
            pnl_emoji = '🔺'; pnl_word = '이익'
        elif pnl_pct < 0:
            pnl_emoji = '🔻'; pnl_word = '손실'
        else:
            pnl_emoji = '➖'; pnl_word = '보합'
        streak_txt = f"연속승 {self.current_streak}" if self.current_streak>0 else (f"연속패 {abs(self.current_streak)}" if self.current_streak<0 else '보합')
        self._notify(
            f"[청산][{tag_mode}] {market} {pnl_emoji} 청산가 {self._short_krw(close)}원 (≈{close:,.2f}) 수익률 {sign}{pnl_pct:.2f}% {pnl_word} 실현손익 {self._short_krw(realized)}원 누적 {self._short_krw(self.total_realized_pnl)}원 (승률 {win_rate:.1f}% / {streak_txt} / 사유: {reasons_ko_txt}) 수량 {pos.volume:.6f}개"
        )

    def process_market(self, market: str, candles_count: int = 120):
        """개별 마켓 처리 (레이트리밋 + 엔트리/엑싯 판정)."""
        now_t = time.time()
        last_t = self._last_fetch.get(market, 0)
        if (now_t - last_t) < self.min_fetch_seconds:
            return
        try:
            candles = self.api.candles(market, interval=self.interval, count=candles_count)
        except Exception as e:
            if '429' in str(e):  # rate limit backoff
                self._last_fetch[market] = now_t + self.min_fetch_seconds * 0.5
            return
        self._last_fetch[market] = now_t
        if len(candles) < self.bb_period + 5:
            return
        df = pd.DataFrame([c.model_dump() for c in candles])
        df['dt'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = compute_indicators(df, self.bb_period, self.bb_k)
        last = df.iloc[-1]
        close = float(last['close'])
        lower = float(last.get('bb_lower') or np.nan)
        upper = float(last.get('bb_upper') or np.nan)
        mid = float(last.get('bb_mid') or np.nan)
        rsi = float(last.get('rsi') or np.nan)
        pos = self.positions.get(market)
        # Entry
        if not pos:
            # 밴드폭 필터 (upper/lower/ mid 모두 유효해야 계산)
            bandwidth_ok = True
            if not math.isnan(lower) and not math.isnan(upper) and not math.isnan(mid) and mid > 0:
                bw = (upper - lower) / mid  # 비율
                if bw < self.min_bandwidth_pct:
                    bandwidth_ok = False
            if bandwidth_ok:
                cond = ((not math.isnan(lower) and close < lower) or (not math.isnan(rsi) and rsi < self.rsi_buy))
                if cond:
                    self._place_entry(market, close, rsi)
                    return
        # Exit
        if pos:
            exit_cond = False
            reasons: List[str] = []
            if self.stop_pct>0 and close <= pos.entry_price * (1 - self.stop_pct):
                exit_cond = True; reasons.append('STOP')
            if not exit_cond and self.take_pct>0 and close >= pos.entry_price * (1 + self.take_pct):
                exit_cond = True; reasons.append('TAKE')
            if not exit_cond:
                if (not math.isnan(upper) and close > upper) or (not math.isnan(rsi) and rsi > self.rsi_sell):
                    exit_cond = True; reasons.append('REVERSAL')
                elif self.exit_mid and (not math.isnan(mid) and close > mid):
                    exit_cond = True; reasons.append('MID')
            if exit_cond:
                self._place_exit(market, close, '/'.join(reasons), pos)
        if self.per_request_sleep > 0:
            time.sleep(self.per_request_sleep)

    def loop(self, markets: List[str], loop_seconds: int = 120):
        mode = '실매매' if self.live_orders else '모의'
        self._notify(
            f"[시작][{mode}] Mean Reversion 감시 시작 인터벌={self.interval} 대상={len(markets)}개 BB={self.bb_period},{self.bb_k} RSI={self.rsi_buy}/{self.rsi_sell} 1회매수={self.krw_per_trade:,.0f}원 최대포지션={self.max_open}개"
        )
        while True:
            for m in markets:
                try:
                    self.process_market(m)
                except Exception as e:
                    print(f"process error {m}: {e}")
            time.sleep(loop_seconds)

    # Streamlit 주기적 단일 패스 실행용 (블로킹 루프 대신 한 번 돌고 metric 반환)
    def run_cycle(self, markets: List[str]):
        start = time.time()
        open_before = len(self.positions)
        trades_before = self.total_trades
        processed = 0
        for m in markets:
            try:
                self.process_market(m)
            except Exception as e:
                print(f"run_cycle error {m}: {e}")
            processed += 1
        dur = time.time() - start
        return {
            'processed': processed,
            'open': len(self.positions),
            'open_delta': len(self.positions) - open_before,
            'trades': self.total_trades,
            'new_trades': self.total_trades - trades_before,
            'dur_ms': int(dur * 1000)
        }

# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(description='Mean Reversion Monitor')
    p.add_argument('--interval', default='minute15')
    p.add_argument('--markets', type=int, default=30, help='거래대금 상위 N개')
    p.add_argument('--loop-seconds', type=int, default=120)
    p.add_argument('--bb-period', type=int, default=20)
    p.add_argument('--bb-k', type=float, default=2.0)
    p.add_argument('--rsi-buy', type=int, default=30)
    p.add_argument('--rsi-sell', type=int, default=70)
    p.add_argument('--exit-mid', action='store_true', default=False)
    p.add_argument('--stop-pct', type=float, default=0.0)
    p.add_argument('--take-pct', type=float, default=0.0)
    p.add_argument('--base', default='KRW')
    p.add_argument('--live-orders', action='store_true', default=False, help='실제 주문 (UPBIT_LIVE=1 필요)')
    p.add_argument('--krw-per-trade', type=float, default=5000.0, help='매수 1회 KRW 할당')
    p.add_argument('--max-open', type=int, default=5, help='최대 동시 포지션 수')
    p.add_argument('--min-fetch-seconds', type=float, default=20.0, help='같은 마켓 최소 재호출 간격 (초)')
    p.add_argument('--per-request-sleep', type=float, default=0.12, help='마켓 별 호출 사이 슬립 (초)')
    p.add_argument('--min-bandwidth-pct', type=float, default=0.5, help='최소 밴드폭 % (이하이면 진입 무시)')
    p.add_argument('--no-exclude-stables', action='store_true', help='USDT/USDC 등 스테이블 마켓도 포함')
    return p.parse_args()

def main():
    args = parse_args()
    api = UpbitAPI(access_key=os.getenv('UPBIT_ACCESS_KEY'), secret_key=os.getenv('UPBIT_SECRET_KEY'))
    mkts = fetch_top_markets(api, base=args.base, limit=args.markets, exclude_stables=not args.no_exclude_stables)
    mon = MRMonitor(api,
                    interval=args.interval,
                    bb_period=args.bb_period,
                    bb_k=args.bb_k,
                    rsi_buy=args.rsi_buy,
                    rsi_sell=args.rsi_sell,
                    exit_mid=args.exit_mid,
                    stop_pct=args.stop_pct,
                    take_pct=args.take_pct,
                    live_orders=args.live_orders,
                    krw_per_trade=args.krw_per_trade,
                    max_open=args.max_open,
                    min_fetch_seconds=args.min_fetch_seconds,
                    per_request_sleep=args.per_request_sleep,
                    min_bandwidth_pct=args.min_bandwidth_pct)
    if args.live_orders and str(os.getenv('UPBIT_LIVE') or '') != '1':
        print('[WARN] --live-orders 지정했지만 환경변수 UPBIT_LIVE=1 이 아님 → 시뮬레이션 모드로 동작')
    mon.loop(mkts, loop_seconds=args.loop_seconds)

if __name__ == '__main__':
    main()
