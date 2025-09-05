from __future__ import annotations
import streamlit as st
import pandas as pd
from upbit_api import UpbitAPI
from utils.formatters import fmt_full_number, fmt_coin_amount, fmt_price, signed_color

api: UpbitAPI | None = None

def init_api(a: UpbitAPI):
    global api
    api = a

@st.cache_data(ttl=30)
def _tickers(markets: list[str]):
    if not api or not markets: return []
    try:
        return api.tickers(markets)
    except Exception:
        # 일괄 호출 실패시 개별 호출로 복구 (문제되는 마켓만 제외)
        out=[]; failed=[]
        for m in markets:
            try:
                out.append(api.ticker(m))
            except Exception:
                failed.append(m)
        # 실패 목록은 세션에 기록 (디버그용)
        st.session_state['account_failed_tickers']=failed
        return out

@st.cache_data(ttl=300)
def _all_markets_set():
    if not api: return set()
    try:
        return {m.get('market') for m in api.markets() if m.get('market')}
    except Exception:
        return set()

def render_account():
    st.header('내 자산')
    if not api:
        st.error('API 초기화 안됨')
        return
    try:
        accounts = api.accounts()
    except Exception as e:
        st.error(f'계좌 조회 실패: {e}')
        return
    if not accounts:
        st.warning('보유 자산이 없습니다.')
        return
    need = []
    for a in accounts:
        cur = a.get('currency'); bal_str = a.get('balance') or '0'; locked_str = a.get('locked') or '0'
        try:
            bal_all = float(bal_str) + float(locked_str)
        except Exception:
            bal_all = 0.0
        if cur and cur != 'KRW' and bal_all>0:
            need.append(f'KRW-{cur}')
    need = list(dict.fromkeys(need))
    tick = _tickers(need)
    price_map = {t['market']: t.get('trade_price') for t in tick if isinstance(t, dict)}
    rows=[]; total_eval=0.0; krw_bal=0.0; total_crypto_eval=0.0; total_purchase_cost=0.0
    for a in accounts:
        cur = a.get('currency')
        try:
            bal = float(a.get('balance') or 0); locked = float(a.get('locked') or 0)
            amt = bal + locked; avg_buy = float(a.get('avg_buy_price') or 0)
            if cur == 'KRW':
                krw_bal += amt; total_eval += amt
                row_data = {
                    '자산':'KRW','보유':fmt_full_number(bal,0),'주문중':fmt_full_number(locked,0) if locked else '',
                    '평균매입가':'-','현재가':'-','평가금액':fmt_full_number(amt,0),'손익%':'-'
                }
                rows.append(row_data)
            else:
                market = f'KRW-{cur}'; price_raw = price_map.get(market)
                try:
                    price = float(price_raw)
                except Exception:
                    price = None
                eval_krw = (amt * price) if price is not None else None
                if eval_krw is not None:
                    total_eval += eval_krw; total_crypto_eval += eval_krw
                purchase_cost = None
                if avg_buy > 0 and amt > 0:
                    purchase_cost = avg_buy * amt; total_purchase_cost += purchase_cost
                pnl_pct = None
                if price is not None and avg_buy > 0:
                    try:
                        pnl_pct = (price / avg_buy - 1) * 100
                    except Exception:
                        pnl_pct = None
                row_data = {
                    '자산':cur,'보유':fmt_coin_amount(bal),'주문중':fmt_coin_amount(locked) if locked else '',
                    '평균매입가':fmt_price(avg_buy) if avg_buy > 0 else '-', '현재가':fmt_price(price) if price is not None else '-',
                    '평가금액':fmt_full_number(eval_krw,0) if eval_krw is not None else '-',
                    '손익%': (f"{pnl_pct:.2f}%" if pnl_pct is not None else '-')
                }
                rows.append(row_data)
        except Exception:
            pass
    # 평가금액이 '-' (가격 미수신) 인 행 제거하여 기존 동작 복원
    rows = [r for r in rows if r.get('평가금액') != '-']
    # KRW 행은 항상 맨 앞에 정렬 유지
    if rows:
        krw_rows=[r for r in rows if r.get('자산')=='KRW']; other=[r for r in rows if r.get('자산')!='KRW']; rows=krw_rows+other
    orderable_krw=0.0
    for a in accounts:
        if a.get('currency')=='KRW':
            try: orderable_krw += float(a.get('balance') or 0)
            except Exception: pass
    total_purchase_cost=total_purchase_cost
    evaluation_profit=None; roi=None
    if total_purchase_cost>0:
        evaluation_profit = total_crypto_eval - total_purchase_cost
        roi = (evaluation_profit/total_purchase_cost)*100
    row1 = st.columns(4); row2 = st.columns(3)
    row1[0].metric('보유 KRW', fmt_full_number(krw_bal,0))
    row1[1].metric('주문가능', fmt_full_number(orderable_krw,0))
    row1[2].metric('총 평가', fmt_full_number(total_eval,0))
    row1[3].metric('총 매수', fmt_full_number(total_purchase_cost,0) if total_purchase_cost>0 else '-')
    row2[0].metric('총 보유자산', fmt_full_number(total_eval,0))
    row2[1].metric('평가손익', fmt_full_number(evaluation_profit,0) if evaluation_profit is not None else '-')
    row2[2].metric('수익률', f"{roi:.2f}%" if roi is not None else '-')
    df = pd.DataFrame(rows)
    if not df.empty and '손익%' in df.columns:
        def style_pnl(col):
            styled=[]
            for v in col:
                if isinstance(v,str) and v.endswith('%'):
                    try:
                        num=float(v[:-1]); color=signed_color(num); styled.append(f'color: {color};' if color!='inherit' else '')
                    except Exception:
                        styled.append('')
                else:
                    styled.append('')
            return styled
        styled_df = df.style.apply(style_pnl, subset=['손익%'])
        st.dataframe(styled_df, hide_index=True, use_container_width=True)
    else:
        st.dataframe(df, hide_index=True, use_container_width=True)
