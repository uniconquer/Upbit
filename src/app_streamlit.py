
from __future__ import annotations
import os
import requests
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from upbit_api import UpbitAPI
try:
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
except Exception:  # st_aggrid가 없을 경우 대비
    AgGrid = None

st.set_page_config(page_title="Upbit Dashboard", layout="wide")

# 전역 스타일 (색상/폰트 통일) - 중복 삽입 방지
if '_base_style_injected' not in st.session_state:
    st.markdown(
        """
        <style>
        .signed-number {font-size:1.4rem; font-weight:600; font-family: inherit; line-height:1;}
        .signed-number.pos {color:red;}
        .signed-number.neg {color:blue;}
        /* 손익% 테이블 값은 Styler로 색칠되지만 폰트 통일 */
        div[data-testid='stDataFrame'] td {font-family: inherit;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.session_state['_base_style_injected'] = True

load_dotenv()
ACCESS = os.getenv("UPBIT_ACCESS_KEY")
SECRET = os.getenv("UPBIT_SECRET_KEY")
api = UpbitAPI(access_key=ACCESS, secret_key=SECRET)

@st.cache_data(ttl=30)
def load_tickers(markets: list[str]):
    """지정된 마켓들의 최신 시세를 반환 (캐시 30초).
    실패 시 세션 상태에 오류 저장 후 개별 마켓 fallback 시도.
    """
    if not markets:
        return []
    try:
        return api.tickers(markets)
    except Exception as e:  # 일괄 호출 실패 -> 개별 재시도
        st.session_state['_ticker_error'] = f"batch_error: {repr(e)}"
        out = []
        failed = []
        for m in markets:
            try:
                t = api.ticker(m)
                out.append(t)
            except Exception as ie:
                failed.append(f"{m}:{type(ie).__name__}")
        if failed:
            st.session_state['_ticker_error'] = st.session_state.get('_ticker_error','') + \
                f"; failed_markets={','.join(failed[:10])}{'...' if len(failed)>10 else ''}"
        return out

@st.cache_data(ttl=300)
def load_all_markets_set() -> set[str]:
    """모든 마켓 문자열 집합 (유효성 검사용)."""
    try:
        return {m.get("market") for m in api.markets() if m.get("market")}
    except Exception:
        return set()

def fmt_full_number(v, decimals_if_needed: int = 2):
    try:
        f = float(v)
    except Exception:
        return "-"
    if f.is_integer():
        return f"{int(f):,}"
    return f"{f:,.{decimals_if_needed}f}"

def fmt_coin_amount(v, max_decimals: int = 8) -> str:
    try:
        f = float(v)
    except Exception:
        return "-"
    s = f"{f:.{max_decimals}f}".rstrip('0').rstrip('.')
    if s == "":
        return "0"
    return s

fmt_price = fmt_full_number  # 동일 포맷 사용
fmt_krw = lambda v: fmt_full_number(v, 0)  # KRW는 기본 0자리 (정수) 표기

# 과거 디버그용 메타/핑/키 마스킹 유틸 제거됨

# -------------------- 색상 헬퍼 (양수=빨강, 음수=파랑) -------------------- #
def signed_color(value: float | int | None) -> str:
    if value is None:
        return 'inherit'
    try:
        v = float(value)
    except Exception:
        return 'inherit'
    if v > 0:
        return 'red'
    if v < 0:
        return 'blue'
    return 'inherit'

def colorize_number(value: float | int | None, *, is_percent: bool = False, decimals: int = 2) -> str:
    """숫자를 span(클래스 기반)으로 감싸 색상만 변경 (폰트/사이즈 전역 CSS)."""
    if value is None:
        return '-'
    try:
        v = float(value)
    except Exception:
        return str(value)
    if is_percent:
        text = f"{v:.{decimals}f}%"
    else:
        if abs(v - int(v)) < 1e-9:
            text = fmt_full_number(int(v), 0)
        else:
            text = fmt_full_number(v, decimals)
    cls = 'pos' if v > 0 else ('neg' if v < 0 else '')
    return f"<span class='signed-number {cls}'>{text}</span>"
with st.sidebar:
    st.title("Upbit Dashboard")
    if "view" not in st.session_state:
        st.session_state["view"] = "account"
    st.caption("좌측 버튼으로 화면 전환")
    if st.button("내 자산", use_container_width=True):
        st.session_state["view"] = "account"
    if st.button("백테스팅", use_container_width=True):
        st.session_state["view"] = "backtest"
    if st.button("라이브", use_container_width=True):
        st.session_state["view"] = "live"
    st.markdown(f"**현재 뷰:** `{st.session_state['view']}`")
view = st.session_state['view']

if view == "account":
    st.header("내 자산")
    # 계좌 조회
    accounts = []
    raw_error = None
    try:
        accounts = api.accounts()
    except Exception as e:
        raw_error = str(e)
    if raw_error:
        st.error(f"계좌 조회 실패: {raw_error}")
    if not accounts:
        st.warning("보유 자산이 없습니다.")
    else:
        # 필요 시세 마켓 수집
        need = []
        for a in accounts:
            cur = a.get("currency")
            bal_str = a.get("balance") or "0"
            locked_str = a.get("locked") or "0"
            try:
                bal_all = float(bal_str) + float(locked_str)
            except Exception:
                bal_all = 0.0
            if cur and cur != "KRW" and bal_all > 0:
                need.append(f"KRW-{cur}")
        need = list(dict.fromkeys(need))
        tick = load_tickers(need)
        price_map = {t["market"]: t.get("trade_price") for t in tick}
        rows = []
        total_eval = 0.0
        krw_bal = 0.0
        total_crypto_eval = 0.0
        total_purchase_cost = 0.0
        for a in accounts:
            try:
                cur = a.get("currency")
                bal = float(a.get("balance") or 0)
                locked = float(a.get("locked") or 0)
                amt = bal + locked
                avg_buy = float(a.get("avg_buy_price") or 0)
                if cur == "KRW":
                    krw_bal += amt
                    total_eval += amt
                    rows.append({
                        "자산": "KRW",
                        "보유": fmt_full_number(bal, 0),
                        "주문중": fmt_full_number(locked, 0) if locked else "",
                        "평균매입가": "-",
                        "현재가": "-",
                        "평가금액": fmt_full_number(amt, 0),
                        "손익%": "-",
                    })
                else:
                    market = f"KRW-{cur}"
                    price_raw = price_map.get(market)
                    try:
                        price = float(price_raw)
                    except Exception:
                        price = None
                    eval_krw = (amt * price) if (price is not None) else None
                    if eval_krw is not None:
                        total_eval += eval_krw
                        total_crypto_eval += eval_krw
                    purchase_cost = None
                    if avg_buy > 0 and amt > 0:
                        purchase_cost = avg_buy * amt
                        total_purchase_cost += purchase_cost
                    pnl_pct = None
                    if price is not None and avg_buy > 0:
                        try:
                            pnl_pct = (price / avg_buy - 1) * 100
                        except Exception:
                            pnl_pct = None
                    rows.append({
                        "자산": cur,
                        "보유": fmt_coin_amount(bal),
                        "주문중": fmt_coin_amount(locked) if locked else "",
                        "평균매입가": fmt_price(avg_buy) if avg_buy > 0 else "-",
                        "현재가": fmt_price(price) if price is not None else "-",
                        "평가금액": fmt_full_number(eval_krw, 0) if eval_krw is not None else "-",
                        "손익%": (f"{pnl_pct:.2f}%" if pnl_pct is not None else "-"),
                    })
            except Exception:
                pass
        if need and not price_map:
            err = st.session_state.get('_ticker_error')
            all_set = load_all_markets_set()
            invalid = [m for m in need if m not in all_set] if all_set else []
            st.warning(
                "시세 데이터를 불러오지 못했습니다 (요청 {n}개). 마켓목록: {ml}\n오류: {err}\n잘못된 마켓: {inv}".format(
                    n=len(need), ml=','.join(need), err=err, inv=','.join(invalid) if invalid else '(없음)')
            )
        # 평가금액 미존재 코인 제거 (KRW 제외)
        if rows:
            rows = [r for r in rows if not (r.get("자산") != "KRW" and r.get("평가금액") == "-")]
        if rows:
            krw_rows = [r for r in rows if r.get("자산") == "KRW"]
            other_rows = [r for r in rows if r.get("자산") != "KRW"]
            rows = krw_rows + other_rows
        # Summary metrics
        orderable_krw = 0.0
        for a in accounts:
            if a.get("currency") == "KRW":
                try:
                    orderable_krw += float(a.get("balance") or 0)
                except Exception:
                    pass
        evaluation_profit = None
        roi = None
        if total_purchase_cost > 0:
            evaluation_profit = total_crypto_eval - total_purchase_cost
            roi = (evaluation_profit / total_purchase_cost) * 100
        row1 = st.columns(4)
        row2 = st.columns(3)
        row1[0].metric("보유 KRW", fmt_full_number(krw_bal, 0))
        row1[1].metric("주문가능", fmt_full_number(orderable_krw, 0))
        row1[2].metric("총 평가", fmt_full_number(total_eval, 0))
        row1[3].metric("총 매수", fmt_full_number(total_purchase_cost, 0) if total_purchase_cost > 0 else "-")
        row2[0].metric("총 보유자산", fmt_full_number(total_eval, 0))
        row2[1].metric("평가손익", fmt_full_number(evaluation_profit, 0) if evaluation_profit is not None else "-")
        row2[2].metric("수익률", f"{roi:.2f}%" if roi is not None else "-")
        # Table
        df = pd.DataFrame(rows)
        if not df.empty and '손익%' in df.columns:
            def style_pnl(col):
                styled = []
                for v in col:
                    if isinstance(v, str) and v.endswith('%'):
                        try:
                            num = float(v[:-1])
                            color = signed_color(num)
                            styled.append(f'color: {color};' if color != 'inherit' else '')
                        except Exception:
                            styled.append('')
                    else:
                        styled.append('')
                return styled
            styled_df = df.style.apply(style_pnl, subset=['손익%'])
            st.dataframe(styled_df, hide_index=True, use_container_width=True)
        else:
            st.dataframe(df, hide_index=True, use_container_width=True)
elif view == "backtest":
    st.header("백테스팅")

    @st.cache_data(ttl=600)
    def load_krw_markets_meta() -> pd.DataFrame:
        try:
            ms = api.markets()
            rows = []
            for m in ms:
                mk = m.get("market", "")
                if not mk.startswith("KRW-"):
                    continue
                rows.append({
                    'market': mk,
                    'symbol': mk.split('-')[1],
                    'korean_name': m.get('korean_name'),
                    'english_name': m.get('english_name')
                })
            return pd.DataFrame(rows)
        except Exception:
            return pd.DataFrame()

    @st.cache_data(ttl=60)
    def load_market_tickers(markets: tuple[str, ...]) -> pd.DataFrame:
        if not markets:
            return pd.DataFrame()
        try:
            # chunk via api.tickers
            data = api.tickers(list(markets))
            rows = []
            for t in data:
                rows.append({
                    'market': t.get('market'),
                    'trade_price': t.get('trade_price'),
                    'acc_trade_volume_24h': t.get('acc_trade_volume_24h'),
                    'acc_trade_price_24h': t.get('acc_trade_price_24h'),
                })
            return pd.DataFrame(rows)
        except Exception:
            return pd.DataFrame()

    @st.cache_data(ttl=60)
    def load_candles_cached(market: str, interval: str, count: int):
        try:
            cds = api.candles(market, interval=interval, count=count)
            data = [{
                "time": c.timestamp,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
            } for c in cds]
            df = pd.DataFrame(data)
            if not df.empty:
                df['dt'] = pd.to_datetime(df['time'], unit='ms')
            return df
        except Exception:
            return pd.DataFrame()

    meta_df = load_krw_markets_meta()
    if meta_df.empty:
        st.error("KRW 마켓 메타를 불러오지 못했습니다.")
    else:
        tk_df = load_market_tickers(tuple(meta_df['market'].tolist()))
        merged = meta_df.merge(tk_df, on='market', how='left')
        # 거래대금 기준 정렬
        merged['acc_trade_price_24h'] = pd.to_numeric(merged['acc_trade_price_24h'], errors='coerce')
        merged = merged.sort_values('acc_trade_price_24h', ascending=False)
        # 표시용 포맷
        disp = merged.copy()
        disp['24h거래대금(KRW)'] = disp['acc_trade_price_24h'].apply(lambda v: fmt_full_number(v,0) if pd.notnull(v) else '-')
        disp['현재가'] = disp['trade_price'].apply(lambda v: fmt_full_number(v,0) if pd.notnull(v) else '-')
        show_cols = ['market','symbol','korean_name','현재가','24h거래대금(KRW)']

        ui_cols = st.columns([2,3])
        with ui_cols[0]:
            st.subheader("코인 목록")
            # 원본 DF -> subset (검색 반영)
            q = st.text_input("검색 (심볼/이름)", '', placeholder="예: BTC 또는 비트")
            table_df = disp.copy()
            if q.strip():
                uq = q.strip().upper()
                table_df = table_df[table_df.apply(lambda r: uq in str(r['market']).upper() or uq in str(r['symbol']).upper() or uq in str(r['korean_name']).upper(), axis=1)]
            subset = table_df.to_dict('records')
            # 선택 상태 초기화
            if 'bt_selected_market' not in st.session_state and subset:
                st.session_state['bt_selected_market'] = subset[0]['market']
            selected_market = st.session_state.get('bt_selected_market')

            def _short_val(v):
                # 간단: 전체 천단위 포맷 (요청에 따라 축약 대신 전체 표시 유지)
                try:
                    return fmt_full_number(float(v), 0)
                except Exception:
                    return '-'

            if AgGrid:
                # df_src 구성 (숨김 numeric + 표시 컬럼)
                df_src = pd.DataFrame([
                    {
                        'market': r['market'],
                        'name': r['korean_name'],
                        'value24h': r.get('acc_trade_price_24h'),
                        'value24h_short': _short_val(r.get('acc_trade_price_24h'))
                    } for r in subset
                ])
                if df_src.empty:
                    st.info("검색 결과가 없습니다.")
                else:
                    gb = GridOptionsBuilder.from_dataframe(df_src)
                    gb.configure_column('market', headerName='마켓', width=120)
                    gb.configure_column('name', headerName='이름', width=130)
                    gb.configure_column('value24h', headerName='value24h', hide=True, type=['numericColumn'])
                    gb.configure_column('value24h_short', headerName='24h거래대금', width=140)
                    # 기본 정렬: value24h desc
                    gb.configure_grid_options(sortModel=[{'colId': 'value24h', 'sort': 'desc'}])
                    gb.configure_selection('single', use_checkbox=False)
                    gb.configure_grid_options(rowHeight=28, suppressCellFocus=True, animateRows=False,
                                               enableCellTextSelection=False, domLayout='normal')
                    gb.configure_default_column(resizable=True, sortable=True, filter=True)
                    # quick filter (client side)
                    if q.strip():
                        gb.configure_grid_options(quickFilterText=q.strip())
                    grid_options = gb.build()
                    grid = AgGrid(
                        df_src,
                        gridOptions=grid_options,
                        height=620,
                        fit_columns_on_grid_load=True,
                        update_mode=GridUpdateMode.SELECTION_CHANGED,
                        allow_unsafe_jscode=False,
                        theme='alpine',
                        key='bt_aggrid'
                    )
                    sel_rows_raw = grid.get('selected_rows', [])
                    if isinstance(sel_rows_raw, pd.DataFrame):
                        sel_rows = sel_rows_raw.to_dict('records')
                    else:
                        sel_rows = sel_rows_raw if isinstance(sel_rows_raw, list) else []
                    if sel_rows:
                        first = sel_rows[0]
                        if isinstance(first, dict) and 'market' in first:
                            new_sel = first['market']
                            if new_sel != selected_market:
                                st.session_state['bt_selected_market'] = new_sel
                    else:
                        # 선택 없으면 현재 state 유지, state 가 subset 밖이면 첫 행으로
                        markets_now = [r['market'] for r in subset]
                        if markets_now and selected_market not in markets_now:
                            st.session_state['bt_selected_market'] = markets_now[0]
            else:
                # Fallback: selectbox
                options = [r['market'] for r in subset]
                if options:
                    if selected_market not in options:
                        selected_market = options[0]
                    new_sel = st.selectbox('코인', options, index=options.index(selected_market))
                    if new_sel != selected_market:
                        st.session_state['bt_selected_market'] = new_sel
                else:
                    st.info("표시할 마켓이 없습니다.")

            selected_market = st.session_state.get('bt_selected_market')
        with ui_cols[1]:
            pcols = st.columns([1,1])
            with pcols[0]:
                interval = st.selectbox("봉 간격", ["day","week","month","minute60","minute15","minute5"], index=0)
            with pcols[1]:
                count = st.slider("캔들 수", 30, 400, 120, 10)
            if selected_market:
                df_c = load_candles_cached(selected_market, interval, count)
                if df_c.empty:
                    st.warning("캔들 데이터를 불러오지 못했습니다.")
                else:
                    import plotly.graph_objects as go
                    fig = go.Figure(data=[go.Candlestick(
                        x=df_c['dt'],
                        open=df_c['open'], high=df_c['high'], low=df_c['low'], close=df_c['close'],
                        increasing_line_color='red', decreasing_line_color='blue'
                    )])
                    fig.update_layout(margin=dict(l=10,r=10,t=30,b=20), height=500, xaxis_rangeslider_visible=False,
                                      title=f"{selected_market} ({interval})")
                    st.plotly_chart(fig, use_container_width=True)
        with st.expander("전략 파라미터 / 시뮬레이션 (추가 예정)"):
            st.write("이곳에 전략 설정(예: 이동평균, RSI 등)과 백테스트 결과(지표, 에쿼티, 트레이드 로그)를 추가할 예정입니다.")
elif view == "live":
    st.header("라이브 (준비중)")
    st.info("라이브 매매 모듈은 아직 구현되지 않았습니다. 이 영역에 모니터링 UI를 추가하세요.")
    with st.expander("예시 Placeholder"):
        st.write("실시간 시세, 포지션, 로그 등이 여기에 표시됩니다.")

# 끝. (마켓/백테스트/라이브 제거됨)
