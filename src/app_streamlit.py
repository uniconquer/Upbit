
import os
import streamlit as st
from dotenv import load_dotenv
from upbit_api import UpbitAPI
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

load_dotenv()

st.set_page_config(page_title="Upbit Markets", layout="wide")


@st.cache_data(ttl=30)
def load_markets_and_tickers(_api: UpbitAPI):  # underscore to prevent hashing custom object
    all_markets = _api.markets()
    krw = [m for m in all_markets if isinstance(m.get('market'), str) and m['market'].startswith('KRW-')]
    markets = [m['market'] for m in krw]
    tickers = _api.tickers(markets)
    # index by market for join of names
    name_map = {m['market']: m.get('korean_name') for m in krw}
    rows = []
    for t in tickers:
        try:
            change_rate = t.get('signed_change_rate') or t.get('change_rate')
            change_pct = float(change_rate) * 100 if change_rate is not None else None
        except Exception:
            change_pct = None
        rows.append({
            'market': t.get('market'),
            'name_ko': name_map.get(t.get('market')),
            'price': t.get('trade_price'),
            'value24h': t.get('acc_trade_price_24h'),
            'volume24h': t.get('acc_trade_volume_24h'),
            'change24h_pct': change_pct,
        })
    # sort by value24h desc
    rows.sort(key=lambda r: float(r['value24h'] or 0), reverse=True)
    return rows


def fmt(v, digits=2):
    if v is None:
        return '-'
    try:
        if isinstance(v, str):
            v = float(v)
        if float(v).is_integer():
            return f"{int(v):,}"
        return f"{v:,.{digits}f}"
    except Exception:
        return v


access = os.getenv('UPBIT_ACCESS_KEY')
secret = os.getenv('UPBIT_SECRET_KEY')
api = UpbitAPI(access_key=access, secret_key=secret)

with st.sidebar:
    load_btn = st.button("마켓 불러오기", width=200)
#    st.divider()
    acct_btn = st.button("내 정보 보기", width=200)
#    st.divider()
    backtest_btn = st.button("백테스트", width=200)

# 뷰 상태 관리 (markets/account/backtest)
if 'active_view' not in st.session_state:
    st.session_state['active_view'] = 'markets'
if load_btn:
    st.session_state['active_view'] = 'markets'
elif 'acct_btn' in locals() and acct_btn:
    st.session_state['active_view'] = 'account'
elif 'backtest_btn' in locals() and backtest_btn:
    st.session_state['active_view'] = 'backtest'

view = st.session_state['active_view']

if view == 'account':
    st.title("내 보유 자산")
elif view == 'backtest':
    st.title("백테스트")
else:
    st.title("KRW Top 10 Markets")

if view == 'markets' and load_btn:
    with st.spinner("불러오는 중..."):
        try:
            rows = load_markets_and_tickers(api)
        except Exception as e:
            st.error(f"API 오류: {e}")
            rows = []
    top10 = rows[:10]
    if not top10:
        st.warning("데이터 없음")
    else:
        # 포맷
        display = []
        for r in top10:
            display.append({
                '마켓': r['market'],
                '이름': r['name_ko'],
                '현재가': fmt(r['price'], 0),
                '24h 거래대금': fmt(r['value24h'], 0),
                '24h 거래량': fmt(r['volume24h'], 4),
                '등락률%': (f"{r['change24h_pct']:.2f}%" if r['change24h_pct'] is not None else '-')
            })
        st.dataframe(display, hide_index=True, use_container_width=True)
elif view == 'markets':
    st.info("좌측 '마켓 불러오기' 버튼을 눌러 주세요")

if view == 'markets':
    st.caption("업데이트 주기: 캐시 30초. 새로고침하려면 버튼 재클릭.")

if view == 'account':
    if not access or not secret:
        st.warning(".env 에 UPBIT_ACCESS_KEY / UPBIT_SECRET_KEY 설정 필요")
    else:
        try:
            accounts = api.accounts()
        except Exception as e:
            st.error(f"계정 조회 실패: {e}")
        else:
            # 시장 목록 가져와 KRW- 매핑
            try:
                mkts = api.markets()
            except Exception:
                mkts = []
            krw_set = {m.get('market') for m in mkts if isinstance(m.get('market'), str) and m['market'].startswith('KRW-')}
            # 수집할 마켓들
            target_markets = []
            for a in accounts:
                cur = a.get('currency')
                if cur and cur != 'KRW':
                    m = f"KRW-{cur}"
                    if m in krw_set:
                        target_markets.append(m)
            prices_map = {}
            if target_markets:
                try:
                    tks = api.tickers(target_markets)
                    for t in tks:
                        prices_map[t.get('market')] = t.get('trade_price')
                except Exception:
                    pass
            rows_acct = []  # raw rows with numeric fields for sorting
            for a in accounts:
                cur = a.get('currency')
                bal = float(a.get('balance') or 0)
                locked = float(a.get('locked') or 0)
                avg = float(a.get('avg_buy_price') or 0)
                market_name = f"KRW-{cur}" if cur and cur != 'KRW' else None
                last_price = prices_map.get(market_name) if market_name else 1.0
                eval_krw = (bal + locked) * (last_price or 0)
                pnl_pct = None
                profit_amt = None
                cost = None
                if avg and last_price and avg > 0 and cur != 'KRW':
                    try:
                        pnl_pct = (last_price/avg - 1) * 100
                        cost = (bal + locked) * avg
                        profit_amt = eval_krw - cost
                    except Exception:
                        pnl_pct = None
                        profit_amt = None
                elif cur == 'KRW':
                    profit_amt = 0.0
                rows_acct.append({
                    '자산': cur,
                    '보유수량': fmt(bal, 6),
                    '주문중': fmt(locked, 6),
                    '평균매수가': fmt(avg, 0),
                    '현재가': fmt(last_price, 0) if cur != 'KRW' else '-',
                    '평가액(KRW)': fmt(eval_krw, 0),
                    '수익금(KRW)': fmt(profit_amt, 0) if profit_amt is not None else '-',
                    '수익률%': (f"{pnl_pct:.2f}%" if pnl_pct is not None else '-'),
                    # raw fields for sorting (prefixed underscore to avoid display collision)
                    '_eval': eval_krw,
                    '_profit': (profit_amt if profit_amt is not None else float('-inf')),
                    '_pnl_pct': (pnl_pct if pnl_pct is not None else float('-inf')),
                })
            # 총 평가액 요약
            total_eval = sum([
                ((float(a.get('balance') or 0) + float(a.get('locked') or 0)) * (prices_map.get(f"KRW-{a.get('currency')}") or 1 if a.get('currency')!='KRW' else 1))
                for a in accounts
            ])
            # 항상 수익금 기준 내림차순 정렬
            rows_acct.sort(key=lambda r: r['_profit'], reverse=True)
            # build display rows without raw keys
            display_rows = [
                {k: v for k, v in r.items() if not k.startswith('_')}
                for r in rows_acct
            ]
            st.metric("총 평가액 (KRW)", fmt(total_eval, 0))
            st.dataframe(display_rows, hide_index=True, use_container_width=True)

if view == 'backtest':
    # 좌측: KRW 마켓 리스트 (24h 거래대금 순), 우측: 선택 마켓 캔들 차트
    @st.cache_data(ttl=60)
    def load_all_krw_markets():
        try:
            rows_all = load_markets_and_tickers(api)  # 이미 24h 거래대금 내림차순 정렬
        except Exception as e:
            st.error(f"마켓 조회 실패: {e}")
            return []
        return rows_all

    @st.cache_data(ttl=60)
    def load_candles_cached(market: str, interval: str, count: int):
        try:
            cds = api.candles(market, interval=interval, count=count)
            # 캐시 직렬화를 위해 dict 로 변환
            return [c.model_dump() for c in cds]
        except Exception as e:
            raise RuntimeError(f"캔들 조회 실패: {e}")

    rows_all = load_all_krw_markets()
    if not rows_all:
        st.warning("마켓 데이터가 없습니다.")
    else:
        # 기본 선택 상태 초기화
        if 'selected_backtest_market' not in st.session_state:
            st.session_state['selected_backtest_market'] = rows_all[0]['market']

        col_left, col_right = st.columns([1, 2])

        def _short_val(v):
            try:
                f=float(v or 0)
                if f>=1e12: return f"{f/1e12:.1f}조"
                if f>=1e8: return f"{f/1e8:.1f}억"
                if f>=1e6: return f"{f/1e6:.1f}백만"
                if f>=1e4: return f"{f/1e4:.1f}만"
                return str(int(f))
            except Exception:
                return '-'

        with col_left:
            st.subheader("마켓 목록")
            max_n = min(400, len(rows_all))
            show_n = st.slider("표시 개수", min_value=20, max_value=max_n, value=min(80, max_n), step=10, key="_bt_show_n")
            subset = rows_all[:show_n]

            # AgGrid 적용
            from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
            # DataFrame 구성
            df_src = pd.DataFrame([
                {
                    'market': r['market'],
                    'name': r['name_ko'],
                    'value24h': r['value24h'],            # numeric (for 정렬)
                    'value24h_short': _short_val(r['value24h'])  # 표시용
                } for r in subset
            ])
            # Grid 옵션 빌드
            gb = GridOptionsBuilder.from_dataframe(df_src)
            gb.configure_column('market', headerName='마켓', width=120)
            gb.configure_column('name', headerName='이름', width=120)
            # 숨김 numeric 컬럼 (정렬용)
            gb.configure_column('value24h', headerName='value24h', hide=True, type=['numericColumn'])
            # 표시용 짧은 표기 컬럼
            gb.configure_column('value24h_short', headerName='24h거래대금', width=120)
            # 기본 정렬: value24h desc
            gb.configure_grid_options(sortModel=[{'colId':'value24h','sort':'desc'}])
            gb.configure_selection('single', use_checkbox=False)
            gb.configure_grid_options(rowHeight=28, suppressCellFocus=True, animateRows=False, enableCellTextSelection=False, domLayout='normal')
            gb.configure_default_column(resizable=True, sortable=True, filter=True)
            # 빠른 필터 텍스트 (옵션)
            q = st.text_input("검색 (심볼/이름)", '')
            if q:
                # 간단 client-side quick filter: gridOptions.quickFilterText 이용
                gb.configure_grid_options(quickFilterText=q)
            grid_options = gb.build()
            grid = AgGrid(
                df_src,
                gridOptions=grid_options,
                height=620,
                fit_columns_on_grid_load=True,
                update_mode=GridUpdateMode.SELECTION_CHANGED,
                allow_unsafe_jscode=False,
                theme='alpine'
            )
            sel_rows_raw = grid.get('selected_rows', [])
            # st_aggrid 가 리스트 또는 DataFrame 형태를 줄 수 있으니 모두 대응
            if isinstance(sel_rows_raw, pd.DataFrame):
                sel_rows = sel_rows_raw.to_dict('records')
            else:
                sel_rows = sel_rows_raw if isinstance(sel_rows_raw, list) else []
            if len(sel_rows) > 0:
                first = sel_rows[0]
                if isinstance(first, dict) and 'market' in first:
                    new_sel = first['market']
                    if new_sel != st.session_state['selected_backtest_market']:
                        st.session_state['selected_backtest_market'] = new_sel
            else:
                if subset and st.session_state['selected_backtest_market'] not in [r['market'] for r in subset]:
                    st.session_state['selected_backtest_market'] = subset[0]['market']

        with col_right:
            sel = st.session_state['selected_backtest_market']
            st.subheader(f"{sel} 캔들 차트")
            # Interval 버튼 그룹 (Top-down 제거)
            interval_options = ["day", "minute15", "minute60", "week", "month"]
            if 'bt_interval' not in st.session_state:
                st.session_state['bt_interval'] = interval_options[0]
            cols_int = st.columns(len(interval_options))
            for i, opt in enumerate(interval_options):
                active = (st.session_state['bt_interval'] == opt)
                btn_label = opt.upper() if not opt.startswith('minute') else opt
                if cols_int[i].button(btn_label, key=f"bt_int_{opt}"):
                    st.session_state['bt_interval'] = opt
            interval = st.session_state['bt_interval']
            count = st.slider("캔들", min_value=30, max_value=400, value=120, step=10)
            refresh = st.button("새로고침")

            try:
                with st.spinner("캔들 불러오는 중..."):
                    if refresh:
                        load_candles_cached.clear()
                    raw_candles = load_candles_cached(sel, interval, count)
            except Exception as e:
                st.error(str(e))
            else:
                if not raw_candles:
                    st.warning("캔들 데이터 없음")
                else:
                    df = pd.DataFrame(raw_candles)
                    df['dt'] = pd.to_datetime(df['timestamp'], unit='ms')

                    # Bollinger Bands 계산 (기간 20, 표준편차 2)
                    bb_period = 20
                    bb_k = 2
                    if len(df) >= bb_period:
                        df['bb_mid'] = df['close'].rolling(bb_period).mean()
                        df['bb_std'] = df['close'].rolling(bb_period).std()
                        df['bb_upper'] = df['bb_mid'] + bb_k * df['bb_std']
                        df['bb_lower'] = df['bb_mid'] - bb_k * df['bb_std']
                    else:
                        df['bb_mid'] = df['bb_upper'] = df['bb_lower'] = None

                    show_bb = st.checkbox("Bollinger Bands (20, 2σ)", value=True, key="_show_bb")

                    # RSI 계산 (Wilder, period=14)
                    rsi_period = 14
                    show_rsi = st.checkbox("RSI (14)", value=True, key="_show_rsi")
                    if show_rsi and len(df) > rsi_period:
                        delta = df['close'].diff()
                        gain = np.where(delta > 0, delta, 0)
                        loss = np.where(delta < 0, -delta, 0)
                        roll_up = pd.Series(gain).ewm(alpha=1/rsi_period, adjust=False).mean()
                        roll_down = pd.Series(loss).ewm(alpha=1/rsi_period, adjust=False).mean()
                        rs = roll_up / roll_down
                        df['rsi'] = 100 - (100 / (1 + rs))
                    else:
                        df['rsi'] = np.nan

                    # 서브플롯: 0=가격,1=거래량,2=RSI (옵션)
                    use_rows = 3 if show_rsi else 2
                    if show_rsi:
                        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.25, 0.15], vertical_spacing=0.02)
                    else:
                        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.02)
                    fig.add_trace(go.Candlestick(
                        x=df['dt'],
                        open=df['open'], high=df['high'], low=df['low'], close=df['close'],
                        name='OHLC',
                        increasing_line_color='#d62728',
                        increasing_fillcolor='rgba(214,39,40,0.7)',
                        decreasing_line_color='#1f77b4',
                        decreasing_fillcolor='rgba(31,119,180,0.7)'
                    ), row=1, col=1)
                    if show_bb and 'bb_mid' in df.columns:
                        # 밴드 영역 (lower 먼저, 그 위에 upper with fill)
                        valid = df['bb_upper'].notna() & df['bb_lower'].notna()
                        if valid.any():
                            fig.add_trace(go.Scatter(
                                x=df.loc[valid, 'dt'], y=df.loc[valid, 'bb_lower'],
                                line=dict(color='rgba(150,150,200,0.2)', width=0),
                                showlegend=False, hoverinfo='skip'
                            ), row=1, col=1)
                            fig.add_trace(go.Scatter(
                                x=df.loc[valid, 'dt'], y=df.loc[valid, 'bb_upper'],
                                line=dict(color='rgba(150,150,200,0.2)', width=0),
                                fill='tonexty', fillcolor='rgba(150,150,200,0.15)',
                                name='BBands', hoverinfo='skip'
                            ), row=1, col=1)
                            fig.add_trace(go.Scatter(
                                x=df.loc[valid, 'dt'], y=df.loc[valid, 'bb_mid'],
                                line=dict(color='#3366cc', width=1),
                                name='BB Mid'
                            ), row=1, col=1)
                    fig.add_trace(go.Bar(x=df['dt'], y=df['volume'], name='Volume', marker_color='#888'), row=2, col=1)
                    if show_rsi:
                        # RSI 라인 & 70/30 레벨 밴드
                        if df['rsi'].notna().any():
                            fig.add_trace(go.Scatter(x=df['dt'], y=df['rsi'], line=dict(color='#ff9900', width=1.2), name='RSI'), row=3, col=1)
                            for level, color in [(70, 'red'), (30, 'green')]:
                                fig.add_hline(y=level, line=dict(color=color, width=1, dash='dot'), row=3, col=1)
                            fig.update_yaxes(range=[0, 100], row=3, col=1, title_text='RSI')
                    fig.update_layout(
                        height=760 if show_rsi else 650,
                        margin=dict(t=30, l=10, r=10, b=10),
                        xaxis_rangeslider_visible=False,
                        showlegend=False,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    last_row = df.iloc[-1]
                    colm1, colm2, colm3, colm4 = st.columns(4)
                    colm1.metric("종가", fmt(last_row['close'], 0))
                    colm2.metric("고가", fmt(last_row['high'], 0))
                    colm3.metric("저가", fmt(last_row['low'], 0))
                    # 거래량 축약 표기
                    try:
                        vol_short = _short_val(last_row['volume'])
                    except Exception:
                        vol_short = fmt(last_row['volume'], 4)
                    colm4.metric("거래량", vol_short)

    st.caption("백테스트(시각화) 데이터 캐시: 마켓/캔들 60초. 새로고침 버튼으로 즉시 갱신 가능.")
