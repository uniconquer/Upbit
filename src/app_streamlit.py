import os
from datetime import datetime, timedelta
import streamlit as st
from dotenv import load_dotenv
from upbit_api import UpbitAPI
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from mr_worker import MRMonitor, fetch_top_markets  # 라이브 모니터 재사용
import time  # live view 주기 실행용

# st_autorefresh 함수가 Streamlit 버전에 따라 없을 수 있으므로 안전하게 import 시도
try:  # type: ignore
    from streamlit import st_autorefresh  # noqa: F401
except Exception:  # pragma: no cover - 호환성 처리
    st_autorefresh = None  # fallback 으로 meta refresh 사용

load_dotenv()

st.set_page_config(page_title="Upbit Markets", layout="wide")


@st.cache_data(ttl=30)
def load_markets_and_tickers(_api: UpbitAPI):
    all_markets = _api.markets()
    krw = [m for m in all_markets if isinstance(m.get('market'), str) and m['market'].startswith('KRW-')]
    markets = [m['market'] for m in krw]
    tickers = _api.tickers(markets)
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

# 단순 캔들 캐시 (스캔 제거 후 최소 기능)
@st.cache_data(ttl=60)
def load_candles_cached(market: str, interval: str, count: int):
    try:
        cds = api.candles(market, interval=interval, count=count)
    except Exception:
        return []
    return [c.model_dump() for c in cds]

def _get_query_params():
    # Streamlit 버전에 따른 호환 처리
    try:
        if hasattr(st, 'query_params'):
            return dict(st.query_params)
        return st.experimental_get_query_params()
    except Exception:
        return {}

def _set_query_params(**kwargs):
    try:
        if hasattr(st, 'query_params'):
            st.query_params.update(kwargs)
        else:
            st.experimental_set_query_params(**kwargs)
    except Exception:
        pass

# 초기 뷰 결정: session_state -> URL -> 기본값
if 'active_view' not in st.session_state:
    qp = _get_query_params()
    st.session_state['active_view'] = qp.get('view', ['markets'])[0] if isinstance(qp.get('view'), list) else qp.get('view', 'markets')

with st.sidebar:
    # 라디오 위젯은 별도 key 사용하여 위젯 상태와 논리 상태 분리
    current = st.session_state.get('active_view', 'markets')
    radio_choice = st.radio(
        "메뉴",
        ("markets", "account", "backtest", "live"),
        index=["markets","account","backtest","live"].index(current) if current in ["markets","account","backtest","live"] else 0,
        format_func=lambda x: {"markets":"마켓 불러오기","account":"내 정보 보기","backtest":"백테스트","live":"라이브"}[x],
        key='nav_choice'
    )
    st.caption("세션 재시작 / 자동 새로고침에도 뷰 유지 (URL 동기화)")

# 라이브 모니터 동작 중이면 뷰 고정 (위젯 선택 무시) / 아니면 라디오 선택 반영
if 'live_monitor' in st.session_state and st.session_state['live_monitor'] is not None:
    if st.session_state.get('active_view') != 'live':
        st.session_state['active_view'] = 'live'
        st.toast('라이브 실행 중: 뷰를 라이브로 고정합니다.', icon='⚙️')
else:
    # 사용자가 선택한 라디오 메뉴 적용
    if st.session_state.get('active_view') != radio_choice:
        st.session_state['active_view'] = radio_choice

view = st.session_state.get('active_view', 'markets')

# URL query params 동기화
_set_query_params(view=view)

if view == 'markets':
    st.title('KRW 마켓 상위 거래대금')
    colA, colB = st.columns([0.15, 0.85])
    with colA:
        refresh_btn = st.button('새로고침', key='_mk_refresh')
    if 'markets_rows' not in st.session_state or refresh_btn:
        if refresh_btn:
            try:
                load_markets_and_tickers.clear()
            except Exception:
                pass
        try:
            st.session_state['markets_rows'] = load_markets_and_tickers(api)
        except Exception as e:
            st.error(f'마켓 로드 실패: {e}')
            st.session_state['markets_rows'] = []
    rows = st.session_state.get('markets_rows', [])
    if not rows:
        st.info('표시할 마켓 데이터가 없습니다.')
    else:
        max_n = min(500, len(rows))
        show_n = st.slider('표시 개수', 10, max_n, value=min(50, max_n), step=10, key='_mk_show_n')
        q = st.text_input('검색 (심볼/이름)', '', key='_mk_search')
        filtered = rows[:show_n]
        if q:
            ql = q.lower()
            filtered = [r for r in filtered if (r.get('market') or '').lower().find(ql) >= 0 or (str(r.get('name_ko') or '').lower().find(ql) >= 0)]
        def _short_val(v):
            try:
                f = float(v or 0)
                if f >= 1e12: return f"{f/1e12:.1f}조"
                if f >= 1e8: return f"{f/1e8:.1f}억"
                if f >= 1e6: return f"{f/1e6:.1f}백만"
                if f >= 1e4: return f"{f/1e4:.1f}만"
                return f"{int(f)}"
            except Exception:
                return '-'
        disp = []
        for r in filtered:
            disp.append({
                '마켓': r.get('market'),
                '이름': r.get('name_ko'),
                '현재가': fmt(r.get('price'), 4),
                '24h거래대금': _short_val(r.get('value24h')),
                '24h변동%': (f"{r['change24h_pct']:.2f}%" if r.get('change24h_pct') is not None else '-')
            })
        st.dataframe(disp, hide_index=True, use_container_width=True)

if view == 'account':
    st.title("내 보유 자산")
    if not access or not secret:
        st.warning("API 키가 설정되지 않았습니다. .env 에 UPBIT_ACCESS_KEY / UPBIT_SECRET_KEY 를 넣고 재시작하세요.")
    else:
        @st.cache_data(ttl=20)
        def load_accounts_cached():
            return api.accounts()
        colA, colB = st.columns([0.15,0.85])
        with colA:
            if st.button('새로고침', key='_acct_refresh'):
                load_accounts_cached.clear()
        try:
            accounts = load_accounts_cached()
        except Exception as e:
            st.error(f"계정 API 오류: {e}")
            accounts = []
        if not accounts:
            st.info('표시할 계정 데이터가 없습니다.')
        else:
            # 가격 조회 대상 (KRW 제외) + 유효 마켓 필터
            try:
                all_rows = load_markets_and_tickers(api)  # KRW-* 만 포함
                valid_market_set = {r['market'] for r in all_rows}
            except Exception:
                valid_market_set = set()
            need_prices = []
            skipped_symbols = []
            for a in accounts:
                try:
                    bal = float(a.get('balance') or 0) + float(a.get('locked') or 0)
                    cur = a.get('currency')
                    if cur and cur != 'KRW' and bal > 0:
                        mkt = f"KRW-{cur}"
                        if valid_market_set and mkt not in valid_market_set:
                            skipped_symbols.append(cur)
                            continue
                        need_prices.append(mkt)
                except Exception:
                    continue
            need_prices = list(dict.fromkeys(need_prices))

            def fetch_prices_safe(markets_list):
                prices_map = {}
                if not markets_list:
                    return prices_map
                try:
                    # bulk 시 하나라도 invalid 면 404 발생 → except 블록에서 세분 처리
                    tks_bulk = api.tickers(markets_list)
                    for t in tks_bulk:
                        try:
                            prices_map[t.get('market')] = float(t.get('trade_price') or 0)
                        except Exception:
                            pass
                    return prices_map
                except Exception:
                    # fallback: 개별 호출
                    for m in markets_list:
                        try:
                            t = api.ticker(m)
                            prices_map[m] = float(t.get('trade_price') or 0)
                        except Exception:
                            pass
                    return prices_map

            prices: dict[str,float] = fetch_prices_safe(need_prices)

            rows=[]; total_krw=0.0; total_eval=0.0
            price_zero_symbols = []  # 현재가 0 또는 미조회로 숨긴 심볼
            # 총매수/평가 계산용
            total_cost_basis = 0.0      # 비 KRW 자산 평균매입 금액 * 수량 합
            total_asset_eval = 0.0      # 비 KRW 자산 현재 평가 합
            for a in accounts:
                try:
                    cur = a.get('currency')
                    bal = float(a.get('balance') or 0)
                    locked = float(a.get('locked') or 0)
                    total_amt = bal + locked
                    avg_buy = float(a.get('avg_buy_price') or 0)
                    if cur == 'KRW':
                        price = 1.0
                    else:
                        price = prices.get(f"KRW-{cur}", 0.0)
                        if price <= 0:
                            # 현재가 없는(0) 자산은 표에서 제외
                            if total_amt > 0:
                                price_zero_symbols.append(cur)
                            continue
                    eval_krw = total_amt * price
                    if cur == 'KRW':
                        total_krw += total_amt
                    else:
                        total_asset_eval += eval_krw
                        if avg_buy > 0 and total_amt > 0:
                            total_cost_basis += avg_buy * total_amt
                    total_eval += eval_krw if cur != 'KRW' else total_amt
                    pnl_pct = None
                    if cur != 'KRW' and avg_buy > 0 and price > 0:
                        pnl_pct = (price/avg_buy - 1) * 100
                    rows.append({
                        '자산': cur,
                        '보유': fmt(bal,8),
                        '주문중': fmt(locked,8) if locked else '',
                        '총수량': fmt(total_amt,8),
                        '평균매입가': fmt(avg_buy,2) if cur!='KRW' else '-',
                        '현재가(KRW)': fmt(price,2) if cur!='KRW' else '-',
                        '평가금액(KRW)': fmt(eval_krw,2) if cur!='KRW' else fmt(total_amt,0),
                        '손익%': (f"{pnl_pct:.2f}%" if pnl_pct is not None else '-')
                    })
                except Exception:
                    pass
            # 상단 요약 (총매수/총평가/평가손익/수익률)
            unrealized_pnl = total_asset_eval - total_cost_basis
            if total_cost_basis > 0:
                return_pct = (unrealized_pnl / total_cost_basis) * 100
            else:
                return_pct = 0.0
            sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
            sum_col1.markdown(f"**총매수**<br>{fmt(total_cost_basis,0)}", unsafe_allow_html=True)
            sum_col2.markdown(f"**총평가**<br>{fmt(total_asset_eval,0)}", unsafe_allow_html=True)
            sum_col3.markdown(f"**평가손익**<br><span style='color:#d62728'>{fmt(unrealized_pnl,0)}</span>", unsafe_allow_html=True)
            sum_col4.markdown(f"**수익률**<br><span style='color:#d62728'>{return_pct:.2f}%</span>", unsafe_allow_html=True)
            st.divider()
            # 정렬: 평가금액 큰 순 (KRW 는 항상 최상단)
            def _eval_sort(r):
                try:
                    v = r['평가금액(KRW)'].replace(',','')
                    return float(v) if v not in ('-','') else 0.0
                except Exception:
                    return 0.0
            krw_rows = [r for r in rows if r.get('자산') == 'KRW']
            other_rows = [r for r in rows if r.get('자산') != 'KRW']
            other_rows.sort(key=_eval_sort, reverse=True)
            rows = (krw_rows + other_rows)
            col1,col2,col3 = st.columns(3)
            col1.metric('KRW 잔액', fmt(total_krw,0))
            col2.metric('총 평가 (KRW)', fmt(total_eval,0))
            if total_eval>0:
                krw_ratio = total_krw/total_eval*100
                col3.metric('현금비중 %', f"{krw_ratio:.1f}%")
            st.dataframe(rows, hide_index=True, use_container_width=True)
            if skipped_symbols:
                st.caption("가격 조회 제외 (상장폐지/비유효): " + ", ".join(sorted(set(skipped_symbols))))
            if price_zero_symbols:
                st.caption("현재가 0으로 숨김: " + ", ".join(sorted(set(price_zero_symbols))))

elif view == 'backtest':
    # 백테스트 / 차트 뷰
    rows_all = load_markets_and_tickers(api)
    if not rows_all:
        st.warning("마켓 데이터를 불러오지 못했습니다.")
    else:
        if 'selected_backtest_market' not in st.session_state:
            st.session_state['selected_backtest_market'] = rows_all[0]['market']
        if 'bt_interval' not in st.session_state:
            st.session_state['bt_interval'] = 'day'

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

        col_left, col_right = st.columns([1,2])
        with col_left:
            st.subheader("마켓 목록")
            max_n = min(400, len(rows_all))
            show_n = st.slider("표시 개수", 20, max_n, value=min(80,max_n), step=10, key="_bt_show_n")
            subset = rows_all[:show_n]
            from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
            df_src = pd.DataFrame([
                {
                    'market': r['market'],
                    'name': r['name_ko'],
                    'value24h': r['value24h'],
                    'value24h_short': _short_val(r['value24h'])
                } for r in subset
            ])
            gb = GridOptionsBuilder.from_dataframe(df_src)
            gb.configure_column('market', headerName='마켓', width=120)
            gb.configure_column('name', headerName='이름', width=120)
            gb.configure_column('value24h', headerName='value24h', hide=True, type=['numericColumn'])
            gb.configure_column('value24h_short', headerName='24h거래대금', width=120)
            gb.configure_grid_options(sortModel=[{'colId':'value24h','sort':'desc'}])
            gb.configure_selection('single', use_checkbox=False)
            gb.configure_grid_options(rowHeight=28, suppressCellFocus=True)
            gb.configure_default_column(resizable=True, sortable=True, filter=True)
            q = st.text_input("검색 (심볼/이름)", '')
            if q:
                gb.configure_grid_options(quickFilterText=q)
            grid = AgGrid(
                df_src,
                gridOptions=gb.build(),
                height=620,
                fit_columns_on_grid_load=True,
                update_mode=GridUpdateMode.SELECTION_CHANGED,
                allow_unsafe_jscode=False,
                theme='alpine'
            )
            sel_rows_raw = grid.get('selected_rows', [])
            if isinstance(sel_rows_raw, pd.DataFrame):
                sel_rows = sel_rows_raw.to_dict('records')
            else:
                sel_rows = sel_rows_raw if isinstance(sel_rows_raw, list) else []
            if sel_rows:
                first = sel_rows[0]
                if isinstance(first, dict) and 'market' in first:
                    st.session_state['selected_backtest_market'] = first['market']
            else:
                if subset and st.session_state['selected_backtest_market'] not in [r['market'] for r in subset]:
                    st.session_state['selected_backtest_market'] = subset[0]['market']

        with col_right:
            sel = st.session_state['selected_backtest_market']
            st.subheader(f"{sel} 캔들 차트")
            interval_options = ["minute1","minute15","minute60","day","week","month"]
            if 'bt_interval' not in st.session_state:
                st.session_state['bt_interval'] = interval_options[0]
            cols_int = st.columns(len(interval_options))
            for i,opt in enumerate(interval_options):
                label = opt.replace('minute','')+'m' if opt.startswith('minute') else opt.upper()
                if cols_int[i].button(label, key=f"bt_int_{opt}"):
                    st.session_state['bt_interval']=opt
            interval = st.session_state['bt_interval']
            range_mode = st.checkbox("기간 선택 모드", value=False)
            if not range_mode:
                count = st.slider("캔들", 30, 400, value=120, step=10)
            else:
                today = datetime.now().date()
                default_start = today - timedelta(days=7)
                c1,c2 = st.columns(2)
                with c1:
                    start_date = st.date_input("시작일", value=default_start, key="_range_start")
                with c2:
                    end_date = st.date_input("종료일", value=today, key="_range_end")
                if interval.startswith('minute'):
                    c3,c4 = st.columns(2)
                    with c3:
                        start_time = st.time_input("시작 시각", value=datetime.now().replace(hour=0,minute=0,second=0,microsecond=0).time(), key="_range_start_time")
                    with c4:
                        end_time = st.time_input("종료 시각", value=datetime.now().time(), key="_range_end_time")
                else:
                    start_time = datetime.min.time(); end_time = datetime.max.time().replace(microsecond=0)
            refresh = st.button("새로고침")

            def fetch_candles_range(market: str, interval: str, start_dt: datetime, end_dt: datetime, max_calls: int=30):
                all_items=[]; cursor=end_dt; last_earliest=None
                for _ in range(max_calls):
                    to_str = cursor.strftime('%Y-%m-%d %H:%M:%S')
                    batch = api.candles(market, interval=interval, count=200, to=to_str)
                    if not batch: break
                    keep=[]
                    for c in batch:
                        c_ts = datetime.fromtimestamp(c.timestamp/1000)
                        if c_ts> end_dt or c_ts< start_dt: continue
                        keep.append(c)
                    all_items.extend(keep)
                    earliest_ts = batch[0].timestamp
                    if last_earliest is not None and earliest_ts >= last_earliest: break
                    last_earliest = earliest_ts
                    earliest_dt = datetime.fromtimestamp(earliest_ts/1000)
                    if earliest_dt <= start_dt: break
                    cursor = earliest_dt - timedelta(seconds=1)
                uniq={(c.timestamp,c.open,c.close):c for c in all_items}
                out=list(uniq.values()); out.sort(key=lambda x:x.timestamp)
                return [c.model_dump() for c in out]

            try:
                with st.spinner("캔들 불러오는 중..."):
                    if refresh:
                        load_candles_cached.clear()
                    if range_mode:
                        if start_date > end_date:
                            st.error("시작일이 종료일보다 늦습니다."); raw_candles=[]
                        else:
                            if interval.startswith('minute'):
                                start_dt = datetime.combine(start_date, start_time); end_dt = datetime.combine(end_date, end_time)
                            else:
                                start_dt = datetime.combine(start_date, datetime.min.time()); end_dt = datetime.combine(end_date, datetime.max.time().replace(microsecond=0))
                            raw_candles = fetch_candles_range(sel, interval, start_dt, end_dt)
                            if len(raw_candles)==0:
                                st.info("범위 내 캔들 없음")
                            elif len(raw_candles)>5000:
                                st.warning(f"수집 캔들 {len(raw_candles)}개 (성능 저하 가능)")
                    else:
                        raw_candles = load_candles_cached(sel, interval, count)
            except Exception as e:
                st.error(str(e)); raw_candles=[]
            if not raw_candles:
                st.warning("캔들 데이터 없음")
            else:
                df = pd.DataFrame(raw_candles)
                df['dt'] = pd.to_datetime(df['timestamp'], unit='ms')
                bb_period=20; bb_k=2
                if len(df) >= bb_period:
                    df['bb_mid']=df['close'].rolling(bb_period).mean()
                    df['bb_std']=df['close'].rolling(bb_period).std()
                    df['bb_upper']=df['bb_mid']+bb_k*df['bb_std']
                    df['bb_lower']=df['bb_mid']-bb_k*df['bb_std']
                else:
                    df['bb_mid']=df['bb_upper']=df['bb_lower']=None
                show_bb = st.checkbox("Bollinger Bands (20, 2σ)", value=True, key="_show_bb")
                rsi_period=14; delta=df['close'].diff(); gain=np.where(delta>0,delta,0); loss=np.where(delta<0,-delta,0)
                if len(df)>rsi_period:
                    roll_up=pd.Series(gain).ewm(alpha=1/rsi_period, adjust=False).mean()
                    roll_down=pd.Series(loss).ewm(alpha=1/rsi_period, adjust=False).mean(); rs=roll_up/roll_down
                    df['rsi']=100-(100/(1+rs))
                else:
                    df['rsi']=np.nan
                show_rsi = st.checkbox("RSI (14)", value=True, key="_show_rsi")
                if show_rsi:
                    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6,0.25,0.15], vertical_spacing=0.02)
                else:
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7,0.3], vertical_spacing=0.02)
                fig.add_trace(go.Candlestick(x=df['dt'],open=df['open'],high=df['high'],low=df['low'],close=df['close'],name='OHLC',increasing_line_color='#d62728',increasing_fillcolor='rgba(214,39,40,0.7)',decreasing_line_color='#1f77b4',decreasing_fillcolor='rgba(31,119,180,0.7)'), row=1,col=1)
                if show_bb and 'bb_mid' in df.columns:
                    valid = df['bb_upper'].notna() & df['bb_lower'].notna()
                    if valid.any():
                        fig.add_trace(go.Scatter(x=df.loc[valid,'dt'], y=df.loc[valid,'bb_lower'], line=dict(color='rgba(150,150,200,0.2)', width=0), showlegend=False, hoverinfo='skip'), row=1,col=1)
                        fig.add_trace(go.Scatter(x=df.loc[valid,'dt'], y=df.loc[valid,'bb_upper'], line=dict(color='rgba(150,150,200,0.2)', width=0), fill='tonexty', fillcolor='rgba(150,150,200,0.15)', name='BBands', hoverinfo='skip'), row=1,col=1)
                        fig.add_trace(go.Scatter(x=df.loc[valid,'dt'], y=df.loc[valid,'bb_mid'], line=dict(color='#3366cc', width=1), name='BB Mid'), row=1,col=1)
                fig.add_trace(go.Bar(x=df['dt'], y=df['volume'], name='Volume', marker_color='#888'), row=2,col=1)
                # Mean Reversion 섹션 (기존 코드 유지)
                st.markdown("**Mean Reversion 전략:** 하단밴드/RSI 과매도 진입 → 중간/상단밴드 혹은 RSI 과매수 청산 (선택적 SL/TP)**")
                show_mr = st.checkbox("Mean Reversion 전략 표시", value=False, key="_show_mr")
                mr_trades=[]
                if show_rsi:
                    # RSI subplot 라인
                    if df['rsi'].notna().any():
                        fig.add_trace(go.Scatter(x=df['dt'], y=df['rsi'], line=dict(color='#ff9900', width=1.2), name='RSI'), row=3, col=1)
                        for level,color in [(70,'red'),(30,'green')]:
                            fig.add_hline(y=level, line=dict(color=color,width=1,dash='dot'), row=3, col=1)
                        fig.update_yaxes(range=[0,100], row=3, col=1, title_text='RSI')
                if show_mr and 'bb_lower' in df.columns and df['bb_lower'].notna().any():
                    colmr1,colmr2,colmr3,colmr4,colmr5 = st.columns(5)
                    with colmr1: mr_bb_period = st.number_input("볼린저 밴드 기간",10,100,20,1,key="_mr_bb_period")
                    with colmr2: mr_bb_k = st.number_input("볼린저 밴드 계수",1.0,4.0,2.0,0.1,key="_mr_bb_k")
                    with colmr3: mr_rsi_buy = st.number_input("RSI 매수 이하",5,50,30,1,key="_mr_rsi_buy")
                    with colmr4: mr_rsi_sell = st.number_input("RSI 매도 이상",50,95,70,1,key="_mr_rsi_sell")
                    with colmr5: mr_capital = st.number_input("1회 진입 수량(가정)",0.0, value=1.0, step=0.1, key="_mr_capital")
                    colsl1,colsl2,colsl3 = st.columns(3)
                    with colsl1: mr_stop_pct = st.number_input("StopLoss % (손절)",0.0,50.0,0.0,0.5,key="_mr_sl_pct")/100.0
                    with colsl2: mr_tp_pct = st.number_input("TakeProfit % (익절)",0.0,200.0,0.0,1.0,key="_mr_tp_pct")/100.0
                    with colsl3: mr_exit_mid = st.checkbox("중심선 도달시 청산", value=True, key="_mr_exit_mid")
                    if mr_bb_period!=20 or mr_bb_k!=2:
                        if len(df) >= mr_bb_period:
                            mid2=df['close'].rolling(int(mr_bb_period)).mean(); std2=df['close'].rolling(int(mr_bb_period)).std()
                            upper2=mid2+mr_bb_k*std2; lower2=mid2-mr_bb_k*std2
                        else:
                            mid2=pd.Series([np.nan]*len(df)); upper2=lower2=mid2
                    else:
                        mid2=df['bb_mid']; upper2=df['bb_upper']; lower2=df['bb_lower']
                    rsi_series=df.get('rsi', pd.Series([np.nan]*len(df)))
                    entry_mask=(df['close']<lower2)|(rsi_series<mr_rsi_buy)
                    exit_mask=(df['close']>upper2)|(rsi_series>mr_rsi_sell)
                    if mr_exit_mid: exit_mask = exit_mask | (df['close']>mid2)
                    in_pos=False; entry_price=0.0; entry_dt=None; entries_x=[]; entries_y=[]; exits_x=[]; exits_y=[]
                    for i in range(len(df)):
                        c=float(df.iloc[i]['close'])
                        if (not in_pos) and entry_mask.iloc[i] and not np.isnan(c):
                            in_pos=True; entry_price=c; entry_dt=df.iloc[i]['dt']; entries_x.append(df.iloc[i]['dt']); entries_y.append(c); continue
                        if in_pos:
                            hit_sl=False; hit_tp=False
                            low=df.iloc[i]['low']; high=df.iloc[i]['high']
                            if mr_stop_pct>0 and not np.isnan(low) and low <= entry_price*(1-mr_stop_pct):
                                c_exit=entry_price*(1-mr_stop_pct); hit_sl=True
                            if (not hit_sl) and mr_tp_pct>0 and not np.isnan(high) and high >= entry_price*(1+mr_tp_pct):
                                c_exit=entry_price*(1+mr_tp_pct); hit_tp=True
                            if not (hit_sl or hit_tp):
                                if exit_mask.iloc[i] and not np.isnan(c):
                                    c_exit=c
                                else:
                                    continue
                            ret_pct=(c_exit/entry_price-1.0)
                            mr_trades.append({'entry_dt':entry_dt,'exit_dt':df.iloc[i]['dt'],'entry':entry_price,'exit':c_exit,'ret_pct':ret_pct*100,'stop':hit_sl,'take':hit_tp})
                            exits_x.append(df.iloc[i]['dt']); exits_y.append(c_exit); in_pos=False
                    if in_pos:
                        last=df.iloc[-1]; ret_pct=(last['close']/entry_price-1.0)
                        mr_trades.append({'entry_dt':entry_dt,'exit_dt':None,'entry':entry_price,'exit':np.nan,'ret_pct':ret_pct*100,'stop':False,'take':False})
                    if entries_x:
                        fig.add_trace(go.Scatter(x=entries_x,y=entries_y,mode='markers',marker=dict(symbol='triangle-up',size=11,color='#00cc96',line=dict(width=1,color='#006644')),name='MR Entry'), row=1,col=1)
                    if exits_x:
                        fig.add_trace(go.Scatter(x=exits_x,y=exits_y,mode='markers',marker=dict(symbol='triangle-down',size=11,color='#ef553b',line=dict(width=1,color='#661a00')),name='MR Exit'), row=1,col=1)
                fig.update_layout(height=760 if show_rsi else 650, margin=dict(t=30,l=10,r=10,b=10), xaxis_rangeslider_visible=False, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                if len(df):
                    last_row=df.iloc[-1]
                    c1,c2,c3,c4=st.columns(4)
                    c1.metric("종가", fmt(last_row['close'],0)); c2.metric("고가", fmt(last_row['high'],0)); c3.metric("저가", fmt(last_row['low'],0))
                    try: vol_short=_short_val(last_row['volume'])
                    except Exception: vol_short=fmt(last_row['volume'],4)
                    c4.metric("거래량", vol_short)
                if show_mr and mr_trades:
                    closed=[t for t in mr_trades if t['exit_dt'] is not None]
                    wins=[t for t in closed if t['ret_pct']>0]; losses=[t for t in closed if t['ret_pct']<=0]
                    win_rate=(len(wins)/len(closed)*100) if closed else 0.0
                    avg_ret=(np.mean([t['ret_pct'] for t in closed]) if closed else 0.0)
                    tot_ret=(np.prod([(1+t['ret_pct']/100) for t in closed]) -1)*100 if closed else 0.0
                    st.markdown(f"**Mean Reversion 트레이드** (완료 {len(closed)}건 / 전체 {len(mr_trades)}건) | 승률 {win_rate:.1f}% | 평균수익 {avg_ret:.2f}% | 누적 {tot_ret:.2f}%")
                    disp=[]
                    for t in sorted(mr_trades, key=lambda x:x['entry_dt'], reverse=True):
                        disp.append({
                            '진입시각': t['entry_dt'].strftime('%Y-%m-%d %H:%M') if t['entry_dt'] else '-',
                            '청산시각': t['exit_dt'].strftime('%Y-%m-%d %H:%M') if t['exit_dt'] else '(보유중)',
                            '진입가': fmt(t['entry'],2),
                            '청산가': fmt(t['exit'],2) if not np.isnan(t['exit']) else '-',
                            '수익률%': f"{t['ret_pct']:.2f}%",
                            'SL': 'Y' if t['stop'] else '',
                            'TP': 'Y' if t['take'] else ''
                        })
                    st.dataframe(pd.DataFrame(disp), hide_index=True, use_container_width=True)

elif view == 'live':
    st.title('라이브 자동 매매 (Mean Reversion)')
    live_col1, live_col2 = st.columns([30, 70])
    with live_col1:
        with st.form('live_params_form', clear_on_submit=False):
            st.subheader('설정')
            interval = st.selectbox('캔들 인터벌', ['minute15','minute30','minute60','day'], index=0, key='_live_interval')
            markets_n = st.number_input('거래대금 상위 N개 대상', 5, 200, 20, 5, key='_live_markets_n')
            loop_seconds = st.number_input('주기(초)', 30, 600, 120, 10, key='_live_loop_seconds')
            bb_period = st.number_input('볼린저 밴드 기간', 10, 100, 20, 1, key='_live_bb_period')
            bb_k = st.number_input('볼린저 밴드 계수', 1.0, 4.0, 2.0, 0.1, key='_live_bb_k')
            rsi_buy = st.number_input('RSI 매수 이하', 5, 50, 30, 1, key='_live_rsi_buy')
            rsi_sell = st.number_input('RSI 매도 이상', 50, 95, 70, 1, key='_live_rsi_sell')
            exit_mid = st.checkbox('중심선 도달시 청산', value=True, key='_live_exit_mid')
            stop_pct = st.number_input('StopLoss % (손절)', 0.0, 80.0, 0.0, 0.5, key='_live_stop_pct') / 100.0
            take_pct = st.number_input('TakeProfit % (익절)', 0.0, 300.0, 0.0, 1.0, key='_live_take_pct') / 100.0
            krw_per_trade = st.number_input('1회 매수 KRW', 1000, 10000000, 5000, 1000, key='_live_krw_per_trade')
            max_open = st.number_input('최대 동시 포지션', 1, 30, 5, 1, key='_live_max_open')
            # ---- Rate Limit Options ----
            min_fetch_seconds = st.number_input('마켓 최소 재호출 간격(초)', 5.0, 600.0, 20.0, 5.0, key='_live_min_fetch_seconds')
            per_request_sleep = st.number_input('마켓 사이 지연(초)', 0.0, 2.0, 0.12, 0.01, key='_live_per_request_sleep')
            live_possible = (os.getenv('UPBIT_LIVE') == '1') and bool(access and secret)
            live_orders_flag = st.checkbox('실제 주문 실행 (UPBIT_LIVE=1 필요, 매우 주의)', value=False, disabled=not live_possible, key='_live_live_orders')
            submitted = st.form_submit_button('시작 / 재시작')
            stop_clicked = st.form_submit_button('중지')
        # 시작/중지 처리
        if stop_clicked and 'live_monitor' in st.session_state:
            for k in ['live_monitor','live_markets','live_loop_seconds','live_last_run','live_messages']:
                st.session_state.pop(k, None)
            st.success('라이브 중지됨')
        if submitted:
            # 초기화 후 새 모니터 생성
            for k in ['live_monitor','live_markets','live_loop_seconds','live_last_run','live_messages']:
                st.session_state.pop(k, None)
            try:
                mkts = fetch_top_markets(api, base='KRW', limit=int(markets_n))
            except Exception as e:
                st.error(f'마켓 조회 실패: {e}')
                mkts = []
            mon = MRMonitor(api,
                            interval=interval,
                            bb_period=int(bb_period), bb_k=float(bb_k),
                            rsi_buy=int(rsi_buy), rsi_sell=int(rsi_sell),
                            exit_mid=exit_mid, stop_pct=stop_pct, take_pct=take_pct,
                            live_orders=bool(live_orders_flag), krw_per_trade=float(krw_per_trade), max_open=int(max_open),
                            min_fetch_seconds=float(min_fetch_seconds), per_request_sleep=float(per_request_sleep))
            # 메시지 캡처
            st.session_state['live_messages'] = []
            def _ui_notify(msg: str):
                st.session_state['live_messages'].append({'t': datetime.utcnow(), 'msg': msg})
                if mon.notifier.available():
                    mon.notifier.send_text(msg)
            mon._notify = _ui_notify
            st.session_state['live_monitor'] = mon
            st.session_state['live_markets'] = mkts
            st.session_state['live_loop_seconds'] = int(loop_seconds)
            st.session_state['live_last_run'] = 0.0
            st.success(f'라이브 시작: 대상 {len(mkts)}개')
    with live_col2:
        st.subheader('상태')
        mon = st.session_state.get('live_monitor')
        if not mon:
            st.info('좌측에서 파라미터 설정 후 시작하세요.')
        else:
            loop_seconds = st.session_state.get('live_loop_seconds', 120)
            markets = st.session_state.get('live_markets', [])
            last_run = st.session_state.get('live_last_run', 0.0)
            now_ts = time.time()
            due = (now_ts - last_run) >= loop_seconds
            # 카운트다운 & 진행률
            if last_run > 0:
                elapsed = now_ts - last_run
                remaining = max(0, int(loop_seconds - elapsed)) if not due else 0
                progress_ratio = min(1.0, elapsed / loop_seconds) if loop_seconds>0 else 0
            else:
                elapsed = 0; remaining = 0; progress_ratio = 0
            cda, cdb, cdc = st.columns(3)
            cda.metric('마지막 실행(UTC)', datetime.utcfromtimestamp(last_run).strftime('%H:%M:%S') if last_run>0 else '-')
            cdb.metric('다음 실행까지', '실행중' if due else f'{remaining}s')
            try:
                cdc.progress(int(progress_ratio*100))
            except Exception:
                pass
            st.write(f"모드: {'LIVE' if mon.live_orders else 'SIM'} | 인터벌: {mon.interval} | 주기: {loop_seconds}s | 대상: {len(markets)} | 재호출간격:{mon.min_fetch_seconds}s")
            if due and markets:
                for m in markets:
                    try:
                        mon.process_market(m)
                    except Exception as e:
                        st.session_state['live_messages'].append({'t': datetime.utcnow(), 'msg': f'ERR {m} {e}'})
                st.session_state['live_last_run'] = now_ts
            st_autorefresh_ms = min(max(loop_seconds, 10), 300) * 1000
            if st_autorefresh:
                st_autorefresh(interval=st_autorefresh_ms, key='_live_autorefresh')
            else:
                st.caption('자동 새로고침: 내장 함수 미지원 → meta refresh 사용 중')
                st.markdown(f"<meta http-equiv='refresh' content='{loop_seconds}'>", unsafe_allow_html=True)
                if st.button('즉시 새로고침', key='_live_manual_refresh'):
                    st.session_state['live_last_run'] = 0
                    try:
                        st.experimental_rerun()
                    except Exception:
                        pass
            if mon.positions:
                try:
                    pos_markets = list(mon.positions.keys())
                    tks = api.tickers(pos_markets)
                    price_map = {t['market']: float(t.get('trade_price') or 0) for t in tks}
                except Exception:
                    price_map = {}
                pos_rows = []
                for mk, p in mon.positions.items():
                    cur_price = price_map.get(mk)
                    pnl_pct = (cur_price / p.entry_price - 1) * 100 if cur_price else 0.0
                    pos_rows.append({
                        '마켓': mk,
                        '진입가': fmt(p.entry_price,4),
                        '현재가': fmt(cur_price,4) if cur_price else '-',
                        '수익률%': f"{pnl_pct:.2f}%",
                        '수량≈': f"{p.volume:.6f}",
                        '진입시각(UTC)': p.entry_time.strftime('%H:%M:%S')
                    })
                st.markdown('**포지션**')
                st.dataframe(pos_rows, hide_index=True, use_container_width=True)
            else:
                st.info('보유 포지션 없음')
            st.markdown('**최근 이벤트**')
            msgs = st.session_state.get('live_messages', [])[-80:]
            if msgs:
                for item in reversed(msgs):
                    st.write(f"[{item['t'].strftime('%H:%M:%S')}] {item['msg']}")
            else:
                st.caption('이벤트 없음')
            if mon.live_orders and os.getenv('UPBIT_LIVE') != '1':
                st.warning('UPBIT_LIVE=1 이 아니므로 실제 주문이 차단되고 있을 수 있습니다.')
            if not mon.live_orders:
                st.caption('SIM 모드: 실제 주문 전송 안 함. 실매매하려면 UPBIT_LIVE=1 + 체크박스.')
