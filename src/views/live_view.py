from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import time, threading
from upbit_api import UpbitAPI
from notifier import get_notifier
from utils.formatters import fmt_full_number

api: UpbitAPI | None = None
flux_indicator = None

# --- 내부 TTL 캐시 (Streamlit 비의존) ---
_MARKETS_CACHE = {'data': None, 'ts': 0.0}
_CANDLES_CACHE = {}  # key=(market, interval, count) => {'data':df,'ts':t}
_CACHE_NOW = time.time  # alias

# 외부에서 init

def init_api(a: UpbitAPI, flux):
    global api, flux_indicator
    api = a; flux_indicator = flux

def _markets_rank():
    ttl=120
    now=_CACHE_NOW()
    if _MARKETS_CACHE['data'] is not None and (now - _MARKETS_CACHE['ts'] < ttl):
        return _MARKETS_CACHE['data']
    if not api:
        return pd.DataFrame()
    try:
        ms=api.markets(); dfm=pd.DataFrame(ms)
        if dfm.empty:
            _MARKETS_CACHE.update({'data':pd.DataFrame(),'ts':now}); return pd.DataFrame()
        dfm=dfm[dfm['market'].str.startswith('KRW-')]
        mk_list=dfm['market'].tolist(); tk=api.tickers(mk_list); dft=pd.DataFrame(tk)
        if not dft.empty:
            dft=dft[['market','acc_trade_price_24h','acc_trade_volume_24h','trade_price']]
        out=dfm.merge(dft, on='market', how='left')
        out['acc_trade_price_24h']=pd.to_numeric(out['acc_trade_price_24h'], errors='coerce')
        out=out.sort_values('acc_trade_price_24h', ascending=False)
        _MARKETS_CACHE.update({'data':out,'ts':now})
        return out
    except Exception:
        return pd.DataFrame()

def _candles(market: str, interval: str, count: int):
    ttl=60; now=_CACHE_NOW(); key=(market, interval, int(count))
    ce=_CANDLES_CACHE.get(key)
    if ce and (now - ce['ts'] < ttl):
        return ce['data']
    if not api:
        return pd.DataFrame()
    try:
        cds=api.candles(market, interval=interval, count=count)
        rows=[{'time':c.timestamp,'open':c.open,'high':c.high,'low':c.low,'close':c.close,'volume':c.volume} for c in cds]
        df=pd.DataFrame(rows)
        if not df.empty:
            df['dt']=pd.to_datetime(df['time'], unit='ms')
            df=df.set_index('dt').sort_index()
        _CANDLES_CACHE[key]={'data':df,'ts':now}
        return df
    except Exception:
        return pd.DataFrame()

def _simple_bt(df_sig: pd.DataFrame, fee: float=0.0005):
    if df_sig.empty or 'close' not in df_sig: return {'trades':0,'total_return_pct':0,'win_rate_pct':0,'max_drawdown_pct':0,'equity':None}
    if 'buy_signal' not in df_sig or 'sell_signal' not in df_sig: return {'trades':0,'total_return_pct':0,'win_rate_pct':0,'max_drawdown_pct':0,'equity':None}
    in_pos=False; entry=0.0; eq=1.0; equity=[]; trades=[]; peak=-1e9; max_dd=0.0
    for ts,row in df_sig.iterrows():
        price=row.get('close')
        if price is None or np.isnan(price): continue
        if (not in_pos) and bool(row.get('buy_signal')): in_pos=True; entry=float(price)
        if in_pos and bool(row.get('sell_signal')):
            gross=float(price)/entry; net=gross*(1-fee)*(1-fee); eq*=net; trades.append((entry,float(price),(net-1)*100)); in_pos=False
        equity.append(eq)
        if eq>peak: peak=eq
        dd=(eq/peak -1)*100 if peak>0 else 0
        if dd<max_dd: max_dd=dd
    if in_pos: pass
    if not equity:
        return {'trades':0,'total_return_pct':0,'win_rate_pct':0,'max_drawdown_pct':0,'equity':None}
    total_return_pct=(equity[-1]-1)*100
    win_rate = (sum(1 for t in trades if t[2]>0)/len(trades)*100) if trades else 0
    return {'trades':len(trades),'total_return_pct':total_return_pct,'win_rate_pct':win_rate,'max_drawdown_pct':max_dd,'equity':pd.Series(equity, index=df_sig.index[:len(equity)])}

def _ensure_lock():
    if 'LIVE_LOCK' not in st.session_state:
        st.session_state['LIVE_LOCK']=threading.Lock()


def _scan_core(params: dict, last_sig_state: dict) -> dict:
    """순수 스캔 로직: streamlit API 비사용. last_sig_state 갱신하며 snapshot dict 반환."""
    rank=_markets_rank()
    if rank.empty:
        return {'table': pd.DataFrame(), 'detail': {}, 'last_sig': last_sig_state, 'notify': []}
    top_df=rank.head(int(params['topn'])).copy()
    sim_rows=[]; detail_cache={}; notifier_msgs=[]
    for _,r in top_df.iterrows():
        mk=r['market']; candles=_candles(mk, params['interval'], int(params['count']))
        if candles.empty:
            sim_rows.append({'market':mk,'trades':0,'total_return_pct':0,'win_rate_pct':0,'max_drawdown_pct':0,'last_signal':'-','price':None,'error':'no_candles'})
            continue
        try:
            raw=candles[['open','high','low','close','volume']]
            indi = flux_indicator(raw, ltf_mult=float(params['ltf_mult']), ltf_length=int(params['ltf_len']), htf_mult=float(params['htf_mult']), htf_length=int(params['htf_len']), htf_rule=params['htf_rule']) if flux_indicator else pd.DataFrame()
            df_all=raw.join(indi, how='left') if not indi.empty else raw
            bt=_simple_bt(df_all, fee=float(params['fee']))
            last_row=df_all.iloc[-1]
            if bool(last_row.get('buy_signal')): sig='BUY'
            elif bool(last_row.get('sell_signal')): sig='SELL'
            else: sig='-'
            prev=last_sig_state.get(mk)
            if sig in ('BUY','SELL') and sig!=prev:
                price_val=last_row.get('close')
                try: price_txt=f"{float(price_val):.4f}" if price_val is not None else 'NA'
                except Exception: price_txt=str(price_val)
                notifier_msgs.append(f"[LIVE] {mk} {sig} price={price_txt} ret={bt['total_return_pct']:.2f}%")
            last_sig_state[mk]=sig
            sim_rows.append({'market':mk,'trades':bt['trades'],'total_return_pct':bt['total_return_pct'],'win_rate_pct':bt['win_rate_pct'],'max_drawdown_pct':bt['max_drawdown_pct'],'last_signal':sig,'price':last_row.get('close')})
            detail_cache[mk]={'df':df_all,'bt':bt}
        except Exception as e:
            sim_rows.append({'market':mk,'trades':0,'total_return_pct':0,'win_rate_pct':0,'max_drawdown_pct':0,'last_signal':'-','price':None,'error':repr(e)})
    res_df=pd.DataFrame(sim_rows)
    if not res_df.empty:
        res_df=res_df.sort_values('total_return_pct', ascending=False)
    return {'table':res_df,'detail':detail_cache,'last_sig': last_sig_state,'notify': notifier_msgs,'last_run': time.time()}

def _scan(params: dict):
    """메인 스레드용 스캔: session_state 업데이트."""
    _ensure_lock()
    with st.session_state['LIVE_LOCK']:
        if 'LIVE_LAST_SIG' not in st.session_state: st.session_state['LIVE_LAST_SIG']={}
        snapshot=_scan_core(params, st.session_state['LIVE_LAST_SIG'])
        st.session_state['LIVE_LAST_SIG']=snapshot['last_sig']
        st.session_state['LIVE_RESULTS']={'table':snapshot['table'],'detail':snapshot['detail']}
        st.session_state['LIVE_LAST_RUN']=snapshot.get('last_run')
        notifier=get_notifier()
        if snapshot['notify'] and notifier and notifier.available():
            for m in snapshot['notify'][:20]:
                try: notifier.send_text(m)
                except Exception: pass

class _Worker:
    def __init__(self):
        self.thread=None
        self.stop_evt=threading.Event()
        self.lock=threading.Lock()
        self.last_sig_state={}
        self.snapshot=None
        self.last_error=None
        self.params={}
        self.interval=30
    def update_params(self, params: dict):
        with self.lock:
            self.params = (params or {}).copy()
    def _get_params(self):
        with self.lock:
            return self.params.copy()
    def start(self, interval: int, initial_params: dict):
        self.stop()
        self.update_params(initial_params)
        self.interval=interval
        self.stop_evt.clear()
        def loop():
            while not self.stop_evt.is_set():
                try:
                    params = self._get_params()
                    snap=_scan_core(params, self.last_sig_state)
                    notifier=get_notifier()
                    if snap['notify'] and notifier and notifier.available():
                        for m in snap['notify'][:20]:
                            try: notifier.send_text(m)
                            except Exception: pass
                    with self.lock:
                        self.snapshot={'table':snap['table'],'detail':snap['detail'],'last_run':snap.get('last_run')}
                except Exception as e:
                    self.last_error=repr(e)
                self.stop_evt.wait(self.interval)
        self.thread=threading.Thread(target=loop, daemon=True)
        self.thread.start()
    def stop(self):
        if self.thread and self.thread.is_alive():
            self.stop_evt.set(); self.thread.join(timeout=0.5)
    def get_snapshot(self):
        with self.lock:
            return self.snapshot
    def get_error(self):
        return self.last_error


def render_live():
    st.header('라이브 (리팩터링 뷰)')
    if not api:
        st.error('API 초기화 안됨')
        return
    lc1,lc2,lc3,lc4,lc5=st.columns(5)
    interval=lc1.selectbox('인터벌',['minute5','minute15','minute30','minute60'],index=0,key='live2_interval')
    count=lc2.number_input('캔들수',50,1000,300,10,key='live2_count')
    fee=lc3.number_input('수수료',0.0,0.01,0.0005,0.0001,format='%.4f',key='live2_fee')
    topn=lc4.number_input('Top N',5,50,20,1,key='live2_topn')
    auto=lc5.checkbox('자동',value=True,key='live2_auto')
    with st.expander('지표 파라미터', expanded=False):
        p1,p2,p3,p4,p5=st.columns(5)
        ltf_len=p1.number_input('LTF Len',5,400,20,1,key='live2_ltf_len')
        ltf_mult=p2.number_input('LTF Mult',0.1,10.0,2.0,0.1,key='live2_ltf_mult')
        htf_len=p3.number_input('HTF Len',5,400,20,1,key='live2_htf_len')
        htf_mult=p4.number_input('HTF Mult',0.1,10.0,2.25,0.1,key='live2_htf_mult')
        htf_rule_disp=p5.selectbox('HTF 주기',['30m','60m','120m','240m','1D'], index=0, key='live2_htf_rule')
        if htf_rule_disp.endswith('m'): htf_rule=htf_rule_disp.replace('m','T')
        else: htf_rule='1D'
    params={'interval':interval,'count':int(count),'ltf_len':int(ltf_len),'ltf_mult':float(ltf_mult),'htf_len':int(htf_len),'htf_mult':float(htf_mult),'htf_rule':htf_rule,'fee':float(fee),'topn':int(topn)}
    prev=st.session_state.get('LIVE_PARAMS') or {}
    changed=any(prev.get(k)!=v for k,v in params.items())
    st.session_state['LIVE_PARAMS']=params
    wkcol1,wkcol2,wkcol3=st.columns([1,1,2])
    worker_interval=wkcol1.number_input('워커주기(초)',5,3600,30,1,key='live2_worker_interval')
    start_btn=wkcol2.button('워커 시작' if not st.session_state.get('LIVE_WORKING') else '재시작', key='live2_start')
    stop_btn=wkcol2.button('워커 정지', key='live2_stop')
    if 'LIVE_WORKER' not in st.session_state:
        st.session_state['LIVE_WORKER']=_Worker()
    worker: _Worker = st.session_state['LIVE_WORKER']
    if start_btn:
        st.session_state['LIVE_WORKING']=True
        worker.start(worker_interval, st.session_state.get('LIVE_PARAMS'))
    if stop_btn:
        st.session_state['LIVE_WORKING']=False
        worker.stop()
    if (changed and auto) and st.session_state.get('LIVE_WORKING'):
        worker.update_params(st.session_state.get('LIVE_PARAMS'))
    if (not st.session_state.get('LIVE_WORKING')) and (changed or 'LIVE_RESULTS' not in st.session_state):
        _scan(params)
    # 워커 스냅샷 반영 (백그라운드 -> 메인)
    if st.session_state.get('LIVE_WORKING'):
        snap=worker.get_snapshot()
        if snap:
            st.session_state['LIVE_RESULTS']= {'table':snap['table'],'detail':snap['detail']}
            st.session_state['LIVE_LAST_RUN']= snap.get('last_run')
    last_run=st.session_state.get('LIVE_LAST_RUN')
    colm=st.columns([1,1,1])
    from datetime import datetime
    colm[0].metric('마지막 실행', datetime.fromtimestamp(last_run).strftime('%H:%M:%S') if last_run else '-')
    colm[1].metric('워커', 'ON' if st.session_state.get('LIVE_WORKING') else 'OFF')
    colm[2].metric('변경', 'Yes' if changed else 'No')
    res=st.session_state.get('LIVE_RESULTS') or {}
    tbl=res.get('table')
    if tbl is not None and isinstance(tbl,pd.DataFrame) and not tbl.empty:
        st.dataframe(tbl[['market','price','trades','total_return_pct','win_rate_pct','max_drawdown_pct','last_signal']], use_container_width=True, hide_index=True)
        sel=st.selectbox('상세', tbl['market'].tolist(), key='live2_detail')
        detail=res.get('detail') or {}
        if sel in detail:
            det=detail[sel]; det_df=det['df']; bt=det['bt']
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            fig=make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75,0.25], vertical_spacing=0.02)
            if {'open','high','low','close'}.issubset(det_df.columns):
                fig.add_trace(go.Candlestick(x=det_df.index, open=det_df['open'], high=det_df['high'], low=det_df['low'], close=det_df['close'], increasing_line_color='red', decreasing_line_color='blue', name='Price'), row=1,col=1)
            else:
                fig.add_trace(go.Scatter(x=det_df.index, y=det_df['close'], name='close'), row=1,col=1)
            for name,color in [('ltf_upper','rgba(255,0,0,0.6)'),('ltf_lower','rgba(0,120,255,0.6)'),('ltf_basis','rgba(200,200,200,0.6)')]:
                if name in det_df: fig.add_trace(go.Scatter(x=det_df.index, y=det_df[name], name=name), row=1,col=1)
            for name,color in [('htf_upper','rgba(255,140,0,0.9)'),('htf_lower','rgba(0,200,255,0.9)')]:
                if name in det_df: fig.add_trace(go.Scatter(x=det_df.index, y=det_df[name], name=name, line=dict(dash='dash')), row=1,col=1)
            for col,color,symbol in [('buy_signal','lime','triangle-up'),('sell_signal','magenta','triangle-down')]:
                if col in det_df:
                    hits=det_df[det_df[col]]
                    if not hits.empty:
                        fig.add_trace(go.Scatter(x=hits.index, y=hits['close'], mode='markers', name=col, marker=dict(symbol=symbol,size=10,color=color,line=dict(color='black',width=1))), row=1,col=1)
            eq=bt.get('equity') if isinstance(bt,dict) else None
            if eq is not None:
                fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name='equity', line=dict(color='orange')), row=2,col=1)
            fig.update_layout(margin=dict(l=10,r=10,t=40,b=20), height=620, xaxis_rangeslider_visible=False, legend_orientation='h')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('결과 없음(실행 필요)')
