"""Research results and explicit paper observations, separate from live controls."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.portfolio_paper import observe_once
from src.paper_learning import run_once as learning_once
from src.minute_learning import observe_once as minute_once


ROOT = Path(__file__).resolve().parents[2]


def render_learning():
    st.write('자동 개선 실험')
    timeframe = st.selectbox('실험 주기', ['1분봉', '일봉 · 이전 기록'], key='learning_timeframe')
    minute = timeframe == '1분봉'
    folder = ROOT / '.runtime' / ('minute-learning' if minute else 'paper-learning')
    st.caption(f'{"1분봉" if minute else "일봉"} 전략 후보를 같은 시세로 관찰합니다. 30일 이상 새 성과가 쌓이면 '
               '비용·낙폭·체결 수를 평가하고 다음 세대의 설정을 생성합니다. '
               '대표 후보 선정은 이 모의 실험 안에서만 적용됩니다.')
    if st.button('학습 관찰 1회', key='learning_once'):
        try:
            with st.spinner('후보들을 같은 시세로 평가합니다…'):
                result = minute_once(folder) if minute else learning_once(folder)
            st.success('이미 이번 주기에 관찰했습니다.' if result['status'] == 'already_observed' else '관찰을 저장했습니다.')
        except Exception as exc:
            st.error(f'학습 관찰 실패: {exc}')
    path = folder / 'state.json'
    worker_path = folder / 'worker.json'
    if minute and worker_path.exists():
        worker = json.loads(worker_path.read_text(encoding='utf-8'))
        age = (pd.Timestamp.now(tz='UTC')-pd.Timestamp(worker['checked_at'])).total_seconds()
        status = '관찰 지연' if age > 180 else worker.get('status')
        st.caption(f"1분봉 관찰 프로세스: {status} · 최근 확인 {worker.get('checked_at')}")
    minute_report = ROOT / '.runtime' / 'minute-research-v1' / 'report.json'
    structural_report = ROOT / '.runtime' / 'strategy-research' / 'latest.json'
    if minute and structural_report.exists():
        result = json.loads(structural_report.read_text(encoding='utf-8'))
        st.write('매일 전략 구조 연구')
        st.caption('다음 연구는 이전 학습 점수를 참고해 설정을 변형하고 새 조합도 탐색합니다. '
                   '기존 비교 기준을 포함해 하루 24개를 평가하며 투자 한도를 자동으로 늘리지 않습니다.')
        st.caption(f"연구일 {result['run_id']} UTC · {len(result['training_ranking'])}개 조합 · "
                   '상위 봉 추세, 비용 대비 변동 폭, 청산 확인을 비교합니다.')
        st.write(f"선택 후보 검증 수익률 {result['test']['return_pct']:+.2f}% · "
                 f"높은 비용 {result['stress']['return_pct']:+.2f}% · "
                 f"완료 매도 {result['test']['closed_trades']}건 · 다음 세대 추천 {len(result['nominees'])}개")
        st.caption('거래가 없어서 0%인 결과는 수익성 입증이 아닙니다. 과거 연구는 새 모의 관찰 실적으로 합산하지 않습니다.')
    if minute and minute_report.exists():
        result = json.loads(minute_report.read_text(encoding='utf-8'))
        cols = st.columns(3)
        cols[0].metric('1분봉 후보 평가 수익률', f"{result['candidate_test']['return_pct']:+.2f}%")
        cols[1].metric('1분봉 완료 매도', result['candidate_test']['closed_trades'])
        cols[2].metric('1분봉 높은 비용 수익률', f"{result['candidate_stress']['return_pct']:+.2f}%")
        st.caption('과거 1분봉 재생 결과이며 현재 시세 관찰 실적과 별개입니다.')
        if result['selection_favors_cash']:
            st.info('첫 과거 실험에서는 모든 후보의 선택 점수가 불리해 현금 보유를 택했습니다.')
    if not path.exists():
        st.info('첫 관찰을 실행하면 후보별 기록이 생성됩니다.')
        return
    try:
        state = json.loads(path.read_text(encoding='utf-8'))
        cols = st.columns(4)
        cols[0].metric('세대', state['generation'])
        cols[1].metric('실험 후보', len(state['candidates']))
        cols[2].metric('현재 세대 관찰', state['observations'])
        cols[3].metric('대표 후보', state['incumbent'] or '현금 / 검증 대기')
        st.caption(f"최근 관찰: {state['last_observation']} · "
                   f"현재 세대 관찰률 {state.get('coverage', 0)*100:.1f}% · "
                   '기본 비용과 높은 비용을 각각 적용한 독립 가상계좌입니다.')
        if state['rankings']:
            rows = pd.DataFrame(state['rankings'])
            configs = {c['id']: c['config'] for c in state['candidates']}
            rows['최대 투자 비중(%)'] = rows.id.map(lambda key: configs[key]['exposure_fraction']*100)
            rows = rows.rename(columns={'id': '후보', 'equity': '가상 평가액', 'return_pct': '수익률(%)',
                                        'stress_return_pct': '높은 비용 수익률(%)', 'drawdown_pct': '최대낙폭(%)',
                                        'closed_trades': '완료 매도'})
            st.dataframe(rows[['후보', '가상 평가액', '수익률(%)', '높은 비용 수익률(%)',
                               '최대낙폭(%)', '완료 매도', '최대 투자 비중(%)']], hide_index=True)
        if state['promotions']:
            with st.expander('대표 후보 변경 기록'):
                st.dataframe(pd.DataFrame(state['promotions']), hide_index=True)
    except (OSError, ValueError, KeyError) as exc:
        st.error(f'학습 기록을 읽을 수 없습니다: {exc}')


def render_portfolio():
    st.subheader('적응형 포트폴리오 · 모의매매')
    st.caption('가상자금 10만 원으로 검증합니다. 실제 계좌나 주문과 연결되지 않습니다.')
    render_learning()
    st.subheader('이전 일봉 포트폴리오 연구')
    folder = ROOT / '.runtime' / 'portfolio-research-v2'
    report_path = folder / 'report.json'
    state_path = ROOT / '.runtime' / 'adaptive-paper-v2' / 'state.json'
    if not report_path.exists():
        st.info('아직 연구 결과가 없습니다. 저장소 폴더에서 아래 명령을 실행하세요.')
        st.code('py -3 -m src.portfolio_research --expanded --output .runtime/portfolio-research-v2')
        return
    try:
        report = json.loads(report_path.read_text(encoding='utf-8'))
        profile_path = folder / 'paper-profile.json'
        profile = json.loads(profile_path.read_text(encoding='utf-8')) if profile_path.exists() else None
    except (OSError, ValueError) as exc:
        st.error(f'연구 보고서를 읽을 수 없습니다: {exc}')
        return
    if report['status'] == 'RESEARCH_ONLY':
        st.warning('연구용 후보입니다. 현재 결과는 실전 투입 기준을 충족하지 못했습니다.')
    else:
        st.info('미래 모의매매 관찰 후보입니다. 실제 주문 승인을 뜻하지 않습니다.')
    development = report['walk_forward']
    final = report['final']
    cols = st.columns(4)
    cols[0].metric('개발 구간 순차 수익률', f"{development['return_pct']:+.2f}%")
    cols[1].metric('마지막 구간 수익률', f"{final['return_pct']:+.2f}%")
    cols[2].metric('마지막 구간 최대낙폭', f"{final['max_drawdown_pct']:.2f}%")
    cols[3].metric('비용 스트레스 수익률', f"{report['stress']['return_pct']:+.2f}%")
    st.caption(f"마지막 평가: {report['final_start'][:10]} ~ {report['final_end'][:10]} UTC. "
               '이미 관찰한 과거 기간이며, 미래 수익을 보장하지 않습니다.')
    with st.expander('전략과 손실 관리', expanded=True):
        st.write('비트코인의 장기 추세가 상승할 때 강한 종목을 고릅니다. '
                 '돌파·추세 진입·눌림목 후보는 과거 구간에서 비교하고 다음 구간에 적용합니다.')
        st.write('거래당 위험 예산 0.75%, 종목당 최대 40%, 전체 투자 최대 60%, 최대 2종목입니다. '
                 '변동성에 따라 금액을 줄이고 손절선을 올립니다. '
                 '일 손실 2% 또는 자산 최고점 대비 10% 하락 시 신규 매수를 중단합니다. '
                 '급락·체결 지연 때문에 실제 손실은 이 수치를 넘을 수 있습니다.')
        config = profile.get('selected_config') if profile else None
        if config:
            st.caption(f"현재 후보: {config['entry_mode']} · 추세 {config['trend_window']}일 · "
                       f"돌파 {config['breakout_window']}일 · 손절 {config['stop_atr']} ATR")
        else:
            st.caption('현재 선택은 현금 보유입니다.')
    chart_path = folder / 'walk-forward-equity.csv'
    if chart_path.exists():
        chart = pd.read_csv(chart_path, index_col='timestamp', parse_dates=True)
        st.write('개발 구간의 순차 모의매매 자산 추이')
        st.line_chart(chart)
    st.write('현재 시세 모의매매')
    st.caption('아래 버튼을 누를 때만 시세를 관찰합니다. 화면을 닫으면 지속 감시하지 않습니다. '
               '실시간 기록은 과거 재생 결과와 별도로 저장하며, 매매 조건이 없으면 현금을 유지합니다.')
    if profile is None:
        st.info('최신 데이터로 선택한 모의매매 프로필이 필요합니다. 연구 도구를 실행해 생성하세요.')
    else:
        st.caption(f"모의매매 프로필 선택 기준일: {profile.get('selection_asof', '')}")
    if st.button('현재 시세로 모의매매 1회', key='portfolio_observe', disabled=profile is None):
        try:
            with st.spinner('완료된 일봉과 현재 시세를 확인합니다…'):
                observation = observe_once(profile, state_path)
            st.success(f"관찰 완료 · 새 모의 체결 {observation['new_trades']}건")
        except Exception as exc:
            st.error(f'모의 관찰을 저장하지 못했습니다: {exc}')
    if state_path.exists():
        try:
            saved = json.loads(state_path.read_text(encoding='utf-8'))
            sim = saved['simulation']
            cols = st.columns(3)
            cols[0].metric('가상자산 평가액', f"{sim['last_equity']:,.0f}원")
            cols[1].metric('가상 현금', f"{sim['cash']:,.0f}원")
            cols[2].metric('보유 종목', len(sim['positions']))
            st.caption(f"마지막 관찰: {saved['observed_at']} · 누적 {saved['observations']}회")
            if sim['halted'] or sim['daily_halted']:
                st.warning('손실 제한으로 신규 매수가 중단되었습니다. 청산 판단은 계속합니다.')
            if sim['blocked_exits']:
                st.warning('최소 주문금액 미만으로 청산하지 못한 모의 포지션이 있습니다.')
            if sim['events']:
                events = pd.DataFrame(sim['events'])
                events['시각'] = pd.to_datetime(events.ts, unit='s', utc=True).dt.tz_convert('Asia/Seoul')
                st.dataframe(events[['시각', 'market', 'side', 'price', 'qty']].tail(30), hide_index=True)
            else:
                st.info('아직 모의 체결이 없습니다.')
        except (OSError, ValueError, KeyError) as exc:
            st.error(f'모의 기록을 읽을 수 없습니다: {exc}')
