"""Streamlit entrypoint for the Upbit dashboard."""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from ui_theme import inject_app_styles
from upbit_api import UpbitAPI

try:
    from flux_bbands_mtf_kalman import indicator as flux_indicator  # type: ignore
    from flux_bbands_mtf_kalman import indicator_with_ema as flux_indicator_with_ema  # type: ignore
except Exception:
    try:
        from src.flux_bbands_mtf_kalman import indicator as flux_indicator  # type: ignore
        from src.flux_bbands_mtf_kalman import indicator_with_ema as flux_indicator_with_ema  # type: ignore
    except Exception:
        flux_indicator = None  # type: ignore
        flux_indicator_with_ema = None  # type: ignore

from views.account_view import init_api as account_init
from views.account_view import render_account
from views.backtest_view import init_api as backtest_init
from views.backtest_view import render_backtest
from views.live_view import init_api as live_init
from views.live_view import render_live


st.set_page_config(page_title="Upbit Studio", layout="wide")
inject_app_styles()

load_dotenv()
api = UpbitAPI(
    access_key=os.getenv("UPBIT_ACCESS_KEY"),
    secret_key=os.getenv("UPBIT_SECRET_KEY"),
)

account_init(api)
backtest_init(api, flux_indicator, flux_indicator_with_ema)
live_init(api, flux_indicator, flux_indicator_with_ema)


with st.sidebar:
    st.title("Upbit Studio")
    st.caption("연구하고, 모의로 검증하고, 안전장치 아래에서만 실거래로 넘어가세요.")
    view = st.radio(
        "작업 공간",
        ["account", "backtest", "live"],
        index=["account", "backtest", "live"].index(st.session_state.get("view", "account")),
        key="view",
        format_func=lambda value: {
            "account": "내 자산",
            "backtest": "백테스트 랩",
            "live": "라이브 데스크",
        }[value],
    )
    live_flag = "ON" if os.getenv("UPBIT_LIVE") == "1" else "OFF"
    st.markdown(f"**UPBIT_LIVE:** `{live_flag}`")


main_slot = st.empty()
with main_slot.container():
    if view == "account":
        render_account()
    elif view == "backtest":
        render_backtest()
    else:
        render_live()
