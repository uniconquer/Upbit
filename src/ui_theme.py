"""Reusable Streamlit and Plotly styling helpers."""

from __future__ import annotations

import streamlit as st


APP_STYLE = """
<style>
:root {
  --bg-0: #06111d;
  --bg-1: #0b1f33;
  --bg-2: rgba(8, 24, 39, 0.84);
  --panel: rgba(9, 27, 43, 0.78);
  --panel-strong: rgba(10, 34, 54, 0.92);
  --line: rgba(125, 211, 252, 0.18);
  --text: #e2f3ff;
  --muted: #96b9d0;
  --accent: #2dd4bf;
  --accent-2: #f59e0b;
  --danger: #fb7185;
}

.stApp {
  background:
    radial-gradient(circle at 12% 18%, rgba(45, 212, 191, 0.12), transparent 28%),
    radial-gradient(circle at 88% 16%, rgba(245, 158, 11, 0.12), transparent 26%),
    linear-gradient(160deg, var(--bg-0) 0%, #091423 35%, #10263d 100%);
  color: var(--text);
  font-family: "Segoe UI Variable", "Segoe UI", "Malgun Gothic", sans-serif;
}

section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(6, 17, 29, 0.96), rgba(7, 26, 42, 0.98));
  border-right: 1px solid var(--line);
}

div[data-testid="stMetric"] {
  background: linear-gradient(180deg, var(--panel), rgba(7, 20, 33, 0.72));
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 0.7rem 0.8rem;
  box-shadow: 0 18px 40px rgba(0, 0, 0, 0.22);
}

div[data-testid="stDataFrame"],
div[data-testid="stPlotlyChart"],
div[data-testid="stExpander"] {
  border-radius: 20px;
  overflow: hidden;
}

div[data-testid="stDataFrame"] {
  border: 1px solid var(--line);
  background: rgba(5, 19, 31, 0.68);
}

.stButton > button, .stDownloadButton > button {
  border-radius: 14px;
  border: 1px solid rgba(45, 212, 191, 0.32);
  background: linear-gradient(180deg, rgba(14, 40, 62, 0.95), rgba(9, 25, 40, 0.95));
  color: var(--text);
  font-weight: 600;
}

.stButton > button:hover {
  border-color: rgba(45, 212, 191, 0.65);
  color: white;
}

.upbit-hero {
  padding: 1.15rem 1.25rem;
  border-radius: 24px;
  border: 1px solid var(--line);
  background:
    linear-gradient(135deg, rgba(5, 19, 31, 0.92), rgba(13, 39, 63, 0.88)),
    radial-gradient(circle at top right, rgba(45, 212, 191, 0.12), transparent 30%);
  box-shadow: 0 18px 40px rgba(0, 0, 0, 0.25);
  margin-bottom: 1rem;
}

.upbit-eyebrow {
  color: var(--accent);
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  margin-bottom: 0.35rem;
}

.upbit-hero h1 {
  font-size: 2rem;
  margin: 0;
  color: white;
}

.upbit-hero p {
  color: var(--muted);
  margin: 0.55rem 0 0;
  max-width: 72ch;
}
</style>
"""


def inject_app_styles() -> None:
    # Streamlit reruns can rebuild the page tree, so inject the theme stylesheet
    # on every run instead of caching it in session state.
    st.markdown(APP_STYLE, unsafe_allow_html=True)


def page_intro(eyebrow: str, title: str, body: str) -> None:
    st.markdown(
        (
            "<div class='upbit-hero'>"
            f"<div class='upbit-eyebrow'>{eyebrow}</div>"
            f"<h1>{title}</h1>"
            f"<p>{body}</p>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def apply_chart_theme(fig, *, height: int = 720):
    fig.update_layout(
        paper_bgcolor="rgba(6, 17, 29, 0.0)",
        plot_bgcolor="rgba(8, 24, 39, 0.72)",
        font={"color": "#dbeafe", "family": "Segoe UI Variable, Segoe UI, Malgun Gothic, sans-serif"},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "bgcolor": "rgba(6, 17, 29, 0.0)",
        },
        margin={"l": 18, "r": 18, "t": 44, "b": 18},
        height=height,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="rgba(148, 163, 184, 0.10)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(148, 163, 184, 0.10)", zeroline=False)
    return fig
