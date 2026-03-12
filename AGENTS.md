# Upbit Repo Guide

## Purpose
- Evolve this repository into a safer and more testable Upbit auto-trading system.
- Prefer incremental upgrades that keep backtest behavior, simulation behavior, and live behavior aligned.

## Default Working Mode
- Assume the team currently works directly on `main` unless the user asks for a separate branch.
- Respect existing dirty changes. Do not revert user work unless explicitly asked.
- Never print secret values from `.env`. Mention only key names or redact values.
- Treat real orders as high risk. Do not enable `UPBIT_LIVE=1` or weaken live-confirmation flows unless the user explicitly asks.

## Environment Variables
- Supported keys are defined by `.env.example` and current code paths:
  - `UPBIT_ACCESS_KEY`
  - `UPBIT_SECRET_KEY`
  - `UPBIT_LIVE`
  - `TELEGRAM_BOT_TOKEN`
  - `TELEGRAM_CHAT_ID`
- The code treats live mode as enabled only when `UPBIT_LIVE` is exactly `1`.
- Read `.env` only when needed for the task. Prefer existence checks, redacted output, or key-name checks over dumping contents.

## Repo Map
- `src/app_streamlit.py`: Streamlit entrypoint and view wiring.
- `src/views/backtest_view.py`: indicator exploration and quick backtest UI.
- `src/views/live_view.py`: live scan UI, risk limits, worker loop, and session-state runtime.
- `src/mr_worker.py`: CLI strategy monitor that shares the refactored strategy, risk, and paper-trading stack.
- `src/upbit_api.py`: Upbit REST/WebSocket client and order submission gate.
- `src/notifier.py`: Telegram notification adapter.
- `src/main.py`: small CLI for markets, ticker, bollinger, accounts, and order smoke checks.
- `tests/`: add or update tests whenever strategy, risk, or order behavior changes.

## Standard Workflow
1. Inspect `git status`, touched files, and the relevant entrypoints before editing.
2. For strategy or risk changes, read both UI and worker paths so signal logic does not drift between `backtest_view.py`, `live_view.py`, `mr_worker.py`, and `upbit_api.py`.
3. Keep changes small and reversible. Extract pure functions when logic is hard to test in-place.
4. Validate with the narrowest useful command first, then broaden only if needed.
5. Report behavior changes, validation run, and remaining risk clearly.

## Safety Rules
- Backtest first, simulation second, live last.
- If a change affects order placement, capital allocation, exits, or notification timing, add or update tests before touching live paths.
- Keep real order execution gated by both `UPBIT_LIVE=1` and an explicit CLI/UI live toggle.
- Prefer `simulate=True` or non-live flows by default.

## Commands
- Install deps: `pip install -r requirements.txt`
- Run tests: `pytest`
- Run Streamlit: `python -m streamlit run .\src\app_streamlit.py --server.headless true`
- CLI smoke: `python .\src\main.py markets --limit 10`
- Notification smoke: `python .\src\send_test_notification.py`

## Codex Skills
- Use `$upbit-dev-routine` for general work in this repository.
- Use `$upbit-strategy-safety` for strategy, risk, notification, and live-trading changes.
