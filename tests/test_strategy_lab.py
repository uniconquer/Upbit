from __future__ import annotations

import random

import pandas as pd

from src.strategy import VOLATILITY_RESET_BREAKOUT_DEFAULTS
from src.strategy_lab import (
    CampaignCandidateResult,
    CandidateResult,
    CandidateSpec,
    EvaluationWindow,
    LAB_CAPITULATION_RECLAIM_DEFAULTS,
    LAB_GUARDED_DRIFT_DEFAULTS,
    LabConfig,
    RoundSummary,
    SplitMetrics,
    best_candidate,
    build_lab_strategy_frame,
    evaluate_candidate_campaign,
    invent_candidate,
    make_offspring,
    mutate_candidate,
    rank_campaign_candidates,
    rank_candidates,
    run_evolution,
    seed_candidates,
    select_survivors,
    split_market_frames,
    _score,
)


def _sample_ohlcv() -> pd.DataFrame:
    closes = [100, 101, 102, 104, 103, 101, 99, 100, 102, 105, 107, 106, 108, 110, 112, 114]
    frame = pd.DataFrame(
        {
            "open": [99] + closes[:-1],
            "high": [value + 1 for value in closes],
            "low": [value - 1 for value in closes],
            "close": closes,
            "volume": [1000 + idx * 20 for idx, _ in enumerate(closes)],
        },
        index=pd.date_range("2026-03-01", periods=len(closes), freq="1h"),
    )
    return frame


def test_seed_candidates_contains_improve_and_invent_tracks():
    seeds = seed_candidates()

    assert any(candidate.track == "improve" and candidate.kind == "engine" for candidate in seeds)
    assert any(candidate.track == "invent" and candidate.kind == "lab" for candidate in seeds)
    assert {candidate.strategy_name for candidate in seeds} >= {
        "relative_strength_guard",
        "lab_breakout_reversion_v1",
        "lab_range_rebound_v1",
        "lab_capitulation_reclaim_v1",
        "lab_guarded_drift_v1",
        "lab_regime_switch_v1",
    }
    volatility_seed = next(candidate for candidate in seeds if candidate.strategy_name == "volatility_reset_breakout")
    assert volatility_seed.params == VOLATILITY_RESET_BREAKOUT_DEFAULTS
    capitulation_seed = next(candidate for candidate in seeds if candidate.strategy_name == "lab_capitulation_reclaim_v1")
    guarded_seed = next(candidate for candidate in seeds if candidate.strategy_name == "lab_guarded_drift_v1")
    assert capitulation_seed.params == dict(LAB_CAPITULATION_RECLAIM_DEFAULTS)
    assert guarded_seed.params == dict(LAB_GUARDED_DRIFT_DEFAULTS)


def test_mutate_candidate_preserves_family_and_advances_generation():
    parent = CandidateSpec(
        candidate_id="parent-1",
        track="improve",
        kind="engine",
        strategy_name="relative_strength_guard",
        params={"guard_fast_ema": 13, "guard_adx_floor": 10.0, "use_macd_filter": True, "spike_quantile": 0.8},
        generation=2,
    )

    child = mutate_candidate(parent, rng=random.Random(7))

    assert child.parent_id == parent.candidate_id
    assert child.generation == 3
    assert child.strategy_name == parent.strategy_name
    assert child.params["guard_fast_ema"] != parent.params["guard_fast_ema"] or child.params["guard_adx_floor"] != parent.params["guard_adx_floor"]
    assert 0.05 <= child.params["spike_quantile"] <= 0.99


def test_make_offspring_limits_family_spread():
    parents = [
        CandidateSpec("p1", "invent", "engine", "volatility_reset_breakout", params={"fast_ema": 12}),
        CandidateSpec("p2", "invent", "engine", "volatility_reset_breakout", params={"fast_ema": 14}),
    ]

    offspring = make_offspring(
        parents,
        config=LabConfig(offspring_per_parent=3, inventions_per_round=0, max_family_offspring=2, random_seed=5),
        rng=random.Random(5),
    )

    assert len(offspring) == 2
    assert all(child.strategy_name == "volatility_reset_breakout" for child in offspring)


def test_invent_candidate_creates_lab_template():
    candidate = invent_candidate(rng=random.Random(3), generation=1)

    assert candidate.kind == "lab"
    assert candidate.track == "invent"
    assert candidate.strategy_name in {
        "lab_breakout_reversion_v1",
        "lab_range_rebound_v1",
        "lab_capitulation_reclaim_v1",
        "lab_guarded_drift_v1",
        "lab_regime_switch_v1",
    }
    assert candidate.generation == 1


def test_mutate_candidate_normalizes_new_family_ranges():
    capitulation = mutate_candidate(
        CandidateSpec(
            "cap",
            "invent",
            "lab",
            "lab_capitulation_reclaim_v1",
            params={
                "fast_ema": 80,
                "slow_ema": 12,
                "rsi_len": 2,
                "oversold": -10.0,
                "bb_len": 4,
                "bb_mult": 9.0,
                "atr_window": 2,
                "atr_mult": 8.0,
                "volume_window": 2,
                "volume_spike": 0.2,
                "wick_ratio": 4.0,
                "panic_drop_pct": 0.1,
                "reclaim_bars": 1,
                "exit_rsi": 10.0,
            },
        ),
        rng=random.Random(11),
        generation=1,
    )
    guarded = mutate_candidate(
        CandidateSpec(
            "drift",
            "invent",
            "lab",
            "lab_guarded_drift_v1",
            params={
                "fast_ema": 70,
                "slow_ema": 18,
                "rsi_len": 2,
                "bb_len": 4,
                "bb_mult": 8.0,
                "adx_window": 2,
                "adx_floor": 35.0,
                "adx_ceiling": 12.0,
                "atr_window": 2,
                "atr_ceiling_pct": 0.1,
                "stop_atr_mult": 7.0,
                "volume_window": 2,
                "volume_threshold": 0.1,
                "pullback_pct": 5.0,
                "drift_window": 2,
                "rsi_ceiling": 40.0,
                "exit_rsi": 80.0,
            },
        ),
        rng=random.Random(13),
        generation=1,
    )

    assert 18.0 <= capitulation.params["oversold"] <= 40.0
    assert 0.15 <= capitulation.params["wick_ratio"] <= 0.85
    assert capitulation.params["reclaim_bars"] >= 2
    assert capitulation.params["fast_ema"] < capitulation.params["slow_ema"]
    assert 45.0 <= capitulation.params["exit_rsi"] <= 85.0

    assert guarded.params["fast_ema"] < guarded.params["slow_ema"]
    assert 8.0 <= guarded.params["adx_floor"] <= 25.0
    assert guarded.params["adx_ceiling"] >= guarded.params["adx_floor"] + 4.0
    assert 0.2 <= guarded.params["pullback_pct"] <= 2.0
    assert guarded.params["exit_rsi"] <= guarded.params["rsi_ceiling"] - 4.0


def test_build_lab_strategy_frame_returns_signals():
    frame = _sample_ohlcv()
    result = build_lab_strategy_frame(
        frame,
        strategy_name="lab_breakout_reversion_v1",
        params={
            "trend_fast_ema": 3,
            "trend_slow_ema": 6,
            "breakout_window": 3,
            "squeeze_window": 3,
            "bb_len": 4,
            "bb_mult": 1.5,
            "atr_window": 3,
            "atr_mult": 1.5,
            "volume_window": 3,
            "volume_threshold": 0.8,
        },
    )

    assert {"ema_fast", "ema_slow", "bb_lower", "buy_signal", "sell_signal", "strategy_score"}.issubset(result.columns)
    assert result["buy_signal"].dtype == bool
    assert result["sell_signal"].dtype == bool


def test_build_lab_strategy_frame_supports_new_invent_families():
    frame = _sample_ohlcv()

    capitulation = build_lab_strategy_frame(
        frame,
        strategy_name="lab_capitulation_reclaim_v1",
        params={
            "fast_ema": 3,
            "slow_ema": 6,
            "rsi_len": 4,
            "oversold": 35.0,
            "bb_len": 4,
            "bb_mult": 1.6,
            "atr_window": 3,
            "atr_mult": 1.5,
            "volume_window": 3,
            "volume_spike": 1.2,
            "wick_ratio": 0.2,
            "panic_drop_pct": 1.0,
            "reclaim_bars": 2,
            "exit_rsi": 55.0,
        },
    )
    guarded = build_lab_strategy_frame(
        frame,
        strategy_name="lab_guarded_drift_v1",
        params={
            "fast_ema": 3,
            "slow_ema": 6,
            "rsi_len": 4,
            "bb_len": 4,
            "bb_mult": 1.2,
            "adx_window": 3,
            "adx_floor": 5.0,
            "adx_ceiling": 40.0,
            "atr_window": 3,
            "atr_ceiling_pct": 5.0,
            "stop_atr_mult": 1.5,
            "volume_window": 3,
            "volume_threshold": 0.7,
            "pullback_pct": 1.5,
            "drift_window": 3,
            "rsi_ceiling": 80.0,
            "exit_rsi": 45.0,
        },
    )

    assert {"panic_event", "panic_recent", "strategy_score", "buy_signal", "sell_signal"}.issubset(capitulation.columns)
    assert {"drift_return_pct", "trend_gap", "atr_pct", "strategy_score", "buy_signal", "sell_signal"}.issubset(guarded.columns)
    assert capitulation["buy_signal"].dtype == bool
    assert guarded["sell_signal"].dtype == bool


def test_rank_candidates_orders_by_holdout_weighted_score():
    a = CandidateResult(
        candidate=CandidateSpec("a", "improve", "engine", "relative_strength_guard"),
        train=SplitMetrics(10000, 11000, 10.0, -5.0, 50.0, 10, 5, 5, 0),
        validation=SplitMetrics(10000, 11200, 12.0, -4.5, 55.0, 9, 5, 4, 0),
        holdout=SplitMetrics(10000, 12000, 20.0, -4.0, 60.0, 8, 4, 4, 0),
        holdout_weighted_score=4.0,
        overfit_gap_pct=0.0,
    )
    b = CandidateResult(
        candidate=CandidateSpec("b", "invent", "lab", "lab_range_rebound_v1"),
        train=SplitMetrics(10000, 13000, 30.0, -6.0, 55.0, 10, 5, 5, 0),
        validation=SplitMetrics(10000, 9800, -2.0, -8.0, 45.0, 10, 5, 5, 0),
        holdout=SplitMetrics(10000, 10500, 5.0, -4.0, 45.0, 8, 4, 4, 0),
        holdout_weighted_score=-2.0,
        overfit_gap_pct=25.0,
    )

    ranked = rank_candidates([b, a])

    assert ranked[0].candidate.candidate_id == "a"
    assert ranked[0].rank == 1
    assert ranked[1].rank == 2


def test_score_penalizes_negative_holdout_and_large_holdout_drawdown():
    config = LabConfig()
    strong_train_bad_holdout = _score(
        SplitMetrics(10000, 15000, 50.0, -10.0, 55.0, 40, 20, 20, 0),
        SplitMetrics(10000, 9800, -2.0, -18.0, 50.0, 30, 15, 15, 0),
        SplitMetrics(10000, 9200, -8.0, -28.0, 45.0, 32, 16, 16, 0),
        config,
    )[0]
    modest_train_positive_holdout = _score(
        SplitMetrics(10000, 10600, 6.0, -8.0, 52.0, 34, 17, 17, 0),
        SplitMetrics(10000, 10400, 4.0, -9.0, 50.0, 24, 12, 12, 0),
        SplitMetrics(10000, 10300, 3.0, -9.0, 50.0, 24, 12, 12, 0),
        config,
    )[0]

    assert modest_train_positive_holdout > strong_train_bad_holdout


def test_split_market_frames_supports_validation_window():
    raw = {"KRW-A": _sample_ohlcv()}

    train, validation, holdout = split_market_frames(
        raw,
        train_start=pd.Timestamp("2026-03-01 00:00:00"),
        train_end=pd.Timestamp("2026-03-01 06:00:00"),
        validation_end=pd.Timestamp("2026-03-01 11:00:00"),
        holdout_end=pd.Timestamp("2026-03-01 16:00:00"),
    )

    assert len(train["KRW-A"]) == 6
    assert len(validation["KRW-A"]) == 5
    assert len(holdout["KRW-A"]) == 5


def test_select_survivors_prefers_family_diversity_when_possible():
    ranked = [
        CandidateResult(
            candidate=CandidateSpec("vol-1", "invent", "engine", "volatility_reset_breakout"),
            train=SplitMetrics(10000, 11200, 12.0, -6.0, 50.0, 10, 5, 5, 0),
            validation=SplitMetrics(10000, 11400, 14.0, -5.0, 55.0, 9, 5, 4, 0),
            holdout=SplitMetrics(10000, 11100, 11.0, -4.0, 60.0, 8, 4, 4, 0),
            holdout_weighted_score=7.0,
            overfit_gap_pct=1.0,
            rank=1,
        ),
        CandidateResult(
            candidate=CandidateSpec("vol-2", "invent", "engine", "volatility_reset_breakout"),
            train=SplitMetrics(10000, 11000, 10.0, -6.0, 50.0, 10, 5, 5, 0),
            validation=SplitMetrics(10000, 11100, 11.0, -5.0, 55.0, 9, 5, 4, 0),
            holdout=SplitMetrics(10000, 10900, 9.0, -4.5, 60.0, 8, 4, 4, 0),
            holdout_weighted_score=6.0,
            overfit_gap_pct=1.0,
            rank=2,
        ),
        CandidateResult(
            candidate=CandidateSpec("sq-1", "improve", "engine", "squeeze_breakout"),
            train=SplitMetrics(10000, 10800, 8.0, -7.0, 50.0, 10, 5, 5, 0),
            validation=SplitMetrics(10000, 10900, 9.0, -6.0, 55.0, 9, 5, 4, 0),
            holdout=SplitMetrics(10000, 10700, 7.0, -5.0, 60.0, 8, 4, 4, 0),
            holdout_weighted_score=5.0,
            overfit_gap_pct=1.0,
            rank=3,
        ),
        CandidateResult(
            candidate=CandidateSpec("range-1", "invent", "lab", "lab_range_rebound_v1"),
            train=SplitMetrics(10000, 10400, 4.0, -8.0, 45.0, 10, 5, 5, 0),
            validation=SplitMetrics(10000, 10500, 5.0, -7.0, 48.0, 9, 5, 4, 0),
            holdout=SplitMetrics(10000, 10300, 3.0, -6.0, 50.0, 8, 4, 4, 0),
            holdout_weighted_score=4.0,
            overfit_gap_pct=1.0,
            rank=4,
        ),
    ]

    survivors = select_survivors(ranked, config=LabConfig(parents_per_track=1, max_family_survivors=1))
    survivor_families = {candidate.strategy_name for candidate in survivors}

    assert "volatility_reset_breakout" in survivor_families
    assert len(survivor_families) >= 2


def test_evaluate_candidate_campaign_aggregates_and_ranks_windows(monkeypatch):
    raw = {"KRW-A": _sample_ohlcv()}
    windows = [
        EvaluationWindow(
            name="w1",
            train_start=pd.Timestamp("2026-03-01 00:00:00"),
            train_end=pd.Timestamp("2026-03-01 06:00:00"),
            validation_end=pd.Timestamp("2026-03-01 10:00:00"),
            holdout_end=pd.Timestamp("2026-03-01 16:00:00"),
        ),
        EvaluationWindow(
            name="w2",
            train_start=pd.Timestamp("2026-03-01 00:00:00"),
            train_end=pd.Timestamp("2026-03-01 08:00:00"),
            validation_end=pd.Timestamp("2026-03-01 12:00:00"),
            holdout_end=pd.Timestamp("2026-03-01 16:00:00"),
        ),
    ]

    def fake_evaluate(candidate, *_args, train_end, **_kwargs):
        if candidate.strategy_name == "volatility_reset_breakout":
            if train_end == pd.Timestamp("2026-03-01 06:00:00"):
                return CandidateResult(
                    candidate,
                    SplitMetrics(10000, 10800, 8.0, -5.0, 50.0, 10, 5, 5, 0),
                    SplitMetrics(10000, 11200, 12.0, -6.0, 55.0, 9, 5, 4, 0),
                    SplitMetrics(10000, 11100, 11.0, -4.0, 60.0, 8, 4, 4, 0),
                    6.0,
                    0.0,
                )
            return CandidateResult(
                candidate,
                SplitMetrics(10000, 10600, 6.0, -4.0, 50.0, 10, 5, 5, 0),
                SplitMetrics(10000, 10900, 9.0, -7.0, 55.0, 9, 5, 4, 0),
                SplitMetrics(10000, 10700, 7.0, -5.0, 60.0, 8, 4, 4, 0),
                3.0,
                0.0,
            )
        return CandidateResult(
            candidate,
            SplitMetrics(10000, 10200, 2.0, -8.0, 45.0, 10, 5, 5, 0),
            SplitMetrics(10000, 9900, -1.0, -10.0, 48.0, 9, 5, 4, 0),
            SplitMetrics(10000, 9800, -2.0, -9.0, 40.0, 8, 4, 4, 0),
            -4.0,
            0.0,
        )

    monkeypatch.setattr("src.strategy_lab.evaluate_candidate", fake_evaluate)

    strong = evaluate_candidate_campaign(
        CandidateSpec("a", "invent", "engine", "volatility_reset_breakout"),
        raw,
        windows=windows,
        config=LabConfig(),
    )
    weak = evaluate_candidate_campaign(
        CandidateSpec("b", "invent", "lab", "lab_range_rebound_v1"),
        raw,
        windows=windows,
        config=LabConfig(),
    )

    ranked = rank_campaign_candidates([weak, strong])

    assert round(strong.avg_holdout_return_pct, 2) == 9.0
    assert strong.min_holdout_return_pct == 7.0
    assert strong.worst_holdout_drawdown_pct == -5.0
    assert ranked[0].candidate.strategy_name == "volatility_reset_breakout"
    assert ranked[0].rank == 1


def test_evaluate_candidate_campaign_penalizes_inactive_holdout_windows(monkeypatch):
    raw = {"KRW-A": _sample_ohlcv()}
    windows = [
        EvaluationWindow(
            name="w1",
            train_start=pd.Timestamp("2026-03-01 00:00:00"),
            train_end=pd.Timestamp("2026-03-01 06:00:00"),
            validation_end=pd.Timestamp("2026-03-01 10:00:00"),
            holdout_end=pd.Timestamp("2026-03-01 16:00:00"),
        ),
        EvaluationWindow(
            name="w2",
            train_start=pd.Timestamp("2026-03-01 00:00:00"),
            train_end=pd.Timestamp("2026-03-01 08:00:00"),
            validation_end=pd.Timestamp("2026-03-01 12:00:00"),
            holdout_end=pd.Timestamp("2026-03-01 16:00:00"),
        ),
    ]

    def fake_evaluate(candidate, *_args, **_kwargs):
        holdout_trades = 0 if candidate.strategy_name == "inactive_family" else 3
        return CandidateResult(
            candidate,
            SplitMetrics(10000, 10200, 2.0, -2.0, 50.0, 4, 2, 2, 0),
            SplitMetrics(10000, 10300, 3.0, -2.5, 55.0, 4, 2, 2, 0),
            SplitMetrics(10000, 10100, 1.0, -1.0, 60.0, holdout_trades, holdout_trades // 2, holdout_trades // 2, 0),
            1.5,
            0.0,
        )

    monkeypatch.setattr("src.strategy_lab.evaluate_candidate", fake_evaluate)

    inactive = evaluate_candidate_campaign(
        CandidateSpec("inactive", "invent", "lab", "inactive_family"),
        raw,
        windows=windows,
        config=LabConfig(),
    )
    active = evaluate_candidate_campaign(
        CandidateSpec("active", "invent", "lab", "active_family"),
        raw,
        windows=windows,
        config=LabConfig(),
    )

    ranked = rank_campaign_candidates([inactive, active])

    assert inactive.avg_holdout_trades == 0.0
    assert inactive.min_holdout_trades == 0
    assert active.avg_holdout_trades == 3.0
    assert active.campaign_score > inactive.campaign_score
    assert ranked[0].candidate.strategy_name == "active_family"


def test_run_evolution_uses_fake_evaluator_and_returns_leaderboards(monkeypatch):
    raw = {
        "KRW-A": _sample_ohlcv(),
        "KRW-B": _sample_ohlcv().assign(close=lambda df: df["close"] + 2),
    }

    def fake_evaluate(candidate, *args, **kwargs):
        base = {
            "relative_strength_guard": 8.0,
            "lab_breakout_reversion_v1": 4.0,
            "lab_range_rebound_v1": 2.0,
            "lab_regime_switch_v1": 6.0,
            "relative_strength_rotation": 5.0,
            "regime_blend_guard": 3.0,
            "squeeze_breakout": 1.0,
            "regime_blend": 0.5,
        }.get(candidate.strategy_name, 0.0)
        train = SplitMetrics(10000, 10000 + base * 100, base * 2, -5.0, 50.0, 10, 5, 5, 0)
        validation = SplitMetrics(10000, 10000 + base * 120, base * 2.5, -4.5, 55.0, 9, 5, 4, 0)
        holdout = SplitMetrics(10000, 10000 + base * 150, base * 3, -4.0, 60.0, 8, 4, 4, 0)
        return CandidateResult(candidate, train, validation, holdout, base, 0.0)

    monkeypatch.setattr("src.strategy_lab.evaluate_candidate", fake_evaluate)

    history = run_evolution(
        raw,
        train_start=pd.Timestamp("2026-03-01 00:00:00"),
        train_end=pd.Timestamp("2026-03-01 06:00:00"),
        validation_end=pd.Timestamp("2026-03-01 10:00:00"),
        holdout_end=pd.Timestamp("2026-03-01 16:00:00"),
        rounds=2,
        config=LabConfig(offspring_per_parent=1, parents_per_track=1, random_seed=11),
    )

    assert len(history) == 2
    assert isinstance(history[0], RoundSummary)
    assert not history[0].leaderboard.empty
    assert history[0].leaderboard.iloc[0]["strategy_name"] == "relative_strength_guard"
    assert best_candidate(history).candidate.strategy_name == "relative_strength_guard"
    assert history[0].survivors
