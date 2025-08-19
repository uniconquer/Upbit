from src.strategy import sma_cross_signals, rsi_signals

def test_no_signals_if_short_ge_long():
    try:
        sma_cross_signals([1,2,3,4,5,6], short=5, long=5)
    except ValueError:
        return
    assert False, "expected ValueError when long == short"


def test_basic_signals_generation():
    prices = [1,1,1,1,1, 2,2,2,2,2, 1,1,1,1,1, 2,2,2,2,2]
    sigs = sma_cross_signals(prices, short=3, long=5)
    assert isinstance(sigs, list)


def test_rsi_no_signals_small_series():
    prices = [1,2,3,4,5]  # shorter than period+2
    sigs = rsi_signals(prices, period=14)
    assert sigs == []


def test_rsi_basic_generation():
    # Construct oscillating prices to trigger oversold/overbought crosses
    prices = [50 + ((-1)**i)*i for i in range(1,80)]
    sigs = rsi_signals(prices, period=6, oversold=30, overbought=70)
    assert isinstance(sigs, list)
