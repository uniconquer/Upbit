import os
import argparse
import statistics
from dotenv import load_dotenv
from rich import print
from upbit_api import UpbitAPI


def cmd_markets(api: UpbitAPI, base: str = "KRW", limit: int = 5):
    all_markets = api.markets()
    base = base.upper()
    filtered = [d for d in all_markets if isinstance(d.get("market"), str) and d["market"].startswith(f"{base}-")]
    if not filtered:
        print({"message": f"no markets for base {base}"})
        return
    markets = [d["market"] for d in filtered]
    name_map = {d["market"]: d.get("korean_name") for d in filtered}
    tks = api.tickers(markets)
    tks_sorted = sorted(tks, key=lambda x: float(x.get('acc_trade_price_24h') or 0), reverse=True)
    for t in tks_sorted[:limit]:
        print({
            'market': t.get('market'),
            'name_ko': name_map.get(t.get('market')),
            'trade_price': t.get('trade_price'),
            'acc_trade_price_24h': t.get('acc_trade_price_24h')
        })


def cmd_ticker(api: UpbitAPI, market: str):
    print(api.ticker(market))


def _bollinger(prices: list[float], period: int = 20, mult: float = 2.0):
    if len(prices) < period:
        return []
    out = []
    for i in range(period - 1, len(prices)):
        win = prices[i - period + 1: i + 1]
        mid = statistics.fmean(win)
        if len(win) > 1:
            var = statistics.fmean([(p - mid) ** 2 for p in win])
            sd = var ** 0.5
        else:
            sd = 0.0
        out.append({'index': i, 'mid': mid, 'upper': mid + mult * sd, 'lower': mid - mult * sd, 'close': prices[i]})
    return out


def cmd_bollinger(api: UpbitAPI, market: str, interval: str, period: int, mult: float, count: int):
    candles = api.candles(market, interval=interval, count=max(count, period + 5))
    closes = [c.close for c in candles][-count:]
    bands = _bollinger(closes, period=period, mult=mult)
    print({
        'market': market,
        'interval': interval,
        'period': period,
        'mult': mult,
        'last': bands[-1] if bands else None
    })


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description='Upbit simplified CLI (markets, ticker, bollinger)')
    sub = parser.add_subparsers(dest='cmd', required=True)

    p1 = sub.add_parser('markets')
    p1.add_argument('--base', default='KRW')
    p1.add_argument('--limit', type=int, default=5)

    p2 = sub.add_parser('ticker')
    p2.add_argument('--market', required=True)

    p3 = sub.add_parser('bollinger')
    p3.add_argument('--market', required=True)
    p3.add_argument('--interval', default='day')
    p3.add_argument('--period', type=int, default=20)
    p3.add_argument('--mult', type=float, default=2.0)
    p3.add_argument('--count', type=int, default=120)

    sub.add_parser('accounts')

    p5 = sub.add_parser('order')
    p5.add_argument('--market', required=True)
    p5.add_argument('--side', choices=['bid', 'ask'], required=True)
    p5.add_argument('--ord-type', choices=['limit', 'price', 'market'], required=True)
    p5.add_argument('--volume')
    p5.add_argument('--price')
    p5.add_argument('--simulate', action='store_true')

    args = parser.parse_args()
    api = UpbitAPI(access_key=os.getenv('UPBIT_ACCESS_KEY'), secret_key=os.getenv('UPBIT_SECRET_KEY'))

    if args.cmd == 'markets':
        cmd_markets(api, args.base, args.limit)
    elif args.cmd == 'ticker':
        cmd_ticker(api, args.market)
    elif args.cmd == 'bollinger':
        cmd_bollinger(api, args.market, args.interval, args.period, args.mult, args.count)
    elif args.cmd == 'accounts':
        try:
            for a in api.accounts():
                print(a)
        except Exception as e:
            print({'error': str(e)})
    elif args.cmd == 'order':
        print(api.create_order(args.market, args.side, args.ord_type, volume=args.volume, price=args.price, simulate=args.simulate))


if __name__ == '__main__':
    main()
