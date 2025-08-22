import time
import hmac
import hashlib
import uuid
import requests
from typing import Any, Dict, Optional
from urllib.parse import urlencode
from pydantic import BaseModel
import json
import threading
from websocket import create_connection, WebSocket
import jwt

UPBIT_API_BASE = "https://api.upbit.com/v1"

class Candle(BaseModel):
    market: str
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float

class UpbitAPI:
    def __init__(self, access_key: Optional[str] = None, secret_key: Optional[str] = None):
        self.access_key = access_key
        self.secret_key = secret_key
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
        })

    # ---------- Public APIs ----------
    def markets(self) -> list[Dict[str, Any]]:
        r = self.session.get(f"{UPBIT_API_BASE}/market/all", params={"isDetails": "false"}, timeout=10)
        r.raise_for_status()
        return r.json()

    def ticker(self, market: str) -> Dict[str, Any]:
        r = self.session.get(f"{UPBIT_API_BASE}/ticker", params={"markets": market}, timeout=10)
        r.raise_for_status()
        data = r.json()[0]
        return data

    def tickers(self, markets: list[str]) -> list[Dict[str, Any]]:
        if not markets:
            return []
        # Upbit allows multiple markets in a single call; to be safe, chunk by 100
        out: list[Dict[str, Any]] = []
        chunk_size = 100
        for i in range(0, len(markets), chunk_size):
            chunk = markets[i:i+chunk_size]
            r = self.session.get(f"{UPBIT_API_BASE}/ticker", params={"markets": ",".join(chunk)}, timeout=15)
            r.raise_for_status()
            out.extend(r.json())
        return out

    def candles(self, market: str, interval: str = "day", count: int = 200, to: Optional[str] = None) -> list[Candle]:
        # interval: minute1|minute3|minute5|minute10|minute15|minute30|minute60|minute240|day|week|month
        endpoint = {
            "day": "candles/days",
            "week": "candles/weeks",
            "month": "candles/months",
        }.get(interval, None)
        if endpoint is None and interval.startswith("minute"):
            unit = interval.replace("minute", "")
            endpoint = f"candles/minutes/{unit}"
        if endpoint is None:
            raise ValueError("invalid interval")
        params = {"market": market, "count": count}
        # Upbit 'to' 파라미터: 마지막 캔들 기준 시간 문자열 (UTC 기준 문서 명시, 실무상 KST 문자열도 허용되는 사례 있음)
        if to:
            params["to"] = to
        r = self.session.get(f"{UPBIT_API_BASE}/{endpoint}", params=params, timeout=10)
        r.raise_for_status()
        items = r.json()
        # Upbit returns newest first
        items = list(reversed(items))
        out: list[Candle] = []
        for it in items:
            out.append(Candle(
                market=it["market"],
                timestamp=int(time.mktime(time.strptime(it["candle_date_time_kst"], "%Y-%m-%dT%H:%M:%S"))) * 1000,
                open=float(it["opening_price"]),
                high=float(it["high_price"]),
                low=float(it["low_price"]),
                close=float(it["trade_price"]),
                volume=float(it["candle_acc_trade_volume"]),
            ))
        return out

    # ---------- WebSocket (simple ticker stream) ----------
    def stream_ticker(self, markets: list[str], on_message, run_seconds: int = 30):
        """Subscribe to ticker websocket; calls on_message(dict) per event. Stops after run_seconds."""
        url = "wss://api.upbit.com/websocket/v1"
        payload = [
            {"ticket": uuid.uuid4().hex},
            {"type": "ticker", "codes": markets},
        ]
        def _run():
            try:
                ws: WebSocket = create_connection(url, timeout=5)
                ws.send(json.dumps(payload))
                import time as _time
                end = _time.time() + run_seconds
                while _time.time() < end:
                    data = ws.recv()
                    try:
                        obj = json.loads(data)
                        on_message(obj)
                    except Exception:
                        pass
                ws.close()
            except Exception:
                pass
        th = threading.Thread(target=_run, daemon=True)
        th.start()
        return th

    # ---------- Private (signed) APIs ----------
    def _jwt_token(self, query: Optional[Dict[str, Any]] = None) -> str:
        if not self.access_key or not self.secret_key:
            raise RuntimeError("API keys not set")
        payload: Dict[str, Any] = {
            "access_key": self.access_key,
            "nonce": str(uuid.uuid4()),
        }
        if query:
            q = urlencode(query)
            query_hash = hashlib.sha512(q.encode()).hexdigest()
            payload.update({
                "query_hash": query_hash,
                "query_hash_alg": "SHA512",
            })
        token = jwt.encode(payload, self.secret_key, algorithm="HS256")
        # PyJWT returns a string in v2+
        return token

    def _auth_headers(self, query: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self._jwt_token(query)}"}

    def accounts(self) -> list[Dict[str, Any]]:
        headers = self._auth_headers()
        r = self.session.get(f"{UPBIT_API_BASE}/accounts", headers=headers, timeout=10)
        r.raise_for_status()
        return r.json()

    # ---------- Trading (LIVE guarded) ----------
    def order_chance(self, market: str) -> Dict[str, Any]:
        headers = self._auth_headers({"market": market})
        r = self.session.get(f"{UPBIT_API_BASE}/orders/chance", params={"market": market}, headers=headers, timeout=10)
        r.raise_for_status()
        return r.json()

    def create_order(self, market: str, side: str, ord_type: str,
                     volume: Optional[str] = None, price: Optional[str] = None,
                     simulate: bool = False) -> Dict[str, Any]:
        """
        Place an order. Upbit rules:
          - Limit: ord_type=limit, need price + volume
          - Market buy: ord_type=price, need price (total KRW) only
          - Market sell: ord_type=market, need volume only
        side: 'bid' (buy) or 'ask' (sell)
        volume/price must be strings per Upbit docs.
        simulate=True will not hit the API.
        LIVE 모드는 환경변수 UPBIT_LIVE=1 이어야 실제 주문.
        """
        if simulate or (str((__import__('os').getenv('UPBIT_LIVE') or '')).strip() != '1'):
            return {
                'simulate': True,
                'market': market,
                'side': side,
                'ord_type': ord_type,
                'volume': volume,
                'price': price,
            }
        if not self.access_key or not self.secret_key:
            raise RuntimeError('API keys required for live order')
        query: Dict[str, Any] = {
            'market': market,
            'side': side,
            'ord_type': ord_type,
        }
        if volume is not None:
            query['volume'] = volume
        if price is not None:
            query['price'] = price
        headers = self._auth_headers(query)
        r = self.session.post(f"{UPBIT_API_BASE}/orders", params=query, headers=headers, timeout=10)
        if r.status_code >= 400:
            try:
                return {'error': r.json(), 'status_code': r.status_code}
            except Exception:
                r.raise_for_status()
        return r.json()
