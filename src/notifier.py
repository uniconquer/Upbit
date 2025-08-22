import os
import requests
from typing import Optional, Protocol

# --- minimal .env loader (first call) ---
_loaded_env = False

def _load_dotenv_once():
    global _loaded_env
    if _loaded_env:
        return
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    try:
        if os.path.exists(env_path):
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line=line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if '=' not in line:
                        continue
                    k,v = line.split('=',1)
                        # skip if already in process env
                    if k not in os.environ:
                        os.environ[k] = v
    except Exception:
        pass
    _loaded_env = True

_load_dotenv_once()

# ------------ 공통 인터페이스 ------------
class Notifier(Protocol):
    def available(self) -> bool: ...
    def send_text(self, text: str) -> dict: ...

# ------------ Telegram ------------
class TelegramNotifier:
    """Telegram Bot 알림.
    필요 환경변수:
      TELEGRAM_BOT_TOKEN
      TELEGRAM_CHAT_ID (개인/그룹 chat id)
    """
    def __init__(self, bot_token: Optional[str] = None, chat_id: Optional[str] = None):
        self.bot_token = bot_token or os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = chat_id or os.getenv('TELEGRAM_CHAT_ID')

    def available(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    def send_text(self, text: str) -> dict:
        if not self.available():
            return {"ok": False, "error": "telegram creds missing"}
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": text[:4000]}
        try:
            r = requests.post(url, json=payload, timeout=5)
            if r.status_code == 200:
                return {"ok": True}
            return {"ok": False, "status": r.status_code, "resp": r.text}
        except Exception as e:
            return {"ok": False, "error": str(e)}

_notifier_singleton: Optional[Notifier] = None

def get_notifier() -> Notifier:
    """Telegram 우선 단일 팩토리. 자격 없으면 Dummy 반환."""
    global _notifier_singleton
    if _notifier_singleton is not None:
        return _notifier_singleton
    tg = TelegramNotifier()
    if tg.available():
        _notifier_singleton = tg
        return _notifier_singleton
    class Dummy:
        def available(self) -> bool: return False
        def send_text(self, text: str) -> dict: return {"ok": False, "error": "no notifier configured"}
    _notifier_singleton = Dummy()
    return _notifier_singleton
