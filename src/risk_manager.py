from dataclasses import dataclass
from typing import Optional

@dataclass
class RiskConfig:
    max_position_pct: float = 100.0   # 포트폴리오 대비 최대 포지션 %
    max_daily_loss_pct: float = 20.0  # 일일 최대 손실 % (피크 대비)
    stop_loss_pct: Optional[float] = None  # 진입 대비 손절 %
    take_profit_pct: Optional[float] = None  # 진입 대비 익절 %

class RiskManager:
    def __init__(self, config: RiskConfig):
        self.cfg = config
        self.start_equity: Optional[float] = None
        self.peak_equity: Optional[float] = None

    def update_equity(self, equity: float):
        if self.start_equity is None:
            self.start_equity = equity
        if self.peak_equity is None or equity > self.peak_equity:
            self.peak_equity = equity

    def can_open(self, proposed_position_value: float, equity: float) -> tuple[bool, str]:
        if equity <= 0:
            return False, "no equity"
        if proposed_position_value / equity * 100 > self.cfg.max_position_pct:
            return False, "position size exceeds max_position_pct"
        if self.peak_equity and equity < self.peak_equity:
            dd_pct = (self.peak_equity - equity) / self.peak_equity * 100
            if dd_pct > self.cfg.max_daily_loss_pct:
                return False, "daily loss limit breached"
        return True, "ok"

    def exit_signal(self, entry_price: float, current_price: float) -> Optional[str]:
        if entry_price <= 0:
            return None
        change_pct = (current_price / entry_price - 1) * 100
        if self.cfg.stop_loss_pct is not None and change_pct <= -abs(self.cfg.stop_loss_pct):
            return "stop_loss"
        if self.cfg.take_profit_pct is not None and change_pct >= abs(self.cfg.take_profit_pct):
            return "take_profit"
        return None
