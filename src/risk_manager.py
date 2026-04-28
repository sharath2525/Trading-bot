"""Centralized risk management for the trading agent.

All safety guards are enforced here, independent of LLM decisions.
The LLM cannot override these limits — they are hard-coded checks
applied before every trade execution.
"""

import json
import logging
import os
from datetime import datetime, timezone

from src.config_loader import CONFIG

_RISK_STATE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "risk_state.json",
)


class RiskManager:
    """Enforces risk limits on every trade before execution."""

    def __init__(self):
        self.max_position_pct = float(CONFIG.get("max_position_pct") or 10)
        self.max_loss_per_position_pct = float(CONFIG.get("max_loss_per_position_pct") or 20)
        self.max_leverage = float(CONFIG.get("max_leverage") or 10)
        self.max_total_exposure_pct = float(CONFIG.get("max_total_exposure_pct") or 50)
        self.daily_loss_circuit_breaker_pct = float(CONFIG.get("daily_loss_circuit_breaker_pct") or 10)
        self.mandatory_sl_pct = float(CONFIG.get("mandatory_sl_pct") or 5)
        self.max_concurrent_positions = int(CONFIG.get("max_concurrent_positions") or 10)
        self.min_balance_reserve_pct = float(CONFIG.get("min_balance_reserve_pct") or 20)

        # Daily tracking
        self.daily_high_value = None
        self.daily_high_date = None
        self.circuit_breaker_active = False
        self.circuit_breaker_date = None
        self._state_file = _RISK_STATE_FILE
        self._load_circuit_state()

    # ── Circuit-breaker persistence ──────────────────────────────────────────

    def _save_circuit_state(self) -> None:
        """Atomically persist circuit breaker and daily high watermark to disk."""
        payload = {
            "circuit_breaker_active": self.circuit_breaker_active,
            "circuit_breaker_date": str(self.circuit_breaker_date) if self.circuit_breaker_date else None,
            "daily_high_value": self.daily_high_value,
            "daily_high_date": str(self.daily_high_date) if self.daily_high_date else None,
        }
        tmp = self._state_file + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(payload, f)
            os.replace(tmp, self._state_file)
        except Exception as e:
            logging.warning("[RISK] failed to persist circuit state: %s", e)

    def _load_circuit_state(self) -> None:
        """Restore circuit breaker state from disk on startup."""
        if not os.path.exists(self._state_file):
            return
        try:
            with open(self._state_file, "r") as f:
                payload = json.load(f)
            self.circuit_breaker_active = bool(payload.get("circuit_breaker_active", False))
            self.daily_high_value = payload.get("daily_high_value")
            from datetime import date
            date_str = payload.get("daily_high_date")
            if date_str:
                self.daily_high_date = date.fromisoformat(date_str)
            cb_date_str = payload.get("circuit_breaker_date")
            if cb_date_str:
                self.circuit_breaker_date = date.fromisoformat(cb_date_str)
            logging.info(
                "[RISK] restored circuit state: breaker=%s daily_high=%s",
                self.circuit_breaker_active, self.daily_high_value,
            )
        except Exception as e:
            logging.warning("[RISK] failed to load risk_state.json: %s — starting fresh", e)

    # ─────────────────────────────────────────────────────────────────────────

    def _reset_daily_if_needed(self, account_value: float):
        """Reset daily high watermark at UTC day boundary."""
        today = datetime.now(timezone.utc).date()
        if self.daily_high_date != today:
            self.daily_high_value = account_value
            self.daily_high_date = today
            self.circuit_breaker_active = False
            self.circuit_breaker_date = None
            self._save_circuit_state()
        elif account_value > (self.daily_high_value or 0):
            self.daily_high_value = account_value
            self._save_circuit_state()

    # ------------------------------------------------------------------
    # Individual checks — each returns (allowed: bool, reason: str)
    # ------------------------------------------------------------------

    def check_position_size(self, alloc_usd: float, account_value: float) -> tuple[bool, str]:
        """Single position cannot exceed max_position_pct of leverage-adjusted buying power.

        Buying power = account_value × max_leverage.
        Max position  = buying_power × max_position_pct / 100.

        Example: $100 account, 5× leverage, 15% cap → $500 buying power → $75 max.
        The old formula ($100 × 15% = $15) ignored leverage entirely.
        """
        if account_value <= 0:
            return False, "Account value is zero or negative"
        buying_power = account_value * self.max_leverage
        max_alloc = buying_power * (self.max_position_pct / 100.0)
        if alloc_usd > max_alloc:
            return False, (
                f"Allocation ${alloc_usd:.2f} exceeds {self.max_position_pct}% "
                f"of buying power ${buying_power:.2f} (max ${max_alloc:.2f})"
            )
        return True, ""

    def check_total_exposure(self, positions: list[dict], new_alloc: float,
                              account_value: float) -> tuple[bool, str]:
        """Sum of all position notionals + new allocation cannot exceed max_total_exposure_pct."""
        current_exposure = 0.0
        for pos in positions:
            qty = abs(float(pos.get("quantity") or pos.get("szi") or 0))
            entry = float(pos.get("entry_price") or pos.get("entryPx") or 0)
            current_exposure += qty * entry
        total = current_exposure + new_alloc
        max_exposure = account_value * (self.max_total_exposure_pct / 100.0)
        if total > max_exposure:
            return False, (
                f"Total exposure ${total:.2f} would exceed {self.max_total_exposure_pct}% "
                f"of account (${max_exposure:.2f})"
            )
        return True, ""

    def check_leverage(self, alloc_usd: float, account_value: float) -> tuple[bool, str]:
        """Effective leverage of new trade cannot exceed max_leverage.

        effective_leverage = allocation / account_value (equity denominator).
        Using balance (same as account_value here) was previously named misleadingly.
        """
        if account_value <= 0:
            return False, "Account value is zero or negative"
        effective_lev = alloc_usd / account_value
        if effective_lev > self.max_leverage:
            return False, (
                f"Effective leverage {effective_lev:.1f}x exceeds max {self.max_leverage}x"
            )
        return True, ""

    def check_daily_drawdown(self, account_value: float) -> tuple[bool, str]:
        """Activate circuit breaker if account drops max % from daily high."""
        self._reset_daily_if_needed(account_value)
        if self.circuit_breaker_active:
            return False, "Daily loss circuit breaker is active — no new trades until tomorrow (UTC)"
        if self.daily_high_value and self.daily_high_value > 0:
            drawdown_pct = ((self.daily_high_value - account_value) / self.daily_high_value) * 100
            if drawdown_pct > self.daily_loss_circuit_breaker_pct:
                self.circuit_breaker_active = True
                self.circuit_breaker_date = datetime.now(timezone.utc).date()
                self._save_circuit_state()
                return False, (
                    f"Daily drawdown {drawdown_pct:.2f}% exceeds circuit breaker "
                    f"threshold of {self.daily_loss_circuit_breaker_pct}%"
                )
        return True, ""

    def check_concurrent_positions(self, current_count: int) -> tuple[bool, str]:
        """Limit number of simultaneous open positions."""
        if current_count >= self.max_concurrent_positions:
            return False, (
                f"Already at max concurrent positions ({self.max_concurrent_positions})"
            )
        return True, ""

    def check_balance_reserve(self, balance: float, initial_balance: float) -> tuple[bool, str]:
        """Don't trade if balance falls below reserve threshold."""
        if initial_balance <= 0:
            return True, ""
        min_balance = initial_balance * (self.min_balance_reserve_pct / 100.0)
        if balance < min_balance:
            return False, (
                f"Balance ${balance:.2f} below minimum reserve "
                f"${min_balance:.2f} ({self.min_balance_reserve_pct}% of initial)"
            )
        return True, ""

    # ------------------------------------------------------------------
    # Stop-loss / Take-profit enforcement
    # ------------------------------------------------------------------

    # Hyperliquid base taker fee (both sides of a round trip = 2×)
    TAKER_FEE_PCT = 0.045

    def enforce_stop_loss(self, sl_price: float | None, entry_price: float,
                           is_buy: bool, atr14: float | None = None) -> float:
        """Ensure every trade has a stop-loss. Auto-set if missing.

        SL distance = max(mandatory_sl_pct% of price, 1.0 × ATR14).
        A flat-percentage SL can be tighter than actual volatility; using the
        larger of the two prevents the SL from sitting inside normal price noise.
        atr14 is optional — falls back to percentage-only when not provided.
        """
        if sl_price is not None:
            return sl_price
        pct_distance = entry_price * (self.mandatory_sl_pct / 100.0)
        atr_distance = float(atr14) if atr14 and atr14 > 0 else 0.0
        sl_distance = max(pct_distance, atr_distance)
        logging.info(
            "RISK: Auto-SL — pct_dist=%.4f atr_dist=%.4f chosen=%.4f",
            pct_distance, atr_distance, sl_distance,
        )
        if is_buy:
            return round(entry_price - sl_distance, 2)
        else:
            return round(entry_price + sl_distance, 2)

    def enforce_take_profit(self, tp_price: float | None, entry_price: float,
                             is_buy: bool) -> float:
        """Ensure TP covers round-trip fees plus a minimum net profit (3× fees)."""
        # round-trip fee = 2 × taker_fee; require at least 3× to clear fees with profit
        min_profit_pct = self.TAKER_FEE_PCT * 2 * 3  # = 0.27 %
        min_distance = entry_price * (min_profit_pct / 100.0)
        if is_buy:
            min_tp = round(entry_price + min_distance, 6)
            if tp_price is None or tp_price < min_tp:
                logging.info("RISK: TP adjusted to %.6f to cover fees (min %.4f%% from entry)",
                             min_tp, min_profit_pct)
                return min_tp
        else:
            max_tp = round(entry_price - min_distance, 6)
            if tp_price is None or tp_price > max_tp:
                logging.info("RISK: TP adjusted to %.6f to cover fees (min %.4f%% from entry)",
                             max_tp, min_profit_pct)
                return max_tp
        return tp_price

    # ------------------------------------------------------------------
    # Force-close losing positions
    # ------------------------------------------------------------------

    def check_losing_positions(self, positions: list[dict]) -> list[dict]:
        """Return positions that should be force-closed due to excessive loss.

        Args:
            positions: List of position dicts with keys:
                coin/symbol, szi/quantity, entryPx/entry_price,
                pnl/unrealized_pnl

        Returns:
            List of positions that exceed the max loss threshold.
        """
        to_close = []
        for pos in positions:
            coin = pos.get("coin") or pos.get("symbol")
            entry_px = float(pos.get("entryPx") or pos.get("entry_price") or 0)
            size = float(pos.get("szi") or pos.get("quantity") or 0)
            pnl = float(pos.get("pnl") or pos.get("unrealized_pnl") or 0)

            if entry_px == 0 or size == 0:
                continue

            notional = abs(size) * entry_px
            if notional == 0:
                continue

            loss_pct = abs(pnl / notional) * 100 if pnl < 0 else 0

            if loss_pct >= self.max_loss_per_position_pct:
                logging.warning(
                    "RISK: Force-closing %s — loss %.2f%% exceeds max %.2f%%",
                    coin, loss_pct, self.max_loss_per_position_pct
                )
                to_close.append({
                    "coin": coin,
                    "size": abs(size),
                    "is_long": size > 0,
                    "loss_pct": round(loss_pct, 2),
                    "pnl": round(pnl, 2),
                })
        return to_close

    # ------------------------------------------------------------------
    # Composite validation — run all checks before a trade
    # ------------------------------------------------------------------

    def validate_trade(self, trade: dict, account_state: dict,
                        initial_balance: float) -> tuple[bool, str, dict]:
        """Run all safety checks on a proposed trade.

        Args:
            trade: LLM trade decision with keys:
                asset, action, allocation_usd, tp_price, sl_price
            account_state: Current account with keys:
                balance, total_value, positions
            initial_balance: Starting balance for reserve check

        Returns:
            (allowed, reason, adjusted_trade)
            adjusted_trade may have modified sl_price if it was missing.
        """
        action = trade.get("action", "hold")
        if action == "hold":
            return True, "", trade

        alloc_usd = float(trade.get("allocation_usd", 0))
        if alloc_usd <= 0:
            return False, "Zero or negative allocation", trade

        # Hyperliquid minimum order size is $10
        if alloc_usd < 11.0:
            alloc_usd = 11.0
            trade = {**trade, "allocation_usd": alloc_usd}
            logging.info("RISK: Bumped allocation to $11 (Hyperliquid $10 minimum)")

        account_value = float(account_state.get("total_value", 0))
        balance = float(account_state.get("balance", 0))
        positions = account_state.get("positions", [])
        is_buy = action == "buy"

        # 1. Daily drawdown circuit breaker
        ok, reason = self.check_daily_drawdown(account_value)
        if not ok:
            return False, reason, trade

        # 2. Balance reserve
        ok, reason = self.check_balance_reserve(balance, initial_balance)
        if not ok:
            return False, reason, trade

        # 3. Position size limit (leverage-aware cap)
        ok, reason = self.check_position_size(alloc_usd, account_value)
        if not ok:
            # Cap to the leverage-adjusted maximum instead of rejecting outright
            max_alloc = account_value * self.max_leverage * (self.max_position_pct / 100.0)
            if max_alloc < 11.0:
                max_alloc = 11.0
            logging.warning("RISK: Capping allocation from $%.2f to $%.2f", alloc_usd, max_alloc)
            alloc_usd = max_alloc
            trade = {**trade, "allocation_usd": alloc_usd}

        # 4. Total exposure
        ok, reason = self.check_total_exposure(positions, alloc_usd, account_value)
        if not ok:
            return False, reason, trade

        # 5. Leverage check — uses account_value (equity) as denominator
        ok, reason = self.check_leverage(alloc_usd, account_value)
        if not ok:
            return False, reason, trade

        # 6. Concurrent positions
        active_count = sum(
            1 for p in positions
            if abs(float(p.get("szi") or p.get("quantity") or 0)) > 0
        )
        ok, reason = self.check_concurrent_positions(active_count)
        if not ok:
            return False, reason, trade

        # 7. Enforce mandatory stop-loss (ATR-aware minimum distance)
        current_price = float(trade.get("current_price", 0))
        entry_price = current_price if current_price > 0 else 1.0
        atr14 = trade.get("atr14")  # injected by main.py from asset_ctx long_term_4h
        sl_price = trade.get("sl_price")
        enforced_sl = self.enforce_stop_loss(sl_price, entry_price, is_buy, atr14)
        if sl_price is None:
            logging.info("RISK: Auto-setting SL at %.6f (atr14=%s)", enforced_sl, atr14)

        # 8. Enforce fee-aware take-profit minimum
        tp_price = trade.get("tp_price")
        enforced_tp = self.enforce_take_profit(tp_price, entry_price, is_buy)

        trade = {**trade, "sl_price": enforced_sl, "tp_price": enforced_tp}

        return True, "", trade

    def get_risk_summary(self) -> dict:
        """Return current risk parameters for inclusion in LLM context."""
        return {
            "max_position_pct": self.max_position_pct,
            "max_loss_per_position_pct": self.max_loss_per_position_pct,
            "max_leverage": self.max_leverage,
            "max_total_exposure_pct": self.max_total_exposure_pct,
            "daily_loss_circuit_breaker_pct": self.daily_loss_circuit_breaker_pct,
            "mandatory_sl_pct": self.mandatory_sl_pct,
            "max_concurrent_positions": self.max_concurrent_positions,
            "min_balance_reserve_pct": self.min_balance_reserve_pct,
            "circuit_breaker_active": self.circuit_breaker_active,
            "taker_fee_pct": self.TAKER_FEE_PCT,
            "min_tp_pct_from_entry": round(self.TAKER_FEE_PCT * 2 * 3, 4),
        }
