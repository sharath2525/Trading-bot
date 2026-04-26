"""Simple per-asset state machine tracking trade lifecycle.

States: IDLE → ENTERED → COOLDOWN → IDLE
"""

import logging
import time


class TradeStateMachine:
    """Per-asset state tracking for the trading bot."""

    IDLE = "IDLE"
    ENTERED = "ENTERED"
    COOLDOWN = "COOLDOWN"

    def __init__(self):
        self._states: dict[str, str] = {}
        self._cooldown_until: dict[str, float] = {}
        self._entry_time: dict[str, float] = {}

    def get_state(self, asset: str) -> str:
        """Return current state for asset, defaulting to IDLE."""
        if asset in self._cooldown_until:
            if time.time() < self._cooldown_until[asset]:
                return self.COOLDOWN
            del self._cooldown_until[asset]
            self._states[asset] = self.IDLE
        return self._states.get(asset, self.IDLE)

    def set_state(self, asset: str, state: str) -> None:
        self._states[asset] = state

    def start_cooldown(self, asset: str, interval_seconds: int = 3600) -> None:
        """Enter cooldown for asset for interval_seconds."""
        self._cooldown_until[asset] = time.time() + interval_seconds
        self._states[asset] = self.COOLDOWN
        logging.info("[STATE] %s → COOLDOWN for %ds", asset, interval_seconds)

    def record_entry(self, asset: str) -> None:
        """Call this when a trade is confirmed entered."""
        self._entry_time[asset] = time.time()
        self.set_state(asset, self.ENTERED)
        logging.info("[STATE] %s → ENTERED", asset)

    def is_trade_expired(self, asset: str, max_hours: int = 12) -> bool:
        """
        Returns True if trade has been open > max_hours
        with no TP1 hit (stuck / dead trade).
        """
        t = self._entry_time.get(asset)
        if not t:
            return False
        elapsed_hours = (time.time() - t) / 3600
        if elapsed_hours > max_hours:
            logging.warning(
                "[TIMEOUT] %s trade open %.1fh > %dh limit — exit",
                asset, elapsed_hours, max_hours
            )
            return True
        return False
