"""Per-asset state machine tracking trade lifecycle with disk persistence.

States: IDLE → ENTERED → COOLDOWN → IDLE

State is written to state.json on every mutation so a crash or restart never
loses cooldown windows or open-position awareness.
"""

import json
import logging
import os
import time


_STATE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "state.json",
)

ACTIVE_TRADES_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "active_trades.json",
)


def save_active_trades(active_trades: list) -> None:
    try:
        with open(ACTIVE_TRADES_FILE, "w") as f:
            json.dump(active_trades, f, default=str)
    except Exception as e:
        logging.warning("[STATE] failed to save active_trades.json: %s", e)


def load_active_trades() -> list:
    if os.path.exists(ACTIVE_TRADES_FILE):
        try:
            with open(ACTIVE_TRADES_FILE) as f:
                return json.load(f)
        except Exception as e:
            logging.warning("[STATE] failed to load active_trades.json: %s — starting fresh", e)
    return []


class TradeStateMachine:
    """Per-asset state tracking for the trading bot."""

    IDLE = "IDLE"
    ENTERED = "ENTERED"
    COOLDOWN = "COOLDOWN"

    def __init__(self, state_file: str = _STATE_FILE):
        self._state_file = state_file
        self._states: dict[str, str] = {}
        self._cooldown_until: dict[str, float] = {}
        self._entry_time: dict[str, float] = {}
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _save(self) -> None:
        """Atomically write full state to disk."""
        payload = {
            "states": self._states,
            "cooldown_until": self._cooldown_until,
            "entry_time": self._entry_time,
        }
        tmp = self._state_file + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(payload, f)
            os.replace(tmp, self._state_file)
        except Exception as e:
            logging.warning("[STATE] failed to persist state: %s", e)

    def _load(self) -> None:
        """Restore state from disk if state.json exists."""
        if not os.path.exists(self._state_file):
            return
        try:
            with open(self._state_file, "r") as f:
                payload = json.load(f)
            self._states = payload.get("states", {})
            self._cooldown_until = {k: float(v) for k, v in payload.get("cooldown_until", {}).items()}
            self._entry_time = {k: float(v) for k, v in payload.get("entry_time", {}).items()}
            # Expire any cooldowns that already passed while the bot was down
            now = time.time()
            expired = [a for a, t in self._cooldown_until.items() if t <= now]
            for a in expired:
                del self._cooldown_until[a]
                self._states[a] = self.IDLE
                logging.info("[STATE] %s cooldown expired while offline — reset to IDLE", a)
            logging.info(
                "[STATE] restored from disk: %s",
                {a: self._states.get(a, self.IDLE) for a in set(self._states) | set(self._entry_time)},
            )
        except Exception as e:
            logging.warning("[STATE] failed to load state.json: %s — starting fresh", e)

    # ── State machine ────────────────────────────────────────────────────────

    def get_state(self, asset: str) -> str:
        """Return current state for asset, defaulting to IDLE."""
        if asset in self._cooldown_until:
            if time.time() < self._cooldown_until[asset]:
                return self.COOLDOWN
            del self._cooldown_until[asset]
            self._states[asset] = self.IDLE
            self._save()
        return self._states.get(asset, self.IDLE)

    def set_state(self, asset: str, state: str) -> None:
        self._states[asset] = state
        self._save()

    def start_cooldown(self, asset: str, interval_seconds: int = 3600) -> None:
        """Enter cooldown for asset for interval_seconds."""
        self._cooldown_until[asset] = time.time() + interval_seconds
        self._states[asset] = self.COOLDOWN
        logging.info("[STATE] %s → COOLDOWN for %ds", asset, interval_seconds)
        self._save()

    def record_entry(self, asset: str) -> None:
        """Call this when a trade is confirmed entered."""
        self._entry_time[asset] = time.time()
        self._states[asset] = self.ENTERED
        logging.info("[STATE] %s → ENTERED", asset)
        self._save()

    def is_trade_expired(self, asset: str, max_hours: int = 12) -> bool:
        """Return True if trade has been open > max_hours with no TP hit."""
        t = self._entry_time.get(asset)
        if not t:
            return False
        elapsed_hours = (time.time() - t) / 3600
        if elapsed_hours > max_hours:
            logging.warning(
                "[TIMEOUT] %s trade open %.1fh > %dh limit — exit",
                asset, elapsed_hours, max_hours,
            )
            return True
        return False
