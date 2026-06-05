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
    tmp = ACTIVE_TRADES_FILE + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(active_trades, f, default=str)
        os.replace(tmp, ACTIVE_TRADES_FILE)
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
            logging.critical("[STATE] CRITICAL: failed to persist state: %s — next restart will not know about open positions", e)

    def _load(self) -> None:
        """Restore state from disk if state.json exists."""
        if not os.path.exists(self._state_file):
            return
        try:
            with open(self._state_file, "r") as f:
                payload = json.load(f)

            # HIGH-1 FIX: Validate structural integrity after JSON parse. Valid JSON with a
            # null/wrong-type "states" key (e.g. written during a crash mid-serialization)
            # would previously set self._states=None, causing AttributeError on first access
            # and masking re-entry on open exchange positions. Fail hard on bad structure.
            _raw_states = payload.get("states", {})
            _raw_cooldown = payload.get("cooldown_until", {})
            _raw_entry = payload.get("entry_time", {})
            if not isinstance(_raw_states, dict):
                raise ValueError(
                    f"'states' field is {type(_raw_states).__name__}, expected dict. "
                    "state.json is structurally corrupt."
                )
            if not isinstance(_raw_cooldown, dict):
                raise ValueError(
                    f"'cooldown_until' field is {type(_raw_cooldown).__name__}, expected dict."
                )
            if not isinstance(_raw_entry, dict):
                raise ValueError(
                    f"'entry_time' field is {type(_raw_entry).__name__}, expected dict."
                )
            # Validate each state value is a known string constant
            _valid_states = {self.IDLE, self.ENTERED, self.COOLDOWN}
            for _asset, _st in _raw_states.items():
                if _st not in _valid_states:
                    raise ValueError(
                        f"Unknown state '{_st}' for asset '{_asset}' — expected one of {_valid_states}."
                    )

            self._states = _raw_states
            self._cooldown_until = {k: float(v) for k, v in _raw_cooldown.items()}
            self._entry_time = {k: float(v) for k, v in _raw_entry.items()}
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
        except json.JSONDecodeError as e:
            # CHAOS-7 FIX: Corrupt state.json (bad JSON) must halt, not silently reset to IDLE.
            # A silent reset clears all ENTERED states — if a position is open on the exchange,
            # the bot re-enters and creates 2× exposure at 5× leverage (= 10× effective leverage).
            # Force a clean exit so the operator can manually inspect and repair state.json.
            logging.critical(
                "[STATE] FATAL: state.json is corrupt (JSON parse error: %s). "
                "Refusing to start to prevent re-entry on existing open positions. "
                "Inspect '%s', fix or delete it, then restart.",
                e, self._state_file,
            )
            import sys as _state_sys
            _state_sys.exit(1)
        except ValueError as e:
            # HIGH-1 FIX: Structurally corrupt state.json (valid JSON, wrong types/values).
            # Same risk as JSONDecodeError — halt to prevent re-entry on open positions.
            logging.critical(
                "[STATE] FATAL: state.json has invalid structure: %s. "
                "Refusing to start to prevent re-entry on existing open positions. "
                "Inspect '%s', fix or delete it, then restart.",
                e, self._state_file,
            )
            import sys as _state_sys
            _state_sys.exit(1)
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

    def clear_entry(self, asset: str) -> None:
        """Remove stale entry_time for an asset without changing its state."""
        if asset in self._entry_time:
            del self._entry_time[asset]
            self._save()
            logging.info("[STATE] %s stale entry_time cleared", asset)

    def is_trade_expired_minutes(self, asset: str, max_minutes: float) -> bool:
        """Return True if trade has been open > max_minutes. Used for candle-count exits."""
        t = self._entry_time.get(asset)
        if not t:
            return False
        elapsed_minutes = (time.time() - t) / 60.0
        if elapsed_minutes > max_minutes:
            logging.warning(
                "[TIMEOUT] %s trade open %.1fmin > %.0fmin limit — exit",
                asset, elapsed_minutes, max_minutes,
            )
            return True
        return False
