"""Entry-point script that wires together the trading agent, data feeds, and API."""

import sys
import argparse
import asyncio
import json
import logging
import math
import os
import pathlib
import signal
import socket as _socket
import time
import traceback
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from aiohttp import web
from collections import deque
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

from src.agent.decision_maker import TradingAgent
from src.alerts import send_alert
from src.config_loader import CONFIG
from src.indicators.local_indicators import compute_all, last_n, latest
from src.risk_manager import RiskManager
from src.strategy import compute_bb_stochrsi_signal, is_trend_regime_active, funding_rate_ok, session_gate_ok
from src.trade_state import TradeStateMachine, load_active_trades, save_active_trades
from src.trading.hyperliquid_api import HyperliquidAPI
from src.utils.prompt_utils import json_default, round_or_none, round_series

load_dotenv()

_root_logger = logging.getLogger()
_root_logger.setLevel(logging.INFO)
for _h in list(_root_logger.handlers):          # clear whatever libraries set up before us
    _root_logger.removeHandler(_h)
_stderr_handler = logging.StreamHandler()
_stderr_handler.setLevel(logging.INFO)
_stderr_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
_root_logger.addHandler(_stderr_handler)
_log_file_handler = RotatingFileHandler("bot.log", maxBytes=5 * 1024 * 1024, backupCount=3)
_log_file_handler.setLevel(logging.INFO)
_log_file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
_root_logger.addHandler(_log_file_handler)

_api_host = os.getenv("API_HOST", "0.0.0.0").strip()

# Telegram alert status — warn at startup if not configured
from src.alerts import _ENABLED as _telegram_enabled
if not _telegram_enabled:
    logging.warning(
        "[ALERTS] ⚠️  Telegram NOT configured — push alerts disabled. "
        "Set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID in .env to enable. "
        "Without alerts: circuit breaker, API failures, and KILLSWITCH events are silent."
    )
    print("[ALERTS] ⚠️  Telegram NOT configured — no push notifications. Add TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID to .env")

_shutdown = False  # Set to True by SIGTERM/SIGINT handler for clean loop exit

# ── C-5: Instance lock — prevents two simultaneous bot instances ──────────────
_INSTANCE_LOCK_SOCK: _socket.socket | None = None
_INSTANCE_LOCK_PORT = 47293  # arbitrary loopback-only port used as a process mutex

def _acquire_instance_lock() -> None:
    """Bind a loopback socket as a process-level mutex. Exits if already bound."""
    global _INSTANCE_LOCK_SOCK
    _INSTANCE_LOCK_SOCK = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    _INSTANCE_LOCK_SOCK.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 0)
    try:
        _INSTANCE_LOCK_SOCK.bind(("127.0.0.1", _INSTANCE_LOCK_PORT))
        logging.info("[LOCK] Instance lock acquired (port %d)", _INSTANCE_LOCK_PORT)
    except OSError:
        logging.critical(
            "[LOCK] Port %d already bound — another bot instance is running. Exiting.",
            _INSTANCE_LOCK_PORT,
        )
        print(
            f"ERROR: Another bot instance is already running "
            f"(lock port {_INSTANCE_LOCK_PORT} in use).\n"
            "Stop the other instance first, or change INSTANCE_LOCK_PORT in .env."
        )
        sys.exit(1)

def _release_instance_lock() -> None:
    if _INSTANCE_LOCK_SOCK:
        try:
            _INSTANCE_LOCK_SOCK.close()
        except OSError:
            pass
# ─────────────────────────────────────────────────────────────────────────────

# ── CRITICAL-7: Daily trade count persistence across restarts ─────────────────
_DAILY_COUNT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "daily_count.json")

def _load_daily_count() -> tuple[int, str | None]:
    """Return (count, date_str) from daily_count.json, or (0, None) if missing/corrupt."""
    try:
        p = os.path.normpath(_DAILY_COUNT_FILE)
        if os.path.exists(p):
            with open(p) as _f:
                d = json.load(_f)
                return int(d.get("count", 0)), d.get("date")
    except Exception as _e:
        logging.warning("[DAILY COUNT] load failed: %s", _e)
    return 0, None

def _save_daily_count(count: int, date: str) -> None:
    """Atomically write daily trade count to disk."""
    try:
        p = os.path.normpath(_DAILY_COUNT_FILE)
        tmp = p + ".tmp"
        with open(tmp, "w") as _f:
            json.dump({"count": count, "date": date}, _f)
        os.replace(tmp, p)
    except Exception as _e:
        logging.warning("[DAILY COUNT] save failed: %s", _e)
# ─────────────────────────────────────────────────────────────────────────────

# ── P3-M2: Diary index — compact per-asset index so guardian avoids full scan ──
_DIARY_INDEX_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "diary_index.json")

def _load_diary_index() -> dict:
    """Return {asset: latest_trade_entry_dict} from diary_index.json."""
    try:
        p = os.path.normpath(_DIARY_INDEX_FILE)
        if os.path.exists(p):
            with open(p) as _f:
                return json.load(_f)
    except Exception as _e:
        logging.warning("[DIARY INDEX] load failed: %s", _e)
    return {}

def _save_diary_index(index: dict) -> None:
    """Atomically write the diary index."""
    try:
        p = os.path.normpath(_DIARY_INDEX_FILE)
        tmp = p + ".tmp"
        with open(tmp, "w") as _f:
            json.dump(index, _f)
        os.replace(tmp, p)
    except Exception as _e:
        logging.warning("[DIARY INDEX] save failed: %s", _e)
# ─────────────────────────────────────────────────────────────────────────────

# ── E-4: Bounded log rotation for append-only JSONL files ────────────────────
_LOG_ROTATION_MAX_MB = 50

def _rotate_if_needed(path: str, max_mb: int = _LOG_ROTATION_MAX_MB) -> None:
    """Rotate path → path.old when it exceeds max_mb. Prevents unbounded disk growth."""
    try:
        if os.path.exists(path) and os.path.getsize(path) > max_mb * 1024 * 1024:
            os.replace(path, path + ".old")
            logging.info("[ROTATE] %s exceeded %dMB — rotated to %s.old", path, max_mb, path)
    except Exception as _re:
        logging.warning("[ROTATE] %s failed: %s", path, _re)
# ─────────────────────────────────────────────────────────────────────────────

# ── FL-4: Persist initial account value across restarts ──────────────────────
_INITIAL_BALANCE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "initial_balance.json"
)

def _load_initial_balance() -> float | None:
    """Return the persisted initial account value, or None if not yet set."""
    try:
        p = os.path.normpath(_INITIAL_BALANCE_FILE)
        if os.path.exists(p):
            with open(p) as _f:
                val = float(json.load(_f).get("initial_account_value", 0))
                return val if val > 0 else None
    except Exception as _e:
        logging.warning("[INIT] failed to load initial_balance.json: %s", _e)
    return None

def _save_initial_balance(value: float) -> None:
    try:
        p = os.path.normpath(_INITIAL_BALANCE_FILE)
        with open(p + ".tmp", "w") as _f:
            json.dump({"initial_account_value": value,
                       "set_at": __import__("datetime").datetime.utcnow().isoformat()}, _f)
        os.replace(p + ".tmp", p)
        logging.info("[INIT] initial_account_value=%.2f persisted to disk", value)
    except Exception as _e:
        logging.warning("[INIT] failed to save initial_balance.json: %s", _e)
# ─────────────────────────────────────────────────────────────────────────────


def clear_terminal():
    """Clear the terminal screen on Windows or POSIX systems."""
    os.system('cls' if os.name == 'nt' else 'clear')


def get_interval_seconds(interval_str):
    """Convert interval strings like '5m' or '1h' to seconds."""
    if interval_str.endswith('m'):
        return int(interval_str[:-1]) * 60
    elif interval_str.endswith('h'):
        return int(interval_str[:-1]) * 3600
    elif interval_str.endswith('d'):
        return int(interval_str[:-1]) * 86400
    else:
        raise ValueError(f"Unsupported interval: {interval_str}")


# _code_decide_direction() REMOVED — 2026-06-01
# New architecture: direction set by compute_bb_stochrsi_signal() in strategy.py.
# BB lower band touch + StochRSI oversold hook → LONG
# BB upper band touch + StochRSI overbought hook → SHORT


# _code_compute_tpsl() REMOVED — 2026-06-01
# New architecture: TP = BB midline (mean reversion target), SL = entry ± 1.5×ATR(5m)
# Both computed inline in the scoring pipeline from compute_bb_stochrsi_signal() output.


# multi_timeframe_confluence() REMOVED — 2026-06-01
# Mean reversion does NOT require trend alignment — it trades AGAINST the trend.
# _build_confluence_fingerprint() REMOVED — no longer used (anomaly check is stateless).


async def _fetch_macro_context() -> dict:
    """Fetch economic calendar events and news headlines from free RSS feeds.

    MED-3/PERF-1 FIX: All feeds are now fetched concurrently via asyncio.gather().
    Old sequential fetch had a 9-second worst-case delay (3 feeds × 3s timeout each).
    Concurrent fetch reduces worst case to a single 3-second timeout.

    MED-4/SEC-1 FIX: defusedxml used instead of stdlib xml.etree to prevent XML bomb
    (billion-laughs) attacks from compromised or MITM-intercepted RSS feeds.
    Falls back to stdlib ET if defusedxml is not installed (logs WARNING once).
    """
    import aiohttp as _aiohttp
    try:
        import defusedxml.ElementTree as ET  # type: ignore  # pip install defusedxml
    except ImportError:
        import xml.etree.ElementTree as ET  # type: ignore
        logging.warning(
            "[MACRO] defusedxml not installed — using stdlib xml.etree (XML bomb risk). "
            "Install with: pip install defusedxml"
        )

    context: dict = {
        "events": [],
        "headlines": [],
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }

    if not CONFIG.get("news_fetch_enabled", True):
        return context

    _headers = {"User-Agent": "Mozilla/5.0"}
    _feeds = [
        ("https://rss.forexfactory.com", "events", "[FF]"),
        ("https://www.coindesk.com/arc/outboundfeeds/rss/", "headlines", "[CoinDesk]"),
        ("https://feeds.reuters.com/reuters/businessNews", "headlines", "[Reuters]"),
    ]

    async def _fetch_one(sess, url, bucket, prefix):
        try:
            async with sess.get(
                url,
                timeout=_aiohttp.ClientTimeout(total=3),
                headers=_headers,
            ) as _resp:
                _text = await _resp.text()
                _root = ET.fromstring(_text)
                _items = []
                for _item in _root.iter("item"):
                    _title = (_item.findtext("title") or "").strip()
                    if _title:
                        _items.append(f"{prefix} {_title}")
                    if len(_items) >= 8:
                        break
                return bucket, _items
        except Exception as _fe:
            logging.debug("[MACRO] feed %s failed: %s", url, _fe)
            return bucket, []

    try:
        async with _aiohttp.ClientSession() as _sess:
            _results = await asyncio.gather(
                *(_fetch_one(_sess, url, bucket, prefix) for url, bucket, prefix in _feeds),
                return_exceptions=True,
            )
        for _r in _results:
            if isinstance(_r, Exception):
                logging.debug("[MACRO] gather error: %s", _r)
                continue
            _bucket, _items = _r
            context[_bucket].extend(_items)
    except Exception as _se:
        logging.debug("[MACRO] session error: %s", _se)

    return context


# ── CORS middleware ───────────────────────────────────────────────────────────
@web.middleware
async def cors_middleware(request, handler):
    """Allow same-origin requests (served from the bot itself) and localhost."""
    if request.method == "OPTIONS":
        response = web.Response()
    else:
        try:
            response = await handler(request)
        except web.HTTPException as ex:
            response = ex
    _port = CONFIG.get("api_port") or "3000"
    # Reflect request origin only for exact localhost/127.0.0.1 origins — substring
    # matching would allow crafted domains like "localhost.attacker.com" to pass.
    _origin = request.headers.get("Origin", "")
    _allowed_origins = {f"http://localhost:{_port}", f"http://127.0.0.1:{_port}"}
    _allowed = _origin if _origin in _allowed_origins else f"http://localhost:{_port}"
    response.headers["Access-Control-Allow-Origin"] = _allowed
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response
# ─────────────────────────────────────────────────────────────────────────────


@web.middleware
async def auth_middleware(request, handler):
    return await handler(request)


def main():
    """Parse CLI args, bootstrap dependencies, and launch the trading loop."""
    _acquire_instance_lock()  # C-5: exits if another instance is running
    clear_terminal()
    _network = os.getenv("HYPERLIQUID_NETWORK", "").strip().lower()
    if _network not in ("mainnet", "testnet"):
        print("WARNING: HYPERLIQUID_NETWORK is not set in .env — defaulting to MAINNET. Real funds at risk.")
        print("         Set HYPERLIQUID_NETWORK=testnet in .env to use the testnet instead.")
        logging.warning("[BOOT] HYPERLIQUID_NETWORK not explicitly set — defaulting to mainnet. Real funds at risk.")
    parser = argparse.ArgumentParser(description="LLM-based Trading Agent on Hyperliquid")
    parser.add_argument("--assets", type=str, nargs="+", required=False, help="Assets to trade, e.g., BTC ETH")
    parser.add_argument("--interval", type=str, required=False, help="Interval period, e.g., 1h")
    args = parser.parse_args()

    # Allow assets/interval via .env (CONFIG) if CLI not provided
    assets_env = CONFIG.get("assets")
    interval_env = CONFIG.get("interval")
    if (not args.assets or len(args.assets) == 0) and assets_env:
        # Support space or comma separated
        if "," in assets_env:
            args.assets = [a.strip() for a in assets_env.split(",") if a.strip()]
        else:
            args.assets = [a.strip() for a in assets_env.split(" ") if a.strip()]
    if not args.interval and interval_env:
        args.interval = interval_env

    if not args.assets or not args.interval:
        parser.error("Please provide --assets and --interval, or set ASSETS and INTERVAL in .env")

    # H-3: Validate that risk parameters are explicitly configured, not relying on permissive defaults.
    _risk_defaults = {
        "MAX_LEVERAGE": ("max_leverage", 5, 5),
        "MAX_POSITION_PCT": ("max_position_pct", 15, 15),
        "MAX_LOSS_PER_POSITION_PCT": ("max_loss_per_position_pct", 8, 8),
        "MAX_TOTAL_EXPOSURE_PCT": ("max_total_exposure_pct", 50, 50),
        "DAILY_LOSS_CIRCUIT_BREAKER_PCT": ("daily_loss_circuit_breaker_pct", 12, 12),
        "MAX_CONCURRENT_POSITIONS": ("max_concurrent_positions", 3, 3),
        "MIN_BALANCE_RESERVE_PCT": ("min_balance_reserve_pct", 20, 20),
    }
    _using_dangerous_defaults = []
    for _env_key, (_cfg_key, _code_default, _safe_value) in _risk_defaults.items():
        # BUG-6 FIX: Check os.getenv(), not CONFIG.get().
        # config_loader always returns a default string — CONFIG.get() is never None,
        # so the old check never fired even when .env was missing or incomplete.
        if os.getenv(_env_key) is None:
            _using_dangerous_defaults.append(
                f"  {_env_key} not set — using code default {_code_default} (recommended: {_safe_value})"
            )
    if _using_dangerous_defaults:
        logging.critical(
            "[CONFIG] DANGEROUS: The following risk parameters are not set in .env and use "
            "permissive code defaults:\n%s\nSet these in .env before live trading.",
            "\n".join(_using_dangerous_defaults),
        )
        print("WARNING: Permissive risk defaults active. See bot.log for details.")

    hyperliquid = HyperliquidAPI()
    agent = TradingAgent(hyperliquid=hyperliquid)
    risk_mgr = RiskManager()
    state_mgr = TradeStateMachine()

    start_time = datetime.now(timezone.utc)
    invocation_count = 0
    trade_log = deque(maxlen=200)  # BUG-7 FIX: bounded — was an unbounded list, never read (Sharpe uses diary.jsonl)
    active_trades = load_active_trades()  # {'asset','is_long','amount','entry_price','tp_oid','sl_oid','exit_plan'}
    # PERF-v10-8 FIX: Use absolute paths so log rotation and writes land in the correct
    # directory regardless of CWD (matches the absolute-path fix in decision_maker.py).
    _MAIN_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    diary_path = os.path.join(_MAIN_PROJECT_ROOT, "diary.jsonl")
    decisions_path = os.path.join(_MAIN_PROJECT_ROOT, "decisions.jsonl")
    _llm_log_path = os.path.join(_MAIN_PROJECT_ROOT, "llm_requests.log")
    _prompts_log_path = os.path.join(_MAIN_PROJECT_ROOT, "prompts.log")
    _signals_path = os.path.join(_MAIN_PROJECT_ROOT, "signals.jsonl")
    initial_account_value = None
    # Perp mid-price history sampled each loop (authoritative, avoids spot/perp basis mismatch)
    price_history = {}

    print(f"Starting trading agent for assets: {args.assets} at interval: {args.interval}")

    def add_event(msg: str):
        logging.info(msg)

    async def _interruptible_sleep(seconds: float) -> None:
        """Sleep for `seconds` but wake within 5s if _shutdown is set by SIGTERM/SIGINT.

        Replaces bare asyncio.sleep() in long waits so the bot shuts down quickly
        instead of waiting up to 5 minutes for the inner-loop tick to complete.
        """
        elapsed = 0.0
        while elapsed < seconds and not _shutdown:
            chunk = min(5.0, seconds - elapsed)
            await asyncio.sleep(chunk)
            elapsed += chunk

    async def run_loop():
        """Main trading loop that gathers data, calls the agent, and executes trades."""
        global _shutdown  # V3-CRITICAL-1 FIX: without this, `_shutdown = True` inside the inner loop
        # creates a local variable, making every `if _shutdown` check raise UnboundLocalError.
        nonlocal invocation_count, initial_account_value

        # Pre-load meta cache for correct order sizing
        await hyperliquid.get_meta_and_ctxs()
        # Pre-load HIP-3 dex meta for any dex:asset in the asset list
        hip3_dexes = set()
        for a in args.assets:
            if ":" in a:
                hip3_dexes.add(a.split(":")[0])
        for dex in hip3_dexes:
            await hyperliquid.get_meta_and_ctxs(dex=dex)
            add_event(f"Loaded HIP-3 meta for dex: {dex}")

        # FL-1 FIX: Set exchange-side leverage for every asset before any order is placed.
        # MAX_LEVERAGE in .env only affected risk calculations — not the actual exchange leverage.
        _configured_leverage = int(float(CONFIG.get("max_leverage") or 5))
        for _lev_asset in args.assets:
            try:
                await hyperliquid.set_leverage(_lev_asset, _configured_leverage)
                add_event(f"[LEVERAGE] {_lev_asset} set to {_configured_leverage}x on exchange")
            except Exception as _lev_err:
                logging.error("[LEVERAGE] Failed to set leverage for %s: %s — ABORTING to avoid uncontrolled leverage", _lev_asset, _lev_err)
                add_event(f"[LEVERAGE] CRITICAL: could not set leverage for {_lev_asset} — bot halted. Fix and restart.")
                return  # Fail closed: do not trade with unknown leverage

        # ── Score-pipeline state (persist across cycles) ──────────────────────────
        # CRITICAL-7: Load daily count from disk to survive process restarts
        _today_init = datetime.now(timezone.utc).date()
        _saved_count, _saved_date = _load_daily_count()
        _daily_trade_count: int = _saved_count if _saved_date == str(_today_init) else 0
        _sl_cooldown_map: dict = {}           # asset -> datetime (blocked until)
        _last_daily_reset = _today_init       # set now so midnight reset works correctly
        # CRITICAL-2: Persistent OI history — last 3 readings per asset for oi_confirmed()
        from collections import deque as _deque
        _oi_history: dict = {}                # asset -> deque(maxlen=3) of OI float values
        _ai_verdict_cache: dict = {}          # asset → {verdict, fingerprint, expires_at}
        _macro_context_cache: dict = {}       # {events, headlines, fetched_at}
        _diary_index: dict = _load_diary_index()  # P3-M2: {asset: latest diary entry} for guardian O(1) lookup
        _outer_cycle_timestamp: float = 0.0  # time.monotonic() at outer cycle data fetch
        _last_ai_call_time: dict = {}         # asset → unix timestamp of last Claude call
        # ─────────────────────────────────────────────────────────────────────────

        # ── Trade-close logging helpers ───────────────────────────────────────────

        _stats_lock = asyncio.Lock()

        async def _update_stats(realized_pnl: float | None, exit_price: float | None,
                                qty: float | None, entry_price: float | None = None) -> None:
            """Atomically update stats.json after every trade close."""
            # LOW-1 FIX: Use absolute path (matches _MAIN_PROJECT_ROOT pattern for all other files).
            stats_path = os.path.join(_MAIN_PROJECT_ROOT, "stats.json")
            async with _stats_lock:
                try:
                    if os.path.exists(stats_path):
                        with open(stats_path) as _sf:
                            stats = json.load(_sf)
                    else:
                        stats = {"total_trades": 0, "wins": 0, "losses": 0,
                                 "win_rate": 0.0, "total_pnl": 0.0, "total_fees": 0.0}
                    stats["total_trades"] = stats.get("total_trades", 0) + 1
                    if realized_pnl is not None:
                        stats["total_pnl"] = round(stats.get("total_pnl", 0.0) + realized_pnl, 4)
                        if realized_pnl > 0:
                            stats["wins"] = stats.get("wins", 0) + 1
                        else:
                            stats["losses"] = stats.get("losses", 0) + 1
                        total = stats["total_trades"]
                        stats["win_rate"] = round(stats["wins"] / total, 4) if total > 0 else 0.0
                    # H-2 FIX: count both entry and exit fee legs (old code only counted exit).
                    # Backtest-validity FIX: read fee from CONFIG so stats stay correct if fee changes.
                    _fee_rate = float(CONFIG.get("taker_fee_pct") or 0.00045)
                    if exit_price and qty:
                        _exit_fee = exit_price * qty * _fee_rate
                        _entry_fee = (entry_price or exit_price) * qty * _fee_rate
                        stats["total_fees"] = round(
                            stats.get("total_fees", 0.0) + _exit_fee + _entry_fee, 4
                        )
                    _tmp = stats_path + ".tmp"
                    with open(_tmp, "w") as _sf:
                        json.dump(stats, _sf, indent=2)
                    os.replace(_tmp, stats_path)
                except Exception as _se:
                    logging.error("[STATS] update failed: %s", _se)

        async def _log_trade_close(tr: dict, exit_type: str,
                                    override_pnl: float | None = None,
                                    override_exit_price: float | None = None) -> None:
            """Write a trade_closed event to diary.jsonl and refresh stats.json.

            Fetches the closing fill from Hyperliquid when no override price is
            supplied. exit_type is refined from 'unknown' to 'tp'/'sl' by comparing
            the fill price against the trade's stored tp_price / sl_price.
            """
            asset = tr.get('asset', '')
            entry_price = float(tr.get('entry_price') or 0)
            qty = float(tr.get('amount') or 0)
            is_long = bool(tr.get('is_long', True))
            opened_at_str = tr.get('opened_at')
            tp_price_tr = tr.get('tp_price')
            sl_price_tr = tr.get('sl_price')

            now = datetime.now(timezone.utc)
            duration_minutes = None
            if opened_at_str:
                try:
                    _odt = datetime.fromisoformat(opened_at_str)
                    if _odt.tzinfo is None:
                        _odt = _odt.replace(tzinfo=timezone.utc)
                    duration_minutes = round((now - _odt).total_seconds() / 60, 1)
                except Exception:
                    pass

            exit_price = override_exit_price
            realized_pnl = override_pnl

            # Fetch closing fill when no price is already known
            _matching: list[dict] = []
            if exit_price is None and qty > 0:
                try:
                    _fills = await hyperliquid.get_recent_fills(limit=50)
                    _opened_ts = None
                    if opened_at_str:
                        try:
                            _opened_ts = datetime.fromisoformat(opened_at_str)
                            if _opened_ts.tzinfo is None:
                                _opened_ts = _opened_ts.replace(tzinfo=timezone.utc)
                        except Exception:
                            pass
                    for _fill in _fills:
                        if (_fill.get('coin') or _fill.get('asset')) != asset:
                            continue
                        # Must be after trade opened
                        _t_raw = _fill.get('time') or _fill.get('timestamp')
                        if _t_raw and _opened_ts:
                            try:
                                _t_int = int(_t_raw)
                                _fdt = datetime.fromtimestamp(
                                    _t_int / 1000 if _t_int > 1e12 else _t_int, tz=timezone.utc
                                )
                                if _fdt < _opened_ts:
                                    continue
                            except Exception:
                                pass
                        # LB-v10-3 FIX: get_recent_fills() normalizes to snake_case 'is_buy';
                        # the original API may return camelCase 'isBuy'. Use dual-lookup so
                        # neither normalization path silently mismatches every fill direction.
                        _fbuy = bool(_fill.get('is_buy', _fill.get('isBuy', False)))
                        if is_long and _fbuy:
                            continue   # long is closed by a sell fill
                        if not is_long and not _fbuy:
                            continue   # short is closed by a buy fill
                        _matching.append(_fill)
                    if _matching:
                        _tqty = sum(float(f.get('sz') or f.get('size') or 0) for f in _matching)
                        _tval = sum(
                            float(f.get('px') or f.get('price') or 0)
                            * float(f.get('sz') or f.get('size') or 0)
                            for f in _matching
                        )
                        if _tqty > 0:
                            exit_price = round(_tval / _tqty, 6)
                except Exception as _fe:
                    logging.warning("[PNL] fill lookup failed for %s: %s", asset, _fe)

            # Net P&L: gross move minus round-trip taker fees and funding cost
            if realized_pnl is None and exit_price and entry_price and qty:
                _gross = ((exit_price - entry_price) * qty if is_long
                          else (entry_price - exit_price) * qty)
                _fee = (entry_price + exit_price) * qty * 0.00045
                realized_pnl = round(_gross - _fee, 4)

            # Funding cost — paid every 8h; positive rate = longs pay, shorts receive
            if realized_pnl is not None and duration_minutes and entry_price and qty:
                _funding_rate = float(tr.get('funding_rate') or 0)
                if _funding_rate != 0:
                    _funding_intervals = (duration_minutes / 60.0) / 8.0
                    _funding_impact = entry_price * qty * _funding_rate * _funding_intervals
                    # Long: positive rate = cost; short: positive rate = income
                    realized_pnl = round(
                        realized_pnl - (_funding_impact if is_long else -_funding_impact), 4
                    )

            # OID-based exit type — more reliable than price proximity
            if exit_type == "unknown" and _matching:
                _tp_oid = tr.get('tp_oid')
                _sl_oid = tr.get('sl_oid')
                for _fill in _matching:
                    _fill_oid = str(_fill.get('oid') or _fill.get('orderId') or '')
                    if _fill_oid and _tp_oid and _fill_oid == str(_tp_oid):
                        exit_type = 'tp'
                        break
                    elif _fill_oid and _sl_oid and _fill_oid == str(_sl_oid):
                        exit_type = 'sl'
                        break

            # Refine exit_type from 'unknown' when prices are available (fallback to price proximity)
            if exit_type == "unknown" and exit_price is not None:
                if is_long:
                    if tp_price_tr and exit_price >= float(tp_price_tr) * 0.999:
                        exit_type = "tp"
                    elif sl_price_tr and exit_price <= float(sl_price_tr) * 1.001:
                        exit_type = "sl"
                else:
                    if tp_price_tr and exit_price <= float(tp_price_tr) * 1.001:
                        exit_type = "tp"
                    elif sl_price_tr and exit_price >= float(sl_price_tr) * 0.999:
                        exit_type = "sl"

            # Per-asset SL cooldown — block re-entry after a confirmed stop-loss or forced loss
            if exit_type in ("sl", "force") and (realized_pnl is None or realized_pnl <= 0):
                _cd_mins = int(CONFIG.get("cooldown_minutes") or 60)
                _sl_cooldown_map[asset] = datetime.now(timezone.utc) + timedelta(minutes=_cd_mins)
                logging.info("[COOLDOWN] %s blocked %d min after %s exit", asset, _cd_mins, exit_type)

            _close_event = {
                "timestamp": now.isoformat(),
                "event": "trade_closed",
                "asset": asset,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "qty": qty,
                "is_long": is_long,
                "realized_pnl": realized_pnl,
                "exit_type": exit_type,
                "duration_minutes": duration_minutes,
                "tp_oid": tr.get('tp_oid'),
                "sl_oid": tr.get('sl_oid'),
            }
            try:
                with open(diary_path, "a") as _df:
                    _df.write(json.dumps(_close_event) + "\n")
            except Exception as _we:
                logging.error("[PNL] diary write failed: %s", _we)

            await _update_stats(realized_pnl, exit_price, qty, entry_price)
            logging.info(
                "[TRADE CLOSE] %s exit_type=%s exit_px=%s pnl=%s duration=%smin",
                asset, exit_type, exit_price, realized_pnl, duration_minutes,
            )

        # ─────────────────────────────────────────────────────────────────────────

        _interval_seconds = get_interval_seconds(args.interval)
        _consecutive_failures = 0
        _MAX_CONSECUTIVE_FAILURES = 5
        _KILLSWITCH_FILE = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "KILLSWITCH"
        )
        while True:
            if _shutdown:
                logging.info("[SHUTDOWN] Signal received — exiting loop cleanly")
                break
            # C-7 FIX: File-based remote kill switch — create a file named KILLSWITCH
            # in the project root to halt all trading without SSH access.
            if os.path.exists(os.path.normpath(_KILLSWITCH_FILE)):
                logging.critical("[KILLSWITCH] KILLSWITCH file detected — halting all trading. Remove the file to re-enable.")
                add_event("[KILLSWITCH] Trading halted by KILLSWITCH file. Remove it to restart.")
                # BUG-P9-2 FIX: Close all open positions and cancel orders before exiting.
                # Previous code only broke the loop — positions kept running with no monitoring.
                # BUG-v7-O1 FIX: Retry each close up to 3 times. If any position cannot be closed,
                # do NOT exit — keep the bot running so the guardian continues protecting it.
                # A silent exit with an open leveraged position is worse than staying up.
                _ks_unclosed = []
                for _ks_tr in list(active_trades):
                    _ks_asset = _ks_tr.get("asset")
                    if not _ks_asset:
                        continue
                    _ks_closed = False
                    for _ks_attempt in range(3):
                        try:
                            await hyperliquid.cancel_all_orders(_ks_asset)
                            await hyperliquid.market_close(_ks_asset)
                            logging.critical("[KILLSWITCH] %s — orders cancelled and position closed", _ks_asset)
                            add_event(f"[KILLSWITCH] {_ks_asset} closed successfully")
                            _ks_closed = True
                            break
                        except Exception as _ks_err:
                            logging.critical("[KILLSWITCH] %s — close attempt %d/3 failed: %s", _ks_asset, _ks_attempt + 1, _ks_err)
                            if _ks_attempt < 2:
                                import asyncio as _ks_asyncio
                                await _ks_asyncio.sleep(2)
                    if not _ks_closed:
                        _ks_unclosed.append(_ks_asset)
                        logging.critical("[KILLSWITCH] %s — ALL 3 CLOSE ATTEMPTS FAILED — position remains open — bot staying alive to monitor", _ks_asset)
                        add_event(f"[KILLSWITCH] {_ks_asset} CLOSE FAILED x3 — MANUAL INTERVENTION REQUIRED")
                if _ks_unclosed:
                    _ks_msg = (
                        f"\U0001f6a8 KILLSWITCH: {len(active_trades) - len(_ks_unclosed)} closed, "
                        f"{len(_ks_unclosed)} FAILED TO CLOSE: {', '.join(_ks_unclosed)}. "
                        f"Bot staying alive to monitor unclosed positions. Manual close required."
                    )
                    await send_alert(_ks_msg)
                    logging.critical("[KILLSWITCH] staying alive — unclosed positions need manual action: %s", _ks_unclosed)
                    # Do NOT break — continue monitoring the unclosed positions
                    continue
                else:
                    _ks_msg = f"\U0001f6a8 KILLSWITCH activated — closed {len(active_trades)} position(s). Remove the KILLSWITCH file to restart."
                    await send_alert(_ks_msg)
                    break
            cycle_start = time.monotonic()
            _outer_cycle_timestamp = cycle_start  # inner ticks read this to check higher-TF freshness
            # C-2 FIX: Do NOT clear the entire AI verdict cache on every outer cycle.
            # Cache entries already carry `expires_at` timestamps and expire naturally.
            # Clearing here negated the 60-minute APPROVE cache, doubling Claude call frequency.
            # _ai_verdict_cache.clear()  ← REMOVED
            invocation_count += 1

            # V6-LOW-2: Re-assert exchange leverage once per outer cycle (hourly).
            # Guards against an operator manually raising leverage in the Hyperliquid
            # web UI mid-session — the risk_manager notional check would still pass
            # but the exchange multiplier would be wrong until restart.
            _lev_cycle = int(float(CONFIG.get("max_leverage") or 5))
            for _lv_asset in args.assets:
                try:
                    await hyperliquid.set_leverage(_lv_asset, _lev_cycle)
                except Exception as _lv_err:
                    logging.warning("[LEVERAGE] re-verify failed for %s: %s — will retry next cycle", _lv_asset, _lv_err)

            # UTC midnight reset of daily trade counter
            _today_utc = datetime.now(timezone.utc).date()
            if _last_daily_reset != _today_utc:
                _daily_trade_count = 0
                _last_daily_reset = _today_utc
                _save_daily_count(0, str(_today_utc))
                logging.info("[DAILY] trade counter reset for %s", _today_utc)

            minutes_since_start = (datetime.now(timezone.utc) - start_time).total_seconds() / 60

            # Global account state — wrap so a sustained API outage skips the cycle
            # instead of propagating an unhandled exception out of run_loop() and
            # killing the process while positions remain open on the exchange.
            try:
                state = await hyperliquid.get_user_state()
            except Exception as _state_err:
                _consecutive_failures += 1
                logging.error(
                    "[LOOP] get_user_state failed (%d/%d) — skipping cycle %d: %s",
                    _consecutive_failures, _MAX_CONSECUTIVE_FAILURES,
                    invocation_count, _state_err,
                )
                add_event(f"[LOOP] API error ({_consecutive_failures}/{_MAX_CONSECUTIVE_FAILURES}), skipping cycle: {_state_err}")
                if _consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                    logging.critical(
                        "[LOOP] Circuit breaker triggered after %d consecutive failures — sleeping 5 minutes",
                        _consecutive_failures,
                    )
                    add_event(f"[LOOP] Circuit breaker: sleeping 5 minutes after {_consecutive_failures} failures")
                    await send_alert(f"⚠️ API circuit breaker: {_consecutive_failures} consecutive failures — sleeping 5 min before retry.")
                    await _interruptible_sleep(300)
                    _consecutive_failures = 0
                else:
                    await _interruptible_sleep(_interval_seconds)
                continue
            total_value = state.get('total_value') or (state.get('balance', 0) + sum(p.get('pnl', 0) for p in state.get('positions', [])))
            sharpe = calculate_sharpe_from_diary(diary_path)

            account_value = total_value
            # FL-4 FIX: Persist initial_account_value across restarts.
            # Old code: re-initialized to current balance on every restart, silently lowering
            # the reserve floor after a loss cycle. Now loads from disk and never resets downward.
            if initial_account_value is None:
                _loaded = _load_initial_balance()
                if _loaded is not None:
                    initial_account_value = _loaded
                    logging.info("[INIT] Loaded initial_account_value=%.2f from disk", initial_account_value)
                else:
                    initial_account_value = account_value
                    _save_initial_balance(initial_account_value)
            total_return_pct = ((account_value - initial_account_value) / initial_account_value * 100.0) if initial_account_value else 0.0

            # E-4 / V3-HIGH-1 FIX: Rotate unbounded log files before they fill the disk.
            _rotate_if_needed(diary_path)
            _rotate_if_needed(decisions_path)
            _rotate_if_needed(_llm_log_path)    # PERF-v10-8: was relative path
            _rotate_if_needed(_prompts_log_path)  # PERF-v10-8: was relative path
            _rotate_if_needed(_signals_path)     # BUG-v7-P5 FIX: was omitted; can grow to GB without rotation

            positions = []
            for pos_wrap in state.get('positions', []):
                pos = pos_wrap
                coin = pos.get('coin')
                try:
                    current_px = await hyperliquid.get_current_price(coin) if coin else None
                except Exception:
                    current_px = None
                positions.append({
                    "symbol": coin,
                    "quantity": round_or_none(pos.get('szi'), 6),
                    "entry_price": round_or_none(pos.get('entryPx'), 2),
                    "current_price": round_or_none(current_px, 2),
                    "liquidation_price": round_or_none(pos.get('liquidationPx') or pos.get('liqPx'), 2),
                    "unrealized_pnl": round_or_none(pos.get('pnl'), 4),
                    "leverage": pos.get('leverage')
                })

            # --- RISK: Force-close positions that exceed max loss ---
            try:
                # LB-v10-2 FIX: Alert on positions where price lookup returned 0 (pnl_unknown).
                # check_losing_positions() sees pnl=0 for these and never triggers — alert operator.
                for _pnl_unk in state.get('positions', []):
                    if _pnl_unk.get("pnl_unknown"):
                        _unk_coin = _pnl_unk.get("coin", "?")
                        logging.warning(
                            "[FORCE-CLOSE BLIND] %s price=0 after all retries — cannot assess loss. MANUAL CHECK REQUIRED.",
                            _unk_coin,
                        )
                        await send_alert(
                            f"🚨 [FORCE-CLOSE BLIND] {_unk_coin} price lookup failed — loss unknown, "
                            f"force-close protection DISABLED. Check position manually on Hyperliquid."
                        )
                positions_to_close = risk_mgr.check_losing_positions(state.get('positions', []))
                for ptc in positions_to_close:
                    coin = ptc["coin"]
                    add_event(f"RISK FORCE-CLOSE: {coin} at {ptc['loss_pct']}% loss (PnL: ${ptc['pnl']})")
                    try:
                        # Use market_close — bypasses _order_retry's idempotency pre-flight
                        # which would otherwise find the SL trigger order and silently skip
                        # the force-close, leaving the position open past the loss threshold.
                        await hyperliquid.market_close(coin)
                        await hyperliquid.cancel_all_orders(coin)
                        # LB-v10-6 FIX: Transition to COOLDOWN (not IDLE) so the bot cannot
                        # immediately re-enter on the same adverse trend that triggered force-close.
                        state_mgr.start_cooldown(
                            coin,
                            interval_seconds=int(CONFIG.get("cooldown_minutes") or 30) * 60,
                        )
                        # Remove from active trades and log the close event
                        for tr in active_trades[:]:
                            if tr.get('asset') == coin:
                                active_trades.remove(tr)
                                save_active_trades(active_trades)
                                await _log_trade_close(
                                    tr, "force",
                                    override_pnl=float(ptc["pnl"]),
                                )
                    except Exception as fc_err:
                        add_event(f"Force-close error for {coin}: {fc_err}")
            except Exception as risk_err:
                add_event(f"Risk check error: {risk_err}")

            recent_diary = []
            try:
                with open(diary_path, "r") as f:
                    lines = f.readlines()
                    for line in lines[-10:]:
                        entry = json.loads(line)
                        recent_diary.append(entry)
            except Exception:
                pass

            open_orders_struct = []
            open_orders_ok = False  # only True when fetch succeeded — gates the guardian
            try:
                open_orders = await hyperliquid.get_open_orders()
                open_orders_ok = True
                for o in open_orders[:50]:
                    open_orders_struct.append({
                        "coin": o.get('coin'),
                        "oid": o.get('oid'),
                        "is_buy": o.get('isBuy'),
                        "size": round_or_none(o.get('sz'), 6),
                        "price": round_or_none(o.get('px'), 2),
                        "trigger_price": round_or_none(o.get('triggerPx'), 2),
                        "order_type": o.get('orderType')
                    })
            except Exception as e:
                logging.warning("Failed to fetch open orders: %s", e)
                open_orders = []

            # Reconcile active trades
            try:
                assets_with_positions = set()
                for pos in state.get('positions', []):
                    try:
                        if abs(float(pos.get('szi') or 0)) > 0:
                            assets_with_positions.add(pos.get('coin'))
                    except Exception as e:
                        logging.warning("Skipped malformed position in reconcile: %s", e)
                        continue
                assets_with_orders = {o.get('coin') for o in (open_orders or []) if o.get('coin')}
                for tr in active_trades[:]:
                    asset = tr.get('asset')
                    if asset not in assets_with_positions and asset not in assets_with_orders:
                        add_event(f"Reconciling stale active trade for {asset} (no position, no orders)")
                        active_trades.remove(tr)
                        save_active_trades(active_trades)
                        # Reset state machine so the asset can trade again next cycle.
                        # Without this, the state stays ENTERED indefinitely and the
                        # state gate blocks all future entries for up to 13 hours.
                        state_mgr.start_cooldown(asset, interval_seconds=3600)
                        state_mgr.clear_entry(asset)
                        logging.info("[RECONCILE] %s — position closed naturally, cooldown started", asset)
                        # Use pending_exit_type if the timeout handler set it; otherwise
                        # let _log_trade_close resolve tp/sl from the fill price.
                        _recon_exit_type = tr.get('pending_exit_type', 'unknown')
                        await _log_trade_close(tr, _recon_exit_type)
                # BUG-P11-1 FIX: TP1 hit detection — set tp1_hit=True and halve tracked amount
                # when TP1 order disappears from open orders but position still exists.
                # Without this, tp1_hit stays False forever and the trailing stop guardian
                # continues placing SL for the full original size after 50% has been closed.
                for _tp1_tr in active_trades:
                    _tp1_tr_asset = _tp1_tr.get("asset")
                    _tp1_tr_oid   = _tp1_tr.get("tp1_oid")
                    if not _tp1_tr_asset or not _tp1_tr_oid or _tp1_tr.get("tp1_hit"):
                        continue  # no TP1 OID, or already marked hit
                    _tp1_still_open = any(
                        str(o.get("oid")) == str(_tp1_tr_oid) for o in (open_orders or [])
                    )
                    if _tp1_still_open:
                        continue  # TP1 order still live
                    if _tp1_tr_asset not in assets_with_positions:
                        continue  # full exit — handled by reconcile loop above
                    # LB-v10-5 FIX: If price gapped through both TP1 and TP2 in one candle,
                    # the position is already fully flat. Check TP2 OID too — if both are
                    # gone and no position exists, the reconcile loop handles the full close.
                    # Only mark tp1_hit when position truly still exists (already guarded above).
                    _tp2_tr_oid = _tp1_tr.get("tp2_oid")
                    _tp2_still_open = _tp2_tr_oid and any(
                        str(o.get("oid")) == str(_tp2_tr_oid) for o in (open_orders or [])
                    )
                    if not _tp2_still_open and _tp2_tr_oid:
                        # TP2 OID is also gone → both filled simultaneously.
                        # Don't mark tp1_hit; reconcile loop already removed this from active_trades.
                        logging.info(
                            "[TP1+TP2 RACE] %s both TP orders gone simultaneously — full close handled by reconcile",
                            _tp1_tr_asset,
                        )
                        continue
                    # TP1 filled: partial close of 50%. Update tracked amount for guardian.
                    _new_amount = float(_tp1_tr.get("half_size") or float(_tp1_tr.get("amount", 0)) / 2)
                    _tp1_tr["tp1_hit"]  = True
                    _tp1_tr["amount"]   = _new_amount
                    _tp1_tr["tp1_oid"]  = None  # cleared — no longer in open orders
                    save_active_trades(active_trades)
                    logging.info(
                        "[TP1 HIT] %s TP1 filled — amount reduced to %.6f for trailing stop guardian",
                        _tp1_tr_asset, _new_amount
                    )
                    add_event(f"[TP1 HIT] {_tp1_tr_asset} TP1 filled — trailing stop now covers remaining 50%")

            except Exception as _rec_err:
                # C-6 FIX: Log reconcile errors instead of silently swallowing them.
                # A bare `pass` left stale ENTERED state that blocked future entries for 13h.
                logging.error("[RECONCILE] exception — stale state may persist: %s\n%s",
                              _rec_err, traceback.format_exc())
                add_event(f"[RECONCILE] ERROR: {_rec_err} — check bot.log, may need manual state.json edit")

            # Time-based exit — force-close trades stuck beyond max_trade_hours
            for _asset_name in list(args.assets):
                if state_mgr.get_state(_asset_name) == "ENTERED":
                    _max_hours = int(CONFIG.get("max_trade_hours") or 12)
                    if state_mgr.is_trade_expired(_asset_name, _max_hours):
                        # Compute actual age for alert message using opened_at from active_trades
                        _timeout_age_str = f"{_max_hours}h"
                        for _atr_t in active_trades:
                            if _atr_t.get("asset") == _asset_name and _atr_t.get("opened_at"):
                                try:
                                    _atr_opened = datetime.fromisoformat(_atr_t["opened_at"].replace("Z", "+00:00"))
                                    _atr_age_h = (datetime.now(timezone.utc) - _atr_opened).total_seconds() / 3600
                                    _timeout_age_str = f"{_atr_age_h:.1f}h"
                                except (ValueError, TypeError):
                                    pass
                                break
                        add_event(
                            f"[TIMEOUT] {_asset_name} force-closing "
                            f"after {_timeout_age_str} — no progress (max={_max_hours}h)"
                        )
                        await send_alert(
                            f"⏰ [MAX DURATION] {_asset_name} — trade open {_timeout_age_str}, auto-closing (max={_max_hours}h)"
                        )
                        try:
                            await hyperliquid.market_close(_asset_name)
                            # BUG-3 FIX: Cancel orphaned TP/SL trigger orders after timeout close.
                            # Without this, trigger orders kept the asset in assets_with_orders,
                            # preventing the reconciler from clearing stale active_trade entries.
                            await hyperliquid.cancel_all_orders(_asset_name)
                            state_mgr.start_cooldown(_asset_name, interval_seconds=3600)
                            # Flag so the reconciler logs the close with the right exit_type
                            for _tr in active_trades:
                                if _tr.get('asset') == _asset_name:
                                    _tr['pending_exit_type'] = 'timeout'
                        except Exception as _te:
                            add_event(f"[TIMEOUT] {_asset_name} close error: {_te}")
                            await send_alert(f"🚨 [MAX DURATION FAIL] {_asset_name} could not auto-close: {_te}")

            # TP/SL GUARDIAN — re-place missing trigger orders for every ENTERED position.
            # TP/SL are placed once at entry; if the exchange dropped the order (rate-limit,
            # connection reset, race), the position runs naked until this catches it.
            for _g_asset in list(args.assets):
                if not open_orders_ok:
                    # open_orders fetch failed — stale [] would make every ENTERED asset
                    # appear to have no TP/SL, triggering mass duplicate order placement.
                    logging.warning("[GUARDIAN] skipping all assets — open_orders fetch failed, cannot safely re-place")
                    break
                if state_mgr.get_state(_g_asset) != "ENTERED":
                    continue
                # Only act if the position still exists on the exchange
                _g_pos_exists = any(
                    abs(float(p.get('szi') or 0)) > 0 and p.get('coin') == _g_asset
                    for p in state.get('positions', [])
                )
                if not _g_pos_exists:
                    # Position gone but state still ENTERED — reset it now.
                    # This covers any path the reconciler missed (e.g. first cycle after restart).
                    logging.info("[GUARDIAN] %s state=ENTERED but no live position — resetting to COOLDOWN", _g_asset)
                    add_event(f"[GUARDIAN] {_g_asset} position gone, resetting state to COOLDOWN")
                    state_mgr.start_cooldown(_g_asset, interval_seconds=3600)
                    continue
                # Classify existing trigger orders for this asset
                _g_has_tp = False
                _g_has_sl = False
                for _g_o in (open_orders or []):
                    if _g_o.get('coin') != _g_asset:
                        continue
                    _g_ot = _g_o.get('orderType')
                    if isinstance(_g_ot, dict):
                        _g_tpsl = (_g_ot.get('trigger') or {}).get('tpsl', '')
                        if _g_tpsl == 'tp':
                            _g_has_tp = True
                        elif _g_tpsl == 'sl':
                            _g_has_sl = True
                if _g_has_tp and _g_has_sl:
                    continue  # Both present — nothing to do
                # P3-M2: Read TP/SL prices from the per-asset diary index (O(1), not O(n) full scan).
                # Index is updated every time a trade is written to diary.jsonl. Falls back to
                # a tail scan of diary.jsonl only if the index is missing (e.g. first run after upgrade).
                _g_diary = _diary_index.get(_g_asset)
                if not _g_diary:
                    try:
                        with open(diary_path, 'r') as _gf:
                            for _gl in reversed(_gf.readlines()[-500:]):
                                try:
                                    _ge = json.loads(_gl)
                                    if _ge.get('asset') == _g_asset and _ge.get('action') in ('buy', 'sell'):
                                        _g_diary = _ge
                                        _diary_index[_g_asset] = _ge  # back-fill index for next cycle
                                        _save_diary_index(_diary_index)
                                        break
                                except Exception:
                                    continue
                    except Exception:
                        pass
                if not _g_diary:
                    logging.warning("[GUARDIAN] %s: no diary entry — attempting fallback SL from live price", _g_asset)
                    if not _g_has_sl:
                        _g_pos = next(
                            (p for p in state.get('positions', [])
                             if p.get('coin') == _g_asset and abs(float(p.get('szi') or 0)) > 0),
                            None,
                        )
                        if _g_pos:
                            _g_szi     = float(_g_pos.get('szi') or 0)
                            _g_fb_long = _g_szi > 0
                            _g_fb_size = abs(_g_szi)
                            _g_fb_px   = asset_prices.get(_g_asset) or 0
                            if _g_fb_px > 0 and _g_fb_size > 0:
                                _g_fb_sl = risk_mgr.enforce_stop_loss(None, _g_fb_px, _g_fb_long)
                                try:
                                    _g_fb_res = await hyperliquid.place_stop_loss(_g_asset, _g_fb_long, _g_fb_size, _g_fb_sl)
                                    _g_fb_oid = (hyperliquid.extract_oids(_g_fb_res) or [None])[0]
                                    add_event(f"[GUARDIAN] {_g_asset} fallback SL placed at {_g_fb_sl} (no diary, size={_g_fb_size:.6f}, oid={_g_fb_oid})")
                                    logging.warning("[GUARDIAN] %s fallback SL placed at %.2f (no diary)", _g_asset, _g_fb_sl)
                                    for _gtr in active_trades:
                                        if _gtr.get('asset') == _g_asset:
                                            _gtr['sl_oid'] = _g_fb_oid
                                    save_active_trades(active_trades)
                                except Exception as _g_fb_err:
                                    add_event(f"[GUARDIAN] {_g_asset} fallback SL failed: {_g_fb_err}")
                    continue
                _g_is_long  = _g_diary.get('action') == 'buy'
                # BUG-v7-L1 FIX: Use active_trades amount (halved post-TP1) not diary amount (full original).
                # After TP1 fills, active_trades["amount"] is halved by the BUG-P11-1 fix.
                # The diary still records the original full amount at entry time.
                # Placing a reduce-only SL for the full amount against a half-size position
                # causes Hyperliquid to reject the order, leaving the position naked.
                _g_active_tr = next((t for t in active_trades if t.get('asset') == _g_asset), None)
                _g_amount   = float((_g_active_tr or {}).get('amount') or _g_diary.get('amount') or 0)
                _g_tp_px    = _g_diary.get('tp_price')
                _g_tp1_px   = _g_diary.get('tp1_price')
                _g_tp2_px   = _g_diary.get('tp2_price')
                _g_sl_px    = _g_diary.get('sl_price')
                _g_half     = round(_g_amount / 2, 6)
                if _g_amount <= 0:
                    logging.warning("[GUARDIAN] %s: zero amount in diary — cannot re-place", _g_asset)
                    continue
                # LB-v10-1 FIX: Count live TP orders and use tp1_hit state to determine
                # the EXPECTED count so TP1 and TP2 are re-placed independently.
                # Old code: single _g_has_tp1 flag → TP2 permanently orphaned when TP1
                # is present but TP2 was dropped (e.g. connection reset between two places).
                # Fix: expected TPs = 2 before TP1 fires, = 1 after TP1 fires.
                # Re-place TP1 only when tp1 not yet hit AND no TPs present.
                # Re-place TP2 independently whenever live count < expected count.
                _g_tp_count = sum(
                    1 for o in (open_orders or [])
                    if o.get('coin') == _g_asset
                    and isinstance(o.get('orderType'), dict)
                    and (o.get('orderType', {}).get('trigger') or {}).get('tpsl') == 'tp'
                )
                _g_tp1_hit = (_g_active_tr or {}).get('tp1_hit', False)
                _g_expected_tp_count = 1 if _g_tp1_hit else 2
                _g_has_tp1 = _g_tp1_hit or _g_tp_count >= 1  # TP1 satisfied if already hit or still open
                _g_has_tp2 = _g_tp_count >= _g_expected_tp_count  # TP2 present iff live count meets expectation
                if not _g_has_tp1 and _g_tp1_px and _g_half > 0:
                    try:
                        _g_tp1_res = await hyperliquid.place_take_profit(_g_asset, _g_is_long, _g_half, float(_g_tp1_px))
                        _g_tp1_oid = (hyperliquid.extract_oids(_g_tp1_res) or [None])[0]
                        add_event(f"[GUARDIAN] {_g_asset} TP1 re-placed at {_g_tp1_px} size={_g_half:.6f} (oid={_g_tp1_oid})")
                        for _gtr in active_trades:
                            if _gtr.get('asset') == _g_asset:
                                _gtr['tp1_oid'] = _g_tp1_oid
                        save_active_trades(active_trades)
                    except Exception as _g_tp1_err:
                        add_event(f"[GUARDIAN] {_g_asset} TP1 re-place failed: {_g_tp1_err}")
                if not _g_has_tp2 and _g_tp2_px and _g_half > 0:
                    try:
                        _g_tp2_res = await hyperliquid.place_take_profit(_g_asset, _g_is_long, _g_half, float(_g_tp2_px))
                        _g_tp2_oid = (hyperliquid.extract_oids(_g_tp2_res) or [None])[0]
                        add_event(f"[GUARDIAN] {_g_asset} TP2 re-placed at {_g_tp2_px} size={_g_half:.6f} (oid={_g_tp2_oid})")
                        for _gtr in active_trades:
                            if _gtr.get('asset') == _g_asset:
                                _gtr['tp2_oid'] = _g_tp2_oid
                        save_active_trades(active_trades)
                    except Exception as _g_tp2_err:
                        add_event(f"[GUARDIAN] {_g_asset} TP2 re-place failed: {_g_tp2_err}")
                # Fallback: single TP (used when tp1/tp2 were never set)
                if not _g_has_tp and not _g_tp1_px and _g_tp_px:
                    try:
                        _g_tp_res = await hyperliquid.place_take_profit(_g_asset, _g_is_long, _g_amount, float(_g_tp_px))
                        _g_tp_oid = (hyperliquid.extract_oids(_g_tp_res) or [None])[0]
                        add_event(f"[GUARDIAN] {_g_asset} TP re-placed at {_g_tp_px} (oid={_g_tp_oid})")
                        for _gtr in active_trades:
                            if _gtr.get('asset') == _g_asset:
                                _gtr['tp_oid'] = _g_tp_oid
                        save_active_trades(active_trades)
                    except Exception as _g_err:
                        add_event(f"[GUARDIAN] {_g_asset} TP re-place failed: {_g_err}")
                if not _g_has_sl and _g_sl_px:
                    try:
                        _g_sl_res = await hyperliquid.place_stop_loss(_g_asset, _g_is_long, _g_amount, float(_g_sl_px))
                        _g_sl_oid = (hyperliquid.extract_oids(_g_sl_res) or [None])[0]
                        add_event(f"[GUARDIAN] {_g_asset} SL re-placed at {_g_sl_px} (oid={_g_sl_oid})")
                        for _gtr in active_trades:
                            if _gtr.get('asset') == _g_asset:
                                _gtr['sl_oid'] = _g_sl_oid
                        save_active_trades(active_trades)
                    except Exception as _g_err:
                        add_event(f"[GUARDIAN] {_g_asset} SL re-place failed: {_g_err}")

            # ── Trailing stop guardian ─────────────────────────────────────────────
            # Runs every outer loop tick for all ENTERED positions.
            # +1.0×ATR from entry → move SL to breakeven
            # +1.5×ATR from entry → trail SL at current_price − 0.5×ATR (buy) / + 0.5×ATR (sell)
            for _tr in active_trades:
                _tr_asset = _tr.get("asset")
                _tr_entry = float(_tr.get("entry_price") or 0)
                _tr_atr   = float(_tr.get("entry_atr") or 0)
                _tr_long  = _tr.get("is_long", True)
                _tr_size  = float(_tr.get("amount") or 0)

                if not _tr_asset or _tr_entry <= 0 or _tr_atr <= 0:
                    continue
                if _tr_size <= 0:
                    # BUG-v9-L1 FIX: amount can round to zero after TP1 halves it on small positions.
                    # Query the exchange for the live position size as a fallback before giving up —
                    # without this the position runs permanently without a trailing stop after TP1.
                    try:
                        _live_positions = await hyperliquid.get_positions()
                        _live_pos = next(
                            (p for p in (_live_positions or []) if p.get("coin") == _tr_asset),
                            None,
                        )
                        _live_sz = abs(float((_live_pos or {}).get("szi") or 0))
                        if _live_sz > 0:
                            _tr_size = _live_sz
                            _tr["amount"] = _live_sz  # repair active_trades entry
                            logging.info("[TRAIL] %s recovered live size %.6f from exchange — trailing stop proceeding", _tr_asset, _live_sz)
                        else:
                            logging.warning("[TRAIL] %s skipped — amount zero and no live position found on exchange", _tr_asset)
                            continue
                    except Exception as _tse:
                        logging.warning("[TRAIL] %s skipped — amount zero and exchange lookup failed: %s", _tr_asset, _tse)
                        continue

                _cur_px = float(asset_prices.get(_tr_asset) or 0)
                if _cur_px <= 0:
                    continue

                _move = (_cur_px - _tr_entry) if _tr_long else (_tr_entry - _cur_px)

                # Stage 1: Breakeven — price moved +1×ATR from entry
                if not _tr.get("trail_breakeven_done") and _move >= _tr_atr:
                    _be_sl = _tr_entry  # move SL to entry price (breakeven)
                    try:
                        # P5-MEDIUM-2: place new SL BEFORE cancelling old one — avoids unprotected window
                        _be_resp = await hyperliquid.place_stop_loss(_tr_asset, _tr_long, _tr_size, _be_sl)
                        _be_oids = hyperliquid.extract_oids(_be_resp)
                        if _tr.get("sl_oid"):
                            try:
                                await hyperliquid.cancel_order(_tr_asset, _tr["sl_oid"])
                            except Exception as _be_ce:
                                logging.warning("[TRAIL] %s could not cancel old SL %s: %s", _tr_asset, _tr["sl_oid"], _be_ce)
                        _tr["sl_price"] = _be_sl
                        _tr["sl_oid"] = _be_oids[0] if _be_oids else _tr.get("sl_oid")
                        _tr["trail_breakeven_done"] = True
                        save_active_trades(active_trades)
                        logging.info("[TRAIL] %s breakeven — SL moved to entry %.4f", _tr_asset, _be_sl)
                        add_event(f"[TRAIL] {_tr_asset} SL moved to breakeven {_be_sl:.4f}")
                    except Exception as _te:
                        logging.warning("[TRAIL] %s breakeven failed: %s", _tr_asset, _te)

                # Stage 2: Trailing — price moved +1.5×ATR, trail SL 0.5×ATR behind
                if _tr.get("trail_breakeven_done") and _move >= 1.5 * _tr_atr:
                    _trail_sl = (
                        round(_cur_px - 0.5 * _tr_atr, 6) if _tr_long
                        else round(_cur_px + 0.5 * _tr_atr, 6)
                    )
                    _cur_sl = float(_tr.get("sl_price") or 0)
                    _improves = (_tr_long and _trail_sl > _cur_sl) or (not _tr_long and _trail_sl < _cur_sl)
                    if _improves:
                        try:
                            # P5-MEDIUM-2: place new SL BEFORE cancelling old one — avoids unprotected window
                            _tsl_resp = await hyperliquid.place_stop_loss(_tr_asset, _tr_long, _tr_size, _trail_sl)
                            _tsl_oids = hyperliquid.extract_oids(_tsl_resp)
                            if _tr.get("sl_oid"):
                                try:
                                    await hyperliquid.cancel_order(_tr_asset, _tr["sl_oid"])
                                except Exception as _tsl_ce:
                                    logging.warning("[TRAIL] %s could not cancel old SL %s: %s", _tr_asset, _tr["sl_oid"], _tsl_ce)
                            _tr["sl_price"] = _trail_sl
                            _tr["sl_oid"] = _tsl_oids[0] if _tsl_oids else _tr.get("sl_oid")
                            _tr["trail_active"] = True
                            save_active_trades(active_trades)
                            logging.info("[TRAIL] %s trailing SL → %.4f", _tr_asset, _trail_sl)
                        except Exception as _te:
                            logging.warning("[TRAIL] %s trailing update failed: %s", _tr_asset, _te)
            # ──────────────────────────────────────────────────────────────────────

            recent_fills_struct = []
            try:
                fills = await hyperliquid.get_recent_fills(limit=50)
                for f_entry in fills[-20:]:
                    try:
                        t_raw = f_entry.get('time') or f_entry.get('timestamp')
                        timestamp = None
                        if t_raw is not None:
                            try:
                                t_int = int(t_raw)
                                if t_int > 1e12:
                                    timestamp = datetime.fromtimestamp(t_int / 1000, tz=timezone.utc).isoformat()
                                else:
                                    timestamp = datetime.fromtimestamp(t_int, tz=timezone.utc).isoformat()
                            except Exception:
                                timestamp = str(t_raw)
                        recent_fills_struct.append({
                            "timestamp": timestamp,
                            "coin": f_entry.get('coin') or f_entry.get('asset'),
                            "is_buy": f_entry.get('isBuy'),
                            "size": round_or_none(f_entry.get('sz') or f_entry.get('size'), 6),
                            "price": round_or_none(f_entry.get('px') or f_entry.get('price'), 2)
                        })
                    except Exception:
                        continue
            except Exception:
                pass

            dashboard = {
                "total_return_pct": round(total_return_pct, 2),
                "balance": round_or_none(state['balance'], 2),
                "account_value": round_or_none(account_value, 2),
                "perps_value": round_or_none(state.get('perps_value'), 2),
                "spot_usdc": round_or_none(state.get('spot_usdc'), 2),
                "sharpe_ratio": round_or_none(sharpe, 3),
                "positions": positions,
                "active_trades": [
                    {
                        "asset": tr.get('asset'),
                        "is_long": tr.get('is_long'),
                        "amount": round_or_none(tr.get('amount'), 6),
                        "entry_price": round_or_none(tr.get('entry_price'), 2),
                        "tp_oid": tr.get('tp_oid'),
                        "sl_oid": tr.get('sl_oid'),
                        "exit_plan": tr.get('exit_plan'),
                        "opened_at": tr.get('opened_at')
                    }
                    for tr in active_trades
                ],
                "open_orders": open_orders_struct,
                "recent_diary": recent_diary,
                "recent_fills": recent_fills_struct,
            }

            # Refresh macro context once per outer cycle (cached for inner ticks)
            try:
                _mc_stale = True
                if _macro_context_cache:
                    try:
                        _mc_dt = datetime.fromisoformat(_macro_context_cache.get("fetched_at", ""))
                        if _mc_dt.tzinfo is None:
                            _mc_dt = _mc_dt.replace(tzinfo=timezone.utc)
                        _mc_age = (datetime.now(timezone.utc) - _mc_dt).total_seconds()
                        _mc_stale = _mc_age > 3600  # re-fetch once per hour
                    except Exception:
                        _mc_stale = True
                if _mc_stale:
                    _macro_context_cache = await _fetch_macro_context()
                    logging.info("[MACRO] refreshed: %d events, %d headlines",
                                 len(_macro_context_cache.get("events", [])),
                                 len(_macro_context_cache.get("headlines", [])))
            except Exception as _mce:
                logging.warning("[MACRO] outer cycle fetch error — Claude will use technicals only: %s", _mce)

            # Gather data for ALL assets first (using Hyperliquid candles + local indicators)
            market_sections = []
            asset_prices = {}
            asset_trends = {}     # 4h EMA trend label per asset for inversion guard
            asset_trends_1d = {}  # daily EMA trend label per asset for macro filter
            asset_adx_1d = {}    # daily ADX per asset — gates macro filter when market is ranging
            asset_candles_5m = {}  # Raw 5m candles kept locally for volume confirmation (not sent to Claude)
            for asset in args.assets:
                try:
                    # Fetch price/OI/funding and all 6 timeframes in parallel — includes 30m (new).
                    (current_price, oi, funding,
                     candles_1h, candles_4h, candles_30m, candles_15m, candles_5m,
                     candles_1d) = await asyncio.gather(
                        hyperliquid.get_current_price(asset),
                        hyperliquid.get_open_interest(asset),
                        hyperliquid.get_funding_rate(asset),
                        hyperliquid.get_candles(asset, "1h",  60),
                        hyperliquid.get_candles(asset, "4h",  60),
                        hyperliquid.get_candles(asset, "30m", 40),
                        hyperliquid.get_candles(asset, "15m", 30),
                        hyperliquid.get_candles(asset, "5m",  100),  # 100 candles sufficient for BB(20)+StochRSI(14,14) warmup
                        hyperliquid.get_candles(asset, "1d",  50),
                    )
                    asset_prices[asset] = current_price
                    asset_candles_5m[asset] = candles_5m

                    # CHAOS-3 / CHAOS-v9-2 FIX: Stale candle watchdog — if last 5m or 1h candle
                    # is more than 3× the interval duration old, skip this asset entirely for this
                    # cycle (signal on 30+ min old OHLCV data is worse than no signal at all).
                    # 4h candles: only warn (4h candles naturally have a longer gap near open).
                    _now_ms = int(time.time() * 1000)
                    _stale_skip = False
                    for _stale_label, _stale_candles, _stale_ms, _stale_halt in [
                        ("5m", candles_5m, 300_000, True),
                        ("1h", candles_1h, 3_600_000, True),
                        ("4h", candles_4h, 14_400_000, False),
                    ]:
                        if _stale_candles:
                            _last_t = _stale_candles[-1].get("t")
                            if _last_t and (_now_ms - int(_last_t)) > 3 * _stale_ms:
                                _age_min = (_now_ms - int(_last_t)) / 60_000
                                if _stale_halt:
                                    logging.warning(
                                        "[STALE DATA] %s %s last candle is %.1f min old — skipping asset this cycle",
                                        asset, _stale_label, _age_min,
                                    )
                                    _stale_skip = True
                                else:
                                    logging.warning(
                                        "[STALE DATA] %s %s last candle is %.1f min old — possible data lag",
                                        asset, _stale_label, _age_min,
                                    )
                    if _stale_skip:
                        continue

                    # CRITICAL-2: Accumulate OI history for oi_confirmed()
                    if asset not in _oi_history:
                        _oi_history[asset] = _deque(maxlen=3)
                    if oi is not None:
                        _oi_history[asset].append(float(oi))
                    if asset not in price_history:
                        price_history[asset] = deque(maxlen=60)
                    price_history[asset].append({"t": datetime.now(timezone.utc).isoformat(), "mid": round_or_none(current_price, 2)})
                    ind_30m = compute_all(candles_30m)
                    ind_15m = compute_all(candles_15m)
                    ind_5m  = compute_all(candles_5m)

                    if len(candles_1h) < 26:
                        add_event(f"Skipping {asset}: only {len(candles_1h)} 1h candles (need 26+)")
                        continue

                    intra = compute_all(candles_1h)
                    lt = compute_all(candles_4h)

                    ema20_1h = latest(intra.get("ema20", []))
                    ema50_1h = latest(intra.get("ema50", []))
                    ema20_4h = latest(lt.get("ema20", []))
                    ema50_4h = latest(lt.get("ema50", []))
                    macd_hist_1h = latest(intra.get("macd_histogram", []))
                    macd_hist_4h = latest(lt.get("macd_histogram", []))
                    macd_sig_1h = latest(intra.get("macd_signal", []))
                    macd_sig_4h = latest(lt.get("macd_signal", []))
                    rsi14_1h = latest(intra.get("rsi14", []))
                    rsi14_4h = latest(lt.get("rsi14", []))

                    # ── Compute trend labels (EMA20 > EMA50 = BULLISH) ───────
                    # EMA20 > EMA50 means the faster average is above the slower → uptrend (BULLISH)
                    # EMA20 < EMA50 means faster is below slower → downtrend (BEARISH)
                    if ema20_4h is not None and ema50_4h is not None:
                        trend_4h = "BULLISH" if ema20_4h > ema50_4h else "BEARISH"
                    else:
                        trend_4h = "UNKNOWN"

                    if ema20_1h is not None and ema50_1h is not None:
                        trend_1h = "BULLISH" if ema20_1h > ema50_1h else "BEARISH"
                    else:
                        trend_1h = "UNKNOWN"

                    # Daily macro trend — same EMA cross logic on 1d candles.
                    # Filters out 4h "bounces" that run counter to the weekly move.
                    ind_1d = compute_all(candles_1d)
                    ema20_1d = latest(ind_1d.get("ema20", []))
                    ema50_1d = latest(ind_1d.get("ema50", []))
                    if ema20_1d is not None and ema50_1d is not None:
                        trend_1d = "BULLISH" if ema20_1d > ema50_1d else "BEARISH"
                    else:
                        trend_1d = "UNKNOWN"

                    # MACD histogram > 0 = MACD line above signal = bullish momentum
                    momentum_4h = (
                        "BULLISH" if macd_hist_4h is not None and macd_hist_4h > 0
                        else "BEARISH" if macd_hist_4h is not None and macd_hist_4h < 0
                        else "NEUTRAL"
                    )

                    asset_trends[asset] = trend_4h
                    asset_trends_1d[asset] = trend_1d
                    asset_adx_1d[asset] = latest(ind_1d.get("adx", []))

                    recent_mids = [entry["mid"] for entry in list(price_history.get(asset, []))[-10:]]
                    funding_annualized = round(funding * 24 * 365 * 100, 2) if funding else None

                    # Spread from impactPxs in cached metadata — no extra API call
                    try:
                        _dex = asset.split(":")[0] if ":" in asset else None
                        _mdata = await hyperliquid.get_meta_and_ctxs(dex=_dex)
                        spread_pct = 0.0
                        if isinstance(_mdata, list) and len(_mdata) >= 2:
                            _umeta, _uctxs = _mdata[0], _mdata[1]
                            _uidx = next(
                                (i for i, u in enumerate(_umeta.get("universe", [])) if u.get("name") == asset),
                                None
                            )
                            if _uidx is not None and _uidx < len(_uctxs):
                                _ipx = _uctxs[_uidx].get("impactPxs")
                                if _ipx and len(_ipx) >= 2 and current_price > 0:
                                    spread_pct = abs(float(_ipx[1]) - float(_ipx[0])) / float(current_price) * 100
                    except Exception:
                        spread_pct = 0.0

                    ema20_15m = latest(ind_15m.get("ema20", []))
                    # BUG-4 FIX: Default to False when EMA is unavailable (startup warmup / data gap).
                    # Previously defaulted to True, silently bypassing the near_ema gate when EMA
                    # data was missing — the E-1 fix in strategy.py never fired because this always
                    # wrote the key with a value, and it was True on unavailable data.
                    # near_ema threshold widened from 0.3% → 0.5%:
                    # 0.3% = $270 for BTC at $90k — a single 15m candle can move that far,
                    # making near_ema permanently False in even mild trends. 0.5% = $450,
                    # which still represents a genuine pullback to the EMA zone while allowing
                    # more realistic trend setups to register near_ema=True and gain the 1.5 score points.
                    near_ema_15m = (
                        abs(current_price - ema20_15m) / current_price < 0.005
                        if (ema20_15m is not None and current_price > 0)
                        else False
                    )

                    # Pre-compute ADX and Bollinger Band values for both timeframes
                    adx_4h  = latest(lt.get("adx", []))
                    adx_1h  = latest(intra.get("adx", []))
                    bb_upper_4h  = latest(lt.get("bbands_upper", []))
                    bb_lower_4h  = latest(lt.get("bbands_lower", []))
                    bb_middle_4h = latest(lt.get("bbands_middle", []))
                    bb_width_pct_4h = (
                        round((bb_upper_4h - bb_lower_4h) / bb_middle_4h * 100, 2)
                        if (bb_upper_4h is not None and bb_lower_4h is not None
                            and bb_middle_4h and bb_middle_4h != 0)
                        else None
                    )

                    # CRITICAL-3: Build BB width series for is_trending_regime()
                    _bbu_ser = lt.get("bbands_upper", [])
                    _bbl_ser = lt.get("bbands_lower", [])
                    _bbm_ser = lt.get("bbands_middle", [])
                    _bb_w_vals = []
                    for _bbi in range(len(_bbu_ser)):
                        _bu = _bbu_ser[_bbi]
                        _bl = _bbl_ser[_bbi] if _bbi < len(_bbl_ser) else None
                        _bm = _bbm_ser[_bbi] if _bbi < len(_bbm_ser) else None
                        if _bu is not None and _bl is not None and _bm is not None and _bm > 0:
                            _bb_w_vals.append(round((_bu - _bl) / _bm * 100, 3))
                    bb_width_series = _bb_w_vals[-20:]

                    market_sections.append({
                        "asset": asset,
                        "current_price": round_or_none(current_price, 2),
                        # Pre-computed trend labels — trust these for directional bias
                        "trend_1d": trend_1d,       # BULLISH=EMA20>EMA50 on 1d → macro bias; hard gate in execution
                        "trend_4h": trend_4h,       # BULLISH=EMA20>EMA50 on 4h → favor BUY; BEARISH→favor SELL
                        "trend_1h": trend_1h,       # BULLISH=EMA20>EMA50 on 1h → confirms entry direction
                        "momentum_4h": momentum_4h, # BULLISH=histogram>0; BEARISH=histogram<0
                        "intraday_1h": {
                            "ema20": round_or_none(ema20_1h, 2),
                            "ema50": round_or_none(ema50_1h, 2),
                            "macd": round_or_none(latest(intra.get("macd", [])), 4),
                            "macd_histogram": round_or_none(macd_hist_1h, 4),
                            "macd_signal": round_or_none(macd_sig_1h, 4),
                            "rsi14": round_or_none(rsi14_1h, 2),
                            "adx": round_or_none(adx_1h, 2),
                            "adx_trending": (adx_1h or 0) > 15,  # BUG-P11-PERF-2 FIX: matches active gate (was 25)
                            "series": {
                                "ema20": round_series(last_n(intra.get("ema20", []), 3), 2),
                                "ema50": round_series(last_n(intra.get("ema50", []), 3), 2),
                                "macd_histogram": round_series(last_n(intra.get("macd_histogram", []), 3), 4),
                                "rsi14": round_series(last_n(intra.get("rsi14", []), 3), 2),
                            }
                        },
                        "long_term_4h": {
                            "ema20": round_or_none(ema20_4h, 2),
                            "ema50": round_or_none(ema50_4h, 2),
                            "atr14": round_or_none(latest(lt.get("atr14", [])), 2),
                            "macd": round_or_none(latest(lt.get("macd", [])), 4),
                            "macd_histogram": round_or_none(macd_hist_4h, 4),
                            "macd_signal": round_or_none(macd_sig_4h, 4),
                            "rsi14": round_or_none(rsi14_4h, 2),
                            "adx": round_or_none(adx_4h, 2),
                            "adx_trending": (adx_4h or 0) > 15,  # BUG-P11-PERF-2 FIX: matches active gate (was 25)
                            "bb_upper": round_or_none(bb_upper_4h, 2),
                            "bb_lower": round_or_none(bb_lower_4h, 2),
                            "bb_width_pct": bb_width_pct_4h,
                            "bb_width_series": bb_width_series,  # CRITICAL-3
                            "macd_histogram_series": round_series(last_n(lt.get("macd_histogram", []), 3), 4),
                            "rsi_series": round_series(last_n(lt.get("rsi14", []), 3), 2),
                        },
                        "open_interest": round_or_none(oi, 2),
                        "oi_series": list(_oi_history.get(asset, [])),  # CRITICAL-2
                        "funding_rate": round_or_none(funding, 8),
                        "funding_annualized_pct": funding_annualized,
                        "recent_mid_prices": recent_mids,
                        "setup_30m": {
                            "ema20":          round_or_none(latest(ind_30m.get("ema20", [])), 2),
                            "ema50":          round_or_none(latest(ind_30m.get("ema50", [])), 2),
                            "macd_histogram": round_or_none(latest(ind_30m.get("macd_histogram", [])), 4),
                            "rsi14":          round_or_none(latest(ind_30m.get("rsi14", [])), 2),
                        },
                        "setup_15m": {
                            "ema20":          round_or_none(latest(ind_15m.get("ema20", [])), 2),
                            "macd_histogram": round_or_none(latest(ind_15m.get("macd_histogram", [])), 4),
                            "rsi14":          round_or_none(latest(ind_15m.get("rsi14", [])), 2),
                            "near_ema":       near_ema_15m,
                        },
                        "trigger_5m": {
                            "macd_histogram": round_or_none(latest(ind_5m.get("macd_histogram", [])), 4),
                            "rsi14":          round_or_none(latest(ind_5m.get("rsi14", [])), 2),
                            "candle_bullish": candles_5m[-1]["close"] > candles_5m[-1]["open"] if candles_5m else False,
                        },
                        "spread_pct": round(spread_pct, 4),
                        "candles_4h": candles_4h,
                        "candles_1h": candles_1h,   # CRITICAL-4: needed for S&R swing H/L gate
                        "candles_5m": candles_5m,   # required for BB+StochRSI signal and regime ADX check
                        "daily_1d": candles_1d[-2] if len(candles_1d) >= 2 else (candles_1d[-1] if candles_1d else {}),  # BUG-C: use yesterday's closed candle not today's incomplete
                    })
                except Exception as e:
                    add_event(f"Data gather error {asset}: {e}")
                    continue

            # V3-HIGH-2 FIX: Removed dead `context_payload` build block.

            # BUG-v8-L5 FIX: BTC correlation filter was bypassed when BTC is not in --assets.
            # market_filter() blocks altcoin BUY when BTC 1h trend is BEARISH. With BTC absent,
            # _btc_trend_1h defaulted to "UNKNOWN" (filter never fires). Fetch BTC 1h candles
            # once per outer cycle when BTC is not already in the tracked asset list.
            if "BTC" not in args.assets:
                try:
                    _btc_corr_1h = await hyperliquid.get_candles("BTC", "1h", 60)
                    _btc_corr_5m = await hyperliquid.get_candles("BTC", "5m", 20)
                    if _btc_corr_1h:
                        from src.indicators.local_indicators import compute_all as _btc_compute
                        _btc_ind = _btc_compute(_btc_corr_1h)
                        _btc_ema20 = (_btc_ind.get("ema20") or [None])[-1]
                        _btc_ema50 = (_btc_ind.get("ema50") or [None])[-1]
                        if _btc_ema20 and _btc_ema50:
                            asset_candles_5m["BTC"] = _btc_corr_5m or []
                            # Inject a minimal BTC entry into market_sections so _btc_trend_1h lookup works
                            _btc_trend_label = "BULLISH" if _btc_ema20 > _btc_ema50 else "BEARISH"
                            if not any(m.get("asset") == "BTC" for m in market_sections):
                                market_sections.append({"asset": "BTC", "trend_1h": _btc_trend_label})
                            logging.debug("[BTC CORR] injected BTC trend_1h=%s for correlation filter", _btc_trend_label)
                except Exception as _btc_err:
                    logging.warning("[BTC CORR] BTC correlation fetch failed — filter will use UNKNOWN: %s", _btc_err)

            add_event(f"[CYCLE {invocation_count}] BB+StochRSI scan — {len(market_sections)} asset(s)")

            # ── BB + StochRSI mean reversion pipeline ─────────────────────────────────
            # Direction set by BB band touch + StochRSI hook — code only.
            # TP = BB midline. SL = 1.5×ATR(5m). Claude only called on >3% price spikes.
            outputs = {"reasoning": "", "trade_decisions": []}
            _max_dt = int(CONFIG.get("max_daily_trades") or 40)

            def _make_hold(asset_name: str, reason: str) -> dict:
                return {"asset": asset_name, "action": "hold", "allocation_usd": 0.0,
                        "order_type": "market", "limit_price": None,
                        "tp_price": None, "sl_price": None,
                        "exit_plan": "", "rationale": reason}

            # Session gate — check once for all assets (same time applies to all)
            _sess_ok, _sess_reason = session_gate_ok(CONFIG)

            for _asset in args.assets:
                _ac = next((m for m in market_sections if m.get("asset") == _asset), None)
                if not _ac:
                    outputs["trade_decisions"].append(_make_hold(_asset, "no market data"))
                    continue

                # Session gate (00:00–06:00 UTC + weekends)
                if not _sess_ok:
                    outputs["trade_decisions"].append(_make_hold(_asset, _sess_reason))
                    continue

                # Daily trade cap
                if _daily_trade_count >= _max_dt:
                    outputs["trade_decisions"].append(_make_hold(_asset, f"daily cap {_daily_trade_count}/{_max_dt}"))
                    continue

                # Per-asset SL cooldown
                _cd_until = _sl_cooldown_map.get(_asset)
                if _cd_until and datetime.now(timezone.utc) < _cd_until:
                    _mins_left = round((_cd_until - datetime.now(timezone.utc)).total_seconds() / 60, 1)
                    outputs["trade_decisions"].append(_make_hold(_asset, f"SL cooldown {_mins_left}min remaining"))
                    continue

                # Regime pause: 1h ADX > per-asset threshold = strong trend = skip reversion
                if is_trend_regime_active(_ac, CONFIG):
                    _adx_val = _ac.get("intraday_1h", {}).get("adx", "?")
                    _adx_threshold = CONFIG.get(f"adx_override_{_asset.lower()}") or CONFIG.get("trend_pause_adx", 30)
                    outputs["trade_decisions"].append(_make_hold(
                        _asset, f"regime pause: 1h ADX={_adx_val} > {_adx_threshold} (trend too strong)"))
                    continue

                # BB + StochRSI signal
                _candles_5m_data = _ac.get("candles_5m", [])
                _sig = compute_bb_stochrsi_signal(_candles_5m_data, CONFIG)
                _direction = _sig.get("signal", "NONE")

                if _direction == "NONE":
                    logging.debug("[BB/SRSI] %s no signal K=%.1f",
                                  _asset, _sig.get("stochrsi_k") or 0)
                    outputs["trade_decisions"].append(_make_hold(
                        _asset, f"no BB/StochRSI signal (K={_sig.get('stochrsi_k')})"))
                    continue

                # Convert LONG/SHORT → buy/sell
                _direction_str = "buy" if _direction == "LONG" else "sell"

                # Funding rate gate
                _funding_val = float(_ac.get("funding_rate") or 0)
                if not funding_rate_ok(_funding_val, _direction_str):
                    outputs["trade_decisions"].append(_make_hold(
                        _asset, f"funding gate: {_funding_val:.5f}/8h blocks {_direction_str}"))
                    continue

                # Compute TP and SL
                _entry  = float(_ac.get("current_price") or 0)
                _atr_5m = float(_sig.get("atr") or 0)
                _bb_mid = float(_sig.get("bb_mid") or 0)
                _fee_buf = _entry * float(CONFIG.get("taker_fee_pct") or 0.00045) * 2

                if _entry <= 0 or _atr_5m <= 0 or _bb_mid <= 0:
                    outputs["trade_decisions"].append(_make_hold(_asset, "missing price, ATR5m, or BB mid"))
                    continue

                if _direction_str == "buy":
                    _tp = round(_bb_mid, 6)          # BB midline above entry
                    _sl = round(_entry - 1.5 * _atr_5m, 6)
                    _tp1 = _tp  # single TP for mean reversion (no partial close split)
                    _tp2 = _tp
                    # Validate: TP must be above entry (mean reversion target)
                    if _tp <= _entry + _fee_buf:
                        outputs["trade_decisions"].append(_make_hold(
                            _asset, f"TP {_tp:.4f} ≤ entry+fees {_entry+_fee_buf:.4f} — no edge"))
                        continue
                else:
                    _tp = round(_bb_mid, 6)          # BB midline below entry
                    _sl = round(_entry + 1.5 * _atr_5m, 6)
                    _tp1 = _tp
                    _tp2 = _tp
                    if _tp >= _entry - _fee_buf:
                        outputs["trade_decisions"].append(_make_hold(
                            _asset, f"TP {_tp:.4f} ≥ entry-fees {_entry-_fee_buf:.4f} — no edge"))
                        continue

                # Position sizing: ATR 1% risk rule + pct cap (risk manager unchanged)
                _buying_power = account_value * float(CONFIG.get("max_leverage") or 5)
                _pct_cap = _buying_power * (float(CONFIG.get("max_position_pct") or 15) / 100.0)
                _atr_sized = risk_mgr.atr_position_size(account_value, _entry, _sl)
                _alloc = min(_pct_cap, _atr_sized)

                # Claude anomaly check — only on sharp moves (>ANOMALY_TRIGGER_PCT in 5 candles)
                _anomaly_pct = float(CONFIG.get("anomaly_trigger_pct") or 3.0)
                _price_chg_5c = 0.0
                if len(_candles_5m_data) >= 5:
                    _p_now  = float(_candles_5m_data[-1]["close"])
                    _p_5ago = float(_candles_5m_data[-5]["close"])
                    if _p_5ago > 0:
                        _price_chg_5c = abs(_p_now - _p_5ago) / _p_5ago * 100

                if _price_chg_5c > _anomaly_pct:
                    _verdict = await asyncio.to_thread(
                        agent.claude_anomaly_check,
                        _asset, _price_chg_5c, _direction_str
                    )
                    if _verdict != "APPROVE":
                        add_event(f"[CLAUDE ANOMALY] {_asset} REJECTED — {_price_chg_5c:.1f}% move in 5 candles")
                        outputs["trade_decisions"].append(_make_hold(
                            _asset, f"anomaly REJECT: {_price_chg_5c:.1f}% in 5 candles"))
                        continue
                    add_event(f"[CLAUDE ANOMALY] {_asset} APPROVED — {_price_chg_5c:.1f}% move ok")

                # Signal is clean — queue the trade
                bb_touch = "lower" if _direction_str == "buy" else "upper"
                add_event(f"[BB/SRSI] {_asset} {_direction_str} — BB {bb_touch} touch K={_sig.get('stochrsi_k')} TP={_tp:.4f} SL={_sl:.4f}")

                # Log to signals.jsonl for win-rate tracking
                try:
                    with open(_signals_path, "a", encoding="utf-8") as _sf:
                        _sf.write(json.dumps({
                            "timestamp":    datetime.now(timezone.utc).isoformat(),
                            "asset":        _asset,
                            "direction":    _direction_str,
                            "stochrsi_k":   _sig.get("stochrsi_k"),
                            "bb_touch":     bb_touch,
                            "atr_5m":       round(_atr_5m, 6),
                            "tp":           round(_tp, 6),
                            "sl":           round(_sl, 6),
                            "action":       "queued",
                        }) + "\n")
                except Exception:
                    pass

                # LIMIT order — 0.15% better than market (0% maker fee on Hyperliquid)
                _lim_off = _entry * 0.0015
                _lim_px = round(_entry - _lim_off, 6) if _direction_str == "buy" else round(_entry + _lim_off, 6)

                outputs["trade_decisions"].append({
                    "asset":          _asset,
                    "action":         _direction_str,
                    "allocation_usd": _alloc,
                    "order_type":     "limit",
                    "limit_price":    _lim_px,
                    "tp_price":       _tp,
                    "tp1_price":      _tp1,
                    "tp2_price":      _tp2,
                    "sl_price":       _sl,
                    "atr14":          _atr_5m,   # 5m ATR used for SL sizing
                    "current_price":  _entry,
                    "score":          8.0,        # fixed score placeholder (not score-based system)
                    "exit_plan":      f"BB-reversion TP={_tp:.4f} SL={_sl:.4f} K={_sig.get('stochrsi_k')}",
                    "rationale":      (f"BB {bb_touch} touch StochRSI K={_sig.get('stochrsi_k')} "
                                      f"direction={_direction_str} atr5m={_atr_5m:.6f}"),
                })
            # ─────────────────────────────────────────────────────────────────────────

            reasoning_text = outputs.get("reasoning", "") if isinstance(outputs, dict) else ""
            if reasoning_text:
                add_event(f"LLM reasoning summary: {reasoning_text}")

            # Log full cycle decisions for the dashboard
            cycle_decisions = []
            for d in outputs.get("trade_decisions", []) if isinstance(outputs, dict) else []:
                cycle_decisions.append({
                    "asset": d.get("asset"),
                    "action": d.get("action", "hold"),
                    "allocation_usd": d.get("allocation_usd", 0),
                    "rationale": d.get("rationale", ""),
                })
            cycle_log = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "cycle": invocation_count,
                "reasoning": reasoning_text[:2000] if reasoning_text else "",
                "decisions": cycle_decisions,
                "account_value": round_or_none(account_value, 2),
                "balance": round_or_none(state['balance'], 2),
                "perps_value": round_or_none(state.get('perps_value'), 2),
                "spot_usdc": round_or_none(state.get('spot_usdc'), 2),
                "withdrawable": round_or_none(state.get('withdrawable'), 2),
                "positions": positions,
                "open_orders": open_orders_struct,
                "recent_fills": recent_fills_struct,
                "positions_count": len([p for p in state.get('positions', []) if abs(float(p.get('szi') or 0)) > 0]),
            }
            try:
                with open(decisions_path, "a") as f:
                    f.write(json.dumps(cycle_log) + "\n")
            except Exception:
                pass

            # Execute trades for each asset
            for output in outputs.get("trade_decisions", []) if isinstance(outputs, dict) else []:
                try:
                    asset = output.get("asset")
                    if not asset or asset not in args.assets:
                        continue
                    action = output.get("action")
                    trend_4h = asset_trends.get(asset, "UNKNOWN")
                    current_price = asset_prices.get(asset, 0)
                    if not current_price or current_price <= 0:
                        add_event(f"Skipping {asset}: invalid/zero price, cannot size order")
                        continue
                    # Mandatory sanity log — confirms direction vs trend in every cycle
                    logging.info("[TRADE] %s action=%s | 4h_trend=%s | entry=%s", asset, action, trend_4h, current_price)
                    print(f"[TRADE] {asset} action={action} | 4h_trend={trend_4h} | entry={current_price}")
                    rationale = output.get("rationale", "")
                    if rationale:
                        add_event(f"Decision rationale for {asset}: {rationale}")
                    if action in ("buy", "sell"):
                        # BUG 1 FIX: Hard gate — skip if state machine says already in position or cooling down
                        _sm_state = state_mgr.get_state(asset)
                        if _sm_state in ("ENTERED", "COOLDOWN"):
                            logging.info("[STATE GATE] %s skipped — state=%s", asset, _sm_state)
                            add_event(f"[STATE GATE] {asset} skipped — state={_sm_state}, no new entry")
                            continue
                        # Spread pre-check — skip entry if spread is unusually wide
                        asset_ctx = next((m for m in market_sections if m.get("asset") == asset), {})
                        _spread = float(asset_ctx.get("spread_pct") or 0)
                        if _spread > 0.15:
                            logging.info("[SPREAD] %s blocked — spread %.3f%% > 0.15%%", asset, _spread)
                            add_event(f"[SPREAD] {asset} blocked — spread {_spread:.3f}% too wide")
                            continue

                        # Candle close gate — signal already used [-2] closed candle, but
                        # confirm the current candle is at least 30% complete before entry
                        # to avoid entering at the very start of a new candle.
                        _secs_into_5m = (datetime.now(timezone.utc).minute % 5) * 60 + datetime.now(timezone.utc).second
                        _candle_age_pct = _secs_into_5m / 300  # 0.0 = just opened, 1.0 = about to close
                        if _candle_age_pct < 0.30:
                            logging.info("[CANDLE GATE] %s waiting — 5m candle only %.0f%% complete (need 30%%)",
                                         asset, _candle_age_pct * 100)
                            continue
                        is_buy = action == "buy"
                        alloc_usd = float(output.get("allocation_usd", 0.0))
                        if alloc_usd <= 0:
                            add_event(f"Holding {asset}: zero/negative allocation")
                            continue

                        # --- RISK: Validate trade before execution ---
                        output["current_price"] = current_price
                        # atr14 is 5m ATR (already set by BB+StochRSI pipeline).
                        # Ensure it's present so enforce_stop_loss can use it for SL validation.
                        if not output.get("atr14"):
                            output["atr14"] = asset_ctx.get("long_term_4h", {}).get("atr14")

                        allowed, reason, output = risk_mgr.validate_trade(
                            output, state, initial_account_value or 0
                        )
                        if not allowed:
                            add_event(f"RISK BLOCKED {asset}: {reason}")
                            with open(diary_path, "a") as f:
                                f.write(json.dumps({
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "asset": asset,
                                    "action": "risk_blocked",
                                    "reason": reason,
                                    "original_alloc_usd": alloc_usd,
                                }) + "\n")
                            continue
                        # Use potentially adjusted values from risk manager
                        alloc_usd = float(output.get("allocation_usd", alloc_usd))
                        amount = alloc_usd / current_price

                        # P4-H-2 / P8-HIGH-1: Second concurrent-position gate immediately before order.
                        # Uses max(exchange count, active_trades count) — state.get("positions") misses
                        # same-cycle entries placed before this asset but not yet visible on exchange.
                        _max_conc = int(CONFIG.get("max_concurrent_positions") or 3)
                        _open_pos_count = max(
                            sum(1 for _p in (state.get("positions") or []) if abs(float(_p.get("szi") or 0)) > 0),
                            len(active_trades),
                        )
                        if _open_pos_count >= _max_conc:
                            add_event(f"[CONC GATE] {asset} blocked — {_open_pos_count}/{_max_conc} positions open")
                            logging.info("[CONC GATE] %s blocked — %d/%d concurrent positions", asset, _open_pos_count, _max_conc)
                            continue

                        # Place market or limit order
                        order_type = output.get("order_type", "market")
                        limit_price = output.get("limit_price")

                        if order_type == "limit" and limit_price:
                            limit_price = float(limit_price)
                            if is_buy:
                                order = await hyperliquid.place_limit_buy(asset, amount, limit_price)
                            else:
                                order = await hyperliquid.place_limit_sell(asset, amount, limit_price)
                            add_event(f"LIMIT {action.upper()} {asset} amount {amount:.4f} at limit ${limit_price}")
                        else:
                            order = await hyperliquid.place_buy_order(asset, amount) if is_buy else await hyperliquid.place_sell_order(asset, amount)

                        # Extract the OID from the entry order response for precise fill matching
                        entry_oids = hyperliquid.extract_oids(order) if order else []
                        entry_oid = entry_oids[0] if entry_oids else None

                        # Confirm fill — poll all 3 attempts regardless of whether fills
                        # were found in an earlier attempt. Each poll may return new
                        # partial fills for the same OID. A seen-tid set prevents
                        # double-counting when the same fill appears in multiple polls.
                        filled_qty = 0.0
                        filled = False
                        _seen_fill_tids: set = set()
                        for _attempt in range(3):
                            await asyncio.sleep(1)
                            try:
                                fills_check = await hyperliquid.get_recent_fills(limit=30)
                                for fc in fills_check:
                                    fc_oid = fc.get('oid') or fc.get('orderId')
                                    if not (entry_oid and fc_oid and str(fc_oid) == str(entry_oid)):
                                        continue
                                    # Deduplicate by trade ID — use stable composite key as fallback
                                    # (id(fc) is NOT stable across poll iterations and causes double-counting)
                                    fc_tid = str(
                                        fc.get('tid') or fc.get('tradeId') or fc.get('hash')
                                        or f"{fc_oid}_{fc.get('sz') or fc.get('size')}_{fc.get('time') or fc.get('timestamp')}"
                                    )
                                    if fc_tid in _seen_fill_tids:
                                        continue
                                    _seen_fill_tids.add(fc_tid)
                                    filled_qty += float(fc.get('sz') or fc.get('size') or 0)
                                    filled = True
                            except Exception:
                                pass
                        # No early break — always complete all 3 polls to capture partial fills

                        # BUG-P11-LIMIT FIX: Do NOT cancel after 3 seconds.
                        # Previous code cancelled the limit immediately after 3×1s polls.
                        # A 0.15% offset limit in a trending market never fills in 3 seconds.
                        # Fix: leave the order live; inner loop cancels after 5 min if still unfilled.
                        if order_type == "limit" and filled_qty == 0 and entry_oid:
                            logging.info("[LIMIT] %s not yet filled — leaving live; inner loop will cancel after 5 min if still unfilled", asset)
                            add_event(f"[LIMIT] {asset} limit pending fill — will cancel if unfilled after 5 min")

                        # CRITICAL-5: Partial fill — cancel unfilled remainder to avoid unprotected second fill
                        if order_type == "limit" and 0 < filled_qty < amount * 0.99 and entry_oid:
                            try:
                                await hyperliquid.cancel_order(asset, entry_oid)
                                logging.info("[LIMIT] %s partial fill %.6f/%.6f — unfilled remainder cancelled", asset, filled_qty, amount)
                                add_event(f"[LIMIT] {asset} partial fill {filled_qty:.4f}/{amount:.4f} — remainder cancelled to avoid unprotected exposure")
                            except Exception as _pce:
                                logging.warning("[LIMIT] %s partial fill cancel failed: %s", asset, _pce)

                        # Use actual filled quantity for TP/SL sizing.
                        # Fall back to the requested amount for resting limit orders
                        # that haven't filled yet, or when OID matching found nothing.
                        tp_sl_size = filled_qty if filled_qty > 0 else amount
                        logging.info(
                            "[FILL] %s entry_oid=%s filled_qty=%.6f requested=%.6f tp_sl_size=%.6f",
                            asset, entry_oid, filled_qty, amount, tp_sl_size,
                        )

                        trade_log.append({"type": action, "price": current_price, "amount": tp_sl_size, "exit_plan": output["exit_plan"], "filled": filled})
                        tp_oid = None
                        sl_oid = None
                        # For resting limit orders (filled_qty=0), skip TP/SL entirely.
                        # Reduce-only orders submitted against a non-existent position are
                        # silently rejected by Hyperliquid. The guardian places them next
                        # cycle once the position is confirmed open on the exchange.
                        tp1_oid = None
                        tp2_oid = None
                        _can_place_tpsl = filled_qty > 0 or order_type != "limit"
                        if _can_place_tpsl:
                            _half = round(tp_sl_size / 2, 6)
                            if output.get("tp1_price") and _half > 0:
                                _tp1r = await hyperliquid.place_take_profit(asset, is_buy, _half, output["tp1_price"])
                                _tp1_oids = hyperliquid.extract_oids(_tp1r)
                                tp1_oid = _tp1_oids[0] if _tp1_oids else None
                                add_event(f"TP1 placed {asset} at {output['tp1_price']} size={_half:.6f} (50%)")
                            if output.get("tp2_price") and _half > 0:
                                _tp2r = await hyperliquid.place_take_profit(asset, is_buy, _half, output["tp2_price"])
                                _tp2_oids = hyperliquid.extract_oids(_tp2r)
                                tp2_oid = _tp2_oids[0] if _tp2_oids else None
                                add_event(f"TP2 placed {asset} at {output['tp2_price']} size={_half:.6f} (50% runner)")
                            # Fallback if tp1/tp2 not set: use single tp_price
                            if not tp1_oid and output.get("tp_price"):
                                _tpr = await hyperliquid.place_take_profit(asset, is_buy, tp_sl_size, output["tp_price"])
                                _tp_oids = hyperliquid.extract_oids(_tpr)
                                tp_oid = _tp_oids[0] if _tp_oids else None
                                add_event(f"TP placed {asset} at {output['tp_price']} size={tp_sl_size:.6f}")
                            if output.get("sl_price"):
                                _sl_placed = False
                                _sl_err = None
                                for _sl_attempt in range(2):  # try twice before market-close fallback
                                    try:
                                        _slr = await hyperliquid.place_stop_loss(asset, is_buy, tp_sl_size, output["sl_price"])
                                        _sl_oids = hyperliquid.extract_oids(_slr)
                                        sl_oid = _sl_oids[0] if _sl_oids else None
                                        add_event(f"SL placed {asset} at {output['sl_price']} size={tp_sl_size:.6f}")
                                        _sl_placed = True
                                        break
                                    except Exception as _sl_err:
                                        logging.warning("[SL RETRY] %s attempt %d failed: %s", asset, _sl_attempt + 1, _sl_err)
                                        if _sl_attempt == 0:
                                            await asyncio.sleep(2)  # brief pause before retry
                                if not _sl_placed:
                                    logging.critical(
                                        "[SL FAIL] %s — SL placement failed after 2 attempts: %s — market-closing to avoid unprotected exposure",
                                        asset, _sl_err
                                    )
                                    add_event(f"[SL FAIL] {asset} SL failed after retry — market-closing position immediately")
                                    await send_alert(
                                        f"\U0001f6a8 SL FAIL {asset} — 2 SL attempts failed, market-closing NOW. Error: {_sl_err}"
                                    )
                                    try:
                                        await hyperliquid.market_close(asset)
                                        await hyperliquid.cancel_all_orders(asset)
                                    except Exception as _mc_err:
                                        logging.critical(
                                            "[SL FAIL] %s — market-close also failed: %s — MANUAL INTERVENTION REQUIRED",
                                            asset, _mc_err
                                        )
                                        add_event(f"[SL FAIL] {asset} MARKET CLOSE ALSO FAILED — close manually on Hyperliquid NOW")
                                        await send_alert(
                                            f"\U0001f198 CRITICAL {asset} — SL fail AND market-close fail. CLOSE MANUALLY ON HYPERLIQUID NOW. Error: {_mc_err}"
                                        )
                                    continue
                        else:
                            logging.info("[LIMIT] %s TP/SL deferred — limit order not yet filled, guardian covers next cycle", asset)
                            add_event(f"[LIMIT] {asset} TP/SL deferred — position not confirmed (guardian places next cycle)")
                        # Reconcile: if opposite-side position exists or TP/SL just filled, clear stale active_trades for this asset
                        for existing in active_trades[:]:
                            if existing.get('asset') == asset:
                                try:
                                    active_trades.remove(existing)
                                except ValueError:
                                    pass
                        active_trades.append({
                            "asset": asset,
                            "is_long": is_buy,
                            "amount": tp_sl_size,
                            "half_size": round(tp_sl_size / 2, 6),
                            "entry_price": current_price,
                            "entry_atr": float(output.get("atr14") or 0),
                            "tp_price": output.get("tp_price"),
                            "tp1_price": output.get("tp1_price"),
                            "tp2_price": output.get("tp2_price"),
                            "sl_price": output.get("sl_price"),
                            "tp_oid": tp_oid,
                            "tp1_oid": tp1_oid,
                            "tp2_oid": tp2_oid,
                            "sl_oid": sl_oid,
                            "tp1_hit": False,
                            "trail_breakeven_done": False,
                            "trail_active": False,
                            "exit_plan": output["exit_plan"],
                            "funding_rate": float(asset_ctx.get("funding_rate") or 0),
                            "opened_at": datetime.now(timezone.utc).isoformat(),
                            # BUG-P11-LIMIT FIX: track pending limit orders for deferred cancel in inner loop
                            "entry_oid": entry_oid if (order_type == "limit" and filled_qty == 0) else None,
                            "limit_placed_at": datetime.now(timezone.utc).isoformat() if (order_type == "limit" and filled_qty == 0) else None,
                        })
                        # CRITICAL-8: record_entry (state.json) BEFORE save_active_trades
                        state_mgr.record_entry(asset)
                        save_active_trades(active_trades)
                        _daily_trade_count += 1
                        _save_daily_count(_daily_trade_count, str(_today_utc))  # CRITICAL-7
                        add_event(f"{action.upper()} {asset} amount {amount:.4f} at ~{current_price} [daily={_daily_trade_count}]")
                        if rationale:
                            add_event(f"Post-trade rationale for {asset}: {rationale}")
                        # Write to diary after confirming fills status
                        with open(diary_path, "a") as f:
                            diary_entry = {
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "asset": asset,
                                "action": action,
                                "order_type": order_type,
                                "limit_price": limit_price,
                                "allocation_usd": alloc_usd,
                                "amount": tp_sl_size,
                                "filled_qty": filled_qty,
                                "requested_qty": amount,
                                "entry_price": current_price,
                                "tp_price": output.get("tp_price"),
                                "tp1_price": output.get("tp1_price"),
                                "tp2_price": output.get("tp2_price"),
                                "tp_oid": tp_oid,
                                "tp1_oid": tp1_oid,
                                "tp2_oid": tp2_oid,
                                "sl_price": output.get("sl_price"),
                                "sl_oid": sl_oid,
                                "exit_plan": output.get("exit_plan", ""),
                                "rationale": output.get("rationale", ""),
                                "order_result": str(order),
                                "opened_at": datetime.now(timezone.utc).isoformat(),
                                "filled": filled
                            }
                            f.write(json.dumps(diary_entry) + "\n")
                        # P3-M2: Update per-asset diary index so guardian reads O(1) not O(n)
                        _diary_index[asset] = diary_entry
                        _save_diary_index(_diary_index)
                    else:
                        add_event(f"Hold {asset}: {output.get('rationale', '')}")
                except Exception as e:
                    add_event(f"Execution error {asset}: {e}")
                    add_event(f"Traceback: {traceback.format_exc()}")

            _consecutive_failures = 0  # reset on any cycle that completes state fetch
            _cycle_dur = time.monotonic() - cycle_start
            logging.info("[CYCLE] completed in %.1fs (interval=%ds)", _cycle_dur, _interval_seconds)
            if _cycle_dur > _interval_seconds:
                logging.warning(
                    "[CYCLE] overrun: %.1fs > interval %ds — next cycle starts immediately",
                    _cycle_dur, _interval_seconds,
                )
            # ── 5-minute inner loop — refresh 5m candles, re-score, re-execute ──────
            # Runs 11 more ticks (outer loop already counted as tick 0).
            # Only re-fetches 5m candles — 4h trend/indicators stay from outer loop.
            for _tick in range(11):
                await _interruptible_sleep(300)  # 5 minutes — wakes within 5s on SIGTERM

                # BUG-5 FIX: Check KILLSWITCH inside the inner loop.
                # Without this, operator creates KILLSWITCH expecting immediate halt but bot
                # continues trading for up to 55 more minutes (11 ticks × 5 min).
                if os.path.exists(os.path.normpath(_KILLSWITCH_FILE)):
                    logging.critical("[KILLSWITCH] detected inside inner loop tick %d — halting", _tick + 1)
                    add_event("[KILLSWITCH] Trading halted by KILLSWITCH file (inner loop).")
                    # BUG-v8-L1 FIX: Mirror outer loop retry logic — 3 attempts per asset.
                    # Old code had a single try/except; one exchange timeout → bot exits with
                    # position open and no alert. Now retries 3× and stays alive if close fails,
                    # so the outer loop guardian continues protecting the unclosed position.
                    _iks_unclosed = []
                    for _iks_tr in list(active_trades):
                        _iks_asset = _iks_tr.get("asset")
                        if not _iks_asset:
                            continue
                        _iks_closed = False
                        for _iks_attempt in range(3):
                            try:
                                await hyperliquid.cancel_all_orders(_iks_asset)
                                await hyperliquid.market_close(_iks_asset)
                                logging.critical("[KILLSWITCH] inner %s — closed (attempt %d)", _iks_asset, _iks_attempt + 1)
                                add_event(f"[KILLSWITCH] inner {_iks_asset} closed")
                                _iks_closed = True
                                break
                            except Exception as _iks_err:
                                logging.critical("[KILLSWITCH] inner %s — close attempt %d/3 failed: %s", _iks_asset, _iks_attempt + 1, _iks_err)
                                if _iks_attempt < 2:
                                    await asyncio.sleep(2)
                        if not _iks_closed:
                            _iks_unclosed.append(_iks_asset)
                            logging.critical("[KILLSWITCH] inner %s — ALL 3 ATTEMPTS FAILED — position remains open", _iks_asset)
                            add_event(f"[KILLSWITCH] inner {_iks_asset} CLOSE FAILED x3 — MANUAL INTERVENTION REQUIRED")
                    if _iks_unclosed:
                        await send_alert(
                            f"\U0001f6a8 KILLSWITCH (inner): {len(_iks_unclosed)} position(s) FAILED TO CLOSE: "
                            f"{', '.join(_iks_unclosed)}. Bot staying alive to monitor. Manual close required."
                        )
                        # Do NOT set _shutdown — keep outer loop running to protect unclosed positions
                    else:
                        await send_alert(
                            f"\U0001f6a8 KILLSWITCH (inner): all {len(active_trades)} position(s) closed. "
                            "Remove KILLSWITCH file to restart."
                        )
                        _shutdown = True
                    break

                # BUG-v7-O5 FIX: Alert when circuit breaker auto-resets at UTC midnight.
                if getattr(risk_mgr, 'circuit_breaker_was_active', False):
                    risk_mgr.circuit_breaker_was_active = False
                    await send_alert(
                        "⚠️ Daily loss circuit breaker AUTO-RESET at UTC midnight — trading has resumed. "
                        "Review yesterday's drawdown before leaving the bot unattended."
                    )
                if risk_mgr.circuit_breaker_active:
                    logging.info("[INNER] circuit breaker active — skipping tick %d", _tick + 1)
                    if _tick == 0:  # alert once per inner-loop session, not every tick
                        await send_alert("⛔ Daily loss circuit breaker active — no new trades until UTC midnight reset.")
                    continue

                logging.info("[INNER %d/11] refreshing 5m candles for %d assets", _tick + 1, len(args.assets))

                # ── SL ORPHAN CHECK — fix unprotected resting limit fills ─────────────
                # Runs every 5-min inner tick. If any active_trade has sl_oid=None (SL was
                # deferred because limit order was not yet filled at entry time), check whether
                # the position now exists on the exchange. If it does, place SL immediately.
                # This closes the 60-minute unprotected window to ≤5 minutes.
                # PERF-v9-2 FIX: Fetch positions once before the orphan loop instead of once
                # per deferred trade. With 3 concurrent deferred trades this was 3 sequential
                # REST calls per inner tick; now it is always exactly 1.
                _orphan_all_positions = None
                _has_deferred = any(
                    t.get("sl_oid") is None and t.get("sl_price") and float(t.get("amount") or 0) > 0
                    for t in active_trades
                )
                if _has_deferred:
                    try:
                        _orphan_all_positions = await hyperliquid.get_positions()
                    except Exception as _oap_err:
                        logging.warning("[SL ORPHAN] get_positions() failed: %s — orphan check skipped this tick", _oap_err)

                for _orphan_tr in list(active_trades):
                    _orphan_asset = _orphan_tr.get("asset")
                    _orphan_sl_oid = _orphan_tr.get("sl_oid")
                    _orphan_sl_px = _orphan_tr.get("sl_price")
                    _orphan_is_long = _orphan_tr.get("is_long", True)
                    _orphan_size = float(_orphan_tr.get("amount") or 0)

                    # Only process trades that were deferred (sl_oid is None) and have an SL price
                    if _orphan_sl_oid is not None or not _orphan_sl_px or _orphan_size <= 0:
                        continue

                    if _orphan_all_positions is None:
                        continue  # positions fetch failed this tick — skip

                    try:
                        _orphan_positions = _orphan_all_positions
                        _orphan_pos_exists = any(
                            p.get("coin") == _orphan_asset and abs(float(p.get("szi") or 0)) > 0
                            for p in (_orphan_positions or [])
                        )
                        if not _orphan_pos_exists:
                            continue  # limit not filled yet — skip

                        logging.warning(
                            "[SL ORPHAN] %s has open position but no SL (deferred from limit entry) — placing SL NOW at %s",
                            _orphan_asset, _orphan_sl_px
                        )
                        await send_alert(
                            f"⚠️ [SL ORPHAN] {_orphan_asset} open position found with no SL — placing SL at {_orphan_sl_px} NOW"
                        )
                        _orphan_sl_resp = await hyperliquid.place_stop_loss(
                            _orphan_asset, _orphan_is_long, _orphan_size, float(_orphan_sl_px)
                        )
                        _orphan_sl_new_oid = (hyperliquid.extract_oids(_orphan_sl_resp) or [None])[0]
                        for _otr in active_trades:
                            if _otr.get("asset") == _orphan_asset:
                                _otr["sl_oid"] = _orphan_sl_new_oid
                        save_active_trades(active_trades)
                        logging.info(
                            "[SL ORPHAN] %s SL placed successfully at %s (oid=%s)",
                            _orphan_asset, _orphan_sl_px, _orphan_sl_new_oid
                        )
                        add_event(f"[SL ORPHAN FIXED] {_orphan_asset} SL placed at {_orphan_sl_px} (oid={_orphan_sl_new_oid})")
                    except Exception as _orphan_err:
                        logging.error(
                            "[SL ORPHAN] %s — failed to place orphan SL: %s — MANUAL CHECK REQUIRED",
                            _orphan_asset, _orphan_err
                        )
                        await send_alert(
                            f"🚨 [SL ORPHAN FAIL] {_orphan_asset} — could not place SL at {_orphan_sl_px}. CHECK MANUALLY on Hyperliquid."
                        )
                # ── End SL orphan check ──────────────────────────────────────────────

                # ── PENDING LIMIT CANCEL — cancel unfilled limits after 1 candle (5 min) ──
                # BUG-P11-LIMIT FIX: Limit orders are NOT cancelled at entry time (3s is too short).
                # Instead, each inner tick checks trades with a pending entry_oid.
                # If the position opened → clear the pending flags (SL orphan check places TP/SL).
                # If still unfilled after ≥4 min → cancel the limit and free the slot.
                for _pl_tr in list(active_trades):
                    _pl_oid   = _pl_tr.get("entry_oid")
                    _pl_at    = _pl_tr.get("limit_placed_at")
                    _pl_asset = _pl_tr.get("asset")
                    if not _pl_oid or not _pl_at or not _pl_asset:
                        continue
                    try:
                        _pl_age_s = (datetime.now(timezone.utc) - datetime.fromisoformat(_pl_at)).total_seconds()
                    except Exception:
                        _pl_age_s = 999
                    if _pl_age_s < 240:  # give at least 4 min before cancelling (inner tick is 5 min)
                        continue
                    # Check whether the limit filled and opened a position.
                    # MED-2/PERF-2 FIX: Reuse _orphan_all_positions fetched above instead of
                    # issuing a second get_positions() call — eliminates duplicate REST call
                    # that could exhaust rate budget before the more time-sensitive cancel check.
                    if _orphan_all_positions is not None:
                        _pl_positions = _orphan_all_positions
                        _pl_pos_exists = any(
                            p.get("coin") == _pl_asset and abs(float(p.get("szi") or 0)) > 0
                            for p in _pl_positions
                        )
                    else:
                        try:
                            _pl_positions = await hyperliquid.get_positions()
                            _pl_pos_exists = any(
                                p.get("coin") == _pl_asset and abs(float(p.get("szi") or 0)) > 0
                                for p in (_pl_positions or [])
                            )
                        except Exception:
                            _pl_pos_exists = False
                    if _pl_pos_exists:
                        # Position opened — limit filled between ticks. Clear pending flags.
                        # SL orphan check (above this block) will place TP/SL on next tick.
                        for _upd_tr in active_trades:
                            if _upd_tr.get("asset") == _pl_asset:
                                _upd_tr["entry_oid"] = None
                                _upd_tr["limit_placed_at"] = None
                        save_active_trades(active_trades)
                        logging.info("[LIMIT PENDING] %s position confirmed open — pending flags cleared, SL orphan check covers TP/SL", _pl_asset)
                        add_event(f"[LIMIT PENDING] {_pl_asset} limit filled — position open, TP/SL to be placed")
                        continue
                    # Position still absent after ≥5 min — limit never filled; cancel and free slot
                    try:
                        await hyperliquid.cancel_order(_pl_asset, _pl_oid)
                        logging.info("[LIMIT PENDING] %s unfilled limit cancelled after %.0fs — slot freed", _pl_asset, _pl_age_s)
                        add_event(f"[LIMIT PENDING] {_pl_asset} unfilled limit cancelled after {int(_pl_age_s)}s — re-evaluating next cycle")
                    except Exception as _pl_ce:
                        logging.warning("[LIMIT PENDING] %s cancel failed: %s", _pl_asset, _pl_ce)
                    try:
                        active_trades.remove(_pl_tr)
                    except ValueError:
                        pass
                    state_mgr.clear_entry(_pl_asset)
                    state_mgr.set_state(_pl_asset, TradeStateMachine.IDLE)  # BUG-v7-L3 FIX: clear_entry() only removes entry_time, not state
                    save_active_trades(active_trades)
                # ── End pending limit cancel ──────────────────────────────────────────

                # ── Lightweight reconcile — detect TP/SL fills between outer cycles ──
                # Full reconciliation runs in the outer loop. This inner-loop check only
                # looks for positions that DISAPPEARED since last check (TP/SL filled).
                # Does NOT re-place orders — just updates local state and starts cooldown.
                try:
                    _inner_positions = await hyperliquid.get_positions()
                    _inner_open_coins = {
                        p.get("coin")
                        for p in (_inner_positions or [])
                        if abs(float(p.get("szi") or 0)) > 0
                    }
                    for _recon_tr in list(active_trades):
                        _recon_asset = _recon_tr.get("asset")
                        # Skip pending limit orders — position may not exist yet
                        if _recon_tr.get("entry_oid"):
                            continue
                        if _recon_asset and _recon_asset not in _inner_open_coins:
                            logging.info(
                                "[INNER RECON] %s position no longer open — marking closed, starting cooldown",
                                _recon_asset
                            )
                            # BUG-P11-2 FIX: Log the close so diary.jsonl and stats.json are updated
                            # for every mid-cycle TP/SL hit. Without this, 55 of every 60 minutes
                            # of exits were invisible to trade history and Claude context.
                            try:
                                await _log_trade_close(_recon_tr, "unknown")
                            except Exception as _ilc_err:
                                logging.warning("[INNER RECON] _log_trade_close failed for %s: %s", _recon_asset, _ilc_err)
                            active_trades.remove(_recon_tr)
                            state_mgr.start_cooldown(
                                _recon_asset,
                                interval_seconds=int(CONFIG.get("cooldown_minutes") or 30) * 60
                            )
                            save_active_trades(active_trades)
                            add_event(f"[INNER RECON] {_recon_asset} closed (TP/SL hit) — logged, slot freed, cooldown started")
                except Exception as _inner_recon_err:
                    logging.debug("[INNER RECON] position check failed: %s", _inner_recon_err)
                # ── End lightweight reconcile ────────────────────────────────────────

                # PERF-2 FIX: Refresh account value and prices BEFORE scoring/sizing loop.
                # Previously this refresh happened after the scoring loop, so sizing at lines
                # _i_buying_power/_i_atr_sized used account_value that was up to 55 min stale.
                try:
                    _istate_pre = await hyperliquid.get_user_state()
                    _iaccval_pre = float(_istate_pre.get("total_value", 0))
                    if _iaccval_pre > 0:
                        account_value = _iaccval_pre
                        state = _istate_pre
                    for _ri_asset in args.assets:
                        try:
                            _rip = await hyperliquid.get_current_price(_ri_asset)
                            if _rip > 0:
                                asset_prices[_ri_asset] = _rip
                                for _rms in market_sections:
                                    if _rms.get("asset") == _ri_asset:
                                        _rms["current_price"] = _rip
                        except Exception:
                            pass
                except Exception as _ire_pre:
                    logging.warning("[INNER tick %d] pre-score state/price refresh failed: %s", _tick + 1, _ire_pre)

                # Refresh 5m candles and recompute trigger_5m per asset
                for _i_asset in args.assets:
                    try:
                        _f5m = await hyperliquid.get_candles(_i_asset, "5m", 100)  # 100 candles sufficient for BB(20)+StochRSI(14,14) warmup
                        if not _f5m:
                            continue
                        _i5m = compute_all(_f5m)
                        asset_candles_5m[_i_asset] = _f5m
                        for _ms in market_sections:
                            if _ms.get("asset") != _i_asset:
                                continue
                            _ms["trigger_5m"] = {
                                "macd_histogram": round_or_none(latest(_i5m.get("macd_histogram", [])), 4),
                                "rsi14":          round_or_none(latest(_i5m.get("rsi14", [])), 2),
                                "candle_bullish": _f5m[-1]["close"] > _f5m[-1]["open"],
                            }
                            _ms["candles_5m"] = _f5m
                            break
                    except Exception as _i5e:
                        logging.warning("[INNER] 5m refresh %s: %s", _i_asset, _i5e)

                # Re-run BB+StochRSI pipeline with fresh 5m data
                _inner_outputs: dict = {"reasoning": "", "trade_decisions": []}
                _isess_ok, _isess_reason = session_gate_ok(CONFIG)
                for _i_asset in args.assets:
                    _iac = next((m for m in market_sections if m.get("asset") == _i_asset), None)
                    if not _iac:
                        continue
                    if risk_mgr.circuit_breaker_active:
                        logging.info("[INNER CB] circuit breaker active — skipping %s tick %d", _i_asset, _tick + 1)
                        continue
                    if _daily_trade_count >= int(CONFIG.get("max_daily_trades") or 40):
                        break
                    if not _isess_ok:
                        continue
                    _cd = _sl_cooldown_map.get(_i_asset)
                    if _cd and datetime.now(timezone.utc) < _cd:
                        continue
                    if is_trend_regime_active(_iac, CONFIG):
                        continue
                    _i_candles_5m = _iac.get("candles_5m", [])
                    _isig = compute_bb_stochrsi_signal(_i_candles_5m, CONFIG)
                    _idir = _isig.get("signal", "NONE")
                    if _idir == "NONE":
                        continue
                    _idir_str = "buy" if _idir == "LONG" else "sell"
                    _ifund = float(_iac.get("funding_rate") or 0)
                    if not funding_rate_ok(_ifund, _idir_str):
                        continue
                    _ie = float(asset_prices.get(_i_asset) or _iac.get("current_price") or 0)
                    _iatr_5m = float(_isig.get("atr") or 0)
                    _ibb_mid = float(_isig.get("bb_mid") or 0)
                    _ifee_buf = _ie * float(CONFIG.get("taker_fee_pct") or 0.00045) * 2
                    if _ie <= 0 or _iatr_5m <= 0 or _ibb_mid <= 0:
                        continue
                    if _idir_str == "buy":
                        _itp = round(_ibb_mid, 6)
                        _isl = round(_ie - 1.5 * _iatr_5m, 6)
                        if _itp <= _ie + _ifee_buf:
                            continue
                    else:
                        _itp = round(_ibb_mid, 6)
                        _isl = round(_ie + 1.5 * _iatr_5m, 6)
                        if _itp >= _ie - _ifee_buf:
                            continue
                    _itp1 = _itp
                    _itp2 = _itp
                    _i_buying_power = account_value * float(CONFIG.get("max_leverage") or 5)
                    _i_pct_cap = _i_buying_power * (float(CONFIG.get("max_position_pct") or 15) / 100.0)
                    _i_atr_sized = risk_mgr.atr_position_size(account_value, _ie, _isl)
                    _ialloc = min(_i_pct_cap, _i_atr_sized)
                    # Anomaly check
                    _i_chg_5c = 0.0
                    if len(_i_candles_5m) >= 5:
                        _ip_now = float(_i_candles_5m[-1]["close"])
                        _ip_5a = float(_i_candles_5m[-5]["close"])
                        if _ip_5a > 0:
                            _i_chg_5c = abs(_ip_now - _ip_5a) / _ip_5a * 100
                    _i_anm_pct = float(CONFIG.get("anomaly_trigger_pct") or 3.0)
                    if _i_chg_5c > _i_anm_pct:
                        _iverd = await asyncio.to_thread(
                            agent.claude_anomaly_check, _i_asset, _i_chg_5c, _idir_str)
                        if _iverd != "APPROVE":
                            continue
                    _ilim_px = round(_ie * (1 - 0.0015), 6) if _idir_str == "buy" else round(_ie * (1 + 0.0015), 6)
                    _inner_outputs["trade_decisions"].append({
                        "asset": _i_asset, "action": _idir_str,
                        "allocation_usd": _ialloc, "order_type": "limit",
                        "limit_price": _ilim_px, "tp_price": _itp, "tp1_price": _itp1, "tp2_price": _itp2,
                        "sl_price": _isl, "atr14": _iatr_5m, "current_price": _ie,
                        "score": 8.0,
                        "exit_plan": f"inner BB-rev TP={_itp:.4f} SL={_isl:.4f} K={_isig.get('stochrsi_k')}",
                        "rationale": f"inner BB+StochRSI K={_isig.get('stochrsi_k')} {_idir_str}",
                    })

                # Execute inner-loop trades (state gate + spread + candle gate + risk)
                for _iout in _inner_outputs.get("trade_decisions", []):
                    _ia = _iout.get("asset")
                    if not _ia or _iout.get("action") not in ("buy", "sell"):
                        continue
                    try:
                        _ism = state_mgr.get_state(_ia)
                        if _ism in ("ENTERED", "COOLDOWN"):
                            continue
                        _iprice = asset_prices.get(_ia, 0)
                        if not _iprice or _iprice <= 0:
                            continue
                        _iact_ctx = next((m for m in market_sections if m.get("asset") == _ia), {})
                        # Spread check
                        _ispread = float(_iact_ctx.get("spread_pct") or 0)
                        if _ispread > 0.15:
                            logging.info("[INNER SPREAD] %s blocked — spread %.3f%%", _ia, _ispread)
                            continue
                        # Candle gate — need at least 30% of 5m candle complete
                        _i_secs_into_5m = (datetime.now(timezone.utc).minute % 5) * 60 + datetime.now(timezone.utc).second
                        if (_i_secs_into_5m / 300) < 0.30:
                            logging.info("[INNER CANDLE GATE] %s candle only %.0f%% complete", _ia, _i_secs_into_5m / 3)
                            continue
                        _iout["current_price"] = _iprice
                        # atr14 already set from 5m signal
                        _iallowed, _ireason, _iout = risk_mgr.validate_trade(_iout, state, initial_account_value or 0)
                        if not _iallowed:
                            add_event(f"[INNER RISK] {_ia}: {_ireason}")
                            continue
                        _iamt = float(_iout["allocation_usd"]) / _iprice
                        # P7-HIGH-1 / P8-HIGH-1: Second concurrent-position gate immediately before order.
                        # Uses max(exchange count, active_trades count) — state.get("positions") misses
                        # same-tick entries placed before this asset but not yet visible on exchange.
                        _i_max_conc = int(CONFIG.get("max_concurrent_positions") or 3)
                        _i_open_pos = max(
                            sum(1 for _ip in (state.get("positions") or []) if abs(float(_ip.get("szi") or 0)) > 0),
                            len(active_trades),
                        )
                        if _i_open_pos >= _i_max_conc:
                            add_event(f"[INNER CONC GATE] {_ia} blocked — {_i_open_pos}/{_i_max_conc} positions open")
                            logging.info("[INNER CONC GATE] %s blocked — %d/%d concurrent positions", _ia, _i_open_pos, _i_max_conc)
                            continue
                        # BONUS-2: Use LIMIT orders in inner loop
                        _i_order_type = _iout.get("order_type", "market")
                        _i_limit_px = _iout.get("limit_price")
                        if _i_order_type == "limit" and _i_limit_px:
                            _i_limit_px = float(_i_limit_px)
                            _iorder = await (hyperliquid.place_limit_buy(_ia, _iamt, _i_limit_px)
                                             if _iout["action"] == "buy"
                                             else hyperliquid.place_limit_sell(_ia, _iamt, _i_limit_px))
                        else:
                            _iorder = await (hyperliquid.place_buy_order(_ia, _iamt)
                                             if _iout["action"] == "buy"
                                             else hyperliquid.place_sell_order(_ia, _iamt))
                        # H-5 FIX: Poll fills to get confirmed quantity before placing TP/SL.
                        # Old code used requested _iamt directly; partial fills left remainder unprotected.
                        _i_entry_oids = hyperliquid.extract_oids(_iorder) if _iorder else []
                        _i_entry_oid = _i_entry_oids[0] if _i_entry_oids else None
                        _i_filled_qty = 0.0
                        _i_seen_tids: set = set()
                        for _iap in range(3):
                            await asyncio.sleep(1)
                            try:
                                _i_fills = await hyperliquid.get_recent_fills(limit=20)
                                for _ifc in _i_fills:
                                    _ifc_oid = _ifc.get('oid') or _ifc.get('orderId')
                                    if not (_i_entry_oid and _ifc_oid and str(_ifc_oid) == str(_i_entry_oid)):
                                        continue
                                    _ifc_tid = str(_ifc.get('tid') or _ifc.get('tradeId') or
                                                   f"{_ifc_oid}_{_ifc.get('sz')}_{_ifc.get('time')}")
                                    if _ifc_tid in _i_seen_tids:
                                        continue
                                    _i_seen_tids.add(_ifc_tid)
                                    _i_filled_qty += float(_ifc.get('sz') or _ifc.get('size') or 0)
                            except Exception:
                                pass
                        # BONUS-2: Cancel unfilled inner limit after 1 candle
                        if _i_order_type == "limit" and _i_filled_qty == 0 and _i_entry_oid:
                            try:
                                await hyperliquid.cancel_order(_ia, _i_entry_oid)
                                logging.info("[INNER LIMIT] %s unfilled limit cancelled", _ia)
                                state_mgr.set_state(_ia, TradeStateMachine.IDLE)
                                continue
                            except Exception as _ice:
                                logging.warning("[INNER LIMIT] %s cancel failed: %s — checking exchange for position", _ia, _ice)
                                # BUG-v9-L3 FIX: Cancel failed — we don't know if the order filled.
                                # Check the exchange before deciding whether to record as ENTERED.
                                # If no position exists, skip (don't record a phantom entry).
                                # If a position exists, fall through and let the SL orphan check
                                # place the SL on the next inner tick.
                                try:
                                    _ice_positions = await hyperliquid.get_positions()
                                    _ice_pos = any(
                                        p.get("coin") == _ia and abs(float(p.get("szi") or 0)) > 0
                                        for p in (_ice_positions or [])
                                    )
                                    if not _ice_pos:
                                        logging.info("[INNER LIMIT] %s no position found after cancel failure — aborting entry", _ia)
                                        state_mgr.set_state(_ia, TradeStateMachine.IDLE)
                                        continue
                                    # LB-v10-4 FIX: Use actual exchange position size, not
                                    # original order size. Cancel may have partially propagated
                                    # leaving a partial fill — using _iamt would size SL wrong.
                                    _ice_pos_obj = next(
                                        (p for p in (_ice_positions or []) if p.get("coin") == _ia),
                                        None,
                                    )
                                    _ice_actual_sz = abs(float((_ice_pos_obj or {}).get("szi") or 0))
                                    if _ice_actual_sz > 0:
                                        _i_filled_qty = _ice_actual_sz
                                        logging.info("[INNER LIMIT] %s cancel-fail: using exchange position size %.6f for SL", _ia, _ice_actual_sz)
                                    logging.warning("[INNER LIMIT] %s position exists despite cancel failure — recording as ENTERED (SL orphan check covers)", _ia)
                                    await send_alert(f"⚠️ [{_ia}] limit cancel failed but position open — SL deferred to orphan check next tick")
                                except Exception as _ice2:
                                    logging.warning("[INNER LIMIT] %s position check failed after cancel failure: %s — skipping entry to be safe", _ia, _ice2)
                                    state_mgr.set_state(_ia, TradeStateMachine.IDLE)
                                    continue
                        # HIGH-1: Cancel partial fill remainder (mirrors outer loop CRITICAL-5 fix)
                        if _i_order_type == "limit" and 0 < _i_filled_qty < _iamt * 0.99 and _i_entry_oid:
                            try:
                                await hyperliquid.cancel_order(_ia, _i_entry_oid)
                                logging.info("[INNER LIMIT] %s partial fill %.6f/%.6f — unfilled remainder cancelled", _ia, _i_filled_qty, _iamt)
                            except Exception as _ipce:
                                logging.warning("[INNER LIMIT] %s partial fill cancel failed: %s", _ia, _ipce)
                        _itp_sl_size = _i_filled_qty if _i_filled_qty > 0 else _iamt
                        _itp1_oid = None
                        _itp2_oid = None
                        _itp_oid = None
                        _isl_oid = None
                        _i_is_buy = _iout["action"] == "buy"
                        _i_half = round(_itp_sl_size / 2, 6)
                        # MEDIUM-1 / P3-H1: Only place TP/SL when fill is confirmed.
                        # When _i_can_place_tpsl is False (cancel failed on an unfilled limit),
                        # fall through to record the trade in active_trades + state_mgr so the
                        # guardian monitors it next cycle — do NOT `continue` (that leaves the
                        # position untracked and allows a double-entry on the next inner tick).
                        _i_can_place_tpsl = _i_filled_qty > 0 or _i_order_type != "limit"
                        if not _i_can_place_tpsl:
                            logging.info("[INNER LIMIT] %s TP/SL deferred — limit order not yet filled, guardian covers next cycle", _ia)
                        if _i_can_place_tpsl and _iout.get("tp1_price") and _i_half > 0:
                            _itp1_res = await hyperliquid.place_take_profit(_ia, _i_is_buy, _i_half, _iout["tp1_price"])
                            _itp1_oid = (hyperliquid.extract_oids(_itp1_res) or [None])[0]
                        if _i_can_place_tpsl and _iout.get("tp2_price") and _i_half > 0:
                            _itp2_res = await hyperliquid.place_take_profit(_ia, _i_is_buy, _i_half, _iout["tp2_price"])
                            _itp2_oid = (hyperliquid.extract_oids(_itp2_res) or [None])[0]
                        if _i_can_place_tpsl and not _itp1_oid and _iout.get("tp_price"):
                            _itp_res = await hyperliquid.place_take_profit(_ia, _i_is_buy, _itp_sl_size, _iout["tp_price"])
                            _itp_oid = (hyperliquid.extract_oids(_itp_res) or [None])[0]
                        if _i_can_place_tpsl and _iout.get("sl_price"):
                            try:
                                _isl_res = await hyperliquid.place_stop_loss(_ia, _i_is_buy, _itp_sl_size, _iout["sl_price"])
                                _isl_oid = (hyperliquid.extract_oids(_isl_res) or [None])[0]
                            except Exception as _isl_err:
                                logging.critical("[INNER SL FAIL] %s — SL failed: %s — market-closing", _ia, _isl_err)
                                try:
                                    await hyperliquid.market_close(_ia)
                                    await hyperliquid.cancel_all_orders(_ia)
                                except Exception:
                                    pass
                                continue
                        active_trades.append({
                            "asset": _ia, "is_long": _i_is_buy,
                            "amount": _itp_sl_size, "half_size": _i_half,
                            "entry_price": _iprice, "entry_atr": float(_iout.get("atr14") or 0),
                            "tp_price": _iout.get("tp_price"), "tp1_price": _iout.get("tp1_price"),
                            "tp2_price": _iout.get("tp2_price"), "sl_price": _iout.get("sl_price"),
                            "tp_oid": _itp_oid, "tp1_oid": _itp1_oid, "tp2_oid": _itp2_oid, "sl_oid": _isl_oid,
                            "tp1_hit": False, "trail_breakeven_done": False, "trail_active": False,
                            "exit_plan": _iout.get("exit_plan", ""),
                            "funding_rate": float(_iact_ctx.get("funding_rate") or 0),
                            "opened_at": datetime.now(timezone.utc).isoformat(),
                        })
                        # CRITICAL-8: record_entry (state.json) BEFORE save_active_trades
                        state_mgr.record_entry(_ia)
                        save_active_trades(active_trades)
                        # CHAOS-v10-6 FIX: Re-check circuit breaker after each trade so a loss
                        # that trips the breaker mid-loop blocks all remaining sibling assets.
                        if risk_mgr.circuit_breaker_active:
                            logging.warning("[INNER CB] circuit breaker tripped after %s trade — aborting inner loop", _ia)
                            break
                        # MEDIUM-2: refresh date inside inner loop to handle UTC midnight crossing
                        _today_utc = datetime.now(timezone.utc).date()
                        if _today_utc != _last_daily_reset:
                            _daily_trade_count = 0
                            _last_daily_reset = _today_utc
                            _save_daily_count(0, str(_today_utc))
                        _daily_trade_count += 1
                        _save_daily_count(_daily_trade_count, str(_today_utc))  # CRITICAL-7
                        add_event(f"[INNER] {_iout['action'].upper()} {_ia} amt={_itp_sl_size:.4f} filled={_i_filled_qty:.4f} score={_iout.get('rationale','')} daily={_daily_trade_count}")
                        _i_diary_entry = {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "asset": _ia, "action": _iout["action"],
                            "order_type": _i_order_type, "limit_price": _i_limit_px,
                            "allocation_usd": float(_iout["allocation_usd"]),
                            "amount": _itp_sl_size, "filled_qty": _i_filled_qty, "entry_price": _iprice,
                            "tp_price": _iout.get("tp_price"),
                            "tp1_price": _iout.get("tp1_price"),
                            "tp2_price": _iout.get("tp2_price"),
                            "sl_price": _iout.get("sl_price"),
                            "exit_plan": _iout.get("exit_plan", ""),
                            "rationale": _iout.get("rationale", ""),
                            "inner_tick": _tick + 1,
                        }
                        with open(diary_path, "a") as _idf:
                            _idf.write(json.dumps(_i_diary_entry) + "\n")
                        # P3-M2: Update per-asset diary index so guardian reads O(1) not O(n)
                        _diary_index[_ia] = _i_diary_entry
                        _save_diary_index(_diary_index)
                        # P4-E-1: Log inner-loop approved trades to signals.jsonl for win-rate analysis
                        try:
                            with open(_signals_path, "a", encoding="utf-8") as _isf:
                                _isf.write(json.dumps({
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "asset": _ia,
                                    "direction": _iout["action"],
                                    "score": round(_iout.get("score", _iscr), 2),
                                    "action": "queued_inner",
                                    "reason": f"inner score={_iout.get('score', _iscr):.1f} tick={_tick+1}",
                                    "trend_4h": _iact_ctx.get("trend_4h"),
                                    "trend_1h": _iact_ctx.get("trend_1h"),
                                }) + "\n")
                        except Exception:
                            pass
                    except Exception as _ie2:
                        add_event(f"[INNER] execution error {_ia}: {_ie2}")
            # ── end 5-minute inner loop ────────────────────────────────────────────

    async def handle_diary(request):
        """Return diary entries as JSON — reads from decisions.jsonl for rich data."""
        try:
            limit = min(int(request.query.get('limit', '200')), 5000)  # V6-LOW-4: cap prevents OOM on large diary files

            entries = []

            # Primary: decisions.jsonl has account_value, positions, reasoning, decisions
            if os.path.exists(decisions_path):
                with open(decisions_path, "r") as f:
                    lines = f.readlines()
                start = max(0, len(lines) - limit)
                for line in lines[start:]:
                    try:
                        entries.append(json.loads(line))
                    except Exception:
                        pass

            # Fallback: plain diary.jsonl
            if not entries and os.path.exists(diary_path):
                with open(diary_path, "r") as f:
                    lines = f.readlines()
                start = max(0, len(lines) - limit)
                for line in lines[start:]:
                    try:
                        entries.append(json.loads(line))
                    except Exception:
                        pass

            return web.json_response(entries)
        except Exception as e:
            return web.json_response([], status=200)

    _ALLOWED_LOG_FILES = frozenset({
        'llm_requests.log',
        'prompts.log',
        'diary.jsonl',
        'decisions.jsonl',
        'risk_state.json',
    })
    # Project root: parent of the src/ directory this file lives in
    _LOG_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    async def handle_logs(request):
        """Stream log files with optional download or tailing behaviour.

        Only filenames explicitly listed in _ALLOWED_LOG_FILES are served.
        Path traversal attempts (e.g. '../../.env', '/etc/passwd') are
        rejected before any filesystem access.
        """
        requested = request.query.get('path', 'llm_requests.log')
        # Reject anything with a directory component — bare filename only
        if os.path.basename(requested) != requested or requested not in _ALLOWED_LOG_FILES:
            logging.warning("[SECURITY] /logs blocked path=%r", requested)
            return web.Response(text="Forbidden", status=403)

        safe_path = os.path.join(_LOG_BASE_DIR, requested)
        try:
            download = request.query.get('download')
            limit_param = request.query.get('limit')
            if not os.path.exists(safe_path):
                return web.Response(text="", content_type="text/plain")
            with open(safe_path, "r", encoding="utf-8", errors="replace") as f:
                data = f.read()
            if download or (limit_param and (limit_param.lower() == 'all' or limit_param == '-1')):
                headers = {}
                if download:
                    headers["Content-Disposition"] = f"attachment; filename={requested}"
                return web.Response(text=data, content_type="text/plain", headers=headers)
            limit = int(limit_param) if limit_param else 2000
            return web.Response(text=data[-limit:], content_type="text/plain")
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_live(request):
        """Return real-time account state fetched directly from Hyperliquid."""
        try:
            state = await hyperliquid.get_user_state()
            positions = []
            for pos in state.get('positions', []):
                coin = pos.get('coin')
                try:
                    current_px = await hyperliquid.get_current_price(coin) if coin else None
                except Exception:
                    current_px = None
                positions.append({
                    "symbol": coin,
                    "quantity": round_or_none(pos.get('szi'), 6),
                    "entry_price": round_or_none(pos.get('entryPx'), 2),
                    "current_price": round_or_none(current_px, 2),
                    "liquidation_price": round_or_none(pos.get('liquidationPx') or pos.get('liqPx'), 2),
                    "unrealized_pnl": round_or_none(pos.get('pnl'), 4),
                    "leverage": pos.get('leverage')
                })
            open_orders = []
            try:
                for o in await hyperliquid.get_open_orders():
                    open_orders.append({
                        "coin": o.get('coin'),
                        "is_buy": o.get('isBuy'),
                        "size": round_or_none(o.get('sz'), 6),
                        "price": round_or_none(o.get('px'), 2),
                        "trigger_price": round_or_none(o.get('triggerPx'), 2),
                        "order_type": o.get('orderType')
                    })
            except Exception:
                pass
            recent_fills = []
            try:
                for f in await hyperliquid.get_recent_fills(limit=50):
                    t_raw = f.get('time') or f.get('timestamp')
                    ts = None
                    if t_raw:
                        try:
                            t_int = int(t_raw)
                            ts = datetime.fromtimestamp(t_int / 1000 if t_int > 1e12 else t_int, tz=timezone.utc).isoformat()
                        except Exception:
                            ts = str(t_raw)
                    recent_fills.append({
                        "timestamp": ts,
                        "coin": f.get('coin') or f.get('asset'),
                        "is_buy": f.get('isBuy'),
                        "size": round_or_none(f.get('sz') or f.get('size'), 6),
                        "price": round_or_none(f.get('px') or f.get('price'), 2)
                    })
            except Exception:
                pass
            _live_av = state.get('total_value') or 0.0
            _live_init = _load_initial_balance()
            _live_ret = round(((_live_av - _live_init) / _live_init * 100.0), 2) if _live_init and _live_av else None
            return web.json_response({
                "account_value":    round_or_none(_live_av, 2),
                "balance":          round_or_none(state.get('balance'), 2),
                "perps_value":      round_or_none(state.get('perps_value'), 2),
                "spot_usdc":        round_or_none(state.get('spot_usdc'), 2),
                "withdrawable":     round_or_none(state.get('withdrawable'), 2),
                "total_return_pct": _live_ret,   # server-authoritative return vs initial deposit
                "positions":        positions,
                "open_orders":      open_orders,
                "recent_fills":     recent_fills,
                "timestamp":        datetime.now(timezone.utc).isoformat()
            })
        except Exception as e:
            logging.error("handle_live error: %s", e)
            return web.json_response({"error": str(e)}, status=500)

    async def handle_fills(request):
        """Return full fill history from Hyperliquid (includes closedPnl, fee, side)."""
        try:
            fills = []
            if hasattr(hyperliquid.info, 'user_fills'):
                fills = await asyncio.to_thread(hyperliquid.info.user_fills, hyperliquid.query_address)
            elif hasattr(hyperliquid.info, 'fills'):
                fills = await asyncio.to_thread(hyperliquid.info.fills, hyperliquid.query_address)
            return web.json_response(fills if isinstance(fills, list) else [])
        except Exception as e:
            logging.error("handle_fills error: %s", e)
            return web.json_response([], status=200)

    async def handle_index(request):
        """Serve the trading dashboard HTML."""
        dashboard = pathlib.Path(__file__).parent.parent / 'dashboard.html'
        try:
            resp = web.Response(text=dashboard.read_text(encoding='utf-8'), content_type='text/html')
        except FileNotFoundError:
            resp = web.Response(text=f'<h1>dashboard.html not found at {dashboard}</h1>', content_type='text/html', status=404)
        # V6-LOW-3: Defensive security headers — low-risk given 127.0.0.1 default bind,
        # but important if API_HOST is ever set to expose the dashboard on a network interface.
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'unsafe-inline' https://cdn.jsdelivr.net; "
            "style-src 'unsafe-inline'; "
            "connect-src 'self'; "
            "img-src 'self' data:; "
            "font-src 'self';"
        )
        return resp

    async def handle_meta(request):
        """Return bot metadata: start date, auth enabled flag."""
        _meta_path = pathlib.Path(__file__).parent.parent / "bot_started.json"
        _started = None
        try:
            if _meta_path.exists():
                _started = json.loads(_meta_path.read_text())
        except Exception:
            pass
        return web.json_response({
            "bot_started": _started.get("started_at") if _started else None,
            "network": (CONFIG.get("hyperliquid_network") or "mainnet").upper(),
            "assets": args.assets or [],
        })

    async def handle_signals(request):
        """Return last N entries from signals.jsonl for the Signal Status panel."""
        try:
            limit = min(int(request.query.get('limit', '100')), 1000)
            sigs = []
            if os.path.exists(_signals_path):
                with open(_signals_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                for line in lines[max(0, len(lines) - limit):]:
                    try:
                        sigs.append(json.loads(line))
                    except Exception:
                        pass
            return web.json_response(sigs)
        except Exception as e:
            return web.json_response([], status=200)

    async def handle_stats(request):
        """Compute trade stats from diary.jsonl: win rate, avg profit, avg loss, profit factor, by exit type."""
        try:
            wins, losses, total_pnl = [], [], 0.0
            by_exit = {"tp": 0, "sl": 0, "time_limit": 0, "unknown": 0}
            if os.path.exists(diary_path):
                with open(diary_path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            e = json.loads(line)
                            if e.get("event") == "trade_closed" and e.get("realized_pnl") is not None:
                                pnl = float(e["realized_pnl"])
                                total_pnl = round(total_pnl + pnl, 4)
                                if pnl > 0:
                                    wins.append(pnl)
                                else:
                                    losses.append(pnl)
                                et = e.get("exit_type", "unknown")
                                by_exit[et] = by_exit.get(et, 0) + 1
                        except Exception:
                            pass
            total = len(wins) + len(losses)
            avg_win  = round(sum(wins) / len(wins), 4) if wins else 0
            avg_loss = round(sum(losses) / len(losses), 4) if losses else 0
            gross_profit = sum(wins)
            gross_loss   = abs(sum(losses))
            profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else None
            return web.json_response({
                "total_trades":   total,
                "wins":           len(wins),
                "losses":         len(losses),
                "win_rate":       round(len(wins) / total * 100, 1) if total else 0,
                "avg_win":        avg_win,
                "avg_loss":       avg_loss,
                "total_pnl":      round(total_pnl, 4),
                "profit_factor":  profit_factor,
                "by_exit_type":   by_exit,
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=200)

    async def handle_trades(request):
        """Return closed trade events from diary.jsonl for the Trade History tab."""
        try:
            limit = min(int(request.query.get('limit', '100')), 1000)
            trades = []
            if os.path.exists(diary_path):
                with open(diary_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                for line in lines:
                    try:
                        e = json.loads(line)
                        if e.get("event") == "trade_closed":
                            trades.append(e)
                    except Exception:
                        pass
            return web.json_response(trades[-limit:])
        except Exception as e:
            return web.json_response([], status=200)

    async def start_api(app):
        """Register HTTP endpoints for observing diary entries and logs."""
        app.router.add_get('/', handle_index)
        app.router.add_get('/meta', handle_meta)
        app.router.add_get('/diary', handle_diary)
        app.router.add_get('/live', handle_live)
        app.router.add_get('/fills', handle_fills)
        app.router.add_get('/logs', handle_logs)
        app.router.add_get('/signals', handle_signals)
        app.router.add_get('/stats', handle_stats)
        app.router.add_get('/trades', handle_trades)
        app.router.add_route('OPTIONS', '/{path_info:.*}', lambda r: web.Response())

    async def main_async():
        """Start the aiohttp server and kick off the trading loop."""

        # Persist bot start date once (survives diary rotation — used by dashboard "Active Since")
        _started_path = pathlib.Path(__file__).parent.parent / "bot_started.json"
        if not _started_path.exists():
            try:
                _tmp = str(_started_path) + ".tmp"
                with open(_tmp, "w", encoding="utf-8") as _sf:
                    json.dump({"started_at": datetime.now(timezone.utc).isoformat()}, _sf)
                os.replace(_tmp, str(_started_path))
            except Exception as _se:
                logging.warning("[BOOT] could not write bot_started.json: %s", _se)

        app = web.Application(middlewares=[cors_middleware, auth_middleware])
        await start_api(app)
        runner = web.AppRunner(app)
        await runner.setup()
        port = int(CONFIG.get("api_port"))
        host = CONFIG.get("api_host")  # defaults to 127.0.0.1 — localhost only
        # Access via SSH tunnel: ssh -L 3000:localhost:3000 user@server
        # To expose on a network interface, set API_HOST=0.0.0.0 in .env (not recommended)
        site = web.TCPSite(runner, host, port)
        await site.start()
        logging.info(f"API server started at http://{host}:{port}")
        await send_alert(f"\U0001f680 Trading bot started — assets: {args.assets} interval: {args.interval} port: {port}")
        try:
            await run_loop()
        finally:
            await send_alert("\U0001f6d1 Trading bot stopped.")

    def calculate_sharpe_from_diary(path: str) -> float:
        """Compute Sharpe ratio from realized P&L recorded in diary.jsonl.

        Reads only trade_closed events with a realized_pnl field — these are
        written by _log_trade_close() on every natural, forced, or timed close.
        Returns 0.0 when fewer than 3 closed trades are available.
        """
        returns: list[float] = []
        try:
            with open(path) as _f:
                for _line in _f:
                    try:
                        _e = json.loads(_line)
                        if _e.get('event') == 'trade_closed' and _e.get('realized_pnl') is not None:
                            returns.append(float(_e['realized_pnl']))
                    except Exception:
                        pass
        except FileNotFoundError:
            return 0.0
        if len(returns) < 3:
            return 0.0
        mean_r = sum(returns) / len(returns)
        std_r = (sum((r - mean_r) ** 2 for r in returns) / len(returns)) ** 0.5
        return round(mean_r / std_r if std_r > 0 else 0.0, 3)

    def _handle_signal(signum, frame):
        global _shutdown
        _shutdown = True
        _release_instance_lock()
        logging.info("[SHUTDOWN] Signal %d received", signum)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    try:
        asyncio.run(main_async())
    finally:
        _release_instance_lock()


if __name__ == "__main__":
    main()