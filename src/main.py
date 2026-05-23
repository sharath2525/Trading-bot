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
from src.strategy import entry_confirmed, market_filter, compute_signal_score, is_trending_regime, oi_confirmed
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


def _code_decide_direction(asset_data: dict) -> str | None:
    """Return 'buy', 'sell', or None from 4h/1h trend alignment plus ADX gate.

    Gates (all must pass before any direction is returned):
    1. 4h EMA20/50 trend alignment
    2. 1h trend agrees with 4h direction
    3. 1h ADX > 20 — must be trending, not ranging
    Returns None when any gate fails.
    """
    trend_4h = asset_data.get("trend_4h", "UNKNOWN")
    trend_1h = asset_data.get("trend_1h", "UNKNOWN")
    if trend_4h == "UNKNOWN":
        return None
    if trend_4h == "BULLISH" and trend_1h in ("BULLISH", "UNKNOWN"):
        _direction = "buy"
    elif trend_4h == "BEARISH" and trend_1h in ("BEARISH", "UNKNOWN"):
        _direction = "sell"
    else:
        return None  # conflicting trends

    # ADX gate — must be trending (ADX > 20) on 1h BEFORE scoring
    _adx_1h = asset_data.get("intraday_1h", {}).get("adx")
    if _adx_1h is not None and float(_adx_1h) < 20:
        logging.info(
            "[DIRECTION] %s blocked — 1h ADX %.1f < 20 (ranging, not trending)",
            asset_data.get("asset", "?"), float(_adx_1h)
        )
        print(f"[DIRECTION] {asset_data.get('asset', '?')} blocked — 1h ADX {float(_adx_1h):.1f} < 20 (ranging)")
        return None

    return _direction


def _code_compute_tpsl(entry: float, atr: float, direction: str, score: float = 7.0) -> tuple[float, float, float, float]:
    """Return (tp1, tp2, sl, tp_main) — score-adaptive TP with partial-close levels.

    tp1     = 1.0×ATR  — close 50% of position (lock profit)
    tp2     = 3.0×ATR  — let remaining 50% run
    tp_main = score-adaptive single TP level used for primary trigger order:
              score >= 10.0 → 2.5×ATR  (perfect base + volume bonus — highest conviction)
              score >= 9.0  → 2.2×ATR
              score >= 8.0  → 2.0×ATR
              score >= 7.0  → 1.8×ATR  (minimum passing setup)
    sl      = 1.0×ATR from entry always (MASTER RULE 4)
    Fee buffer baked into all levels. Code owns all TP/SL (MASTER RULE 2).
    Score scale is 0–11; tiers anchored to updated scale (MASTER RULES 2026-05-20).
    """
    fee_buffer = entry * float(CONFIG.get("taker_fee_pct") or 0.00045) * 2

    if score >= 10.0:
        tp_mult = 2.5
    elif score >= 9.0:
        tp_mult = 2.2
    elif score >= 8.0:
        tp_mult = 2.0
    elif score >= 7.0:
        tp_mult = 1.8
    else:
        tp_mult = 2.0

    if direction == "buy":
        tp1     = round(entry + 1.0 * atr + fee_buffer, 6)
        tp2     = round(entry + 3.0 * atr + fee_buffer, 6)
        tp_main = round(entry + tp_mult * atr + fee_buffer, 6)
        sl      = round(entry - 1.0 * atr - fee_buffer, 6)
    else:
        tp1     = round(entry - 1.0 * atr - fee_buffer, 6)
        tp2     = round(entry - 3.0 * atr - fee_buffer, 6)
        tp_main = round(entry - tp_mult * atr - fee_buffer, 6)
        sl      = round(entry + 1.0 * atr + fee_buffer, 6)

    return tp1, tp2, sl, tp_main


def multi_timeframe_confluence(asset_data: dict, direction: str, require_30m: bool = False) -> bool:
    """Return True when 4h, 1h, and 15m MACD all agree on direction.

    Reduced from 5 timeframes to 3: 4h (macro trend) + 1h (intraday trend) + 15m MACD
    (short-term momentum). 30m is correlated with 4h/1h (redundant). 5m candle direction
    is noise — bullish 5m candles occur 40-50% of the time even in strong downtrends.
    """
    is_buy = direction == "buy"

    # 4h: macro trend alignment
    trend_4h = asset_data.get("trend_4h", "UNKNOWN")
    if is_buy and trend_4h != "BULLISH":
        return False
    if not is_buy and trend_4h != "BEARISH":
        return False

    # 1h: intraday trend alignment
    trend_1h = asset_data.get("trend_1h", "UNKNOWN")
    if is_buy and trend_1h != "BULLISH":
        return False
    if not is_buy and trend_1h != "BEARISH":
        return False

    # 15m: short-term momentum confirmation via MACD histogram
    macd_15m = asset_data.get("setup_15m", {}).get("macd_histogram")
    if macd_15m is not None:
        if is_buy and macd_15m <= 0:
            return False
        if not is_buy and macd_15m >= 0:
            return False

    return True


def _build_confluence_fingerprint(asset: str, direction: str,
                                   trend_4h: str, trend_1h: str, score: float) -> str:
    """Cache key that changes when market character genuinely shifts.

    score_bucket rounds to nearest 0.5 — avoids cache misses from tiny score drift.
    date_hour component expires the key after ~1 hour regardless.
    """
    score_bucket = round(score * 2) / 2
    date_hour = datetime.now(timezone.utc).strftime("%Y%m%d%H")
    return f"{asset}_{direction}_{trend_4h}_{trend_1h}_{score_bucket}_{date_hour}"


async def _fetch_macro_context() -> dict:
    """Fetch economic calendar events and news headlines from free RSS feeds.

    Timeout: 3 seconds per feed. Returns empty lists on any failure — never blocks trading.
    """
    import xml.etree.ElementTree as ET
    import aiohttp as _aiohttp

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

    try:
        async with _aiohttp.ClientSession() as _sess:
            for _url, _bucket, _prefix in _feeds:
                try:
                    async with _sess.get(
                        _url,
                        timeout=_aiohttp.ClientTimeout(total=3),
                        headers=_headers,
                    ) as _resp:
                        _text = await _resp.text()
                        _root = ET.fromstring(_text)
                        for _item in _root.iter("item"):
                            _title = (_item.findtext("title") or "").strip()
                            if _title:
                                context[_bucket].append(f"{_prefix} {_title}")
                            if len(context[_bucket]) >= 8:
                                break
                except Exception as _fe:
                    logging.debug("[MACRO] feed %s failed: %s", _url, _fe)
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


# ── Auth middleware (V4-HIGH-1) ───────────────────────────────────────────────
@web.middleware
async def auth_middleware(request, handler):
    """Bearer-token gate for all non-OPTIONS routes.

    Set DASHBOARD_TOKEN in .env to enable. Unset = no auth (backward compatible).
    The dashboard HTML reads the token from localStorage and sends it as
    'Authorization: Bearer <token>' on every API request.
    The root '/' route (dashboard HTML itself) is always allowed so operators
    can load the page to enter their token.
    """
    _token = CONFIG.get("dashboard_token")
    if not _token or request.method == "OPTIONS" or request.path == "/":
        return await handler(request)
    _auth = request.headers.get("Authorization", "")
    if _auth == f"Bearer {_token}":
        return await handler(request)
    return web.json_response({"error": "Unauthorized"}, status=401)
# ─────────────────────────────────────────────────────────────────────────────


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
    diary_path = "diary.jsonl"
    decisions_path = "decisions.jsonl"
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
            stats_path = "stats.json"
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
                    if exit_price and qty:
                        _exit_fee = exit_price * qty * 0.00045
                        _entry_fee = (entry_price or exit_price) * qty * 0.00045
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
                        # Closing fill is opposite direction to the entry
                        _fbuy = bool(_fill.get('isBuy', False))
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
                await send_alert("\U0001f6a8 KILLSWITCH activated — all trading halted. Remove the KILLSWITCH file to restart.")
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
            _rotate_if_needed("llm_requests.log")   # V3-HIGH-1: was missed in E-4 fix
            _rotate_if_needed("prompts.log")

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
                        add_event(
                            f"[TIMEOUT] {_asset_name} force-closing "
                            f"after {_max_hours}h — no progress"
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
                _g_amount   = float(_g_diary.get('amount') or 0)
                _g_tp_px    = _g_diary.get('tp_price')
                _g_tp1_px   = _g_diary.get('tp1_price')
                _g_tp2_px   = _g_diary.get('tp2_price')
                _g_sl_px    = _g_diary.get('sl_price')
                _g_half     = round(_g_amount / 2, 6)
                if _g_amount <= 0:
                    logging.warning("[GUARDIAN] %s: zero amount in diary — cannot re-place", _g_asset)
                    continue
                # P3-M1: Re-place TP1/TP2 (two-stage exit) when both are missing from exchange
                # Only re-place the two-stage pair when neither is present to avoid double-orders
                _g_has_tp1 = any(
                    o.get('coin') == _g_asset
                    and isinstance(o.get('orderType'), dict)
                    and (o.get('orderType', {}).get('trigger') or {}).get('tpsl') == 'tp'
                    for o in (open_orders or [])
                )
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
                if not _g_has_tp1 and _g_tp2_px and _g_half > 0:
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

                if not _tr_asset or _tr_entry <= 0 or _tr_atr <= 0 or _tr_size <= 0:
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
                        hyperliquid.get_candles(asset, "5m",  20),
                        hyperliquid.get_candles(asset, "1d",  50),
                    )
                    asset_prices[asset] = current_price
                    asset_candles_5m[asset] = candles_5m
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
                    near_ema_15m = (
                        abs(current_price - ema20_15m) / current_price < 0.003
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
                            "adx_trending": (adx_1h or 0) > 25,
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
                            "adx_trending": (adx_4h or 0) > 25,
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
                        "candles_5m": candles_5m,   # BUG-B: required for volume/pattern bonuses in compute_signal_score()
                        "daily_1d": candles_1d[-2] if len(candles_1d) >= 2 else (candles_1d[-1] if candles_1d else {}),  # BUG-C: use yesterday's closed candle not today's incomplete
                    })
                except Exception as e:
                    add_event(f"Data gather error {asset}: {e}")
                    continue

            # V3-HIGH-2 FIX: Removed dead `context_payload` build block.
            # The old code built a large JSON dict and serialized it every cycle, but it was
            # never sent to Claude — it was a remnant of the pre-2026-04-30 full-context design.
            # `confirm_trade()` writes its own prompts.log entry with the actual Claude input.
            # The misleading "Combined prompt length: X chars" log line is also removed.
            add_event(f"[CYCLE {invocation_count}] scoring {len(market_sections)} asset(s) — code-first pipeline")

            # ── Score-gated code-first pipeline (replaces unconditional Claude call) ──
            # Claude is called when score >= MIN_AI_SCORE and multi-timeframe confluence is confirmed.
            # All direction/TP/SL/size decisions are made by code.
            outputs = {"reasoning": "", "trade_decisions": []}
            _min_sig = float(CONFIG.get("min_signal_score") or 7)
            _max_dt  = int(CONFIG.get("max_daily_trades") or 10)

            def _make_hold(asset_name: str, reason: str) -> dict:
                return {"asset": asset_name, "action": "hold", "allocation_usd": 0.0,
                        "order_type": "market", "limit_price": None,
                        "tp_price": None, "sl_price": None,
                        "exit_plan": "", "rationale": reason}

            for _asset in args.assets:
                _ac = next((m for m in market_sections if m.get("asset") == _asset), None)
                if not _ac:
                    outputs["trade_decisions"].append(_make_hold(_asset, "no market data"))
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

                # 4h hard gate — direction must align with trend_4h or return None
                _direction = _code_decide_direction(_ac)
                if _direction is None:
                    _t4h = _ac.get('trend_4h', 'UNKNOWN')
                    _t1h = _ac.get('trend_1h', 'UNKNOWN')
                    _gate_label = "trend conflict" if _t4h != _t1h else "secondary gate blocked (ADX)"
                    outputs["trade_decisions"].append(_make_hold(
                        _asset,
                        f"4h gate: trend_4h={_t4h} trend_1h={_t1h} — {_gate_label}"
                    ))
                    continue

                # Weighted score gate (0-10 float)
                _score = compute_signal_score(_ac, _direction)
                # Kronos-mini forecast modifier (±0.5, code-only, clamped to [0,11] — MASTER RULE 1)
                try:
                    from src.indicators.kronos_forecast import get_kronos_modifier
                    _kmod = get_kronos_modifier(candles=_ac.get("candles_4h", []), direction=_direction)
                    if _kmod != 0.0:
                        logging.info("[KRONOS] %s modifier=%.1f score %.1f → %.1f", _asset, _kmod, _score, _score + _kmod)
                    _score = min(11.0, max(0.0, _score + _kmod))
                except ImportError:
                    pass
                if _score < _min_sig:
                    logging.info("[SCORE] %s %s score=%.1f < %.1f → HOLD", _asset, _direction, _score, _min_sig)
                    outputs["trade_decisions"].append(_make_hold(_asset, f"score={_score:.1f} < min {_min_sig:.0f}"))
                    # Signal logging — log every score≥7 signal including HOLDs, for 2-month win-rate analysis
                    if _score >= 7.0:
                        try:
                            with open("signals.jsonl", "a", encoding="utf-8") as _sf:
                                _sf.write(json.dumps({
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "asset": _asset,
                                    "direction": _direction,
                                    "score": round(_score, 2),
                                    "action": "hold" if _score < _min_sig else "pending",
                                    "reason": f"score={_score:.1f} min={_min_sig}",
                                    "trend_4h": _ac.get("trend_4h"),
                                    "trend_1h": _ac.get("trend_1h"),
                                }) + "\n")
                        except Exception:
                            pass
                    continue

                # Code computes trade parameters
                _entry = float(_ac.get("current_price") or 0)
                _atr   = float(_ac.get("long_term_4h", {}).get("atr14") or 0)
                if _entry <= 0 or _atr <= 0:
                    outputs["trade_decisions"].append(_make_hold(_asset, "missing price or ATR14"))
                    continue

                _tp1, _tp2, _sl, _tp = _code_compute_tpsl(_entry, _atr, _direction, _score)
                # Position sizing: target = 15% of buying_power (account × leverage).
                # ATR 1% risk rule acts as a safety ceiling — reduces size when SL is distant.
                _buying_power = account_value * float(CONFIG.get("max_leverage") or 5)
                _pct_cap = _buying_power * (float(CONFIG.get("max_position_pct") or 15) / 100.0)
                _atr_sized = risk_mgr.atr_position_size(account_value, _entry, _sl)
                _alloc = min(_pct_cap, _atr_sized)
                # Scale allocation by signal strength (min(score,10)/10): score-7 → 70%, score-10..11 → 100%
                # Scores above 10.0 do NOT grant above-100% sizing (MASTER RULE 2)
                _alloc = _alloc * (min(_score, 10.0) / 10.0)
                # ADX ranging market guard: half-size if ADX weak and score not at maximum
                _adx_1h_val = float(_ac.get("intraday_1h", {}).get("adx") or 25)
                _adx_thr    = float(CONFIG.get("adx_half_size_threshold") or 20)
                if _adx_1h_val < _adx_thr and _score < 9.0:
                    _alloc *= 0.5
                    logging.info("[SIZE] %s ADX %.1f < %.0f + score %.1f < 9 → half-size applied",
                                 _asset, _adx_1h_val, _adx_thr, _score)

                # Confluence gate: all timeframes must agree before calling Claude
                _confluence_ok = multi_timeframe_confluence(_ac, _direction)
                if not _confluence_ok:
                    logging.info("[CONFLUENCE] %s %s — TFs not aligned → HOLD", _asset, _direction)
                    outputs["trade_decisions"].append(_make_hold(_asset, "confluence failed — TFs not aligned"))
                    continue

                # MIN_AI_SCORE gate — separate from MIN_SIGNAL_SCORE so Claude call frequency is independently tunable
                _min_ai = float(CONFIG.get("min_ai_score") or 7)
                if _score < _min_ai:
                    logging.info("[AI GATE] %s score=%.1f < MIN_AI_SCORE %.1f → HOLD", _asset, _score, _min_ai)
                    outputs["trade_decisions"].append(_make_hold(_asset, f"score={_score:.1f} < MIN_AI_SCORE {_min_ai:.1f}"))
                    continue

                # Build fingerprint and check AI verdict cache
                _fingerprint = _build_confluence_fingerprint(
                    _asset, _direction,
                    _ac.get("trend_4h", "UNKNOWN"), _ac.get("trend_1h", "UNKNOWN"), _score
                )
                _now_utc = datetime.now(timezone.utc)
                _cache_entry = _ai_verdict_cache.get(_asset)
                _use_cached = False
                if _cache_entry:
                    _cfp     = _cache_entry.get("fingerprint")
                    _cexp    = _cache_entry.get("expires_at")
                    if _cfp == _fingerprint and _cexp and _now_utc < _cexp:
                        if _cache_entry.get("verdict") == "APPROVE":
                            logging.info("[AI CACHE] %s APPROVE (exp %s)", _asset, _cexp.isoformat())
                            _use_cached = True
                        else:
                            logging.info("[AI CACHE] %s REJECT → HOLD", _asset)
                            outputs["trade_decisions"].append(_make_hold(_asset, "AI cached REJECT"))
                            continue

                if not _use_cached:
                    # Hard minimum gap between Claude calls per asset
                    _gap_mins   = int(CONFIG.get("min_ai_call_gap_minutes") or 30)
                    _last_ts    = _last_ai_call_time.get(_asset, 0)
                    _gap_elapsed = time.time() - _last_ts
                    if _last_ts > 0 and _gap_elapsed < _gap_mins * 60:
                        _prev = _ai_verdict_cache.get(_asset)
                        if _prev and _prev.get("verdict") == "APPROVE":
                            logging.info("[AI GAP] %s — %.0f min gap (min %d) → using prev APPROVE",
                                         _asset, _gap_elapsed / 60, _gap_mins)
                            _use_cached = True
                        else:
                            logging.info("[AI GAP] %s — %.0f min gap (min %d) → HOLD",
                                         _asset, _gap_elapsed / 60, _gap_mins)
                            outputs["trade_decisions"].append(
                                _make_hold(_asset, f"AI call gap {_gap_elapsed/60:.0f}min < {_gap_mins}min"))
                            continue

                if not _use_cached:
                    # Call Claude with full market analysis context
                    _verdict = await asyncio.to_thread(
                        agent.confirm_trade, _asset, _direction, _entry, _tp, _sl, _score, {},
                        _macro_context_cache, _ac
                    )
                    _last_ai_call_time[_asset] = time.time()
                    _app_mins = int(CONFIG.get("ai_approve_cache_minutes") or 60)
                    _rej_mins = int(CONFIG.get("ai_reject_cache_minutes") or 30)
                    _cm = _app_mins if _verdict == "APPROVE" else _rej_mins
                    _ai_verdict_cache[_asset] = {
                        "verdict":     _verdict,
                        "fingerprint": _fingerprint,
                        "expires_at":  _now_utc + timedelta(minutes=_cm),
                    }
                    if _verdict != "APPROVE":
                        add_event(f"[CLAUDE] {_asset} score={_score:.1f} REJECTED by market analysis")
                        outputs["trade_decisions"].append(
                            _make_hold(_asset, f"AI REJECT score={_score:.1f}"))
                        continue

                add_event(f"[SCORE] {_asset} {_direction} score={_score:.1f} → queuing trade")
                # Signal logging — log approved trades to signals.jsonl for win-rate analysis (#78)
                try:
                    import json as _sjson
                    with open("signals.jsonl", "a", encoding="utf-8") as _sf:
                        _sf.write(_sjson.dumps({
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "asset": _asset,
                            "direction": _direction,
                            "score": round(_score, 2),
                            "action": "queued",
                            "reason": f"score={_score:.1f} APPROVED",
                            "trend_4h": _ac.get("trend_4h"),
                            "trend_1h": _ac.get("trend_1h"),
                        }) + "\n")
                except Exception:
                    pass
                # Limit order: buy 0.15% below ask, sell 0.15% above bid
                _lim_off = _entry * 0.0015
                _lim_px  = round(_entry - _lim_off, 6) if _direction == "buy" else round(_entry + _lim_off, 6)
                outputs["trade_decisions"].append({
                    "asset":        _asset,
                    "action":       _direction,
                    "allocation_usd": _alloc,
                    "order_type":   "limit",
                    "limit_price":  _lim_px,
                    "tp_price":     _tp,
                    "tp1_price":    _tp1,
                    "tp2_price":    _tp2,
                    "sl_price":     _sl,
                    "atr14":        _atr,
                    "current_price": _entry,
                    "score":        round(_score, 2),  # P6-HIGH-1: store per-asset score so execution loop uses correct value
                    "exit_plan":    f"code TP={_tp:.4f} TP1={_tp1:.4f} TP2={_tp2:.4f} SL={_sl:.4f} score={_score:.1f}",
                    "rationale":    (f"score={_score:.1f} trend_4h={_ac.get('trend_4h')} "
                                     f"trend_1h={_ac.get('trend_1h')}"),
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
                        # Block entirely when 4h EMA data is unavailable — UNKNOWN means
                        # insufficient candle history (startup or exchange gap). Both
                        # inversion guards below evaluate to False for UNKNOWN, so without
                        # this check Claude's action passes through with no validation.
                        if trend_4h == "UNKNOWN":
                            logging.warning(
                                "[TREND GUARD] %s blocked — trend_4h=UNKNOWN (insufficient 4h candle data)",
                                asset,
                            )
                            add_event(f"[TREND GUARD] {asset} {action} blocked — trend_4h=UNKNOWN, candle data insufficient")
                            continue
                        # Inversion assertion — fires if trend and order direction are opposite.
                        # BULLISH (EMA20>EMA50) must produce buy; BEARISH must produce sell.
                        # If this raises, an inversion is still present somewhere in the signal chain.
                        if trend_4h == "BULLISH" and action == "sell":
                            raise ValueError(f"INVERSION BUG DETECTED: {asset} trend=BULLISH but action=sell")
                        if trend_4h == "BEARISH" and action == "buy":
                            raise ValueError(f"INVERSION BUG DETECTED: {asset} trend=BEARISH but action=buy")

                        # Daily macro trend gate — block trades that fight the daily EMA cross.
                        # Only applied when the daily trend has actual momentum (ADX > 20).
                        # A near-cross with low ADX is a ranging market — blocking all longs
                        # or shorts would unnecessarily suppress valid intraday setups.
                        _trend_1d = asset_trends_1d.get(asset, "UNKNOWN")
                        _adx_1d = asset_adx_1d.get(asset)
                        _macro_trending = _adx_1d is not None and float(_adx_1d) > 20
                        if _macro_trending:
                            if _trend_1d == "BEARISH" and action == "buy":
                                logging.info(
                                    "[DAILY FILTER] %s BUY blocked — daily trend BEARISH ADX=%.1f",
                                    asset, float(_adx_1d) if _adx_1d else 0,
                                )
                                add_event(f"[DAILY FILTER] {asset} BUY skipped — daily trend BEARISH")
                                continue
                            if _trend_1d == "BULLISH" and action == "sell":
                                logging.info(
                                    "[DAILY FILTER] %s SELL blocked — daily trend BULLISH ADX=%.1f",
                                    asset, float(_adx_1d) if _adx_1d else 0,
                                )
                                add_event(f"[DAILY FILTER] {asset} SELL skipped — daily trend BULLISH")
                                continue

                        asset_ctx = next((m for m in market_sections if m.get("asset") == asset), {})
                        # Attach raw 5m candles for volume confirmation inside entry_confirmed.
                        # Done here, not in market_sections, so they are never serialised into
                        # the Claude context payload (which would waste tokens on raw OHLCV data).
                        # Inject score so market_filter() can apply the off-session score≥9 gate (#3)
                        asset_ctx_local = {**asset_ctx, "candles_5m": asset_candles_5m.get(asset, []), "_current_score": output.get("score", _score)}
                        # ATR spike + spread pre-flight — market_filter() was dead code (only
                        # called from make_decision() which is never invoked); wire it directly.
                        _btc_trend_1h = next(
                            (m.get("trend_1h", "UNKNOWN") for m in market_sections if m.get("asset") == "BTC"),
                            "UNKNOWN"
                        )
                        _btc_5m_candles = asset_candles_5m.get("BTC", [])
                        _mf_pass, _mf_reason = market_filter(
                            asset_ctx_local,
                            btc_trend_1h=_btc_trend_1h,
                            btc_candles_5m=_btc_5m_candles,
                            direction=action
                        )
                        if not _mf_pass:
                            logging.warning("[MARKET FILTER] %s %s blocked — %s", asset, action, _mf_reason)
                            add_event(f"[MARKET FILTER] {asset} {action} blocked — {_mf_reason}")
                            # P4-E-2: log filter-blocked signals so win-rate analysis captures all score≥7 signals
                            try:
                                import json as _fsjson
                                with open("signals.jsonl", "a", encoding="utf-8") as _fsf:
                                    _fsf.write(_fsjson.dumps({
                                        "timestamp": datetime.now(timezone.utc).isoformat(),
                                        "asset": asset, "direction": action,
                                        "score": round(output.get("score", _score), 2), "action": "filtered",
                                        "reason": f"market_filter: {_mf_reason}",
                                        "trend_4h": asset_ctx.get("trend_4h"),
                                        "trend_1h": asset_ctx.get("trend_1h"),
                                    }) + "\n")
                            except Exception:
                                pass
                            continue
                        # Entry confirmation (15m/5m layers + volume gate)
                        if not entry_confirmed(asset_ctx_local, action):
                            logging.info(
                                "[ENTRY] %s direction=%s blocked — "
                                "15m/5m not confirmed, waiting for pullback",
                                asset, action
                            )
                            add_event(f"[ENTRY] {asset} {action} blocked — 15m/5m not confirmed")
                            # P4-E-2: log entry-confirmation blocks
                            try:
                                import json as _ecjson
                                with open("signals.jsonl", "a", encoding="utf-8") as _ecf:
                                    _ecf.write(_ecjson.dumps({
                                        "timestamp": datetime.now(timezone.utc).isoformat(),
                                        "asset": asset, "direction": action,
                                        "score": round(output.get("score", _score), 2), "action": "filtered",
                                        "reason": "entry_confirmed: 15m/5m not confirmed",
                                        "trend_4h": asset_ctx.get("trend_4h"),
                                        "trend_1h": asset_ctx.get("trend_1h"),
                                    }) + "\n")
                            except Exception:
                                pass
                            continue
                        # Candle close gate — only fire when the 5m trigger candle has fully closed
                        # Prevents entering mid-candle on false signals that reverse before close
                        _now_sec = datetime.now(timezone.utc).second + datetime.now(timezone.utc).minute % 5 * 60
                        _secs_into_5m = (datetime.now(timezone.utc).minute % 5) * 60 + datetime.now(timezone.utc).second
                        _candle_age_pct = _secs_into_5m / 300  # 0.0 = just opened, 1.0 = about to close
                        if _candle_age_pct < 0.70:  # candle must be at least 70% complete (210 of 300s)
                            logging.info(
                                "[CANDLE GATE] %s waiting — 5m candle only %.0f%% complete",
                                asset, _candle_age_pct * 100
                            )
                            continue
                        # OI confirmation gate — requires OI increasing, blocks spikes
                        _oi_ok, _oi_reason = oi_confirmed(asset_ctx_local, action)
                        if not _oi_ok:
                            logging.info("[OI GATE] %s blocked — %s", asset, _oi_reason)
                            add_event(f"[OI GATE] {asset} blocked — {_oi_reason}")
                            continue
                        is_buy = action == "buy"
                        alloc_usd = float(output.get("allocation_usd", 0.0))
                        if alloc_usd <= 0:
                            add_event(f"Holding {asset}: zero/negative allocation")
                            continue

                        # --- RISK: Validate trade before execution ---
                        output["current_price"] = current_price
                        # Inject ATR14 from 4h data so enforce_stop_loss can use
                        # max(pct_floor, 1×ATR) instead of a flat percentage only.
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

                        # Cancel unfilled limit orders after 1 candle (5 minutes)
                        if order_type == "limit" and filled_qty == 0 and entry_oid:
                            try:
                                await hyperliquid.cancel_order(asset, entry_oid)
                                logging.info("[LIMIT] %s unfilled limit cancelled after 1 candle — skipping trade", asset)
                                add_event(f"[LIMIT] {asset} unfilled limit cancelled — price moved away")
                                continue
                            except Exception as _ce:
                                logging.warning("[LIMIT] %s cancel failed: %s", asset, _ce)

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
                                try:
                                    _slr = await hyperliquid.place_stop_loss(asset, is_buy, tp_sl_size, output["sl_price"])
                                    _sl_oids = hyperliquid.extract_oids(_slr)
                                    sl_oid = _sl_oids[0] if _sl_oids else None
                                    add_event(f"SL placed {asset} at {output['sl_price']} size={tp_sl_size:.6f}")
                                except Exception as _sl_err:
                                    logging.critical("[SL FAIL] %s — SL placement failed: %s — market-closing position to avoid unprotected exposure", asset, _sl_err)
                                    add_event(f"[SL FAIL] {asset} SL placement failed — market-closing position immediately")
                                    await send_alert(f"\U0001f6a8 SL FAIL {asset} — SL placement failed, market-closing position NOW. Error: {_sl_err}")
                                    try:
                                        await hyperliquid.market_close(asset)
                                        await hyperliquid.cancel_all_orders(asset)
                                    except Exception as _mc_err:
                                        logging.critical("[SL FAIL] %s — market-close also failed: %s — MANUAL INTERVENTION REQUIRED", asset, _mc_err)
                                        add_event(f"[SL FAIL] {asset} MARKET CLOSE ALSO FAILED — close manually on Hyperliquid NOW")
                                        await send_alert(f"\U0001f198 CRITICAL {asset} — SL fail AND market-close failed. CLOSE MANUALLY ON HYPERLIQUID NOW. Error: {_mc_err}")
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
                            "opened_at": datetime.now(timezone.utc).isoformat()
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
                    _shutdown = True
                    break

                if risk_mgr.circuit_breaker_active:
                    logging.info("[INNER] circuit breaker active — skipping tick %d", _tick + 1)
                    if _tick == 0:  # alert once per inner-loop session, not every tick
                        await send_alert("⛔ Daily loss circuit breaker active — no new trades until UTC midnight reset.")
                    continue

                logging.info("[INNER %d/11] refreshing 5m candles for %d assets", _tick + 1, len(args.assets))

                # Refresh 5m candles and recompute trigger_5m per asset
                for _i_asset in args.assets:
                    try:
                        _f5m = await hyperliquid.get_candles(_i_asset, "5m", 20)
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

                # Re-run score-gated pipeline with fresh 5m data
                _inner_outputs: dict = {"reasoning": "", "trade_decisions": []}
                for _i_asset in args.assets:
                    _iac = next((m for m in market_sections if m.get("asset") == _i_asset), None)
                    if not _iac:
                        continue
                    # P4-E-5: Circuit breaker re-checked per-asset so a breaker tripped mid-loop
                    # (e.g. after a loss updates risk_mgr) blocks remaining assets in the same tick.
                    if risk_mgr.circuit_breaker_active:
                        logging.info("[INNER CB] circuit breaker active — skipping %s tick %d", _i_asset, _tick + 1)
                        continue
                    if _daily_trade_count >= int(CONFIG.get("max_daily_trades") or 10):
                        break
                    _cd = _sl_cooldown_map.get(_i_asset)
                    if _cd and datetime.now(timezone.utc) < _cd:
                        continue
                    _idir = _code_decide_direction(_iac)
                    if _idir is None:
                        continue
                    _iscr = compute_signal_score(_iac, _idir)
                    try:
                        from src.indicators.kronos_forecast import get_kronos_modifier
                        _iscr = min(11.0, max(0.0, _iscr + get_kronos_modifier(candles=_iac.get("candles_4h", []), direction=_idir)))
                    except ImportError:
                        pass
                    if _iscr < float(CONFIG.get("min_signal_score") or 7):
                        continue
                    # BUG-2 FIX: Use the fresh price already refreshed by the C-3/C-4 block above,
                    # not the stale outer-loop price. TP/SL were anchored to an old price, placing
                    # the SL up to 1 full ATR away from the actual fill price during volatile moves.
                    _ie = float(asset_prices.get(_i_asset) or _iac.get("current_price") or 0)
                    _iatr = float(_iac.get("long_term_4h", {}).get("atr14") or 0)
                    if _ie <= 0 or _iatr <= 0:
                        continue
                    _itp1, _itp2, _isl, _itp = _code_compute_tpsl(_ie, _iatr, _idir, _iscr)
                    _i_buying_power = account_value * float(CONFIG.get("max_leverage") or 5)
                    _i_pct_cap = _i_buying_power * (float(CONFIG.get("max_position_pct") or 15) / 100.0)
                    _i_atr_sized = risk_mgr.atr_position_size(account_value, _ie, _isl)
                    _ialloc = min(_i_pct_cap, _i_atr_sized) * (min(_iscr, 10.0) / 10.0)
                    # ADX ranging market guard (inner loop)
                    _iadx_1h = float(_iac.get("intraday_1h", {}).get("adx") or 25)
                    _iadx_thr = float(CONFIG.get("adx_half_size_threshold") or 20)
                    if _iadx_1h < _iadx_thr and _iscr < 9.0:
                        _ialloc *= 0.5
                        logging.info("[INNER SIZE] %s ADX %.1f < %.0f + score %.1f < 9 → half-size",
                                     _i_asset, _iadx_1h, _iadx_thr, _iscr)

                    # Confluence gate (inner loop)
                    if not multi_timeframe_confluence(_iac, _idir):
                        continue

                    # MIN_AI_SCORE gate (inner loop)
                    _imin_ai = float(CONFIG.get("min_ai_score") or 7)
                    if _iscr < _imin_ai:
                        logging.info("[INNER AI GATE] %s score=%.1f < MIN_AI_SCORE %.1f → HOLD", _i_asset, _iscr, _imin_ai)
                        continue

                    # AI verdict: cache → gap → stale-TF check → call Claude
                    _ifp  = _build_confluence_fingerprint(
                        _i_asset, _idir,
                        _iac.get("trend_4h", "UNKNOWN"), _iac.get("trend_1h", "UNKNOWN"), _iscr
                    )
                    _inow = datetime.now(timezone.utc)
                    _ic   = _ai_verdict_cache.get(_i_asset)
                    _iuse = False
                    if _ic:
                        _icfp, _icexp = _ic.get("fingerprint"), _ic.get("expires_at")
                        if _icfp == _ifp and _icexp and _inow < _icexp:
                            if _ic.get("verdict") == "APPROVE":
                                _iuse = True
                            else:
                                continue  # cached REJECT
                    if not _iuse:
                        _igap  = int(CONFIG.get("min_ai_call_gap_minutes") or 30)
                        _ilast = _last_ai_call_time.get(_i_asset, 0)
                        _igsec = time.time() - _ilast
                        if _ilast > 0 and _igsec < _igap * 60:
                            _ipc = _ai_verdict_cache.get(_i_asset)
                            if _ipc and _ipc.get("verdict") == "APPROVE":
                                _iuse = True
                            else:
                                continue
                    if not _iuse:
                        # Block AI call if higher-TF data is too stale
                        _istale = int(CONFIG.get("ai_stale_tf_minutes") or 55)
                        _iage   = time.monotonic() - _outer_cycle_timestamp
                        if _iage > _istale * 60:
                            logging.info("[INNER AI STALE] %s %.0f min old → HOLD", _i_asset, _iage / 60)
                            continue
                        _iverd = await asyncio.to_thread(
                            agent.confirm_trade, _i_asset, _idir, _ie, _itp, _isl, _iscr, {},
                            _macro_context_cache, _iac
                        )
                        _last_ai_call_time[_i_asset] = time.time()
                        _iapm = int(CONFIG.get("ai_approve_cache_minutes") or 60)
                        _irjm = int(CONFIG.get("ai_reject_cache_minutes") or 30)
                        _icm  = _iapm if _iverd == "APPROVE" else _irjm
                        _ai_verdict_cache[_i_asset] = {
                            "verdict": _iverd, "fingerprint": _ifp,
                            "expires_at": _inow + timedelta(minutes=_icm),
                        }
                        if _iverd != "APPROVE":
                            continue

                    _ilim_px = round(_ie * (1 - 0.0015), 6) if _idir == "buy" else round(_ie * (1 + 0.0015), 6)
                    _inner_outputs["trade_decisions"].append({
                        "asset": _i_asset, "action": _idir,
                        "allocation_usd": _ialloc, "order_type": "limit",   # BONUS-2
                        "limit_price": _ilim_px, "tp_price": _itp, "tp1_price": _itp1, "tp2_price": _itp2,
                        "sl_price": _isl, "atr14": _iatr, "current_price": _ie,
                        "score": round(_iscr, 2),  # P5-HIGH-1: store per-asset score so execution loop uses correct value
                        "exit_plan": f"inner TP={_itp:.4f} TP1={_itp1:.4f} TP2={_itp2:.4f} SL={_isl:.4f} score={_iscr:.1f}",
                        "rationale": f"inner score={_iscr:.1f}",
                    })

                # C-3 + C-4 FIX: Refresh account state and prices before inner-loop execution.
                # Old code used stale outer-loop values (up to 55 min old) for risk checks and
                # TP/SL calculation, defeating total-exposure and concurrent-position guards.
                try:
                    _istate_fresh = await hyperliquid.get_user_state()
                    _iaccval_fresh = float(_istate_fresh.get("total_value", 0))
                    if _iaccval_fresh > 0:
                        account_value = _iaccval_fresh
                        state = _istate_fresh
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
                except Exception as _ire:
                    logging.warning("[INNER tick %d] state/price refresh failed: %s", _tick + 1, _ire)

                # Execute inner-loop trades (state gate + market_filter + entry_confirmed + risk)
                for _iout in _inner_outputs.get("trade_decisions", []):
                    _ia = _iout.get("asset")
                    if not _ia or _iout.get("action") not in ("buy", "sell"):
                        continue
                    try:
                        _ism = state_mgr.get_state(_ia)
                        if _ism in ("ENTERED", "COOLDOWN"):
                            continue
                        # C-4: use refreshed price (updated above for this tick)
                        _iprice = asset_prices.get(_ia, 0)
                        if not _iprice or _iprice <= 0:
                            continue
                        _iact_ctx = next((m for m in market_sections if m.get("asset") == _ia), {})
                        _itrend_1d = asset_trends_1d.get(_ia, "UNKNOWN")
                        _iadx_1d = asset_adx_1d.get(_ia)
                        _imacro_trending = _iadx_1d is not None and float(_iadx_1d) > 20
                        if _imacro_trending:
                            if _itrend_1d == "BEARISH" and _iout["action"] == "buy":
                                logging.info("[INNER DAILY FILTER] %s BUY blocked — daily BEARISH", _ia)
                                continue
                            if _itrend_1d == "BULLISH" and _iout["action"] == "sell":
                                logging.info("[INNER DAILY FILTER] %s SELL blocked — daily BULLISH", _ia)
                                continue
                        _iact_ctx_local = {**_iact_ctx, "candles_5m": asset_candles_5m.get(_ia, []), "_current_score": _iout.get("score", _iscr)}
                        _btc_trend_1h_inner = next(
                            (m.get("trend_1h", "UNKNOWN") for m in market_sections if m.get("asset") == "BTC"),
                            "UNKNOWN"
                        )
                        _mf_ok, _mf_why = market_filter(
                            _iact_ctx_local,
                            btc_trend_1h=_btc_trend_1h_inner,
                            btc_candles_5m=asset_candles_5m.get("BTC", []),
                            direction=_iout["action"]
                        )
                        if not _mf_ok:
                            logging.info("[INNER MKTFILTER] %s blocked: %s", _ia, _mf_why)
                            continue
                        if not entry_confirmed(_iact_ctx_local, _iout["action"]):
                            logging.info("[INNER ENTRY] %s entry_confirmed failed — 15m/5m conditions not met", _ia)
                            continue
                        # BONUS-1: Candle-close gate (same 85% logic as outer loop)
                        _i_secs_into_5m = (datetime.now(timezone.utc).minute % 5) * 60 + datetime.now(timezone.utc).second
                        if (_i_secs_into_5m / 300) < 0.70:
                            logging.info("[INNER CANDLE GATE] %s candle only %.0f%% complete", _ia, _i_secs_into_5m / 3)
                            continue
                        # BONUS-1: OI confirmation gate
                        _i_oi_ok, _i_oi_reason = oi_confirmed(_iact_ctx_local, _iout["action"])
                        if not _i_oi_ok:
                            logging.info("[INNER OI GATE] %s blocked — %s", _ia, _i_oi_reason)
                            continue
                        _iout["current_price"] = _iprice
                        _iout["atr14"] = _iact_ctx.get("long_term_4h", {}).get("atr14")
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
                                continue
                            except Exception as _ice:
                                logging.warning("[INNER LIMIT] %s cancel failed: %s", _ia, _ice)
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
                            with open("signals.jsonl", "a", encoding="utf-8") as _isf:
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
            "auth_enabled": bool(CONFIG.get("dashboard_token")),
            "network": (CONFIG.get("hyperliquid_network") or "mainnet").upper(),
        })

    async def start_api(app):
        """Register HTTP endpoints for observing diary entries and logs."""
        app.router.add_get('/', handle_index)
        app.router.add_get('/meta', handle_meta)
        app.router.add_get('/diary', handle_diary)
        app.router.add_get('/live', handle_live)
        app.router.add_get('/fills', handle_fills)
        app.router.add_get('/logs', handle_logs)
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