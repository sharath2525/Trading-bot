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
import time
import traceback
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from aiohttp import web
from collections import deque
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

from src.agent.decision_maker import TradingAgent
from src.config_loader import CONFIG
from src.indicators.local_indicators import compute_all, last_n, latest
from src.risk_manager import RiskManager
from src.strategy import entry_confirmed, market_filter, compute_signal_score
from src.trade_state import TradeStateMachine, load_active_trades, save_active_trades
from src.trading.hyperliquid_api import HyperliquidAPI
from src.utils.prompt_utils import json_default, round_or_none, round_series

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
_log_file_handler = RotatingFileHandler("bot.log", maxBytes=5 * 1024 * 1024, backupCount=3)
_log_file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.getLogger().addHandler(_log_file_handler)

_shutdown = False  # Set to True by SIGTERM/SIGINT handler for clean loop exit


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
    """Return 'buy', 'sell', or None from 4h/1h trend alignment.

    Returns None when trend_4h is UNKNOWN or trends conflict.
    Called before scoring — no counter-trend trades ever reach the score gate.
    """
    trend_4h = asset_data.get("trend_4h", "UNKNOWN")
    trend_1h = asset_data.get("trend_1h", "UNKNOWN")
    if trend_4h == "UNKNOWN":
        return None
    if trend_4h == "BULLISH" and trend_1h in ("BULLISH", "UNKNOWN"):
        return "buy"
    if trend_4h == "BEARISH" and trend_1h in ("BEARISH", "UNKNOWN"):
        return "sell"
    return None  # conflicting trends — 4h and 1h disagree


def _code_compute_tpsl(entry: float, atr: float, direction: str) -> tuple[float, float]:
    """Return (tp_price, sl_price) using 2×ATR TP / 1×ATR SL with round-trip fee buffer baked in."""
    fee_buffer = entry * float(CONFIG.get("taker_fee_pct") or 0.00045) * 2  # entry + exit fee
    if direction == "buy":
        return round(entry + 2.0 * atr + fee_buffer, 6), round(entry - 1.0 * atr - fee_buffer, 6)
    return round(entry - 2.0 * atr - fee_buffer, 6), round(entry + 1.0 * atr + fee_buffer, 6)


def multi_timeframe_confluence(asset_data: dict, direction: str, require_30m: bool = True) -> bool:
    """Return True only when all active timeframes agree on direction.

    4h: EMA20 vs EMA50 strict alignment
    1h: EMA20 vs EMA50 strict alignment
    30m (optional): EMA cross OR MACD histogram direction
    15m: MACD histogram direction
    5m:  bullish/bearish candle OR MACD histogram direction
    All rows must pass simultaneously.
    """
    is_buy = direction == "buy"

    trend_4h = asset_data.get("trend_4h", "UNKNOWN")
    if is_buy and trend_4h != "BULLISH":
        return False
    if not is_buy and trend_4h != "BEARISH":
        return False

    trend_1h = asset_data.get("trend_1h", "UNKNOWN")
    if is_buy and trend_1h != "BULLISH":
        return False
    if not is_buy and trend_1h != "BEARISH":
        return False

    if require_30m:
        s30m = asset_data.get("setup_30m", {})
        ema20_30m = s30m.get("ema20")
        ema50_30m = s30m.get("ema50")
        macd_30m  = s30m.get("macd_histogram")
        if ema20_30m is not None and ema50_30m is not None:
            ok_30m = (is_buy and ema20_30m > ema50_30m) or (not is_buy and ema20_30m < ema50_30m)
        elif macd_30m is not None:
            ok_30m = (is_buy and macd_30m > 0) or (not is_buy and macd_30m < 0)
        else:
            ok_30m = True  # insufficient data — don't block
        if not ok_30m:
            return False

    macd_15m = asset_data.get("setup_15m", {}).get("macd_histogram")
    if macd_15m is not None:
        if is_buy and macd_15m <= 0:
            return False
        if not is_buy and macd_15m >= 0:
            return False

    t5m = asset_data.get("trigger_5m", {})
    candle_bullish = t5m.get("candle_bullish", False)
    macd_5m = t5m.get("macd_histogram")
    if is_buy:
        ok_5m = candle_bullish or (macd_5m is not None and macd_5m > 0)
    else:
        ok_5m = (not candle_bullish) or (macd_5m is not None and macd_5m < 0)
    if not ok_5m:
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
    """Restrict CORS to the dashboard origin only (localhost on the configured port)."""
    if request.method == "OPTIONS":
        response = web.Response()
    else:
        try:
            response = await handler(request)
        except web.HTTPException as ex:
            response = ex
    _port = CONFIG.get("api_port") or "3000"
    response.headers["Access-Control-Allow-Origin"] = f"http://localhost:{_port}"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response
# ─────────────────────────────────────────────────────────────────────────────


def main():
    """Parse CLI args, bootstrap dependencies, and launch the trading loop."""
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

    hyperliquid = HyperliquidAPI()
    agent = TradingAgent(hyperliquid=hyperliquid)
    risk_mgr = RiskManager()
    state_mgr = TradeStateMachine()

    start_time = datetime.now(timezone.utc)
    invocation_count = 0
    trade_log = []   # For Sharpe ratio: list of trade result dicts
    active_trades = load_active_trades()  # {'asset','is_long','amount','entry_price','tp_oid','sl_oid','exit_plan'}
    diary_path = "diary.jsonl"
    decisions_path = "decisions.jsonl"
    initial_account_value = None
    # Perp mid-price history sampled each loop (authoritative, avoids spot/perp basis mismatch)
    price_history = {}

    print(f"Starting trading agent for assets: {args.assets} at interval: {args.interval}")

    def add_event(msg: str):
        logging.info(msg)

    async def run_loop():
        """Main trading loop that gathers data, calls the agent, and executes trades."""
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

        # ── Score-pipeline state (persist across cycles) ──────────────────────────
        _daily_trade_count: int = 0           # resets at UTC midnight
        _sl_cooldown_map: dict = {}           # asset -> datetime (blocked until)
        _last_daily_reset = None              # date of last counter reset
        _ai_verdict_cache: dict = {}          # asset → {verdict, fingerprint, expires_at}
        _macro_context_cache: dict = {}       # {events, headlines, fetched_at}
        _outer_cycle_timestamp: float = 0.0  # time.monotonic() at outer cycle data fetch
        _last_ai_call_time: dict = {}         # asset → unix timestamp of last Claude call
        # ─────────────────────────────────────────────────────────────────────────

        # ── Trade-close logging helpers ───────────────────────────────────────────

        _stats_lock = asyncio.Lock()

        async def _update_stats(realized_pnl: float | None, exit_price: float | None,
                                qty: float | None) -> None:
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
                    if exit_price and qty:
                        stats["total_fees"] = round(
                            stats.get("total_fees", 0.0) + exit_price * qty * 0.00045, 4
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

            await _update_stats(realized_pnl, exit_price, qty)
            logging.info(
                "[TRADE CLOSE] %s exit_type=%s exit_px=%s pnl=%s duration=%smin",
                asset, exit_type, exit_price, realized_pnl, duration_minutes,
            )

        # ─────────────────────────────────────────────────────────────────────────

        _interval_seconds = get_interval_seconds(args.interval)
        _consecutive_failures = 0
        _MAX_CONSECUTIVE_FAILURES = 5
        while True:
            if _shutdown:
                logging.info("[SHUTDOWN] Signal received — exiting loop cleanly")
                break
            cycle_start = time.monotonic()
            _outer_cycle_timestamp = cycle_start  # inner ticks read this to check higher-TF freshness
            _ai_verdict_cache.clear()              # fresh outer cycle data → invalidate all verdict caches
            invocation_count += 1

            # UTC midnight reset of daily trade counter
            _today_utc = datetime.now(timezone.utc).date()
            if _last_daily_reset != _today_utc:
                _daily_trade_count = 0
                _last_daily_reset = _today_utc
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
                    await asyncio.sleep(300)
                    _consecutive_failures = 0
                else:
                    await asyncio.sleep(_interval_seconds)
                continue
            total_value = state.get('total_value') or (state.get('balance', 0) + sum(p.get('pnl', 0) for p in state.get('positions', [])))
            sharpe = calculate_sharpe_from_diary(diary_path)

            account_value = total_value
            if initial_account_value is None:
                initial_account_value = account_value
            total_return_pct = ((account_value - initial_account_value) / initial_account_value * 100.0) if initial_account_value else 0.0

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
            except Exception:
                pass

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
                # Read TP/SL prices from the most recent buy/sell diary entry for this asset
                _g_diary = None
                try:
                    with open(diary_path, 'r') as _gf:
                        for _gl in reversed(_gf.readlines()):
                            try:
                                _ge = json.loads(_gl)
                                if _ge.get('asset') == _g_asset and _ge.get('action') in ('buy', 'sell'):
                                    _g_diary = _ge
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
                _g_is_long = _g_diary.get('action') == 'buy'
                _g_amount  = float(_g_diary.get('amount') or 0)
                _g_tp_px   = _g_diary.get('tp_price')
                _g_sl_px   = _g_diary.get('sl_price')
                if _g_amount <= 0:
                    logging.warning("[GUARDIAN] %s: zero amount in diary — cannot re-place", _g_asset)
                    continue
                if not _g_has_tp and _g_tp_px:
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
                logging.debug("[MACRO] outer cycle fetch error: %s", _mce)

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
                    near_ema_15m = (
                        abs(current_price - ema20_15m) / current_price < 0.003
                        if (ema20_15m is not None and current_price > 0)
                        else True
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
                            "macd_histogram_series": round_series(last_n(lt.get("macd_histogram", []), 3), 4),
                            "rsi_series": round_series(last_n(lt.get("rsi14", []), 3), 2),
                        },
                        "open_interest": round_or_none(oi, 2),
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
                    })
                except Exception as e:
                    add_event(f"Data gather error {asset}: {e}")
                    continue

            # Single LLM call with all assets
            context_payload = {
                "invocation": {
                    "minutes_since_start": round(minutes_since_start, 2),
                    "current_time": datetime.now(timezone.utc).isoformat(),
                    "invocation_count": invocation_count,
                },
                "account": dashboard,
                "risk_limits": risk_mgr.get_risk_summary(),
                "market_data": market_sections,
                "instructions": {
                    "assets": args.assets,
                    "requirement": "Decide actions for all assets and return a strict JSON object matching the schema.",
                },
            }
            context = json.dumps(context_payload, default=json_default)
            add_event(f"Combined prompt length: {len(context)} chars for {len(args.assets)} assets")
            _prompts_log = "prompts.log"
            try:
                if os.path.exists(_prompts_log) and os.path.getsize(_prompts_log) > 10 * 1024 * 1024:
                    os.replace(_prompts_log, _prompts_log + ".old")
            except Exception:
                pass
            with open(_prompts_log, "a") as f:
                f.write(f"\n\n--- {datetime.now()} - ALL ASSETS ---\n{json.dumps(context_payload, indent=2, default=json_default)}\n")

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
                    outputs["trade_decisions"].append(_make_hold(
                        _asset,
                        f"4h gate: trend_4h={_ac.get('trend_4h')} trend_1h={_ac.get('trend_1h')} conflict"
                    ))
                    continue

                # Weighted score gate (0-10 float)
                _score = compute_signal_score(_ac, _direction)
                if _score < _min_sig:
                    logging.debug("[SCORE] %s %s score=%.1f < %.1f → HOLD", _asset, _direction, _score, _min_sig)
                    outputs["trade_decisions"].append(_make_hold(_asset, f"score={_score:.1f} < min {_min_sig:.0f}"))
                    continue

                # Code computes trade parameters
                _entry = float(_ac.get("current_price") or 0)
                _atr   = float(_ac.get("long_term_4h", {}).get("atr14") or 0)
                if _entry <= 0 or _atr <= 0:
                    outputs["trade_decisions"].append(_make_hold(_asset, "missing price or ATR14"))
                    continue

                _tp, _sl = _code_compute_tpsl(_entry, _atr, _direction)
                # 1% risk rule — allocation sized so SL hit = 1% account loss
                _alloc = risk_mgr.atr_position_size(account_value, _entry, _sl)
                # Scale allocation by signal strength (score/10): score-7 → 70%, score-10 → 100%
                _alloc = _alloc * (_score / 10.0)
                # ADX ranging market guard: half-size if ADX weak and score not at maximum
                _adx_1h_val = float(_ac.get("intraday_1h", {}).get("adx") or 25)
                _adx_thr    = float(CONFIG.get("adx_half_size_threshold") or 20)
                if _adx_1h_val < _adx_thr and _score < 9.0:
                    _alloc *= 0.5
                    logging.info("[SIZE] %s ADX %.1f < %.0f + score %.1f < 9 → half-size applied",
                                 _asset, _adx_1h_val, _adx_thr, _score)

                # Confluence gate: all timeframes must agree before calling Claude
                _require_30m = bool(CONFIG.get("confluence_require_30m", True))
                _confluence_ok = multi_timeframe_confluence(_ac, _direction, _require_30m)
                if not _confluence_ok:
                    logging.debug("[CONFLUENCE] %s %s — TFs not aligned → HOLD", _asset, _direction)
                    outputs["trade_decisions"].append(_make_hold(_asset, "confluence failed — TFs not aligned"))
                    continue

                # MIN_AI_SCORE gate — separate from MIN_SIGNAL_SCORE so Claude call frequency is independently tunable
                _min_ai = float(CONFIG.get("min_ai_score") or 7)
                if _score < _min_ai:
                    logging.debug("[AI GATE] %s score=%.1f < MIN_AI_SCORE %.1f → HOLD", _asset, _score, _min_ai)
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
                outputs["trade_decisions"].append({
                    "asset":        _asset,
                    "action":       _direction,
                    "allocation_usd": _alloc,
                    "order_type":   "market",
                    "limit_price":  None,
                    "tp_price":     _tp,
                    "sl_price":     _sl,
                    "atr14":        _atr,
                    "current_price": _entry,
                    "exit_plan":    f"code TP={_tp:.4f} SL={_sl:.4f} score={_score:.1f}",
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
                        _macro_trending = _adx_1d is None or float(_adx_1d) > 20
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
                        asset_ctx_local = {**asset_ctx, "candles_5m": asset_candles_5m.get(asset, [])}
                        # ATR spike + spread pre-flight — market_filter() was dead code (only
                        # called from make_decision() which is never invoked); wire it directly.
                        _mf_pass, _mf_reason = market_filter(asset_ctx_local)
                        if not _mf_pass:
                            logging.warning("[MARKET FILTER] %s %s blocked — %s", asset, action, _mf_reason)
                            add_event(f"[MARKET FILTER] {asset} {action} blocked — {_mf_reason}")
                            continue
                        # Entry confirmation (15m/5m layers + volume gate)
                        if not entry_confirmed(asset_ctx_local, action):
                            logging.info(
                                "[ENTRY] %s direction=%s blocked — "
                                "15m/5m not confirmed, waiting for pullback",
                                asset, action
                            )
                            add_event(f"[ENTRY] {asset} {action} blocked — 15m/5m not confirmed")
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
                        _can_place_tpsl = filled_qty > 0 or order_type != "limit"
                        if _can_place_tpsl:
                            if output.get("tp_price"):
                                tp_order = await hyperliquid.place_take_profit(asset, is_buy, tp_sl_size, output["tp_price"])
                                tp_oids = hyperliquid.extract_oids(tp_order)
                                tp_oid = tp_oids[0] if tp_oids else None
                                add_event(f"TP placed {asset} at {output['tp_price']} size={tp_sl_size:.6f}")
                            if output.get("sl_price"):
                                sl_order = await hyperliquid.place_stop_loss(asset, is_buy, tp_sl_size, output["sl_price"])
                                sl_oids = hyperliquid.extract_oids(sl_order)
                                sl_oid = sl_oids[0] if sl_oids else None
                                add_event(f"SL placed {asset} at {output['sl_price']} size={tp_sl_size:.6f}")
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
                            "entry_price": current_price,
                            "tp_price": output.get("tp_price"),
                            "sl_price": output.get("sl_price"),
                            "tp_oid": tp_oid,
                            "sl_oid": sl_oid,
                            "exit_plan": output["exit_plan"],
                            "funding_rate": float(asset_ctx.get("funding_rate") or 0),
                            "opened_at": datetime.now(timezone.utc).isoformat()
                        })
                        save_active_trades(active_trades)
                        state_mgr.record_entry(asset)
                        _daily_trade_count += 1
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
                                "tp_oid": tp_oid,
                                "sl_price": output.get("sl_price"),
                                "sl_oid": sl_oid,
                                "exit_plan": output.get("exit_plan", ""),
                                "rationale": output.get("rationale", ""),
                                "order_result": str(order),
                                "opened_at": datetime.now(timezone.utc).isoformat(),
                                "filled": filled
                            }
                            f.write(json.dumps(diary_entry) + "\n")
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
                await asyncio.sleep(300)  # 5 minutes

                if risk_mgr.circuit_breaker_active:
                    logging.info("[INNER] circuit breaker active — skipping tick %d", _tick + 1)
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
                    if _daily_trade_count >= int(CONFIG.get("max_daily_trades") or 10):
                        break
                    _cd = _sl_cooldown_map.get(_i_asset)
                    if _cd and datetime.now(timezone.utc) < _cd:
                        continue
                    _idir = _code_decide_direction(_iac)
                    if _idir is None:
                        continue
                    _iscr = compute_signal_score(_iac, _idir)
                    if _iscr < float(CONFIG.get("min_signal_score") or 7):
                        continue
                    _ie = float(_iac.get("current_price") or 0)
                    _iatr = float(_iac.get("long_term_4h", {}).get("atr14") or 0)
                    if _ie <= 0 or _iatr <= 0:
                        continue
                    _itp, _isl = _code_compute_tpsl(_ie, _iatr, _idir)
                    _ialloc = risk_mgr.atr_position_size(account_value, _ie, _isl) * (_iscr / 10.0)
                    # ADX ranging market guard (inner loop)
                    _iadx_1h = float(_iac.get("intraday_1h", {}).get("adx") or 25)
                    _iadx_thr = float(CONFIG.get("adx_half_size_threshold") or 20)
                    if _iadx_1h < _iadx_thr and _iscr < 9.0:
                        _ialloc *= 0.5
                        logging.info("[INNER SIZE] %s ADX %.1f < %.0f + score %.1f < 9 → half-size",
                                     _i_asset, _iadx_1h, _iadx_thr, _iscr)

                    # Confluence gate (inner loop)
                    _irq30m = bool(CONFIG.get("confluence_require_30m", True))
                    if not multi_timeframe_confluence(_iac, _idir, _irq30m):
                        continue

                    # MIN_AI_SCORE gate (inner loop)
                    _imin_ai = float(CONFIG.get("min_ai_score") or 7)
                    if _iscr < _imin_ai:
                        logging.debug("[INNER AI GATE] %s score=%.1f < MIN_AI_SCORE %.1f → HOLD", _i_asset, _iscr, _imin_ai)
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

                    _inner_outputs["trade_decisions"].append({
                        "asset": _i_asset, "action": _idir,
                        "allocation_usd": _ialloc, "order_type": "market",
                        "limit_price": None, "tp_price": _itp, "sl_price": _isl,
                        "atr14": _iatr, "current_price": _ie,
                        "exit_plan": f"inner TP={_itp:.4f} SL={_isl:.4f} score={_iscr:.1f}",
                        "rationale": f"inner score={_iscr:.1f}",
                    })

                # Execute inner-loop trades (state gate + market_filter + entry_confirmed + risk)
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
                        _itrend_1d = asset_trends_1d.get(_ia, "UNKNOWN")
                        _iadx_1d = asset_adx_1d.get(_ia)
                        _imacro_trending = _iadx_1d is None or float(_iadx_1d) > 20
                        if _imacro_trending:
                            if _itrend_1d == "BEARISH" and _iout["action"] == "buy":
                                logging.info("[INNER DAILY FILTER] %s BUY blocked — daily BEARISH", _ia)
                                continue
                            if _itrend_1d == "BULLISH" and _iout["action"] == "sell":
                                logging.info("[INNER DAILY FILTER] %s SELL blocked — daily BULLISH", _ia)
                                continue
                        _iact_ctx_local = {**_iact_ctx, "candles_5m": asset_candles_5m.get(_ia, [])}
                        _mf_ok, _mf_why = market_filter(_iact_ctx_local)
                        if not _mf_ok:
                            logging.info("[INNER MKTFILTER] %s blocked: %s", _ia, _mf_why)
                            continue
                        if not entry_confirmed(_iact_ctx_local, _iout["action"]):
                            logging.debug("[INNER ENTRY] %s not confirmed", _ia)
                            continue
                        _iout["current_price"] = _iprice
                        _iout["atr14"] = _iact_ctx.get("long_term_4h", {}).get("atr14")
                        _iallowed, _ireason, _iout = risk_mgr.validate_trade(_iout, state, initial_account_value or 0)
                        if not _iallowed:
                            add_event(f"[INNER RISK] {_ia}: {_ireason}")
                            continue
                        _iamt = float(_iout["allocation_usd"]) / _iprice
                        _iorder = await (hyperliquid.place_buy_order(_ia, _iamt)
                                         if _iout["action"] == "buy"
                                         else hyperliquid.place_sell_order(_ia, _iamt))
                        await asyncio.sleep(1)
                        _itp_oid = None
                        _isl_oid = None
                        if _iout.get("tp_price"):
                            _itp_res = await hyperliquid.place_take_profit(
                                _ia, _iout["action"] == "buy", _iamt, _iout["tp_price"])
                            _itp_oid = (hyperliquid.extract_oids(_itp_res) or [None])[0]
                        if _iout.get("sl_price"):
                            _isl_res = await hyperliquid.place_stop_loss(
                                _ia, _iout["action"] == "buy", _iamt, _iout["sl_price"])
                            _isl_oid = (hyperliquid.extract_oids(_isl_res) or [None])[0]
                        active_trades.append({
                            "asset": _ia, "is_long": _iout["action"] == "buy",
                            "amount": _iamt, "entry_price": _iprice,
                            "tp_price": _iout.get("tp_price"), "sl_price": _iout.get("sl_price"),
                            "tp_oid": _itp_oid, "sl_oid": _isl_oid,
                            "exit_plan": _iout.get("exit_plan", ""),
                            "funding_rate": float(_iact_ctx.get("funding_rate") or 0),
                            "opened_at": datetime.now(timezone.utc).isoformat(),
                        })
                        save_active_trades(active_trades)
                        state_mgr.record_entry(_ia)
                        _daily_trade_count += 1
                        add_event(f"[INNER] {_iout['action'].upper()} {_ia} amt={_iamt:.4f} score={_iout.get('rationale','')} daily={_daily_trade_count}")
                        with open(diary_path, "a") as _idf:
                            _idf.write(json.dumps({
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "asset": _ia, "action": _iout["action"],
                                "order_type": "market", "allocation_usd": float(_iout["allocation_usd"]),
                                "amount": _iamt, "entry_price": _iprice,
                                "tp_price": _iout.get("tp_price"), "sl_price": _iout.get("sl_price"),
                                "exit_plan": _iout.get("exit_plan", ""),
                                "rationale": _iout.get("rationale", ""),
                                "inner_tick": _tick + 1,
                            }) + "\n")
                    except Exception as _ie2:
                        add_event(f"[INNER] execution error {_ia}: {_ie2}")
            # ── end 5-minute inner loop ────────────────────────────────────────────

    async def handle_diary(request):
        """Return diary entries as JSON — reads from decisions.jsonl for rich data."""
        try:
            limit = int(request.query.get('limit', '200'))

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
            return web.json_response({
                "account_value": round_or_none(state.get('total_value'), 2),
                "balance":       round_or_none(state.get('balance'), 2),
                "perps_value":   round_or_none(state.get('perps_value'), 2),
                "spot_usdc":     round_or_none(state.get('spot_usdc'), 2),
                "withdrawable":  round_or_none(state.get('withdrawable'), 2),
                "positions": positions,
                "open_orders": open_orders,
                "recent_fills": recent_fills,
                "timestamp": datetime.now(timezone.utc).isoformat()
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
            return web.Response(text=dashboard.read_text(encoding='utf-8'), content_type='text/html')
        except FileNotFoundError:
            return web.Response(text=f'<h1>dashboard.html not found at {dashboard}</h1>', content_type='text/html', status=404)

    async def start_api(app):
        """Register HTTP endpoints for observing diary entries and logs."""
        app.router.add_get('/', handle_index)
        app.router.add_get('/diary', handle_diary)
        app.router.add_get('/live', handle_live)
        app.router.add_get('/fills', handle_fills)
        app.router.add_get('/logs', handle_logs)
        app.router.add_route('OPTIONS', '/{path_info:.*}', lambda r: web.Response())

    async def main_async():
        """Start the aiohttp server and kick off the trading loop."""
        # Pass cors_middleware here so every response gets CORS headers
        app = web.Application(middlewares=[cors_middleware])
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
        await run_loop()

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
        logging.info("[SHUTDOWN] Signal %d received", signum)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    asyncio.run(main_async())


if __name__ == "__main__":
    main()