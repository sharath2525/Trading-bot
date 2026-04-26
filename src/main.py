"""Entry-point script that wires together the trading agent, data feeds, and API."""

import sys
import argparse
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from src.agent.decision_maker import TradingAgent
from src.indicators.local_indicators import compute_all, last_n, latest
from src.risk_manager import RiskManager
from src.trading.hyperliquid_api import HyperliquidAPI
import asyncio
import logging
from collections import deque, OrderedDict
from datetime import datetime, timezone
import math  # For Sharpe
from dotenv import load_dotenv
import os
import json
from aiohttp import web
from src.utils.formatting import format_number as fmt, format_size as fmt_sz
from src.utils.prompt_utils import json_default, round_or_none, round_series
from src.strategy import entry_confirmed
from src.trade_state import TradeStateMachine

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


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


# ── CORS middleware ───────────────────────────────────────────────────────────
@web.middleware
async def cors_middleware(request, handler):
    """Allow any origin to call the local API (needed for dashboard.html)."""
    if request.method == "OPTIONS":
        response = web.Response()
    else:
        try:
            response = await handler(request)
        except web.HTTPException as ex:
            response = ex
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response
# ─────────────────────────────────────────────────────────────────────────────


def main():
    """Parse CLI args, bootstrap dependencies, and launch the trading loop."""
    clear_terminal()
    parser = argparse.ArgumentParser(description="LLM-based Trading Agent on Hyperliquid")
    parser.add_argument("--assets", type=str, nargs="+", required=False, help="Assets to trade, e.g., BTC ETH")
    parser.add_argument("--interval", type=str, required=False, help="Interval period, e.g., 1h")
    args = parser.parse_args()

    # Allow assets/interval via .env (CONFIG) if CLI not provided
    from src.config_loader import CONFIG
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
    trade_log = []  # For Sharpe: list of returns
    active_trades = []  # {'asset','is_long','amount','entry_price','tp_oid','sl_oid','exit_plan'}
    recent_events = deque(maxlen=200)
    diary_path = "diary.jsonl"
    decisions_path = "decisions.jsonl"
    initial_account_value = None
    # Perp mid-price history sampled each loop (authoritative, avoids spot/perp basis mismatch)
    price_history = {}

    print(f"Starting trading agent for assets: {args.assets} at interval: {args.interval}")

    def add_event(msg: str):
        """Log an informational event and push it into the recent events deque."""
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

        while True:
            invocation_count += 1
            minutes_since_start = (datetime.now(timezone.utc) - start_time).total_seconds() / 60

            # Global account state
            state = await hyperliquid.get_user_state()
            total_value = state.get('total_value') or (state.get('balance', 0) + sum(p.get('pnl', 0) for p in state.get('positions', [])))
            sharpe = calculate_sharpe(trade_log)

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
                    size = ptc["size"]
                    is_long = ptc["is_long"]
                    add_event(f"RISK FORCE-CLOSE: {coin} at {ptc['loss_pct']}% loss (PnL: ${ptc['pnl']})")
                    try:
                        if is_long:
                            await hyperliquid.place_sell_order(coin, size)
                        else:
                            await hyperliquid.place_buy_order(coin, size)
                        await hyperliquid.cancel_all_orders(coin)
                        # Remove from active trades
                        for tr in active_trades[:]:
                            if tr.get('asset') == coin:
                                active_trades.remove(tr)
                        with open(diary_path, "a") as f:
                            f.write(json.dumps({
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "asset": coin,
                                "action": "risk_force_close",
                                "loss_pct": ptc["loss_pct"],
                                "pnl": ptc["pnl"],
                            }) + "\n")
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
            try:
                open_orders = await hyperliquid.get_open_orders()
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
                        with open(diary_path, "a") as f:
                            f.write(json.dumps({
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "asset": asset,
                                "action": "reconcile_close",
                                "reason": "no_position_no_orders",
                                "opened_at": tr.get('opened_at')
                            }) + "\n")
            except Exception:
                pass

            # FIX 3: Time-based exit — force-close trades stuck beyond max_trade_hours
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
                        except Exception as _te:
                            add_event(f"[TIMEOUT] {_asset_name} close error: {_te}")

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

            # Gather data for ALL assets first (using Hyperliquid candles + local indicators)
            market_sections = []
            asset_prices = {}
            asset_trends = {}  # Stores computed 4h trend label per asset for sanity checks
            for asset in args.assets:
                try:
                    current_price = await hyperliquid.get_current_price(asset)
                    asset_prices[asset] = current_price
                    if asset not in price_history:
                        price_history[asset] = deque(maxlen=60)
                    price_history[asset].append({"t": datetime.now(timezone.utc).isoformat(), "mid": round_or_none(current_price, 2)})
                    oi = await hyperliquid.get_open_interest(asset)
                    funding = await hyperliquid.get_funding_rate(asset)

                    # Fetch candles — 1h for intraday signals, 4h for structure, 15m/5m for entry precision
                    candles_1h  = await hyperliquid.get_candles(asset, "1h",  60)
                    candles_4h  = await hyperliquid.get_candles(asset, "4h",  60)
                    candles_15m = await hyperliquid.get_candles(asset, "15m", 30)
                    candles_5m  = await hyperliquid.get_candles(asset, "5m",  20)
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

                    # MACD histogram > 0 = MACD line above signal = bullish momentum
                    momentum_4h = (
                        "BULLISH" if macd_hist_4h is not None and macd_hist_4h > 0
                        else "BEARISH" if macd_hist_4h is not None and macd_hist_4h < 0
                        else "NEUTRAL"
                    )

                    asset_trends[asset] = trend_4h

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

                    market_sections.append({
                        "asset": asset,
                        "current_price": round_or_none(current_price, 2),
                        # Pre-computed trend labels — trust these for directional bias
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
                            "macd_histogram_series": round_series(last_n(lt.get("macd_histogram", []), 3), 4),
                            "rsi_series": round_series(last_n(lt.get("rsi14", []), 3), 2),
                        },
                        "open_interest": round_or_none(oi, 2),
                        "funding_rate": round_or_none(funding, 8),
                        "funding_annualized_pct": funding_annualized,
                        "recent_mid_prices": recent_mids,
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
            context_payload = OrderedDict([
                ("invocation", {
                    "minutes_since_start": round(minutes_since_start, 2),
                    "current_time": datetime.now(timezone.utc).isoformat(),
                    "invocation_count": invocation_count
                }),
                ("account", dashboard),
                ("risk_limits", risk_mgr.get_risk_summary()),
                ("market_data", market_sections),
                ("instructions", {
                    "assets": args.assets,
                    "requirement": "Decide actions for all assets and return a strict JSON object matching the schema."
                })
            ])
            context = json.dumps(context_payload, default=json_default)
            add_event(f"Combined prompt length: {len(context)} chars for {len(args.assets)} assets")
            with open("prompts.log", "a") as f:
                f.write(f"\n\n--- {datetime.now()} - ALL ASSETS ---\n{json.dumps(context_payload, indent=2, default=json_default)}\n")

            def _is_failed_outputs(outs):
                """Return True when outputs are missing or clearly invalid."""
                if not isinstance(outs, dict):
                    return True
                decisions = outs.get("trade_decisions")
                if not isinstance(decisions, list) or not decisions:
                    return True
                try:
                    return all(
                        isinstance(o, dict)
                        and (o.get('action') == 'hold')
                        and ('parse error' in (o.get('rationale', '').lower()))
                        for o in decisions
                    )
                except Exception:
                    return True

            try:
                outputs = agent.decide_trade(args.assets, context)
                if not isinstance(outputs, dict):
                    add_event(f"Invalid output format (expected dict): {outputs}")
                    outputs = {}
            except Exception as e:
                import traceback
                add_event(f"Agent error: {e}")
                add_event(f"Traceback: {traceback.format_exc()}")
                outputs = {}

            # Retry once on failure/parse error with a stricter instruction prefix
            if _is_failed_outputs(outputs):
                add_event("Retrying LLM once due to invalid/parse-error output")
                context_retry_payload = OrderedDict([
                    ("retry_instruction", "Return ONLY the JSON array per schema with no prose."),
                    ("original_context", context_payload)
                ])
                context_retry = json.dumps(context_retry_payload, default=json_default)
                try:
                    outputs = agent.decide_trade(args.assets, context_retry)
                    if not isinstance(outputs, dict):
                        add_event(f"Retry invalid format: {outputs}")
                        outputs = {}
                except Exception as e:
                    import traceback
                    add_event(f"Retry agent error: {e}")
                    add_event(f"Retry traceback: {traceback.format_exc()}")
                    outputs = {}

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
                        # Inversion assertion — fires if trend and order direction are opposite.
                        # BULLISH (EMA20>EMA50) must produce buy; BEARISH must produce sell.
                        # If this raises, an inversion is still present somewhere in the signal chain.
                        if trend_4h == "BULLISH" and action == "sell":
                            raise ValueError(f"INVERSION BUG DETECTED: {asset} trend=BULLISH but action=sell")
                        if trend_4h == "BEARISH" and action == "buy":
                            raise ValueError(f"INVERSION BUG DETECTED: {asset} trend=BEARISH but action=buy")
                        # FIX 2: Spread filter
                        asset_ctx = next((m for m in market_sections if m.get("asset") == asset), {})
                        _sp = asset_ctx.get("spread_pct", 0)
                        if _sp and float(_sp) > 0.15:
                            add_event(f"[SPREAD] {asset} blocked — spread {float(_sp):.3f}% too wide")
                            continue
                        # FIX 1: Entry confirmation (15m/5m layers)
                        if not entry_confirmed(asset_ctx, action):
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

                        # Confirm fill — retry up to 3 times with 1s intervals
                        filled = False
                        for _attempt in range(3):
                            await asyncio.sleep(1)
                            try:
                                fills_check = await hyperliquid.get_recent_fills(limit=20)
                                for fc in reversed(fills_check):
                                    if fc.get('coin') == asset or fc.get('asset') == asset:
                                        filled = True
                                        break
                            except Exception:
                                pass
                            if filled:
                                break
                        trade_log.append({"type": action, "price": current_price, "amount": amount, "exit_plan": output["exit_plan"], "filled": filled})
                        tp_oid = None
                        sl_oid = None
                        if output.get("tp_price"):
                            tp_order = await hyperliquid.place_take_profit(asset, is_buy, amount, output["tp_price"])
                            tp_oids = hyperliquid.extract_oids(tp_order)
                            tp_oid = tp_oids[0] if tp_oids else None
                            add_event(f"TP placed {asset} at {output['tp_price']}")
                        if output.get("sl_price"):
                            sl_order = await hyperliquid.place_stop_loss(asset, is_buy, amount, output["sl_price"])
                            sl_oids = hyperliquid.extract_oids(sl_order)
                            sl_oid = sl_oids[0] if sl_oids else None
                            add_event(f"SL placed {asset} at {output['sl_price']}")
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
                            "amount": amount,
                            "entry_price": current_price,
                            "tp_oid": tp_oid,
                            "sl_oid": sl_oid,
                            "exit_plan": output["exit_plan"],
                            "opened_at": datetime.now(timezone.utc).isoformat()
                        })
                        state_mgr.record_entry(asset)
                        add_event(f"{action.upper()} {asset} amount {amount:.4f} at ~{current_price}")
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
                                "amount": amount,
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
                    import traceback
                    add_event(f"Execution error {asset}: {e}")
                    add_event(f"Traceback: {traceback.format_exc()}")

            await asyncio.sleep(get_interval_seconds(args.interval))

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

    async def handle_logs(request):
        """Stream log files with optional download or tailing behaviour."""
        try:
            path = request.query.get('path', 'llm_requests.log')
            download = request.query.get('download')
            limit_param = request.query.get('limit')
            if not os.path.exists(path):
                return web.Response(text="", content_type="text/plain")
            with open(path, "r") as f:
                data = f.read()
            if download or (limit_param and (limit_param.lower() == 'all' or limit_param == '-1')):
                headers = {}
                if download:
                    headers["Content-Disposition"] = f"attachment; filename={os.path.basename(path)}"
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
                "balance": round_or_none(state.get('balance'), 2),
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
        from src.config_loader import CONFIG as CFG
        runner = web.AppRunner(app)
        await runner.setup()
        port = int(CFG.get("api_port"))
        site = web.TCPSite(runner, CFG.get("api_host"), port)
        await site.start()
        logging.info(f"API server started at http://localhost:{port}")
        await run_loop()

    def calculate_sharpe(returns):
        """Compute a naive Sharpe-like ratio from the trade log."""
        if not returns:
            return 0
        vals = [r.get('pnl', 0) if 'pnl' in r else 0 for r in returns]
        if not vals:
            return 0
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        std = math.sqrt(var) if var > 0 else 0
        return mean / std if std > 0 else 0

    asyncio.run(main_async())


if __name__ == "__main__":
    main()