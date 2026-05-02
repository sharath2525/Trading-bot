"""Pre-trade filters and multi-timeframe entry confirmation for the live trading loop."""

import logging
from src.config_loader import CONFIG


def market_filter(asset_data: dict) -> tuple[bool, str]:
    """Block any entry when market conditions are unfavourable.

    Returns (allowed: bool, reason: str).
    """
    lt = asset_data.get("long_term_4h", {})
    current_price = float(asset_data.get("current_price") or 0)

    atr14 = lt.get("atr14")
    if atr14 and current_price > 0:
        atr_pct = float(atr14) / current_price * 100
        if atr_pct > 5.0:
            return False, f"ATR spike {atr_pct:.2f}% of price — too volatile"

    spread_pct = asset_data.get("spread_pct", 0)
    if spread_pct and float(spread_pct) > 0.15:
        return False, f"spread {float(spread_pct):.3f}% too wide"

    return True, ""


def _compute_signal_score(asset_data: dict, direction: str) -> int:
    """Return an integer score 0–5 counting how many entry conditions are met.

    Each of five conditions contributes 1 point. MIN_TRADE_SCORE sets the
    minimum number of conditions that must pass before entry is allowed.
    """
    s15 = asset_data.get("setup_15m", {})
    t5  = asset_data.get("trigger_5m", {})
    current_price = float(asset_data.get("current_price") or 0)
    macd_threshold = current_price * 0.001 if current_price > 0 else 0.0

    macd_15m = float(s15.get("macd_histogram") or 0)
    near_ema  = bool(s15.get("near_ema", False))
    macd_5m   = float(t5.get("macd_histogram") or 0)
    bull_5m   = bool(t5.get("candle_bullish", False))
    trend_4h  = asset_data.get("trend_4h", "UNKNOWN")
    trend_1h  = asset_data.get("trend_1h", "UNKNOWN")

    score = 0
    if direction == "buy":
        if trend_4h == "BULLISH":           score += 1
        if trend_1h == "BULLISH":           score += 1
        if macd_15m > macd_threshold:       score += 1
        if near_ema:                        score += 1
        if bull_5m or macd_5m > 0:         score += 1
    elif direction == "sell":
        if trend_4h == "BEARISH":           score += 1
        if trend_1h == "BEARISH":           score += 1
        if macd_15m < -macd_threshold:      score += 1
        if near_ema:                        score += 1
        if (not bull_5m) or macd_5m < 0:   score += 1
    return score


def compute_signal_score(asset_data: dict, direction: str) -> float:
    """Return a weighted float score 0–10 for the pre-gate in main.py.

    Weights: trend_4h=3, trend_1h=2, MACD_15m=2, near_ema=1.5, trigger_5m=1.5.
    Reachable values: 0, 1.5, 2, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 8, 8.5, 10.
    Score 9 is mathematically unreachable. Every path to >=7 requires trend_4h aligned.
    MIN_SIGNAL_SCORE (default 7) is the execution threshold in main.py.
    Do NOT call this from entry_confirmed() — that uses _compute_signal_score() (0–5 system).
    """
    s15 = asset_data.get("setup_15m", {})
    t5  = asset_data.get("trigger_5m", {})
    current_price = float(asset_data.get("current_price") or 0)
    macd_threshold = current_price * 0.001 if current_price > 0 else 0.0

    macd_15m = float(s15.get("macd_histogram") or 0)
    near_ema  = bool(s15.get("near_ema", False))
    macd_5m   = float(t5.get("macd_histogram") or 0)
    bull_5m   = bool(t5.get("candle_bullish", False))
    trend_4h  = asset_data.get("trend_4h", "UNKNOWN")
    trend_1h  = asset_data.get("trend_1h", "UNKNOWN")

    score = 0.0
    if direction == "buy":
        if trend_4h == "BULLISH":           score += 3.0
        if trend_1h == "BULLISH":           score += 2.0
        if macd_15m > macd_threshold:       score += 2.0
        if near_ema:                        score += 1.5
        if bull_5m or macd_5m > 0:         score += 1.5
    elif direction == "sell":
        if trend_4h == "BEARISH":           score += 3.0
        if trend_1h == "BEARISH":           score += 2.0
        if macd_15m < -macd_threshold:      score += 2.0
        if near_ema:                        score += 1.5
        if (not bull_5m) or macd_5m < 0:   score += 1.5
    return score


def entry_confirmed(asset_data: dict, direction: str) -> bool:
    """Return True only when 15m and 5m confirm the higher-timeframe direction.

    Returns False when indicator data is missing — block entry rather than
    allow through with no confirmation.
    """
    s15 = asset_data.get("setup_15m", {})
    t5  = asset_data.get("trigger_5m", {})

    if not s15 or not t5:
        return False

    # Signal score gate — requires minimum aligned conditions before entry
    _score = _compute_signal_score(asset_data, direction)
    _min_score = int(CONFIG.get("min_trade_score") or 3)
    if _score < _min_score:
        logging.debug(
            "[SCORE] %s %s blocked — score %d < min %d",
            asset_data.get("asset", "?"), direction, _score, _min_score,
        )
        return False

    # RSI gate — block chasing into overbought longs or oversold shorts
    rsi_15m = s15.get("rsi14")
    if rsi_15m is not None:
        if direction == "buy" and float(rsi_15m) > 70:
            logging.debug("buy blocked — 15m RSI %.1f overbought", float(rsi_15m))
            return False
        if direction == "sell" and float(rsi_15m) < 30:
            logging.debug("sell blocked — 15m RSI %.1f oversold", float(rsi_15m))
            return False

    # ADX gate — block entries in ranging (non-trending) markets
    adx_1h = asset_data.get("intraday_1h", {}).get("adx")
    if adx_1h is not None and float(adx_1h) < 20:
        logging.debug("entry blocked — 1h ADX %.1f below 20 (ranging market)", float(adx_1h))
        return False

    macd_15m = float(s15.get("macd_histogram") or 0)
    near_ema  = s15.get("near_ema", True)
    macd_5m   = float(t5.get("macd_histogram") or 0)
    bull_5m   = t5.get("candle_bullish", True)

    # Price-relative MACD threshold (0.1% of price).
    # A fixed ±50 is meaningless for high-priced assets where MACD swings in hundreds.
    current_price = float(asset_data.get("current_price") or 0)
    macd_threshold = current_price * 0.001 if current_price > 0 else 50.0

    # Volume confirmation: trigger candle must have at least 70% of the recent
    # average volume. This filters dead/low-liquidity candles that produce fake
    # MACD crossovers without real buying/selling pressure behind them.
    # Candle dicts use the key "volume" (mapped from raw Hyperliquid "v" field).
    candles_5m = asset_data.get("candles_5m", [])
    if len(candles_5m) >= 5:
        recent_vols = [c.get("volume", 0) for c in candles_5m[:-1]]
        avg_vol = sum(recent_vols) / len(recent_vols) if recent_vols else 0
        trigger_vol = candles_5m[-1].get("volume", 0)
        vol_ok = trigger_vol >= avg_vol * 0.7 if avg_vol > 0 else True
        if not vol_ok:
            logging.debug(
                "Entry rejected: low volume on 5m trigger (%.0f vs avg %.0f)",
                trigger_vol, avg_vol,
            )
    else:
        vol_ok = True  # not enough candles to judge, allow through

    if direction == "buy":
        return vol_ok and (near_ema and macd_15m > -macd_threshold) and (bull_5m or macd_5m > 0)

    if direction == "sell":
        return vol_ok and (near_ema and macd_15m < macd_threshold) and ((not bull_5m) or macd_5m < 0)

    return True
