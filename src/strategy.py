"""BB + StochRSI mean reversion signal logic for the live trading loop.

Architecture: Code decides everything. Claude (Sonnet 4.6) is ONLY called on anomalous
price moves (>3% in 5 candles) as a binary sanity check — not for direction or TP/SL.

Signal: Bollinger Bands (20, 2.0) + Stochastic RSI (14, 14, 3, 3) on 5m candles.
Entry: Price touches outer BB band AND StochRSI hooks out of OS/OB zone.
TP:    BB midline (mean reversion target).
SL:    1.5 × ATR(14, 5m) — code-only.
Regime pause: 1h ADX > 30 = strong trend environment = bad for reversion → skip.
"""

import logging
from src.indicators.local_indicators import bbands, stoch_rsi, atr as _atr_fn, latest


def compute_bb_stochrsi_signal(candles_5m: list, cfg: dict) -> dict:
    """Compute BB + StochRSI mean reversion signal from 5m candles.

    Returns a dict:
        signal       : 'LONG' | 'SHORT' | 'NONE'
        bb_upper     : current upper BB band price
        bb_lower     : current lower BB band price
        bb_mid       : BB midline (= TP target)
        stochrsi_k   : current smoothed %K value (0–100)
        stochrsi_d   : current %D value (0–100)
        atr          : current ATR14 on 5m (used for SL sizing)
        current_price: last closed candle close price
    """
    no_signal = {
        "signal": "NONE",
        "bb_upper": None, "bb_lower": None, "bb_mid": None,
        "stochrsi_k": None, "stochrsi_d": None, "atr": None,
        "current_price": None,
    }

    bb_period = int(cfg.get("bb_period") or 20)
    bb_std    = float(cfg.get("bb_std") or 2.0)
    srsi_ob   = float(cfg.get("stochrsi_ob") or 80.0)
    srsi_os   = float(cfg.get("stochrsi_os") or 20.0)

    # Need at least bb_period + 28 candles for StochRSI to warm up
    min_candles = bb_period + 30
    if len(candles_5m) < min_candles:
        logging.debug("[BB_SIGNAL] insufficient candles: %d < %d", len(candles_5m), min_candles)
        return no_signal

    try:
        bb_data  = bbands(candles_5m, bb_period, bb_std)
        sr_data  = stoch_rsi(candles_5m,
                             rsi_period=int(cfg.get("stochrsi_period") or 14),
                             stoch_period=int(cfg.get("stochrsi_period") or 14),
                             k_smooth=int(cfg.get("stochrsi_k_smooth") or 3),
                             d_smooth=int(cfg.get("stochrsi_d_smooth") or 3))
        atr_data = _atr_fn(candles_5m, 14)

        bb_upper = latest(bb_data["upper"])
        bb_lower = latest(bb_data["lower"])
        bb_mid   = latest(bb_data["middle"])

        # Use last 2 non-None K values to detect the "hook" (reversal turn)
        k_vals = [v for v in sr_data["k"] if v is not None]
        d_vals = [v for v in sr_data["d"] if v is not None]

        if len(k_vals) < 2:
            return no_signal

        k_cur  = k_vals[-1]   # latest smoothed %K
        k_prev = k_vals[-2]   # previous bar %K (hook detection)
        d_cur  = d_vals[-1] if d_vals else None

        atr_val = latest(atr_data)

        # Use the LAST CLOSED candle ([-2]) not the forming candle ([-1])
        # to avoid mid-candle false signals that reverse before close.
        if len(candles_5m) < 2:
            return no_signal
        current_price = float(candles_5m[-2]["close"])

        if not all([bb_upper, bb_lower, bb_mid, atr_val, current_price]):
            return no_signal

        # ── Signal conditions ─────────────────────────────────────────────────
        signal = "NONE"

        # LONG: price at or below lower BB AND StochRSI was oversold AND is now turning up
        if current_price <= bb_lower and k_cur <= srsi_os and k_cur > k_prev:
            signal = "LONG"

        # SHORT: price at or above upper BB AND StochRSI was overbought AND is now turning down
        elif current_price >= bb_upper and k_cur >= srsi_ob and k_cur < k_prev:
            signal = "SHORT"

        if signal != "NONE":
            logging.info(
                "[BB_SIGNAL] %s signal=%s price=%.4f bb_%s=%.4f K=%.1f (prev=%.1f) ATR=%.6f",
                "?", signal,
                current_price,
                "lower" if signal == "LONG" else "upper",
                bb_lower if signal == "LONG" else bb_upper,
                k_cur, k_prev, atr_val,
            )

        return {
            "signal":        signal,
            "bb_upper":      round(bb_upper, 6),
            "bb_lower":      round(bb_lower, 6),
            "bb_mid":        round(bb_mid, 6),
            "stochrsi_k":    round(k_cur, 2),
            "stochrsi_d":    round(d_cur, 2) if d_cur is not None else None,
            "atr":           round(atr_val, 8),
            "current_price": current_price,
        }

    except Exception as e:
        logging.warning("[BB_SIGNAL] compute error: %s", e)
        return no_signal


def is_trend_regime_active(asset_data: dict, cfg: dict) -> bool:
    """Return True if 1h ADX is above the asset's ADX threshold — strong trend, bad for mean reversion.

    Per-asset thresholds set in .env as ADX_OVERRIDE_<ASSET>=<value>
    e.g. ADX_OVERRIDE_PAXG=50, ADX_OVERRIDE_SPX=48, ADX_OVERRIDE_BTC=38
    Falls back to TREND_PAUSE_ADX (default 30) if no per-asset override exists.
    Returns False (allow entry) when ADX data is unavailable — fail open for data gaps.
    """
    adx_1h = asset_data.get("intraday_1h", {}).get("adx")
    if adx_1h is None:
        return False  # no data → don't block

    asset_name = (asset_data.get("asset", "") or "").upper()
    # Check per-asset override first: ADX_OVERRIDE_PAXG=50, ADX_OVERRIDE_SPX=48, etc.
    override_key = f"adx_override_{asset_name.lower()}"
    per_asset = cfg.get(override_key)
    trend_pause = float(per_asset) if per_asset is not None else float(cfg.get("trend_pause_adx") or 30.0)

    is_trending = float(adx_1h) > trend_pause
    if is_trending:
        logging.info(
            "[REGIME] %s 1h ADX=%.1f > %.0f — strong trend active, pausing reversion entries",
            asset_data.get("asset", "?"), float(adx_1h), trend_pause,
        )
    else:
        logging.info(
            "[REGIME] %s 1h ADX=%.1f <= %.0f — regime OK, entries allowed",
            asset_data.get("asset", "?"), float(adx_1h), trend_pause,
        )
    return is_trending


def funding_rate_ok(funding_rate: float, direction: str) -> bool:
    """Return False when funding rate is adverse enough to erode mean reversion edge.

    Thresholds (per 8h):
      LONG  blocked if funding > +0.0005  (longs paying heavily — crowded)
      SHORT blocked if funding < -0.0005  (shorts paying heavily — crowded)
    """
    fr = float(funding_rate or 0)
    if direction == "buy"  and fr >  0.0005:
        logging.info("[FUNDING] BUY blocked — rate %.5f/8h too positive (crowded longs)", fr)
        return False
    if direction == "sell" and fr < -0.0005:
        logging.info("[FUNDING] SELL blocked — rate %.5f/8h too negative (crowded shorts)", fr)
        return False
    return True


def session_gate_ok(cfg: dict) -> tuple[bool, str]:
    """Return (True, '') when current UTC time is outside the low-liquidity block.

    Default block: 00:00–06:00 UTC (configurable via SESSION_BLOCK_START/END_UTC).
    Also blocks Fri 20:00 UTC → Sun 08:00 UTC (weekend low liquidity).
    """
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    h = now.hour
    wd = now.weekday()  # 0=Mon, 5=Sat, 6=Sun

    start_block = int(cfg.get("session_block_start_utc") or 0)
    end_block   = int(cfg.get("session_block_end_utc") or 6)

    if start_block <= h < end_block:
        return False, f"session gate — UTC {h:02d}:xx blocked ({start_block:02d}:00–{end_block:02d}:00 UTC)"

    # Weekend block: Fri 20:00 UTC → Sun 08:00 UTC
    is_weekend = (
        (wd == 4 and h >= 20) or  # Friday from 20:00
        (wd == 5) or               # All Saturday
        (wd == 6 and h < 8)        # Sunday until 08:00
    )
    if is_weekend:
        return False, "weekend gate — Fri 20:00→Sun 08:00 UTC (low liquidity)"

    return True, ""
