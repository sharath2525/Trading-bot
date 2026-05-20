"""Kronos-mini forecast wrapper — ACTIVE Tier 1 score modifier (±0.5).

Model: KronosResearch/Kronos-mini (HuggingFace, 4.1M params, CPU, $0 cost)
Install: pip install torch transformers

This is a REQUIRED Tier 1 component (MASTER RULE 1 — 2026-05-20).
Graceful degradation: if model fails to load, logs WARNING and returns 0.0.
Bot continues operating at full capacity — Kronos modifier is simply 0.0.

Kronos is a time-series foundation model (AAAI 2026).
HuggingFace: https://huggingface.co/KronosResearch/Kronos-mini
"""

import logging

_kronos_pipeline = None
_kronos_loaded = False
_kronos_failed = False


def _load_kronos():
    """Attempt to load Kronos-mini from HuggingFace. Sets module-level flags."""
    global _kronos_pipeline, _kronos_loaded, _kronos_failed
    if _kronos_loaded or _kronos_failed:
        return
    try:
        from transformers import pipeline  # type: ignore
        _kronos_pipeline = pipeline(
            "time-series-prediction",
            model="KronosResearch/Kronos-mini",
            trust_remote_code=True,
        )
        _kronos_loaded = True
        logging.info("[KRONOS] Kronos-mini loaded from HuggingFace (Tier 1 active)")
    except Exception as _e:
        _kronos_failed = True
        logging.warning(
            "WARNING: Kronos not available — modifier = 0.0 (%s). "
            "Install with: pip install torch transformers",
            _e,
        )


def get_kronos_modifier(candles: list, direction: str) -> float:
    """Return +0.5 if Kronos agrees with direction, -0.5 if disagrees, 0.0 if unavailable.

    Input: list of OHLCV candle dicts with keys open/high/low/close/volume.
    Uses last 400 candles (or all available if < 400).
    Modifier is code-computed only — never involves Claude (MASTER RULE 2 compliant).
    """
    _load_kronos()
    if not _kronos_loaded or _kronos_pipeline is None:
        return 0.0
    if len(candles) < 30:
        return 0.0
    try:
        _recent = candles[-400:]
        _closes = [float(c.get("close", 0)) for c in _recent]
        if len(_closes) < 10 or _closes[-1] <= 0:
            return 0.0

        _result = _kronos_pipeline(_closes)

        if isinstance(_result, (list, tuple)) and len(_result) > 0:
            _forecast_val = _result[-1]
            if isinstance(_forecast_val, dict):
                _forecast_val = _forecast_val.get("label") or _forecast_val.get("score") or 0.0
            _forecast_val = float(_forecast_val)
        else:
            return 0.0

        _cur_close = _closes[-1]
        _chg_pct = (_forecast_val - _cur_close) / _cur_close * 100 if _cur_close > 0 else 0.0

        if abs(_chg_pct) < 0.1:
            return 0.0

        _fore_dir = "buy" if _chg_pct > 0 else "sell"
        modifier = +0.5 if _fore_dir == direction else -0.5
        logging.debug("[KRONOS] forecast_chg=%.3f%% fore_dir=%s code_dir=%s modifier=%.1f",
                      _chg_pct, _fore_dir, direction, modifier)
        return modifier

    except Exception as _e:
        logging.debug("[KRONOS] inference error: %s", _e)
        return 0.0
