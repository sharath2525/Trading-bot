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
    """Attempt to load Kronos-mini from HuggingFace. Sets module-level flags.

    BUG-P9-4 FIX: "time-series-prediction" is not a valid transformers pipeline task.
    Kronos-mini is a time-series foundation model that requires direct AutoModel loading
    with trust_remote_code=True. We store the model+tokenizer tuple as _kronos_pipeline.
    Falls back to chronos-forecasting package if available (pip install chronos-forecasting).
    """
    global _kronos_pipeline, _kronos_loaded, _kronos_failed
    if _kronos_loaded or _kronos_failed:
        return

    # Attempt 1: chronos-forecasting package (most reliable for Kronos-family models)
    try:
        import torch  # type: ignore
        from chronos import ChronosPipeline  # type: ignore
        _kronos_pipeline = ChronosPipeline.from_pretrained(
            "KronosResearch/Kronos-mini",
            device_map="cpu",
            torch_dtype=torch.float32,
        )
        _kronos_loaded = True
        logging.info("[KRONOS] Kronos-mini loaded via chronos-forecasting (Tier 1 active)")
        return
    except ImportError:
        pass  # chronos-forecasting not installed — try direct transformers loading
    except Exception as _e1:
        logging.debug("[KRONOS] chronos-forecasting load failed: %s", _e1)

    # BUG-v7-P1 FIX: AutoModel fallback removed. Tokenizing str(_closes) into a seq2seq LM
    # is not time-series forecasting — the output logit is random noise with no predictive
    # value. Injecting a random ±0.5 modifier pollutes the score system. Return 0.0 instead.

    _kronos_failed = True
    logging.warning(
        "WARNING: Kronos not available — modifier = 0.0. "
        "Install with: pip install chronos-forecasting torch transformers"
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

        # ChronosPipeline path only — AutoModel fallback removed (BUG-v7-P1)
        import torch  # type: ignore
        import concurrent.futures as _cf
        _context = torch.tensor(_closes, dtype=torch.float32).unsqueeze(0)
        # PERF-v10-7 FIX: per-call timeout so thermal throttle / memory pressure cannot
        # block the Kronos thread indefinitely (only the pre-warm has a timeout otherwise).
        try:
            with _cf.ThreadPoolExecutor(max_workers=1) as _kex:
                _forecast = _kex.submit(
                    _kronos_pipeline.predict, _context, 1
                ).result(timeout=30)
        except _cf.TimeoutError:
            logging.warning("[KRONOS] inference timed out after 30s — returning 0.0")
            return 0.0
        # predict() returns a tensor of shape (batch, num_samples, prediction_length)
        _forecast_val = float(_forecast[0].mean().item())

        if _forecast_val == 0.0:
            return 0.0

        _cur_close = _closes[-1]
        _chg_pct = (_forecast_val - _cur_close) / _cur_close * 100 if _cur_close > 0 else 0.0

        if abs(_chg_pct) < 0.1:
            return 0.0

        _fore_dir = "buy" if _chg_pct > 0 else "sell"
        modifier = +0.5 if _fore_dir == direction else -0.5
        logging.info("[KRONOS] forecast_chg=%.3f%% fore_dir=%s code_dir=%s modifier=%.1f",
                     _chg_pct, _fore_dir, direction, modifier)
        return modifier

    except Exception as _e:
        logging.info("[KRONOS] inference error: %s", _e)
        return 0.0
