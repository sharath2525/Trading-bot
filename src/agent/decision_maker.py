"""Trading agent — anomaly detector and regime classifier only.

MEAN REVERSION ARCHITECTURE:
Code decides all direction/TP/SL/size. Claude has two narrow roles:
  1. claude_anomaly_check()  — called when price moves >3% in 5 candles.
                               Binary APPROVE/REJECT. max_tokens=10. cost ~$0.0001/call.
  2. claude_regime_classify() — optional 2-hourly regime check.
                               Returns CHOP|TREND_UP|TREND_DOWN|HIGH_RISK. max_tokens=20.
Neither function ever sets direction, TP, SL, or allocation.
"""

import os
import threading
import anthropic
from src.config_loader import CONFIG
import logging
from datetime import datetime, timezone

# Absolute paths so logs land correctly regardless of CWD (systemd, Docker, etc.)
_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
_LLM_LOG_PATH = os.path.join(_PROJECT_ROOT, "llm_requests.log")
_PROMPTS_LOG_PATH = os.path.join(_PROJECT_ROOT, "prompts.log")

# Serialize log writes across concurrent calls
_LOG_WRITE_LOCK = threading.Lock()


class TradingAgent:
    """Wrapper for Claude anomaly detection. Code owns all trade decisions."""

    def __init__(self, hyperliquid=None):
        self.client = anthropic.Anthropic(api_key=CONFIG["anthropic_api_key"])
        self.hyperliquid = hyperliquid
        logging.info("TradingAgent initialized — anomaly detector + regime classifier only (mean reversion arch)")

    def claude_anomaly_check(self, asset: str, price_change_pct: float,
                             direction: str, news_context: str = "") -> str:
        """Binary sanity check when price moves sharply before a mean reversion entry.

        Called ONLY when abs(price_change_5_candles) > ANOMALY_TRIGGER_PCT (default 3%).
        A sharp move before a reversion entry could mean:
          - News event (FOMC, hack, regulation, liquidation cascade) → REJECT
          - Normal volatility spike that the bands caught → APPROVE

        Returns 'APPROVE' or 'REJECT'. Fails closed on any error.
        max_tokens=10 — only needs one word answer.
        """
        _model = "claude-sonnet-4-6"
        _max_tok = int(CONFIG.get("ai_anomaly_max_tokens") or 10)
        _now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        _prompt = (
            f"Asset: {asset} | Direction: {direction.upper()} | UTC: {_now_utc}\n"
            f"Price moved {price_change_pct:.1f}% in the last 5 candles (5-minute bars).\n"
            f"A mean reversion entry is queued AGAINST this move.\n"
            f"News context: {news_context or 'none available'}\n\n"
            f"Is this a manipulative/news-driven spike that makes reversion entry UNSAFE?\n"
            f"Reply with exactly one word: APPROVE (safe to enter) or REJECT (too risky)."
        )

        try:
            resp = self.client.messages.create(
                model=_model,
                max_tokens=_max_tok,
                system="You are a crypto risk filter. Reply with exactly one word: APPROVE or REJECT.",
                messages=[{"role": "user", "content": _prompt}],
                timeout=15.0,
            )
            answer = resp.content[0].text.strip().upper() if resp.content else ""
            verdict = "APPROVE" if "APPROVE" in answer else "REJECT"

            input_tokens  = resp.usage.input_tokens
            output_tokens = resp.usage.output_tokens
            cost_usd = (input_tokens * 0.000003) + (output_tokens * 0.000015)

            logging.info("[ANOMALY] %s %s | chg=%.1f%% → %s | cost=$%.5f",
                         asset, direction, price_change_pct, verdict, cost_usd)

            with _LOG_WRITE_LOCK:
                with open(_LLM_LOG_PATH, "a", encoding="utf-8") as _lf:
                    _lf.write(
                        f"\n=== ANOMALY CHECK {asset} {_now_utc} ===\n"
                        f"direction={direction} price_chg={price_change_pct:.1f}% verdict={verdict}\n"
                        f"input={input_tokens} output={output_tokens} cost=${cost_usd:.5f}\n"
                        f"answer={answer}\n{'='*50}\n"
                    )

            return verdict

        except Exception as e:
            logging.warning("[ANOMALY] %s error — failing closed (REJECT): %s", asset, e)
            return "REJECT"

    def claude_regime_classify(self, asset: str, adx_1h: float,
                                bb_width_pct: float, atr_pct: float) -> str:
        """Classify current market regime from indicators.

        Called every REGIME_CHECK_INTERVAL_MINUTES (default 120 min).
        Used to inform dashboard/logging only — does NOT gate entries.
        Returns one of: CHOP | TREND_UP | TREND_DOWN | HIGH_RISK

        max_tokens=20 — short label only.
        """
        _model = "claude-sonnet-4-6"
        _max_tok = int(CONFIG.get("ai_regime_max_tokens") or 20)
        _now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        _prompt = (
            f"Asset: {asset} | UTC: {_now_utc}\n"
            f"1h ADX: {adx_1h:.1f} | BB width: {bb_width_pct:.2f}% | ATR%: {atr_pct:.3f}%\n\n"
            f"Classify the current market regime. Reply with exactly one word:\n"
            f"CHOP (low ADX, narrow bands, ranging)\n"
            f"TREND_UP (ADX > 25, price expanding upward)\n"
            f"TREND_DOWN (ADX > 25, price expanding downward)\n"
            f"HIGH_RISK (extreme ATR or BB width, manipulation risk)"
        )

        try:
            resp = self.client.messages.create(
                model=_model,
                max_tokens=_max_tok,
                system="You are a market regime classifier. Reply with exactly one word from: CHOP, TREND_UP, TREND_DOWN, HIGH_RISK.",
                messages=[{"role": "user", "content": _prompt}],
                timeout=15.0,
            )
            answer = (resp.content[0].text.strip().upper() if resp.content else "").split()[0]
            valid = {"CHOP", "TREND_UP", "TREND_DOWN", "HIGH_RISK"}
            regime = answer if answer in valid else "CHOP"

            input_tokens  = resp.usage.input_tokens
            output_tokens = resp.usage.output_tokens
            cost_usd = (input_tokens * 0.000003) + (output_tokens * 0.000015)

            logging.info("[REGIME] %s → %s | cost=$%.5f", asset, regime, cost_usd)

            with _LOG_WRITE_LOCK:
                with open(_LLM_LOG_PATH, "a", encoding="utf-8") as _lf:
                    _lf.write(
                        f"\n=== REGIME CLASSIFY {asset} {_now_utc} ===\n"
                        f"adx={adx_1h:.1f} bb_width={bb_width_pct:.2f}% atr%={atr_pct:.3f}% → {regime}\n"
                        f"input={input_tokens} output={output_tokens} cost=${cost_usd:.5f}\n{'='*50}\n"
                    )

            return regime

        except Exception as e:
            logging.warning("[REGIME] %s error — defaulting to CHOP: %s", asset, e)
            return "CHOP"
