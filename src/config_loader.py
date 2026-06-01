"""Centralized environment variable loading for the trading agent configuration."""

import json
import os
from dotenv import load_dotenv

load_dotenv()


def _get_env(name, default=None, required=False):
    value = os.getenv(name, default)
    if required and (value is None or value == ""):
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _get_bool(name, default=False):
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name, default=None):
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"Invalid integer for {name}: {raw}") from exc


def _get_list(name, default=None):
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    raw = raw.strip()
    if raw.startswith("[") and raw.endswith("]"):
        try:
            import json as _json
            parsed = _json.loads(raw)
            return [str(i).strip().strip("\"\'") for i in parsed if str(i).strip()]
        except Exception:
            pass
    values = [i.strip().strip("\"\'") for i in raw.split(",") if i.strip()]
    return values or default


CONFIG = {
    # Hyperliquid
    "hyperliquid_private_key": _get_env("HYPERLIQUID_PRIVATE_KEY") or _get_env("LIGHTER_PRIVATE_KEY"),
    "mnemonic": _get_env("MNEMONIC"),
    "hyperliquid_base_url": _get_env("HYPERLIQUID_BASE_URL"),
    "hyperliquid_network": _get_env("HYPERLIQUID_NETWORK", "mainnet"),
    "hyperliquid_vault_address": _get_env("HYPERLIQUID_VAULT_ADDRESS"),

    # LLM
    "anthropic_api_key": _get_env("ANTHROPIC_API_KEY", required=True),

    # Runtime
    "assets": _get_env("ASSETS"),
    "interval": _get_env("INTERVAL", "5m"),

    # Risk management
    "max_position_pct": _get_env("MAX_POSITION_PCT", "15"),
    "max_loss_per_position_pct": _get_env("MAX_LOSS_PER_POSITION_PCT", "8"),
    "max_leverage": _get_env("MAX_LEVERAGE", "5"),
    "max_total_exposure_pct": _get_env("MAX_TOTAL_EXPOSURE_PCT", "50"),
    "daily_loss_circuit_breaker_pct": _get_env("DAILY_LOSS_CIRCUIT_BREAKER_PCT", "12"),
    "mandatory_sl_pct": _get_env("MANDATORY_SL_PCT", "3"),
    "max_concurrent_positions": _get_env("MAX_CONCURRENT_POSITIONS", "3"),
    "min_balance_reserve_pct": _get_env("MIN_BALANCE_RESERVE_PCT", "20"),

    # API server
    "api_host": _get_env("API_HOST", "127.0.0.1"),
    "api_port": _get_env("APP_PORT") or _get_env("API_PORT") or "3000",
    "dashboard_token": _get_env("DASHBOARD_TOKEN"),

    # Telegram
    "telegram_bot_token": _get_env("TELEGRAM_BOT_TOKEN", ""),
    "telegram_chat_id": _get_env("TELEGRAM_CHAT_ID", ""),

    # Time-based exit backstop
    "max_trade_hours": _get_int("MAX_TRADE_HOURS", 1),

    # Execution frequency
    "taker_fee_pct": float(_get_env("TAKER_FEE_PCT", "0.00045")),
    "cooldown_minutes": _get_int("COOLDOWN_MINUTES", 15),
    "max_daily_trades": _get_int("MAX_DAILY_TRADES", 40),

    # BB + StochRSI signal parameters
    "bb_period": _get_int("BB_PERIOD", 20),
    "bb_std": float(_get_env("BB_STD", "2.0")),
    "stochrsi_period": _get_int("STOCHRSI_PERIOD", 14),
    "stochrsi_k_smooth": _get_int("STOCHRSI_K", 3),
    "stochrsi_d_smooth": _get_int("STOCHRSI_D", 3),
    "stochrsi_ob": float(_get_env("STOCHRSI_OB", "80.0")),
    "stochrsi_os": float(_get_env("STOCHRSI_OS", "20.0")),
    "time_limit_candles": _get_int("TIME_LIMIT_CANDLES", 8),

    # Regime + session controls
    "trend_pause_adx": float(_get_env("TREND_PAUSE_ADX", "30.0")),
    "session_block_start_utc": _get_int("SESSION_BLOCK_START_UTC", 0),
    "session_block_end_utc": _get_int("SESSION_BLOCK_END_UTC", 6),

    # AI anomaly detector + regime classifier
    "anomaly_trigger_pct": float(_get_env("ANOMALY_TRIGGER_PCT", "3.0")),
    "ai_anomaly_max_tokens": _get_int("AI_ANOMALY_MAX_TOKENS", 10),
    "ai_regime_max_tokens": _get_int("AI_REGIME_MAX_TOKENS", 20),
    "regime_check_interval_minutes": _get_int("REGIME_CHECK_INTERVAL_MINUTES", 120),
    "news_fetch_enabled": _get_bool("NEWS_FETCH_ENABLED", True),
}
