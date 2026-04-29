"""
scan_signals.py – Standalone signal scanner for GitHub Actions cron.

Required env vars:
  TELEGRAM_TOKEN   – Bot token from @BotFather
  TELEGRAM_CHAT_ID – Your chat ID
  WATCHLIST        – Comma-separated tickers, e.g. "AAPL,TSLA,NVDA" (optional;
                     falls back to DEFAULT_WATCHLIST below)

Optional:
  MIN_RSI_OVERSOLD   – RSI threshold for oversold alert (default 35)
  MAX_RSI_OVERBOUGHT – RSI threshold for overbought alert (default 70)
  PRICE_CHANGE_PCT   – % daily change threshold for big-move alert (default 3.0)
"""

import os
import sys
import math
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_WATCHLIST = [
    # US 大型科技
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "TSLA", "META", "AMD", "INTC", "ORCL",
    # 金融
    "JPM", "BAC", "GS", "MS", "V", "MA",
    # 醫療
    "JNJ", "PFE", "MRNA", "UNH", "ABBV",
    # 能源
    "XOM", "CVX", "OXY", "SLB",
    # 消費
    "WMT", "COST", "TGT", "NKE", "SBUX",
    # ETF
    "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "ARKK",
    # 高波動個股
    "PLTR", "COIN", "MSTR", "RBLX", "RIVN", "LCID", "SOFI", "HOOD",
    # 中概 / 台股 ADR
    "BABA", "JD", "PDD", "BIDU", "TSM",
]

TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

raw_watchlist = os.environ.get("WATCHLIST", "")
WATCHLIST = [t.strip().upper() for t in raw_watchlist.split(",") if t.strip()] \
            if raw_watchlist else DEFAULT_WATCHLIST

RSI_OVERSOLD   = float(os.environ.get("MIN_RSI_OVERSOLD")   or 40)
RSI_OVERBOUGHT = float(os.environ.get("MAX_RSI_OVERBOUGHT") or 65)
PRICE_CHANGE_THRESHOLD = float(os.environ.get("PRICE_CHANGE_PCT") or 2.0)

# ---------------------------------------------------------------------------
# Signal computation helpers
# ---------------------------------------------------------------------------

def _rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff().dropna()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean().iloc[-1]
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - 100 / (1 + rs), 1)


def _ma_cross(close: pd.Series) -> str:
    """Return 'golden', 'death', or 'neutral'."""
    if len(close) < 52:
        return "neutral"
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    if ma20.iloc[-2] < ma50.iloc[-2] and ma20.iloc[-1] > ma50.iloc[-1]:
        return "golden"
    if ma20.iloc[-2] > ma50.iloc[-2] and ma20.iloc[-1] < ma50.iloc[-1]:
        return "death"
    return "neutral"


def _daily_change(close: pd.Series) -> float:
    if len(close) < 2:
        return 0.0
    return round((close.iloc[-1] / close.iloc[-2] - 1) * 100, 2)


def _vol_spike(close: pd.Series) -> bool:
    """True when latest 5-day vol > 1.5× trailing 20-day vol."""
    if len(close) < 25:
        return False
    returns = close.pct_change().dropna()
    recent_vol  = returns.iloc[-5:].std() * math.sqrt(252)
    baseline_vol = returns.iloc[-25:-5].std() * math.sqrt(252)
    return recent_vol > 1.5 * baseline_vol if baseline_vol > 0 else False


# ---------------------------------------------------------------------------
# Main scan
# ---------------------------------------------------------------------------

def scan(tickers: list[str]) -> list[dict]:
    """Download data for all tickers and return list of signal dicts."""
    print(f"Downloading data for {len(tickers)} tickers …")
    raw = yf.download(
        tickers,
        period="3mo",
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    results = []
    for ticker in tickers:
        try:
            # Handle both single and multi-ticker DataFrames
            if isinstance(raw.columns, pd.MultiIndex):
                close = raw["Close"][ticker].dropna()
            else:
                close = raw["Close"].dropna()

            if len(close) < 15:
                print(f"  {ticker}: insufficient data, skipping")
                continue

            rsi    = _rsi(close)
            cross  = _ma_cross(close)
            chg    = _daily_change(close)
            vol_sp = _vol_spike(close)
            price  = round(close.iloc[-1], 2)

            signals = []
            if rsi <= RSI_OVERSOLD:
                signals.append(f"RSI 超賣 ({rsi})")
            elif rsi >= RSI_OVERBOUGHT:
                signals.append(f"RSI 超買 ({rsi})")

            if cross == "golden":
                signals.append("黃金交叉 (MA20↑MA50)")
            elif cross == "death":
                signals.append("死亡交叉 (MA20↓MA50)")

            if abs(chg) >= PRICE_CHANGE_THRESHOLD:
                direction = "暴漲" if chg > 0 else "暴跌"
                signals.append(f"單日{direction} {chg:+.1f}%")

            if vol_sp:
                signals.append("波動率飆升")

            results.append({
                "ticker":  ticker,
                "price":   price,
                "rsi":     rsi,
                "chg":     chg,
                "cross":   cross,
                "vol_sp":  vol_sp,
                "signals": signals,
            })
            print(f"  {ticker}: ${price}  RSI={rsi}  chg={chg:+.1f}%  signals={signals}")

        except Exception as exc:
            print(f"  {ticker}: error – {exc}")

    return results


# ---------------------------------------------------------------------------
# Telegram notification
# ---------------------------------------------------------------------------

def _send_telegram(token: str, chat_id: str, text: str) -> bool:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.post(
            url,
            json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
            timeout=15,
        )
        if r.ok:
            print("Telegram message sent.")
            return True
        print(f"Telegram error: {r.status_code} {r.text}")
        return False
    except Exception as exc:
        print(f"Telegram exception: {exc}")
        return False


def _build_message(results: list[dict], timestamp: str) -> str | None:
    """Build the Telegram message. Returns None if nothing to report."""
    flagged = [r for r in results if r["signals"]]
    if not flagged:
        return None

    lines = [
        f"🚨 *RBS 自動訊號掃描* — {timestamp}",
        "",
    ]
    for r in flagged:
        signal_str = " | ".join(r["signals"])
        chg_emoji = "🟢" if r["chg"] > 0 else ("🔴" if r["chg"] < 0 else "⚪")
        lines.append(
            f"{chg_emoji} *{r['ticker']}* ${r['price']}  ({r['chg']:+.1f}%)"
        )
        lines.append(f"   ↳ {signal_str}")

    lines += [
        "",
        f"共掃描 {len(results)} 支，觸發 {len(flagged)} 支",
        "_由 GitHub Actions 自動執行_",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> int:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"=== RBS Signal Scanner  {now} ===")
    print(f"Watchlist ({len(WATCHLIST)}): {', '.join(WATCHLIST)}")
    print()

    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print(
            "WARNING: TELEGRAM_TOKEN or TELEGRAM_CHAT_ID not set.\n"
            "Signals will be printed but not sent."
        )

    results = scan(WATCHLIST)
    message = _build_message(results, now)

    if message is None:
        print("\nNo signals triggered. Nothing to send.")
        return 0

    print("\n--- Telegram message preview ---")
    print(message)
    print("--------------------------------\n")

    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        _send_telegram(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, message)
    else:
        print("Skipping Telegram: TELEGRAM_TOKEN or TELEGRAM_CHAT_ID not configured.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
