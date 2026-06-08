"""
scan_signals.py – RBS 自動訊號掃描器（GitHub Actions cron）

Telegram 指令（傳給 Bot）：
  /add AAPL TSLA     – 加入觀察清單
  /remove TSLA       – 移除
  /list              – 列出目前清單
  /threshold         – 查看門檻
  /set rsi_oversold 32       – 修改 RSI 超賣門檻
  /set rsi_overbought 68     – 修改 RSI 超買門檻
  /set price_change_pct 2.5  – 修改單日漲跌 % 門檻
  /set macd on|off           – 開關 MACD 訊號
  /set bb on|off             – 開關布林通道訊號
  /set atr on|off            – 開關 ATR 進出場訊號
  /scan              – 立即掃描並回傳結果
  /help              – 指令說明

環境變數（GitHub Secrets）：
  TELEGRAM_TOKEN    必填
  TELEGRAM_CHAT_ID  必填
  WATCHLIST         選填（初始清單，之後用 /add 管理）
"""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ── Config ──────────────────────────────────────────────────────────────────

STATE_FILE = Path(__file__).parent / "watchlist_state.json"

DEFAULT_WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "TSLA", "META", "AMD", "JPM", "SPY",
    "QQQ", "PLTR", "COIN", "TSM", "ARKK",
]

DEFAULT_THRESHOLDS = {
    "rsi_oversold":    35.0,
    "rsi_overbought":  68.0,
    "price_change_pct": 3.0,
    "macd_enabled":    True,
    "bb_enabled":      True,
    "atr_enabled":     True,
    "vol_spike_ratio": 2.0,
}

TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


# ── State persistence ────────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"State load error: {e}, using defaults")
    # Bootstrap from env var if first run
    raw_wl = os.environ.get("WATCHLIST", "")
    watchlist = [t.strip().upper() for t in raw_wl.split(",") if t.strip()] or DEFAULT_WATCHLIST
    return {
        "watchlist":     watchlist,
        "thresholds":    DEFAULT_THRESHOLDS.copy(),
        "last_update_id": 0,
    }


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"State saved → {STATE_FILE}")


# ── Telegram helpers ─────────────────────────────────────────────────────────

def _tg_get(token: str, method: str, params: dict | None = None) -> dict:
    url = f"https://api.telegram.org/bot{token}/{method}"
    r = requests.get(url, params=params, timeout=15)
    return r.json() if r.ok else {}


def _tg_send(token: str, chat_id: str, text: str) -> bool:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.post(
            url,
            json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
            timeout=15,
        )
        return r.ok
    except Exception as e:
        print(f"Telegram send error: {e}")
        return False


# ── Command processing ───────────────────────────────────────────────────────

def _cmd_help() -> str:
    return (
        "📋 *RBS Bot 指令說明*\n\n"
        "`/add AAPL TSLA` — 加入觀察清單\n"
        "`/remove TSLA` — 移除標的\n"
        "`/list` — 列出觀察清單\n"
        "`/threshold` — 查看目前門檻設定\n"
        "`/set rsi_oversold 32` — 修改 RSI 超賣門檻\n"
        "`/set rsi_overbought 68` — 修改 RSI 超買門檻\n"
        "`/set price_change_pct 2.5` — 修改單日漲跌門檻（%）\n"
        "`/set macd on|off` — 開關 MACD 訊號\n"
        "`/set bb on|off` — 開關布林通道訊號\n"
        "`/set atr on|off` — 開關 ATR 進出場提示\n"
        "`/set vol_spike_ratio 2.0` — 成交量爆量倍數\n"
        "`/scan` — 立即掃描並回傳結果\n"
        "`/help` — 顯示此說明"
    )


def _cmd_list(state: dict) -> str:
    wl = state["watchlist"]
    th = state["thresholds"]
    lines = [f"👁 *觀察清單（{len(wl)} 支）*", ""]
    lines += [f"• {t}" for t in wl]
    lines += [
        "",
        f"📊 RSI 超賣 ≤ {th['rsi_oversold']}",
        f"📊 RSI 超買 ≥ {th['rsi_overbought']}",
        f"📊 單日漲跌 ≥ {th['price_change_pct']}%",
        f"📊 MACD: {'✅' if th.get('macd_enabled',True) else '❌'}  "
        f"BB: {'✅' if th.get('bb_enabled',True) else '❌'}  "
        f"ATR: {'✅' if th.get('atr_enabled',True) else '❌'}",
    ]
    return "\n".join(lines)


def _cmd_threshold(state: dict) -> str:
    th = state["thresholds"]
    return (
        "⚙️ *目前門檻設定*\n\n"
        f"`rsi_oversold`     = {th['rsi_oversold']}\n"
        f"`rsi_overbought`   = {th['rsi_overbought']}\n"
        f"`price_change_pct` = {th['price_change_pct']}%\n"
        f"`vol_spike_ratio`  = {th.get('vol_spike_ratio', 2.0)}x\n"
        f"`macd_enabled`     = {th.get('macd_enabled', True)}\n"
        f"`bb_enabled`       = {th.get('bb_enabled', True)}\n"
        f"`atr_enabled`      = {th.get('atr_enabled', True)}\n\n"
        "用 `/set <key> <value>` 修改"
    )


def process_commands(token: str, chat_id: str, state: dict) -> tuple[dict, bool]:
    """Pull unread Telegram messages and process commands. Returns (state, changed)."""
    changed = False
    last_id = state.get("last_update_id", 0)
    data = _tg_get(token, "getUpdates", {"offset": last_id + 1, "timeout": 5})
    updates = data.get("result", [])

    for upd in updates:
        state["last_update_id"] = upd["update_id"]
        msg = upd.get("message", {})
        text = msg.get("text", "").strip()
        src_chat = str(msg.get("chat", {}).get("id", ""))

        if not text.startswith("/"):
            continue

        parts = text.split()
        cmd = parts[0].lower().split("@")[0]  # handle /cmd@botname format
        args = parts[1:]

        print(f"Command: {cmd} {args} from chat {src_chat}")
        reply = ""

        if cmd == "/help":
            reply = _cmd_help()

        elif cmd == "/list":
            reply = _cmd_list(state)

        elif cmd == "/threshold":
            reply = _cmd_threshold(state)

        elif cmd == "/add" and args:
            added = []
            for t in args:
                t = t.upper()
                if t not in state["watchlist"]:
                    state["watchlist"].append(t)
                    added.append(t)
            changed = True
            if added:
                reply = f"✅ 已加入：{', '.join(added)}\n現有 {len(state['watchlist'])} 支"
            else:
                reply = "⚠️ 標的已在清單中，無需重複新增"

        elif cmd == "/remove" and args:
            removed = []
            for t in args:
                t = t.upper()
                if t in state["watchlist"]:
                    state["watchlist"].remove(t)
                    removed.append(t)
            changed = True
            if removed:
                reply = f"🗑 已移除：{', '.join(removed)}\n剩餘 {len(state['watchlist'])} 支"
            else:
                reply = "⚠️ 找不到該標的"

        elif cmd == "/set" and len(args) >= 2:
            key, val = args[0].lower(), args[1].lower()
            th = state["thresholds"]
            bool_keys = {"macd_enabled", "bb_enabled", "atr_enabled"}
            float_keys = {"rsi_oversold", "rsi_overbought", "price_change_pct", "vol_spike_ratio"}
            if key in bool_keys:
                th[key] = val in ("on", "true", "1", "yes")
                changed = True
                reply = f"✅ `{key}` → {'開啟' if th[key] else '關閉'}"
            elif key in float_keys:
                try:
                    th[key] = float(val)
                    changed = True
                    reply = f"✅ `{key}` → {th[key]}"
                except ValueError:
                    reply = f"❌ 無效數值：{val}"
            else:
                reply = f"❌ 未知參數：{key}\n可用：{', '.join(bool_keys | float_keys)}"

        elif cmd == "/scan":
            reply = "🔍 掃描中，請稍候約 30 秒…"
            _tg_send(token, src_chat or chat_id, reply)
            results = scan(state["watchlist"], state["thresholds"])
            reply = _build_message(results, "手動觸發") or "✅ 掃描完成，目前無訊號觸發"

        else:
            reply = f"❓ 未知指令：{cmd}\n輸入 /help 查看說明"

        if reply:
            _tg_send(token, src_chat or chat_id, reply)

    return state, changed


# ── Signal algorithms ────────────────────────────────────────────────────────

def _rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff().dropna()
    gain  = delta.clip(lower=0).ewm(alpha=1/period, min_periods=period).mean().iloc[-1]
    loss  = (-delta).clip(lower=0).ewm(alpha=1/period, min_periods=period).mean().iloc[-1]
    if loss == 0:
        return 100.0
    return round(100 - 100 / (1 + gain / loss), 1)


def _macd(close: pd.Series) -> dict:
    """MACD(12,26,9). Returns signal info."""
    if len(close) < 35:
        return {"signal": "neutral", "histogram": 0.0}
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal_line

    prev_hist, curr_hist = float(hist.iloc[-2]), float(hist.iloc[-1])
    prev_macd = float(macd_line.iloc[-2])
    curr_macd = float(macd_line.iloc[-1])

    if prev_hist < 0 and curr_hist > 0:
        return {"signal": "golden", "label": f"MACD 金叉 (hist:{curr_hist:+.3f})", "histogram": curr_hist}
    if prev_hist > 0 and curr_hist < 0:
        return {"signal": "death", "label": f"MACD 死叉 (hist:{curr_hist:+.3f})", "histogram": curr_hist}
    # Momentum strengthening / weakening
    if curr_macd > 0 and curr_hist > prev_hist > 0:
        return {"signal": "bullish_momentum", "label": f"MACD 多頭加速", "histogram": curr_hist}
    if curr_macd < 0 and curr_hist < prev_hist < 0:
        return {"signal": "bearish_momentum", "label": f"MACD 空頭加速", "histogram": curr_hist}
    return {"signal": "neutral", "histogram": curr_hist}


def _bollinger(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> dict:
    """Bollinger Bands. Returns band position and breakout info."""
    if len(close) < period:
        return {"signal": "neutral", "pct_b": 0.5}
    ma  = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = ma + std_dev * std
    lower = ma - std_dev * std

    price = float(close.iloc[-1])
    u, l, m = float(upper.iloc[-1]), float(lower.iloc[-1]), float(ma.iloc[-1])
    bandwidth = (u - l) / m if m != 0 else 0
    pct_b = (price - l) / (u - l) if (u - l) > 0 else 0.5

    prev_price = float(close.iloc[-2])
    prev_upper = float(upper.iloc[-2])
    prev_lower = float(lower.iloc[-2])

    if price > u and prev_price <= prev_upper:
        return {"signal": "breakout_upper", "label": f"BB 突破上軌 ({price:.2f}>{u:.2f})", "pct_b": pct_b}
    if price < l and prev_price >= prev_lower:
        return {"signal": "breakout_lower", "label": f"BB 跌破下軌 ({price:.2f}<{l:.2f})", "pct_b": pct_b}
    if pct_b < 0.05:
        return {"signal": "near_lower", "label": f"BB 接近下軌 (超賣區)", "pct_b": pct_b}
    if pct_b > 0.95:
        return {"signal": "near_upper", "label": f"BB 接近上軌 (超買區)", "pct_b": pct_b}
    return {"signal": "neutral", "pct_b": pct_b}


def _atr_levels(close: pd.Series, high: pd.Series = None,
                low: pd.Series = None, period: int = 14) -> dict:
    """ATR-based entry zone, stop-loss, target. Uses close if H/L unavailable."""
    if len(close) < period + 1:
        return {"signal": "neutral"}
    if high is None or low is None:
        high = close
        low  = close
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = float(tr.rolling(period).mean().iloc[-1])
    price = float(close.iloc[-1])
    ma50  = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else price

    # Entry zone: within 1 ATR of 50-day MA from below (potential support)
    if abs(price - ma50) < atr and price < ma50:
        stop   = round(price - 1.5 * atr, 2)
        target = round(price + 2.5 * atr, 2)
        rr = round((target - price) / (price - stop), 1) if price > stop else 0
        return {
            "signal":  "entry_zone",
            "label":   f"ATR 進場區 (MA50支撐)｜止損{stop}｜目標{target}｜R:R={rr}",
            "stop":    stop,
            "target":  target,
            "rr":      rr,
        }
    return {"signal": "neutral", "atr": round(atr, 2)}


def _vol_spike(close: pd.Series, volume: pd.Series, ratio: float = 2.0) -> dict:
    """True when today's volume > ratio × 20-day avg volume."""
    if volume is None or len(volume) < 21:
        return {"signal": "neutral"}
    avg_vol  = float(volume.iloc[-21:-1].mean())
    curr_vol = float(volume.iloc[-1])
    if avg_vol > 0 and curr_vol > ratio * avg_vol:
        direction = "放量上漲" if float(close.iloc[-1]) > float(close.iloc[-2]) else "放量下跌"
        return {
            "signal": "vol_spike",
            "label":  f"爆量 {direction} ({curr_vol/avg_vol:.1f}x 均量)",
        }
    return {"signal": "neutral"}


def _ma_trend(close: pd.Series) -> dict:
    """MA20/50/200 alignment trend."""
    if len(close) < 52:
        return {"signal": "neutral", "label": ""}
    ma20 = float(close.rolling(20).mean().iloc[-1])
    ma50 = float(close.rolling(50).mean().iloc[-1])
    price = float(close.iloc[-1])

    prev_ma20 = float(close.rolling(20).mean().iloc[-2])
    prev_ma50 = float(close.rolling(50).mean().iloc[-2])

    if prev_ma20 < prev_ma50 and ma20 > ma50:
        return {"signal": "golden_cross", "label": "MA20/50 黃金交叉"}
    if prev_ma20 > prev_ma50 and ma20 < ma50:
        return {"signal": "death_cross",  "label": "MA20/50 死亡交叉"}
    return {"signal": "neutral"}


# ── Main scan ────────────────────────────────────────────────────────────────

def _col(df: pd.DataFrame, price: str, ticker: str) -> pd.Series | None:
    """Safely extract a price series from a yfinance multi-ticker DataFrame."""
    try:
        if isinstance(df.columns, pd.MultiIndex):
            # Default yfinance layout: (price_type, ticker)
            if (price, ticker) in df.columns:
                return df[(price, ticker)].dropna()
            # group_by="ticker" layout: (ticker, price_type)
            if (ticker, price) in df.columns:
                return df[(ticker, price)].dropna()
        else:
            return df[price].dropna() if price in df.columns else None
    except Exception:
        return None


def scan(tickers: list[str], thresholds: dict) -> list[dict]:
    rsi_lo  = thresholds.get("rsi_oversold",    35)
    rsi_hi  = thresholds.get("rsi_overbought",  68)
    chg_th  = thresholds.get("price_change_pct", 3.0)
    macd_on = thresholds.get("macd_enabled",  True)
    bb_on   = thresholds.get("bb_enabled",    True)
    atr_on  = thresholds.get("atr_enabled",   True)
    vol_r   = thresholds.get("vol_spike_ratio", 2.0)

    print(f"Batch-downloading {len(tickers)} tickers (6mo)…")
    try:
        raw = yf.download(tickers, period="6mo", auto_adjust=True,
                          progress=False, threads=True)
    except Exception as e:
        print(f"Batch download failed: {e}")
        return []

    # Single-ticker download returns flat columns; wrap for uniform handling
    single = len(tickers) == 1

    results = []
    for ticker in tickers:
        try:
            if single:
                close  = raw["Close"].squeeze().dropna()
                high   = raw.get("High",   pd.Series()).squeeze().dropna() or None
                low    = raw.get("Low",    pd.Series()).squeeze().dropna() or None
                volume = raw.get("Volume", pd.Series()).squeeze().dropna() or None
            else:
                close  = _col(raw, "Close",  ticker)
                high   = _col(raw, "High",   ticker)
                low    = _col(raw, "Low",    ticker)
                volume = _col(raw, "Volume", ticker)

            if close is None or len(close) < 20:
                print(f"  {ticker}: insufficient data, skipping")
                continue

            price  = round(float(close.iloc[-1]), 2)
            prev   = float(close.iloc[-2])
            chg    = round((price / prev - 1) * 100, 2)
            rsi    = _rsi(close)

            signals: list[str] = []

            # ── RSI ──────────────────────────────────────────────
            if rsi <= rsi_lo:
                signals.append(f"RSI 超賣 ({rsi}≤{rsi_lo})")
            elif rsi >= rsi_hi:
                signals.append(f"RSI 超買 ({rsi}≥{rsi_hi})")

            # ── Price change ──────────────────────────────────────
            if abs(chg) >= chg_th:
                signals.append(f"單日{'暴漲' if chg>0 else '暴跌'} {chg:+.1f}%")

            # ── MA cross ─────────────────────────────────────────
            ma = _ma_trend(close)
            if ma["signal"] in ("golden_cross", "death_cross"):
                signals.append(ma["label"])

            # ── MACD ─────────────────────────────────────────────
            if macd_on:
                mc = _macd(close)
                if mc["signal"] in ("golden", "death", "bullish_momentum", "bearish_momentum"):
                    signals.append(mc.get("label", ""))

            # ── Bollinger Bands ───────────────────────────────────
            if bb_on:
                bb = _bollinger(close)
                if bb["signal"] in ("breakout_upper", "breakout_lower", "near_lower", "near_upper"):
                    signals.append(bb.get("label", ""))

            # ── ATR entry zone ────────────────────────────────────
            if atr_on:
                at = _atr_levels(close, high, low)
                if at["signal"] == "entry_zone":
                    signals.append(at.get("label", ""))

            # ── Volume spike ──────────────────────────────────────
            if volume is not None:
                vs = _vol_spike(close, volume, vol_r)
                if vs["signal"] == "vol_spike":
                    signals.append(vs.get("label", ""))

            results.append({
                "ticker":  ticker,
                "price":   price,
                "rsi":     rsi,
                "chg":     chg,
                "signals": [s for s in signals if s],
            })
            flag = "🚨" if signals else "  "
            print(f"{flag} {ticker}: ${price}  RSI={rsi}  chg={chg:+.1f}%  signals={len(signals)}")

        except Exception as exc:
            print(f"  {ticker}: error – {exc}")

    return results


# ── Message builder ──────────────────────────────────────────────────────────

def _build_message(results: list[dict], timestamp: str) -> str | None:
    flagged = [r for r in results if r["signals"]]
    if not flagged:
        return None

    lines = [f"🚨 *RBS 自動訊號掃描* — {timestamp}", ""]
    for r in flagged:
        arrow = "🟢" if r["chg"] > 0 else ("🔴" if r["chg"] < 0 else "⚪")
        lines.append(f"{arrow} *{r['ticker']}* ${r['price']}  ({r['chg']:+.1f}%)  RSI={r['rsi']}")
        for s in r["signals"]:
            lines.append(f"   ↳ {s}")
        lines.append("")

    no_signal = [r["ticker"] for r in results if not r["signals"]]
    lines.append(f"共掃描 *{len(results)}* 支，觸發 *{len(flagged)}* 支")
    if no_signal:
        lines.append(f"_無訊號：{', '.join(no_signal[:10])}{'…' if len(no_signal)>10 else ''}_")
    lines.append("_由 GitHub Actions 自動執行 · 輸入 /help 管理清單_")
    return "\n".join(lines)


# ── Entrypoint ───────────────────────────────────────────────────────────────

def main() -> int:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"=== RBS Signal Scanner  {now} ===\n")

    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("WARNING: TELEGRAM_TOKEN / TELEGRAM_CHAT_ID not set. Signals printed only.")

    state = load_state()
    state_changed = False

    # Step 1: Process incoming Telegram commands
    if TELEGRAM_TOKEN:
        print("── Processing Telegram commands ──")
        state, changed = process_commands(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, state)
        if changed:
            state_changed = True
            save_state(state)
            print("State updated from commands.")

    tickers = state["watchlist"]
    thresholds = state["thresholds"]
    print(f"\nWatchlist ({len(tickers)}): {', '.join(tickers)}\n")
    print(f"Thresholds: RSI {thresholds['rsi_oversold']}/{thresholds['rsi_overbought']}  "
          f"Chg≥{thresholds['price_change_pct']}%  "
          f"MACD={'on' if thresholds.get('macd_enabled') else 'off'}  "
          f"BB={'on' if thresholds.get('bb_enabled') else 'off'}  "
          f"ATR={'on' if thresholds.get('atr_enabled') else 'off'}\n")

    # Step 2: Scan
    print("── Running signal scan ──")
    results = scan(tickers, thresholds)

    # Step 3: Build & send message
    message = _build_message(results, now)
    if message is None:
        print("\nNo signals triggered. Nothing to send.")
    else:
        print(f"\n{len([r for r in results if r['signals']])} tickers flagged.")
        if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
            ok = _tg_send(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, message)
            print("Telegram sent." if ok else "Telegram send failed.")

    # Step 4: Save state (update_id always changes)
    save_state(state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
