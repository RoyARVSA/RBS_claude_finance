"""
scan_signals.py – RBS 自動訊號掃描器（GitHub Actions cron）

Telegram 指令（傳給 Bot）：
  /add AAPL TSLA          – 加入觀察清單
  /remove TSLA            – 移除
  /list                   – 列出目前清單 + 門檻
  /threshold              – 查看門檻設定
  /set rsi_oversold 32    – 修改 RSI 超賣門檻
  /set rsi_overbought 68  – 修改 RSI 超買門檻
  /set price_change_pct 2.5 – 修改單日漲跌門檻（%）
  /set macd on|off        – 開關 MACD 訊號
  /set bb on|off          – 開關布林通道訊號
  /set atr on|off         – 開關 ATR 進出場提示
  /set vol_spike_ratio 2  – 成交量爆量倍數
  /set scan_market_only on|off – 只在美股開盤時間掃描
  /mute [小時數]          – 靜音 N 小時（預設 8）
  /unmute                 – 解除靜音
  /status                 – 查看 Bot 狀態 + 市場狀態
  /top [N]                – 顯示今日漲跌幅前 N 名
  /rank                   – 綜合評分排名（趨勢/MACD/RSI/布林/動量 合成 -1~+1）
  /scan                   – 立即掃描（忽略靜音與市場狀態）
  /clear                  – 清空觀察清單
  /help                   – 顯示此說明

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
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

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
    "rsi_oversold":       35.0,
    "rsi_overbought":     68.0,
    "price_change_pct":   3.0,
    "macd_enabled":       True,
    "bb_enabled":         True,
    "atr_enabled":        True,
    "vol_spike_ratio":    2.0,
    "scan_market_only":   True,   # skip scan when US market closed
}

ET = ZoneInfo("America/New_York")


# ── Market calendar helpers ───────────────────────────────────────────────────

def _us_holidays(year: int) -> set[date]:
    """Compute approximate NYSE holidays for a given year."""
    def _nth_weekday(year: int, month: int, n: int, weekday: int) -> date:
        """nth occurrence (1-based) of weekday (0=Mon) in month."""
        first = date(year, month, 1)
        delta = (weekday - first.weekday()) % 7
        return first + timedelta(days=delta + (n - 1) * 7)

    def _last_weekday(year: int, month: int, weekday: int) -> date:
        last = date(year, month + 1, 1) - timedelta(days=1) if month < 12 else date(year, 12, 31)
        delta = (last.weekday() - weekday) % 7
        return last - timedelta(days=delta)

    def _observed(d: date) -> date:
        """Shift weekend holidays to nearest weekday."""
        if d.weekday() == 5:   # Saturday → Friday
            return d - timedelta(days=1)
        if d.weekday() == 6:   # Sunday → Monday
            return d + timedelta(days=1)
        return d

    holidays = {
        _observed(date(year, 1, 1)),                           # New Year's Day
        _nth_weekday(year, 1, 3, 0),                           # MLK Day (3rd Mon Jan)
        _nth_weekday(year, 2, 3, 0),                           # Presidents' Day (3rd Mon Feb)
        _last_weekday(year, 5, 0) - timedelta(days=2),         # Good Friday (approx, Fri before Easter)
        _last_weekday(year, 5, 0),                             # Memorial Day (last Mon May)
        _observed(date(year, 6, 19)),                          # Juneteenth
        _observed(date(year, 7, 4)),                           # Independence Day
        _nth_weekday(year, 9, 1, 0),                           # Labor Day (1st Mon Sep)
        _nth_weekday(year, 11, 4, 3),                          # Thanksgiving (4th Thu Nov)
        _nth_weekday(year, 11, 4, 3) + timedelta(days=1),      # Black Friday (half-day, skip for safety)
        _observed(date(year, 12, 25)),                         # Christmas
    }
    return holidays


def market_status() -> dict:
    """Return current NYSE market status."""
    now_et = datetime.now(ET)
    today  = now_et.date()
    weekday = now_et.weekday()   # 0=Mon, 6=Sun

    if weekday >= 5:
        return {"open": False, "reason": f"週末（{'週六' if weekday==5 else '週日'}）"}

    holidays = _us_holidays(today.year)
    if today in holidays:
        return {"open": False, "reason": f"美股假日"}

    market_open  = now_et.replace(hour=9,  minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0,  second=0, microsecond=0)

    if now_et < market_open:
        opens_in = int((market_open - now_et).total_seconds() / 60)
        return {"open": False, "reason": f"盤前（{opens_in} 分鐘後開盤）"}
    if now_et > market_close:
        return {"open": False, "reason": "收盤後"}

    mins_left = int((market_close - now_et).total_seconds() / 60)
    return {"open": True, "reason": f"交易中（距收盤 {mins_left} 分鐘）", "now_et": now_et.strftime("%H:%M ET")}

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
        "📌 *清單管理*\n"
        "`/add AAPL TSLA` — 加入觀察清單\n"
        "`/remove TSLA` — 移除標的\n"
        "`/clear` — 清空清單\n"
        "`/list` — 列出清單 + 門檻\n\n"
        "📊 *門檻設定*\n"
        "`/threshold` — 查看目前設定\n"
        "`/set rsi_oversold 32` — RSI 超賣門檻\n"
        "`/set rsi_overbought 68` — RSI 超買門檻\n"
        "`/set price_change_pct 2.5` — 單日漲跌 %\n"
        "`/set macd on|off` — MACD 訊號\n"
        "`/set bb on|off` — 布林通道訊號\n"
        "`/set atr on|off` — ATR 進出場提示\n"
        "`/set vol_spike_ratio 2.0` — 爆量倍數\n"
        "`/set scan_market_only on|off` — 只在開盤時掃描\n\n"
        "🔕 *靜音控制*\n"
        "`/mute` — 靜音 8 小時\n"
        "`/mute 4` — 靜音 4 小時\n"
        "`/unmute` — 立即解除靜音\n\n"
        "📈 *資訊查詢*\n"
        "`/status` — Bot 狀態 + 市場狀態\n"
        "`/top 5` — 今日漲跌幅前 5 名\n"
        "`/rank` — 綜合評分排名（-1~+1）\n"
        "`/scan` — 立即掃描（忽略靜音）\n\n"
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
        f"`rsi_oversold`      = {th['rsi_oversold']}\n"
        f"`rsi_overbought`    = {th['rsi_overbought']}\n"
        f"`price_change_pct`  = {th['price_change_pct']}%\n"
        f"`vol_spike_ratio`   = {th.get('vol_spike_ratio', 2.0)}x\n"
        f"`macd_enabled`      = {th.get('macd_enabled', True)}\n"
        f"`bb_enabled`        = {th.get('bb_enabled', True)}\n"
        f"`atr_enabled`       = {th.get('atr_enabled', True)}\n"
        f"`scan_market_only`  = {th.get('scan_market_only', True)}\n\n"
        "用 `/set <key> <value>` 修改"
    )


def _cmd_status(state: dict) -> str:
    ms = market_status()
    mute_until = state.get("mute_until")
    muted = ""
    if mute_until:
        mu = datetime.fromisoformat(mute_until)
        if datetime.now(timezone.utc) < mu:
            mins = int((mu - datetime.now(timezone.utc)).total_seconds() / 60)
            muted = f"\n🔕 靜音中（剩餘 {mins} 分鐘）"
        else:
            muted = ""

    market_icon = "🟢" if ms["open"] else "🔴"
    return (
        "📡 *RBS Bot 狀態*\n\n"
        f"{market_icon} 市場：{ms['reason']}\n"
        f"👁 觀察清單：{len(state['watchlist'])} 支\n"
        f"🕐 最後更新：{state.get('last_scan_time', '尚未掃描')}"
        f"{muted}"
    )


def _cmd_top(state: dict, n: int = 5) -> str:
    tickers = state["watchlist"]
    if not tickers:
        return "❌ 觀察清單為空"
    try:
        raw = yf.download(tickers, period="2d", auto_adjust=True, progress=False)
        rows = []
        for t in tickers:
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    s = raw["Close"][t].dropna() if ("Close", t) in raw.columns or t in raw["Close"].columns else None
                else:
                    s = raw["Close"].dropna()
                if s is None or len(s) < 2:
                    continue
                chg = (float(s.iloc[-1]) / float(s.iloc[-2]) - 1) * 100
                rows.append((t, round(float(s.iloc[-1]), 2), round(chg, 2)))
            except Exception:
                continue
        if not rows:
            return "❌ 無法取得數據"
        rows.sort(key=lambda x: -x[2])
        lines = [f"📈 *今日漲跌排行 Top {n}*\n"]
        for t, price, chg in rows[:n]:
            icon = "🟢" if chg > 0 else "🔴"
            lines.append(f"{icon} *{t}* ${price}  ({chg:+.1f}%)")
        if len(rows) > n:
            lines.append(f"\n📉 *墊底 {n} 名*")
            for t, price, chg in rows[-n:]:
                icon = "🟢" if chg > 0 else "🔴"
                lines.append(f"{icon} *{t}* ${price}  ({chg:+.1f}%)")
        return "\n".join(lines)
    except Exception as e:
        return f"❌ 查詢失敗：{e}"


def _cmd_rank(state: dict) -> str:
    """Run a full composite-score scan and return the ranked board."""
    tickers = state["watchlist"]
    if not tickers:
        return "❌ 觀察清單為空"
    results = scan(tickers, state["thresholds"])
    if not results:
        return "❌ 無法取得數據"
    ranked = sorted(results, key=lambda r: -r.get("score", 0))
    lines = ["🏆 *綜合評分排名*（-1 極空 ~ +1 極多）\n"]
    for r in ranked:
        bar_len = int((r["score"] + 1) / 2 * 10)
        bar = "█" * bar_len + "░" * (10 - bar_len)
        lines.append(f"{r['emoji']} *{r['ticker']}*  `{bar}` {r['score']:+.2f}")
        lines.append(f"   {r['rating']} · ${r['price']} ({r['chg']:+.1f}%)")
    lines.append("\n_評分綜合趨勢/MACD/RSI/布林/動量_")
    return "\n".join(lines)


def _is_muted(state: dict) -> bool:
    mute_until = state.get("mute_until")
    if not mute_until:
        return False
    try:
        mu = datetime.fromisoformat(mute_until)
        return datetime.now(timezone.utc) < mu
    except Exception:
        return False


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

        elif cmd == "/status":
            reply = _cmd_status(state)

        elif cmd == "/add" and args:
            added = []
            for t in args:
                t = t.upper()
                if t not in state["watchlist"]:
                    state["watchlist"].append(t)
                    added.append(t)
            changed = True
            reply = (f"✅ 已加入：{', '.join(added)}\n現有 {len(state['watchlist'])} 支"
                     if added else "⚠️ 標的已在清單中")

        elif cmd == "/remove" and args:
            removed = []
            for t in args:
                t = t.upper()
                if t in state["watchlist"]:
                    state["watchlist"].remove(t)
                    removed.append(t)
            changed = True
            reply = (f"🗑 已移除：{', '.join(removed)}\n剩餘 {len(state['watchlist'])} 支"
                     if removed else "⚠️ 找不到該標的")

        elif cmd == "/clear":
            count = len(state["watchlist"])
            state["watchlist"] = []
            changed = True
            reply = f"🗑 已清空觀察清單（共 {count} 支）"

        elif cmd == "/mute":
            hours = int(args[0]) if args and args[0].isdigit() else 8
            until = datetime.now(timezone.utc) + timedelta(hours=hours)
            state["mute_until"] = until.isoformat()
            changed = True
            reply = f"🔕 靜音 {hours} 小時，直到 {until.strftime('%H:%M UTC')}"

        elif cmd == "/unmute":
            state.pop("mute_until", None)
            changed = True
            reply = "🔔 已解除靜音，恢復正常通知"

        elif cmd == "/top":
            n = int(args[0]) if args and args[0].isdigit() else 5
            reply = "🔍 查詢中…"
            _tg_send(token, src_chat or chat_id, reply)
            reply = _cmd_top(state, min(n, 10))

        elif cmd == "/rank":
            reply = "🏆 計算綜合評分中，約需 30 秒…"
            _tg_send(token, src_chat or chat_id, reply)
            reply = _cmd_rank(state)

        elif cmd == "/set" and len(args) >= 2:
            key, val = args[0].lower(), args[1].lower()
            th = state["thresholds"]
            bool_keys = {"macd_enabled", "bb_enabled", "atr_enabled", "scan_market_only"}
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


# ── Composite scoring ─────────────────────────────────────────────────────────

def _composite_score(close: pd.Series, high: pd.Series | None,
                     low: pd.Series | None, volume: pd.Series | None) -> dict:
    """
    Blend every indicator into a single -1 (極空) .. +1 (極多) score.
    Returns {"score", "rating", "emoji", "components"}.

    各子分數權重（偏趨勢跟隨，RSI/布林只在極端區作用以免與趨勢打架）：
      趨勢 (MA 排列)      35%
      MACD 動能           25%
      動量 (1個月報酬)    20%
      RSI 極端反轉        10%
      布林通道極端        10%
    成交量爆量作為「信心放大器」，最多 ±15% 加權。
    """
    comps: dict[str, float] = {}
    price = float(close.iloc[-1])

    # ── 1. 趨勢：價格相對 MA20/50/200 ───────────────────────────
    trend = 0.0
    for span, w in [(20, 0.4), (50, 0.35), (200, 0.25)]:
        if len(close) >= span:
            ma = float(close.rolling(span).mean().iloc[-1])
            trend += w * (1.0 if price > ma else -1.0)
    comps["trend"] = round(float(trend), 3)

    # ── 2. MACD 動能（histogram 正規化）─────────────────────────
    macd_s = 0.0
    if len(close) >= 35:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        sig_line  = macd_line.ewm(span=9, adjust=False).mean()
        h = float((macd_line - sig_line).iloc[-1])
        norm = h / price * 100 if price else 0
        macd_s = max(-1.0, min(1.0, norm * 4))
    comps["macd"] = round(float(macd_s), 3)

    # ── 3. 動量：1個月（約22交易日）報酬 ────────────────────────
    mom_s = 0.0
    if len(close) >= 22:
        ret_1m = price / float(close.iloc[-22]) - 1
        mom_s = max(-1.0, min(1.0, ret_1m * 8))   # ±12.5% → ±1
    comps["momentum"] = round(float(mom_s), 3)

    # ── 4. RSI：只在極端區作用（<35 偏多反彈、>65 偏空）─────────
    rsi = _rsi(close)
    if rsi < 35:
        rsi_s = (35 - rsi) / 25          # rsi=10 → +1
    elif rsi > 65:
        rsi_s = -(rsi - 65) / 25         # rsi=90 → -1
    else:
        rsi_s = 0.0                      # 35~65 中性，不干擾趨勢
    rsi_s = max(-1.0, min(1.0, rsi_s))
    comps["rsi"] = round(float(rsi_s), 3)

    # ── 5. 布林通道：只在貼邊（極端）時作用 ─────────────────────
    bb_s = 0.0
    if len(close) >= 20:
        ma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        u = float((ma20 + 2 * std20).iloc[-1])
        l = float((ma20 - 2 * std20).iloc[-1])
        if u > l:
            pct_b = (price - l) / (u - l)
            if pct_b < 0.15:
                bb_s = (0.15 - pct_b) / 0.15      # 貼下軌 → 偏多
            elif pct_b > 0.85:
                bb_s = -(pct_b - 0.85) / 0.15     # 貼上軌 → 偏空
    bb_s = max(-1.0, min(1.0, bb_s))
    comps["bollinger"] = round(float(bb_s), 3)

    # ── 加權合成 ─────────────────────────────────────────────────
    score = (
        0.35 * comps["trend"] +
        0.25 * comps["macd"] +
        0.20 * comps["momentum"] +
        0.10 * comps["rsi"] +
        0.10 * comps["bollinger"]
    )

    # ── 成交量信心放大器 ─────────────────────────────────────────
    if volume is not None and len(volume) >= 21:
        avg_vol  = float(volume.iloc[-21:-1].mean())
        curr_vol = float(volume.iloc[-1])
        if avg_vol > 0:
            vol_ratio = curr_vol / avg_vol
            if vol_ratio > 1.5:
                # amplify the existing direction up to +15%
                amp = min(0.15, (vol_ratio - 1.5) * 0.1)
                score *= (1 + amp)
    score = max(-1.0, min(1.0, score))

    # ── 評級 ─────────────────────────────────────────────────────
    if score >= 0.5:
        rating, emoji = "強力買進", "🟢🟢"
    elif score >= 0.2:
        rating, emoji = "買進", "🟢"
    elif score > -0.2:
        rating, emoji = "中性", "⚪"
    elif score > -0.5:
        rating, emoji = "賣出", "🔴"
    else:
        rating, emoji = "強力賣出", "🔴🔴"

    return {"score": round(score, 3), "rating": rating, "emoji": emoji, "components": comps}


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

            # ── Composite score (always computed) ─────────────────
            cs = _composite_score(close, high, low, volume)

            results.append({
                "ticker":  ticker,
                "price":   price,
                "rsi":     rsi,
                "chg":     chg,
                "score":   cs["score"],
                "rating":  cs["rating"],
                "emoji":   cs["emoji"],
                "signals": [s for s in signals if s],
            })
            flag = "🚨" if signals else "  "
            print(f"{flag} {ticker}: ${price}  RSI={rsi}  chg={chg:+.1f}%  "
                  f"score={cs['score']:+.2f}({cs['rating']})  signals={len(signals)}")

        except Exception as exc:
            print(f"  {ticker}: error – {exc}")

    return results


# ── Message builder ──────────────────────────────────────────────────────────

def _build_message(results: list[dict], timestamp: str) -> str | None:
    flagged = [r for r in results if r["signals"]]
    if not flagged:
        return None

    # Sort flagged signals by absolute conviction (strongest first)
    flagged_sorted = sorted(flagged, key=lambda r: -abs(r.get("score", 0)))

    lines = [f"🚨 *RBS 自動訊號掃描* — {timestamp}", ""]
    for r in flagged_sorted:
        arrow = "🟢" if r["chg"] > 0 else ("🔴" if r["chg"] < 0 else "⚪")
        score = r.get("score", 0)
        rating = r.get("rating", "")
        lines.append(
            f"{arrow} *{r['ticker']}* ${r['price']}  ({r['chg']:+.1f}%)  "
            f"RSI={r['rsi']}\n   📊 評分 *{score:+.2f}* ({rating})"
        )
        for s in r["signals"]:
            lines.append(f"   ↳ {s}")
        lines.append("")

    # ── Ranking board: top bullish / bearish across ALL scanned ─────
    ranked = sorted(results, key=lambda r: -r.get("score", 0))
    bullish = [r for r in ranked if r.get("score", 0) >= 0.2][:5]
    bearish = [r for r in ranked if r.get("score", 0) <= -0.2][-5:]

    if bullish:
        lines.append("📈 *今日最看多 Top 5*")
        for r in bullish:
            lines.append(f"   {r['emoji']} {r['ticker']}  {r['score']:+.2f} ({r['rating']})")
        lines.append("")
    if bearish:
        lines.append("📉 *今日最看空 Top 5*")
        for r in reversed(bearish):
            lines.append(f"   {r['emoji']} {r['ticker']}  {r['score']:+.2f} ({r['rating']})")
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

    tickers    = state["watchlist"]
    thresholds = state["thresholds"]
    ms = market_status()
    print(f"Market status: {ms['reason']}")

    # Step 2: Check mute & market hours
    if _is_muted(state):
        mu = datetime.fromisoformat(state["mute_until"])
        mins = int((mu - datetime.now(timezone.utc)).total_seconds() / 60)
        print(f"Muted for {mins} more minutes. Skipping scan.")
        save_state(state)
        return 0

    if thresholds.get("scan_market_only", True) and not ms["open"]:
        print(f"Market closed ({ms['reason']}). Skipping scan.")
        save_state(state)
        return 0

    print(f"\nWatchlist ({len(tickers)}): {', '.join(tickers)}")
    print(f"Thresholds: RSI {thresholds['rsi_oversold']}/{thresholds['rsi_overbought']}  "
          f"Chg≥{thresholds['price_change_pct']}%  "
          f"MACD={'on' if thresholds.get('macd_enabled') else 'off'}  "
          f"BB={'on' if thresholds.get('bb_enabled') else 'off'}  "
          f"ATR={'on' if thresholds.get('atr_enabled') else 'off'}\n")

    # Step 3: Scan
    print("── Running signal scan ──")
    results = scan(tickers, thresholds)
    state["last_scan_time"] = now

    # Step 4: Build & send message
    message = _build_message(results, now)
    if message is None:
        print("\nNo signals triggered. Nothing to send.")
    else:
        print(f"\n{len([r for r in results if r['signals']])} tickers flagged.")
        if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
            ok = _tg_send(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, message)
            print("Telegram sent." if ok else "Telegram send failed.")

    # Step 5: Save state
    save_state(state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
