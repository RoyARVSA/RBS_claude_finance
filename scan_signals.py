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
  /calibrate              – 用歷史回測勝率校準各訊號權重（自我優化迴圈）
  /protections            – 查看防護機制（訊號冷卻 / 大盤風險濾網）狀態
  /risk [帳戶 風險%]       – 設定/查看部位風險（訊號附建議部位股數）
  /fundamentals TICKER    – 查公司基本面摘要（健康評分/ROE/估值，快取一天）（別名 /f）
  /options TICKER         – 選擇權情緒：Put/Call 比、隱含波動偏斜、情緒分數（別名 /opt）
  /insider TICKER         – SEC 內部人交易（Form 4，僅美股，買賣/cluster buy）（別名 /ins）
  /whales [編號]           – 超級投資人 13F 季度持倉增減（巴菲特/Burry/Ackman…）
  /alert TICKER 價位      – 到價警報：突破/跌破時推播，觸發後自動移除（/alert 看清單）
  /earnings [天數]        – 觀察清單近期財報日（晨報也會自動提醒 N 天內財報）
  /autotrade on|off       – Alpaca 模擬自動交易總開關（預設關）
  /positions /pnl /closeall – 模擬持倉 / 帳戶報酬 / 一鍵平倉
  /journal [N]            – 交易日誌（每筆自動交易的評分與原因）
  /rebalance [配置法]     – 再平衡顧問：Alpaca 持倉 vs HRP/Sharpe/風險平價 → 加減碼清單
  /briefing               – 立即生成每日 AI 晨報（每交易日 ET 08:30 自動推送）
  /today [帳戶 風險%]     – 當日交易計畫：VWAP/ORB/RVOL 訂單票（別名 /plan）
  /plantest [apply|clear] – 當日計畫 60 日歷史回測；apply 套用校準（每週亦自動跑）
  /plantest opt [apply]   – 參數尋優（ORB 分鐘×停損 ATR×目標 R:R，walk-forward 把關）
  /weekly                 – 立即生成每週深度週報（每週日 ET 18:00 後自動推送）
  /committee TICKER       – 機構決策會議：分析師×4→對辯→交易員→風控→PM（別名 /cmt）
  /set mtf_enabled on/off – 週線同向確認（日線分數與週線同向加強、背離減弱）
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
import os
import sys
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf

# ── Config ──────────────────────────────────────────────────────────────────

STATE_FILE = Path(__file__).parent / "watchlist_state.json"
JOURNAL_FILE = Path(__file__).parent / "trade_journal.json"

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
    # ── Protections (freqtrade-style) ──
    "cooldown_enabled":   True,   # 同標的同類訊號冷卻，防洗版
    "cooldown_hours":     6.0,    # 冷卻時數
    "regime_filter_enabled": True,  # 大盤風險濾網（加狀態頭）
    # ── Position sizing ──
    "position_sizing_enabled": True,  # 訊號附建議部位
    "account_size":       100000.0,   # 帳戶總值（USD）
    "risk_pct":           0.01,       # 單筆風險比例
    "atr_mult":           1.5,        # ATR 停損倍數
    # ── Daily briefing ──
    "briefing_enabled":   True,       # 每日 AI 晨報
    "briefing_hour_et":   8.5,        # 觸發時間（ET 小數時，8.5 = 08:30）
    # ── Multi-timeframe ──
    "mtf_enabled":        True,       # 週線同向確認（軟性調整評分）
    # ── Earnings ──
    "earnings_alert_days": 5.0,       # 財報前 N 天提醒（晨報 + /earnings）
    # ── Alpaca paper trading（預設關閉，須 /autotrade on）──
    "autotrade_enabled":  False,      # 自動下模擬單總開關
    "at_buy_threshold":   0.5,        # 評分 ≥ 此值 → 開多
    "at_exit_threshold": -0.2,        # 評分 ≤ 此值 → 平倉
    "at_max_positions":   10.0,       # 最多持倉檔數
    "at_max_position_pct": 0.15,      # 每檔最多佔淨值
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
            state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            # 向後相容：補上新版才有的 threshold 預設鍵（不覆蓋既有值）
            th = state.setdefault("thresholds", {})
            for k, v in DEFAULT_THRESHOLDS.items():
                th.setdefault(k, v)
            state.setdefault("signal_history", {})
            state.setdefault("last_update_id", 0)
            state.setdefault("watchlist", DEFAULT_WATCHLIST.copy())
            return state
        except Exception as e:
            print(f"State load error: {e}, using defaults")
    # Bootstrap from env var if first run
    raw_wl = os.environ.get("WATCHLIST", "")
    watchlist = [t.strip().upper() for t in raw_wl.split(",") if t.strip()] or DEFAULT_WATCHLIST
    return {
        "watchlist":      watchlist,
        "thresholds":     DEFAULT_THRESHOLDS.copy(),
        "signal_history": {},
        "last_update_id": 0,
    }


def save_state(state: dict) -> None:
    # 修剪只增不減的每日快取（fund/earnings）：留 35 天內的，移除的標的不再永久佔位
    cutoff = (datetime.now(timezone.utc) - timedelta(days=35)).strftime("%Y-%m-%d")
    for ck, datekey in (("fund_cache", "date"), ("earnings_cache", "checked")):
        c = state.get(ck)
        if isinstance(c, dict):
            state[ck] = {k: v for k, v in c.items()
                         if isinstance(v, dict) and str(v.get(datekey, "")) >= cutoff}
    # 原子寫入：先寫暫存檔再 os.replace，中途被砍不會留下截斷的 JSON
    tmp = STATE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, STATE_FILE)
    print(f"State saved → {STATE_FILE}")


# ── Telegram helpers ─────────────────────────────────────────────────────────

def _tg_get(token: str, method: str, params: dict | None = None) -> dict:
    url = f"https://api.telegram.org/bot{token}/{method}"
    try:
        r = requests.get(url, params=params, timeout=15)
        return r.json() if r.ok else {}
    except Exception as e:          # Telegram 瞬斷不該廢掉整輪掃描
        print(f"_tg_get {method} error: {e}")
        return {}


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
        "`/set mtf_enabled on|off` — 週線同向確認\n"
        "`/set vol_spike_ratio 2.0` — 爆量倍數\n"
        "`/set scan_market_only on|off` — 只在開盤時掃描\n\n"
        "🔕 *靜音控制*\n"
        "`/mute` — 靜音 8 小時\n"
        "`/mute 4` — 靜音 4 小時\n"
        "`/unmute` — 立即解除靜音\n\n"
        "🛡 *防護機制*\n"
        "`/protections` — 查看冷卻/大盤濾網狀態\n"
        "`/set cooldown_hours 4` — 訊號冷卻時數\n"
        "`/set cooldown_enabled off` — 關閉冷卻\n"
        "`/set regime_filter_enabled off` — 關閉大盤濾網\n\n"
        "💰 *部位風險*\n"
        "`/risk 100000 1` — 設定帳戶 $10萬、單筆風險 1%\n"
        "`/risk` — 查看目前部位設定\n\n"
        "📈 *資訊查詢*\n"
        "`/status` — Bot 狀態 + 市場狀態\n"
        "`/top 5` — 今日漲跌幅前 5 名\n"
        "`/rank` — 綜合評分排名（-1~+1）\n"
        "`/fundamentals AAPL`（或 `/f`）— 公司基本面摘要\n"
        "`/options AAPL`（或 `/opt`）— 選擇權情緒（Put/Call、IV 偏斜）\n"
        "`/insider AAPL`（或 `/ins`）— SEC 內部人交易（Form 4，僅美股）\n"
        "`/whales [編號]` — 超級投資人 13F 季度增減倉（巴菲特/Burry…）\n"
        "`/alert AAPL 200` — 到價警報（`/alert` 看清單、`/alert del AAPL` 刪）\n"
        "`/earnings [天數]` — 觀察清單近期財報日\n"
        "`/briefing` — 立即生成每日晨報\n"
        "`/today [帳戶 風險%]`（或 `/plan`）— 當日交易計畫：VWAP/ORB 進場票（進場/停損/停利/股數）\n"
        "`/plantest [apply|clear]` — 當日計畫 60 日回測；apply 套用校準（每週自動跑，`/set plan_autocal_enabled off` 關）\n"
        "`/plantest opt [apply]` — 參數尋優：ORB×停損×R:R 掃 27 組，驗證段勝過預設才推薦\n"
        "`/weekly` — 立即生成每週深度週報（指數/強弱/計分板/RRG/下週行事曆）\n"
        "`/committee NVDA`（或 `/cmt`）— 開一場機構決策會議（需 LLM key，約 1-3 分）\n\n"
        "🤖 *模擬交易（Alpaca paper）*\n"
        "`/autotrade on|off` — 自動下模擬單（預設關）\n"
        "`/positions` — 目前持倉 + 損益\n"
        "`/pnl` — 帳戶淨值 + 報酬\n"
        "`/journal [N]` — 交易日誌（含評分/原因）\n"
        "`/rebalance [hrp|max_sharpe|min_vol|erc|equal]` — 再平衡顧問（持倉 vs 目標權重 → 加減碼清單）\n"
        "`/closeall` — 一鍵平倉\n"
        "`/scan` — 立即掃描（忽略靜音/冷卻）\n"
        "`/calibrate` — 回測校準訊號權重（自我優化）\n\n"
        "☀️ *晨報*：每交易日 ET 08:30 自動推送\n"
        "`/set briefing_enabled off` — 關閉晨報\n"
        "`/set briefing_hour_et 9` — 改晨報時間（ET 小數時）\n\n"
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


def _cmd_fundamentals(state: dict, ticker: str) -> str:
    """查詢單檔基本面摘要（快取一天，避免重複慢呼叫）。"""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cache = state.setdefault("fund_cache", {})
    hit = cache.get(ticker)
    if hit and hit.get("date") == today:
        return hit["text"]

    try:
        import fundamentals as fa
    except Exception:
        return "❌ 找不到 fundamentals.py（請同步該檔案）"

    try:
        d = fa.fetch_fundamentals(ticker)
    except Exception as e:
        return f"❌ {ticker} 查詢失敗：{e}"

    if not d.get("ok"):
        return f"⚠️ {ticker}：{d.get('error') or '無基本面資料'}"

    hs = fa.health_score(d)

    def _pct(x):
        return f"{x*100:.1f}%" if x is not None else "—"
    def _num(x, f="{:.2f}"):
        return f.format(x) if x is not None else "—"

    score_line = (f"{hs['score']:.0f}（{hs['rating']}，{hs['covered']}/5構面）"
                  if hs["score"] is not None else "資料不足")
    text = (
        f"🏢 *{d['name']}* ({ticker})\n"
        f"{d.get('sector') or ''} {('· ' + d.get('industry')) if d.get('industry') else ''}\n\n"
        f"📊 *財務健康*：{score_line}\n"
        f"ROE：{_pct(d.get('roe'))}　淨利率：{_pct(d.get('net_margin'))}\n"
        f"營收成長：{_pct(d.get('revenue_growth'))}　盈餘成長：{_pct(d.get('earnings_growth'))}\n"
        f"P/E：{_num(d.get('pe'))}　PEG：{_num(d.get('peg'))}　P/B：{_num(d.get('pb'))}\n"
        f"負債權益比：{_num(d.get('debt_to_equity'))}　流動比：{_num(d.get('current_ratio'))}\n"
    )
    tgt, pr = d.get("target_mean"), d.get("price")
    if tgt and pr:
        text += f"分析師目標：{tgt:.2f}（{(tgt/pr-1)*100:+.1f}%）\n"
    text += "\n_資料 yfinance，季報非即時 · 僅供參考_"

    cache[ticker] = {"date": today, "text": text}
    return text


def _cmd_options(ticker: str) -> str:
    """查詢單檔選擇權情緒（Put/Call、隱含波動偏斜、情緒分數）。"""
    try:
        import options_sentiment as ops
    except Exception:
        return "❌ 找不到 options_sentiment.py（請同步該檔案）"
    try:
        summ = ops.fetch_options(ticker)
    except Exception as e:
        return f"❌ {ticker} 選擇權查詢失敗：{e}"
    if not summ:
        return f"⚠️ {ticker}：查無選擇權資料（非選擇權標的/外股/無報價）"
    sent = ops.sentiment(summ)
    return f"🎭 *選擇權情緒* {ticker}\n" + ops.format_options_text(summ, sent) \
        + "\n_定位訊號，非投資建議_"


def _cmd_insider(ticker: str) -> str:
    """查詢單檔 SEC Form 4 內部人交易摘要（僅美股）。"""
    try:
        import sec_insider as si
    except Exception:
        return "❌ 找不到 sec_insider.py（請同步該檔案）"
    if "." in ticker:
        return f"⚠️ {ticker}：SEC Form 4 僅涵蓋美股掛牌公司"
    try:
        summ = si.fetch_insider(ticker)
    except Exception as e:
        return f"❌ {ticker} 內部人查詢失敗：{e}"
    if not summ:
        return f"⚠️ {ticker}：查無近期 Form 4（無內部人申報或代碼無法對應 CIK）"
    win = f"近{summ.get('window_days', 90)}天"
    return f"🕵️ *內部人交易* {ticker}\n" + si.format_insider_text(summ, win) \
        + "\n_輔助訊號，非投資建議_"


def _cmd_committee(state: dict, ticker: str) -> tuple[str, bool]:
    """
    Bot 版機構決策委員會（標準流程 ~9 次 LLM 呼叫，約 1-3 分鐘）。
    重用 committee.py 全套角色提示與 web 端相同的資料模組；
    裁決自動記入 state 反思記憶（source=committee）→ 計分板統一累積。
    回 (回覆文字, state_changed)。
    """
    if not os.environ.get("LLM_API_KEY"):
        return "⚠️ 未設定 LLM_API_KEY（GitHub Secrets），無法開會", False
    try:
        import committee as cmt
        import sector_scan as ssc
    except Exception as e:
        return f"❌ 模組缺失：{e}", False

    # ① 資料層
    tk = ticker.upper()
    try:
        raw = yf.download(tk, period="1y", auto_adjust=True, progress=False)
        import backtest as bt
        df = bt.normalize_ohlc(raw, tk)
        close = df["Close"].dropna()
        if len(close) < 60:
            return f"⚠️ {tk} 歷史資料不足", False
    except Exception as e:
        return f"❌ {tk} 資料抓取失敗：{e}", False
    tech = ssc.price_metrics(close) or {}
    quant = None
    try:
        quant = _composite_score(df["Close"], df.get("High"), df.get("Low"), df.get("Volume"))
    except Exception:
        pass
    px = float(close.iloc[-1])
    tech_dom = (f"現價 {px:.2f}　近1月 {tech.get('return_1m')}　近3月 {tech.get('return_3m')}　"
                f"年化波動 {tech.get('ann_vol')}　RSI {tech.get('rsi')}\n"
                f"量化綜合評分 {(quant or {}).get('score', '無')}（{(quant or {}).get('rating', '')}）")
    fund_dom = "（基本面抓取失敗）"
    try:
        import fundamentals as fa
        fd = fa.fetch_fundamentals(tk)
        hsd = fa.health_score(fd) if fd.get("ok") else {}
        import analyst_data as ad
        an = ad.fetch_analyst(tk)
        rat, tgt, sur = an.get("ratings"), an.get("targets"), an.get("surprises")
        fund_dom = (f"財務健康 {hsd.get('score', '無')}　P/E {fd.get('pe')}　ROE {fd.get('roe')}　"
                    f"營收成長 {fd.get('revenue_growth')}\n"
                    f"分析師共識 {(rat or {}).get('score', '無')}　"
                    f"目標價上檔 {(tgt or {}).get('upside_mean', '無')}　"
                    f"EPS Beat {(sur or {}).get('beat_rate', '無')}")
    except Exception:
        pass
    chips_parts = []
    try:
        if tk.endswith((".TW", ".TWO")):
            import tw_flows as twf
            acc = twf.fetch_flows(tk)
            t_ = twf.flows_text(acc) if acc else None
            if t_:
                chips_parts.append(t_)
        else:
            import options_sentiment as ops
            o = ops.fetch_options(tk)
            if o:
                chips_parts.append(ops.format_options_text(o))
            import sec_insider as si
            i_ = si.fetch_insider(tk, max_filings=8)
            if i_:
                chips_parts.append(si.format_insider_text(i_))
    except Exception:
        pass
    chips_dom = "\n".join(chips_parts) or "（籌碼資料暫缺）"
    macro_dom = "（無 FRED key）"
    try:
        fk = os.environ.get("FRED_API_KEY", "")
        if fk:
            import macro as mc
            md_ = mc.fetch_macro(fk)
            if md_:
                macro_dom = mc.macro_summary_text(md_)
    except Exception:
        pass

    # ② 委員會流程（重用 committee.py 提示；每次呼叫獨立容錯）
    def call(p, mt=380):
        r = _llm_complete(p, max_tokens=mt)
        if not r:
            raise RuntimeError("LLM 呼叫失敗")
        return r

    try:
        domains = {"technical": tech_dom, "fundamental": fund_dom,
                   "chips": chips_dom, "macro": macro_dom}
        analysts = {d: call(cmt.analyst_prompt(d) + f"\n\n=== {tk} 資料 ===\n" + t_)
                    for d, t_ in domains.items()}
        all_rep = "\n\n".join(f"【{cmt.ANALYST_ROLES[d][0]}】\n{t_}"
                              for d, t_ in analysts.items())
        bull = call(cmt.RESEARCHER_BULL + "\n\n" + all_rep, 400)
        bear = call(cmt.RESEARCHER_BEAR + "\n\n" + all_rep, 400)
        trader = call(cmt.TRADER_PROMPT + "\n\n" + all_rep
                      + f"\n\n【多方】{bull}\n【空方】{bear}", 420)
        hard = cmt.hard_risk_check({"quant_score": (quant or {}).get("score"),
                                    "ann_vol": tech.get("ann_vol")})
        risk = call(cmt.risk_prompt(hard) + "\n\n【交易員提案】\n" + trader, 320)
        pm = call(cmt.PM_PROMPT + "\n\n" + all_rep
                  + f"\n\n【多方】{bull}\n【空方】{bear}\n\n【交易員】{trader}\n\n【風控】{risk}", 520)
    except Exception as e:
        return f"❌ 會議中斷：{e}", False

    verdict = cmt.parse_verdict(pm)
    cross = cmt.compare_with_quant(verdict.get("verdict"), (quant or {}).get("score"))
    # ③ 裁決入反思記憶（與 web 端計分板統一）
    changed = False
    try:
        import reflection as rfl
        vmap = {"買進": 0.8, "迴避": -0.8}
        sc = vmap.get(verdict.get("verdict"))
        if sc is not None and px > 0:
            changed = rfl.record_pick(state, tk, sc, px,
                                      datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                                      source="committee")
    except Exception:
        pass

    stances = "　".join(f"{cmt.ANALYST_ROLES[d][0][:2]} "
                        f"{(f'{s:+.1f}' if (s := cmt.parse_stance(t_)) is not None else '—')}"
                        for d, t_ in analysts.items())
    reply = (f"🏛 *委員會裁決 {tk}*\n"
             f"結論：*{verdict.get('verdict') or '—'}*　信心：{verdict.get('confidence') or '—'}"
             + (f"　時間框架：{verdict['horizon']}" if verdict.get("horizon") else "") + "\n"
             f"量化評分 {(quant or {}).get('score', '—')}　{cross.get('agreement', '')}\n"
             f"立場：{stances}\n\n{pm[:1200]}\n"
             + ("\n⚠️ 硬性風控：" + "；".join(hard) if hard else "")
             + "\n_9 次 LLM 呼叫 · 已記入計分板 · 非投資建議_")
    return reply, changed


def _cmd_positions() -> str:
    """Alpaca 目前持倉 + 未實現損益。"""
    key, secret = _alpaca_keys()
    if not key or not secret:
        return "⚠️ 未設定 Alpaca key（ALPACA_KEY_ID / ALPACA_SECRET_KEY）"
    try:
        import alpaca_trader as at
    except Exception:
        return "❌ 找不到 alpaca_trader.py"
    pos = at.get_positions(key, secret)
    if not pos:
        return "📭 目前無持倉"
    lines = ["📊 *模擬持倉*\n"]
    total_pl = 0.0
    for sym, p in pos.items():
        pl = p.get("unrealized_pl") or 0
        plpc = p.get("unrealized_plpc")
        total_pl += pl
        icon = "🟢" if pl >= 0 else "🔴"
        pct = f"（{plpc*100:+.1f}%）" if plpc is not None else ""
        qty = p.get("qty") or 0
        lines.append(f"{icon} *{sym}* x{qty:g}　損益 {pl:+.2f}{pct}")
    lines.append(f"\n未實現損益合計：*{total_pl:+.2f}*")
    return "\n".join(lines)


def _cmd_pnl() -> str:
    """Alpaca 帳戶淨值與報酬。"""
    key, secret = _alpaca_keys()
    if not key or not secret:
        return "⚠️ 未設定 Alpaca key"
    try:
        import alpaca_trader as at
    except Exception:
        return "❌ 找不到 alpaca_trader.py"
    acc = at.get_account(key, secret)
    if not acc:
        return "❌ 帳戶讀取失敗（確認 key 與網路）"
    r = at.account_return(acc)
    cash = at._f(acc.get("cash"))
    lines = ["💰 *模擬帳戶*\n"]
    if r["equity"] is not None:
        lines.append(f"淨值：${r['equity']:,.2f}")
    if r["day_change"] is not None:
        icon = "🟢" if r["day_change"] >= 0 else "🔴"
        pct = f"（{r['day_pct']*100:+.2f}%）" if r["day_pct"] is not None else ""
        lines.append(f"{icon} 今日：{r['day_change']:+,.2f}{pct}")
    if cash is not None:
        lines.append(f"現金：${cash:,.2f}")
    return "\n".join(lines)


def _cmd_rank(state: dict) -> str:
    """Run a full composite-score scan and return the ranked board."""
    tickers = state["watchlist"]
    if not tickers:
        return "❌ 觀察清單為空"
    results = scan(tickers, state["thresholds"], calibration=_calibration_weights(state))
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

        # 安全：只接受授權聊天室（任何人都找得到 bot username；未授權者可下 /closeall 等指令）
        if src_chat != str(chat_id):
            print(f"Ignored message from unauthorized chat {src_chat}")
            continue

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

        elif cmd == "/alert":
            alerts = state.setdefault("price_alerts", [])
            if not args:
                if not alerts:
                    reply = ("🎯 尚無到價警報。\n用法：`/alert AAPL 200`（依現價自動判斷突破/跌破）\n"
                             "`/alert AAPL >200`、`/alert AAPL <150` 指定方向；`/alert del AAPL` 刪除")
                else:
                    lines = ["🎯 *到價警報*"]
                    for a in alerts:
                        arrow = "≥" if a.get("op") == ">=" else "≤"
                        lines.append(f"• {a['ticker']} {arrow} {float(a['level']):.2f}")
                    lines.append("_觸發後自動移除 · `/alert del AAPL` 刪除_")
                    reply = "\n".join(lines)
            elif args[0].lower() in ("del", "delete", "rm") and len(args) >= 2:
                tk = args[1].upper()
                before = len(alerts)
                state["price_alerts"] = [a for a in alerts if a.get("ticker") != tk]
                changed = True
                reply = f"🗑 已刪除 {tk} 的 {before - len(state['price_alerts'])} 個警報"
            elif len(args) >= 2:
                tk = args[0].upper()
                lvl_s = args[1]
                op = None
                if lvl_s.startswith(">"):
                    op, lvl_s = ">=", lvl_s[1:]
                elif lvl_s.startswith("<"):
                    op, lvl_s = "<=", lvl_s[1:]
                try:
                    level = float(lvl_s)
                except ValueError:
                    level = None
                if level is None or level <= 0:
                    reply = "用法：`/alert AAPL 200` 或 `/alert AAPL >200`、`/alert AAPL <150`"
                elif len(alerts) >= 20:
                    reply = "⚠️ 警報上限 20 個，請先用 `/alert del TICKER` 清理"
                else:
                    if op is None:                       # 依現價自動判方向
                        px = _alert_prices([tk]).get(tk)
                        op = ">=" if (px is None or px < level) else "<="
                    alerts.append({"ticker": tk, "op": op, "level": level})
                    changed = True
                    arrow = "向上突破 ≥" if op == ">=" else "向下跌破 ≤"
                    reply = f"🎯 已設定：{tk} {arrow} {level:.2f} 時通知（觸發後自動移除）"
            else:
                reply = "用法：`/alert AAPL 200`、`/alert`（看清單）、`/alert del AAPL`"

        elif cmd == "/mute":
            hours = int(args[0]) if args and args[0].isdigit() else 8
            hours = max(1, min(hours, 720))   # clamp：超大數字會讓 timedelta 溢位、毒掉之後每輪 cron
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
            bool_keys = {"macd_enabled", "bb_enabled", "atr_enabled", "scan_market_only",
                         "cooldown_enabled", "regime_filter_enabled",
                         "position_sizing_enabled", "briefing_enabled", "mtf_enabled",
                         "autotrade_enabled", "weekly_enabled", "plan_autocal_enabled"}
            float_keys = {"rsi_oversold", "rsi_overbought", "price_change_pct",
                          "vol_spike_ratio", "cooldown_hours",
                          "account_size", "risk_pct", "atr_mult", "briefing_hour_et",
                          "earnings_alert_days", "at_buy_threshold", "at_exit_threshold",
                          "at_max_positions", "at_max_position_pct"}
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
            results = scan(state["watchlist"], state["thresholds"],
                           calibration=_calibration_weights(state))
            reply = _build_message(results, "手動觸發") or "✅ 掃描完成，目前無訊號觸發"

        elif cmd == "/calibrate":
            reply = "🔬 回測校準中，每支需 2-5 秒，請稍候…"
            _tg_send(token, src_chat or chat_id, reply)
            maybe_calibrate(state, force=True)
            changed = True
            cal = _calibration_weights(state)
            if cal:
                lines = ["✅ *校準完成*（訊號權重已依歷史勝率調整）\n"]
                for tk, mult in list(cal.items())[:15]:
                    parts = [f"{k}×{v}" for k, v in mult.items()]
                    lines.append(f"• *{tk}*: {', '.join(parts)}")
                reply = "\n".join(lines)
            else:
                reply = "⚠️ 校準無結果（資料不足或 backtest.py 缺失）"

        elif cmd == "/risk":
            th = state["thresholds"]
            if len(args) >= 2:
                try:
                    th["account_size"] = float(args[0])
                    th["risk_pct"] = float(args[1]) / 100 if float(args[1]) > 1 else float(args[1])
                    changed = True
                except ValueError:
                    pass
            reply = (
                "💰 *部位風險設定*\n\n"
                f"帳戶總值：${th.get('account_size', 100000):,.0f}\n"
                f"單筆風險：{th.get('risk_pct', 0.01):.1%}\n"
                f"ATR 停損倍數：{th.get('atr_mult', 1.5)}\n"
                f"部位提示：{'✅ 開' if th.get('position_sizing_enabled', True) else '❌ 關'}\n\n"
                "設定範例：`/risk 100000 1`（帳戶 $10萬、單筆風險 1%）\n"
                "或 `/set atr_mult 2`、`/set position_sizing_enabled off`"
            )

        elif cmd == "/protections":
            th = state["thresholds"]
            hist = state.get("signal_history", {})
            cd_h = float(th.get("cooldown_hours", 6))
            _now = datetime.now(timezone.utc)
            active = 0
            for cats in hist.values():
                for ts in cats.values():
                    try:
                        if (_now - datetime.fromisoformat(ts)).total_seconds() / 3600 < cd_h:
                            active += 1
                    except Exception:
                        pass
            reply = (
                "🛡 *防護機制狀態*\n\n"
                f"🔕 訊號冷卻：{'✅ 開' if th.get('cooldown_enabled', True) else '❌ 關'}"
                f"（{th.get('cooldown_hours', 6)} 小時內同類訊號不重發）\n"
                f"🌊 大盤濾網：{'✅ 開' if th.get('regime_filter_enabled', True) else '❌ 關'}\n"
                f"🕐 市場時段限制：{'✅ 開' if th.get('scan_market_only', True) else '❌ 關'}\n\n"
                f"目前追蹤 {active} 個冷卻中訊號\n\n"
                "調整：`/set cooldown_hours 4`、`/set cooldown_enabled off`、"
                "`/set regime_filter_enabled off`"
            )

        elif cmd in ("/fundamentals", "/f"):
            if not args:
                reply = "用法：`/fundamentals AAPL` 或 `/f AAPL`"
            else:
                tkr = args[0].upper()
                reply = f"🏢 查詢 {tkr} 基本面中…"
                _tg_send(token, src_chat or chat_id, reply)
                reply = _cmd_fundamentals(state, tkr)
                changed = True   # 可能更新快取

        elif cmd in ("/options", "/opt"):
            if not args:
                reply = "用法：`/options AAPL` 或 `/opt AAPL`"
            else:
                tkr = args[0].upper()
                _tg_send(token, src_chat or chat_id, f"🎭 查詢 {tkr} 選擇權情緒中…")
                reply = _cmd_options(tkr)

        elif cmd == "/whales":
            try:
                import whales_13f as wf
                whales = list(wf.WHALES.items())          # [(cik, name), ...]
                if not args or not args[0].isdigit():
                    lines = ["🐋 *超級投資人 13F*（`/whales 編號` 查詢季度增減倉）"]
                    lines += [f"{i+1}. {name}" for i, (_c, name) in enumerate(whales)]
                    reply = "\n".join(lines)
                else:
                    i = int(args[0]) - 1
                    if not (0 <= i < len(whales)):
                        reply = f"編號範圍 1-{len(whales)}"
                    else:
                        cik, name = whales[i]
                        _tg_send(token, src_chat or chat_id, f"🐋 抓取 {name} 的 13F 中…")
                        res = wf.fetch_whale(cik, name)
                        reply = (wf.format_whale_text(name, res, res.get("period", ""))
                                 + "\n_SEC 13F · 45天延遲 · 非投資建議_") if res else \
                            f"⚠️ 查無 {name} 的 13F（EDGAR 暫時無回應或無申報）"
            except Exception as e:
                reply = f"❌ 13F 查詢失敗：{e}"

        elif cmd in ("/insider", "/ins"):
            if not args:
                reply = "用法：`/insider AAPL` 或 `/ins AAPL`"
            else:
                tkr = args[0].upper()
                _tg_send(token, src_chat or chat_id, f"🕵️ 查詢 {tkr} 內部人交易中…")
                reply = _cmd_insider(tkr)

        elif cmd == "/briefing":
            reply = "☀️ 生成晨報中，約需 20-40 秒…"
            _tg_send(token, src_chat or chat_id, reply)
            reply = daily_briefing(state, force=True) or "晨報生成失敗"

        elif cmd in ("/today", "/plan"):
            th = state["thresholds"]
            acct = float(th.get("account_size", 100000))
            rk = float(th.get("risk_pct", 0.01))
            if args:
                try:
                    acct = float(args[0].replace(",", ""))
                except ValueError:
                    pass
            if len(args) >= 2:
                try:
                    rk = float(args[1]) / 100.0
                except ValueError:
                    pass
            _tg_send(token, src_chat or chat_id,
                     "🎯 產生當日交易計畫（VWAP/ORB/RVOL，約 30-60 秒）…")
            try:
                import trade_plan as _tpl
                wl = state["watchlist"][:10]
                if not wl:
                    reply = "觀察清單是空的——先 `/add AAPL NVDA`"
                else:
                    _tickets, _tp_src = _tpl.build_plans(wl, acct, rk,
                                                         calib=state.get("plan_calib"))
                    reply = (_tpl.plan_text(_tickets, acct, rk, _tp_src)
                             if _tickets else "抓不到盤中資料（休市或資料源異常）")
                    # 進場票記入決策計分板（source=day_plan，隔日結算）
                    try:
                        import reflection as rfl
                        _today_s = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                        for _t in _tickets or []:
                            if _t["action"] in ("買進", "小量試單"):
                                if rfl.record_pick(state, _t["ticker"],
                                                   round(_t["confidence"] / 5.0, 2),
                                                   _t["last"], _today_s,
                                                   source="day_plan", horizon=1):
                                    changed = True
                    except Exception:
                        pass
            except Exception as e:
                reply = f"❌ 計畫產生失敗：{e}"

        elif cmd in ("/committee", "/cmt"):
            if not args:
                reply = "用法：`/committee NVDA`（約 1-3 分鐘、9 次 LLM 呼叫；台股用 2330.TW）"
            else:
                _tg_send(token, src_chat or chat_id,
                         f"🏛 召開 {args[0].upper()} 投資決策會議（約 1-3 分鐘）…")
                reply, _cmt_changed = _cmd_committee(state, args[0])
                if _cmt_changed:
                    changed = True

        elif cmd == "/weekly":
            reply = "📒 生成週報中，約需 30-60 秒…"
            _tg_send(token, src_chat or chat_id, reply)
            try:
                reply = weekly_report(state) or "週報生成失敗"
            except Exception as e:
                reply = f"❌ 週報生成失敗：{e}"

        elif cmd == "/earnings":
            days = int(args[0]) if args and args[0].isdigit() else 14
            reply = "📅 查詢近期財報中…"
            _tg_send(token, src_chat or chat_id, reply)
            earn = _upcoming_earnings(state, max_days=days)
            changed = True   # 更新快取
            if earn:
                lines = [f"📅 *{days} 天內財報*\n"]
                for tk, ed, d in earn:
                    when = "今天" if d == 0 else ("明天" if d == 1 else f"{d} 天後")
                    lines.append(f"• *{tk}* — {ed.isoformat()}（{when}）")
                reply = "\n".join(lines)
            else:
                reply = f"📅 觀察清單 {days} 天內無財報公布"

        elif cmd == "/positions":
            reply = "📊 查詢持倉中…"
            _tg_send(token, src_chat or chat_id, reply)
            reply = _cmd_positions()

        elif cmd == "/pnl":
            reply = _cmd_pnl()

        elif cmd == "/autotrade":
            th = state["thresholds"]
            if args and args[0].lower() in ("on", "off"):
                th["autotrade_enabled"] = (args[0].lower() == "on")
                changed = True
            on = th.get("autotrade_enabled", False)
            key, secret = _alpaca_keys()
            key_ok = "✅" if (key and secret) else "❌ 未設 key"
            reply = (
                f"🤖 *自動交易*：{'✅ 開啟' if on else '⏸ 關閉'}\n"
                f"Alpaca key：{key_ok}\n"
                f"買進門檻 ≥{th.get('at_buy_threshold',0.5)}　出場 ≤{th.get('at_exit_threshold',-0.2)}\n"
                f"最多 {int(th.get('at_max_positions',10))} 檔，每檔 ≤{th.get('at_max_position_pct',0.15):.0%}\n\n"
                "⚠️ 開啟後僅在美股開盤時，依掃描評分自動下*模擬*單\n"
                "`/autotrade on`｜`/autotrade off`｜`/positions`｜`/pnl`｜`/closeall`"
            )

        elif cmd == "/closeall":
            key, secret = _alpaca_keys()
            if not key or not secret:
                reply = "⚠️ 未設定 Alpaca key"
            else:
                try:
                    import alpaca_trader as at
                    ok, msg = at.close_all(key, secret)
                    reply = "✅ 已送出全部平倉" if ok else f"❌ 平倉失敗：{msg}"
                except Exception as e:
                    reply = f"❌ {e}"

        elif cmd == "/journal":
            n = max(1, int(args[0])) if args and args[0].isdigit() else 10  # 避免 /journal 0 顯示全部
            try:
                import alpaca_trader as at
                log = at.load_journal(JOURNAL_FILE)
            except Exception:
                log = []
            if not log:
                reply = "📒 尚無自動交易紀錄（開 /autotrade on 後才會記錄）"
            else:
                lines = [f"📒 *交易日誌*（最近 {min(n, len(log))} 筆）\n"]
                for e in log[-n:][::-1]:
                    sc = e.get("score")
                    sc_s = f"{sc:+.2f}" if isinstance(sc, (int, float)) else "—"
                    icon = "✅" if e.get("submitted") else "❌"
                    t = (e.get("time") or "")[:16].replace("T", " ")
                    lines.append(f"{icon} {t}　{e.get('side','').upper()} {e.get('symbol')} "
                                 f"x{e.get('qty')}　評分 {sc_s}")
                reply = "\n".join(lines)

        elif cmd == "/plantest":
            sub = args[0].lower() if args else ""
            if sub == "clear":
                had = state.pop("plan_calib", None)
                changed = bool(had)
                reply = "🧹 已清除當日計畫校準與參數（/today 回到原始規則）" if had \
                        else "目前沒有已套用的校準"
            elif sub == "opt":
                wl = state["watchlist"][:8]
                do_apply = len(args) >= 2 and args[1].lower() == "apply"
                if not wl:
                    reply = "觀察清單是空的——先 `/add AAPL NVDA`"
                else:
                    _tg_send(token, src_chat or chat_id,
                             f"🔧 參數尋優（{len(wl)} 檔 × 27 組參數 × ~60 日，約 2-4 分鐘）…")
                    try:
                        # 長操作前先落盤 last_update_id：萬一 runner 超時被殺，
                        # 這則指令不會每 15 分鐘被重新處理（毒訊息迴圈）
                        save_state(state)
                        import plan_backtest as pbt
                        opt = pbt.optimize(wl)
                        reply = pbt.opt_text(opt)
                        if do_apply:
                            rec = opt.get("recommend")
                            if rec:
                                cal_o = pbt.walk_forward_calibrate(rec["trades"])
                                cal_o["as_of"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                                cal_o["params"] = rec["params"]
                                state["plan_calib"] = cal_o
                                changed = True
                                reply += ("\n\n✅ 已套用推薦參數＋對應型態校準"
                                          "（`/plantest clear` 還原）")
                            else:
                                reply += "\n\n（無明確勝過預設的組合，未套用任何變更）"
                    except Exception as e:
                        reply = f"❌ 尋優失敗：{e}"
            else:
                wl = state["watchlist"][:12]
                if not wl:
                    reply = "觀察清單是空的——先 `/add AAPL NVDA`"
                else:
                    _tg_send(token, src_chat or chat_id,
                             f"📜 回測當日計畫（{len(wl)} 檔 × ~60 交易日，約 1-3 分鐘）…")
                    try:
                        save_state(state)      # 防超時毒訊息迴圈（同 opt）
                        import plan_backtest as pbt
                        agg_, cal_, _tr = pbt.run(wl)
                        reply = pbt.stats_text(agg_, cal_, n_tickers=len(wl))
                        if sub == "apply":
                            if cal_.get("setups"):
                                cal_["as_of"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                                state["plan_calib"] = cal_
                                changed = True
                                reply += ("\n\n✅ 校準已套用——之後的 /today 會吃這份實證"
                                          "（`/plantest clear` 還原）")
                            else:
                                reply += "\n\n（樣本不足，未產生可套用的校準）"
                        else:
                            reply += "\n\n（僅報告；`/plantest apply` 套用、`/plantest clear` 還原）"
                    except Exception as e:
                        reply = f"❌ 回測失敗：{e}"

        elif cmd == "/rebalance":
            key, secret = _alpaca_keys()
            import rebalance as rbl
            sch = args[0].lower() if args else "hrp"
            if not key or not secret:
                reply = "⚠️ 未設定 Alpaca key（此指令以 Alpaca 模擬持倉為基準）"
            elif sch not in rbl.SCHEMES:
                reply = ("用法：`/rebalance [配置法]`\n" +
                         "\n".join(f"`{k}` — {v}" for k, v in rbl.SCHEMES.items()))
            else:
                _tg_send(token, src_chat or chat_id,
                         f"⚖️ 計算再平衡中（目標：{rbl.SCHEMES[sch]}）…")
                try:
                    import alpaca_trader as at
                    pos = at.get_positions(key, secret)
                    qty = {s: p["qty"] for s, p in pos.items() if p["qty"] > 0}  # 空頭不進權重
                    if len(qty) < 2:
                        reply = "⚠️ Alpaca 模擬多頭持倉不足 2 檔，無法做配置優化（先 /autotrade on 建倉）"
                    else:
                        px = {s: pos[s]["market_value"] / pos[s]["qty"] for s in qty}
                        raw = yf.download(list(qty), period="1y",
                                          auto_adjust=True, progress=False)
                        close = (raw["Close"] if isinstance(raw.columns, pd.MultiIndex)
                                 else raw[["Close"]])
                        rets = close.pct_change().dropna(how="all").dropna(axis=1)
                        tw = rbl.target_weights(rets, sch)
                        if tw is None:
                            reply = "❌ 目標權重計算失敗（歷史數據不足：需 ≥2 檔、≥40 交易日）"
                        else:
                            no_hist = [s for s in qty if s not in rets.columns]
                            res = rbl.rebalance_orders(qty, px, tw.to_dict(),
                                                       no_data=no_hist)
                            reply = rbl.rebalance_text(res, rbl.SCHEMES[sch])
                except Exception as e:
                    reply = f"❌ 再平衡失敗：{e}"

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
    u, l = float(upper.iloc[-1]), float(lower.iloc[-1])
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

    prev_ma20 = float(close.rolling(20).mean().iloc[-2])
    prev_ma50 = float(close.rolling(50).mean().iloc[-2])

    if prev_ma20 < prev_ma50 and ma20 > ma50:
        return {"signal": "golden_cross", "label": "MA20/50 黃金交叉"}
    if prev_ma20 > prev_ma50 and ma20 < ma50:
        return {"signal": "death_cross",  "label": "MA20/50 死亡交叉"}
    return {"signal": "neutral"}


# ── Composite scoring ─────────────────────────────────────────────────────────

def _weekly_trend(close: pd.Series) -> int:
    """
    把日線 resample 成週線，回傳週線偏向 -2~+2：
      週價 > 週MA10 +1 / 否則 -1；週MACD histogram > 0 +1 / 否則 -1。
    資料不足或非時間索引回 0（中性，不影響）。
    """
    if not isinstance(close.index, pd.DatetimeIndex):
        return 0
    try:
        wk = close.resample("W").last().dropna()
        if len(wk) < 12:
            return 0
        price = float(wk.iloc[-1])
        ma10 = float(wk.rolling(10).mean().iloc[-1])
        ema12 = wk.ewm(span=12, adjust=False).mean()
        ema26 = wk.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        hist = float((macd_line - macd_line.ewm(span=9, adjust=False).mean()).iloc[-1])
        bias = (1 if price > ma10 else -1) + (1 if hist > 0 else -1)
        return bias
    except Exception:
        return 0


def _composite_score(close: pd.Series, high: pd.Series | None,
                     low: pd.Series | None, volume: pd.Series | None,
                     edge_weights: dict | None = None, mtf: bool = False) -> dict:
    """
    Blend every indicator into a single -1 (極空) .. +1 (極多) score.
    Returns {"score", "rating", "emoji", "components"}.

    edge_weights: 各元件的歷史勝率乘數（來自回測校準），如 {"macd":1.4,"rsi":0.7}。
                  有提供時會放大歷史表現好的元件、縮小表現差的，並重新正規化。

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

    # ── 加權合成（可被回測校準的 edge_weights 調整）──────────────
    base_w = {"trend": 0.35, "macd": 0.25, "momentum": 0.20,
              "rsi": 0.10, "bollinger": 0.10}
    if edge_weights:
        adj = {k: base_w[k] * float(edge_weights.get(k, 1.0)) for k in base_w}
        tot = sum(adj.values()) or 1.0
        w = {k: adj[k] / tot for k in adj}   # 重新正規化使總和=1，分數仍 -1~+1
    else:
        w = base_w
    score = sum(w[k] * comps[k] for k in comps)

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

    # ── 多時間框架確認（軟性調整：日線分數 vs 週線偏向）────────────
    mtf_note = None
    if mtf:
        wt = _weekly_trend(close)            # -2~+2
        if score > 0.1 and wt >= 1:
            score *= 1.1; mtf_note = "✅ 週線同向"
        elif score < -0.1 and wt <= -1:
            score *= 1.1; mtf_note = "✅ 週線同向(偏空)"
        elif score > 0.1 and wt <= -1:
            score *= 0.8; mtf_note = "⚠️ 週線背離"
        elif score < -0.1 and wt >= 1:
            score *= 0.8; mtf_note = "⚠️ 週線背離"
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

    return {"score": round(score, 3), "rating": rating, "emoji": emoji,
            "components": comps, "mtf_note": mtf_note}


# ── 回測校準：把歷史勝率回饋成元件權重 ────────────────────────────────────────

# 回測規則 → 評分元件 的對應
_RULE_TO_COMPONENT = {
    "MA20/50 黃金交叉":          "trend",
    "黃金交叉+站上200MA":        "trend",
    "⭐三層確認(MACD+RSI+趨勢)":  "trend",
    "MACD 金叉":                 "macd",
    "MACD 死叉(空)":             "macd",
    "RSI<30 超賣反彈":           "rsi",
    "RSI>70 超買回落(空)":       "rsi",
    "布林下軌反彈":              "bollinger",
    "布林上軌突破":              "bollinger",
}


def calibrate_ticker(df) -> dict:
    """
    對單一標的跑回測，把各規則的 edge 分數聚合成「元件權重乘數」。
    回傳如 {"macd":1.35,"rsi":0.72,...}，乘數範圍約 0.5~1.5。
    勝率高的元件 >1（加重），表現差的 <1（縮小）。
    """
    try:
        import backtest as bt
        edges = bt.rule_edge_scores(df)   # {rule: edge(-1~+1)}
    except Exception as e:
        print(f"  calibrate 失敗（backtest 不可用）：{e}")
        return {}

    comp_sum: dict[str, float] = {}
    comp_cnt: dict[str, int] = {}
    for rule, edge in edges.items():
        comp = _RULE_TO_COMPONENT.get(rule)
        if comp is None:
            continue
        comp_sum[comp] = comp_sum.get(comp, 0.0) + edge
        comp_cnt[comp] = comp_cnt.get(comp, 0) + 1

    mult = {}
    for comp, total in comp_sum.items():
        avg = total / comp_cnt[comp]
        # edge -0.5~+0.5 → 乘數 0.5~1.5
        mult[comp] = round(1 + max(-0.5, min(0.5, avg)), 3)
    return mult


def calibrate(tickers: list[str], period: str = "2y") -> dict:
    """
    對清單每支標的跑回測校準，回傳 {ticker: {component: multiplier}}。
    這是較重的操作（每支下載 2 年資料），建議每天/每週跑一次，不要每次掃描都跑。
    """
    print(f"校準 {len(tickers)} 支標的的訊號權重（回測 {period}）…")
    import backtest as bt
    result = {}
    for tk in tickers:
        try:
            raw = yf.download(tk, period=period, auto_adjust=True, progress=False)
            if raw.empty or len(raw) < 60:
                continue
            # MultiIndex 下 `"Close" in raw.columns` 是部分鍵比對、恆為 True，
            # 舊寫法會讓 MultiIndex 直接漏過去 → 校準悄悄算出垃圾。統一用 normalize_ohlc。
            df = bt.normalize_ohlc(raw, tk)
            mult = calibrate_ticker(df)
            if mult:
                result[tk] = mult
                print(f"  {tk}: {mult}")
        except Exception as e:
            print(f"  {tk}: 校準錯誤 {e}")
    return result


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


def _atr_value(close, high, low, period: int = 14) -> float:
    """ATR(14) 數值（H/L 缺失時退回用收盤近似）。"""
    if high is None or low is None:
        high = close
        low = close
    tr = pd.concat([high - low,
                    (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    return float(atr) if not pd.isna(atr) else 0.0


def _position_hint(close, high, low, price: float, thresholds: dict) -> dict | None:
    """ATR 風險基準的建議部位（共用 quant_tools，與 dashboard 一致）。"""
    try:
        import quant_tools as qt
    except Exception:
        return None
    atr = _atr_value(close, high, low)
    if atr <= 0 or price <= 0:
        return None
    acct = float(thresholds.get("account_size", 100000))
    risk = float(thresholds.get("risk_pct", 0.01))
    mult = float(thresholds.get("atr_mult", 1.5))
    ps = qt.atr_position_size(acct, risk, price, atr, mult)
    ann_vol = float(close.pct_change().dropna().std() * (252 ** 0.5)) if len(close) > 5 else 0.0
    return {
        "shares":   ps["shares"],
        "pct":      ps["pct_of_account"],
        "stop":     ps["stop_price"],
        "ann_vol":  round(ann_vol, 3),
    }


def scan(tickers: list[str], thresholds: dict, calibration: dict | None = None) -> list[dict]:
    rsi_lo  = thresholds.get("rsi_oversold",    35)
    rsi_hi  = thresholds.get("rsi_overbought",  68)
    chg_th  = thresholds.get("price_change_pct", 3.0)
    macd_on = thresholds.get("macd_enabled",  True)
    bb_on   = thresholds.get("bb_enabled",    True)
    atr_on  = thresholds.get("atr_enabled",   True)
    vol_r   = thresholds.get("vol_spike_ratio", 2.0)
    mtf_on  = thresholds.get("mtf_enabled",   True)

    # 1y：週線指標（MACD 26週）需足夠歷史
    print(f"Batch-downloading {len(tickers)} tickers (1y)…")
    try:
        raw = yf.download(tickers, period="1y", auto_adjust=True,
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
                def _ser(field):
                    s = raw.get(field, pd.Series()).squeeze().dropna()
                    return s if not s.empty else None
                close  = raw["Close"].squeeze().dropna()
                high   = _ser("High")
                low    = _ser("Low")
                volume = _ser("Volume")
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

            # ── Composite score (calibration-weighted + MTF 確認) ──
            edge_w = (calibration or {}).get(ticker)
            cs = _composite_score(close, high, low, volume,
                                  edge_weights=edge_w, mtf=mtf_on)

            # ── Position sizing hint (ATR risk-based) ─────────────
            pos = None
            if thresholds.get("position_sizing_enabled", True):
                pos = _position_hint(close, high, low, price, thresholds)

            results.append({
                "ticker":  ticker,
                "price":   price,
                "rsi":     rsi,
                "chg":     chg,
                "score":   cs["score"],
                "rating":  cs["rating"],
                "emoji":   cs["emoji"],
                "mtf_note": cs.get("mtf_note"),
                "position": pos,
                "signals": [s for s in signals if s],
            })
            flag = "🚨" if signals else "  "
            print(f"{flag} {ticker}: ${price}  RSI={rsi}  chg={chg:+.1f}%  "
                  f"score={cs['score']:+.2f}({cs['rating']})  signals={len(signals)}")

        except Exception as exc:
            print(f"  {ticker}: error – {exc}")

    return results


# ── Protections (freqtrade-style) ─────────────────────────────────────────────

def _signal_category(label: str) -> str:
    """把訊號文字歸類，供冷卻去重使用。"""
    if "RSI" in label:
        return "rsi"
    if "MACD" in label:
        return "macd"
    if "黃金交叉" in label or "死亡交叉" in label:
        return "ma_cross"
    if "BB" in label or "布林" in label:
        return "bollinger"
    if "ATR" in label:
        return "atr"
    if "爆量" in label or "波動率" in label:
        return "volume"
    if "暴漲" in label or "暴跌" in label:
        return "price"
    return "other"


def apply_cooldown(results: list[dict], state: dict, now: datetime | None = None) -> tuple[list[dict], int]:
    """
    CooldownPeriod：同標的同類訊號在 cooldown_hours 內只發一次，防洗版。
    只在「實際保留（會發送）」的訊號上更新時間戳；被壓制者不更新。
    回傳 (過濾後的 results, 被壓制的訊號數)。會就地更新 state["signal_history"]。
    """
    now = now or datetime.now(timezone.utc)
    cd_hours = float(state["thresholds"].get("cooldown_hours", 6))
    hist = state.setdefault("signal_history", {})
    suppressed = 0

    for r in results:
        tk_hist = hist.setdefault(r["ticker"], {})
        kept = []
        newly_sent: dict[str, str] = {}   # 本次掃描新發送的類別 → 時間戳
        for label in r.get("signals", []):
            cat = _signal_category(label)
            # 只根據「先前掃描」留下的歷史判斷冷卻，避免同次掃描同類互相壓制
            last = tk_hist.get(cat)
            if last:
                try:
                    age_h = (now - datetime.fromisoformat(last)).total_seconds() / 3600
                    if age_h < cd_hours:
                        suppressed += 1
                        continue  # 冷卻中，壓制（不更新時間戳）
                except Exception:
                    pass
            kept.append(label)
            newly_sent[cat] = now.isoformat()
        tk_hist.update(newly_sent)   # 本次掃描結束後才寫回歷史
        r["signals"] = kept

    # 清理 >7 天舊紀錄，避免 state 膨脹
    cutoff = now - timedelta(days=7)
    for tk in list(hist.keys()):
        for cat in list(hist[tk].keys()):
            try:
                if datetime.fromisoformat(hist[tk][cat]) < cutoff:
                    del hist[tk][cat]
            except Exception:
                del hist[tk][cat]
        if not hist[tk]:
            del hist[tk]

    return results, suppressed


def market_regime() -> dict | None:
    """
    MaxDrawdown-style 大盤風險濾網：用 SPY 相對 MA50 與近月動量判斷 risk-on/off。
    回傳 {"regime","emoji","label"} 或 None（資料不足）。
    """
    try:
        raw = yf.download("SPY", period="3mo", auto_adjust=True, progress=False)
        if raw.empty:
            return None
        close = raw["Close"].squeeze().dropna()
        if len(close) < 50:
            return None
        price = float(close.iloc[-1])
        ma50 = float(close.rolling(50).mean().iloc[-1])
        ret_1m = price / float(close.iloc[-22]) - 1 if len(close) >= 22 else 0.0
        if price > ma50 and ret_1m > 0:
            return {"regime": "risk_on", "emoji": "🟢", "label": "偏多（大盤站上 MA50）"}
        if price < ma50 and ret_1m < -0.03:
            return {"regime": "risk_off", "emoji": "🔴", "label": "偏空（大盤跌破 MA50 且弱勢）"}
        return {"regime": "neutral", "emoji": "🟡", "label": "中性震盪"}
    except Exception:
        return None


# ── Daily AI briefing ─────────────────────────────────────────────────────────

def _llm_complete(prompt: str, max_tokens: int = 900) -> str | None:
    """
    經 requests 呼叫 LLM（cron 環境無 openai SDK，故用 requests）。
    自動辨識 OpenAI / Anthropic 相容端點。需環境變數 LLM_API_KEY；
    可選 LLM_BASE_URL、LLM_MODEL。失敗回 None（晨報退回純數據版）。
    """
    key = os.environ.get("LLM_API_KEY", "")
    # 清掉貼上金鑰時混入的隱形/全形字元（同 app.py _clean_secret；HTTP header 只吃 ASCII）
    for _ch in ("​", "‌", "‍", "﻿", " ", "　"):
        key = key.replace(_ch, "")
    key = key.strip()
    if not key:
        return None
    base = os.environ.get("LLM_BASE_URL", "").strip().rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3].rstrip("/")   # 容忍帶 /v1 的 base（app.py 的 SDK 慣例），這裡自己補路徑
    model = os.environ.get("LLM_MODEL", "")
    is_anthropic = ("anthropic" in base.lower()) or key.startswith("sk-ant") or model.startswith("claude")
    if not model:
        model = "claude-3-5-haiku-20241022" if is_anthropic else "gpt-4o-mini"
    try:
        if is_anthropic:
            url = (base or "https://api.anthropic.com") + "/v1/messages"
            r = requests.post(url, timeout=45,
                              headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                                       "content-type": "application/json"},
                              json={"model": model, "max_tokens": max_tokens,
                                    "messages": [{"role": "user", "content": prompt}]})
            if r.ok:
                return r.json()["content"][0]["text"]
        else:
            url = ((base + "/v1") if base else "https://api.openai.com/v1") + "/chat/completions"
            r = requests.post(url, timeout=45,
                              headers={"Authorization": f"Bearer {key}"},
                              json={"model": model, "max_tokens": max_tokens, "temperature": 0.3,
                                    "messages": [{"role": "user", "content": prompt}]})
            if r.ok:
                return r.json()["choices"][0]["message"]["content"]
        print(f"LLM error {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f"LLM exception: {e}")
    return None


def _index_snapshot() -> dict:
    """抓主要指數的最新漲跌（晨報用）。回 {名稱: (價, 日變動)}。"""
    idx_map = {"^GSPC": "S&P500", "^IXIC": "NASDAQ", "^DJI": "道瓊", "^VIX": "VIX"}
    out = {}
    for tk, name in idx_map.items():
        try:
            raw = yf.download(tk, period="5d", auto_adjust=True, progress=False)
            if raw.empty:
                continue
            s = raw["Close"].squeeze().dropna()
            if len(s) >= 2:
                out[name] = (float(s.iloc[-1]), float(s.iloc[-1] / s.iloc[-2] - 1))
        except Exception:
            continue
    return out


def _upcoming_earnings(state: dict, max_days: int | None = None) -> list:
    """
    回傳 watchlist 中 max_days 天內要公布財報的標的：[(ticker, date, days_until), ...]。
    每日快取（earnings_cache）避免重複慢呼叫。就地更新 state。
    """
    import datetime as _dt
    if max_days is None:
        max_days = int(state["thresholds"].get("earnings_alert_days", 5))
    today = _dt.date.today()
    today_s = today.isoformat()
    cache = state.setdefault("earnings_cache", {})
    try:
        import fundamentals as fa
    except Exception:
        return []
    out = []
    for tk in state.get("watchlist", []):
        c = cache.get(tk)
        if c and c.get("checked") == today_s:
            ed_s = c.get("earnings")
        else:
            ed = fa.next_earnings_date(tk)
            ed_s = ed.isoformat() if ed else None
            cache[tk] = {"checked": today_s, "earnings": ed_s}
        if ed_s:
            try:
                ed = _dt.date.fromisoformat(ed_s)
                days = (ed - today).days
                if 0 <= days <= max_days:
                    out.append((tk, ed, days))
            except Exception:
                continue
    out.sort(key=lambda x: x[2])
    return out


def daily_briefing(state: dict, force: bool = False) -> str | None:
    """組裝每日晨報（大盤 + 指數 + 觀察清單排名 + 訊號 + AI 解讀）。"""
    now_et = datetime.now(ET)
    date_str = now_et.strftime("%Y-%m-%d (%a)")

    regime = market_regime()
    indices = _index_snapshot()
    results = scan(state["watchlist"], state["thresholds"],
                   calibration=_calibration_weights(state)) if state["watchlist"] else []
    ranked = sorted(results, key=lambda r: -r.get("score", 0))
    top = ranked[:5]
    bottom = [r for r in ranked if r.get("score", 0) < -0.1][-3:]
    flagged = [r for r in results if r.get("signals")]

    # 結構化數據（給 AI 與純數據版共用）
    lines = [f"☀️ *RBS 每日晨報* — {date_str}"]
    if regime:
        lines.append(f"{regime['emoji']} 大盤風險：{regime['label']}")
    if indices:
        idx_str = "　".join(f"{n} {c:+.2%}" for n, (p, c) in indices.items())
        lines.append(f"📈 {idx_str}")
    lines.append("")

    # AI 解讀（可選；force=手動 /briefing 即使關閉也產生）
    if force or state["thresholds"].get("briefing_enabled", True):
        ctx = [f"日期：{date_str}"]
        if regime:
            ctx.append(f"大盤風險：{regime['label']}")
        if indices:
            ctx.append("指數：" + ", ".join(f"{n} {c:+.2%}" for n, (p, c) in indices.items()))
        if top:
            ctx.append("觀察清單最強：" + ", ".join(
                f"{r['ticker']}({r['score']:+.2f})" for r in top))
        if bottom:
            ctx.append("最弱：" + ", ".join(
                f"{r['ticker']}({r['score']:+.2f})" for r in bottom))
        if flagged:
            ctx.append("觸發訊號：" + ", ".join(
                f"{r['ticker']}" for r in flagged[:8]))
        prompt = (
            "你是專業晨間市場分析師。僅根據以下數據，用繁體中文寫一段 100-150 字的"
            "開盤前晨報，點出今天該留意什麼、整體情緒、風險。不得編造未提供的事實。\n\n"
            + "\n".join(ctx)
        )
        ai = _llm_complete(prompt)
        if ai:
            lines.append(ai.strip())
            lines.append("")

    # 排名（純數據，永遠附上）
    if top:
        lines.append("🏆 *觀察清單最強 Top 5*")
        for r in top:
            lines.append(f"   {r.get('emoji','')} {r['ticker']}  {r['score']:+.2f} ({r.get('rating','')})")
        lines.append("")
    if bottom:
        lines.append("⚠️ *最弱*")
        for r in reversed(bottom):
            lines.append(f"   {r.get('emoji','')} {r['ticker']}  {r['score']:+.2f} ({r.get('rating','')})")
        lines.append("")
    if flagged:
        lines.append(f"🚨 今日 {len(flagged)} 檔觸發訊號（詳見後續掃描）")

    # 內部人交易亮點：只查「最強且偏多的美股」1 檔（每日 1 次呼叫、全程防呆）
    try:
        cand = next((r for r in top
                     if r.get("score", 0) >= 0.3 and "." not in r.get("ticker", "")), None)
        if cand:
            import sec_insider as _si
            ins = _si.fetch_insider(cand["ticker"], max_filings=8)  # 限縮抓取數，控制晨報延遲
            if ins and ins.get("n_buys") and (ins.get("cluster_buy")
                                              or (ins.get("net_value") or 0) > 0):
                net = ins.get("net_value")
                lines.append("")
                lines.append(f"🕵️ *內部人*：{cand['ticker']} {ins.get('label','')}"
                             + (f"（淨 {_si._money(net)}）" if net else ""))
    except Exception:
        pass

    # AI 判斷回顧（反思記憶）
    try:
        import reflection as rfl
        s_ref = rfl.summary_text(state)
        if s_ref:
            lines.append("")
            lines.append(f"🪞 {s_ref}")
    except Exception:
        pass

    # 本週重要總經數據發布日（FRED 行事曆；CPI/非農/GDP/PCE…）
    try:
        _fred_k = os.environ.get("FRED_API_KEY", "")
        if _fred_k:
            import macro as _mc
            _rel = _mc.fetch_release_calendar(_fred_k, days_ahead=7)
            if _rel:
                lines.append("")
                lines.append("🗓 *本週總經*：" + "、".join(f"{n}({d[5:]})" for d, n in _rel[:5]))
    except Exception:
        pass

    # 近期財報提醒
    try:
        earn = _upcoming_earnings(state)
        if earn:
            lines.append("")
            lines.append("📅 *近期財報*")
            for tk, ed, days in earn:
                when = "今天" if days == 0 else ("明天" if days == 1 else f"{days} 天後")
                lines.append(f"   • {tk} — {ed.isoformat()}（{when}）")
    except Exception:
        pass

    if not state["watchlist"]:
        lines.append("_觀察清單為空，用 /add 新增標的_")
    lines.append("_每日晨報 · /set briefing_enabled off 可關閉_")
    return "\n".join(lines)


def weekly_report(state: dict) -> str:
    """
    每週深度週報：指數週漲跌 + 觀察清單強弱 + 決策計分板 + RRG 板塊輪動
    + 下週財報/總經行事曆。組件全走既有模組；任何區塊失敗都跳過不擋整報。
    """
    from sector_scan import _batch_closes
    now_et = datetime.now(ET)
    lines = [f"📒 *RBS 每週深度週報* — {now_et.strftime('%Y-%m-%d')}"]

    # 指數本週表現
    try:
        idx_map = {"^GSPC": "S&P500", "^IXIC": "NASDAQ", "^DJI": "道瓊", "^VIX": "VIX"}
        closes = _batch_closes(list(idx_map), "5d", min_len=2)
        segs = []
        for tk, name in idx_map.items():
            s = closes.get(tk)
            if s is not None and len(s) >= 2:
                segs.append(f"{name} {float(s.iloc[-1] / s.iloc[0] - 1):+.1%}")
        if segs:
            lines.append("📈 本週：" + "　".join(segs))
    except Exception:
        pass

    # 觀察清單本週最強/最弱（綜合評分）
    try:
        results = scan(state["watchlist"], state["thresholds"],
                       calibration=_calibration_weights(state))
        ranked = sorted(results, key=lambda r: -r.get("score", 0))
        if len(ranked) >= 6:
            top = "、".join(f"{r['ticker']}({r['score']:+.2f})" for r in ranked[:3])
            bot = "、".join(f"{r['ticker']}({r['score']:+.2f})" for r in ranked[-3:][::-1])
            lines.append(f"🏆 最強：{top}\n🥶 最弱：{bot}")
        elif ranked:                              # 清單太短就列全表，避免最強/最弱重疊
            lines.append("📊 評分：" + "、".join(
                f"{r['ticker']}({r['score']:+.2f})" for r in ranked))
    except Exception:
        pass

    # 決策計分板（反思記憶＋委員會紀錄——app 會把 committee_log.json commit 進 repo，
    # cron 的 checkout 讀得到）
    try:
        import json as _json_w
        from pathlib import Path as _P_w

        import reflection as rfl
        hist_w = list(state.get("reflections", {}).get("history", []))
        try:
            _clf = _P_w("committee_log.json")
            if _clf.exists():
                hist_w += (_json_w.loads(_clf.read_text(encoding="utf-8"))
                           .get("reflections", {}).get("history", []))
        except Exception:
            pass
        sb = rfl.scoreboard(hist_w)
        for r_ in sb:
            if r_["hit_rate"] is not None:
                lines.append(f"🎯 {rfl.SOURCE_LABELS.get(r_['source'], r_['source'])}"
                             f"近 {r_['n']} 次命中率 {r_['hit_rate']:.0%}"
                             + (f"、平均對齊報酬 {r_['avg_fwd']:+.1%}"
                                if r_["avg_fwd"] is not None else ""))
    except Exception:
        pass

    # RRG 板塊輪動（週線象限）
    try:
        from sector_scan import rrg_metrics
        _SEC = {"XLK": "科技", "XLF": "金融", "XLE": "能源", "XLV": "醫療",
                "XLY": "非必需", "XLP": "必需", "XLI": "工業", "XLU": "公用",
                "XLRE": "房產", "XLB": "原物料", "XLC": "通訊"}
        closes = _batch_closes(list(_SEC) + ["SPY"], "1y", min_len=60)
        bench = closes.get("SPY")
        quad: dict = {}
        if bench is not None:
            for etf, nm in _SEC.items():
                m = rrg_metrics(closes.get(etf), bench)
                if m:
                    quad.setdefault(m["quadrant"], []).append(nm)
        segs = [f"{q}：{'、'.join(v)}" for q, v in quad.items() if v]
        if segs:
            lines.append("🔄 *板塊輪動（RRG vs SPY）*\n" + "\n".join(f"　{s}" for s in segs))
    except Exception:
        pass

    # 下週財報 + 總經
    try:
        earn = _upcoming_earnings(state, max_days=7)
        if earn:
            lines.append("📅 下週財報：" + "、".join(
                f"{tk}({ed.strftime('%m/%d')})" for tk, ed, _dd in earn[:6]))
    except Exception:
        pass
    try:
        _fk = os.environ.get("FRED_API_KEY", "")
        if _fk:
            import macro as _mc
            rel = _mc.fetch_release_calendar(_fk, days_ahead=7)
            if rel:
                lines.append("🗓 下週總經：" + "、".join(f"{n}({d[5:]})" for d, n in rel[:6]))
    except Exception:
        pass

    lines.append("_每週日推送 · /weekly 隨時手動 · /set weekly_enabled off 關閉 · 非投資建議_")
    return "\n\n".join(lines)


def _should_send_weekly(state: dict) -> bool:
    """週日 ET 18:00 後、本週未發過、未靜音、開關開啟。"""
    th = state["thresholds"]
    if not th.get("weekly_enabled", True) or _is_muted(state):
        return False
    now_et = datetime.now(ET)
    if now_et.weekday() != 6 or now_et.hour < 18:      # 週日=6
        return False
    week_id = f"{now_et.isocalendar().year}-W{now_et.isocalendar().week}"
    return state.get("last_weekly") != week_id


def _should_send_briefing(state: dict) -> bool:
    """判斷現在是否該發晨報：啟用 + 交易日 + 過了設定時間 + 今天還沒發 + 未靜音。"""
    th = state["thresholds"]
    if not th.get("briefing_enabled", True):
        return False
    if _is_muted(state):
        return False
    ms = market_status()
    # 週末/假日不發（market_status 的 reason 會標明；用 open 與 reason 判斷）
    now_et = datetime.now(ET)
    if now_et.weekday() >= 5:
        return False
    if "假日" in ms.get("reason", ""):
        return False
    # 過了設定時間（ET 小數時）
    now_dec = now_et.hour + now_et.minute / 60
    if now_dec < float(th.get("briefing_hour_et", 8.5)):
        return False
    # 今天還沒發
    today_et = now_et.strftime("%Y-%m-%d")
    if state.get("last_briefing_date") == today_et:
        return False
    return True


# ── Message builder ──────────────────────────────────────────────────────────

def _build_message(results: list[dict], timestamp: str,
                   regime: dict | None = None, suppressed: int = 0) -> str | None:
    flagged = [r for r in results if r.get("signals")]
    if not flagged:
        return None

    # Sort flagged signals by absolute conviction (strongest first)
    flagged_sorted = sorted(flagged, key=lambda r: -abs(r.get("score", 0)))

    lines = [f"🚨 *RBS 自動訊號掃描* — {timestamp}"]
    if regime:
        lines.append(f"{regime['emoji']} 大盤風險：{regime['label']}")
    lines.append("")
    for r in flagged_sorted:
        arrow = "🟢" if r["chg"] > 0 else ("🔴" if r["chg"] < 0 else "⚪")
        score = r.get("score", 0)
        rating = r.get("rating", "")
        mtf = r.get("mtf_note")
        mtf_str = f"  {mtf}" if mtf else ""
        lines.append(
            f"{arrow} *{r['ticker']}* ${r['price']}  ({r['chg']:+.1f}%)  "
            f"RSI={r['rsi']}\n   📊 評分 *{score:+.2f}* ({rating}){mtf_str}"
        )
        for s in r.get("signals", []):
            lines.append(f"   ↳ {s}")
        pos = r.get("position")
        if pos and pos.get("shares", 0) > 0:
            lines.append(
                f"   💰 建議 {pos['shares']:.0f} 股（佔帳戶 {pos['pct']:.1%}）"
                f"｜停損 ${pos['stop']:.2f}｜年化波動 {pos['ann_vol']:.0%}"
            )
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

    no_signal = [r["ticker"] for r in results if not r.get("signals")]
    lines.append(f"共掃描 *{len(results)}* 支，觸發 *{len(flagged)}* 支")
    if suppressed:
        lines.append(f"_🔕 {suppressed} 個重複訊號因冷卻被合併_")
    if no_signal:
        lines.append(f"_無訊號：{', '.join(no_signal[:10])}{'…' if len(no_signal)>10 else ''}_")
    lines.append("_輸入 /help 管理清單 · /protections 查看防護_")
    return "\n".join(lines)


# ── Auto-calibration helper ──────────────────────────────────────────────────

def maybe_calibrate(state: dict, max_age_days: int = 7, force: bool = False) -> bool:
    """
    若校準資料超過 max_age_days 天（或不存在），重新跑回測校準。
    回傳 True 表示有更新（呼叫端應 save_state）。
    """
    cal = state.get("calibration", {})
    updated = cal.get("_updated")
    stale = True
    if updated and not force:
        try:
            age = datetime.now(timezone.utc) - datetime.fromisoformat(updated)
            stale = age.days >= max_age_days
        except Exception:
            stale = True
    if not (stale or force):
        return False

    new_cal = calibrate(state["watchlist"])
    new_cal["_updated"] = datetime.now(timezone.utc).isoformat()
    state["calibration"] = new_cal
    return True


def _calibration_weights(state: dict) -> dict:
    """取出可傳入 scan() 的 {ticker: {component: mult}}（去掉 _updated）。"""
    cal = dict(state.get("calibration", {}))
    cal.pop("_updated", None)
    return cal


def maybe_plan_calibrate(state: dict, max_age_days: int = 7) -> str | None:
    """
    當日計畫的每週自動回測校準（/plantest 的自動版；`/set plan_autocal_enabled off` 關閉）。
    節奏與 maybe_calibrate 相同（7 天）；只在校準「動作有變」時回傳通知文字。
    手動 /plantest opt apply 的參數（params）會被保留。
    """
    th = state["thresholds"]
    if not th.get("plan_autocal_enabled", True) or not state["watchlist"]:
        return None
    old = state.get("plan_calib") or {}
    upd = old.get("_auto_checked") or old.get("as_of")
    if upd:
        try:
            age = datetime.now(timezone.utc) - datetime.fromisoformat(upd).replace(
                tzinfo=timezone.utc)
            if age.days < max_age_days:
                return None
        except ValueError:
            pass
    now_s = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        import plan_backtest as pbt
        prm = old.get("params")                    # 尊重手動尋優參數
        _agg2, cal_new, _tr2 = pbt.run(state["watchlist"][:12], params=prm)
    except Exception as e:
        print(f"plan auto-calibration failed: {e}")
        return None
    if not cal_new.get("setups"):
        # 樣本不足：只蓋時間戳避免每 15 分重試，不動現有校準
        state.setdefault("plan_calib", {})["_auto_checked"] = now_s
        return None
    cal_new["as_of"] = now_s
    cal_new["_auto_checked"] = now_s
    cal_new["auto"] = True
    if prm:
        cal_new["params"] = prm
    changed_actions = []
    for s_, c_ in cal_new["setups"].items():
        o_ = (old.get("setups") or {}).get(s_, {"enabled": True, "conf_delta": 0})
        if (c_["enabled"], c_["conf_delta"]) != (o_.get("enabled", True),
                                                 o_.get("conf_delta", 0)):
            if not c_["enabled"]:
                changed_actions.append(f"{s_} → ⛔ 停用（{c_['why']}）")
            elif c_["conf_delta"]:
                changed_actions.append(f"{s_} → ⬇️ 信心-1（{c_['why']}）")
            else:
                changed_actions.append(f"{s_} → ✅ 恢復正常")
    state["plan_calib"] = cal_new
    if changed_actions:
        return ("🔧 *當日計畫自動校準*（每週回測 60 日）\n" +
                "\n".join(f"・{a}" for a in changed_actions) +
                "\n`/plantest` 看完整報告｜`/set plan_autocal_enabled off` 關閉自動校準")
    return None


# ── 自動掃描循環（main 與 daemon 共用，確保兩邊行為一致）─────────────────────────

def evaluate_price_alerts(alerts: list, prices: dict) -> tuple[list, list]:
    """檢查到價警報（純函數，離線可測）。回 (觸發清單, 保留清單)。
    prices 缺該標的時保留警報等下次；壞資料（level 非數字）直接丟棄。"""
    triggered, remaining = [], []
    for a in alerts:
        try:
            level = float(a["level"])
            op = a.get("op", ">=")
            tk = a["ticker"]
        except Exception:
            continue
        px = prices.get(tk)
        if px is None:
            remaining.append(a)
        elif (op == ">=" and px >= level) or (op == "<=" and px <= level):
            triggered.append({**a, "price": float(px)})
        else:
            remaining.append(a)
    return triggered, remaining


def _alert_prices(tickers: list) -> dict:
    """批次抓警報標的最新收盤。失敗回 {}（警報保留到下次）。"""
    try:
        from sector_scan import _batch_closes
        closes = _batch_closes(sorted(set(tickers)), "5d", min_len=1)
        return {tk: float(s.iloc[-1]) for tk, s in closes.items() if len(s)}
    except Exception as e:
        print(f"alert price fetch error: {e}")
        return {}


def scan_and_report(state: dict, timestamp: str) -> tuple[str | None, list[dict]]:
    """
    執行一次完整的自動掃描：校準 → 掃描 → 大盤濾網 → 冷卻去重 → 組訊息。
    就地更新 state（calibration / signal_history / last_scan_time）。
    回傳 (訊息字串或 None, results)。供 main() 與 bot_daemon 共用。
    """
    th = state["thresholds"]

    # 每週自動回測校準（自我優化迴圈）
    if maybe_calibrate(state):
        print("Calibration refreshed from backtest edges.")

    # 當日計畫每週自動校準（有動作變更時併入通知）
    plan_cal_note = None
    try:
        plan_cal_note = maybe_plan_calibrate(state)
    except Exception as e:
        print(f"plan autocal error: {e}")

    # 掃描（校準加權評分）
    results = scan(state["watchlist"], th, calibration=_calibration_weights(state))
    state["last_scan_time"] = timestamp

    # 大盤風險濾網
    regime = market_regime() if th.get("regime_filter_enabled", True) else None
    if regime:
        print(f"Market regime: {regime['label']}")

    # 冷卻去重（防洗版）
    suppressed = 0
    if th.get("cooldown_enabled", True):
        results, suppressed = apply_cooldown(results, state)
        if suppressed:
            print(f"Cooldown suppressed {suppressed} duplicate signals.")

    message = _build_message(results, timestamp, regime=regime, suppressed=suppressed)

    # 到價警報（/alert 設定；觸發即通知並自動移除，不受靜音/冷卻影響）
    alerts = state.get("price_alerts") or []
    if alerts:
        trig, remain = evaluate_price_alerts(alerts, _alert_prices(
            [a.get("ticker") for a in alerts if a.get("ticker")]))
        if trig or len(remain) != len(alerts):
            state["price_alerts"] = remain
        if trig:
            lines = ["🎯 *到價警報觸發*"]
            for t in trig:
                arrow = "≥" if t.get("op") == ">=" else "≤"
                lines.append(f"• *{t['ticker']}* 現價 {t['price']:.2f}，已{arrow} {float(t['level']):.2f}")
            alert_msg = "\n".join(lines)
            message = (message + "\n\n" + alert_msg) if message else alert_msg

    # 反思記憶（FinMem 式）：記錄強判斷 → 5 日後結算命中率 → 晨報/助理回饋
    try:
        import reflection as rfl
        today_s = (timestamp or "")[:10] or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        strong = sorted([x for x in results if abs(x.get("score", 0)) >= 0.5],
                        key=lambda x: -abs(x.get("score", 0)))[:3]
        for s_ in strong:
            if s_.get("price"):
                rfl.record_pick(state, s_["ticker"], float(s_["score"]),
                                float(s_["price"]), today_s)
        pend = {p["ticker"] for p in state.get("reflections", {}).get("pending", [])}
        if pend:
            rfl.evaluate_pending(state, _alert_prices(sorted(pend)), today_s)
    except Exception as e:
        print(f"reflection error: {e}")

    if plan_cal_note:
        message = (message + "\n\n" + plan_cal_note) if message else plan_cal_note
    return message, results


# ── Alpaca 自動交易 ───────────────────────────────────────────────────────────

def _alpaca_keys() -> tuple[str, str]:
    return os.environ.get("ALPACA_KEY_ID", ""), os.environ.get("ALPACA_SECRET_KEY", "")


def run_autotrade(state: dict, results: list[dict]) -> str | None:
    """
    依掃描評分自動下模擬單（僅在 autotrade 開啟 + 有 key + 市場開盤時）。
    回傳執行摘要字串或 None。安全預設：autotrade_enabled 預設 False。
    """
    th = state["thresholds"]
    if not th.get("autotrade_enabled", False):
        return None
    key, secret = _alpaca_keys()
    if not key or not secret:
        print("Autotrade: 未設定 Alpaca key，跳過")
        return None
    if not market_status().get("open", False):
        print("Autotrade: 市場未開盤，跳過")
        return None
    try:
        import alpaca_trader as at
    except Exception as e:
        print(f"Autotrade: alpaca_trader 不可用 {e}")
        return None

    account = at.get_account(key, secret)
    if not account:
        print("Autotrade: 帳戶讀取失敗")
        return None
    positions = at.get_positions(key, secret)
    equity = at._f(account.get("equity")) or 0.0
    bp = at._f(account.get("buying_power")) or 0.0

    scored = []
    for r in results:
        pos = r.get("position") or {}
        price = r.get("price")
        rps = (price - pos["stop"]) if (pos.get("stop") and price) else None
        scored.append({"ticker": r["ticker"], "score": r.get("score", 0),
                       "price": price, "risk_per_share": rps})

    config = {
        "buy_threshold":    th.get("at_buy_threshold", 0.5),
        "exit_threshold":   th.get("at_exit_threshold", -0.2),
        "max_positions":    int(th.get("at_max_positions", 10)),
        "max_position_pct": th.get("at_max_position_pct", 0.15),
        "risk_pct":         th.get("risk_pct", 0.01),
    }
    orders = at.decide_orders(scored, positions, equity, bp, config)
    if not orders:
        print("Autotrade: 無符合下單條件")
        return None

    score_by_sym = {s["ticker"]: s.get("score") for s in scored}
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = ["🤖 *自動交易執行*（Alpaca 模擬）"]
    journal_entries = []
    for o in orders:
        ok, msg = at.submit_order(key, secret, o["symbol"], o["qty"], o["side"])
        icon = "✅" if ok else "❌"
        tail = "" if ok else f"（{msg}）"
        lines.append(f"{icon} {o['side'].upper()} {o['symbol']} x{int(o['qty'])} — {o['reason']}{tail}")
        journal_entries.append({
            "time": now_iso, "symbol": o["symbol"], "side": o["side"],
            "qty": int(o["qty"]), "score": score_by_sym.get(o["symbol"]),
            "reason": o["reason"], "submitted": ok,
            "error": None if ok else msg,
        })
    if journal_entries:
        at.append_journal(JOURNAL_FILE, journal_entries)
    print(f"Autotrade: 送出 {len(orders)} 筆")
    return "\n".join(lines)


# ── Entrypoint ───────────────────────────────────────────────────────────────

def main() -> int:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"=== RBS Signal Scanner  {now} ===\n")

    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("WARNING: TELEGRAM_TOKEN / TELEGRAM_CHAT_ID not set. Signals printed only.")

    state = load_state()

    # Step 1: Process incoming Telegram commands
    if TELEGRAM_TOKEN:
        print("── Processing Telegram commands ──")
        try:
            state, changed = process_commands(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, state)
        except Exception as e:
            # 毒訊息防護：state 是就地修改，last_update_id 已前進——照樣存檔，
            # 讓壞訊息被消耗掉，而不是讓之後每 15 分鐘的 cron 重複崩潰
            print(f"Command processing error: {e}")
            changed = True
        if changed:
            save_state(state)
            print("State updated from commands.")

    tickers    = state["watchlist"]
    thresholds = state["thresholds"]
    ms = market_status()
    print(f"Market status: {ms['reason']}")

    # Step 1.5: Daily AI briefing（盤前獨立推送，不受開盤掃描限制）
    if _should_send_briefing(state):
        print("── Sending daily briefing ──")
        bmsg = daily_briefing(state)
        if bmsg and TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
            _tg_send(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, bmsg)
            print("Daily briefing sent.")
        state["last_briefing_date"] = datetime.now(ET).strftime("%Y-%m-%d")
        save_state(state)

    # Step 1.6: 每週深度週報（週日 ET 晚間）
    if _should_send_weekly(state):
        print("── Sending weekly report ──")
        try:
            wmsg = weekly_report(state)
            if wmsg and TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
                _tg_send(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, wmsg)
                print("Weekly report sent.")
        except Exception as e:
            print(f"Weekly report error: {e}")
        _now_w = datetime.now(ET)
        state["last_weekly"] = f"{_now_w.isocalendar().year}-W{_now_w.isocalendar().week}"
        save_state(state)

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

    # Step 3: Scan + protections (shared with daemon)
    print("── Running signal scan ──")
    message, results = scan_and_report(state, now)

    # Step 4: Send
    if message is None:
        print("\nNo signals triggered (or all in cooldown). Nothing to send.")
    else:
        print(f"\n{len([r for r in results if r['signals']])} tickers flagged.")
        if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
            ok = _tg_send(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, message)
            print("Telegram sent." if ok else "Telegram send failed.")

    # Step 4.5: Auto-trade (Alpaca paper; gated by autotrade_enabled + keys + open)
    at_msg = run_autotrade(state, results)
    if at_msg and TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        _tg_send(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, at_msg)

    # Step 5: Save state
    save_state(state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
