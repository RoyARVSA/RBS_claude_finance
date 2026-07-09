"""
trade_plan.py — 當日交易計畫（盤中決策建議）
RBS Finance Dashboard

把「現在盤中該不該動作」轉成具體訂單票（教育模擬用，非投資建議）：
  日線趨勢閘門（MA20/50 + RSI + MACD）決定方向偏多/觀望，
  盤中確認（VWAP / 開盤區間 ORB / 相對量能 RVOL / 跳空）決定進場型態，
  ATR 風險基準給出 進場區間 / 停損 / 停利 / 建議股數 / 風險回報比。

慣例：純邏輯（intraday_metrics / daily_gate / build_ticket / plan_text）離線可測；
抓取層（fetch_plan_data / alpaca_latest_prices）需網路。
資料延遲誠實揭露：yfinance 盤中約延遲 15 分鐘；設 Alpaca key 時以免費 IEX
即時報價校正「現價」。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest import rsi as _rsi
from quant_tools import atr_position_size

# ── 純邏輯：盤中指標 ─────────────────────────────────────────────────────────

ORB_MINUTES = 30          # 開盤區間分鐘數
RVOL_STRONG = 1.5         # 相對量能門檻
STOP_ATR_MULT = 1.5       # 停損 = 1.5×ATR（與 Bot / 交易工具一致）
TARGET_RR = 2.0           # 停利 = 2R
MAX_NOTIONAL_PCT = 0.30   # 單一部位名目金額上限（佔帳戶比例）
_NON_US_SUFFIX = {"TW", "TWO", "HK", "T", "KS", "L", "SS", "SZ"}   # Alpaca 不支援


def _session_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """取最後一個交易日的盤中 K 棒（bars 可含前一日）。"""
    if bars is None or bars.empty:
        return pd.DataFrame()
    b = bars.copy()
    if isinstance(b.columns, pd.MultiIndex):          # yfinance 單檔仍可能多層
        b.columns = b.columns.get_level_values(0)
    b = b.dropna(subset=["Close"])
    if b.empty:
        return b
    last_day = b.index[-1].date()
    return b[[d.date() == last_day for d in b.index]]


def intraday_metrics(bars: pd.DataFrame, daily: pd.DataFrame,
                     orb_minutes: int = ORB_MINUTES,
                     session_minutes: float = 390.0) -> dict | None:
    """由 5 分 K + 日線計算盤中決策指標。回傳 None 表示資料不足。
    session_minutes：全日交易分鐘數（美股 390、台股 270）——影響 RVOL 分母。"""
    sess = _session_bars(bars)
    if len(sess) < 2 or daily is None or len(daily) < 60:
        return None
    d = daily.copy()
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)

    last = float(sess["Close"].iloc[-1])
    typ = (sess["High"] + sess["Low"] + sess["Close"]) / 3.0
    vol = sess["Volume"].astype(float)
    cum_v = float(vol.sum())
    vwap = float((typ * vol).sum() / cum_v) if cum_v > 0 else last

    bar_minutes = 5
    if len(sess) >= 2:
        step = (sess.index[1] - sess.index[0]).total_seconds() / 60.0
        if step > 0:
            bar_minutes = step
    n_orb = max(1, int(round(orb_minutes / bar_minutes)))
    orb = sess.iloc[:n_orb]
    orb_high, orb_low = float(orb["High"].max()), float(orb["Low"].min())

    # 日線基準
    close_d = d["Close"].astype(float)
    prev_close = float(close_d.iloc[-1]) if close_d.index[-1].date() != sess.index[-1].date() \
        else float(close_d.iloc[-2])
    today_open = float(sess["Open"].iloc[0])
    gap_pct = (today_open / prev_close - 1.0) * 100 if prev_close > 0 else 0.0

    # 日線 ATR(14)
    hi, lo, cl = d["High"].astype(float), d["Low"].astype(float), close_d
    tr = pd.concat([hi - lo, (hi - cl.shift()).abs(), (lo - cl.shift()).abs()], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1])

    # 相對量能：今日截至目前累積量 vs 20 日均量 × 已進行時間比例
    avg_v20 = float(d["Volume"].astype(float).tail(20).mean())
    elapsed_frac = min(1.0, len(sess) * bar_minutes / max(session_minutes, bar_minutes))
    rvol = (cum_v / (avg_v20 * elapsed_frac)) if avg_v20 > 0 and elapsed_frac > 0 else 0.0

    return {"last": last, "vwap": vwap, "orb_high": orb_high, "orb_low": orb_low,
            "gap_pct": gap_pct, "rvol": rvol, "atr": atr,
            "above_vwap": last >= vwap, "prev_close": prev_close,
            "session_date": str(sess.index[-1].date()), "bars": len(sess)}


def daily_gate(daily: pd.DataFrame) -> dict:
    """日線趨勢閘門：只在日線不反對時給進場票（多頭偏好；本專案不做放空建議）。"""
    d = daily.copy()
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    close = d["Close"].astype(float)
    ma20 = float(close.rolling(20).mean().iloc[-1])
    ma50 = float(close.rolling(50).mean().iloc[-1])
    r = float(_rsi(close, 14).iloc[-1])
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_hist = float((ema12 - ema26 - (ema12 - ema26).ewm(span=9, adjust=False).mean()).iloc[-1])

    score = 0
    score += 1 if ma20 > ma50 else -1
    score += 1 if close.iloc[-1] > ma20 else -1
    score += 1 if macd_hist > 0 else -1
    if r >= 75:
        score -= 1        # 過熱降級
    if r <= 30:
        score += 1        # 超賣回升候選

    bias = "bullish" if score >= 2 else ("neutral" if score >= 0 else "bearish")
    return {"bias": bias, "score": score, "rsi": r, "ma20": ma20, "ma50": ma50,
            "macd_hist": macd_hist}


def build_ticket(ticker: str, m: dict, gate: dict,
                 account: float = 100_000.0, risk_pct: float = 0.01,
                 days_to_earnings: int | None = None,
                 calib: dict | None = None,
                 stop_atr_mult: float = STOP_ATR_MULT,
                 target_rr: float = TARGET_RR) -> dict:
    """盤中指標 + 日線閘門 → 當日訂單票。永遠回傳 dict（action 可能是 觀望/迴避）。
    days_to_earnings：距下次財報天數（0=今天）；0-1 天內一律迴避（事件跳空風險）。
    calib：plan_backtest walk-forward 校準（{"setups": {setup: {enabled, conf_delta}}}）；
    只降不升——負期望型態停用、不穩定型態降信心。
    stop_atr_mult / target_rr：停損 ATR 倍數與目標 R:R——/plantest opt 參數尋優可調。"""
    reasons: list[str] = []
    t = {"ticker": ticker, "action": "觀望", "setup": "", "entry_lo": None,
         "entry_hi": None, "stop": None, "target": None, "shares": 0,
         "rr": None, "confidence": 0, "reasons": reasons,
         "last": m["last"], "vwap": m["vwap"], "rvol": m["rvol"],
         "gap_pct": m["gap_pct"], "session_date": m["session_date"],
         "valid": "當日有效（DAY）"}

    if days_to_earnings is not None and 0 <= days_to_earnings <= 1:
        t["action"] = "迴避"
        when = "今日" if days_to_earnings == 0 else "明日"
        reasons.append(f"⚠️ {when}財報——事件跳空風險，當日票一律迴避")
        return t

    if gate["bias"] == "bearish":
        t["action"] = "迴避"
        reasons.append(f"日線趨勢偏空（score {gate['score']:+d}，MA20{'<' if gate['ma20'] < gate['ma50'] else '>'}MA50）")
        return t

    last, vwap, atr = m["last"], m["vwap"], m["atr"]
    if atr <= 0 or last <= 0:
        reasons.append("ATR/價格資料異常")
        return t

    conf = 0
    if gate["bias"] == "bullish":
        conf += 2
        reasons.append(f"日線多頭（MA20>MA50、RSI {gate['rsi']:.0f}）")
    else:
        conf += 1
        reasons.append(f"日線中性（score {gate['score']:+d}）——僅低信心試單")
    if m["rvol"] >= RVOL_STRONG:
        conf += 1
        reasons.append(f"量能放大（RVOL {m['rvol']:.1f}×）")
    if m["gap_pct"] >= 1.0:
        conf += 1
        reasons.append(f"跳空上漲 {m['gap_pct']:+.1f}%")
    elif m["gap_pct"] <= -1.5:
        conf -= 1
        reasons.append(f"跳空下跌 {m['gap_pct']:+.1f}%（追多風險高）")

    # 進場型態
    if m["above_vwap"] and last > m["orb_high"]:
        setup = "ORB 突破"
        entry_lo, entry_hi = last, round(last + 0.25 * atr, 2)
        stop = round(max(m["orb_low"], last - stop_atr_mult * atr), 2)
        reasons.append(f"站上 VWAP 且突破開盤區間高點 {m['orb_high']:.2f}")
        conf += 1
    elif m["above_vwap"]:
        setup = "VWAP 回踩"
        entry_lo, entry_hi = round(vwap, 2), round(min(last, vwap + 0.3 * atr), 2)
        if entry_hi < entry_lo:
            entry_hi = round(entry_lo + 0.1 * atr, 2)
        stop = round(vwap - stop_atr_mult * atr, 2)
        reasons.append(f"價格在 VWAP {vwap:.2f} 之上，等回踩不追高")
    else:
        t["action"] = "觀望"
        reasons.append(f"價格在 VWAP {vwap:.2f} 之下——盤中弱勢，今日不進場")
        return t

    # 歷史回測校準（/plantest apply 後生效；只降不升）
    c_ = ((calib or {}).get("setups") or {}).get(setup)
    if c_:
        if not c_.get("enabled", True):
            t["action"] = "觀望"
            t["setup"] = setup
            t["confidence"] = conf
            reasons.append(f"📜 歷史回測負期望——{setup} 型態暫停用（/plantest 校準）")
            return t
        d_ = int(c_.get("conf_delta", 0))
        if d_:
            conf += d_
            reasons.append(f"📜 歷史回測不穩定——信心 {d_:+d}（/plantest 校準）")

    if conf < 2:
        t["action"] = "觀望"
        reasons.append("綜合信心不足（<2），寧可錯過")
        t["confidence"] = conf
        return t

    entry_mid = (entry_lo + entry_hi) / 2.0
    risk_ps = entry_mid - stop
    if risk_ps <= 0:
        t["action"] = "觀望"
        reasons.append("停損距離異常（entry ≤ stop）")
        return t
    target = round(entry_mid + target_rr * risk_ps, 2)
    pos = atr_position_size(account, risk_pct, entry_mid, atr, atr_mult=stop_atr_mult)
    shares = int(pos["shares"])
    # 用實際票面停損重新校正股數（風險預算 / 每股風險）
    shares = int(min(shares if shares > 0 else 0,
                     (account * risk_pct) / risk_ps)) if risk_ps > 0 else 0
    # 名目金額上限：低波動股票的風險式股數可能超過帳戶購買力（現金帳戶不可行）
    shares = min(shares, int(account * MAX_NOTIONAL_PCT / entry_mid))

    t.update({"action": "買進" if gate["bias"] == "bullish" else "小量試單",
              "setup": setup, "entry_lo": round(entry_lo, 2), "entry_hi": round(entry_hi, 2),
              "stop": stop, "target": target, "shares": max(shares, 0),
              "rr": round((target - entry_mid) / risk_ps, 1), "confidence": conf})
    return t


def plan_text(tickets: list[dict], account: float, risk_pct: float,
              realtime_src: str | None = None) -> str:
    """訂單票 → 文字版（Bot / 下載共用）。"""
    lines = [f"⚡ 當日交易計畫（帳戶 ${account:,.0f}、單筆風險 {risk_pct:.1%}）"]
    if realtime_src:
        lines.append(f"現價來源：{realtime_src}")
    actionable = [t for t in tickets if t["action"] in ("買進", "小量試單")]
    watch = [t for t in tickets if t["action"] == "觀望"]
    avoid = [t for t in tickets if t["action"] == "迴避"]
    for t in actionable:
        lines.append(
            f"\n🟢 {t['ticker']} {t['action']}｜{t['setup']}（信心 {t['confidence']}/5）\n"
            f"   進場 {t['entry_lo']}–{t['entry_hi']}｜停損 {t['stop']}｜停利 {t['target']}"
            f"（R:R {t['rr']}）｜{t['shares']} 股｜{t['valid']}")
        for r in t["reasons"]:
            lines.append(f"   • {r}")
    if watch:
        lines.append("\n🟡 觀望：" + "、".join(
            f"{t['ticker']}（{t['reasons'][-1] if t['reasons'] else '—'}）" for t in watch))
    if avoid:
        lines.append("🔴 迴避：" + "、".join(t["ticker"] for t in avoid))
    if not actionable:
        lines.append("\n今日無符合條件的進場票——空手也是部位。")
    lines.append("\n⚠️ 模擬教育用途，非投資建議；盤中數據可能延遲約 15 分鐘。")
    return "\n".join(lines)


# ── 抓取層（需網路）──────────────────────────────────────────────────────────

STALE_TRADE_MIN = 10   # IEX 成交價超過 N 分鐘視為過舊（低流動性股票常見），棄用


def _parse_alpaca_ts(t: str):
    """RFC-3339（可能帶奈秒）→ aware datetime；解析失敗回 None。"""
    from datetime import datetime, timezone
    try:
        return datetime.strptime(t[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def alpaca_latest_prices(tickers: list[str]) -> dict[str, float]:
    """Alpaca 免費 IEX 即時成交價（設 ALPACA_KEY_ID/SECRET 才用；失敗回空 dict）。

    事實查證（官方 SDK/文件）：端點 /v2/stocks/trades/latest?symbols=..&feed=iex，
    標頭 APCA-API-KEY-ID/SECRET（paper key 可用），回應 {"trades": {SYM: {"p":…, "t":…}}}。
    免費方案顯式帶 feed=iex 最穩（帶 sip 會 403）；IEX 對低流動性標的可能回
    過舊成交價 → 檢查 t 新鮮度，過舊棄用改回 yfinance 現價。
    """
    import os
    from datetime import datetime, timedelta, timezone
    import requests
    key, sec = os.environ.get("ALPACA_KEY_ID", ""), os.environ.get("ALPACA_SECRET_KEY", "")
    if not key or not sec or not tickers:
        return {}
    try:
        r = requests.get(
            "https://data.alpaca.markets/v2/stocks/trades/latest",
            params={"symbols": ",".join(tickers), "feed": "iex"},
            headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec},
            timeout=10)
        if not r.ok:
            return {}
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=STALE_TRADE_MIN)
        out = {}
        for sym, obj in (r.json() or {}).get("trades", {}).items():
            p, ts = obj.get("p"), _parse_alpaca_ts(obj.get("t", ""))
            if p and ts and ts >= cutoff:
                out[sym] = float(p)
        return out
    except Exception:
        return {}


def fetch_plan_data(tickers: list[str]) -> dict[str, dict]:
    """一次抓齊 5 分 K + 日線。回傳 {ticker: {"bars": df, "daily": df}}。"""
    import yfinance as yf
    out: dict[str, dict] = {}
    for tk in tickers:
        try:
            bars = yf.download(tk, period="2d", interval="5m",
                               auto_adjust=True, progress=False)
            daily = yf.download(tk, period="6mo", interval="1d",
                                auto_adjust=True, progress=False)
            if bars is not None and not bars.empty and daily is not None and not daily.empty:
                out[tk] = {"bars": bars, "daily": daily}
        except Exception:
            continue
    return out


def build_plans(tickers: list[str], account: float = 100_000.0,
                risk_pct: float = 0.01,
                calib: dict | None = None) -> tuple[list[dict], str | None]:
    """端到端：抓資料 → 訂單票。回傳 (tickets, 即時價來源標籤或 None)。
    calib：plan_backtest 校準（見 build_ticket）。"""
    data = fetch_plan_data(tickers)
    # 非美股（.TW 等）不能進 Alpaca 請求——一顆壞代碼會讓整批 400
    us_only = [t for t in data if t.split(".")[-1] not in _NON_US_SUFFIX or "." not in t]
    live = alpaca_latest_prices(us_only)
    src = "Alpaca IEX 即時" if live else None
    tickets = []
    for tk, dd in data.items():
        sess_min = 270.0 if tk.upper().endswith((".TW", ".TWO")) else 390.0
        # 日線閘門/ATR/量能基準只用「今日以前」的日線——與 plan_backtest 重放同一母體，
        # 也修正 avg_v20 把今日未完成量算進分母而虛增 RVOL 的偏差
        daily_tk = dd["daily"]
        try:
            _sd = dd["bars"].index[-1].date()
            _hist = daily_tk[daily_tk.index.date < _sd]
            if len(_hist) >= 60:
                daily_tk = _hist
        except Exception:
            pass
        prm = (calib or {}).get("params") or {}   # /plantest opt apply 的尋優參數
        m = intraday_metrics(dd["bars"], daily_tk,
                             orb_minutes=int(prm.get("orb_minutes", ORB_MINUTES)),
                             session_minutes=sess_min)
        if not m:
            continue
        if tk in live:                       # 用即時價覆蓋延遲現價
            m["last"] = live[tk]
            m["above_vwap"] = m["last"] >= m["vwap"]
        gate = daily_gate(daily_tk)
        d2e = None
        try:                                 # 財報日事件閘門（抓不到就略過）
            from fundamentals import next_earnings_date
            import datetime as _dt
            ed = next_earnings_date(tk)
            if ed:
                d2e = (ed - _dt.date.today()).days
        except Exception:
            pass
        tickets.append(build_ticket(
            tk, m, gate, account, risk_pct, days_to_earnings=d2e, calib=calib,
            stop_atr_mult=float(prm.get("stop_atr_mult", STOP_ATR_MULT)),
            target_rr=float(prm.get("target_rr", TARGET_RR))))
    order = {"買進": 0, "小量試單": 1, "觀望": 2, "迴避": 3}
    tickets.sort(key=lambda t: (order.get(t["action"], 9), -(t["confidence"] or 0)))
    return tickets, src


# ── CLI 自我測試（離線純邏輯）─────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(11)

    # 合成日線：上升趨勢 120 天
    n = 120
    drift = rng.normal(0.002, 0.012, n)
    close = 100 * np.cumprod(1 + drift)
    d_idx = pd.bdate_range("2025-01-01", periods=n)
    daily = pd.DataFrame({
        "Open": close * (1 - rng.uniform(0, 0.005, n)),
        "High": close * (1 + rng.uniform(0, 0.01, n)),
        "Low":  close * (1 - rng.uniform(0, 0.01, n)),
        "Close": close,
        "Volume": rng.integers(2_000_000, 4_000_000, n).astype(float),
    }, index=d_idx)

    # 合成 5 分 K：跳空上漲後沿 VWAP 走高、尾段突破開盤區間
    base = close[-1] * 1.015
    steps = 60
    intr = base * np.cumprod(1 + np.abs(rng.normal(0.0006, 0.0008, steps)))
    i_idx = pd.date_range(d_idx[-1] + pd.Timedelta(days=1, hours=9, minutes=30),
                          periods=steps, freq="5min")
    bars = pd.DataFrame({
        "Open": intr * 0.9995, "High": intr * 1.0012, "Low": intr * 0.9988,
        "Close": intr, "Volume": rng.integers(80_000, 150_000, steps).astype(float),
    }, index=i_idx)

    m = intraday_metrics(bars, daily)
    assert m and m["bars"] == steps, m
    assert m["orb_high"] >= m["orb_low"] > 0, m
    assert m["vwap"] > 0 and m["atr"] > 0, m
    assert m["gap_pct"] > 0.5, m["gap_pct"]

    gate = daily_gate(daily)
    assert gate["bias"] in ("bullish", "neutral"), gate   # 上升趨勢不應判空

    tk = build_ticket("TEST", m, gate, account=100_000, risk_pct=0.01)
    assert tk["action"] in ("買進", "小量試單", "觀望"), tk["action"]
    if tk["action"] in ("買進", "小量試單"):
        assert tk["stop"] < tk["entry_lo"] <= tk["entry_hi"] < tk["target"], tk
        assert tk["shares"] >= 0 and tk["rr"] and tk["rr"] >= 1.5, tk
        # 風險預算驗證：股數 × 每股風險 ≤ 帳戶 × 風險% (+5% 容差)
        mid = (tk["entry_lo"] + tk["entry_hi"]) / 2
        assert tk["shares"] * (mid - tk["stop"]) <= 100_000 * 0.01 * 1.05, tk

    # 空頭日線 → 必須迴避
    bear = daily.copy()
    bear["Close"] = bear["Close"].iloc[::-1].to_numpy()
    bear["High"], bear["Low"] = bear["Close"] * 1.01, bear["Close"] * 0.99
    bear["Open"] = bear["Close"]
    g2 = daily_gate(bear)
    t2 = build_ticket("BEAR", m, g2)
    assert g2["bias"] == "bearish" and t2["action"] == "迴避", (g2, t2["action"])

    # VWAP 之下 → 觀望
    m3 = dict(m); m3["above_vwap"] = False
    t3 = build_ticket("WEAK", m3, gate)
    assert t3["action"] == "觀望", t3

    # 校準回饋：型態停用 → 觀望；conf_delta=-1 → 信心降級（只降不升）
    _cal = {"setups": {"ORB 突破": {"enabled": False},
                       "VWAP 回踩": {"enabled": True, "conf_delta": -1}}}
    t_cal = build_ticket("CAL", m, gate, calib=_cal)
    assert t_cal["action"] == "觀望" and any("暫停用" in r for r in t_cal["reasons"]), t_cal
    _cal2 = {"setups": {"ORB 突破": {"enabled": True, "conf_delta": -1}}}
    t_cal2 = build_ticket("CAL2", m, gate, calib=_cal2)
    t_nocal = build_ticket("NC", m, gate)
    assert t_cal2["confidence"] == t_nocal["confidence"] - 1, (t_cal2, t_nocal)

    # 參數化：target_rr 直接反映在停利與 R:R；stop_atr_mult 拉大 → VWAP 型停損更遠
    t_rr = build_ticket("RR", m, gate, target_rr=3.0)
    assert t_rr["rr"] == 3.0 and t_rr["target"] > t_nocal["target"], (t_rr, t_nocal)
    # 強制走 VWAP 回踩分支（orb_high 抬高到 last 之上，ORB 突破不成立）
    m_vwap = {**m, "last": m["vwap"] * 1.001, "orb_high": m["vwap"] * 1.10}
    tw_a = build_ticket("SA", m_vwap, gate, stop_atr_mult=1.0)
    tw_b = build_ticket("SB", m_vwap, gate, stop_atr_mult=2.0)
    assert tw_a["setup"] == tw_b["setup"] == "VWAP 回踩", (tw_a["setup"], tw_b["setup"])
    assert tw_a["stop"] is not None and tw_b["stop"] is not None, (tw_a, tw_b)
    assert tw_b["stop"] < tw_a["stop"], (tw_a["stop"], tw_b["stop"])

    # 財報日迴避：明日財報 → 就算盤面完美也擋
    t5 = build_ticket("ERN", m, gate, days_to_earnings=1)
    assert t5["action"] == "迴避" and "財報" in t5["reasons"][0], t5
    t5b = build_ticket("ERN2", m, gate, days_to_earnings=7)   # 一週後 → 不影響
    assert t5b["action"] != "迴避" or "財報" not in (t5b["reasons"][0] if t5b["reasons"] else ""), t5b

    # 低波動大型股：風險式股數不得超過名目上限（30% 帳戶）
    m4 = dict(m); m4["atr"] = m4["last"] * 0.004      # ATR 僅 0.4% 價格
    m4["orb_low"] = m4["last"] * 0.998                # 極窄停損
    t4 = build_ticket("LOWVOL", m4, gate, account=100_000, risk_pct=0.01)
    if t4["action"] in ("買進", "小量試單") and t4["shares"]:
        mid4 = (t4["entry_lo"] + t4["entry_hi"]) / 2
        assert t4["shares"] * mid4 <= 100_000 * MAX_NOTIONAL_PCT * 1.01, \
            (t4["shares"], t4["shares"] * mid4)

    txt = plan_text([tk, t2, t3], 100_000, 0.01, realtime_src="Alpaca IEX 即時")
    assert "當日交易計畫" in txt and "非投資建議" in txt and "迴避" in txt, txt[:200]

    print("✅ trade_plan 離線自我測試通過"
          f"（ticket={tk['action']}/{tk.get('setup', '')} conf={tk['confidence']}）")
