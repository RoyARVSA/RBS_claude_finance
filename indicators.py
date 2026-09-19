"""
indicators.py – 技術指標與綜合評分（自 scan_signals 抽取；審查團架構師第一刀）

抽取動機（2026-08 七人審查）：
  • scan_signals 3120 行是全專案交易決策的心臟，卻是唯一零斷言的模組
  • app.py 兩處直接偷用私有 `_ss._composite_score`——跨檔私有依賴，
    任何人重構 scan_signals 都會不知情炸掉委員會頁

內容：純指標（_rsi/_macd/_bollinger/_atr_levels/_vol_spike/_ma_trend/
_weekly_trend）、綜合評分 _composite_score（趨勢 35%/MACD 25%/RSI 15%/
布林 10%/量能 15%，可吃回測校準的 edge_weights）、部位提示 _position_hint、
回測校準 calibrate_ticker/calibrate、掃描編排 scan（含 yfinance 批次抓取——
唯一的網路面，其餘全部純邏輯離線可測）。

公開名：composite_score / position_hint 等不帶底線別名——外部（app.py）
一律走公開名；底線名保留供 scan_signals re-export 向後相容。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

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
        mom_s = max(-1.0, min(1.0, ret_1m * 8)) if ret_1m == ret_1m else 0.0   # ±12.5% → ±1；NaN → 0（否則 max/min 會傳出 +1）
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


# ── 向量化評分序列（A/B 段夜間工作流用；與 _composite_score 逐日切片逐位相等）─────────

def _py_round(s: pd.Series, nd: int) -> pd.Series:
    """用 Python round（正確捨入）逐值取整，確保與 _composite_score 的 round() 逐位一致。"""
    return s.map(lambda v: round(float(v), nd) if pd.notna(v) else v)


def _weekly_trend_series(close: pd.Series) -> pd.Series:
    """_weekly_trend 的逐日向量化版：第 i 日的週線序列 = 已完成週的週收 + [close_i]（與切片 resample 同義）。"""
    n = len(close)
    if not isinstance(close.index, pd.DatetimeIndex) or n == 0:
        return pd.Series(0, index=close.index, dtype=int)
    W = close.resample("W").last().dropna()                    # 全序列週收（含最後一個部分週）
    if len(W) == 0:
        return pd.Series(0, index=close.index, dtype=int)
    a12, a26, a9 = 2 / 13, 2 / 27, 2 / 10
    e12 = W.ewm(span=12, adjust=False).mean()
    e26 = W.ewm(span=26, adjust=False).mean()
    macd = e12 - e26
    sig = macd.ewm(span=9, adjust=False).mean()
    wk_end = close.index.to_period("W").end_time.normalize()   # 每日所屬週的週末
    W_end = W.index                                             # 週收索引（週日）
    # 已完成週數 c = 週末 < 該日週末 的週收數
    c = np.searchsorted(W_end.values, wk_end.values, side="left")
    Wv, e12v, e26v, sigv = W.values, e12.values, e26.values, sig.values
    csum = np.concatenate([[0.0], np.cumsum(Wv)])
    out = np.zeros(n, dtype=int)
    px = close.values
    for i in range(n):
        ci = int(c[i])
        if ci + 1 < 12 or np.isnan(px[i]):
            continue
        ma10 = (csum[ci] - csum[ci - 9] + px[i]) / 10.0
        E12 = a12 * px[i] + (1 - a12) * e12v[ci - 1]
        E26 = a26 * px[i] + (1 - a26) * e26v[ci - 1]
        m_i = E12 - E26
        S_i = a9 * m_i + (1 - a9) * sigv[ci - 1]
        out[i] = (1 if px[i] > ma10 else -1) + (1 if (m_i - S_i) > 0 else -1)
    return pd.Series(out, index=close.index, dtype=int)


def composite_series(close: pd.Series, high: pd.Series | None = None, low: pd.Series | None = None,
                     volume: pd.Series | None = None, edge_weights: dict | None = None,
                     mtf: bool = False) -> pd.Series:
    """
    _composite_score 的逐日向量化：回每個位置 i 的評分（= _composite_score(close[:i+1], …)["score"]）。
    只用 i 以前含 i 的資料（無前視）；夜間工作流 415 檔 × 504 日由 ~24 分縮到 ~1 分。
    前提：close 無 NaN（呼叫端先 dropna；engine_backtest.fetch_history 已如此）——收盤含 NaN 時
    _rsi 的 dropna 與 ewm 的 ignore_na 語意不同，之後 ~25 根會有暫態差。成交量可含 NaN。
    自測以逐日切片逐位比對（含 NaN 成交量、edge_weights、mtf）。
    """
    n = len(close)
    idx = close.index
    pos = pd.Series(np.arange(1, n + 1), index=idx)             # 切片長度 len(close[:i+1])
    price = close.astype(float)
    # 1. 趨勢
    trend = pd.Series(0.0, index=idx)
    for span, w in [(20, 0.4), (50, 0.35), (200, 0.25)]:
        ma = price.rolling(span).mean()
        contrib = np.where(price > ma, 1.0, -1.0) * w
        trend = trend + pd.Series(np.where(pos >= span, contrib, 0.0), index=idx)
    trend = _py_round(trend, 3)
    # 2. MACD
    ema12 = price.ewm(span=12, adjust=False).mean()
    ema26 = price.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    h = macd_line - macd_line.ewm(span=9, adjust=False).mean()
    norm = pd.Series(np.where(price != 0, h / price * 100, 0.0), index=idx)
    macd_s = (norm * 4).clip(-1.0, 1.0)
    macd_s = _py_round(pd.Series(np.where(pos >= 35, macd_s, 0.0), index=idx), 3)
    # 3. 動量
    ret_1m = price / price.shift(21) - 1
    mom_s = (ret_1m * 8).clip(-1.0, 1.0)
    mom_s = _py_round(pd.Series(np.where(pos >= 22, mom_s, 0.0), index=idx).fillna(0.0), 3)
    # 4. RSI（與 _rsi 同：ewm alpha=1/14, min_periods=14；round 1 位）
    delta = price.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, min_periods=14).mean()
    loss = (-delta).clip(lower=0).ewm(alpha=1 / 14, min_periods=14).mean()
    rsi = pd.Series(np.where(loss == 0, 100.0, 100 - 100 / (1 + gain / loss)), index=idx)
    rsi = _py_round(rsi, 1)
    rsi_s = pd.Series(0.0, index=idx)
    rsi_s = rsi_s.mask(rsi < 35, (35 - rsi) / 25).mask(rsi > 65, -(rsi - 65) / 25)
    rsi_s = _py_round(rsi_s.clip(-1.0, 1.0).fillna(0.0), 3)
    # 5. 布林
    ma20 = price.rolling(20).mean()
    std20 = price.rolling(20).std()
    u, l = ma20 + 2 * std20, ma20 - 2 * std20
    pct_b = (price - l) / (u - l)
    bb_s = pd.Series(0.0, index=idx)
    bb_s = bb_s.mask((u > l) & (pct_b < 0.15), (0.15 - pct_b) / 0.15) \
               .mask((u > l) & (pct_b > 0.85), -(pct_b - 0.85) / 0.15)
    bb_s = _py_round(pd.Series(np.where(pos >= 20, bb_s.clip(-1.0, 1.0), 0.0), index=idx).fillna(0.0), 3)
    # 加權
    base_w = {"trend": 0.35, "macd": 0.25, "momentum": 0.20, "rsi": 0.10, "bollinger": 0.10}
    if edge_weights:
        adj = {k: base_w[k] * float(edge_weights.get(k, 1.0)) for k in base_w}
        tot = sum(adj.values()) or 1.0
        w = {k: adj[k] / tot for k in adj}
    else:
        w = base_w
    score = w["trend"] * trend + w["macd"] * macd_s + w["momentum"] * mom_s + w["rsi"] * rsi_s + w["bollinger"] * bb_s
    # 成交量放大器
    if volume is not None and len(volume) == n:
        v = volume.astype(float)
        avg_vol = v.rolling(20, min_periods=1).mean().shift(1)
        ratio = v / avg_vol
        amp = (ratio - 1.5) * 0.1
        amp = amp.clip(upper=0.15)
        cond = (pos >= 21) & (avg_vol > 0) & (ratio > 1.5)
        score = score * (1 + pd.Series(np.where(cond, amp, 0.0), index=idx))
    score = score.clip(-1.0, 1.0)
    # MTF
    if mtf:
        wt = _weekly_trend_series(price)
        f = pd.Series(1.0, index=idx)
        f = f.mask((score > 0.1) & (wt >= 1), 1.1).mask((score < -0.1) & (wt <= -1), 1.1) \
             .mask((score > 0.1) & (wt <= -1), 0.8).mask((score < -0.1) & (wt >= 1), 0.8)
        score = (score * f).clip(-1.0, 1.0)
    return _py_round(score, 3)


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


def _extension(close, high, low) -> dict:
    """
    進場延伸度（追高濾網用，純函數）：距 MA20 幾個 ATR、近 5 日報酬。
    2026-09 診斷實案：強訊號進場後 5 日平均報酬為負、硬停損 6/14 在進場 1–2 天內被打到
    ——問題不在方向而在「買在延伸端」。資料不足的欄回 None（引擎視為不濾）。
    """
    out = {"atr": None, "ma20": None, "ext_atr": None, "ret_5d": None,
           "vol_60": None, "mom_12_1": None, "ret_1m": None}        # 後三欄：meta-labeling 線上特徵（與夜間訓練同定義）
    try:
        if len(close) >= 20:
            atr = _atr_value(close, high, low)
            ma20 = float(close.rolling(20).mean().iloc[-1])
            px = float(close.iloc[-1])
            out["atr"], out["ma20"] = round(atr, 4), round(ma20, 4)
            if atr > 0:
                out["ext_atr"] = round((px - ma20) / atr, 2)
        if len(close) >= 6:
            out["ret_5d"] = round(float(close.iloc[-1] / close.iloc[-6] - 1), 4)
        if len(close) >= 61:
            out["vol_60"] = round(float(close.pct_change().iloc[-60:].std()), 5)
        if len(close) >= 22:
            out["ret_1m"] = round(float(close.iloc[-1] / close.iloc[-22] - 1), 4)
        if len(close) >= 253:
            out["mom_12_1"] = round(float(close.iloc[-22] / close.iloc[-253] - 1), 4)
    except Exception:
        pass
    return out


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


def scan(tickers: list[str], thresholds: dict, calibration: dict | None = None, quiet: bool = False) -> list[dict]:
    """quiet=True：不逐檔 print（候選池／持倉補掃用——公開 Actions 日誌不印持倉代碼，PITFALLS D14）。"""
    rsi_lo  = thresholds.get("rsi_oversold",    35)
    rsi_hi  = thresholds.get("rsi_overbought",  68)
    chg_th  = thresholds.get("price_change_pct", 3.0)
    macd_on = thresholds.get("macd_enabled",  True)
    bb_on   = thresholds.get("bb_enabled",    True)
    atr_on  = thresholds.get("atr_enabled",   True)
    vol_r   = thresholds.get("vol_spike_ratio", 2.0)
    mtf_on  = thresholds.get("mtf_enabled",   True)

    # 15mo：週線指標（MACD 26 週）與 12-1 動能（需 ≥253 根；1y 剛好卡在門檻上，對抗驗證 H1）
    if not quiet:
        print(f"Batch-downloading {len(tickers)} tickers (15mo)…")
    try:
        raw = yf.download(tickers, period="15mo", auto_adjust=True,
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
                if not quiet:
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
                **_extension(close, high, low),      # atr/ma20/ext_atr/ret_5d（引擎追高濾網）
            })
            flag = "🚨" if signals else "  "
            if quiet:
                continue
            print(f"{flag} {ticker}: ${price}  RSI={rsi}  chg={chg:+.1f}%  "
                  f"score={cs['score']:+.2f}({cs['rating']})  signals={len(signals)}")

        except Exception as exc:
            if not quiet:
                print(f"  {ticker}: error – {exc}")

    return results


# ── Protections (freqtrade-style) ─────────────────────────────────────────────


# ── 公開別名（外部呼叫走這些；底線名僅為 scan_signals 向後相容）──────────────
rsi = _rsi
macd = _macd
bollinger = _bollinger
composite_score = _composite_score
position_hint = _position_hint
extension = _extension


# ── 自我測試（合成 K 線；評分心臟首次有斷言）────────────────────────────────

if __name__ == "__main__":

    def _mk(px_path, vol=None, n=None):
        n = n or len(px_path)
        idx = pd.bdate_range("2025-01-01", periods=n)
        s = pd.Series(px_path, index=idx, dtype=float)
        v = pd.Series(vol if vol is not None else [1e6] * n, index=idx, dtype=float)
        return s, v

    rng = np.random.default_rng(3)
    up = 100 * np.cumprod(1 + rng.normal(0.004, 0.006, 260))     # 強升趨勢
    dn = 100 * np.cumprod(1 + rng.normal(-0.004, 0.006, 260))    # 強跌趨勢
    s_up, v_up = _mk(up)
    s_dn, v_dn = _mk(dn)

    # 1) RSI：連漲高檔、連跌低檔、常數不炸
    assert _rsi(s_up) > 55 and _rsi(s_dn) < 45
    flat, _ = _mk([100.0] * 60)
    assert 0 <= _rsi(flat) <= 100
    print("✅ 1 RSI 方向與邊界")

    # 2) MACD / 布林 / ATR 欄位契約（依實際回傳形狀）
    m = _macd(s_up)
    assert {"histogram", "signal"} <= set(m), m
    b = _bollinger(s_up)
    assert {"pct_b", "signal"} <= set(b) and 0 <= b["pct_b"] <= 1.5, b
    a = _atr_levels(s_up, s_up * 1.01, s_up * 0.99)
    assert {"atr", "signal"} <= set(a) and a["atr"] > 0, a
    print("✅ 2 MACD/布林/ATR 契約")

    # 3) 綜合評分：升趨勢顯著為正、跌趨勢顯著為負、含元件明細
    hi_u, lo_u = s_up * 1.01, s_up * 0.99
    hi_d, lo_d = s_dn * 1.01, s_dn * 0.99
    cs_u = _composite_score(s_up, hi_u, lo_u, v_up)
    cs_d = _composite_score(s_dn, hi_d, lo_d, v_dn)
    assert cs_u["score"] > 0.2, cs_u
    assert cs_d["score"] < -0.2, cs_d
    assert cs_u["score"] > cs_d["score"] + 0.5
    assert "components" in cs_u and cs_u["rating"]
    print(f"✅ 3 綜合評分（升 {cs_u['score']:+.2f} vs 跌 {cs_d['score']:+.2f}）")

    # 4) edge_weights 校準乘數：放大 MACD 應偏移分數且不出界
    cs_w = _composite_score(s_up, hi_u, lo_u, v_up,
                            edge_weights={"macd": 1.5, "rsi": 0.5})
    assert -1 <= cs_w["score"] <= 1 and cs_w["score"] != cs_u["score"]
    # mtf 分支不炸且出界防護
    cs_m = _composite_score(s_up, hi_u, lo_u, v_up, mtf=True)
    assert -1 <= cs_m["score"] <= 1
    print("✅ 4 edge_weights/mtf 分支")

    # 5) 部位提示：股數為正、停損低於現價、風險金額一致
    ph = _position_hint(s_up, hi_u, lo_u, float(s_up.iloc[-1]),
                        {"position_sizing_enabled": True, "account_size": 100000,
                         "risk_pct": 0.01, "atr_mult": 1.5})
    assert ph and ph["shares"] > 0 and ph["stop"] < float(s_up.iloc[-1]), ph
    # 無效價格 → None（sizing 開關在呼叫端 scan()，不在本函數）
    assert _position_hint(s_up, hi_u, lo_u, 0.0, {}) is None
    print("✅ 5 部位提示")

    # 6) 回測校準：鍵名與乘數範圍契約（不只「非空」）
    df = pd.DataFrame({"Close": s_up, "High": hi_u, "Low": lo_u, "Volume": v_up})
    cal = calibrate_ticker(df)
    assert isinstance(cal, dict) and cal, cal
    for _k, _v in cal.items():
        assert isinstance(_k, str), cal
        _m = _v.get("mult") if isinstance(_v, dict) else _v
        if isinstance(_m, (int, float)):
            assert 0 <= _m <= 5, (_k, _m)     # 校準乘數不得出現負值/爆表
    print("✅ 6 calibrate_ticker 契約（鍵名+乘數範圍）")

    # 7) 短序列/垃圾輸入不炸
    tiny, tv = _mk([100, 101, 99, 102, 98])
    cs_t = _composite_score(tiny, None, None, None)
    assert -1 <= cs_t["score"] <= 1
    print("✅ 7 短序列防炸")

    # 8) 延伸度：平穩上升趨勢 ext 小；末端跳空拉升 → ext_atr 顯著 > 2；短序列回 None
    ex_up = _extension(s_up, hi_u, lo_u)
    assert {"atr", "ma20", "ext_atr", "ret_5d"} <= set(ex_up) and ex_up["atr"] > 0
    spike = list(up[:200]) + [up[199] * (1 + 0.06 * (i + 1)) for i in range(3)]   # 3 天拉 +18%
    s_sp, _ = _mk(spike)
    ex_sp = _extension(s_sp, s_sp * 1.01, s_sp * 0.99)
    assert ex_sp["ext_atr"] > 2.0 and ex_sp["ret_5d"] > 0.10, ex_sp
    assert ex_sp["ext_atr"] > ex_up["ext_atr"]
    assert _extension(flat.iloc[:10], None, None)["ext_atr"] is None
    assert ex_up["vol_60"] is not None and ex_up["mom_12_1"] is not None and ex_up["ret_1m"] is not None
    assert abs(ex_up["mom_12_1"] - (float(s_up.iloc[-22]) / float(s_up.iloc[-253]) - 1)) < 1e-4
    print(f"✅ 8 延伸度（趨勢 ext {ex_up['ext_atr']:+.1f} ATR、噴出 {ex_sp['ext_atr']:+.1f} ATR）")

    # 9) composite_series 與逐日切片逐位相等（含 NaN 成交量、edge_weights、mtf）
    rng9 = np.random.default_rng(11)
    n9 = 320
    px9 = 100 * np.cumprod(1 + rng9.normal(0.0005, 0.02, n9))
    idx9 = pd.bdate_range("2025-01-06", periods=n9)
    c9 = pd.Series(px9, index=idx9)
    h9, l9 = c9 * (1 + np.abs(rng9.normal(0, 0.008, n9))), c9 * (1 - np.abs(rng9.normal(0, 0.008, n9)))
    v9 = pd.Series(rng9.integers(5e5, 5e6, n9).astype(float), index=idx9)
    v9.iloc[[40, 41, 150]] = np.nan
    v9.iloc[200] = v9.iloc[199] * 4                                  # 爆量日
    for ew, mt in [(None, False), (None, True), ({"macd": 1.4, "rsi": 0.7, "trend": 1.1}, True)]:
        vec = composite_series(c9, h9, l9, v9, edge_weights=ew, mtf=mt)
        worst = 0.0
        for i in range(20, n9):
            ref = _composite_score(c9.iloc[:i + 1], h9.iloc[:i + 1], l9.iloc[:i + 1], v9.iloc[:i + 1],
                                   edge_weights=ew, mtf=mt)["score"]
            worst = max(worst, abs(float(vec.iloc[i]) - ref))
            assert abs(float(vec.iloc[i]) - ref) < 1e-9, (i, ew, mt, vec.iloc[i], ref)
    assert (composite_series(c9, None, None, None) != 0).any()
    print(f"✅ 9 composite_series 逐位對齊逐日評分（300 日 × 3 組設定，最大差 {worst:.1e}）")

    print("\nindicators selftest OK ✅")
