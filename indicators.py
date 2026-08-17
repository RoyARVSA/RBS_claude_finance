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


# ── 公開別名（外部呼叫走這些；底線名僅為 scan_signals 向後相容）──────────────
rsi = _rsi
macd = _macd
bollinger = _bollinger
composite_score = _composite_score
position_hint = _position_hint


# ── 自我測試（合成 K 線；評分心臟首次有斷言）────────────────────────────────

if __name__ == "__main__":
    import numpy as np

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

    # 6) 回測校準：合成資料出勝率結構、鍵齊全
    df = pd.DataFrame({"Close": s_up, "High": hi_u, "Low": lo_u, "Volume": v_up})
    cal = calibrate_ticker(df)
    assert isinstance(cal, dict) and cal, cal
    print("✅ 6 calibrate_ticker 契約")

    # 7) 短序列/垃圾輸入不炸
    tiny, tv = _mk([100, 101, 99, 102, 98])
    cs_t = _composite_score(tiny, None, None, None)
    assert -1 <= cs_t["score"] <= 1
    print("✅ 7 短序列防炸")

    print("\nindicators selftest OK ✅")
