"""
sector_scan.py – 產業類別總覽掃描（可重用，無 Streamlit 依賴）

一次掃描整個市場/產業的所有標的，用產業為單位看強弱輪動與風險分佈。
資料源為 stock_db 的產業→標的清單。

設計：
  • 純邏輯（price_metrics / aggregate_by_industry）可離線單元測試
  • 抓取層 scan_universe 用「批次 + 並行」下載，基本面（P/E,ROE）並行且可選
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── 純邏輯（離線可測）──────────────────────────────────────────────────────────

def price_metrics(close: pd.Series) -> dict | None:
    """從收盤序列算價格/風險指標（純函數）。資料不足回 None。"""
    if close is None:
        return None
    close = close.dropna()
    if len(close) < 20:
        return None
    price = float(close.iloc[-1])
    r = close.pct_change().dropna()

    def _ret(n):
        return float(close.iloc[-1] / close.iloc[-n] - 1) if len(close) >= n else None

    ann_vol = float(r.std() * np.sqrt(252)) if len(r) > 1 else None
    sharpe = float(r.mean() / r.std() * np.sqrt(252)) if (len(r) > 1 and r.std() > 0) else None
    max_dd = float((close / close.cummax() - 1).min())
    # RSI(14)
    d = close.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / 14, min_periods=14).mean().iloc[-1]
    loss = (-d).clip(lower=0).ewm(alpha=1 / 14, min_periods=14).mean().iloc[-1]
    rsi = 100.0 if (loss == 0 or pd.isna(loss)) else float(100 - 100 / (1 + gain / loss))

    return {
        "price": round(price, 2),
        "return_1m": _ret(22),
        "return_3m": _ret(63),
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "rsi": round(rsi, 1) if not pd.isna(rsi) else None,
    }


def aggregate_by_industry(rows: list[dict]) -> list[dict]:
    """把個股列（含 industry 欄）依產業彙總平均指標（純函數）。"""
    from collections import defaultdict
    groups: dict[str, list] = defaultdict(list)
    for r in rows:
        groups[r.get("industry", "?")].append(r)

    def _mean(items, key):
        vals = [i[key] for i in items
                if i.get(key) is not None and not (isinstance(i[key], float) and np.isnan(i[key]))]
        return sum(vals) / len(vals) if vals else None

    out = []
    for ind, items in groups.items():
        out.append({
            "industry": ind,
            "count": len(items),
            "return_1m": _mean(items, "return_1m"),
            "return_3m": _mean(items, "return_3m"),
            "ann_vol":   _mean(items, "ann_vol"),
            "sharpe":    _mean(items, "sharpe"),
        })
    out.sort(key=lambda x: (x["return_3m"] if x["return_3m"] is not None else -9), reverse=True)
    return out


# ── 抓取層（需網路；此環境代理擋 yfinance，需部署後實測）────────────────────────

def _batch_closes(tickers: list[str], period: str, min_len: int = 20) -> dict:
    """批次下載多檔收盤序列，穩健處理 MultiIndex 兩種版面。
    全專案共用的唯一實作（app.py 市場快照/篩選器皆 import 此函數）。"""
    import yfinance as yf
    try:
        raw = yf.download(tickers, period=period, auto_adjust=True,
                          progress=False, threads=True)
    except Exception:
        return {}
    if raw is None or raw.empty:
        return {}
    out = {}
    is_multi = isinstance(raw.columns, pd.MultiIndex)
    for tkr in tickers:
        try:
            if is_multi:
                if ("Close", tkr) in raw.columns:
                    s = raw[("Close", tkr)].dropna()
                elif (tkr, "Close") in raw.columns:
                    s = raw[(tkr, "Close")].dropna()
                else:
                    continue
            else:
                if len(tickers) != 1:   # 平面欄位只在單檔時才對應該 ticker
                    continue
                s = raw["Close"].squeeze().dropna()
            if len(s) >= min_len:
                out[tkr] = s
        except Exception:
            continue
    return out


def scan_universe(industry_map: dict, period: str = "6mo",
                  with_fundamentals: bool = False) -> tuple[list[dict], list[dict]]:
    """
    掃描 {產業: [代碼,...]} 全部標的。
    回傳 (個股列 rows, 產業彙總 industry_rows)。
    with_fundamentals=True 時並行抓 P/E、ROE（較慢）。
    """
    # 反查表：ticker → industry（同股可能屬多產業，取第一個）
    tkr_industry, all_tickers = {}, []
    for ind, tkrs in industry_map.items():
        for t in tkrs:
            if t not in tkr_industry:
                tkr_industry[t] = ind
                all_tickers.append(t)

    closes = _batch_closes(all_tickers, period)

    # 基本面（可選、並行）
    fund = {}
    if with_fundamentals and all_tickers:
        try:
            import fundamentals as fa
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=8) as ex:
                results = ex.map(lambda t: (t, fa.quick_valuation(t)), list(closes.keys()))
                fund = dict(results)
        except Exception:
            fund = {}

    rows = []
    for tkr in all_tickers:
        s = closes.get(tkr)
        m = price_metrics(s)
        if m is None:
            continue
        m["ticker"] = tkr
        m["industry"] = tkr_industry.get(tkr, "?")
        if with_fundamentals:
            fv = fund.get(tkr, {})
            m["pe"] = fv.get("pe")
            m["roe"] = fv.get("roe")
        rows.append(m)

    industry_rows = aggregate_by_industry(rows)
    return rows, industry_rows


# ── CLI 自我測試（純邏輯）──────────────────────────────────────────────────────

if __name__ == "__main__":
    idx = pd.date_range("2024-01-01", periods=80, freq="B")
    up = pd.Series(np.linspace(100, 130, 80), index=idx)
    down = pd.Series(np.linspace(130, 110, 80), index=idx)

    print("=== price_metrics（上升）===")
    print(" ", {k: (round(v, 3) if isinstance(v, float) else v)
                 for k, v in price_metrics(up).items()})
    print("=== price_metrics（下降）===")
    print(" ", {k: (round(v, 3) if isinstance(v, float) else v)
                 for k, v in price_metrics(down).items()})
    print("=== price_metrics（資料不足）===")
    print(" ", price_metrics(up.iloc[:10]))

    print("\n=== aggregate_by_industry ===")
    rows = [
        {"ticker": "A", "industry": "半導體", "return_1m": 0.05, "return_3m": 0.12, "ann_vol": 0.3, "sharpe": 1.2},
        {"ticker": "B", "industry": "半導體", "return_1m": 0.03, "return_3m": 0.08, "ann_vol": 0.35, "sharpe": 0.9},
        {"ticker": "C", "industry": "金融", "return_1m": -0.01, "return_3m": 0.02, "ann_vol": 0.18, "sharpe": 0.3},
    ]
    for a in aggregate_by_industry(rows):
        print(" ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in a.items()})
