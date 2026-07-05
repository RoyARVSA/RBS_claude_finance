"""
analyst_data.py – 分析師共識與財報 EPS surprise（學自 OpenBB 的免費資料面）

資料源：yfinance（analyst_price_targets / recommendations_summary /
upgrades_downgrades / earnings_dates），Finnhub 免費端點備援
（/stock/recommendation、/stock/earnings）。
彙總層為純函數（離線可測），抓取層需網路。分析教育用途，非投資建議。
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd


# ── 純彙總 ─────────────────────────────────────────────────────────────────────

def summarize_ratings(counts: dict) -> dict | None:
    """
    評等分佈 → 共識分數。counts 鍵：strongBuy/buy/hold/sell/strongSell。
    分數 = (2×強力買進 + 買進 − 賣出 − 2×強力賣出) / (2×總數)，範圍 -1~+1。
    """
    keys = ("strongBuy", "buy", "hold", "sell", "strongSell")
    vals = {k: int(counts.get(k) or 0) for k in keys}
    total = sum(vals.values())
    if total == 0:
        return None
    score = (2 * vals["strongBuy"] + vals["buy"]
             - vals["sell"] - 2 * vals["strongSell"]) / (2 * total)
    label = ("強力買進傾向" if score >= 0.5 else
             "買進傾向" if score >= 0.2 else
             "賣出傾向" if score <= -0.2 else "持有傾向")
    return {"total": total, "score": round(score, 2), "label": label, "dist": vals}


def summarize_targets(targets: dict, price=None) -> dict | None:
    """目標價共識 → 相對現價的上檔空間。targets 鍵：low/high/mean/median/current。"""
    if not targets:
        return None
    mean = targets.get("mean")
    cur = price if price is not None else targets.get("current")
    if mean is None or not cur:
        return None
    out = {"mean": float(mean), "current": float(cur),
           "upside_mean": float(mean) / float(cur) - 1}
    for k in ("low", "high", "median"):
        if targets.get(k) is not None:
            out[k] = float(targets[k])
    return out


def summarize_surprises(rows: list) -> dict | None:
    """
    EPS surprise 歷史 → beat 率。rows：[{"estimate": x, "actual": y}, ...]
    （actual 為 None 的未公布期自動略過）。
    """
    done = [r for r in rows
            if r.get("actual") is not None and r.get("estimate") is not None]
    if not done:
        return None
    beats = sum(1 for r in done if float(r["actual"]) > float(r["estimate"]))
    # 每筆 surprise 夾在 ±300%：接近零的預估值會產生 4900% 這類極端值灌爆平均
    sur = [max(-3.0, min(3.0,
           (float(r["actual"]) - float(r["estimate"])) / abs(float(r["estimate"]))))
           for r in done if float(r["estimate"]) != 0]
    return {"n": len(done), "beats": beats,
            "beat_rate": beats / len(done),
            "avg_surprise": float(np.mean(sur)) if sur else None,
            "rows": done[:8]}


# ── 抓取層（需網路）───────────────────────────────────────────────────────────

def _yf_analyst(ticker: str) -> dict:
    import yfinance as yf
    tk = yf.Ticker(ticker)
    out: dict = {}
    try:
        t = tk.analyst_price_targets
        if isinstance(t, dict) and t:
            out["targets"] = t
    except Exception:
        pass
    try:
        rs = tk.recommendations_summary
        if rs is not None and len(rs):
            row = rs.iloc[0]                      # period=0m（最新月）
            out["ratings"] = {k: row.get(k) for k in
                              ("strongBuy", "buy", "hold", "sell", "strongSell")}
    except Exception:
        pass
    try:
        ud = tk.upgrades_downgrades
        if ud is not None and len(ud):
            recent = ud.sort_index(ascending=False).head(6)
            out["upgrades"] = [
                {"date": str(idx.date() if hasattr(idx, "date") else idx),
                 "firm": r.get("Firm"), "to": r.get("ToGrade"),
                 "from": r.get("FromGrade"), "action": r.get("Action")}
                for idx, r in recent.iterrows()]
    except Exception:
        pass
    try:
        ed = tk.earnings_dates
        if ed is not None and len(ed):
            rows = []
            for idx, r in ed.iterrows():
                rows.append({"date": str(idx.date() if hasattr(idx, "date") else idx),
                             "estimate": _f(r.get("EPS Estimate")),
                             "actual": _f(r.get("Reported EPS"))})
            out["surprise_rows"] = rows
    except Exception:
        pass
    return out


def _f(x):
    try:
        v = float(x)
        return None if pd.isna(v) else v
    except (TypeError, ValueError):
        return None


def _finnhub_fallback(ticker: str, out: dict) -> dict:
    """yfinance 缺的欄位用 Finnhub 免費端點補。"""
    key = os.environ.get("FINNHUB_API_KEY", "")
    if not key:
        return out
    import requests
    base = "https://finnhub.io/api/v1"
    if "ratings" not in out:
        try:
            r = requests.get(f"{base}/stock/recommendation",
                             params={"symbol": ticker, "token": key}, timeout=15)
            data = r.json() if r.ok else []
            if data:
                d0 = data[0]
                out["ratings"] = {k: d0.get(k) for k in
                                  ("strongBuy", "buy", "hold", "sell", "strongSell")}
        except Exception:
            pass
    if "surprise_rows" not in out:
        try:
            r = requests.get(f"{base}/stock/earnings",
                             params={"symbol": ticker, "token": key}, timeout=15)
            data = r.json() if r.ok else []
            if isinstance(data, list) and data:
                out["surprise_rows"] = [
                    {"date": d.get("period"), "estimate": _f(d.get("estimate")),
                     "actual": _f(d.get("actual"))} for d in data]
        except Exception:
            pass
    return out


def fetch_analyst(ticker: str) -> dict:
    """完整抓取＋彙總。回 {targets, ratings, upgrades, surprises}（缺項為 None）。"""
    raw = _yf_analyst(ticker)
    raw = _finnhub_fallback(ticker, raw)
    return {
        "targets":   summarize_targets(raw.get("targets") or {}),
        "ratings":   summarize_ratings(raw.get("ratings") or {}),
        "upgrades":  raw.get("upgrades") or [],
        "surprises": summarize_surprises(raw.get("surprise_rows") or []),
    }


# ── CLI 自我測試（純彙總）─────────────────────────────────────────────────────

if __name__ == "__main__":
    r = summarize_ratings({"strongBuy": 10, "buy": 20, "hold": 8, "sell": 2, "strongSell": 0})
    print("ratings:", r)
    assert r["total"] == 40 and r["score"] == round((20 + 20 - 2) / 80, 2)
    assert "買進" in r["label"]
    assert summarize_ratings({}) is None

    t = summarize_targets({"low": 150, "high": 250, "mean": 200, "current": 180})
    print("targets:", t)
    assert abs(t["upside_mean"] - (200 / 180 - 1)) < 1e-9
    assert summarize_targets({}) is None
    assert summarize_targets({"mean": 100}) is None       # 無現價

    s = summarize_surprises([
        {"estimate": 1.0, "actual": 1.1},
        {"estimate": 2.0, "actual": 1.8},
        {"estimate": 1.5, "actual": 1.6},
        {"estimate": 1.2, "actual": None},                 # 未公布 → 略過
    ])
    print("surprises:", {k: v for k, v in s.items() if k != "rows"})
    assert s["n"] == 3 and s["beats"] == 2
    assert abs(s["beat_rate"] - 2 / 3) < 1e-9
    assert summarize_surprises([]) is None

    print("\n✅ analyst_data 純彙總測試通過")
