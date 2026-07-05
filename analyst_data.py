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


# ── 新聞逐篇多空標注（學自 TradingAgents 情緒分析師的輸出形式）────────────────

def build_news_tag_prompt(ticker: str, headlines: list[str]) -> str:
    """組出「逐篇標注利多/利空」的 LLM 提示。headlines 最多取 8 則。"""
    items = "\n".join(f"{i + 1}. {h}" for i, h in enumerate(headlines[:8]))
    return (f"你是新聞分析師。逐篇判斷下列新聞標題對 {ticker} 的方向性影響。\n"
            "每則輸出一行，格式嚴格為：「<編號>. [利多|利空|中性] <15字內理由>」。\n"
            "只看標題資訊，不得腦補內文；與該標的無關的標題標為中性。\n\n" + items)


def parse_news_tags(text: str, n: int) -> list[dict]:
    """解析標注輸出 → [{"i", "tag", "reason"}]（i 為 0-based，超出 n 或格式壞的行略過）。"""
    import re as _re
    out = []
    for m in _re.finditer(r"(\d+)\s*[.、]\s*[\[【（(]?(利多|利空|中性)[\]】）)]?\s*(.*)",
                          text or ""):
        i = int(m.group(1)) - 1
        if 0 <= i < n:
            out.append({"i": i, "tag": m.group(2), "reason": m.group(3).strip()[:40]})
    return out


def format_tagged_news(headlines: list[str], tags: list[dict]) -> str:
    """把標注結果組回「[利多] 標題 — 理由」清單文字（給分析師 context / 顯示用）。"""
    tag_map = {t["i"]: t for t in tags}
    lines = []
    for i, h in enumerate(headlines[:8]):
        t = tag_map.get(i)
        if t:
            lines.append(f"[{t['tag']}] {h}" + (f" — {t['reason']}" if t["reason"] else ""))
        else:
            lines.append(f"[未標注] {h}")
    return "\n".join(lines)


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

    heads = ["Foxconn Q2 revenue jumps 39.8% on AI demand",
             "Michael Burry bets against Nvidia and Micron",
             "Fed holds rates steady"]
    p = build_news_tag_prompt("NVDA", heads)
    assert "1. Foxconn" in p and "利多|利空|中性" in p
    tags = parse_news_tags("1. [利多] AI需求驗證\n2. [利空] 知名空頭做空\n3. [中性] 與個股無直接關係\n9. [利多] 超出範圍", 3)
    assert len(tags) == 3 and tags[0]["tag"] == "利多" and tags[1]["tag"] == "利空"
    ftxt = format_tagged_news(heads, tags)
    assert "[利多] Foxconn" in ftxt and "[利空] Michael Burry" in ftxt
    # 容錯：全形括號與頓號
    t2 = parse_news_tags("1、【利空】財報不如預期", 1)
    assert t2 and t2[0]["tag"] == "利空"
    assert parse_news_tags("亂七八糟沒格式", 3) == []
    print("news tagging OK")
    print("\n✅ analyst_data 純彙總測試通過")
