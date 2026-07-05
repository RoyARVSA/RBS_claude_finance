"""
reflection.py – AI 判斷的反思記憶（FinMem/FinAgent 文獻中證據最強的組件）

把「AI/訊號過去的方向判斷 vs 實際 N 日後結果」存進 bot 狀態檔，
彙總成命中率與近期失誤，回饋給晨報與 AI 助理的 context——
讓模型看得到自己最近錯在哪，而不是每次都自信滿滿地重來。

純邏輯（離線可測）；狀態存於 watchlist_state.json 的 "reflections" 鍵
（workflow 已會 commit 該檔，網頁端可讀取同一檔）。無 streamlit 依賴。
"""

from __future__ import annotations

import datetime as _dt

HORIZON_DAYS = 5      # 幾個「日曆日」後結算（近似一週交易日）
HISTORY_CAP = 60
PENDING_CAP = 30


def _refl(state: dict) -> dict:
    r = state.setdefault("reflections", {})
    r.setdefault("pending", [])
    r.setdefault("history", [])
    return r


def record_pick(state: dict, ticker: str, score: float, price: float,
                date: str) -> bool:
    """記錄一筆方向判斷（同日同代碼去重）。score 正=偏多、負=偏空。"""
    if not ticker or price is None or price <= 0:
        return False
    r = _refl(state)
    if any(p["ticker"] == ticker and p["date"] == date for p in r["pending"]):
        return False
    r["pending"].append({"ticker": ticker.upper(), "score": round(float(score), 2),
                         "price": float(price), "date": date})
    r["pending"] = r["pending"][-PENDING_CAP:]
    return True


def evaluate_pending(state: dict, prices: dict, today: str) -> int:
    """
    把滿 HORIZON_DAYS 的 pending 結算進 history。
    prices：{ticker: 現價}；抓不到價的先留著下次再算。回結算筆數。
    """
    r = _refl(state)
    today_d = _dt.date.fromisoformat(today)
    matured, keep = [], []
    for p in r["pending"]:
        try:
            age = (today_d - _dt.date.fromisoformat(p["date"])).days
        except ValueError:
            continue                      # 壞日期直接丟
        px = prices.get(p["ticker"])
        if age >= HORIZON_DAYS and px:
            fwd = float(px) / p["price"] - 1
            hit = (p["score"] > 0) == (fwd > 0) if p["score"] != 0 else None
            matured.append({**p, "fwd_ret": round(fwd, 4),
                            "hit": hit, "settled": today})
        elif age <= HORIZON_DAYS * 6:     # 太舊又一直抓不到價 → 放棄
            keep.append(p)
    r["pending"] = keep
    r["history"] = (r["history"] + matured)[-HISTORY_CAP:]
    return len(matured)


def summary_text(state: dict, n: int = 20) -> str | None:
    """近 n 次已結算判斷的命中率 + 最近兩筆失誤（給晨報/助理 context）。無資料回 None。"""
    hist = [h for h in _refl(state)["history"] if h.get("hit") is not None]
    if len(hist) < 3:
        return None
    recent = hist[-n:]
    hits = sum(1 for h in recent if h["hit"])
    rate = hits / len(recent)
    misses = [h for h in reversed(recent) if not h["hit"]][:2]
    parts = [f"AI 判斷回顧（近 {len(recent)} 次、{HORIZON_DAYS} 日後結算）：命中率 {rate:.0%}"]
    for m in misses:
        direction = "看多" if m["score"] > 0 else "看空"
        parts.append(f"最近失誤：{m['ticker']} {direction}（評分 {m['score']:+.2f}）"
                     f"→ 實際 {m['fwd_ret']:+.1%}")
    return "；".join(parts)


# ── CLI 自我測試 ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    state: dict = {}
    assert record_pick(state, "AAPL", 0.6, 200.0, "2026-06-20")
    assert not record_pick(state, "AAPL", 0.6, 200.0, "2026-06-20")   # 同日去重
    record_pick(state, "TSLA", -0.5, 300.0, "2026-06-20")
    record_pick(state, "NVDA", 0.7, 150.0, "2026-07-01")              # 未滿 5 天

    n = evaluate_pending(state, {"AAPL": 210.0, "TSLA": 310.0}, "2026-07-04")
    assert n == 2
    hist = state["reflections"]["history"]
    a = next(h for h in hist if h["ticker"] == "AAPL")
    t = next(h for h in hist if h["ticker"] == "TSLA")
    assert a["hit"] is True                     # 看多 +5% → 命中
    assert t["hit"] is False                    # 看空卻 +3.3% → 失誤
    assert len(state["reflections"]["pending"]) == 1   # NVDA 未到期

    # summary 需 ≥3 筆
    assert summary_text(state) is None
    record_pick(state, "AMD", 0.4, 100.0, "2026-06-25")
    evaluate_pending(state, {"AMD": 90.0}, "2026-07-04")
    s = summary_text(state)
    print(s)
    assert "命中率 33%" in s and "TSLA" in s and "AMD" in s

    # 抓不到價 → 留在 pending；太舊 → 放棄
    state2: dict = {}
    record_pick(state2, "OLD", 0.5, 10.0, "2026-01-01")
    record_pick(state2, "WAIT", 0.5, 10.0, "2026-07-01")
    evaluate_pending(state2, {}, "2026-07-04")
    tick = [p["ticker"] for p in state2["reflections"]["pending"]]
    assert "WAIT" in tick and "OLD" not in tick

    print("✅ reflection 純邏輯測試通過")
