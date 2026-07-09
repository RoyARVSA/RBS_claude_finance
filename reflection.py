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
                date: str, source: str = "quant",
                horizon: int | None = None) -> bool:
    """記錄一筆方向判斷（同日同代碼同來源去重）。score 正=偏多、負=偏空。
    source：quant=量化訊號、committee=委員會、day_plan=當日計畫…（計分板分組用）。
    horizon：此筆的結算天數；None 用預設 HORIZON_DAYS（當日計畫用 1）。"""
    if not ticker or price is None or price <= 0:
        return False
    ticker = ticker.upper()          # 先正規化再去重，小寫呼叫端才不會繞過
    r = _refl(state)
    if any(p["ticker"] == ticker and p["date"] == date
           and p.get("source", "quant") == source for p in r["pending"]):
        return False
    rec = {"ticker": ticker, "score": round(float(score), 2),
           "price": float(price), "date": date, "source": source}
    if horizon is not None:
        rec["horizon"] = max(1, int(horizon))
    r["pending"].append(rec)
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
        hz = max(1, int(p.get("horizon", HORIZON_DAYS)))
        if age >= hz and px:
            fwd = float(px) / p["price"] - 1
            hit = (p["score"] > 0) == (fwd > 0) if p["score"] != 0 else None
            matured.append({**p, "fwd_ret": round(fwd, 4),
                            "hit": hit, "settled": today})
        elif age <= max(hz * 6, 10):      # 太舊又一直抓不到價 → 放棄
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
    parts = [f"AI 判斷回顧（近 {len(recent)} 次、各筆到期結算）：命中率 {rate:.0%}"]
    for m in misses:
        direction = "看多" if m["score"] > 0 else "看空"
        parts.append(f"最近失誤：{m['ticker']} {direction}（評分 {m['score']:+.2f}）"
                     f"→ 實際 {m['fwd_ret']:+.1%}")
    return "；".join(parts)


def scoreboard(history: list) -> list[dict]:
    """
    決策者計分板（純函數）。history：已結算清單（可混多來源）。
    回 [{source, n, hits, hit_rate, avg_fwd}]，依 hit_rate 排序；無資料回 []。
    """
    by_src: dict = {}
    for h in history or []:
        if h.get("hit") is None:
            continue
        s = h.get("source", "quant")
        g = by_src.setdefault(s, {"n": 0, "hits": 0, "fwd": []})
        g["n"] += 1
        g["hits"] += 1 if h["hit"] else 0
        # 統一成「方向對齊報酬」：看多時 fwd、看空時 -fwd
        try:
            aligned = h["fwd_ret"] * (1 if h["score"] > 0 else -1)
            g["fwd"].append(aligned)
        except (KeyError, TypeError):
            pass
    out = []
    for s, g in by_src.items():
        out.append({"source": s, "n": g["n"], "hits": g["hits"],
                    "hit_rate": g["hits"] / g["n"] if g["n"] else None,
                    "avg_fwd": (sum(g["fwd"]) / len(g["fwd"])) if g["fwd"] else None})
    out.sort(key=lambda x: -(x["hit_rate"] or 0))
    return out


SOURCE_LABELS = {"quant": "量化訊號", "committee": "委員會", "analyst": "AI 分析師",
                 "day_plan": "當日計畫"}


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

    # 計分板：兩來源分組、方向對齊報酬
    hist = [
        {"ticker": "A", "score": 0.6, "fwd_ret": 0.05, "hit": True,  "source": "quant"},
        {"ticker": "B", "score": -0.5, "fwd_ret": 0.03, "hit": False, "source": "quant"},
        {"ticker": "C", "score": 0.8, "fwd_ret": 0.04, "hit": True,  "source": "committee"},
        {"ticker": "D", "score": 0.8, "fwd_ret": -0.02, "hit": False, "source": "committee"},
        {"ticker": "E", "score": 0.8, "fwd_ret": 0.06, "hit": True,  "source": "committee"},
        {"ticker": "F", "score": 0.1, "fwd_ret": 0.01, "hit": None},          # 未結算略過
    ]
    sb = scoreboard(hist)
    cm = next(x for x in sb if x["source"] == "committee")
    qt = next(x for x in sb if x["source"] == "quant")
    assert cm["n"] == 3 and abs(cm["hit_rate"] - 2 / 3) < 1e-9
    assert qt["n"] == 2 and abs(qt["hit_rate"] - 0.5) < 1e-9
    assert abs(qt["avg_fwd"] - (0.05 + (-0.03)) / 2) < 1e-9    # 看空 B 的 +3% → 對齊 -3%
    assert sb[0]["source"] == "committee"                       # 命中率排序
    # 同日同代碼不同來源 → 不互相去重
    st2: dict = {}
    assert record_pick(st2, "NVDA", 0.8, 100, "2026-07-05", source="quant")
    assert record_pick(st2, "NVDA", 0.8, 100, "2026-07-05", source="committee")
    assert not record_pick(st2, "NVDA", 0.8, 100, "2026-07-05", source="committee")
    print("scoreboard OK:", [(x['source'], round(x['hit_rate'], 2)) for x in sb])

    # 自訂 horizon：day_plan 隔日即結算、預設 5 日的同日紀錄不受影響
    st3: dict = {}
    assert record_pick(st3, "SPY", 0.8, 500.0, "2026-07-06", source="day_plan", horizon=1)
    assert record_pick(st3, "SPY", 0.8, 500.0, "2026-07-06", source="quant")   # 預設 5 日
    n3 = evaluate_pending(st3, {"SPY": 505.0}, "2026-07-07")
    assert n3 == 1, n3                                          # 只有 day_plan 到期
    h3 = st3["reflections"]["history"][0]
    assert h3["source"] == "day_plan" and h3["hit"] is True, h3
    assert len(st3["reflections"]["pending"]) == 1              # quant 那筆還在等

    print("✅ reflection 純邏輯測試通過")
