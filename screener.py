"""
screener.py – 候選篩選（估值層 P5：選股池 + 主題層 → 進 watchlist 的候選）

候選池 = 選股池快照的動能前 N（universe.top）∪ stock_db AI 供應鏈瓶頸主題 − 現有 watchlist。
Stage 3（限額、有網路時）：每檔抓 PIT 三表 → 品質分（quality）、yfinance 預估快照 → 修正動能、
加上選股池的 12-1 動能與距 52 週高。綜合分 = 品質 0.4 + 修正動能 0.3 + 動能 0.3（缺成分只降信心）。
輸出排名與理由；**只建議、不自動加 watchlist**——`/add` 之後才會進建模輪替與佈局計畫。
純邏輯離線可測；stage3 需網路。教育用途，非投資建議。
"""

from __future__ import annotations

import math
from datetime import datetime

DEFAULTS = {"screen_max_fetch": 8, "screen_ttl_days": 7, "screen_top": 12}
W = {"quality": 0.4, "rev": 0.3, "mom": 0.3}


def _f(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def candidate_pool(universe: dict | None, themes: dict | None, watchlist: list[str], extra: list[str] | None = None) -> list[dict]:
    """去除 watchlist 後的候選（帶來源標記與選股池動能欄位）。"""
    wl = {t.upper() for t in (watchlist or [])}
    pool: dict[str, dict] = {}
    for r in ((universe or {}).get("top") or []):
        t = str(r.get("t") or "").upper()
        if t and t not in wl:
            pool[t] = {"ticker": t, "src": ["universe"], "mom": _f(r.get("mom")), "off_high": _f(r.get("oh")),
                       "rank": r.get("rank"), "sector": r.get("s")}
    for name, syms in (themes or {}).items():
        for t in syms:
            t = str(t).upper()
            if t in wl or "." in t:                    # 主題內非美股（IQE.L / SOI.PA）略過
                continue
            pool.setdefault(t, {"ticker": t, "src": [], "mom": None, "off_high": None, "rank": None, "sector": None})
            pool[t]["src"].append(f"主題:{name}")
    for t in (extra or []):
        t = str(t).upper()
        if t and t not in wl:
            pool.setdefault(t, {"ticker": t, "src": ["manual"], "mom": None, "off_high": None, "rank": None, "sector": None})
    return list(pool.values())


def score_row(row: dict) -> dict:
    """綜合分（-1..1）與信心（可得成分比例）。"""
    parts, ws = [], []
    q = row.get("quality_score")
    if q is not None:
        parts.append(max(-1.0, min(1.0, q))); ws.append(W["quality"])
    r = row.get("rev_score")
    if r is not None:
        parts.append(max(-1.0, min(1.0, r))); ws.append(W["rev"])
    m = row.get("mom")
    if m is not None:
        parts.append(max(-1.0, min(1.0, m / 0.5))); ws.append(W["mom"])     # 12-1 動能 +50% → 滿分
    score = (sum(p * w for p, w in zip(parts, ws)) / sum(ws)) if ws else None
    row["score"] = score
    row["confidence"] = len(ws) / 3.0
    return row


def rank_rows(rows: list[dict], top: int = 12) -> list[dict]:
    """品質否決者剔除；依 score×confidence 排序。"""
    ok = [score_row(dict(r)) for r in rows if not r.get("veto")]
    ok = [r for r in ok if r.get("score") is not None]
    ok.sort(key=lambda r: -(r["score"] * (0.5 + 0.5 * r["confidence"])))
    for i, r in enumerate(ok[:top], 1):
        r["rank_out"] = i
    return ok[:top]


def screen_text(res: dict) -> str:
    rows = res.get("rows") or []
    lines = [f"🔎 *候選篩選*（{res.get('as_of', '')}；池 {res.get('pool', 0)} 檔、已分析 {res.get('analyzed', 0)}、"
             f"品質否決 {res.get('vetoed', 0)}）"]
    if not rows:
        lines.append("尚無可排名候選（選股池未建或 Stage 3 尚未累積；`/universe rebuild` 後再試）")
    for r in rows:
        bits = []
        if r.get("quality_score") is not None:
            bits.append(f"質 {r['quality_score']:+.2f}")
        if r.get("rev_score") is not None:
            bits.append(f"修正 {r['rev_score']:+.2f}")
        if r.get("mom") is not None:
            bits.append(f"動能 {r['mom']:+.0%}")
        if r.get("off_high") is not None:
            bits.append(f"距高 {r['off_high']:+.0%}")
        src = "、".join(x.replace("_", "·") for x in r.get("src") or [])
        flag = ("⚠️" + "、".join(str(x).replace("_", "·") for x in r["flags"]) + " ") if r.get("flags") else ""
        lines.append(f"{r.get('rank_out', 0)}. {r['ticker']} 綜合 {r['score']:+.2f}（成分 {int(r['confidence'] * 3)}/3）"
                     f"｜{'｜'.join(bits)} {flag}[{src}]")
    lines.append("只建議不自動加入：`/add TICKER` 後才進建模輪替與佈局計畫；非投資建議")
    return "\n".join(lines)


# ── Stage 3 抓取（需網路；限額）────────────────────────────────────────────

def stage3_fetch(ticker: str, today: str) -> dict:
    """單檔：品質（PIT 三表）+ 修正動能（yfinance 預估快照）。任何失敗只缺該欄。"""
    out = {}
    try:
        import fin_data as fd
        import quality as ql
        store = fd.get_financials(ticker, today)
        periods = fd.pit_view(store, None, "A")
        if len(periods) >= 2:
            q = ql.quality_summary(periods, None, None)
            out.update({"quality_score": q.get("score"), "veto": q.get("veto"), "flags": q.get("flags", [])})
    except Exception:
        pass
    try:
        import estimates_ledger as el
        snap = el.fetch_yf_snapshot(ticker)
        if snap:
            m = el.revision_momentum({"latest": snap, "rows": [], "ts": today})
            if m and m.get("score") is not None:
                out["rev_score"] = m["score"]
    except Exception:
        pass
    return out


def run_screen(state: dict, today: str, universe: dict | None, themes: dict | None, cfg: dict | None = None,
               fetch=None, force: bool = False) -> dict:
    """
    候選池 → 對「未分析或過期」者最舊優先做 Stage 3（≤ screen_max_fetch 檔）→ 合併快取 → 排名。
    快取 state["screen"] = {"as_of", "cache": {t: {...,"ts"}}}（公開資料衍生，明文）。
    """
    c = {**DEFAULTS, **(cfg or {})}
    sc = state.setdefault("screen", {"as_of": None, "cache": {}})
    cache = sc.setdefault("cache", {})
    pool = candidate_pool(universe, themes, state.get("watchlist") or [])
    pool_t = {r["ticker"] for r in pool}
    for k in [k for k in cache if k not in pool_t]:          # 離開候選池的清掉
        del cache[k]
    stale = []
    for r in pool:
        ent = cache.get(r["ticker"])
        try:
            age = (datetime.strptime(today, "%Y-%m-%d") - datetime.strptime(str(ent.get("ts")), "%Y-%m-%d")).days if ent else 1e9
        except Exception:
            age = 1e9
        if force or age >= int(c["screen_ttl_days"]):
            stale.append((r["ticker"], age))
    stale.sort(key=lambda x: -x[1])
    picked = [t for t, _ in stale[: int(c["screen_max_fetch"])]]
    f = fetch or stage3_fetch
    for t in picked:
        try:
            got = f(t, today) or {}
        except Exception:
            got = {}
        cache[t] = {**got, "ts": today}
    rows = []
    for r in pool:
        rows.append({**r, **{k: v for k, v in (cache.get(r["ticker"]) or {}).items() if k != "ts"}})
    ranked = rank_rows(rows, int(c["screen_top"]))
    sc["as_of"] = today
    return {"as_of": today, "pool": len(pool), "analyzed": sum(1 for r in pool if r["ticker"] in cache),
            "vetoed": sum(1 for r in rows if r.get("veto")), "fetched": picked, "rows": ranked}


def should_refresh(state: dict, today: str, cfg: dict | None = None) -> bool:
    c = {**DEFAULTS, **(cfg or {})}
    last = (state.get("screen") or {}).get("as_of")
    try:
        return not last or (datetime.strptime(today, "%Y-%m-%d") - datetime.strptime(str(last), "%Y-%m-%d")).days >= int(c["screen_ttl_days"])
    except Exception:
        return True


if __name__ == "__main__":
    uni = {"top": [{"t": "AAA", "rank": 1, "mom": 0.6, "oh": -0.05, "s": "Technology"},
                   {"t": "BBB", "rank": 2, "mom": 0.3, "oh": -0.2, "s": "Industrials"},
                   {"t": "WL1", "rank": 3, "mom": 0.9}]}
    themes = {"AI電力/散熱": ["VRT", "BBB", "IQE.L"], "HBM/記憶儲存": ["MU"]}
    pool = candidate_pool(uni, themes, ["WL1", "vrt"])
    by = {r["ticker"]: r for r in pool}
    assert set(by) == {"AAA", "BBB", "MU"} and by["BBB"]["src"] == ["universe", "主題:AI電力/散熱"] and by["MU"]["mom"] is None
    print("✅ 1 候選池（去 watchlist、主題合併、非美股略過）")

    calls = []

    def fake_fetch(t, today):
        calls.append(t)
        return {"AAA": {"quality_score": 0.7, "veto": False, "flags": [], "rev_score": 0.5},
                "BBB": {"quality_score": -0.9, "veto": True, "flags": ["beneish_high"]},
                "MU": {"quality_score": 0.2}}.get(t, {})
    st = {"watchlist": ["WL1", "VRT"]}
    res = run_screen(st, "2026-09-09", uni, themes, {"screen_max_fetch": 2}, fetch=fake_fetch)
    assert calls == ["AAA", "BBB"] and res["analyzed"] == 2 and res["vetoed"] == 1
    assert [r["ticker"] for r in res["rows"]] == ["AAA"]                                # BBB 否決；MU 未抓且無動能 → 無分數不入榜
    # 第二輪輪到 MU
    calls.clear()
    res2 = run_screen(st, "2026-09-09", uni, themes, {"screen_max_fetch": 2}, fetch=fake_fetch)
    assert calls == ["MU"] and [r["ticker"] for r in res2["rows"]][0] == "AAA" and any(r["ticker"] == "MU" for r in res2["rows"])
    assert res2["rows"][0]["confidence"] == 1.0 and res2["rows"][0]["score"] > 0.5
    assert should_refresh(st, "2026-09-10") is False and should_refresh(st, "2026-09-20") is True
    # 離開候選池的快取被清；watchlist 新增後不再是候選
    st["watchlist"].append("AAA")
    res3 = run_screen(st, "2026-09-20", uni, themes, {"screen_max_fetch": 0}, fetch=fake_fetch)
    assert "AAA" not in st["screen"]["cache"] and all(r["ticker"] != "AAA" for r in res3["rows"])
    print("✅ 2 Stage 3 限額輪替 / 否決剔除 / 快取清理")
    t = screen_text(res2)
    assert t.count("*") % 2 == 0 and "_" not in t.replace("`/add TICKER`", "").replace("`/universe rebuild`", "")
    print(t)
    print("\nscreener selftest OK ✅")
