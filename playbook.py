"""
playbook.py – 佈局計畫整合層（純邏輯、離線可測）

網頁上各功能是分頁、Bot 上各功能是指令；本模組把它們串成**一份**佈局計畫：
  選股池（universe）→ 分析師修正動能（estimates_ledger）→ 公司模型（val_hist：MoS/區間位置/判定）
  → 品質與會計旗標（quality on data/fin）→ 技術評分（last_scores）→ 大盤 regime（weather）
  → 現有持倉（engine.pos）→ 論點（theses）
每檔輸出：四象限（估值/論點 × 價格/動能）、層級（迴避／減碼／累積候選／持有／觀察）、
conviction（0–1，只由可得的成分構成並附成分數）、權重帶（有界；依 regime 打折；單檔 ≤10%、主題 ≤25%）。
組合層：各層級人數、現金目標（regime）、主題集中、需要更新的模型。

鐵律：這是**參考**，不下單、不改引擎參數；進場仍由技術訊號過門檻、出場仍由價格機制；
估值層預設關閉、未過 walk-forward holdout 不接引擎（VALUATION_PLAN §3）。教育用途，非投資建議。
"""

from __future__ import annotations

import math
from datetime import datetime

DEFAULTS = {
    "base_weight": 0.05,          # 一檔的基準權重
    "max_single": 0.10,           # 單檔硬上限
    "max_theme": 0.25,            # 主題硬上限
    "stale_days": 30,             # 模型多久算過期
    "tech_buy": 0.5,              # 技術評分進場門檻（與引擎 buy_threshold 一致）
    "cash_target": {"risk_on": 0.10, "neutral": 0.25, "risk_off": 0.40, None: 0.20},
    "regime_mult": {"risk_on": 1.0, "neutral": 0.8, "risk_off": 0.5, None: 0.8},
    "weights": {"val": 0.35, "quality": 0.25, "rev": 0.20, "tech": 0.20},
}
TIER_ORDER = ("迴避", "減碼", "累積候選", "持有", "觀察")
TIER_EMOJI = {"迴避": "🚫", "減碼": "🟠", "累積候選": "🟢", "持有": "🔵", "觀察": "⚪"}


def _f(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _clip(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))


def _days(a: str | None, b: str) -> int | None:
    try:
        return (datetime.strptime(b[:10], "%Y-%m-%d") - datetime.strptime(str(a)[:10], "%Y-%m-%d")).days
    except Exception:
        return None


# ── 1. 彙整每檔輸入 ─────────────────────────────────────────────────────────

def collect_row(ticker: str, state: dict, today: str, ledger: dict | None = None,
                quality: dict | None = None, universe: dict | None = None, cfg: dict | None = None) -> dict:
    """把散落各處的資料收成一列（缺什麼就 None，不臆測）。"""
    c = {**DEFAULTS, **(cfg or {})}
    row = {"ticker": ticker, "components": []}
    # 技術評分
    ls = (state.get("last_scores") or {}).get(ticker) or {}
    row["tech"] = _f(ls.get("score"))
    row["price"] = _f(ls.get("price"))
    if row["tech"] is not None:
        row["components"].append("tech")
    # 持倉（引擎簿記）
    pos = ((state.get("engine") or {}).get("pos") or {}).get(ticker)
    row["held"] = bool(pos)
    row["held_days"] = _days(pos.get("opened"), today) if pos else None
    row["scaled_out"] = bool(pos and pos.get("scaled_out"))
    # 估值（val_hist 最新）
    vh = ((state.get("val_hist") or {}).get(ticker) or [])
    v = vh[-1] if vh else None
    if v and v.get("base"):
        age = _days(v.get("d"), today)
        row["val"] = {"as_of": v.get("d"), "base": _f(v.get("base")), "bear": _f(v.get("bear")),
                      "bull": _f(v.get("bull")), "mos": _f(v.get("mos")), "verdict": v.get("verdict"),
                      "method": v.get("method"), "age_days": age, "stale": (age is None or age > int(c["stale_days"]))}
        px = row["price"] or _f(v.get("px"))
        b, u = row["val"]["bear"], row["val"]["bull"]
        row["val"]["range_pos"] = ((px - b) / (u - b)) if (px and b is not None and u is not None and u > b) else None
        if row["val"]["mos"] is None and px and row["val"]["base"]:
            row["val"]["mos"] = row["val"]["base"] / px - 1
        row["components"].append("val")
    else:
        row["val"] = None
    # 品質
    row["quality"] = quality if (quality and quality.get("score") is not None) else None
    if row["quality"]:
        row["components"].append("quality")
    # 修正動能
    row["rev"] = None
    try:
        import estimates_ledger as el
        ent = ((ledger or {}).get("tickers") or {}).get(ticker)
        m = el.revision_momentum(ent, row["price"]) if ent else None
        if m and m.get("score") is not None:
            row["rev"] = {"score": m["score"], "breadth": m.get("breadth_30d"), "chg90": m.get("eps_chg_90d"),
                          "tgt_upside": m.get("tgt_upside")}
            row["components"].append("rev")
    except Exception:
        pass
    # 選股池
    top = (universe or {}).get("top") or []
    rk = next((r.get("rank") for r in top if r.get("t") == ticker), None)
    row["universe_rank"] = rk
    row["in_universe"] = rk is not None or ticker in {r.get("t") for r in ((universe or {}).get("broad") or [])}
    # 論點
    th = ((state.get("theses") or {}).get(ticker) or {})
    row["thesis"] = {"direction": th.get("direction"), "conviction": th.get("conviction"), "stop": _f(th.get("stop")),
                     "status": th.get("status")} if th else None
    return row


# ── 2. 分類與 conviction ────────────────────────────────────────────────────

def conviction(row: dict, cfg: dict | None = None) -> tuple[float | None, float]:
    """回 (conviction 0–1 | None, confidence 0–1=可得成分比例)。各成分先映到 [-1,1] 再加權。"""
    c = {**DEFAULTS, **(cfg or {})}
    w = c["weights"]
    parts, ws = [], []
    if row.get("val") and row["val"].get("mos") is not None:
        parts.append(_clip(row["val"]["mos"] / 0.5)); ws.append(w["val"])
    if row.get("quality"):
        parts.append(_clip(row["quality"]["score"])); ws.append(w["quality"])
    if row.get("rev"):
        parts.append(_clip(row["rev"]["score"])); ws.append(w["rev"])
    if row.get("tech") is not None:
        parts.append(_clip(row["tech"])); ws.append(w["tech"])
    if not ws:
        return None, 0.0
    c01 = (sum(p * ww for p, ww in zip(parts, ws)) / sum(ws) + 1) / 2
    return c01, len(ws) / 4.0


def quadrant(row: dict) -> str:
    """估值/論點（對＝MoS ≥ 0 或 accumulate）× 價格/動能（對＝技術評分 ≥ 0 且修正動能 ≥ 0）。"""
    v = row.get("val") or {}
    thesis_ok = None
    if v.get("mos") is not None:
        thesis_ok = v["mos"] >= 0 and v.get("verdict") not in ("exit", "trim")
    mom = []
    if row.get("tech") is not None:
        mom.append(row["tech"] >= 0)
    if row.get("rev"):
        mom.append(row["rev"]["score"] >= 0)
    price_ok = (all(mom) if mom else None)
    if thesis_ok is None or price_ok is None:
        return "資料不足"
    return {(True, True): "論點對·價格對", (True, False): "論點對·價格錯", (False, True): "論點錯·價格對",
            (False, False): "論點錯·價格錯"}[(thesis_ok, price_ok)]


def classify(row: dict, regime: str | None, cfg: dict | None = None) -> dict:
    """層級 + 權重帶 + 理由。"""
    c = {**DEFAULTS, **(cfg or {})}
    v, q, r = row.get("val") or {}, row.get("quality") or {}, row.get("rev") or {}
    reasons = []
    conv, conf = conviction(row, c)
    tier = "觀察"
    # 迴避：會計/破產否決、市價高於牛市情境、論點失效價已破
    stop_hit = bool(row.get("thesis") and row["thesis"].get("stop") and row.get("price")
                    and row["thesis"].get("direction") in ("多", "long", "bull")
                    and row["price"] < row["thesis"]["stop"])
    if q.get("veto"):
        tier, reasons = "迴避", ["品質否決：" + "、".join(str(x).replace("_", "·") for x in (q.get("flags") or []))]
    elif v.get("verdict") == "exit":
        tier, reasons = "迴避", ["市價高於牛市情境"]
    elif stop_hit:
        tier, reasons = "迴避", ["跌破論點失效價"]
    elif row.get("held"):
        if v.get("verdict") == "trim" or (v.get("range_pos") is not None and v["range_pos"] > 0.7):
            tier, reasons = "減碼", [f"區間位置 {v.get('range_pos', 0):.2f} > 0.7（估值偏貴）"]
        else:
            tier = "持有"
            if v.get("verdict") == "accumulate":
                reasons.append("估值仍有安全邊際，可接受引擎加碼")
            if r and r.get("score", 0) < -0.3:
                reasons.append("分析師下修中，注意論點")
    else:
        if v.get("verdict") == "accumulate" and (q.get("score") is None or q["score"] >= 0) and (not r or r.get("score", 0) > -0.3):
            tier = "累積候選"
            reasons.append(f"MoS {v.get('mos', 0):+.0%}" + ("；技術訊號已達門檻" if (row.get("tech") or -9) >= float(c["tech_buy"]) else "；等技術訊號（引擎觸發進場）"))
        elif v.get("verdict") in ("hold",) and (row.get("tech") or -9) >= float(c["tech_buy"]) and (q.get("score") or 0) >= 0:
            tier, reasons = "觀察", ["技術訊號在但估值無安全邊際——只做引擎標準部位"]
        elif not v:
            reasons.append("未建模（/model 或 /playbook build）")
    if v.get("stale"):
        reasons.append(f"模型 {v.get('age_days')} 天未更新")
    if conv is not None and conf < 0.75:
        reasons.append(f"成分僅 {len(row.get('components', []))}/4")
    # 權重帶（參考）
    band = None
    if tier in ("累積候選", "持有"):
        mult = (0.5 + (conv if conv is not None else 0.5)) * float(c["regime_mult"].get(regime, c["regime_mult"][None]))
        mid = min(float(c["base_weight"]) * mult, float(c["max_single"]))
        band = (round(mid * 0.8, 4), round(min(mid * 1.2, float(c["max_single"])), 4))
    return {"tier": tier, "reasons": reasons, "conviction": conv, "confidence": conf,
            "quadrant": quadrant(row), "weight_band": band}


# ── 3. 組合層 ───────────────────────────────────────────────────────────────

def build_plan(state: dict, today: str, ledger: dict | None = None, quality_map: dict | None = None,
               universe: dict | None = None, themes: dict | None = None, cfg: dict | None = None) -> dict:
    c = {**DEFAULTS, **(cfg or {})}
    regime = ((state.get("weather") or {}).get("regime") or {}).get("regime")
    tickers = list(dict.fromkeys(list(state.get("watchlist") or []) + sorted(((state.get("engine") or {}).get("pos") or {}).keys())))
    rows = []
    for t in tickers:
        row = collect_row(t, state, today, ledger, (quality_map or {}).get(t), universe, c)
        row.update(classify(row, regime, c))
        rows.append(row)
    tiers = {k: [r for r in rows if r["tier"] == k] for k in TIER_ORDER}
    for k in tiers:
        tiers[k].sort(key=lambda r: -(r.get("conviction") or 0))
    # 主題集中（以「累積候選 + 持有」的權重帶中點估）
    theme_exp = {}
    for r in tiers["累積候選"] + tiers["持有"]:
        if not r.get("weight_band"):
            continue
        mid = sum(r["weight_band"]) / 2
        for name, syms in (themes or {}).items():
            if r["ticker"] in syms:
                theme_exp[name] = theme_exp.get(name, 0.0) + mid
    over = {k: v for k, v in theme_exp.items() if v > float(c["max_theme"])}
    need_model = [r["ticker"] for r in rows if not r.get("val")]
    stale = [r["ticker"] for r in rows if r.get("val") and r["val"].get("stale")]
    return {"as_of": today, "regime": regime, "cash_target": float(c["cash_target"].get(regime, c["cash_target"][None])),
            "rows": rows, "tiers": tiers, "theme_exposure": theme_exp, "theme_over": over,
            "need_model": need_model, "stale_models": stale,
            "n": len(rows), "coverage": (sum(1 for r in rows if r.get("val")) / len(rows)) if rows else 0.0}


# ── 4. 文字（Telegram legacy Markdown：單 *、無底線）────────────────────────

def _one(r: dict) -> str:
    v = r.get("val") or {}
    bits = []
    if v.get("mos") is not None:
        bits.append(f"MoS {v['mos']:+.0%}")
    if r.get("tech") is not None:
        bits.append(f"技 {r['tech']:+.2f}")
    if r.get("rev"):
        bits.append(f"修正 {r['rev']['score']:+.2f}")
    if r.get("quality"):
        bits.append(f"質 {r['quality']['score']:+.2f}")
    if r.get("weight_band"):
        bits.append(f"權重 {r['weight_band'][0]:.1%}–{r['weight_band'][1]:.1%}")
    why = f"（{r['reasons'][0]}）" if r.get("reasons") else ""
    return f"・{r['ticker']} " + "｜".join(bits) + why


def plan_text(plan: dict, max_per_tier: int = 6) -> str:
    reg_lab = {"risk_on": "🟢 偏多", "neutral": "🟡 中性", "risk_off": "🔴 偏空", None: "— 未知"}
    lines = [f"🧭 *佈局計畫*（{plan['as_of']}；大盤 {reg_lab.get(plan['regime'], '—')}，現金目標 {plan['cash_target']:.0%}）",
             f"覆蓋 {plan['n']} 檔，{plan['coverage']:.0%} 已建模"]
    for k in TIER_ORDER:
        rs = plan["tiers"].get(k) or []
        if not rs:
            continue
        lines.append(f"{TIER_EMOJI[k]} *{k}*（{len(rs)}）")
        lines.extend(_one(r) for r in rs[:max_per_tier])
        if len(rs) > max_per_tier:
            lines.append(f"  …另 {len(rs) - max_per_tier} 檔")
    if plan.get("theme_over"):
        lines.append("⚠️ 主題集中超過 25%：" + "、".join(f"{k.replace('_', '·')} {v:.0%}" for k, v in plan["theme_over"].items()))
    if plan.get("need_model"):
        lines.append(f"未建模 {len(plan['need_model'])} 檔：{' '.join(plan['need_model'][:8])}（`/playbook build` 逐批建模）")
    if plan.get("stale_models"):
        lines.append(f"模型過期 {len(plan['stale_models'])} 檔：{' '.join(plan['stale_models'][:8])}")
    lines.append("四象限與權重帶為佈局參考、不下單；進場仍由技術訊號、出場仍由價格機制；估值層未過 holdout 不接引擎；非投資建議")
    txt = "\n".join(lines)
    if len(txt) > 3800:
        cut = txt.rfind("\n", 0, 3800)
        txt = txt[:cut] + "\n…"
    return txt


def plan_brief(plan: dict) -> str:
    """給週報的一段摘要。"""
    t = plan["tiers"]
    return (f"🧭 佈局：累積候選 {len(t['累積候選'])}、持有 {len(t['持有'])}、減碼 {len(t['減碼'])}、迴避 {len(t['迴避'])}；"
            f"現金目標 {plan['cash_target']:.0%}；{plan['coverage']:.0%} 已建模"
            + (f"；候選：{' '.join(r['ticker'] for r in t['累積候選'][:5])}" if t["累積候選"] else ""))


# ── 5. 自我測試 ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    today = "2026-09-09"
    state = {
        "watchlist": ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"],
        "weather": {"regime": {"regime": "risk_on"}},
        "engine": {"pos": {"BBB": {"opened": "2026-08-01", "scaled_out": False}, "DDD": {"opened": "2026-07-01"}}},
        "last_scores": {"AAA": {"score": 0.62, "price": 100}, "BBB": {"score": 0.1, "price": 50}, "CCC": {"score": -0.4, "price": 20},
                        "DDD": {"score": 0.3, "price": 80}, "EEE": {"score": 0.7, "price": 30}},
        "val_hist": {"AAA": [{"d": "2026-09-05", "base": 140, "bear": 90, "bull": 190, "mos": 0.40, "verdict": "accumulate", "px": 100}],
                     "BBB": [{"d": "2026-07-01", "base": 45, "bear": 30, "bull": 60, "mos": -0.10, "verdict": "hold", "px": 50}],
                     "CCC": [{"d": "2026-09-01", "base": 12, "bear": 8, "bull": 16, "mos": -0.40, "verdict": "exit", "px": 20}],
                     "DDD": [{"d": "2026-09-08", "base": 85, "bear": 60, "bull": 84, "mos": 0.06, "verdict": "trim", "px": 80}]},
        "theses": {"EEE": {"direction": "多", "stop": 35, "status": "active", "conviction": "高"}},
    }
    ledger = {"tickers": {"AAA": {"latest": {"eps": {"0y": {"cur": 5.5, "d30": 5.2, "d90": 5.0}}, "rev": {"0y": {"up30": 8, "down30": 1}},
                                             "est": {"0y": {"eps_n": 20}}, "tgt": {}, "rec": {}}, "rows": [], "ts": today}}}
    qmap = {"AAA": {"score": 0.6, "flags": [], "veto": False}, "FFF": {"score": -0.8, "flags": ["beneish_high"], "veto": True}}
    uni = {"top": [{"t": "AAA", "rank": 3}], "broad": [{"t": "BBB"}]}
    themes = {"AI電力/散熱": ["AAA", "BBB"]}
    plan = build_plan(state, today, ledger, qmap, uni, themes)
    by = {r["ticker"]: r for r in plan["rows"]}
    assert by["AAA"]["tier"] == "累積候選" and by["AAA"]["quadrant"] == "論點對·價格對" and by["AAA"]["weight_band"]
    assert by["AAA"]["confidence"] == 1.0 and 0.6 < by["AAA"]["conviction"] <= 1.0 and by["AAA"]["universe_rank"] == 3
    assert by["BBB"]["tier"] == "持有" and by["BBB"]["val"]["stale"] and any("未更新" in x for x in by["BBB"]["reasons"])
    assert by["CCC"]["tier"] == "迴避" and by["DDD"]["tier"] == "減碼"
    assert by["EEE"]["tier"] == "迴避" and "失效價" in by["EEE"]["reasons"][0]          # 價 30 < 失效價 35
    assert by["FFF"]["tier"] == "迴避" and "品質否決" in by["FFF"]["reasons"][0] and by["FFF"]["val"] is None
    assert plan["cash_target"] == 0.10 and plan["need_model"] == ["EEE", "FFF"] and plan["stale_models"] == ["BBB"]
    assert by["AAA"]["weight_band"][1] <= 0.10 and by["AAA"]["weight_band"][0] < by["AAA"]["weight_band"][1]
    print(f"✅ 1 分層：{ {k: [r['ticker'] for r in v] for k, v in plan['tiers'].items()} }")

    # regime 打折：risk_off 權重帶縮、現金目標升；資料不足四象限；缺 tech 的成分數
    state2 = dict(state); state2["weather"] = {"regime": {"regime": "risk_off"}}
    plan2 = build_plan(state2, today, ledger, qmap, uni, themes)
    a1, a2 = by["AAA"]["weight_band"], {r["ticker"]: r for r in plan2["rows"]}["AAA"]["weight_band"]
    assert a2[1] < a1[1] and plan2["cash_target"] == 0.40
    r_none = collect_row("ZZZ", {"watchlist": ["ZZZ"]}, today)
    assert quadrant(r_none) == "資料不足" and conviction(r_none) == (None, 0.0)
    assert classify(r_none, "neutral")["tier"] == "觀察" and "未建模" in classify(r_none, "neutral")["reasons"][0]
    print("✅ 2 regime 打折 / 資料不足 / 缺成分")

    # 主題集中檢查：兩檔同主題各 ~6% 不超；強迫超過 → 警示
    assert plan["theme_over"] == {}
    plan3 = build_plan(state, today, ledger, qmap, uni, themes, cfg={"base_weight": 0.2, "max_single": 0.5})
    assert "AI電力/散熱" in plan3["theme_over"]
    print("✅ 3 主題集中")

    # 文字 Markdown 安全
    t1, t2 = plan_text(plan), plan_brief(plan)
    assert t1.count("*") % 2 == 0 and "_" not in t1.replace("`/playbook build`", "") and "_" not in t2
    print(t1)
    print("\nplaybook selftest OK ✅")
