"""
quality.py – 品質 / 會計風險 / 破產風險評分（估值層 P1；純邏輯、離線可測）

輸入為 fin_data 的年期別（新→舊，每筆 dict）。公式來源（VALUATION_LANDSCAPE §5）：
  • Piotroski F（2000）9 項：獲利 4 / 槓桿流動 3 / 效率 2
  • Altman Z（製造業上市，含市值）與 Z''（非製造/新興，帳面權益）：Z''>2.6 安全、<1.1 危險
  • Beneish M-Score（8 指數）：> −1.78 高機率操縱、−2.22 ~ −1.78 灰區
  • Sloan 應計（簡化：(NI − CFO)/平均總資產）：最高十分位盈餘品質差；此處 >0.10 標紅
  • 現金轉換 FCF/NI 三年中位、SBC 占營收、稀釋率、利息保障、ROIC 與 ROIC−WACC 價差趨勢

設計原則：缺欄不臆測——該項回 None 並計入 `missing`；分數只由可算的項目構成並附上 n。
FinanceToolkit（MIT）的公式對照移植，不引入套件（使用者拍板）。教育用途，非投資建議。
"""

from __future__ import annotations

import math

NEUTRAL_TAX = 0.25


def _f(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _div(a, b):
    a, b = _f(a), _f(b)
    if a is None or b is None or abs(b) < 1e-12:
        return None
    return a / b


def _avg(a, b):
    a, b = _f(a), _f(b)
    if a is None and b is None:
        return None
    if a is None or b is None:
        return a if a is not None else b
    return (a + b) / 2


def _g(p: dict | None, k: str):
    return _f((p or {}).get(k))


# ── 1. Piotroski F ─────────────────────────────────────────────────────────

def piotroski(periods: list[dict]) -> dict:
    """回 {score, max, items:{name: 0/1/None}, missing:[...]}；需至少 2 年。"""
    if len(periods) < 2:
        return {"score": None, "max": 9, "items": {}, "missing": ["need_2_years"]}
    c, p = periods[0], periods[1]
    ta_c, ta_p = _g(c, "total_assets"), _g(p, "total_assets")
    roa_c = _div(_g(c, "net_income"), ta_c)          # 兩期同用期末 TA（分母一致，對抗驗證 C1）
    roa_p = _div(_g(p, "net_income"), ta_p)
    items = {
        "roa_pos": None if roa_c is None else int(roa_c > 0),
        "cfo_pos": None if _g(c, "cfo") is None else int(_g(c, "cfo") > 0),
        "roa_up": None if (roa_c is None or roa_p is None) else int(roa_c > roa_p),
        "accrual": None if (_g(c, "cfo") is None or _g(c, "net_income") is None) else int(_g(c, "cfo") > _g(c, "net_income")),
        "leverage_down": None,
        "liquidity_up": None,
        "no_dilution": None,
        "margin_up": None,
        "turnover_up": None,
    }
    lev_c, lev_p = _div(_g(c, "total_debt"), ta_c), _div(_g(p, "total_debt"), ta_p)
    if lev_c is not None and lev_p is not None:
        items["leverage_down"] = int(lev_c <= lev_p)
    cr_c = _div(_g(c, "current_assets"), _g(c, "current_liabilities"))
    cr_p = _div(_g(p, "current_assets"), _g(p, "current_liabilities"))
    if cr_c is not None and cr_p is not None:
        items["liquidity_up"] = int(cr_c > cr_p)
    sh_c = _g(c, "diluted_shares") or _g(c, "shares_out")
    sh_p = _g(p, "diluted_shares") or _g(p, "shares_out")
    if sh_c is not None and sh_p is not None and sh_p > 0:
        items["no_dilution"] = int(sh_c <= sh_p * 1.01)          # 容忍 1% 內（員工計畫）
    gm_c, gm_p = _div(_g(c, "gross_profit"), _g(c, "revenue")), _div(_g(p, "gross_profit"), _g(p, "revenue"))
    if gm_c is not None and gm_p is not None:
        items["margin_up"] = int(gm_c > gm_p)
    to_c = _div(_g(c, "revenue"), ta_c)
    to_p = _div(_g(p, "revenue"), ta_p)
    if to_c is not None and to_p is not None:
        items["turnover_up"] = int(to_c > to_p)
    known = {k: v for k, v in items.items() if v is not None}
    return {"score": sum(known.values()) if known else None, "max": 9, "n_known": len(known),
            "items": items, "missing": [k for k, v in items.items() if v is None]}


# ── 2. Altman Z / Z'' ─────────────────────────────────────────────────────

def altman(periods: list[dict], mkt_cap: float | None = None, manufacturing: bool = False) -> dict:
    """製造業上市（有市值）用原始 Z；否則 Z''（帳面權益，適用非製造/新興）。"""
    if not periods:
        return {"z": None, "model": None, "zone": None}
    c = periods[0]
    ta = _g(c, "total_assets")
    if not ta:
        return {"z": None, "model": None, "zone": None, "missing": ["total_assets"]}
    wc = (_g(c, "current_assets") or 0) - (_g(c, "current_liabilities") or 0) if (_g(c, "current_assets") is not None and _g(c, "current_liabilities") is not None) else None
    re_ = _g(c, "retained_earnings")
    ebit = _g(c, "ebit") if _g(c, "ebit") is not None else _g(c, "operating_income")
    eq = _g(c, "total_equity")
    tl = (ta - eq) if eq is not None else None
    a = _div(wc, ta); b = _div(re_, ta); cc = _div(ebit, ta)
    missing = [k for k, v in (("working_capital", a), ("retained_earnings", b), ("ebit", cc)) if v is None]
    if manufacturing and mkt_cap and tl:
        d = mkt_cap / tl
        e = _div(_g(c, "revenue"), ta)
        if missing or e is None:
            return {"z": None, "model": "Z", "zone": None, "missing": missing + (["revenue"] if e is None else [])}
        z = 1.2 * a + 1.4 * b + 3.3 * cc + 0.6 * d + 1.0 * e
        zone = "safe" if z > 2.99 else ("grey" if z >= 1.81 else "distress")
        return {"z": z, "model": "Z", "zone": zone}
    d = _div(eq, tl)
    if missing or d is None:
        return {"z": None, "model": "Z''", "zone": None, "missing": missing + (["equity_or_liabilities"] if d is None else [])}
    z = 3.25 + 6.56 * a + 3.26 * b + 6.72 * cc + 1.05 * d
    zone = "safe" if z > 2.6 else ("grey" if z >= 1.1 else "distress")
    return {"z": z, "model": "Z''", "zone": zone}


# ── 3. Beneish M ──────────────────────────────────────────────────────────

def beneish(periods: list[dict]) -> dict:
    """8 指數；缺的指數以 1.0（中性）代入並列入 missing。M > −1.78 高風險、≥ −2.22 灰區。"""
    if len(periods) < 2:
        return {"m": None, "zone": None, "missing": ["need_2_years"]}
    c, p = periods[0], periods[1]
    rev_c, rev_p = _g(c, "revenue"), _g(p, "revenue")
    idx, missing = {}, []

    def put(name, val):
        if val is None or val <= 0 or not math.isfinite(val):
            idx[name] = 1.0
            missing.append(name)
        else:
            idx[name] = val
    put("DSRI", _div(_div(_g(c, "receivables"), rev_c), _div(_g(p, "receivables"), rev_p)))
    put("GMI", _div(_div(_g(p, "gross_profit"), rev_p), _div(_g(c, "gross_profit"), rev_c)))
    def aq(x):
        ta, ca, ppe = _g(x, "total_assets"), _g(x, "current_assets"), _g(x, "net_ppe")
        return None if (ta is None or ca is None or ppe is None or ta <= 0) else 1 - (ca + ppe) / ta
    put("AQI", _div(aq(c), aq(p)))
    put("SGI", _div(rev_c, rev_p))
    def dep_rate(x):
        da, ppe = _g(x, "da"), _g(x, "net_ppe")
        return None if (da is None or ppe is None or da + ppe <= 0) else da / (da + ppe)
    put("DEPI", _div(dep_rate(p), dep_rate(c)))
    put("SGAI", _div(_div(_g(c, "sga"), rev_c), _div(_g(p, "sga"), rev_p)))
    def lev(x):
        ta = _g(x, "total_assets"); d = _g(x, "total_debt"); cl = _g(x, "current_liabilities")
        return None if (ta is None or ta <= 0 or d is None or cl is None) else (d + cl) / ta
    put("LVGI", _div(lev(c), lev(p)))
    ni, cfo, ta = _g(c, "net_income"), _g(c, "cfo"), _g(c, "total_assets")
    tata = None if (ni is None or cfo is None or not ta) else (ni - cfo) / ta
    if tata is None:
        missing.append("TATA"); tata = 0.0
    m = (-4.84 + 0.92 * idx["DSRI"] + 0.528 * idx["GMI"] + 0.404 * idx["AQI"] + 0.892 * idx["SGI"]
         + 0.115 * idx["DEPI"] - 0.172 * idx["SGAI"] + 4.679 * tata - 0.327 * idx["LVGI"])
    zone = "high" if m > -1.78 else ("grey" if m >= -2.22 else "low")
    return {"m": m, "zone": zone, "indices": {**idx, "TATA": tata}, "missing": missing,
            "reliable": len(missing) <= 2}


# ── 4. 其他品質項 ─────────────────────────────────────────────────────────

def sloan_accrual(periods: list[dict]) -> float | None:
    if not periods:
        return None
    c = periods[0]
    ta = _avg(_g(c, "total_assets"), _g(periods[1], "total_assets")) if len(periods) > 1 else _g(c, "total_assets")
    ni, cfo = _g(c, "net_income"), _g(c, "cfo")
    return None if (ni is None or cfo is None or not ta) else (ni - cfo) / ta


def fcf_conversion(periods: list[dict], n: int = 3) -> float | None:
    """FCF/NI 中位數（NI ≤ 0 的年份略過）。"""
    vals = []
    for p in periods[:n]:
        ni, fcf = _g(p, "net_income"), _g(p, "fcf")
        if ni and ni > 0 and fcf is not None:
            vals.append(fcf / ni)
    if not vals:
        return None
    vals.sort()
    return vals[len(vals) // 2] if len(vals) % 2 else (vals[len(vals) // 2 - 1] + vals[len(vals) // 2]) / 2


def sbc_pct(periods: list[dict]) -> float | None:
    return _div(_g(periods[0], "sbc"), _g(periods[0], "revenue")) if periods else None


def dilution_yoy(periods: list[dict]) -> float | None:
    if len(periods) < 2:
        return None
    a = _g(periods[0], "diluted_shares") or _g(periods[0], "shares_out")
    b = _g(periods[1], "diluted_shares") or _g(periods[1], "shares_out")
    return (a / b - 1) if (a and b) else None


def interest_coverage(periods: list[dict]) -> float | None:
    if not periods:
        return None
    ebit = _g(periods[0], "ebit") if _g(periods[0], "ebit") is not None else _g(periods[0], "operating_income")
    ie = _g(periods[0], "interest_expense")
    if ebit is None or ie is None:
        return None
    ie = abs(ie)
    return None if ie < 1e-9 else ebit / ie


def invested_capital(p: dict) -> float | None:
    eq, d, cash = _g(p, "total_equity"), _g(p, "total_debt"), (_g(p, "cash_and_sti") if _g(p, "cash_and_sti") is not None else _g(p, "cash"))
    if eq is None:
        return None
    return eq + (_g(p, "minority_interest") or 0) + (d or 0) - (cash or 0)   # 債為全體 → 權益加回少數股權


def roic_series(periods: list[dict], tax_rate: float | None = None) -> list[tuple[str, float]]:
    """[(period_end, ROIC)] 舊→新：NOPAT / 平均投入資本；稅率預設有效稅率夾 [0.1, 0.35] 或 25%。"""
    out = []
    for i, p in enumerate(periods):
        ebit = _g(p, "ebit") if _g(p, "ebit") is not None else _g(p, "operating_income")
        if ebit is None:
            continue
        t = tax_rate
        if t is None:
            et = _div(_g(p, "tax_provision"), _g(p, "pretax_income"))
            t = min(max(et, 0.10), 0.35) if et is not None else NEUTRAL_TAX
        ic_c = invested_capital(p)
        ic_p = invested_capital(periods[i + 1]) if i + 1 < len(periods) else None
        ic = _avg(ic_c, ic_p)
        if ic is None or ic <= 0:
            continue
        out.append((p["period_end"], ebit * (1 - t) / ic))
    return list(reversed(out))


def slope(series: list[tuple[str, float]]) -> float | None:
    """簡單線性斜率（每期變化）；<3 點回 None。"""
    ys = [v for _, v in series if v is not None]
    n = len(ys)
    if n < 3:
        return None
    xs = list(range(n))
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else None


# ── 5. 彙整與旗標 ─────────────────────────────────────────────────────────

FLAG_RULES = {
    "beneish_high":     "Beneish M 高於 −1.78（盈餘操縱高機率）",
    "beneish_grey":     "Beneish M 落在灰區",
    "altman_distress":  "Altman 危險區",
    "accrual_high":     "應計項目占資產 >10%（盈餘與現金背離）",
    "fcf_conv_low":     "FCF/NI 三年中位 <0.8",
    "sbc_high":         "SBC 占營收 >10%",
    "dilution_high":    "稀釋率 >3%/年",
    "coverage_low":     "利息保障 <3 倍",
    "roic_below_wacc":  "ROIC 低於 WACC（成長毀值）",
}


def quality_summary(periods: list[dict], mkt_cap: float | None = None, wacc: float | None = None,
                    manufacturing: bool = False) -> dict:
    """全套品質評分 + 旗標 + 有界品質分（-1..1，只由可算項目構成）。"""
    f = piotroski(periods)
    z = altman(periods, mkt_cap, manufacturing)
    b = beneish(periods)
    acc = sloan_accrual(periods)
    conv = fcf_conversion(periods)
    sbc = sbc_pct(periods)
    dil = dilution_yoy(periods)
    cov = interest_coverage(periods)
    rs = roic_series(periods)
    roic = rs[-1][1] if rs else None
    spread = (roic - wacc) if (roic is not None and wacc is not None) else None
    flags = []
    if b.get("zone") == "high" and b.get("reliable"):
        flags.append("beneish_high")
    elif b.get("zone") == "grey" and b.get("reliable"):
        flags.append("beneish_grey")
    if z.get("zone") == "distress":
        flags.append("altman_distress")
    if acc is not None and acc > 0.10:
        flags.append("accrual_high")
    if conv is not None and conv < 0.8:
        flags.append("fcf_conv_low")
    if sbc is not None and sbc > 0.10:
        flags.append("sbc_high")
    if dil is not None and dil > 0.03:
        flags.append("dilution_high")
    if cov is not None and cov < 3:
        flags.append("coverage_low")
    if spread is not None and spread < 0:
        flags.append("roic_below_wacc")
    # 有界品質分：Piotroski（0..1）、ROIC 價差、旗標扣分
    parts, w = [], []
    if f.get("score") is not None and f.get("n_known", 0) >= 5:
        parts.append((f["score"] / f["n_known"]) * 2 - 1); w.append(0.5)
    if spread is not None:
        parts.append(max(-1.0, min(1.0, spread / 0.10))); w.append(0.3)
    if conv is not None:
        parts.append(max(-1.0, min(1.0, (conv - 0.8) / 0.4))); w.append(0.2)
    score = (sum(p * ww for p, ww in zip(parts, w)) / sum(w)) if w else None
    if score is not None:
        score = max(-1.0, min(1.0, score - 0.25 * sum(1 for x in flags if x in ("beneish_high", "altman_distress", "accrual_high"))))
    veto = any(x in flags for x in ("beneish_high", "altman_distress"))
    return {"piotroski": f, "altman": z, "beneish": b, "sloan_accrual": acc, "fcf_conversion_3y": conv,
            "sbc_pct_rev": sbc, "dilution_yoy": dil, "interest_coverage": cov,
            "roic": roic, "roic_series": rs, "roic_slope": slope(rs), "roic_wacc_spread": spread,
            "flags": flags, "veto": veto, "score": score, "years": len(periods)}


def quality_text(q: dict, ticker: str = "") -> str:
    """Telegram legacy Markdown（單 *、無底線）。"""
    f, z, b = q.get("piotroski", {}), q.get("altman", {}), q.get("beneish", {})
    lines = [f"🧬 *{ticker} 品質與會計風險*（{q.get('years', 0)} 年資料）"]
    if f.get("score") is not None:
        lines.append(f"Piotroski F {f['score']}/{f.get('n_known', 9)}"
                     + (f"（{9 - f.get('n_known', 9)} 項缺資料）" if f.get("n_known", 9) < 9 else ""))
    if z.get("z") is not None:
        zone = {"safe": "安全", "grey": "灰區", "distress": "危險"}.get(z["zone"], "—")
        lines.append(f"Altman {z['model']} {z['z']:.2f}（{zone}）")
    if b.get("m") is not None:
        zone = {"low": "低", "grey": "灰區", "high": "高"}.get(b["zone"], "—")
        lines.append(f"Beneish M {b['m']:.2f}（操縱風險{zone}"
                     + ("" if b.get("reliable") else f"；{len(b.get('missing', []))} 指數缺資料，僅供參考") + "）")
    bits = []
    if q.get("roic") is not None:
        bits.append(f"ROIC {q['roic']:.1%}" + (f"（vs WACC {q['roic_wacc_spread']:+.1%}）" if q.get("roic_wacc_spread") is not None else ""))
    if q.get("fcf_conversion_3y") is not None:
        bits.append(f"FCF/NI {q['fcf_conversion_3y']:.2f}")
    if q.get("sloan_accrual") is not None:
        bits.append(f"應計 {q['sloan_accrual']:+.1%}")
    if q.get("sbc_pct_rev") is not None:
        bits.append(f"SBC {q['sbc_pct_rev']:.1%}")
    if q.get("dilution_yoy") is not None:
        bits.append(f"稀釋 {q['dilution_yoy']:+.1%}")
    if q.get("interest_coverage") is not None:
        bits.append(f"利息保障 {q['interest_coverage']:.1f}x")
    if bits:
        lines.append("｜".join(bits))
    if q.get("flags"):
        lines.append("⚠️ " + "；".join(FLAG_RULES.get(x, x) for x in q["flags"]))
    if q.get("veto"):
        lines.append("🚫 會計/破產旗標：估值層對此檔不給 accumulate")
    if q.get("score") is not None:
        lines.append(f"品質分 {q['score']:+.2f}（−1 ~ +1）")
    lines.append("非投資建議")
    return "\n".join(lines)


# ── 6. 自我測試（合成年期別）───────────────────────────────────────────────

if __name__ == "__main__":
    def yr(pe, rev, gp, oi, ni, cfo, capex, da, ar, inv, ap, ca, cl, ppe, ta, eq, debt, cash, sga, sh, ie, tax, pti, re_):
        return {"period_end": pe, "freq": "A", "revenue": rev, "gross_profit": gp, "operating_income": oi, "ebit": oi,
                "net_income": ni, "cfo": cfo, "capex": -capex, "fcf": cfo - capex, "da": da, "receivables": ar,
                "inventory": inv, "payables": ap, "current_assets": ca, "current_liabilities": cl, "net_ppe": ppe,
                "total_assets": ta, "total_equity": eq, "total_debt": debt, "cash": cash, "sga": sga,
                "diluted_shares": sh, "interest_expense": ie, "tax_provision": tax, "pretax_income": pti,
                "retained_earnings": re_, "sbc": rev * 0.01}
    # 健康公司：成長、利潤率升、去槓桿、現金好、股數不變
    good = [yr("2025-12-31", 12000, 4800, 2400, 1800, 2200, 300, 250, 1500, 800, 900, 4000, 2500, 1200, 10000, 6000, 1000, 2000, 1500, 100, 40, 500, 2300, 3000),
            yr("2024-12-31", 10000, 3800, 1800, 1300, 1600, 250, 220, 1300, 700, 800, 3300, 2300, 1100, 9000, 5000, 1500, 1500, 1400, 100, 60, 400, 1700, 1800),
            yr("2023-12-31", 8000, 2900, 1200, 900, 1100, 200, 200, 1100, 600, 700, 2800, 2100, 1000, 8000, 4200, 1800, 1200, 1300, 100, 80, 300, 1200, 900)]
    q = quality_summary(good, mkt_cap=50000, wacc=0.10)
    assert q["piotroski"]["score"] >= 8 and q["piotroski"]["n_known"] == 9, q["piotroski"]
    assert q["altman"]["model"] == "Z''" and q["altman"]["zone"] == "safe"
    assert q["beneish"]["zone"] in ("low", "grey") and q["beneish"]["reliable"]
    assert q["flags"] == [] and q["score"] > 0.5 and q["veto"] is False
    assert q["roic"] > 0.15 and q["roic_wacc_spread"] > 0 and q["roic_slope"] is not None
    print(f"✅ 1 健康公司：F={q['piotroski']['score']} Z''={q['altman']['z']:.2f} M={q['beneish']['m']:.2f} 品質分 {q['score']:+.2f}")

    # 操縱樣態：應收暴增、毛利率降、SG&A 升、應計高、槓桿升 → Beneish 高 + 應計旗標 + veto
    bad = [yr("2025-12-31", 12000, 3000, 900, 1500, -200, 300, 150, 5000, 800, 900, 7000, 2500, 1200, 14000, 5000, 5000, 300, 2600, 120, 300, 100, 1600, 1000),
           yr("2024-12-31", 10000, 3800, 1800, 1300, 1600, 250, 220, 1300, 700, 800, 3300, 2300, 1100, 9000, 5000, 1500, 1500, 1400, 100, 60, 400, 1700, 1800)]
    qb = quality_summary(bad, mkt_cap=20000, wacc=0.10)
    assert qb["beneish"]["zone"] == "high" and "beneish_high" in qb["flags"] and "accrual_high" in qb["flags"]
    assert qb["veto"] is True and qb["score"] < 0 and "dilution_high" in qb["flags"]
    print(f"✅ 2 操縱樣態：M={qb['beneish']['m']:.2f} 旗標 {qb['flags']}")

    # 缺欄：只有損益表 → Piotroski 部分項 None、Altman None、Beneish 多指數中性且 reliable=False
    thin = [{"period_end": "2025-12-31", "freq": "A", "revenue": 100, "net_income": 10, "operating_income": 12},
            {"period_end": "2024-12-31", "freq": "A", "revenue": 90, "net_income": 8, "operating_income": 10}]
    qt = quality_summary(thin)
    assert qt["piotroski"]["score"] is None or qt["piotroski"]["n_known"] < 5
    assert qt["altman"]["z"] is None and qt["beneish"]["reliable"] is False and qt["score"] is None
    assert quality_summary([])["piotroski"]["score"] is None and quality_summary([good[0]])["beneish"]["m"] is None
    print("✅ 3 缺欄不臆測（None + missing，不拋例外）")

    # Altman 製造業原始 Z（含市值）與危險區
    dist = [yr("2025-12-31", 1000, 100, -200, -300, -250, 50, 40, 300, 300, 400, 600, 900, 300, 2000, 200, 1500, 50, 250, 100, 120, 0, -320, -800)]
    az = altman(dist, mkt_cap=100, manufacturing=True)
    assert az["model"] == "Z" and az["zone"] == "distress"
    assert "altman_distress" in quality_summary(dist, mkt_cap=100, manufacturing=True)["flags"]
    print(f"✅ 4 Altman 製造業 Z={az['z']:.2f}（危險區）")

    # 文字輸出 Markdown 安全
    for qq, name in ((q, "GOOD"), (qb, "BAD"), (qt, "THIN")):
        t = quality_text(qq, name)
        assert t.count("*") % 2 == 0 and "_" not in t, t
    print(quality_text(q, "GOOD"))
    print("\nquality selftest OK ✅")
