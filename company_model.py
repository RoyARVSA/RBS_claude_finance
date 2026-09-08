"""
company_model.py – 公司模型引擎（估值層 P1；純邏輯、離線可測）

把使用者手工 VRT 三表模型的「驅動層」泛化到任意代碼（VALUATION_PLAN §1、LANDSCAPE §5–6）：
  1. derive_drivers：由 fin_data 年期別 + 市場資料 + 分析師共識推預設驅動（3 年均值、共識覆蓋前兩年、
     線性淡出）；使用者 /model set 覆蓋任何驅動
  2. project：5 年顯性 + 5 年淡出的 FCFF（EBIT→NOPAT +D&A −Capex −ΔNWC；SBC 視為成本＝不加回）
  3. 專業級 WACC：rf 即時/預設、ERP 常數、Blume 收縮 β（0.67β+0.33）、Rd = rf + 合成信評利差
     （利息保障倍數→利差）、市值權重（重用 valuation.calc_wacc）
  4. DCF：期中折現；終值 = NOPAT₁₁ × (1 − g/ROIC_TV) / (WACC − g)，ROIC_TV 預設收斂至 WACC
     （終值成長價值中性——Damodaran / implied-expectations 約束），不再用「最後一年 FCF ×(1+g)」
  5. 三情境（bear 是「論點錯了的世界」，非 base −10%）+ 機率加權 + 蒙地卡羅 + WACC×g 敏感度
  6. 反向 DCF：市價隱含的 5 年營收 CAGR（固定其他驅動，二分法求解，不依賴 scipy）
  7. 九條審核清單：任一不過 → verdict=review（Bot 不推 accumulate）
  8. 訊號：MoS、區間位置、上/下行比、品質分層門檻（25/35/50%）、品質旗標否決

產業路由：金融 / 地產 → DCF 不適用（RIM/AFFO 待 P1.5），輸出 review 並說明。
輸出永遠掛「非投資建議」。VRT Excel 為回歸測試基準（折現引擎對齊其 $117.7）。
"""

from __future__ import annotations

import math
import random

try:
    from valuation import DEFAULT_ERP, DEFAULT_RF, calc_wacc
except Exception:                       # 獨立測試環境
    DEFAULT_ERP, DEFAULT_RF = 0.055, 0.042
    calc_wacc = None

DEFAULTS = {
    "years": 5, "fade_years": 5, "terminal_growth": 0.03, "long_growth": 0.06,
    "erp": DEFAULT_ERP, "rf": DEFAULT_RF, "sbc_as_cost": True, "mid_year": True,
    "beta_floor": 0.6, "beta_cap": 2.0, "growth_floor": -0.30, "growth_cap": 0.60,
    "roic_tv": None,               # None → 收斂至 WACC（價值中性）；可覆蓋（護城河證據才用）
    "prob": (0.25, 0.50, 0.25),    # bear / base / bull
    "mc_n": 2000, "mc_seed": 7,
}
NON_DCF_SECTORS = {"Financial Services": "銀行/保險無法定義 FCFF——用 RIM（剩餘收益）與 P/B–ROE；P1.5 實作",
                   "Real Estate": "REIT 用 AFFO/DDM；P1.5 實作"}
# Damodaran 合成信評利差（大型企業，簡化 10 級）：利息保障倍數 → 違約利差
SPREAD_TABLE = [(8.5, 0.0075), (6.5, 0.0100), (5.5, 0.0125), (4.25, 0.0160), (3.0, 0.0225),
                (2.5, 0.0300), (2.25, 0.0400), (2.0, 0.0500), (1.75, 0.0650), (1.5, 0.0800),
                (-1e9, 0.1100)]
PARAM_LABELS = {"rev_g": "營收成長路徑", "opm_target": "目標營益率", "beta": "β", "wacc": "WACC",
                "tgr": "終端成長", "tax": "稅率", "guidance_rev": "指引營收", "long_growth": "第 6–10 年成長",
                "roic_tv": "終值 ROIC", "capex_pct": "Capex/營收", "da_pct": "D&A/營收", "nwc_pct": "NWC/增量營收"}


def _f(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _g(p, k):
    return _f((p or {}).get(k))


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def _avg(vals):
    v = [x for x in vals if x is not None]
    return sum(v) / len(v) if v else None


# ── 1. 驅動推導 ─────────────────────────────────────────────────────────────

def blume_beta(raw: float | None, floor: float = 0.6, cap: float = 2.0) -> float:
    if raw is None or raw <= 0:
        return 1.0
    return _clip(0.67 * raw + 0.33, floor, cap)


def synthetic_spread(coverage: float | None) -> float:
    if coverage is None:
        return 0.0225                       # 無資料：BBB 級
    for th, sp in SPREAD_TABLE:
        if coverage > th:
            return sp
    return SPREAD_TABLE[-1][1]


def derive_drivers(periods: list[dict], profile: dict | None = None, estimates: dict | None = None,
                   overrides: dict | None = None, cfg: dict | None = None) -> dict:
    """
    periods：fin_data 年期別（新→舊，≥1 期）。profile：{price, mkt_cap, beta, shares, sector}。
    estimates：estimates_ledger 的 latest 快照（用 est.0y/+1y 的 rev_growth 覆蓋前兩年）。
    overrides：使用者 /model set（rev_g 逗號路徑、opm_target、beta、wacc、tgr、tax、guidance_rev…）。
    回 drivers dict（含 provenance 欄說明每個驅動哪裡來）。
    """
    c = {**DEFAULTS, **(cfg or {})}
    ov = dict(overrides or {})
    pf = profile or {}
    if not periods:
        raise ValueError("no financial periods")
    cur = periods[0]
    prov = {}
    rev = _g(cur, "revenue")
    if not rev or rev <= 0:
        raise ValueError("revenue missing")

    # 歷史成長
    revs = [_g(p, "revenue") for p in periods if _g(p, "revenue")]
    yoy = (revs[0] / revs[1] - 1) if len(revs) >= 2 and revs[1] > 0 else None
    cagr = ((revs[0] / revs[-1]) ** (1 / (len(revs) - 1)) - 1) if len(revs) >= 3 and revs[-1] > 0 else yoy

    # 前兩年成長：指引 > 共識 > 歷史
    est = (estimates or {}).get("est") or {}
    e0, e1 = est.get("0y") or {}, est.get("+1y") or {}
    g1 = g2 = None
    # 共識用「水準」推導：0y 營收共識 / 基期營收 − 1；與 Yahoo 的 growth 欄差 >5pp 代表快照與
    # 年報的會計年度錯位（FY 滾動前後一週內），改用 +1y 或退回歷史（對抗驗證 Med-3）
    lvl0, lvl1 = _f(e0.get("rev_avg")), _f(e1.get("rev_avg"))
    gr0, gr1 = _f(e0.get("rev_growth")), _f(e1.get("rev_growth"))
    if lvl0 and lvl0 > 0:
        cand = lvl0 / rev - 1
        if gr0 is None or abs(cand - gr0) <= 0.05:
            g1, g2 = cand, ((lvl1 / lvl0 - 1) if lvl1 and lvl1 > 0 else gr1)
            prov["g1"] = "consensus"
        elif lvl1 and lvl1 > 0 and abs(lvl1 / rev - 1 - (gr0 or 0)) <= 0.05:
            g1, g2 = lvl1 / rev - 1, None                         # 快照已滾到下一 FY：用 +1y 當第一年
            prov["g1"] = "consensus(+1y aligned)"
    if g1 is None and lvl0 is None and gr0 is not None:
        g1, g2 = gr0, gr1                                            # 只有成長率無水準：較弱證據，仍優於歷史
        prov["g1"] = "consensus(growth only)"
    if ov.get("guidance_rev"):
        g1 = _f(ov["guidance_rev"]) / rev - 1
        prov["g1"] = "guidance"
    elif g1 is None:
        g1 = yoy if yoy is not None else 0.05
        prov["g1"] = "history"
    prov["g2"] = "consensus" if g2 is not None else "fade"
    lo, hi = float(c["growth_floor"]), float(c["growth_cap"])
    g1 = _clip(g1, lo, hi)
    long_g = _f(ov.get("long_growth")) if ov.get("long_growth") is not None else float(c["long_growth"])
    if g2 is None:
        g2 = g1 + (long_g - g1) * 0.25
    g2 = _clip(g2, lo, hi)
    years = int(c["years"])
    path = [g1, g2] + [g2 + (long_g - g2) * (i / (years - 2)) for i in range(1, years - 1)]
    path = path[:years]
    if ov.get("rev_g"):
        user = [_f(x) for x in str(ov["rev_g"]).split(",")] if isinstance(ov["rev_g"], str) else list(ov["rev_g"])
        user = [_clip(x, lo, hi) for x in user if x is not None]
        if user:
            path = (user + [user[-1]] * years)[:years]
            prov["rev_g"] = "override"

    # 利潤率
    opm_hist = [_g(p, "operating_income") / _g(p, "revenue") for p in periods[:3]
                if _g(p, "operating_income") is not None and _g(p, "revenue")]
    opm_last = opm_hist[0] if opm_hist else 0.10
    opm_avg = _avg(opm_hist) if opm_hist else opm_last
    if ov.get("opm_target") is not None:
        opm_target, prov["opm_target"] = _f(ov["opm_target"]), "override"
    elif opm_last > opm_avg:
        opm_target, prov["opm_target"] = (opm_last + opm_avg) / 2, "mid(last, 3y avg)"   # 不把高點永久化（Med-1）
    else:
        opm_target, prov["opm_target"] = opm_avg, "3y avg"                              # 下滑者假設回到均值
    opm_target = _clip(opm_target, -0.5, 0.6)

    # 稅率
    et = [_g(p, "tax_provision") / _g(p, "pretax_income") for p in periods[:3]
          if _g(p, "tax_provision") is not None and _g(p, "pretax_income") and _g(p, "pretax_income") > 0]
    tax = _f(ov.get("tax")) if ov.get("tax") is not None else (_clip(_avg(et), 0.10, 0.35) if et else 0.25)

    # 比率
    def ratio(field, default, lo_, hi_, n=3, sign=1.0):
        vals = [sign * _g(p, field) / _g(p, "revenue") for p in periods[:n] if _g(p, field) is not None and _g(p, "revenue")]
        return _clip(_avg(vals), lo_, hi_) if vals else default
    da_pct = _f(ov.get("da_pct")) if ov.get("da_pct") is not None else ratio("da", 0.04, 0.0, 0.25)
    capex_pct = _f(ov.get("capex_pct")) if ov.get("capex_pct") is not None else ratio("capex", 0.05, 0.0, 0.40, sign=-1.0)
    sbc_pct = ratio("sbc", 0.0, 0.0, 0.30)
    nwc = None
    if all(_g(cur, k) is not None for k in ("receivables", "inventory", "payables")):
        nwc = (_g(cur, "receivables") + _g(cur, "inventory") - _g(cur, "payables") - (_g(cur, "deferred_revenue") or 0)) / rev
    nwc_pct = _f(ov.get("nwc_pct")) if ov.get("nwc_pct") is not None else (_clip(nwc, -0.3, 0.5) if nwc is not None else 0.05)

    # 資本結構與 WACC
    shares = _g(cur, "diluted_shares") or _f(pf.get("shares")) or _g(cur, "shares_out")
    debt = _g(cur, "total_debt") or 0.0
    cash = _g(cur, "cash_and_sti") if _g(cur, "cash_and_sti") is not None else (_g(cur, "cash") or 0.0)
    minority = _g(cur, "minority_interest") or 0.0
    net_debt = debt - cash + minority
    raw_beta = _f(ov.get("beta")) if ov.get("beta") is not None else _f(pf.get("beta"))
    beta = blume_beta(raw_beta, float(c["beta_floor"]), float(c["beta_cap"])) if ov.get("beta") is None else _clip(raw_beta, 0.3, 3.0)
    rf = _f(ov.get("rf")) if ov.get("rf") is not None else float(c["rf"])
    erp = float(c["erp"])
    ebit = _g(cur, "ebit") if _g(cur, "ebit") is not None else _g(cur, "operating_income")
    ie = _g(cur, "interest_expense")
    coverage = (ebit / abs(ie)) if (ebit is not None and ie and abs(ie) > 1e-9) else None
    rd = rf + synthetic_spread(coverage)
    mkt_cap = _f(pf.get("mkt_cap")) or ((_f(pf.get("price")) or 0) * (shares or 0))
    if calc_wacc and mkt_cap:
        w = calc_wacc(rf, beta, mkt_cap, debt, cash, erp, cost_debt_pre=rd, tax_rate=tax)
        wacc = w["wacc"]
    else:
        wacc = rf + beta * erp
    clamped = {}
    if ov.get("wacc") is not None:
        wacc = _clip(_f(ov["wacc"]), 0.04, 0.30)
        prov["wacc"] = "override"
    tgr = _clip(_f(ov.get("tgr")), 0.0, 0.05) if ov.get("tgr") is not None else float(c["terminal_growth"])
    if tgr > wacc - 0.02:
        tgr = wacc - 0.02
        if ov.get("tgr") is not None:
            clamped["tgr"] = tgr
    roic_tv = _f(ov.get("roic_tv")) if ov.get("roic_tv") is not None else (c["roic_tv"] if c["roic_tv"] else wacc)
    if roic_tv < wacc:                                # 終值 ROIC 不得低於 WACC（否則成長毀值）
        roic_tv = wacc
        if ov.get("roic_tv") is not None:
            clamped["roic_tv"] = roic_tv

    return {"ticker": pf.get("ticker"), "base_revenue": rev, "period_end": cur.get("period_end"),
            "rev_g": [round(x, 4) for x in path], "long_growth": long_g, "opm_last": opm_last,
            "opm_target": opm_target, "tax": tax, "da_pct": da_pct, "capex_pct": capex_pct,
            "sbc_pct": sbc_pct, "nwc_pct": nwc_pct, "shares": shares, "net_debt": net_debt,
            "debt": debt, "cash": cash, "beta_raw": raw_beta, "beta": beta, "rf": rf, "erp": erp,
            "cost_debt": rd, "coverage": coverage, "wacc": wacc, "tgr": tgr, "roic_tv": roic_tv,
            "price": _f(pf.get("price")), "mkt_cap": mkt_cap, "sector": pf.get("sector"),
            "hist_cagr": cagr, "hist_yoy": yoy, "sbc_as_cost": bool(c["sbc_as_cost"]),
            "years": years, "fade_years": int(c["fade_years"]), "mid_year": bool(c["mid_year"]),
            "provenance": prov, "overrides": {k: v for k, v in ov.items() if v is not None},
            "clamped": clamped}


# ── 2. 投影 ─────────────────────────────────────────────────────────────────

def project(d: dict) -> list[dict]:
    """5 年顯性 + fade_years 淡出（成長線性淡到 tgr、利潤率線性到 target 後持平）。"""
    rows, rev = [], d["base_revenue"]
    years, fade = d["years"], d["fade_years"]
    g_path = list(d["rev_g"])
    last_g = g_path[-1]
    for i in range(fade):
        g_path.append(last_g + (d["tgr"] - last_g) * ((i + 1) / fade))
    for yr, g in enumerate(g_path, 1):
        prev = rev
        rev = rev * (1 + g)
        opm = d["opm_last"] + (d["opm_target"] - d["opm_last"]) * min(1.0, yr / years)
        ebit = rev * opm
        nopat = ebit * (1 - d["tax"])
        da, capex = rev * d["da_pct"], rev * d["capex_pct"]
        dnwc = (rev - prev) * d["nwc_pct"]
        # GAAP 營業利益已扣 SBC 費用：sbc_as_cost=True 表「不加回」（視為真實成本）；
        # False 才把 SBC 當非現金加回（對抗驗證 High-1：原本再扣一次＝重複扣除）
        sbc = rev * d["sbc_pct"]
        fcf = nopat + da - capex - dnwc + (0.0 if d.get("sbc_as_cost") else sbc)
        rows.append({"year": yr, "growth": g, "revenue": rev, "opm": opm, "ebit": ebit, "nopat": nopat,
                     "da": da, "capex": capex, "dnwc": dnwc, "sbc": sbc, "fcf": fcf})
    return rows


# ── 3. DCF（期中折現 + 價值中性終值）──────────────────────────────────────────

def dcf(rows: list[dict], wacc: float, tgr: float, net_debt: float, shares: float | None,
        roic_tv: float | None = None, mid_year: bool = True, terminal_fcf: float | None = None) -> dict:
    if tgr >= wacc:
        raise ValueError("g ≥ WACC")
    n = len(rows)
    pv = sum(r["fcf"] / (1 + wacc) ** ((i - 0.5) if mid_year else i) for i, r in enumerate(rows, 1))
    last = rows[-1]
    if terminal_fcf is None:
        nopat_next = last["nopat"] * (1 + tgr)
        reinvest = (tgr / roic_tv) if (roic_tv and roic_tv > 0) else 0.0
        terminal_fcf = nopat_next * (1 - reinvest)             # 再投資率 = g / ROIC_TV
    tv = terminal_fcf / (wacc - tgr)
    pv_tv = tv / (1 + wacc) ** ((n - 0.5) if mid_year else n)
    ev = pv + pv_tv
    eq = ev - net_debt
    ps = (eq / shares) if shares and shares > 0 else None
    return {"pv_fcfs": pv, "pv_terminal": pv_tv, "tv": tv, "ev": ev, "equity": eq, "per_share": ps,
            "tv_pct": (pv_tv / ev) if ev > 0 else None, "terminal_fcf": terminal_fcf,
            "implied_tv_ebitda": (tv / (last["ebit"] + last["da"])) if (last["ebit"] + last["da"]) > 0 else None}


def value_drivers(d: dict, **tweak) -> dict:
    """套用臨時調整（情境/敏感度用）後估值：回 dcf() 結果 + rows。"""
    dd = {**d, **tweak}
    rows = project(dd)
    out = dcf(rows, dd["wacc"], dd["tgr"], dd["net_debt"], dd["shares"], dd.get("roic_tv"), dd.get("mid_year", True))
    out["rows"] = rows
    return out


# ── 4. 情境 / 蒙地卡羅 / 敏感度 / 反向 DCF ────────────────────────────────

def scenarios(d: dict, prob=None) -> dict:
    """bear：成長 −50%|g|、目標營益率 −3pp、WACC +1pp；bull：成長 +25%|g|、+2pp、−0.5pp（不對稱、有界；|g| 讓負成長不反轉）。"""
    p = prob or DEFAULTS["prob"]
    lo, hi = DEFAULTS["growth_floor"], DEFAULTS["growth_cap"]
    bear = value_drivers(d, rev_g=[_clip(g - 0.5 * abs(g), lo, hi) for g in d["rev_g"]],   # 衝擊用 |g|（Med-2）
                         opm_target=d["opm_target"] - 0.03, wacc=d["wacc"] + 0.01,
                         tgr=min(d["tgr"], d["wacc"] + 0.01 - 0.02), roic_tv=max(d["roic_tv"], d["wacc"] + 0.01))
    base = value_drivers(d)
    bull = value_drivers(d, rev_g=[_clip(g + 0.25 * abs(g), lo, hi) for g in d["rev_g"]],
                         opm_target=d["opm_target"] + 0.02, wacc=max(d["wacc"] - 0.005, d["tgr"] + 0.02))
    vals = [x["per_share"] for x in (bear, base, bull)]
    ev_w = sum(pp * v for pp, v in zip(p, vals)) if all(v is not None for v in vals) else None
    return {"bear": bear["per_share"], "base": base["per_share"], "bull": bull["per_share"],
            "prob": list(p), "ev_weighted": ev_w, "base_detail": base,
            "width": (bull["per_share"] / bear["per_share"]) if (bear["per_share"] and bear["per_share"] > 0 and bull["per_share"]) else None}


def monte_carlo(d: dict, n: int | None = None, seed: int | None = None) -> dict:
    """成長乘數 N(1, 0.25)、營益率 N(0, 2pp)、WACC N(0, 1pp)、終端 g U(1.5%, 3.5%)（相關性：成長↑→營益率↑ 0.4）。"""
    n = int(n or DEFAULTS["mc_n"])
    rng = random.Random(DEFAULTS["mc_seed"] if seed is None else seed)
    lo, hi = DEFAULTS["growth_floor"], DEFAULTS["growth_cap"]
    vals = []
    for _ in range(n):
        gm = rng.gauss(1.0, 0.25)
        z = rng.gauss(0, 1)
        opm_shift = 0.02 * (0.4 * (gm - 1.0) / 0.25 + math.sqrt(1 - 0.16) * z)
        wacc = max(d["wacc"] + rng.gauss(0, 0.01), 0.05)
        tgr = min(rng.uniform(0.015, 0.035), wacc - 0.02)
        try:
            v = value_drivers(d, rev_g=[_clip(g + (gm - 1.0) * abs(g), lo, hi) for g in d["rev_g"]],
                              opm_target=d["opm_target"] + opm_shift, wacc=wacc, tgr=tgr,
                              roic_tv=max(d["roic_tv"], wacc))["per_share"]
        except Exception:
            v = None
        if v is not None and math.isfinite(v):
            vals.append(v)
    if len(vals) < 50:
        return {"n": len(vals)}
    vals.sort()
    q = lambda p: vals[min(len(vals) - 1, int(p * len(vals)))]
    px = d.get("price")
    return {"n": len(vals), "p5": q(0.05), "p25": q(0.25), "p50": q(0.5), "p75": q(0.75), "p95": q(0.95),
            "prob_undervalued": (sum(1 for v in vals if v > px) / len(vals)) if px else None,
            "price_pctile": (sum(1 for v in vals if v < px) / len(vals)) if px else None}


def sensitivity(d: dict, wacc_steps=(-0.02, -0.01, 0, 0.01, 0.02), g_steps=(-0.01, -0.005, 0, 0.005, 0.01)) -> dict:
    waccs = [round(d["wacc"] + s, 4) for s in wacc_steps]
    gs = [round(d["tgr"] + s, 4) for s in g_steps]
    grid = []
    for w in waccs:
        row = []
        for g in gs:
            if g >= w - 0.01 or g < 0:
                row.append(None)
                continue
            try:
                row.append(value_drivers(d, wacc=w, tgr=g, roic_tv=max(d["roic_tv"], w))["per_share"])
            except Exception:
                row.append(None)
        grid.append(row)
    return {"wacc": waccs, "g": gs, "grid": grid}


def reverse_dcf(d: dict, price: float | None = None) -> dict:
    """市價隱含：以乘數 k 縮放 5 年成長路徑使每股價值 = 市價（二分法）；回隱含 5 年營收 CAGR。
    另解「固定成長、隱含目標營益率」。"""
    px = price if price else d.get("price")
    if not px or px <= 0:
        return {"implied_cagr": None, "note": "no price"}
    lo_g, hi_g = DEFAULTS["growth_floor"], DEFAULTS["growth_cap"]

    def val_k(k):
        try:
            return value_drivers(d, rev_g=[_clip(g * k, lo_g, hi_g) for g in d["rev_g"]])["per_share"]
        except Exception:
            return None

    def solve(fn, lo, hi, iters=60):
        flo, fhi = fn(lo), fn(hi)
        if flo is None or fhi is None:
            return None
        if (flo - px) * (fhi - px) > 0:
            return None
        for _ in range(iters):
            mid = (lo + hi) / 2
            fm = fn(mid)
            if fm is None:
                return None
            if (fm - px) * (flo - px) <= 0:
                hi, fhi = mid, fm
            else:
                lo, flo = mid, fm
        return (lo + hi) / 2
    k = solve(val_k, -3.0, 6.0)
    implied_path = [_clip(g * k, lo_g, hi_g) for g in d["rev_g"]] if k is not None else None
    implied_cagr = (math.prod(1 + g for g in implied_path) ** (1 / len(implied_path)) - 1) if implied_path else None
    base_cagr = math.prod(1 + g for g in d["rev_g"]) ** (1 / len(d["rev_g"])) - 1

    def val_m(m):
        try:
            return value_drivers(d, opm_target=m)["per_share"]
        except Exception:
            return None
    m = solve(val_m, -0.5, 0.6)
    return {"implied_cagr": implied_cagr, "model_cagr": base_cagr,
            "gap_pp": (implied_cagr - base_cagr) if implied_cagr is not None else None,
            "implied_opm": m, "model_opm": d["opm_target"],
            "note": None if k is not None else "市價超出可解範圍（成長乘數 −3~6 內無解）"}


# ── 5. 審核清單與訊號 ───────────────────────────────────────────────────────

def audit(d: dict, base: dict, sc: dict) -> dict:
    checks = {
        "g_lt_wacc": d["tgr"] < d["wacc"],
        "g_le_cap": d["tgr"] <= min(max(d["rf"], 0.02) + 0.01, 0.04),
        "tv_pct_ok": base.get("tv_pct") is not None and base["tv_pct"] <= 0.75,   # >75% 才算失敗；偏低只是保守
        "implied_tv_multiple_ok": base.get("implied_tv_ebitda") is None or base["implied_tv_ebitda"] <= 25,
        "roic_consistency_ok": d["roic_tv"] >= d["wacc"],
        "scenario_width_ok": sc.get("width") is not None and sc["width"] >= 1.8,
        "shares_ok": bool(d.get("shares")) and d["shares"] > 0,
        "equity_positive": base.get("equity") is not None and base["equity"] > 0,
        "growth_sane": all(DEFAULTS["growth_floor"] <= g <= DEFAULTS["growth_cap"] for g in d["rev_g"]),
    }
    failed = [k for k, v in checks.items() if not v]
    return {"checks": checks, "failed": failed, "passed": not failed}


def mos_threshold(quality_score: float | None, veto: bool) -> float:
    if veto:
        return 9.9
    if quality_score is None:
        return 0.35
    return 0.25 if quality_score >= 0.5 else (0.35 if quality_score >= 0 else 0.50)


def signal(sc: dict, price: float | None, quality: dict | None = None, audit_passed: bool = True) -> dict:
    base, bear, bull = sc.get("base"), sc.get("bear"), sc.get("bull")
    q = quality or {}
    if not price or base is None:
        return {"verdict": "review", "reason": "缺市價或估值"}
    mos = base / price - 1
    pos = ((price - bear) / (bull - bear)) if (bear is not None and bull is not None and bull > bear) else None
    updown = ((bull - price) / (price - bear)) if (bear is not None and bull is not None and bear < price < bull) else None
    th = mos_threshold(q.get("score"), bool(q.get("veto")))
    if not audit_passed:
        v, why = "review", "審核未過"
    elif q.get("veto"):
        v, why = "review", "會計/破產旗標否決"
    elif pos is not None and pos > 1.0:
        v, why = "exit", "市價高於 bull 情境"
    elif pos is not None and pos > 0.7:
        v, why = "trim", "區間位置 >0.7"
    elif mos >= th and (updown is None or updown >= 2.0):
        v, why = "accumulate", f"MoS {mos:+.0%} ≥ 門檻 {th:.0%}"
    else:
        v, why = "hold", f"MoS {mos:+.0%} < 門檻 {th:.0%}" if mos < th else "上/下行比不足 2:1"
    return {"verdict": v, "reason": why, "mos": mos, "mos_threshold": th, "range_pos": pos,
            "up_down_ratio": updown, "fair_value": base}


# ── 6. 端到端 ───────────────────────────────────────────────────────────────

def run_model(periods: list[dict], profile: dict, estimates: dict | None = None, overrides: dict | None = None,
              quality: dict | None = None, cfg: dict | None = None, mc: bool = True) -> dict:
    sector = (profile or {}).get("sector")
    if sector in NON_DCF_SECTORS:
        return {"ticker": (profile or {}).get("ticker"), "method": "not_applicable", "sector": sector,
                "note": NON_DCF_SECTORS[sector], "signal": {"verdict": "review", "reason": "產業路由：DCF 不適用"}}
    d = derive_drivers(periods, profile, estimates, overrides, cfg)
    sc = scenarios(d)
    base = sc["base_detail"]
    au = audit(d, base, sc)
    sig = signal(sc, d.get("price"), quality, au["passed"])
    out = {"ticker": (profile or {}).get("ticker"), "method": "dcf_fcff", "drivers": d,
           "base": {k: v for k, v in base.items() if k != "rows"}, "rows": base["rows"],
           "scenarios": {k: sc[k] for k in ("bear", "base", "bull", "prob", "ev_weighted", "width")},
           "reverse": reverse_dcf(d), "sensitivity": sensitivity(d), "audit": au, "signal": sig,
           "quality_score": (quality or {}).get("score"), "quality_flags": (quality or {}).get("flags", [])}
    if mc:
        out["mc"] = monte_carlo(d, cfg.get("mc_n") if cfg else None)
    return out


def _pct(x, d=1):
    return "—" if x is None else f"{x:+.{d}%}"


def _money(x) -> str:
    """金額自適應單位（yfinance 為原始美元；手工/測試資料可能已是百萬）。"""
    if x is None:
        return "—"
    ax = abs(x)
    if ax >= 1e9:
        return f"{x / 1e9:.2f}B"
    if ax >= 1e6:
        return f"{x / 1e6:.0f}M"
    return f"{x:,.0f}"


def model_text(res: dict, ticker: str = "") -> str:
    """Telegram legacy Markdown（單 *、無底線）。"""
    t = ticker or res.get("ticker") or ""
    if res.get("method") == "not_applicable":
        return f"🏛️ *{t} 公司模型*\n{res.get('note')}\n{res['signal']['reason']}；非投資建議"
    d, b, sc, rv, sig, au = res["drivers"], res["base"], res["scenarios"], res["reverse"], res["signal"], res["audit"]
    px = d.get("price")
    lines = [f"🏛️ *{t} 公司模型*（FCFF DCF；基期 {d.get('period_end')}）"]
    lines.append(f"營收 {_money(d['base_revenue'])} → 5 年成長 " + "/".join(f"{g:+.0%}" for g in d["rev_g"])
                 + f"（{d['provenance'].get('g1', '')}）｜營益率 {d['opm_last']:.1%}→{d['opm_target']:.1%}")
    braw = "" if d.get("beta_raw") is None else f" 原 {d['beta_raw']:.2f}"
    lines.append(f"WACC {d['wacc']:.1%}（β {d['beta']:.2f}{braw}、"
                 f"rf {d['rf']:.2%}、Rd {d['cost_debt']:.2%}）｜終端 g {d['tgr']:.1%}、ROIC 終值 {d['roic_tv']:.1%}｜稅 {d['tax']:.0%}")
    if sc.get("base") is not None:
        lines.append(f"公允價值：熊 {sc['bear']:.0f}｜*基 {sc['base']:.0f}*｜牛 {sc['bull']:.0f}"
                     + (f"｜機率加權 {sc['ev_weighted']:.0f}" if sc.get("ev_weighted") else "")
                     + (f"｜現價 {px:.0f}（MoS {_pct(sig.get('mos'), 0)}）" if px else ""))
    if res.get("mc", {}).get("p50"):
        m = res["mc"]
        lines.append(f"蒙地卡羅 P5–P95 {m['p5']:.0f}–{m['p95']:.0f}（中位 {m['p50']:.0f}）"
                     + (f"｜低估機率 {m['prob_undervalued']:.0%}" if m.get("prob_undervalued") is not None else ""))
    if rv.get("implied_cagr") is not None:
        lines.append(f"反向 DCF：市價隱含 5 年營收 CAGR {rv['implied_cagr']:+.1%}（模型 {rv['model_cagr']:+.1%}，差 {_pct(rv['gap_pp'])}）"
                     + (f"；隱含營益率 {rv['implied_opm']:.1%}" if rv.get("implied_opm") is not None else ""))
    elif rv.get("note"):
        lines.append(f"反向 DCF：{rv['note']}")
    lines.append(f"終值占 EV {b['tv_pct']:.0%}｜隱含終值 EV/EBITDA {b['implied_tv_ebitda']:.1f}x" if b.get("tv_pct") is not None and b.get("implied_tv_ebitda") else "")
    if sig.get("range_pos") is not None:
        lines.append(f"區間位置 {sig['range_pos']:.2f}（0=熊 1=牛）｜上/下行比 {sig['up_down_ratio']:.1f}" if sig.get("up_down_ratio") else f"區間位置 {sig['range_pos']:.2f}")
    verdict_lab = {"accumulate": "🟢 可累積", "hold": "🟡 持有", "trim": "🟠 減碼", "exit": "🔴 高於牛市情境", "review": "⚪ 待審"}
    lines.append(f"判定 {verdict_lab.get(sig['verdict'], sig['verdict'])}：{sig.get('reason', '')}")
    if not au["passed"]:
        lines.append("審核未過：" + "、".join(k.replace("_", "·") for k in au["failed"]))
    if res.get("quality_flags"):
        lines.append("品質旗標：" + "、".join(x.replace("_", "·") for x in res["quality_flags"]))
    if d.get("overrides"):
        cl = d.get("clamped") or {}
        lines.append("人工覆蓋：" + "、".join(
            f"{PARAM_LABELS.get(k, k).replace('_', '·')}={v}" + (f"（已夾至 {cl[k]:.1%}）" if k in cl else "")
            for k, v in d["overrides"].items()))
    lines.append("估值層只調部位、不觸發進場；DCF 對 WACC 極敏感，看區間不看單點；非投資建議")
    return "\n".join(x for x in lines if x)


# ── 7. 自我測試 ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1) 折現引擎對齊使用者 VRT Excel：其 FCF 流 2026–2035、WACC 14.83%、g 3%、期末折現、
    #    終值 = 最後一年 FCF×(1+g)、淨負債 1,330、382.6m 股。
    #    Excel 明確現金流 PV 19,500、終值 61,590 須對齊；但 Excel 的 DCF!G44 = G45*R10 把終值用
    #    2031E（n=6）的折現因子折現（應為 2035E、n=10）→ 每股 117.7 被高估，正確為 ≈ 87.9
    #    （VALUATION_PLAN §1.3 #9）。引擎以正確數學為準，並重現 Excel 的錯誤值作為對照。
    vrt_fcf = [1867, 2184, 2834, 3639, 4358, 5201, 5617, 6066, 6552, 7076]
    rows = [{"fcf": f, "nopat": f, "ebit": f, "da": 0} for f in vrt_fcf]
    r = dcf(rows, 0.1483, 0.03, 1330, 382.6, mid_year=False, terminal_fcf=vrt_fcf[-1] * 1.03)
    assert abs(r["pv_fcfs"] - 19500) / 19500 < 0.01 and abs(r["tv"] - 61590) / 61590 < 0.01, (r["pv_fcfs"], r["tv"])
    assert abs(r["per_share"] - 87.9) / 87.9 < 0.01, r["per_share"]
    excel_replica = (r["pv_fcfs"] + r["tv"] / 1.1483 ** 6 - 1330) / 382.6
    assert abs(excel_replica - 117.7) / 117.7 < 0.01, excel_replica
    r_mid = dcf(rows, 0.1483, 0.03, 1330, 382.6, mid_year=True, terminal_fcf=vrt_fcf[-1] * 1.03)
    assert 1.03 < r_mid["per_share"] / r["per_share"] < 1.10            # 期中慣例 +3–10%
    print(f"✅ 1 折現引擎對齊 VRT Excel：明確 PV {r['pv_fcfs']:,.0f}、終值 {r['tv']:,.0f}；"
          f"正確每股 {r['per_share']:.1f}（Excel 因終值折現期錯誤得 {excel_replica:.1f}）；期中慣例 {r_mid['per_share']:.1f}")

    # 2) 驅動推導：指引 > 共識 > 歷史；Blume β；合成利差；覆蓋
    def yr(pe, rev, oi, ni, cfo, capex, da, ar, inv, ap, ta, eq, debt, cash, sh, ie, tax, pti, sbc=0):
        return {"period_end": pe, "freq": "A", "revenue": rev, "operating_income": oi, "ebit": oi, "net_income": ni,
                "cfo": cfo, "capex": -capex, "da": da, "receivables": ar, "inventory": inv, "payables": ap,
                "total_assets": ta, "total_equity": eq, "total_debt": debt, "cash_and_sti": cash,
                "diluted_shares": sh, "interest_expense": ie, "tax_provision": tax, "pretax_income": pti, "sbc": sbc}
    periods = [yr("2025-12-31", 10229, 1830, 1333, 2114, 220, 309, 3109, 1456, 1756, 12212, 3941, 2913, 1828, 390.65, 86, 409, 1742, 46),
               yr("2024-12-31", 8012, 1367, 496, 1319, 167, 277, 2363, 1244, 1316, 9132, 2434, 2928, 1228, 386.3, 150, 270, 765, 35),
               yr("2023-12-31", 6863, 872, 460, 900, 128, 271, 2185, 884, 986, 7998, 2015, 2941, 780, 386.2, 180, 74, 534, 25)]
    profile = {"ticker": "VRT", "price": 268.8, "mkt_cap": 98550, "beta": 2.08, "shares": 382.6, "sector": "Industrials"}
    d = derive_drivers(periods, profile)
    assert d["provenance"]["g1"] == "history" and abs(d["rev_g"][0] - (10229 / 8012 - 1)) < 1e-4
    assert abs(d["beta"] - (0.67 * 2.08 + 0.33)) < 1e-9 and d["wacc"] < 0.1483 and d["tgr"] < d["wacc"]
    assert d["roic_tv"] >= d["wacc"] and d["net_debt"] == 2913 - 1828
    d_est = derive_drivers(periods, profile, estimates={"est": {"0y": {"rev_growth": 0.37}, "+1y": {"rev_growth": 0.29}}})
    assert d_est["provenance"]["g1"].startswith("consensus") and abs(d_est["rev_g"][1] - 0.29) < 1e-4
    d_gd = derive_drivers(periods, profile, overrides={"guidance_rev": 14000, "opm_target": 0.24, "beta": 1.4, "tgr": 0.03})
    assert d_gd["provenance"]["g1"] == "guidance" and abs(d_gd["rev_g"][0] - (14000 / 10229 - 1)) < 1e-4 and d_gd["beta"] == 1.4
    assert synthetic_spread(50) < synthetic_spread(2.1) < synthetic_spread(0.5)
    # 共識水準推導：0y 營收共識 14,000 → g1 = 36.9%；快照錯位（0y 其實是基期那年）→ 用 +1y 對齊
    d_lvl = derive_drivers(periods, profile, estimates={"est": {"0y": {"rev_avg": 14000, "rev_growth": 0.37},
                                                                 "+1y": {"rev_avg": 18110, "rev_growth": 0.29}}})
    assert d_lvl["provenance"]["g1"] == "consensus" and abs(d_lvl["rev_g"][0] - (14000 / 10229 - 1)) < 1e-3
    assert abs(d_lvl["rev_g"][1] - (18110 / 14000 - 1)) < 1e-3
    d_mis = derive_drivers(periods, profile, estimates={"est": {"0y": {"rev_avg": 10229, "rev_growth": 0.277},
                                                                 "+1y": {"rev_avg": 13060, "rev_growth": 0.277}}})
    assert d_mis["provenance"]["g1"].startswith("consensus(+1y") and abs(d_mis["rev_g"][0] - (13060 / 10229 - 1)) < 1e-3
    # High-1：SBC 不重複扣——SBC 由 0 → 15% 營收，FCF 不變（sbc_as_cost=True 即 GAAP EBIT 已含）
    hi_sbc = [dict(p, sbc=p["revenue"] * 0.15) for p in periods]
    v0 = value_drivers(derive_drivers(periods, profile))["per_share"]
    v1 = value_drivers(derive_drivers(hi_sbc, profile))["per_share"]
    assert abs(v0 - v1) < 1e-6, (v0, v1)
    v2 = value_drivers(derive_drivers(hi_sbc, profile, cfg={"sbc_as_cost": False}))["per_share"]
    assert v2 > v1                                                    # 加回 SBC 才會變高
    # Med-1：營益率目標——最新高於均值時取中點，不鎖高點
    assert d["provenance"]["opm_target"] == "mid(last, 3y avg)" and d["opm_last"] > d["opm_target"] > 0.12
    # Med-2：負成長公司 bear < base < bull 且寬度合理
    shrink = [dict(p) for p in periods]; shrink[0]["revenue"], shrink[1]["revenue"], shrink[2]["revenue"] = 6000, 7500, 9000
    d_neg = derive_drivers(shrink, {**profile, "price": 30})
    assert d_neg["rev_g"][0] < 0
    sc_neg = scenarios(d_neg)
    assert sc_neg["bear"] < sc_neg["base"] < sc_neg["bull"] and sc_neg["width"] > 1.3, sc_neg
    print(f"✅ 2 驅動推導（WACC {d['wacc']:.1%}、β Blume {d['beta']:.2f}、Rd {d['cost_debt']:.2%}）")

    # 3) 投影/DCF/情境：單調性（WACC↑價值↓、成長↑價值↑）、bear<base<bull、寬度、終值價值中性
    res = run_model(periods, profile, overrides={"rev_g": "0.368,0.294,0.224,0.187,0.150", "opm_target": 0.24}, mc=True)
    sc, b = res["scenarios"], res["base"]
    assert sc["bear"] < sc["base"] < sc["bull"] and sc["width"] > 1.5, sc
    assert value_drivers(res["drivers"], wacc=res["drivers"]["wacc"] + 0.02)["per_share"] < b["per_share"]
    assert len(res["rows"]) == 10 and abs(res["rows"][-1]["growth"] - res["drivers"]["tgr"]) < 1e-9
    hi_roic = value_drivers(res["drivers"], roic_tv=0.5)["per_share"]
    assert hi_roic > b["per_share"]                                   # 終值 ROIC 高於 WACC 才有成長價值
    assert res["mc"]["n"] > 1000 and res["mc"]["p5"] < res["mc"]["p50"] < res["mc"]["p95"]
    print(f"✅ 3 情境 熊/基/牛 {sc['bear']:.0f}/{sc['base']:.0f}/{sc['bull']:.0f}（寬 {sc['width']:.1f}x）"
          f"｜MC P5–P95 {res['mc']['p5']:.0f}–{res['mc']['p95']:.0f}｜終值占比 {b['tv_pct']:.0%}")

    # 4) 反向 DCF：把市價設成 base 公允價 → 隱含 CAGR ≈ 模型 CAGR；市價越高隱含越高
    rv_eq = reverse_dcf(res["drivers"], price=sc["base"])
    assert rv_eq["implied_cagr"] is not None and abs(rv_eq["gap_pp"]) < 0.005, rv_eq
    rv_hi = reverse_dcf(res["drivers"], price=sc["base"] * 1.5)
    assert rv_hi["implied_cagr"] > rv_eq["implied_cagr"]
    print(f"✅ 4 反向 DCF（市價 268.8 隱含 CAGR {res['reverse']['implied_cagr']:+.1%} vs 模型 {res['reverse']['model_cagr']:+.1%}）"
          if res["reverse"]["implied_cagr"] is not None else "✅ 4 反向 DCF（市價超出可解範圍，已標示）")

    # 5) 審核與訊號：g≥WACC 被夾、審核失敗→review、品質否決→review、區間位置判定
    au = res["audit"]; assert "g_lt_wacc" not in au["failed"] and "roic_consistency_ok" not in au["failed"]
    s_ok = signal({"bear": 80, "base": 130, "bull": 200}, 100, {"score": 0.6, "veto": False}, True)
    assert s_ok["verdict"] == "accumulate" and abs(s_ok["mos"] - 0.30) < 1e-9
    assert signal({"bear": 80, "base": 130, "bull": 200}, 100, {"score": 0.6, "veto": True}, True)["verdict"] == "review"
    assert signal({"bear": 80, "base": 130, "bull": 200}, 100, {"score": 0.6, "veto": False}, False)["verdict"] == "review"
    assert signal({"bear": 80, "base": 130, "bull": 200}, 210, {"score": 0.6}, True)["verdict"] == "exit"
    assert signal({"bear": 80, "base": 130, "bull": 200}, 180, {"score": 0.6}, True)["verdict"] == "trim"
    assert signal({"bear": 80, "base": 130, "bull": 200}, 120, {"score": -0.5}, True)["verdict"] == "hold"   # 低品質門檻 50%
    assert run_model(periods, {**profile, "sector": "Financial Services"})["method"] == "not_applicable"
    print("✅ 5 審核清單 + 訊號判定 + 產業路由")

    # 6) 文字輸出 Markdown 安全
    txt = model_text(res, "VRT"); txt2 = model_text(run_model(periods, {**profile, "sector": "Real Estate"}), "O")
    for tx in (txt, txt2):
        assert tx.count("*") % 2 == 0 and "_" not in tx, tx
    print(txt)
    print("\ncompany_model selftest OK ✅")
