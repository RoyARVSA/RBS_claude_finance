"""
valuation.py — 內在價值估值：DCF + 可比公司分析（Comps）
RBS Finance Dashboard

方法論依據 Anthropic 官方 financial-services skills（anthropics/financial-services，
Apache 2.0）的 dcf-model 與 comps-analysis：
- FCF 五步驟：EBIT → NOPAT → +D&A − CapEx − ΔNWC = 無槓桿自由現金流
- WACC：CAPM 股權成本（rf + β×ERP）+ 稅後債務成本，市值權重
- 期中折現慣例（mid-year convention）：折現期 0.5, 1.5, 2.5…
- 終值：永續成長法（Gordon），硬約束 g < WACC；終值佔 EV 50-70% 為健康區間
- 股權橋：EV − 淨負債 → 股權價值 ÷ 稀釋股數 = 隱含股價
- 敏感度表：WACC × 終端成長雙軸
- Comps：同業倍數（EV/Rev、EV/EBITDA、P/E）統計區間 → 中位數倍數回推隱含價值

純邏輯（project_fcf / calc_wacc / dcf_value / sensitivity_grid / comps_table）離線可測；
fetch_dcf_inputs / fetch_comps 需網路（yfinance）。教育用途，非投資建議。
"""
from __future__ import annotations

import math

# 預設假設（官方 skill 的典型區間）
DEFAULT_ERP = 0.055          # 股權風險溢酬 5.5%（市場慣例 5.0-6.0%）
DEFAULT_RF = 0.042           # 無風險利率備援值（抓不到 ^TNX 時）
DEFAULT_TERM_GROWTH = 0.025  # 終端成長 2.5%（保守=GDP 區間）
DEFAULT_YEARS = 5            # 標準預測期


# ── 純邏輯：FCF 投影 ─────────────────────────────────────────────────────────

def project_fcf(base_revenue: float, rev_growth: list[float] | float,
                ebit_margin: float, tax_rate: float,
                da_pct: float = 0.04, capex_pct: float = 0.05,
                nwc_pct: float = 0.01, years: int = DEFAULT_YEARS) -> list[dict]:
    """
    EBIT → NOPAT → +D&A − CapEx − ΔNWC = 無槓桿 FCF。
    rev_growth：單一成長率或逐年清單（不足補最後一個）。
    D&A / CapEx 以營收比、ΔNWC 以「營收增量」比（官方 skill 慣例，典型 -2%~+2%）。
    回逐年 [{year, revenue, ebit, nopat, da, capex, dnwc, fcf}]。
    """
    if base_revenue <= 0 or years < 1:
        return []
    g = list(rev_growth) if isinstance(rev_growth, (list, tuple)) else [float(rev_growth)]
    g += [g[-1]] * (years - len(g))
    rows, rev_prev = [], float(base_revenue)
    for yr in range(1, years + 1):
        rev = rev_prev * (1 + g[yr - 1])
        ebit = rev * ebit_margin
        nopat = ebit * (1 - tax_rate)
        da = rev * da_pct
        capex = rev * capex_pct
        dnwc = (rev - rev_prev) * nwc_pct        # ΔNWC 按營收「增量」
        fcf = nopat + da - capex - dnwc
        rows.append({"year": yr, "revenue": rev, "ebit": ebit, "nopat": nopat,
                     "da": da, "capex": capex, "dnwc": dnwc, "fcf": fcf})
        rev_prev = rev
    return rows


# ── 純邏輯：WACC（CAPM）─────────────────────────────────────────────────────

def calc_wacc(rf: float, beta: float, mkt_cap: float,
              total_debt: float = 0.0, cash: float = 0.0,
              erp: float = DEFAULT_ERP, cost_debt_pre: float | None = None,
              tax_rate: float = 0.21) -> dict:
    """
    Cost of Equity = rf + β×ERP；稅後債務成本 = 稅前 ×(1−稅率)。
    權重用市值：EV = 市值 + 淨負債。淨現金（Cash>Debt）或無債 → WACC = 股權成本
    （官方 skill：No Debt: WACC = Cost of Equity；淨現金的負權重 WACC 業界普遍
    視為病態，捨棄不用——此慣例會比負權重法給出略低的折現率/略高的估值）。
    cost_debt_pre 缺省用 rf + 1.5% 信用利差近似。
    """
    beta = beta if beta and beta > 0 else 1.0
    coe = rf + beta * erp
    net_debt = float(total_debt) - float(cash)
    if net_debt <= 0 or mkt_cap <= 0:
        return {"wacc": coe, "cost_equity": coe, "cost_debt_at": 0.0,
                "w_equity": 1.0, "w_debt": 0.0, "net_debt": net_debt}
    cdp = cost_debt_pre if cost_debt_pre is not None else rf + 0.015
    cda = cdp * (1 - tax_rate)
    ev = mkt_cap + net_debt
    we, wd = mkt_cap / ev, net_debt / ev
    return {"wacc": coe * we + cda * wd, "cost_equity": coe, "cost_debt_at": cda,
            "w_equity": we, "w_debt": wd, "net_debt": net_debt}


# ── 純邏輯：DCF 核心 ─────────────────────────────────────────────────────────

def dcf_value(fcfs: list[float], wacc: float, terminal_growth: float,
              net_debt: float, shares: float,
              mid_year: bool = True) -> dict:
    """
    期中折現 + 永續成長終值 + 股權橋。
    回 {pv_fcfs, pv_terminal, ev, equity_value, per_share, tv_pct, warnings}。
    g ≥ WACC 直接 raise ValueError（數學上無意義——官方 skill 的硬約束）。
    """
    if not fcfs:
        raise ValueError("FCF 清單為空")
    if terminal_growth >= wacc:
        raise ValueError(f"終端成長 {terminal_growth:.1%} ≥ WACC {wacc:.1%}——"
                         "永續成長法無解（官方硬約束：g < WACC）")
    n = len(fcfs)
    pv_sum = 0.0
    for i, fcf in enumerate(fcfs, start=1):
        period = i - 0.5 if mid_year else i
        pv_sum += fcf / (1 + wacc) ** period
    term_fcf = fcfs[-1] * (1 + terminal_growth)
    tv = term_fcf / (wacc - terminal_growth)
    final_period = (n - 0.5) if mid_year else n     # 官方：5 年期中慣例用 4.5
    pv_tv = tv / (1 + wacc) ** final_period
    ev = pv_sum + pv_tv
    equity = ev - net_debt
    per_share = equity / shares if shares and shares > 0 else None
    tv_pct = pv_tv / ev if ev > 0 else None

    warnings = []
    if tv_pct is not None:
        if tv_pct > 0.75:
            warnings.append(f"終值佔 EV {tv_pct:.0%}（>75%）——估值過度依賴終端假設")
        elif tv_pct < 0.40:
            warnings.append(f"終值佔 EV {tv_pct:.0%}（<40%）——終端假設可能過度保守")
    if equity <= 0:
        warnings.append("股權價值 ≤ 0（淨負債高於企業價值）")
    return {"pv_fcfs": pv_sum, "pv_terminal": pv_tv, "ev": ev,
            "equity_value": equity, "per_share": per_share,
            "tv_pct": tv_pct, "warnings": warnings}


def sensitivity_grid(fcfs: list[float], net_debt: float, shares: float,
                     wacc_list: list[float], growth_list: list[float],
                     mid_year: bool = True) -> list[list[float | None]]:
    """WACC（列）× 終端成長（欄）的每股價值格網；g ≥ WACC 的格為 None。"""
    grid = []
    for w in wacc_list:
        row = []
        for g in growth_list:
            if g >= w:
                row.append(None)
                continue
            try:
                row.append(dcf_value(fcfs, w, g, net_debt, shares,
                                     mid_year=mid_year)["per_share"])
            except (ValueError, ZeroDivisionError):
                row.append(None)
        grid.append(row)
    return grid


# ── 純邏輯：可比公司分析（Comps）──────────────────────────────────────────────

def _pctile(sorted_vals: list[float], q: float) -> float:
    """線性插值百分位（sorted_vals 已排序、非空）。"""
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def comps_table(target: dict, peers: list[dict]) -> dict:
    """
    target/peers 元素：{ticker, mkt_cap, ev, revenue, ebitda, net_income}
    （ev 缺省 = mkt_cap + net_debt，由呼叫端先算好；缺值傳 None）。
    計算各檔 EV/Rev、EV/EBITDA、P/E → 統計區間（min/P25/中位/P75/max）→
    以同業「中位數」倍數回推 target 的隱含股權價值（官方 skill 的標準輸出）。
    回 {rows, stats, implied}；同業 <2 檔回 implied=None。
    """
    def _mult(p):
        ev, mc = p.get("ev"), p.get("mkt_cap")
        rev, ebitda, ni = p.get("revenue"), p.get("ebitda"), p.get("net_income")
        return {
            "ticker": p.get("ticker", "?"),
            "ev_rev": (ev / rev) if ev and rev and rev > 0 else None,
            "ev_ebitda": (ev / ebitda) if ev and ebitda and ebitda > 0 else None,
            "pe": (mc / ni) if mc and ni and ni > 0 else None,
        }

    rows = [_mult(p) for p in peers]
    stats = {}
    for k in ("ev_rev", "ev_ebitda", "pe"):
        vals = sorted(r[k] for r in rows if r[k] is not None and r[k] > 0)
        stats[k] = None if not vals else {
            "min": vals[0], "p25": _pctile(vals, 0.25),
            "median": _pctile(vals, 0.50), "p75": _pctile(vals, 0.75),
            "max": vals[-1], "n": len(vals)}

    implied = None
    usable = [k for k in stats if stats[k] and stats[k]["n"] >= 2]
    if usable:
        net_debt = (target.get("ev") or 0) - (target.get("mkt_cap") or 0)
        parts = {}
        if stats.get("ev_rev") and stats["ev_rev"]["n"] >= 2 and target.get("revenue"):
            parts["ev_rev"] = stats["ev_rev"]["median"] * target["revenue"] - net_debt
        if stats.get("ev_ebitda") and stats["ev_ebitda"]["n"] >= 2 and target.get("ebitda"):
            parts["ev_ebitda"] = stats["ev_ebitda"]["median"] * target["ebitda"] - net_debt
        if stats.get("pe") and stats["pe"]["n"] >= 2 and target.get("net_income") \
                and target["net_income"] > 0:
            parts["pe"] = stats["pe"]["median"] * target["net_income"]
        if parts:
            implied = {"by_multiple": parts,
                       "avg_equity": sum(parts.values()) / len(parts)}
    return {"rows": rows, "stats": stats, "implied": implied,
            "target": _mult(target)}


# ── 文字輸出（Bot / 下載共用）─────────────────────────────────────────────────

def dcf_text(ticker: str, assumptions: dict, wacc_d: dict, val: dict,
             current_price: float | None = None) -> str:
    a = assumptions
    lines = [f"💰 *{ticker} DCF 估值*（{a.get('years', 5)} 年期中折現＋永續成長）",
             f"營收成長 {a['rev_growth']:.1%} → EBIT 利潤率 {a['ebit_margin']:.1%} → "
             f"稅率 {a['tax_rate']:.0%}",
             f"WACC {wacc_d['wacc']:.1%}（股權成本 {wacc_d['cost_equity']:.1%}"
             f"，β={a.get('beta', 1.0):.2f}）｜終端成長 {a['terminal_growth']:.1%}",
             f"企業價值 ${val['ev'] / 1e9:.1f}B = 預測期 PV ${val['pv_fcfs'] / 1e9:.1f}B"
             f" + 終值 PV ${val['pv_terminal'] / 1e9:.1f}B"
             + (f"（佔 {val['tv_pct']:.0%}）" if val["tv_pct"] is not None else "（EV≤0）"),
             f"− 淨負債 ${wacc_d['net_debt'] / 1e9:.1f}B → 股權 ${val['equity_value'] / 1e9:.1f}B"]
    if val["per_share"] is not None:
        lines.append(f"*隱含股價 ${val['per_share']:.2f}*")
        if current_price and current_price > 0:
            up = val["per_share"] / current_price - 1
            lines.append(f"現價 ${current_price:.2f} → 隱含空間 {up:+.0%}")
    for w in val["warnings"]:
        lines.append(f"⚠️ {w}")
    lines.append("_假設可調（DCF 對假設極敏感）；教育用途，非投資建議。_")
    return "\n".join(lines)


# ── 抓取層（需網路）───────────────────────────────────────────────────────────

def fetch_dcf_inputs(ticker: str) -> dict | None:
    """
    從 yfinance 抓 DCF 所需輸入並給出可調整的預設假設。
    回 {base_revenue, rev_growth, ebit_margin, tax_rate, da_pct, capex_pct,
        beta, rf, mkt_cap, total_debt, cash, shares, price, years, terminal_growth,
        hist:{...}} 或 None。
    """
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        info = tk.info or {}
        fin = tk.income_stmt
        cf = tk.cashflow

        def _row(df, names):
            for nm in names:
                if df is not None and hasattr(df, "index") and nm in df.index:
                    s = df.loc[nm].dropna()
                    if len(s):
                        return [float(x) for x in s.tolist()]
            return []

        revs = _row(fin, ["Total Revenue", "Operating Revenue"])   # 最新在前
        ebits = _row(fin, ["EBIT", "Operating Income"])
        taxes = _row(fin, ["Tax Provision"])
        pretax = _row(fin, ["Pretax Income"])
        da = _row(cf, ["Depreciation And Amortization", "Depreciation Amortization Depletion"])
        capex = _row(cf, ["Capital Expenditure"])
        if not revs or revs[0] <= 0:
            return None
        base_rev = revs[0]
        # 歷史營收 CAGR（最多 4 年），保守夾在 0-25%
        growth = 0.05
        if len(revs) >= 2 and revs[-1] > 0:
            yrs = len(revs) - 1
            growth = (revs[0] / revs[-1]) ** (1 / yrs) - 1
        growth = max(0.0, min(growth, 0.25))
        ebit_margin = (ebits[0] / base_rev) if ebits else 0.15
        ebit_margin = max(0.02, min(ebit_margin, 0.60))
        tax_rate = 0.21
        if taxes and pretax and pretax[0] > 0:
            tax_rate = max(0.10, min(taxes[0] / pretax[0], 0.35))
        da_pct = min(abs(da[0]) / base_rev, 0.15) if da else 0.04
        capex_pct = min(abs(capex[0]) / base_rev, 0.15) if capex else 0.05

        rf = DEFAULT_RF
        try:
            import yfinance as yf2
            tnx = yf2.Ticker("^TNX").history(period="5d")
            if len(tnx):
                rf = float(tnx["Close"].iloc[-1]) / 100.0
        except Exception:
            pass

        shares = info.get("sharesOutstanding") or 0
        return {"base_revenue": base_rev, "rev_growth": round(growth, 4),
                "ebit_margin": round(ebit_margin, 4), "tax_rate": round(tax_rate, 4),
                "da_pct": round(da_pct, 4), "capex_pct": round(capex_pct, 4),
                "nwc_pct": 0.01, "beta": float(info.get("beta") or 1.0),
                "rf": round(rf, 4), "erp": DEFAULT_ERP,
                "terminal_growth": DEFAULT_TERM_GROWTH, "years": DEFAULT_YEARS,
                "mkt_cap": float(info.get("marketCap") or 0),
                "total_debt": float(info.get("totalDebt") or 0),
                "cash": float(info.get("totalCash") or 0),
                "shares": float(shares), "price": float(info.get("currentPrice")
                                                        or info.get("regularMarketPrice") or 0)}
    except Exception:
        return None


def run_dcf(ticker: str, overrides: dict | None = None) -> tuple[dict, dict, dict, dict] | None:
    """端到端：抓輸入（可覆蓋）→ FCF → WACC → DCF。回 (assumptions, wacc_d, val, rows)。"""
    a = fetch_dcf_inputs(ticker)
    if not a:
        return None
    if overrides:
        a.update({k: v for k, v in overrides.items() if v is not None})
    rows = project_fcf(a["base_revenue"], a["rev_growth"], a["ebit_margin"],
                       a["tax_rate"], a["da_pct"], a["capex_pct"],
                       a["nwc_pct"], a["years"])
    wacc_d = calc_wacc(a["rf"], a["beta"], a["mkt_cap"], a["total_debt"],
                       a["cash"], a["erp"], tax_rate=a["tax_rate"])
    val = dcf_value([r["fcf"] for r in rows], wacc_d["wacc"],
                    a["terminal_growth"], wacc_d["net_debt"], a["shares"])
    return a, wacc_d, val, {"fcf_rows": rows}


def fetch_comps(target: str, peers: list[str]) -> dict | None:
    """抓 target+peers 的市值/EV/營收/EBITDA/淨利 → comps_table。"""
    try:
        import yfinance as yf

        def _one(sym):
            info = yf.Ticker(sym).info or {}
            mc = info.get("marketCap")
            debt, cash = info.get("totalDebt") or 0, info.get("totalCash") or 0
            return {"ticker": sym.upper(), "mkt_cap": mc,
                    "ev": (mc + debt - cash) if mc else None,
                    "revenue": info.get("totalRevenue"),
                    "ebitda": info.get("ebitda"),
                    "net_income": info.get("netIncomeToCommon"),
                    "price": info.get("currentPrice") or info.get("regularMarketPrice"),
                    "shares": info.get("sharesOutstanding")}

        t = _one(target)
        ps = []
        for p in peers:
            try:
                d = _one(p)
                if d["mkt_cap"]:
                    ps.append(d)
            except Exception:
                continue
        if not t["mkt_cap"] or len(ps) < 2:
            return None
        out = comps_table(t, ps)
        out["target_raw"] = t
        return out
    except Exception:
        return None


# ── CLI 自我測試（離線純邏輯，手算對照）────────────────────────────────────────

if __name__ == "__main__":
    # FCF 投影手算對照：營收 1000、成長 10%、EBIT 20%、稅 25%、DA 4%、CapEx 5%、NWC 1%
    rows = project_fcf(1000, 0.10, 0.20, 0.25, 0.04, 0.05, 0.01, years=2)
    r1 = rows[0]
    assert abs(r1["revenue"] - 1100) < 1e-9
    # EBIT=220, NOPAT=165, DA=44, CapEx=55, ΔNWC=(1100-1000)*1%=1 → FCF=153
    assert abs(r1["fcf"] - 153.0) < 1e-9, r1
    r2 = rows[1]                                   # rev=1210, ΔNWC=1.1
    assert abs(r2["fcf"] - (1210 * 0.20 * 0.75 + 1210 * 0.04 - 1210 * 0.05 - 1.1)) < 1e-9

    # WACC 手算：rf 4%、β1.2、ERP 5.5% → CoE=10.6%；市值 800、債 300、現金 100
    w = calc_wacc(0.04, 1.2, 800, 300, 100, erp=0.055, cost_debt_pre=0.05, tax_rate=0.25)
    # 淨負債 200、EV 1000、we=0.8、wd=0.2、稅後債息 3.75% → WACC = 8.48%+0.75% = 9.23%
    assert abs(w["wacc"] - (0.106 * 0.8 + 0.0375 * 0.2)) < 1e-12, w
    # 淨現金 → WACC = CoE
    w2 = calc_wacc(0.04, 1.0, 800, 50, 300)
    assert w2["wacc"] == w2["cost_equity"] and w2["net_debt"] < 0

    # DCF 手算：FCF 100 恆定 5 年、WACC 10%、g 2%、無債、100 股
    fcfs = [100.0] * 5
    v = dcf_value(fcfs, 0.10, 0.02, net_debt=0, shares=100, mid_year=True)
    pv_expl = sum(100 / 1.10 ** (i - 0.5) for i in range(1, 6))
    tv = 100 * 1.02 / 0.08
    pv_tv = tv / 1.10 ** 4.5
    assert abs(v["pv_fcfs"] - pv_expl) < 1e-9 and abs(v["pv_terminal"] - pv_tv) < 1e-9
    assert abs(v["per_share"] - (pv_expl + pv_tv) / 100) < 1e-9
    assert v["tv_pct"] > 0.5                       # 終值占比合理區間

    # 期中 vs 年末：期中折現的 PV 必然較高
    v_ye = dcf_value(fcfs, 0.10, 0.02, 0, 100, mid_year=False)
    assert v["ev"] > v_ye["ev"]

    # 硬約束：g ≥ WACC → ValueError
    try:
        dcf_value(fcfs, 0.05, 0.05, 0, 100)
        raise AssertionError("g>=WACC 未擋下")
    except ValueError:
        pass

    # 終值佔比警告：短期高 WACC 低成長 → TV 低佔比警告；高 g 接近 WACC → 高佔比警告
    v_hi = dcf_value(fcfs, 0.07, 0.045, 0, 100)
    assert any(">75%" in s or "過度依賴" in s for s in v_hi["warnings"]), v_hi

    # 敏感度格網：g >= WACC 的格為 None；WACC 越低價值越高
    grid = sensitivity_grid(fcfs, 0, 100, [0.08, 0.10, 0.12], [0.02, 0.03, 0.09])
    assert grid[0][2] is None                                 # g 9% ≥ WACC 8% → 無解格
    assert grid[2][2] is not None                             # g 9% < WACC 12% → 合法
    assert grid[0][0] > grid[1][0] > grid[2][0]               # WACC 越低估值越高

    # Comps：三檔同業手算
    peers = [
        {"ticker": "A", "mkt_cap": 1000, "ev": 1100, "revenue": 500, "ebitda": 200, "net_income": 80},
        {"ticker": "B", "mkt_cap": 2000, "ev": 2400, "revenue": 800, "ebitda": 300, "net_income": 150},
        {"ticker": "C", "mkt_cap": 600, "ev": 500, "revenue": 400, "ebitda": 100, "net_income": 40},
    ]
    tgt = {"ticker": "T", "mkt_cap": 900, "ev": 1000, "revenue": 450, "ebitda": 180, "net_income": 70}
    ct = comps_table(tgt, peers)
    assert ct["stats"]["ev_ebitda"]["n"] == 3
    med = ct["stats"]["ev_ebitda"]["median"]        # 5.5, 8.0, 5.0 → 中位 5.5
    assert abs(med - 5.5) < 1e-9, ct["stats"]["ev_ebitda"]
    # 隱含：EV/EBITDA 5.5×180=990 − 淨負債(1000-900=100) = 890
    assert abs(ct["implied"]["by_multiple"]["ev_ebitda"] - 890) < 1e-9, ct["implied"]
    # 虧損公司不進 P/E 統計
    peers_neg = peers + [{"ticker": "D", "mkt_cap": 300, "ev": 350,
                          "revenue": 100, "ebitda": 20, "net_income": -50}]
    ct2 = comps_table(tgt, peers_neg)
    assert ct2["stats"]["pe"]["n"] == 3             # D 被排除

    # 文字輸出
    txt = dcf_text("TEST", {"rev_growth": 0.1, "ebit_margin": 0.2, "tax_rate": 0.25,
                            "terminal_growth": 0.025, "beta": 1.1, "years": 5},
                   w, v, current_price=8.0)
    assert "DCF 估值" in txt and "非投資建議" in txt

    # 全負 FCF → EV<0、tv_pct=None：dcf_text 不得崩潰（Med 修復迴歸）
    v_neg = dcf_value([-100.0] * 5, 0.10, 0.02, net_debt=0, shares=100)
    assert v_neg["tv_pct"] is None and any("≤ 0" in s for s in v_neg["warnings"])
    txt_neg = dcf_text("NEG", {"rev_growth": 0.0, "ebit_margin": 0.02, "tax_rate": 0.25,
                               "terminal_growth": 0.02, "beta": 1.0, "years": 5},
                       w, v_neg)
    assert "EV≤0" in txt_neg

    print(f"✅ valuation 離線自我測試通過（DCF 每股 {v['per_share']:.2f}、"
          f"WACC {w['wacc']:.2%}、comps 中位 EV/EBITDA {med:.1f}x）")
