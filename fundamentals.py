"""
fundamentals.py – 公司基本面分析（yfinance 免費資料，可重用，無 Streamlit 依賴）

設計（依自我批判修正）：
  • 抓取層與純邏輯層分離 → 純邏輯可用合成資料完整測試（不依賴 yfinance）
  • 三表列名用「多候選關鍵字」模糊比對，不寫死
  • 評分採「分級給分 + 缺資料重正規化」：缺的構面不計零分，只就有資料的算
  • 估值用絕對合理區間旗標（不做產業平均，避免大量慢呼叫）

對外主要函數：
  fetch_fundamentals(ticker)   抓取並整理（呼叫 yfinance；無法離線測）
  health_score(metrics)        財務健康評分 0~100（純邏輯，可測）
  valuation_flags(metrics)     估值旗標（純邏輯，可測）
  pick_row(df, candidates)     三表模糊取列（純邏輯，可測）
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── 純邏輯工具（可離線單元測試）────────────────────────────────────────────────

def _num(x):
    """安全轉 float，無效值回 None。"""
    try:
        if x is None:
            return None
        f = float(x)
        if np.isnan(f) or np.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _as_series(obj):
    """df.loc[label] 在 index 有重複科目名時會回 DataFrame；取第一列收斂成 Series。"""
    if isinstance(obj, pd.DataFrame):
        return obj.iloc[0] if len(obj) else pd.Series(dtype=float)
    return obj


def pick_row(df: pd.DataFrame, candidates: list[str]):
    """
    從三表 DataFrame（index 為科目名）依候選關鍵字模糊取列（最新一期）。
    不分大小寫、忽略空白；回傳 (label, value) 或 (None, None)。
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None, None
    norm = {str(idx).lower().replace(" ", ""): idx for idx in df.index}
    for cand in candidates:
        key = cand.lower().replace(" ", "")
        # 完全相符優先
        if key in norm:
            row = _as_series(df.loc[norm[key]])
            val = _num(row.iloc[0]) if len(row) else None
            return norm[key], val
    # 退而求其次：包含關係
    for cand in candidates:
        key = cand.lower().replace(" ", "")
        for nk, orig in norm.items():
            if key in nk:
                row = _as_series(df.loc[orig])
                val = _num(row.iloc[0]) if len(row) else None
                return orig, val
    return None, None


def series_row(df: pd.DataFrame, candidates: list[str], n: int = 4):
    """取某科目最近 n 期的數值序列（最舊→最新），供畫趨勢用。"""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return [], []
    norm = {str(idx).lower().replace(" ", ""): idx for idx in df.index}
    target = None
    for cand in candidates:
        key = cand.lower().replace(" ", "")
        if key in norm:
            target = norm[key]; break
    if target is None:
        for cand in candidates:
            key = cand.lower().replace(" ", "")
            for nk, orig in norm.items():
                if key in nk:
                    target = orig; break
            if target is not None:
                break
    if target is None:
        return [], []
    row = _as_series(df.loc[target])   # 重複科目名 → 收斂成 Series，避免軸錯亂
    vals = [_num(v) for v in row.iloc[:n]]
    # yfinance 欄位由新到舊 → 反轉成舊到新
    periods = [str(c)[:10] for c in row.index[:n]][::-1]
    return periods, vals[::-1]


def _score_bands(value, bands: list[tuple[float, float]], reverse: bool = False):
    """
    分級給分。bands = [(門檻, 分數), ...] 由高門檻到低門檻。
    reverse=True 表示「越小越好」（如負債比、P/E）。
    value=None 回 None（代表此子項無資料）。
    """
    v = _num(value)
    if v is None:
        return None
    if not reverse:
        for threshold, pts in bands:          # 高→低
            if v >= threshold:
                return pts
        return 0.0
    else:
        for threshold, pts in bands:          # 低→高（越小分越高）
            if v <= threshold:
                return pts
        return 0.0


def health_score(m: dict) -> dict:
    """
    財務健康評分 0~100。5 構面各 20 分，分級給分；
    缺資料的子項/構面不計零分，最後就「有資料的構面」重新正規化。
    m 需含（缺則 None）：roe, net_margin, revenue_growth, earnings_growth,
                        debt_to_equity, current_ratio, fcf, fcf_margin, pe, peg
    """
    dims: dict[str, float] = {}

    def _dim(subs: list):
        """subs = [(分數 or None, 滿分)]；回 (得分, 滿分) 只計有資料子項。"""
        got, full = 0.0, 0.0
        for pts, cap in subs:
            if pts is not None:
                got += pts
                full += cap
        return (got, full) if full > 0 else (None, None)

    # 1. 獲利能力（ROE 12 + 淨利率 8）
    roe = _score_bands(m.get("roe"), [(0.20, 12), (0.15, 9.6), (0.10, 7.2), (0.05, 4.8), (0.0001, 2.4)])
    nm  = _score_bands(m.get("net_margin"), [(0.20, 8), (0.10, 6), (0.05, 4), (0.0001, 2)])
    dims["獲利能力"] = _dim([(roe, 12), (nm, 8)])

    # 2. 成長性（營收成長 10 + EPS 成長 10）
    rg = _score_bands(m.get("revenue_growth"), [(0.20, 10), (0.10, 7.5), (0.05, 5), (0.0001, 2.5)])
    eg = _score_bands(m.get("earnings_growth"), [(0.20, 10), (0.10, 7.5), (0.05, 5), (0.0001, 2.5)])
    dims["成長性"] = _dim([(rg, 10), (eg, 10)])

    # 3. 財務安全（負債權益比 10 越低越好 + 流動比 10）
    de = _score_bands(m.get("debt_to_equity"), [(0.5, 10), (1.0, 7.5), (1.5, 5), (2.5, 2.5)], reverse=True)
    cr = _score_bands(m.get("current_ratio"), [(2.0, 10), (1.5, 8), (1.0, 5), (0.5, 2)])
    dims["財務安全"] = _dim([(de, 10), (cr, 10)])

    # 4. 現金流（FCF 正 12 + FCF 利潤率 8）
    fcf_val = _num(m.get("fcf"))
    fcf_pts = (12.0 if fcf_val > 0 else 0.0) if fcf_val is not None else None
    fcfm = _score_bands(m.get("fcf_margin"), [(0.15, 8), (0.08, 6), (0.03, 4), (0.0001, 2)])
    dims["現金流"] = _dim([(fcf_pts, 12), (fcfm, 8)])

    # 5. 估值（P/E 合理 10 越低越好但非越低越好；PEG 10）
    pe = _score_bands(m.get("pe"), [(10, 10), (15, 8), (25, 6), (40, 3)], reverse=True)
    peg = _score_bands(m.get("peg"), [(1.0, 10), (1.5, 7), (2.0, 4), (3.0, 2)], reverse=True)
    dims["估值"] = _dim([(pe, 10), (peg, 10)])

    # 重正規化：只就有資料的構面加總
    got_total, full_total = 0.0, 0.0
    covered = 0
    breakdown = {}
    for name, (got, full) in dims.items():
        if got is None:
            breakdown[name] = None
            continue
        breakdown[name] = round(got / full * 100, 1) if full > 0 else None  # 該構面 0~100
        got_total += got
        full_total += full
        covered += 1   # 直接計數有資料的構面（避免部分構面滿分非 20 導致誤算）

    score = round(got_total / full_total * 100, 1) if full_total > 0 else None

    if score is None:
        rating = "資料不足"
    elif score >= 75:
        rating = "優"
    elif score >= 55:
        rating = "良"
    elif score >= 40:
        rating = "中"
    else:
        rating = "弱"

    return {
        "score": score,
        "rating": rating,
        "breakdown": breakdown,           # 各構面 0~100（None=缺資料）
        "covered": covered,               # 有資料的構面數（0~5）
    }


def valuation_flags(m: dict) -> dict:
    """估值旗標：用絕對合理區間，回每項 cheap/fair/expensive/na。"""
    def _flag(v, cheap, expensive):
        v = _num(v)
        if v is None or v <= 0:
            return "na"
        if v < cheap:
            return "cheap"
        if v > expensive:
            return "expensive"
        return "fair"

    return {
        "pe":  _flag(m.get("pe"), 15, 30),
        "pb":  _flag(m.get("pb"), 1.5, 4),
        "peg": _flag(m.get("peg"), 1.0, 2.0),
        "ev_ebitda": _flag(m.get("ev_ebitda"), 10, 18),
    }


# ── 抓取層（呼叫 yfinance；此環境代理擋網路，需 merge 後實測）────────────────────

def fetch_fundamentals(ticker: str) -> dict:
    """
    抓 yfinance 基本面並整理。回傳統一 dict；缺欄位皆 None，不拋例外。
    含 quote_type 以辨識 ETF/指數（無基本面）。
    """
    import yfinance as yf

    out = {
        "ticker": ticker, "ok": False, "quote_type": None, "name": ticker,
        "currency": None, "sector": None, "industry": None,
        "price": None, "market_cap": None, "high_52w": None, "low_52w": None,
        # 估值
        "pe": None, "pb": None, "peg": None, "ev_ebitda": None,
        # 獲利
        "roe": None, "roa": None, "net_margin": None, "gross_margin": None, "op_margin": None,
        # 成長
        "revenue_growth": None, "earnings_growth": None,
        # 財務安全
        "debt_to_equity": None, "current_ratio": None,
        # 現金流
        "fcf": None, "fcf_margin": None,
        # 股利
        "dividend_yield": None,
        # 分析師
        "target_mean": None, "recommendation": None,
        # 趨勢（最近數期）
        "revenue_series": ([], []), "netincome_series": ([], []), "fcf_series": ([], []),
        "roe_note": None,
        "error": None,
    }

    try:
        tk = yf.Ticker(ticker)
        try:
            info = tk.info or {}
        except Exception:
            info = {}

        out["quote_type"] = info.get("quoteType")
        out["name"] = info.get("longName") or info.get("shortName") or ticker
        out["currency"] = info.get("currency")
        out["sector"] = info.get("sector")
        out["industry"] = info.get("industry")
        out["price"] = _num(info.get("currentPrice") or info.get("regularMarketPrice"))
        out["market_cap"] = _num(info.get("marketCap"))
        out["high_52w"] = _num(info.get("fiftyTwoWeekHigh"))
        out["low_52w"] = _num(info.get("fiftyTwoWeekLow"))

        # ETF / 指數 / 基金：無基本面，提早回
        if out["quote_type"] in ("ETF", "INDEX", "MUTUALFUND", "CURRENCY", "CRYPTOCURRENCY"):
            out["ok"] = False
            out["error"] = f"{out['quote_type']} 無公司基本面資料"
            return out

        # 估值（.info）
        out["pe"]  = _num(info.get("trailingPE"))
        out["pb"]  = _num(info.get("priceToBook"))
        out["peg"] = _num(info.get("trailingPegRatio") or info.get("pegRatio"))
        out["ev_ebitda"] = _num(info.get("enterpriseToEbitda"))

        # 獲利（.info，比例為小數，如 0.25）
        out["roe"] = _num(info.get("returnOnEquity"))
        out["roa"] = _num(info.get("returnOnAssets"))
        out["net_margin"] = _num(info.get("profitMargins"))
        out["gross_margin"] = _num(info.get("grossMargins"))
        out["op_margin"] = _num(info.get("operatingMargins"))

        # 成長
        out["revenue_growth"] = _num(info.get("revenueGrowth"))
        out["earnings_growth"] = _num(info.get("earningsGrowth") or info.get("earningsQuarterlyGrowth"))

        # 財務安全（yfinance debtToEquity 是百分比，如 120 = 1.2）
        de_raw = _num(info.get("debtToEquity"))
        out["debt_to_equity"] = de_raw / 100 if de_raw is not None else None
        out["current_ratio"] = _num(info.get("currentRatio"))

        # 現金流（.info）
        out["fcf"] = _num(info.get("freeCashflow"))
        rev = _num(info.get("totalRevenue"))
        if out["fcf"] is not None and rev:
            out["fcf_margin"] = out["fcf"] / rev

        # 股利率：優先用 dividendRate/price（無歧義的 %），
        # 退回 dividendYield 並正規化（yfinance 版本有時回小數 0.005、有時回百分比 0.5）
        div_rate = _num(info.get("dividendRate"))
        if div_rate is not None and out["price"]:
            out["dividend_yield"] = div_rate / out["price"] * 100
        else:
            dy = _num(info.get("dividendYield"))
            out["dividend_yield"] = (dy * 100 if (dy is not None and dy < 1) else dy)
        out["target_mean"] = _num(info.get("targetMeanPrice"))
        out["recommendation"] = info.get("recommendationKey")

        # 三表趨勢（容錯：列名模糊比對）
        try:
            fin = tk.financials
            out["revenue_series"] = series_row(fin, ["Total Revenue", "TotalRevenue", "Revenue"])
            out["netincome_series"] = series_row(fin, ["Net Income", "NetIncome",
                                                       "Net Income Common Stockholders"])
        except Exception:
            pass
        try:
            cf = tk.cashflow
            fcf_p, fcf_v = series_row(cf, ["Free Cash Flow", "FreeCashFlow"])
            if not fcf_v:
                # 自行用營運現金流 - 資本支出 估
                op_p, op_v = series_row(cf, ["Operating Cash Flow", "Total Cash From Operating Activities"])
                cx_p, cx_v = series_row(cf, ["Capital Expenditure", "Capital Expenditures"])
                if op_v and cx_v and len(op_v) == len(cx_v):
                    fcf_p = op_p
                    fcf_v = [(o + c) if (o is not None and c is not None) else None
                             for o, c in zip(op_v, cx_v)]   # capex 為負，相加=扣除
            out["fcf_series"] = (fcf_p, fcf_v)
        except Exception:
            pass

        # 若 .info 缺 ROE，嘗試用淨利/股東權益估（資產負債表）
        if out["roe"] is None:
            try:
                bs = tk.balance_sheet
                _, equity = pick_row(bs, ["Stockholders Equity", "Total Stockholder Equity",
                                          "Common Stock Equity"])
                ni_p, ni_v = out["netincome_series"]
                ni = ni_v[-1] if ni_v else None
                if equity and ni is not None and equity != 0:
                    out["roe"] = ni / equity
                    out["roe_note"] = "由淨利/股東權益估算"
            except Exception:
                pass

        out["ok"] = True
        return out

    except Exception as e:
        out["error"] = str(e)
        return out


# ── CLI 自我測試（純邏輯，不需網路）────────────────────────────────────────────

if __name__ == "__main__":
    print("=== pick_row 模糊比對 ===")
    df = pd.DataFrame({"2024": [1000, 200], "2023": [900, 150]},
                      index=["Total Revenue", "Net Income"])
    print(" exact:", pick_row(df, ["Total Revenue"]))
    print(" fuzzy:", pick_row(df, ["revenue"]))
    print(" miss :", pick_row(df, ["EBITDA"]))

    print("\n=== health_score 完整資料（優質公司）===")
    good = dict(roe=0.25, net_margin=0.22, revenue_growth=0.18, earnings_growth=0.25,
                debt_to_equity=0.4, current_ratio=2.2, fcf=5e9, fcf_margin=0.18, pe=18, peg=1.1)
    print(" ", health_score(good))

    print("\n=== health_score 缺一半資料（重正規化）===")
    partial = dict(roe=0.25, net_margin=0.22, revenue_growth=None, earnings_growth=None,
                   debt_to_equity=None, current_ratio=None, fcf=5e9, fcf_margin=0.18,
                   pe=18, peg=1.1)
    hs = health_score(partial)
    print(" ", hs, "← covered 應 < 5")

    print("\n=== health_score 弱公司 ===")
    weak = dict(roe=-0.05, net_margin=-0.1, revenue_growth=-0.2, earnings_growth=-0.3,
                debt_to_equity=3.0, current_ratio=0.6, fcf=-1e9, fcf_margin=-0.05, pe=80, peg=4)
    print(" ", health_score(weak))

    print("\n=== valuation_flags ===")
    print(" ", valuation_flags(dict(pe=12, pb=1.2, peg=0.8, ev_ebitda=8)))
    print(" ", valuation_flags(dict(pe=45, pb=6, peg=3, ev_ebitda=25)))
