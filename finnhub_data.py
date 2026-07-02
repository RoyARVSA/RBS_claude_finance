"""
finnhub_data.py – Finnhub 基本面備援（免費 API，可重用，無 Streamlit 依賴）

yfinance 的 .info 在雲端常被 Yahoo 限流，導致市值/P/E/ROE 抓不到。
本模組用 Finnhub（不同來源）當備援。需 FINNHUB_API_KEY。

解析層 _normalize 為純函數（可離線測試），抓取層 fetch 走 requests。
輸出採本專案慣例：比率用小數（roe 0.15 = 15%），股利率用百分比。
"""

from __future__ import annotations

BASE = "https://finnhub.io/api/v1"


def _f(x):
    try:
        return float(x) if x is not None else None
    except (TypeError, ValueError):
        return None


def _pct_to_frac(x):
    """Finnhub 的 roe/margin 以百分比給（156.08=156%）→ 轉小數 1.5608。"""
    v = _f(x)
    return v / 100 if v is not None else None


def _normalize(profile: dict, quote: dict, metric: dict) -> dict:
    """把 Finnhub 三個端點的原始 JSON 正規化成本專案欄位（純函數）。"""
    profile = profile or {}
    quote = quote or {}
    m = (metric or {}).get("metric", {}) if metric else {}

    mc = _f(profile.get("marketCapitalization"))      # 單位：百萬
    return {
        "price":          _f(quote.get("c")),
        "market_cap":     mc * 1e6 if mc else None,
        "pe":             _f(m.get("peTTM") or m.get("peBasicExclExtraTTM")),
        "eps":            _f(m.get("epsTTM") or m.get("epsBasicExclExtraItemsTTM")),
        "pb":             _f(m.get("pbAnnual") or m.get("pbQuarterly")),
        "roe":            _pct_to_frac(m.get("roeTTM")),
        "roa":            _pct_to_frac(m.get("roaTTM")),
        "net_margin":     _pct_to_frac(m.get("netProfitMarginTTM")),
        "gross_margin":   _pct_to_frac(m.get("grossMarginTTM")),
        "op_margin":      _pct_to_frac(m.get("operatingMarginTTM")),
        "revenue_growth": _pct_to_frac(m.get("revenueGrowthTTMYoy")),
        "high_52w":       _f(m.get("52WeekHigh")),
        "low_52w":        _f(m.get("52WeekLow")),
        "beta":           _f(m.get("beta")),
        "dividend_yield_pct": _f(m.get("currentDividendYieldTTM") or m.get("dividendYieldIndicatedAnnual")),
        "name":           profile.get("name"),
        "sector":         profile.get("finnhubIndustry"),
        "currency":       profile.get("currency"),
    }


# ── 抓取層（需網路 + key）──────────────────────────────────────────────────────

def fetch(ticker: str, key: str) -> dict:
    """抓 Finnhub 基本面並正規化。無 key 或失敗回 {}。"""
    if not key or not ticker:
        return {}
    import requests

    def _get(path, params):
        try:
            r = requests.get(f"{BASE}/{path}", params={**params, "token": key}, timeout=15)
            return r.json() if r.ok else {}
        except Exception:
            return {}

    profile = _get("stock/profile2", {"symbol": ticker})
    quote   = _get("quote", {"symbol": ticker})
    metric  = _get("stock/metric", {"symbol": ticker, "metric": "all"})
    if not (profile or metric):
        return {}
    return _normalize(profile, quote, metric)


# ── CLI 自我測試（純解析）──────────────────────────────────────────────────────

if __name__ == "__main__":
    profile = {"marketCapitalization": 2900000, "name": "Apple Inc",
               "finnhubIndustry": "Technology", "currency": "USD"}
    quote = {"c": 190.5}
    metric = {"metric": {"peTTM": 29.3, "epsTTM": 6.5, "roeTTM": 156.08,
                         "netProfitMarginTTM": 25.3, "revenueGrowthTTMYoy": 8.1,
                         "52WeekHigh": 237.2, "52WeekLow": 164.1, "beta": 1.25,
                         "currentDividendYieldTTM": 0.44}}
    out = _normalize(profile, quote, metric)
    for k, v in out.items():
        print(f"  {k:18} {v}")
    assert out["market_cap"] == 2.9e12
    assert abs(out["roe"] - 1.5608) < 1e-9        # 156.08% → 1.5608
    assert out["pe"] == 29.3
    assert _normalize({}, {}, {})["pe"] is None   # 空輸入安全
    print("\n✅ _normalize 正確（單位換算：市值×1e6、roe/margin ÷100）")
