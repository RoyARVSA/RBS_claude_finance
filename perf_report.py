"""
perf_report.py – QuantStats 風格績效報告（純邏輯，離線可測）

輸入日/週/月報酬序列（pd.Series，DatetimeIndex），輸出專業績效指標：
CAGR、Sortino、Calmar、月報酬表、前 N 大回撤期間、滾動 Sharpe。
無 Streamlit / 網路依賴；學習自 quantstats 的指標慣例。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── 基礎指標 ──────────────────────────────────────────────────────────────────

def cagr(returns: pd.Series, ppy: int = 252):
    """年化複合報酬。樣本不足或淨值歸零回 None。"""
    r = returns.dropna()
    if len(r) < 2:
        return None
    growth = float((1 + r).prod())
    if growth <= 0:
        return None
    return growth ** (ppy / len(r)) - 1


def max_drawdown(returns: pd.Series) -> float:
    curve = (1 + returns.dropna()).cumprod()
    if curve.empty:
        return 0.0
    return float((curve / curve.cummax() - 1).min())


def sharpe(returns: pd.Series, rf_annual: float = 0.0, ppy: int = 252):
    r = returns.dropna() - rf_annual / ppy
    sd = r.std(ddof=1)
    if len(r) < 2 or sd == 0 or not np.isfinite(sd):
        return None
    return float(r.mean() / sd * np.sqrt(ppy))


def sortino(returns: pd.Series, rf_annual: float = 0.0, ppy: int = 252):
    """Sortino：分母只計下檔波動（quantstats 慣例：以全樣本數為分母）。"""
    r = returns.dropna() - rf_annual / ppy
    if len(r) < 2:
        return None
    downside = r[r < 0]
    dd = np.sqrt(float((downside ** 2).sum()) / len(r))
    if dd == 0 or not np.isfinite(dd):
        return None
    return float(r.mean() / dd * np.sqrt(ppy))


def calmar(returns: pd.Series, ppy: int = 252):
    c = cagr(returns, ppy)
    mdd = max_drawdown(returns)
    if c is None or mdd == 0:
        return None
    return c / abs(mdd)


def rolling_sharpe(returns: pd.Series, window: int = 63, ppy: int = 252) -> pd.Series:
    r = returns.dropna()
    mu = r.rolling(window).mean()
    sd = r.rolling(window).std(ddof=1)
    return (mu / sd * np.sqrt(ppy)).dropna()


# ── 彙總 ──────────────────────────────────────────────────────────────────────

def perf_stats(returns: pd.Series, benchmark: pd.Series | None = None,
               rf_annual: float = 0.0, ppy: int = 252) -> dict:
    """一次算完 tearsheet 主要指標。樣本不足的欄位為 None，不拋例外。"""
    r = returns.dropna()
    out = {
        "n_periods":  int(len(r)),
        "total_ret":  float((1 + r).prod() - 1) if len(r) else None,
        "cagr":       cagr(r, ppy),
        "ann_vol":    float(r.std(ddof=1) * np.sqrt(ppy)) if len(r) > 1 else None,
        "sharpe":     sharpe(r, rf_annual, ppy),
        "sortino":    sortino(r, rf_annual, ppy),
        "calmar":     calmar(r, ppy),
        "max_dd":     max_drawdown(r) if len(r) else None,
        "win_rate":   float((r > 0).mean()) if len(r) else None,
        "best":       float(r.max()) if len(r) else None,
        "worst":      float(r.min()) if len(r) else None,
        "skew":       float(r.skew()) if len(r) > 2 else None,
        "var95":      float(np.percentile(r, 5)) if len(r) > 1 else None,
        "alpha_ann":  None, "beta": None, "corr_bench": None,
    }
    if benchmark is not None:
        b = benchmark.dropna()
        df = pd.concat([r, b], axis=1).dropna()
        if len(df) > 2:
            rp, rb = df.iloc[:, 0], df.iloc[:, 1]
            var_b = float(rb.var(ddof=1))
            if var_b > 0:
                beta = float(np.cov(rp, rb, ddof=1)[0, 1]) / var_b
                out["beta"] = beta
                out["alpha_ann"] = float((rp.mean() - beta * rb.mean()) * ppy)
            out["corr_bench"] = float(rp.corr(rb))
    return out


def monthly_table(returns: pd.Series) -> pd.DataFrame:
    """月報酬表：列=年、欄=1..12 月 + YTD（小數）。空序列回空表。"""
    r = returns.dropna()
    if r.empty or not isinstance(r.index, pd.DatetimeIndex):
        return pd.DataFrame()
    m = r.resample("ME").apply(lambda x: float((1 + x).prod() - 1))
    tbl = pd.DataFrame({
        "year": m.index.year, "month": m.index.month, "ret": m.values
    }).pivot_table(index="year", columns="month", values="ret", aggfunc="first")
    tbl = tbl.reindex(columns=range(1, 13))
    ytd = r.groupby(r.index.year).apply(lambda x: float((1 + x).prod() - 1))
    tbl["YTD"] = ytd
    return tbl


def drawdown_periods(returns: pd.Series, top_n: int = 5) -> list[dict]:
    """前 N 大回撤期間：{start, trough, end(None=未回復), depth, days}。"""
    r = returns.dropna()
    if r.empty:
        return []
    curve = (1 + r).cumprod()
    dd = curve / curve.cummax() - 1
    periods, start = [], None
    for i, (dt, v) in enumerate(dd.items()):
        if v < 0 and start is None:
            start = dd.index[i - 1] if i > 0 else dt   # 高點日
        elif v == 0 and start is not None:
            seg = dd.loc[start:dt]
            periods.append({"start": start, "trough": seg.idxmin(),
                            "end": dt, "depth": float(seg.min()),
                            "days": int((dt - start).days)})
            start = None
    if start is not None:                               # 尚未回復的回撤
        seg = dd.loc[start:]
        periods.append({"start": start, "trough": seg.idxmin(),
                        "end": None, "depth": float(seg.min()),
                        "days": int((dd.index[-1] - start).days)})
    periods.sort(key=lambda p: p["depth"])
    return periods[:top_n]


# ── CLI 自我測試 ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    idx = pd.date_range("2024-01-01", periods=504, freq="B")
    rng = np.random.default_rng(7)
    r = pd.Series(rng.normal(0.0006, 0.012, 504), index=idx)
    r.iloc[100:110] = -0.03          # 製造一段明確回撤

    s = perf_stats(r, benchmark=r * 0.5 + rng.normal(0, 0.004, 504))
    for k, v in s.items():
        print(f"  {k:12} {v if not isinstance(v, float) else round(v, 4)}")
    assert s["max_dd"] < -0.2 and s["n_periods"] == 504
    assert s["beta"] is not None and 1.2 < s["beta"] < 2.8   # rp ≈ 2×rb → beta≈2
    assert s["sortino"] is not None and s["sharpe"] is not None

    # 確定性驗證：+10%、-50%、+10%
    det = pd.Series([0.10, -0.50, 0.10],
                    index=pd.date_range("2024-01-31", periods=3, freq="ME"))
    st_ = perf_stats(det, ppy=12)
    assert abs(st_["total_ret"] - (1.1 * 0.5 * 1.1 - 1)) < 1e-9
    assert abs(st_["max_dd"] - (-0.5)) < 1e-9

    mt = monthly_table(r)
    assert "YTD" in mt.columns and len(mt) == 2            # 2024、2025 兩年
    # 月表複合應等於總報酬
    total_from_ytd = float((1 + mt["YTD"]).prod() - 1)
    assert abs(total_from_ytd - s["total_ret"]) < 1e-9

    dps = drawdown_periods(r, top_n=3)
    assert dps and dps[0]["depth"] == s["max_dd"]
    assert dps[0]["days"] > 0
    print(f"\n  top drawdown: {dps[0]['depth']:.1%} ({dps[0]['days']} days)")

    # 空輸入安全
    empty = pd.Series(dtype=float)
    assert perf_stats(empty)["cagr"] is None and drawdown_periods(empty) == []
    assert monthly_table(empty).empty

    print("\n✅ perf_report 純邏輯測試通過")
