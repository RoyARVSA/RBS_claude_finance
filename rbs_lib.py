"""
rbs_lib.py – Core financial analytics library for RBS Finance App
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Union, Dict, Optional, Tuple
from scipy.stats import norm
from dataclasses import dataclass

ArrayLike = Union[np.ndarray, pd.Series, list]

# ─────────────────────────── Data ────────────────────────────

def load_price_data(tickers, start: str = "2020-01-01") -> pd.DataFrame:
    import yfinance as yf
    data = yf.download(tickers, start=start, auto_adjust=False, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        col = "Adj Close" if "Adj Close" in data.columns.get_level_values(0) else "Close"
        out = data[col].dropna(how="all")
    else:
        out = data.dropna(how="all")
        take = "Adj Close" if "Adj Close" in out.columns else "Close"
        out = out[[take]]
        name = tickers if isinstance(tickers, str) else (tickers[0] if tickers else "PX")
        out.columns = [name]
    if isinstance(out, pd.Series):
        out = out.to_frame()
    return out.ffill()


def to_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    return price_df.pct_change().dropna(how="all")


# ─────────────────── Single-asset risk ───────────────────────

def historical_var(returns: ArrayLike, confidence_level: float = 0.95) -> float:
    arr = np.asarray(returns, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.percentile(arr, (1 - confidence_level) * 100))


def conditional_var(returns: ArrayLike, confidence_level: float = 0.95) -> float:
    arr = np.asarray(returns, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return np.nan
    q = historical_var(arr, confidence_level)
    tail = arr[arr <= q]
    return float(np.mean(tail)) if tail.size else np.nan


def delta_normal_var(returns: ArrayLike, confidence_level: float = 0.95) -> float:
    arr = np.asarray(returns, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return np.nan
    mu, sigma = float(np.mean(arr)), float(np.std(arr, ddof=1))
    z = norm.ppf(1 - confidence_level)
    return float(mu + z * sigma)


# ─────────────────── Volatility / Covariance ─────────────────

def calculate_volatility(
    returns: pd.Series, window: int = 20, annualized: bool = True
) -> pd.Series:
    s = pd.Series(returns, dtype=float).dropna()
    vol = s.rolling(window, min_periods=window).std()
    return vol * np.sqrt(252.0) if annualized else vol


def ewma_cov(returns: pd.DataFrame, lam: float = 0.94) -> pd.DataFrame:
    """EWMA covariance (RiskMetrics)."""
    r = returns.dropna(how="any")
    if r.empty:
        return pd.DataFrame(np.nan, index=returns.columns, columns=returns.columns)
    mu = r.mean()
    x = r - mu
    cov = (1 - lam) * (x.iloc[0:1].T @ x.iloc[0:1])
    for i in range(1, len(x)):
        cov = lam * cov + (1 - lam) * (x.iloc[i : i + 1].T @ x.iloc[i : i + 1])
    cov.index = returns.columns
    cov.columns = returns.columns
    return cov


def lw_cov(returns: pd.DataFrame) -> pd.DataFrame:
    """Ledoit–Wolf shrinkage covariance."""
    try:
        from sklearn.covariance import LedoitWolf
        r = returns.dropna(how="any").values
        if r.shape[0] < 2:
            raise ValueError
        lw = LedoitWolf().fit(r)
        return pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)
    except Exception:
        return returns.cov()


# ─────────────────── Portfolio Risk ──────────────────────────

@dataclass
class PortRiskOut:
    var_pct: float
    cvar_pct: float
    vol_ann: float
    value_var: float
    value_cvar: float
    cov: pd.DataFrame


def portfolio_var(
    price_df: pd.DataFrame,
    weights: pd.Series,
    alpha: float = 0.95,
    hold_days: int = 1,
    window: Optional[int] = None,
    cov_method: str = "hist",   # hist | ewma | lw
    lam: float = 0.94,
    as_of_value: Optional[float] = None,
    use_historical_cvar: bool = True,
) -> PortRiskOut:
    px = price_df[weights.index].dropna(how="all")
    ret = px.pct_change().dropna()
    if window and window < len(ret):
        ret = ret.tail(window)

    agg = (
        (1 + ret).rolling(hold_days).apply(np.prod, raw=True) - 1
        if hold_days > 1
        else ret
    )
    agg = agg.dropna()

    if cov_method == "ewma":
        cov = ewma_cov(ret, lam)
    elif cov_method == "lw":
        cov = lw_cov(ret)
    else:
        cov = ret.cov()

    w = weights.values.reshape(-1, 1)
    mu = agg.mean().values.reshape(-1, 1)
    cov_h = cov * hold_days
    port_mu = float((w.T @ mu).ravel())
    port_sigma = float(np.sqrt(w.T @ cov_h.values @ w))
    z = norm.ppf(1 - alpha)
    var_dn = port_mu + z * port_sigma
    vol_ann = float(np.sqrt(w.T @ (ret.cov().values * 252) @ w))

    port_ret_hist = (agg @ weights).dropna()
    if use_historical_cvar and not port_ret_hist.empty:
        var_hist = np.percentile(port_ret_hist, (1 - alpha) * 100)
        cvar = float(port_ret_hist[port_ret_hist <= var_hist].mean())
        var_pct = float(var_hist)
    else:
        cvar = float(
            port_mu + (norm.pdf(norm.ppf(alpha)) / (1 - alpha)) * port_sigma * -1
        )
        var_pct = float(var_dn)

    value = (
        as_of_value
        if as_of_value is not None
        else float((px.iloc[-1] * weights).sum())
    )
    return PortRiskOut(
        var_pct=var_pct,
        cvar_pct=cvar,
        vol_ann=vol_ann,
        value_var=value * (-var_pct),
        value_cvar=value * (-cvar),
        cov=cov,
    )


# ─────────────────── Monte Carlo ─────────────────────────────

def mc_portfolio_pnl(
    price_df: pd.DataFrame,
    weights: pd.Series,
    days: int = 1,
    n: int = 10000,
    cov_method: str = "hist",
    lam: float = 0.94,
    window: Optional[int] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    px = price_df[weights.index]
    ret = px.pct_change().dropna()
    if window and window < len(ret):
        ret = ret.tail(window)
    mu = ret.mean().values
    if cov_method == "ewma":
        cov = ewma_cov(ret, lam).values
    elif cov_method == "lw":
        cov = lw_cov(ret).values
    else:
        cov = ret.cov().values
    L = np.linalg.cholesky(cov + 1e-12 * np.eye(cov.shape[0]))
    z = rng.standard_normal((n, cov.shape[0]))
    draws = (z @ L.T) + mu
    d_ret = draws * np.sqrt(days)
    port = d_ret @ weights.values
    current_value = float((px.iloc[-1] * weights).sum())
    return current_value * (np.exp(port) - 1.0)


# ─────────────────── Backtesting ─────────────────────────────

@dataclass
class KupiecResult:
    exceptions: int
    expected: float
    ratio: float
    p_value: float


def kupiec_pof_test(
    returns: pd.Series, var_series: pd.Series, alpha: float
) -> KupiecResult:
    df = pd.concat([returns, var_series], axis=1).dropna()
    df.columns = ["r", "var"]
    I = (df["r"] < df["var"]).astype(int)
    x = I.sum()
    n = len(I)
    pi = 1 - alpha
    p_hat = x / n if n else np.nan
    if n == 0 or p_hat in (0, 1):
        p_value = np.nan
    else:
        LR = -2 * (
            (x * np.log(pi) + (n - x) * np.log(1 - pi))
            - (x * np.log(p_hat) + (n - x) * np.log(1 - p_hat))
        )
        from scipy.stats import chi2
        p_value = 1 - chi2.cdf(LR, df=1)
    return KupiecResult(
        int(x),
        float(n * pi),
        float(p_hat),
        float(p_value) if not np.isnan(p_value) else np.nan,
    )


def rolling_portfolio_var(
    price_df: pd.DataFrame,
    weights: pd.Series,
    alpha: float = 0.95,
    window: int = 250,
    cov_method: str = "hist",
    lam: float = 0.94,
) -> pd.Series:
    px = price_df[weights.index]
    ret = px.pct_change().dropna()
    vs, idx = [], []
    for i in range(window, len(ret)):
        sub = ret.iloc[i - window : i]
        if cov_method == "ewma":
            cov = ewma_cov(sub, lam).values
        elif cov_method == "lw":
            cov = lw_cov(sub).values
        else:
            cov = sub.cov().values
        w = weights.values.reshape(-1, 1)
        mu = float(sub.mean().values @ weights.values)
        sigma = float(np.sqrt(w.T @ cov @ w))
        z = norm.ppf(1 - alpha)
        vs.append(mu + z * sigma)
        idx.append(ret.index[i])
    return pd.Series(vs, index=idx, name="VaR")


# ─────────────────── Scenarios & Stress ──────────────────────

def apply_shocks(last_prices: pd.Series, shocks: Dict[str, float]) -> pd.Series:
    new = last_prices.copy()
    for t, s in shocks.items():
        if t in new.index:
            new.loc[t] = new.loc[t] * (1 + float(s))
    return new


def scenario_pnl(
    price_df: pd.DataFrame, weights: pd.Series, shocks: Dict[str, float]
) -> float:
    last = price_df[weights.index].iloc[-1]
    new = apply_shocks(last, shocks)
    return float((new * weights).sum()) - float((last * weights).sum())


def scenario_pnl_value(
    weights: pd.Series, shocks: Dict[str, float], notional: float
) -> float:
    if notional is None or notional <= 0:
        raise ValueError("notional must be positive")
    aligned = pd.Series(
        {t: float(shocks.get(t, 0.0)) for t in weights.index},
        index=weights.index,
        dtype=float,
    )
    return notional * float(np.dot(weights.values, aligned.values))


def historical_replay(
    price_df: pd.DataFrame,
    weights: pd.Series,
    start: str,
    end: str,
    notional: float = 100_000,
) -> Dict[str, float]:
    px = price_df[weights.index]
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    sub = px.loc[(px.index >= start_dt) & (px.index <= end_dt)]
    if sub.shape[0] < 2:
        return {"PnL": np.nan, "Return": np.nan, "MaxDD": np.nan, "Rows": float(sub.shape[0])}
    r = sub.pct_change().dropna()
    port_r = (r @ weights).astype(float)
    nav = (1 + port_r).cumprod()
    ret = float(nav.iloc[-1] / nav.iloc[0] - 1.0)
    pnl = float(notional * ret)
    maxdd = float((nav / nav.cummax() - 1).min())
    return {"PnL": pnl, "Return": ret, "MaxDD": maxdd, "Rows": float(sub.shape[0])}


# ─────────────────── Credit Risk ─────────────────────────────

def scorecard_transform(
    pd_array,
    base_score: float = 600,
    base_odds: float = 20,
    pdo: float = 50,
    clip: tuple = (1e-6, 1 - 1e-6),
) -> np.ndarray:
    p = np.clip(np.asarray(pd_array, dtype=float), clip[0], clip[1])
    odds = (1 - p) / p
    factor = pdo / np.log(2.0)
    offset = base_score + factor * np.log(base_odds)
    return offset - factor * np.log(odds)


def calculate_woe_iv(
    data: pd.DataFrame, feature: str, target: str, bins: int = 10
) -> Tuple[pd.DataFrame, float]:
    df = data[[feature, target]].copy()
    if pd.api.types.is_numeric_dtype(df[feature]):
        df["bin"] = pd.qcut(df[feature], q=bins, duplicates="drop")
    else:
        df["bin"] = df[feature].astype("category")
    total_good = (df[target] == 0).sum()
    total_bad = (df[target] == 1).sum()
    eps = 1e-9
    g = df.groupby("bin", observed=True)
    woe_df = pd.DataFrame(
        {
            "good": g.apply(lambda x: (x[target] == 0).sum()),
            "bad": g.apply(lambda x: (x[target] == 1).sum()),
        }
    )
    woe_df["dist_good"] = woe_df["good"] / max(total_good, 1)
    woe_df["dist_bad"] = woe_df["bad"] / max(total_bad, 1)
    woe_df["woe"] = np.log(
        (woe_df["dist_good"] + eps) / (woe_df["dist_bad"] + eps)
    )
    woe_df["iv"] = (woe_df["dist_good"] - woe_df["dist_bad"]) * woe_df["woe"]
    return woe_df, float(woe_df["iv"].sum()) if len(woe_df) else 0.0
