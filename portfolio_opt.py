"""
portfolio_opt.py – 均值-變異數效率前緣（純邏輯 + scipy，離線可測）

現代投資組合理論（Markowitz）：給定各資產日報酬矩陣，求
最小波動組合、最大 Sharpe 組合、效率前緣曲線。學習自 PyPortfolioOpt 的介面精神。
限制：權重和=1、預設不放空（bounds 0~1）。scipy 已在 requirements 中。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _annualize(returns_df: pd.DataFrame, ppy: int = 252):
    mu = returns_df.mean().to_numpy() * ppy
    cov = returns_df.cov().to_numpy() * ppy
    return mu, cov


def port_perf(w: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> tuple[float, float]:
    """(年化報酬, 年化波動)。"""
    ret = float(w @ mu)
    vol = float(np.sqrt(max(w @ cov @ w, 0.0)))
    return ret, vol


def _solve(objective, n: int, bounds, constraints):
    from scipy.optimize import minimize
    w0 = np.repeat(1.0 / n, n)
    res = minimize(objective, w0, method="SLSQP", bounds=bounds,
                   constraints=constraints, options={"maxiter": 300})
    if not res.success:
        return None
    w = np.clip(res.x, 0, None)
    s = w.sum()
    return w / s if s > 0 else None


def min_vol_weights(returns_df: pd.DataFrame, ppy: int = 252) -> pd.Series | None:
    """最小波動組合權重（不放空）。解不出來回 None。"""
    mu, cov = _annualize(returns_df, ppy)
    n = len(mu)
    w = _solve(lambda w: w @ cov @ w, n,
               [(0.0, 1.0)] * n, [{"type": "eq", "fun": lambda w: w.sum() - 1}])
    return pd.Series(w, index=returns_df.columns) if w is not None else None


def max_sharpe_weights(returns_df: pd.DataFrame, rf: float = 0.0,
                       ppy: int = 252) -> pd.Series | None:
    """最大 Sharpe 組合權重（不放空）。"""
    mu, cov = _annualize(returns_df, ppy)
    n = len(mu)

    def neg_sharpe(w):
        ret, vol = port_perf(w, mu, cov)
        return -(ret - rf) / vol if vol > 1e-12 else 1e9

    w = _solve(neg_sharpe, n,
               [(0.0, 1.0)] * n, [{"type": "eq", "fun": lambda w: w.sum() - 1}])
    return pd.Series(w, index=returns_df.columns) if w is not None else None


def efficient_frontier(returns_df: pd.DataFrame, n_points: int = 25,
                       ppy: int = 252) -> pd.DataFrame:
    """
    效率前緣：在最小波動組合報酬 ~ 最高單一資產報酬之間取 n_points 個目標報酬，
    各解最小波動。回 DataFrame(ret, vol, weights)；解不出的點略過。
    """
    mu, cov = _annualize(returns_df, ppy)
    n = len(mu)
    wmv = min_vol_weights(returns_df, ppy)
    if wmv is None:
        return pd.DataFrame(columns=["ret", "vol", "weights"])
    ret_lo, _ = port_perf(wmv.to_numpy(), mu, cov)
    ret_hi = float(mu.max())
    if ret_hi <= ret_lo:                       # 全部資產期望報酬相近 → 前緣退化成一點
        ret_hi = ret_lo + abs(ret_lo) * 0.01 + 1e-6

    rows = []
    for tgt in np.linspace(ret_lo, ret_hi, n_points):
        w = _solve(lambda w: w @ cov @ w, n, [(0.0, 1.0)] * n,
                   [{"type": "eq", "fun": lambda w: w.sum() - 1},
                    {"type": "eq", "fun": lambda w, t=tgt: w @ mu - t}])
        if w is None:
            continue
        ret, vol = port_perf(w, mu, cov)
        rows.append({"ret": ret, "vol": vol, "weights": w})
    return pd.DataFrame(rows)


# ── CLI 自我測試 ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(11)
    n_days = 756
    # 三資產：低波低報酬 / 中波中報酬 / 高波高報酬，低相關
    base = rng.normal(size=(n_days, 3))
    rets = pd.DataFrame({
        "BOND":  0.0001 + 0.003 * base[:, 0],
        "EQ":    0.0004 + 0.010 * (0.2 * base[:, 0] + 0.98 * base[:, 1]),
        "GROW":  0.0008 + 0.020 * (0.1 * base[:, 0] + 0.99 * base[:, 2]),
    }, index=pd.date_range("2023-01-02", periods=n_days, freq="B"))

    mu, cov = _annualize(rets)
    w_eq = np.repeat(1 / 3, 3)
    ret_eq, vol_eq = port_perf(w_eq, mu, cov)

    wmv = min_vol_weights(rets)
    assert wmv is not None and abs(wmv.sum() - 1) < 1e-6 and (wmv >= -1e-9).all()
    ret_mv, vol_mv = port_perf(wmv.to_numpy(), mu, cov)
    print(f"  等權   vol={vol_eq:.4f}  minvol vol={vol_mv:.4f}")
    assert vol_mv <= vol_eq + 1e-9            # 最小波動 ≤ 等權
    assert wmv["BOND"] > 0.5                  # 低波資產應占大頭

    wms = max_sharpe_weights(rets)
    assert wms is not None and abs(wms.sum() - 1) < 1e-6
    ret_ms, vol_ms = port_perf(wms.to_numpy(), mu, cov)
    sh_ms = ret_ms / vol_ms
    sh_eq = ret_eq / vol_eq
    print(f"  等權 Sharpe={sh_eq:.3f}  maxSharpe={sh_ms:.3f}")
    assert sh_ms >= sh_eq - 1e-9              # 最大 Sharpe ≥ 等權

    ef = efficient_frontier(rets, n_points=15)
    assert len(ef) >= 10
    assert ef["ret"].is_monotonic_increasing
    # 前緣上任一點的波動不小於最小波動組合
    assert (ef["vol"] >= vol_mv - 1e-9).all()
    print(f"  前緣 {len(ef)} 點：ret {ef['ret'].iloc[0]:.3f}→{ef['ret'].iloc[-1]:.3f}")

    print("\n✅ portfolio_opt 純邏輯測試通過")
