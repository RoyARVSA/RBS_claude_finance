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


def hrp_weights(returns_df: pd.DataFrame) -> pd.Series | None:
    """
    HRP 階層風險平價（López de Prado；Riskfolio 的招牌）。
    相關性距離 → 階層聚類 → 準對角化 → 遞迴二分反變異數配置。
    不需期望報酬（比均值-變異數穩健）、不需矩陣求逆。失敗回 None。
    """
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import squareform
        corr = returns_df.corr()
        cov = returns_df.cov()
        n = len(corr)
        if n < 2:
            return None
        dist = np.sqrt(np.clip(0.5 * (1 - corr.values), 0.0, 1.0))   # fp 噪聲防 NaN
        np.fill_diagonal(dist, 0.0)
        link = linkage(squareform(dist, checks=False), method="single")
        order = list(leaves_list(link))                     # 準對角化排序
        tickers = [returns_df.columns[i] for i in order]
        w = pd.Series(1.0, index=tickers)

        def _cluster_var(items):
            sub = cov.loc[items, items].values
            ivp = 1 / np.diag(sub)
            ivp /= ivp.sum()
            return float(ivp @ sub @ ivp)

        clusters = [tickers]
        while clusters:
            nxt = []
            for cl in clusters:
                if len(cl) < 2:
                    continue
                mid = len(cl) // 2
                left, right = cl[:mid], cl[mid:]
                vl, vr = _cluster_var(left), _cluster_var(right)
                alpha = 1 - vl / (vl + vr) if (vl + vr) > 0 else 0.5
                w[left] *= alpha
                w[right] *= (1 - alpha)
                nxt += [left, right]
            clusters = nxt
        w = w / w.sum()
        return w.reindex(returns_df.columns)
    except Exception:
        return None


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

    whrp = hrp_weights(rets)
    assert whrp is not None and abs(whrp.sum() - 1) < 1e-9 and (whrp >= 0).all()
    ret_hp, vol_hp = port_perf(whrp.to_numpy(), mu, cov)
    print(f"  HRP  vol={vol_hp:.4f}  權重={dict(whrp.round(3))}")
    assert vol_hp <= vol_eq + 1e-9              # HRP 分散應優於等權
    assert whrp["BOND"] == whrp.max()           # 低波動資產拿最大權重
    assert hrp_weights(rets[["BOND"]]) is None  # 單資產無法聚類

    print("\n✅ portfolio_opt 純邏輯測試通過")

# ── Black-Litterman（絕對觀點 + Idzorek 信心）────────────────────────────────

def market_implied_returns(cov: np.ndarray, w_mkt: np.ndarray, delta: float = 2.5) -> np.ndarray:
    """先驗 Π = δ Σ w_mkt（反向最佳化）。"""
    return delta * cov @ w_mkt


def _cap_weights(w: np.ndarray, max_w: float, iters: int = 50) -> np.ndarray:
    """單檔上限（注水法）：超上限者封頂，多出的比例按未封頂者重分配，直到無人超限；
    n×max_w < 1 時退回等於 max_w 的可行解（總和 < 1，餘為現金）。"""
    w = np.asarray(w, dtype=float).copy()
    n = len(w)
    if n == 0 or max_w <= 0:
        return w
    if n * max_w < 1.0 - 1e-12:
        return np.minimum(w / w.sum() if w.sum() > 0 else w, max_w)
    for _ in range(iters):
        over = w > max_w + 1e-12
        if not over.any():
            break
        excess = float((w[over] - max_w).sum())
        w[over] = max_w
        free = ~over & (w > 0)
        if not free.any():
            break
        w[free] += excess * w[free] / w[free].sum()
    return w


def black_litterman(cov: np.ndarray, w_mkt: np.ndarray, views: dict, tickers: list[str],
                    confidences: dict | None = None, delta: float = 2.5, tau: float = 0.05,
                    long_only: bool = True, max_w: float = 0.10) -> dict:
    """
    絕對觀點 BL：views = {ticker: 年化期望報酬}（估值層：(公允/市價)^(1/T) − 1），
    confidences = {ticker: 0..1}（Idzorek 2005：信心 → Ω 對角，經由「100% 信心下的權重傾斜 × 信心」反解）。
    回 {mu_prior, mu_bl, w_bl, w_mkt, omega}。純 numpy；無觀點 → 回市場先驗。
    """
    n = len(tickers)
    cov = np.asarray(cov, dtype=float)
    w_mkt = np.asarray(w_mkt, dtype=float)
    w_mkt = w_mkt / w_mkt.sum() if w_mkt.sum() > 0 else np.ones(n) / n
    pi = market_implied_returns(cov, w_mkt, delta)
    idx = {t: i for i, t in enumerate(tickers)}
    vk = [t for t in views if t in idx and views[t] is not None and np.isfinite(views[t])]
    if not vk:
        return {"mu_prior": pi, "mu_bl": pi, "w_bl": w_mkt, "w_mkt": w_mkt, "omega": None, "n_views": 0}
    P = np.zeros((len(vk), n))
    Q = np.zeros(len(vk))
    for k, t in enumerate(vk):
        P[k, idx[t]] = 1.0
        Q[k] = float(views[t])
    tau_cov = tau * cov
    # Idzorek：每個觀點的 Ω_k 由「信心 c_k」反解——c_k=1 時 Ω_k→0（完全採信），c_k=0 時 Ω_k→∞（忽略）
    omega = np.zeros((len(vk), len(vk)))
    for k, t in enumerate(vk):
        c = float((confidences or {}).get(t, 0.5) or 0.0)
        c = min(max(c, 0.0), 1.0)
        base = float(P[k] @ tau_cov @ P[k].T)
        if c >= 0.999:
            omega[k, k] = base * 1e-6
        elif c <= 0.001:
            omega[k, k] = base * 1e6
        else:
            omega[k, k] = base * (1.0 - c) / c          # Idzorek 閉式近似（Walters 2014）
    inv_tc = np.linalg.inv(tau_cov)
    inv_om = np.linalg.inv(omega)
    post_cov = np.linalg.inv(inv_tc + P.T @ inv_om @ P)
    mu_bl = post_cov @ (inv_tc @ pi + P.T @ inv_om @ Q)
    w = np.linalg.solve(delta * cov, mu_bl)
    if long_only:
        w = np.clip(w, 0.0, None)
    if w.sum() > 0:
        w = w / w.sum()
    w = _cap_weights(w, max_w)
    return {"mu_prior": pi, "mu_bl": mu_bl, "w_bl": w, "w_mkt": w_mkt, "omega": np.diag(omega), "n_views": len(vk)}



if __name__ == "__main__":
    # Black-Litterman：無觀點＝市場先驗；正觀點高信心 → 權重上升；信心 0 → 幾乎不動；上限 max_w
    rng = np.random.default_rng(5)
    A = rng.normal(size=(6, 4))
    cov = (A.T @ A) / 200 + np.eye(4) * 0.02
    tk = ["A", "B", "C", "D"]
    w_mkt = np.array([0.4, 0.3, 0.2, 0.1])
    r0 = black_litterman(cov, w_mkt, {}, tk)
    assert r0["n_views"] == 0 and np.allclose(r0["w_bl"], w_mkt)
    r1 = black_litterman(cov, w_mkt, {"D": 0.30}, tk, {"D": 0.9}, max_w=0.6)
    assert r1["w_bl"][3] > w_mkt[3] and r1["mu_bl"][3] > r1["mu_prior"][3]
    r2 = black_litterman(cov, w_mkt, {"D": 0.30}, tk, {"D": 0.0}, max_w=0.6)
    assert abs(r2["w_bl"][3] - w_mkt[3]) < 1e-3                      # 零信心 ≈ 先驗
    r3 = black_litterman(cov, w_mkt, {"D": 0.30, "A": -0.20}, tk, {"D": 1.0, "A": 1.0}, max_w=0.35)
    assert r3["w_bl"].max() <= 0.35 + 1e-9 and abs(r3["w_bl"].sum() - 1) < 1e-9 and r3["w_bl"][0] < w_mkt[0]
    assert black_litterman(cov, w_mkt, {"ZZ": 0.1, "D": float("nan")}, tk)["n_views"] == 0
    print("✅ BL（無觀點=先驗、信心單調、上限、未知/NaN 觀點忽略）")
    print("portfolio_opt selftest OK ✅")
