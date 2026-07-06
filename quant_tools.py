"""
quant_tools.py – 部位配置與風險管理工具（可重用，無 Streamlit 依賴）

學習自：
  • freqtrade — stake_amount 動態部位、波動率調整
  • Riskfolio-Lib / Eiten — 風險平價（Risk Parity）配置
  • Kelly (1956) / Thorp — 凱利公式與分數凱利
  • López de Prado — 風險貢獻分解

提供：
  • atr_position_size      ATR 風險基準部位（每筆固定風險）
  • volatility_target      波動率目標槓桿（拉齊組合波動）
  • kelly_fraction         凱利下注比例（含分數凱利上限）
  • inverse_vol_weights    反波動加權（簡易風險平價）
  • risk_parity_weights    等風險貢獻 ERC（含相關性）
  • risk_contributions     檢視各資產風險貢獻（驗證用）
"""

from __future__ import annotations

import numpy as np


# ── 單筆部位 ─────────────────────────────────────────────────────────────────

def atr_position_size(account: float, risk_pct: float, entry: float,
                      atr: float, atr_mult: float = 1.5) -> dict:
    """
    ATR 風險基準部位：每筆交易只冒帳戶 risk_pct 的風險。
      風險/股 = ATR × atr_mult（停損距離）
      股數    = (帳戶 × risk_pct) / 風險/股
    回傳股數、部位金額、佔帳戶比例、停損價、風險金額。
    """
    if entry <= 0 or atr <= 0 or account <= 0 or risk_pct <= 0:
        return {"shares": 0, "position_value": 0.0, "pct_of_account": 0.0,
                "stop_price": 0.0, "risk_amount": 0.0}
    risk_per_share = atr * atr_mult
    risk_budget = account * risk_pct
    shares = risk_budget / risk_per_share
    position_value = shares * entry
    return {
        "shares":         round(shares, 2),
        "position_value": round(position_value, 2),
        "pct_of_account": round(position_value / account, 4),
        "stop_price":     round(entry - risk_per_share, 2),
        "risk_amount":    round(risk_budget, 2),
    }


def volatility_target(realized_vol: float, target_vol: float = 0.15,
                      max_leverage: float = 2.0) -> float:
    """
    波動率目標槓桿係數 = target_vol / realized_vol，上限 max_leverage。
    高波動標的 → 係數 <1（少買）；低波動 → 係數 >1（可加碼，受上限約束）。
    realized_vol / target_vol 皆為「年化」波動率（如 0.15 = 15%）。
    """
    if realized_vol <= 0:
        return 0.0
    return round(min(target_vol / realized_vol, max_leverage), 3)


def kelly_fraction(win_rate: float, win_loss_ratio: float,
                   cap: float = 0.25) -> dict:
    """
    凱利公式：f* = (p(b+1) − 1) / b ，其中 p=勝率, b=賺賠比。
    回傳完整凱利、分數凱利（受 cap 約束，預設 1/4 Kelly）。
    f* < 0 代表此策略期望為負，不應下注。
    """
    p = max(0.0, min(1.0, win_rate))
    b = win_loss_ratio
    if b <= 0:
        return {"full_kelly": 0.0, "fractional": 0.0, "edge": False}
    f = (p * (b + 1) - 1) / b
    frac = max(0.0, min(f, cap))   # 不放空、且不超過 cap
    return {
        "full_kelly":  round(f, 4),
        "fractional":  round(frac, 4),
        "edge":        f > 0,
    }


# ── 組合配置 ─────────────────────────────────────────────────────────────────

def inverse_vol_weights(vols) -> np.ndarray:
    """反波動加權（風險平價的無相關近似）：w_i ∝ 1/σ_i。"""
    vols = np.asarray(vols, dtype=float)
    vols = np.where(vols <= 0, np.nan, vols)
    inv = 1.0 / vols
    inv = np.nan_to_num(inv, nan=0.0)
    s = inv.sum()
    return inv / s if s > 0 else np.full(len(vols), 1.0 / len(vols))


def risk_contributions(weights, cov) -> np.ndarray:
    """各資產對組合總風險的貢獻（總和 = 組合波動率 σ_p）。"""
    w = np.asarray(weights, dtype=float)
    cov = np.asarray(cov, dtype=float)
    port_var = float(w @ cov @ w)
    if port_var <= 0:
        return np.zeros(len(w))
    sigma = np.sqrt(port_var)
    mrc = cov @ w / sigma          # 邊際風險貢獻
    return w * mrc                 # 風險貢獻（加總 = sigma）


def risk_parity_weights(cov, max_iter: int = 2000, tol: float = 1e-10) -> np.ndarray:
    """
    等風險貢獻（Equal Risk Contribution, ERC）配置。

    用「循環座標下降法」（Spinu 2013 / Griveau-Billion 2013）求解，對任意
    正半定共變異數矩陣都可證明收斂——逐一更新每個權重至其風險貢獻達 1/N：
        w_i = (-c_i + √(c_i² + 4·σ_ii/N)) / (2·σ_ii)
    其中 c_i = Σ_{j≠i} σ_ij·w_j。僅做多、最終正規化至總和 1。

    （先前的乘法更新法在 n≥3、相關性較高時會震盪並塌縮到單一資產角解，
      已改用此可證明收斂的解法。）
    """
    cov = np.asarray(cov, dtype=float)
    n = cov.shape[0]
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])

    # 退化檢查：對角線（變異數）需為正且矩陣有限
    var = np.diag(cov)
    if np.any(var <= 0) or not np.all(np.isfinite(cov)):
        return inverse_vol_weights(np.sqrt(np.clip(var, 1e-12, None)))

    w = 1.0 / np.sqrt(var)        # 反波動起始（未正規化）
    for _ in range(max_iter):
        w_old = w.copy()
        for i in range(n):
            ci = cov[i] @ w - cov[i, i] * w[i]      # Σ_{j≠i} σ_ij·w_j
            w[i] = (-ci + np.sqrt(ci * ci + 4.0 * cov[i, i] / n)) / (2.0 * cov[i, i])
        if np.max(np.abs(w - w_old)) < tol:
            break
    w = np.clip(w, 0, None)
    s = w.sum()
    return w / s if s > 0 else np.full(n, 1.0 / n)


# ── CLI 自我測試 ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== ATR 部位 ===")
    print(atr_position_size(account=100000, risk_pct=0.01, entry=150, atr=4.0))

    print("\n=== 波動率目標 ===")
    for v in [0.10, 0.15, 0.30, 0.60]:
        print(f"  realized_vol={v:.0%} → leverage {volatility_target(v):.2f}")

    print("\n=== Kelly ===")
    print(" win55/ratio1.8:", kelly_fraction(0.55, 1.8))
    print(" win40/ratio1.0:", kelly_fraction(0.40, 1.0))

    print("\n=== Risk Parity 驗證（風險貢獻應相等）===")
    # 用較高相關性 + 5 檔測試（先前乘法解法會在此塌縮，座標下降法不會）
    vols = np.array([0.10, 0.20, 0.40, 0.15, 0.30])
    corr = np.array([
        [1.0, 0.6, 0.3, 0.5, 0.2],
        [0.6, 1.0, 0.4, 0.3, 0.5],
        [0.3, 0.4, 1.0, 0.2, 0.6],
        [0.5, 0.3, 0.2, 1.0, 0.4],
        [0.2, 0.5, 0.6, 0.4, 1.0],
    ])
    cov = np.outer(vols, vols) * corr
    w_erc = risk_parity_weights(cov)
    rc = risk_contributions(w_erc, cov)
    print("  weights:", np.round(w_erc, 4), "  sum:", round(float(w_erc.sum()), 6))
    print("  risk contrib %:", np.round(rc / rc.sum(), 4), "(應全部 ~0.20)")
    print("  inverse-vol:", np.round(inverse_vol_weights(vols), 4))
