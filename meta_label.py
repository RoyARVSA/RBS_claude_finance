"""
meta_label.py – 交易層 meta-labeling（B 段；規劃見 ALPHA_SPINE.md §2）

主模型＝現行引擎進場條件（評分 ≥ 門檻、非延伸端、非 FOMC 靜默日）。
次模型只回答一件事：「在這種情境下，主模型發出的訊號過去成功率多少」→ 部位倍數。
  樣本：整個選股池逐日的訊號時點（同檔連續成立只取首日，之後每 10 日一筆）
  標籤：用引擎規則模擬單筆出場（硬停損 / +1R 追蹤 8% / +2R 收緊 5% / 45 日時間柵欄）→ R 倍數
  特徵：延伸度、5 日漲幅、評分、波動、12-1 動能、1 月反轉、SPY/MA50 三態、廣度、品質(PIT)、預估修正(PIT)
        —— 訓練與線上共用同一個 features()（特徵漂移＝最常見的線上失效原因）
  模型：numpy 邏輯迴歸（L2、牛頓法）——無新依賴、係數可讀；purged walk-forward 4 折 + embargo
  閘門：OOS AUC ≥ 0.55 且 勝率尺寸化 Sharpe > 等額 且 n_oos ≥ 300；未過 → 線上不用

純邏輯、離線可測（合成資料植入效應 → AUC 顯著；打亂標籤 → 閘門不過）。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

META_FILE = Path(__file__).parent / "data" / "alpha" / "meta.json"

FEATURES = ("ext_atr", "ret_5d", "score", "vol_60", "mom_12_1", "rev_1m",
            "regime_on", "regime_off", "breadth_pct", "quality", "quality_na", "rev", "rev_na", "mom_na",
            "rsi", "hi_dist_252", "atr_pct", "volratio_20")      # 後四欄：RSI14／距 52 週高／ATR 占價比／20 日量比
GBM_PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 7, "min_data_in_leaf": 50,
              "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
              "verbose": -1, "seed": 7, "num_threads": 2}
GBM_MAX_ROUNDS, GBM_DEFAULT_ROUNDS, GBM_EARLY_STOP = 400, 150, 30
LABEL_CFG = {                      # 與 trade_engine.ENGINE_DEFAULTS 對齊（出場規則）
    "buy_threshold": 0.5, "entry_max_ext_atr": 2.0, "entry_max_ret5d": 0.06,
    "stop_mult": 1.0, "trail_activate_r": 1.0, "trail_pct": 0.08, "trail_tight_pct": 0.05, "trail_tighten_r": 2.0,
    "exit_threshold": -0.2, "dead_money_days": 30, "dead_money_ret": 0.02,      # 訊號轉弱只在獲利時了結／死錢釋放（與引擎同）
    "max_days": 45, "cost_side": 0.0005, "atr_mult": 1.5, "resample_days": 10,
}
GATE = {"auc_min": 0.55, "n_oos_min": 300}
SIZE = {"lo": 0.40, "hi": 0.65, "max_mult": 1.25}   # 預設；訓練時改為相對基準勝率（見 size_cfg_from_base）
GATE_MAX_SKIP = 0.80                                 # 尺寸化不可靠「什麼都不做」取勝：跳過率上限


def size_cfg_from_base(base_rate: float | None) -> dict:
    """尺寸門檻相對基準勝率：lo = base − 8pp（≥0.25）、hi = base + 12pp（≤0.90）。基準未知 → 預設。"""
    if base_rate is None or not math.isfinite(float(base_rate)):
        return dict(SIZE)
    b = float(base_rate)
    return {"lo": round(max(0.25, b - 0.08), 3), "hi": round(min(0.90, b + 0.12), 3), "max_mult": SIZE["max_mult"]}
N_FOLDS = 4
EMBARGO_DAYS = 46                  # ≥ max_days + 1：訓練標籤結算日不得落在測試段內


# ── 1. 特徵（訓練＝線上同一函數）──────────────────────────────────────────────

def _num(x, default=0.0):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def features(row: dict, ctx: dict | None = None) -> list[float]:
    """
    row：ext_atr / ret_5d / score / vol_60 / mom_12_1 / ret_1m 或 rev_1m / quality / rev（缺→0＋旗標）。
    ctx：regime ∈ {risk_on, neutral, risk_off, None}、breadth_pct（缺 → 0.5 中性）。
    回與 FEATURES 同序的數值列。
    """
    ctx = ctx or {}
    rev_1m = row.get("rev_1m")
    if rev_1m is None and row.get("ret_1m") is not None:
        rev_1m = -_num(row.get("ret_1m"))
    q, r = row.get("quality"), row.get("rev")
    reg = ctx.get("regime")
    mom = row.get("mom_12_1")
    mom_na = 1.0 if (mom is None or not math.isfinite(_num(mom, float("nan")))) else 0.0   # 訓練前 252 天／線上歷史不足皆補 0＋旗標
    return [
        _num(row.get("ext_atr")), _num(row.get("ret_5d")), _num(row.get("score")), _num(row.get("vol_60")),
        _num(mom), _num(rev_1m),
        1.0 if reg == "risk_on" else 0.0, 1.0 if reg == "risk_off" else 0.0,
        _num(ctx.get("breadth_pct"), 0.5),
        _num(q), 0.0 if q is not None else 1.0, _num(r), 0.0 if r is not None else 1.0,
        mom_na,
        _num(row.get("rsi"), 50.0), _num(row.get("hi_dist_252")), _num(row.get("atr_pct")), _num(row.get("volratio_20"), 1.0),
    ]


# ── 2. 標籤：引擎規則的單筆出場模擬（無同棒前視：次日開盤進場）─────────────

def simulate_exit(close: np.ndarray, open_: np.ndarray, i: int, rps: float, cfg: dict | None = None,
                  score: np.ndarray | None = None) -> dict | None:
    """
    訊號在第 i 根收盤確認 → 第 i+1 根開盤進場；之後逐日以收盤檢查（順序同 trade_engine.decide）：
      硬停損（追蹤啟動前）→ 追蹤（+1R 起、地板保本；+2R 收緊）→ 訊號轉弱且獲利中了結（score 給了才有）
      → 死錢釋放（≥30 日、<+2%、評分 < 門檻）→ 時間柵欄。
    未模擬：分批鎖利（只影響尺寸）、曝險非 ACTIVE 的收緊、盤中 15 分鐘價（ALPHA_SPINE §2.1）。
    回 {r_mult, days, mech, entry, exit, truncated}；資料不足回 None。
    """
    c = {**LABEL_CFG, **(cfg or {})}
    n = len(close)
    if i + 2 >= n or not (rps and rps > 0):
        return None
    entry = float(open_[i + 1]) if open_ is not None and not np.isnan(open_[i + 1]) else float(close[i + 1])
    if not (entry > 0):
        return None
    stop_line = entry - float(c["stop_mult"]) * rps
    peak = entry
    last = min(i + 1 + int(c["max_days"]), n - 1)
    # 時間柵欄出場價＝區間內最後一個有效收盤（寬表以宇宙日期聯集對齊，close[last] 可能缺值 → NaN 標籤，#55）
    mech, exit_px, j_exit = "time", None, last
    last_valid = None
    for j in range(i + 1, last + 1):
        px = float(close[j])
        if np.isnan(px):
            continue
        last_valid = (j, px)
        peak = max(peak, px)
        r_peak = (peak - entry) / rps
        if r_peak < float(c["trail_activate_r"]) and px <= stop_line:
            mech, exit_px, j_exit = "stop_loss", px, j
            break
        if r_peak >= float(c["trail_activate_r"]):
            pct = float(c["trail_tight_pct"]) if r_peak >= float(c["trail_tighten_r"]) else float(c["trail_pct"])
            trail = max(entry, peak * (1 - pct))
            if px <= trail:
                mech, exit_px, j_exit = "trailing_stop", px, j
                break
        if score is not None and j > i + 1:
            sc_j = float(score[j]) if score[j] == score[j] else 0.0
            if sc_j <= float(c["exit_threshold"]) and px > entry:
                mech, exit_px, j_exit = "signal_exit", px, j
                break
            if (j - (i + 1)) >= int(c["dead_money_days"]) and (px / entry - 1) < float(c["dead_money_ret"]) \
                    and sc_j < float(c["buy_threshold"]):
                mech, exit_px, j_exit = "dead_money", px, j
                break
    if exit_px is None:                         # 沒有觸發任何出場 → 時間柵欄
        if last_valid is None:
            return None                         # 進場後完全沒有有效收盤：無法標籤
        j_exit, exit_px = last_valid            # 尾段缺值（資料缺口或下市）→ 以最後有效價結算
        if j_exit < last:
            mech = "data_end"                   # 與完整 45 日時間出場分開統計
    days = j_exit - (i + 1)
    truncated = (mech == "time" and last == n - 1 and (last - (i + 1)) < int(c["max_days"]))
    cost_r = 2 * float(c["cost_side"]) * entry / rps
    return {"r_mult": (exit_px - entry) / rps - cost_r, "days": int(days), "mech": mech,
            "entry": entry, "exit": exit_px, "truncated": bool(truncated)}


def signal_days(score: np.ndarray, ext: np.ndarray, ret5: np.ndarray, dates: list[str],
                cfg: dict | None = None, blackout_fn=None) -> list[int]:
    """引擎進場條件成立的位置（去重：連續成立取首日，之後每 resample_days 一筆；靜默日跳過）。"""
    c = {**LABEL_CFG, **(cfg or {})}
    out, last_taken = [], -10**9
    for i in range(len(score)):
        s = score[i]
        on = (s == s) and s >= float(c["buy_threshold"])
        if on and ext is not None:
            e, r5 = ext[i], (ret5[i] if ret5 is not None else float("nan"))
            if (e == e) and e > float(c["entry_max_ext_atr"]) and (not (r5 == r5) or r5 > float(c["entry_max_ret5d"])):
                on = False
        if on and blackout_fn is not None:
            try:
                if blackout_fn(dates[i]):
                    on = False
            except Exception:
                pass
        if on and (i - last_taken) >= int(c["resample_days"]):      # 同檔兩筆樣本至少隔 resample_days（門檻閃爍不灌水，M3）
            out.append(i); last_taken = i
    return out


def build_samples(frames: dict[str, pd.DataFrame], scores: pd.DataFrame, ctx_by_date: dict[str, dict],
                  fund_lookup=None, cfg: dict | None = None, blackout_fn=None) -> list[dict]:
    """
    frames：alpha_spine.wide_frames（close/high/low/open 可缺）；scores：composite_series 寬表（同 index/columns）。
    ctx_by_date：{date: {regime, breadth_pct}}；fund_lookup(sym, date) → {"quality","rev"}（PIT，可 None）。
    回樣本列 [{date, pos, sym, x, r_mult, win, days, mech}]。
    """
    import alpha_spine as asp
    c = {**LABEL_CFG, **(cfg or {})}
    close = frames["close"]
    opens = frames.get("open")
    a = asp.atr14(close, frames.get("high"), frames.get("low"))
    pf = asp.price_factors(frames)
    ext_w = -pf["ext_atr"]                       # 還原成正向延伸度（引擎語意）
    ret5_w = close / close.shift(5) - 1
    import factor_lab as _fl
    rsi_w = _fl._rsi_wide(close, 14)
    hi_w = close / close.rolling(252).max() - 1
    atrp_w = a / close.replace(0, np.nan)
    vol_f = frames.get("volume")
    vr_w = (vol_f / vol_f.rolling(20).mean().shift(1).replace(0, np.nan)) if (vol_f is not None and not vol_f.empty) else None
    dates = [str(d.date()) for d in close.index]
    out = []
    for sym in close.columns:
        if sym not in scores.columns:
            continue
        cl = close[sym].values.astype(float)
        op = opens[sym].reindex(close.index).values.astype(float) if (opens is not None and sym in opens) else None
        sc = scores[sym].reindex(close.index).values.astype(float)
        ex = ext_w[sym].values.astype(float)
        r5 = ret5_w[sym].values.astype(float)
        rps_w = (a[sym] * float(c["atr_mult"])).values.astype(float)
        vol60 = -pf["vol_60"][sym].values.astype(float)
        mom = pf["mom_12_1"][sym].values.astype(float)
        rev1 = pf["rev_1m"][sym].values.astype(float)
        rsi_v = rsi_w[sym].values.astype(float) if sym in rsi_w else np.full(len(cl), np.nan)
        hid = hi_w[sym].values.astype(float)
        atrp = atrp_w[sym].values.astype(float)
        vr = vr_w[sym].values.astype(float) if (vr_w is not None and sym in vr_w) else np.full(len(cl), np.nan)
        for i in signal_days(sc, ex, r5, dates, c, blackout_fn):
            rps = rps_w[i]
            if not (rps == rps) or rps <= 0:
                continue
            lab = simulate_exit(cl, op, i, float(rps), c, score=sc)
            if lab is None or lab["truncated"] or not np.isfinite(lab["r_mult"]):   # 非有限標籤一律丟棄（#55 防禦）
                continue
            f = fund_lookup(sym, dates[i]) if fund_lookup else {}
            row = {"ext_atr": ex[i], "ret_5d": r5[i], "score": sc[i], "vol_60": vol60[i], "mom_12_1": mom[i],
                   "rev_1m": rev1[i], "quality": (f or {}).get("quality"), "rev": (f or {}).get("rev"),
                   "rsi": rsi_v[i], "hi_dist_252": hid[i], "atr_pct": atrp[i], "volratio_20": vr[i]}
            out.append({"date": dates[i], "pos": i, "sym": sym, "x": features(row, ctx_by_date.get(dates[i]) or {}),
                        "r_mult": float(lab["r_mult"]), "win": bool(lab["r_mult"] > 0), "days": lab["days"], "mech": lab["mech"]})
    out.sort(key=lambda s: (s["pos"], s["sym"]))
    return out


# ── 3. 邏輯迴歸（numpy、L2、牛頓法）───────────────────────────────────────────

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 25) -> dict:
    """標準化 → 牛頓法；截距不正則。回 {mean, std, coef, intercept}。"""
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    Z = (X - mu) / sd
    n, k = Z.shape
    A = np.hstack([np.ones((n, 1)), Z])
    w = np.zeros(k + 1)
    R = np.eye(k + 1) * l2; R[0, 0] = 0.0
    for _ in range(iters):
        p = _sigmoid(A @ w)
        g = A.T @ (p - y) + R @ w
        W = p * (1 - p)
        H = (A * W[:, None]).T @ A + R + np.eye(k + 1) * 1e-8
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        w -= step
        if float(np.abs(step).max()) < 1e-8:
            break
    return {"mean": mu.tolist(), "std": sd.tolist(), "coef": w[1:].tolist(), "intercept": float(w[0]), "l2": l2}


def predict_logit(model: dict, x) -> float:
    z = (np.asarray(x, dtype=float) - np.asarray(model["mean"])) / np.asarray(model["std"])
    return float(_sigmoid(float(np.dot(z, np.asarray(model["coef"])) + float(model["intercept"]))))


# ── 3b. LightGBM 候選（夜間可用時訓練；線上以純 Python 樹遍歷推論、不加依賴）─────

def gbm_available() -> bool:
    try:
        import lightgbm  # noqa: F401
        return True
    except Exception:
        return False


def _compact_trees(dump: dict) -> list[list[list]]:
    """LightGBM dump_model → 精簡節點表：內部 [feat, thr, left, right, default_left]；葉 [-1, value, 0, 0, 0]。"""
    trees = []
    for t in dump.get("tree_info", []):
        nodes: list[list] = []

        def _walk(nd) -> int:
            idx = len(nodes)
            if "leaf_value" in nd and "split_feature" not in nd:
                nodes.append([-1, float(nd["leaf_value"]), 0, 0, 0])
                return idx
            # 第 5 欄：NaN 走向——missing_type "NaN" → default_left 分支；"None"（訓練無缺值）→ LightGBM 把 NaN 當 0.0 比較
            mt = str(nd.get("missing_type", "None"))
            nodes.append([int(nd["split_feature"]), float(nd["threshold"]), 0, 0,
                          (1 if nd.get("default_left", True) else 0) if mt != "None" else 2])
            if str(nd.get("decision_type", "<=")) != "<=":
                raise ValueError(f"unsupported decision_type {nd.get('decision_type')}")
            left = _walk(nd["left_child"]); right = _walk(nd["right_child"])
            nodes[idx][2], nodes[idx][3] = left, right
            return idx
        _walk(t["tree_structure"])
        trees.append(nodes)
    return trees


def predict_gbm(model: dict, x) -> float:
    xs = [float(v) for v in x]
    raw = 0.0
    for nodes in model["trees"]:
        i = 0
        while nodes[i][0] != -1:
            f, thr, l, r, dl = nodes[i]
            v = xs[f] if f < len(xs) else float("nan")
            if v != v:
                if dl == 2:                       # missing_type None：與 LightGBM 同，NaN 視為 0.0 比較
                    i = l if 0.0 <= thr else r
                else:
                    i = l if dl else r
            else:
                i = l if v <= thr else r
        raw += nodes[i][1]
    return float(_sigmoid(raw))


def fit_gbm(X: np.ndarray, y: np.ndarray, rounds: int | None = None, valid: tuple | None = None,
            keep_booster: bool = False) -> dict:
    """valid=(Xv, yv) 時 early stopping（最多 GBM_MAX_ROUNDS）；否則固定 rounds。回 {kind, trees, best_iter}。"""
    import lightgbm as lgb
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    dtr = lgb.Dataset(X, label=y, free_raw_data=False)
    if valid is not None and len(valid[0]) >= 30:
        dv = lgb.Dataset(np.asarray(valid[0], dtype=float), label=np.asarray(valid[1], dtype=float), reference=dtr)
        bst = lgb.train(GBM_PARAMS, dtr, num_boost_round=GBM_MAX_ROUNDS, valid_sets=[dv],
                        callbacks=[lgb.early_stopping(GBM_EARLY_STOP, verbose=False)])
        best = int(bst.best_iteration or bst.num_trees())
    else:
        best = int(rounds or GBM_DEFAULT_ROUNDS)
        bst = lgb.train(GBM_PARAMS, dtr, num_boost_round=best)
    out = {"kind": "gbm", "trees": _compact_trees(bst.dump_model(num_iteration=best)), "best_iter": best,
           "n_trees": best, "params": dict(GBM_PARAMS)}
    if keep_booster:
        out["_booster"] = bst
    return out


def predict_p(model: dict, x) -> float:
    """依 model["kind"]（logit 預設／gbm）推論。"""
    return predict_gbm(model, x) if model.get("kind") == "gbm" else predict_logit(model, x)


def _fit_kind(kind: str, X, y, l2: float = 1.0, valid: tuple | None = None, rounds: int | None = None) -> dict:
    if kind == "gbm":
        return fit_gbm(X, y, rounds=rounds, valid=valid)
    m = fit_logistic(X, y, l2); m["kind"] = "logit"
    return m


def size_from_p(p: float | None, lo: float | None = None, hi: float | None = None, max_mult: float | None = None) -> float:
    """勝率 → 部位倍數：p<lo → 0（跳過）；lo..hi 線性 0.5→max；≥hi → max。None → 1.0（不干預）。"""
    if p is None:
        return 1.0
    lo = SIZE["lo"] if lo is None else lo; hi = SIZE["hi"] if hi is None else hi
    mx = SIZE["max_mult"] if max_mult is None else max_mult
    if p < lo:
        return 0.0
    if hi <= lo:
        return mx
    return round(min(mx, 0.5 + (mx - 0.5) * min(1.0, (p - lo) / (hi - lo))), 3)


# ── 4. 驗證：purged walk-forward ─────────────────────────────────────────────

def auc_score(y: np.ndarray, p: np.ndarray) -> float | None:
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    if n1 == 0 or n0 == 0:
        return None
    ranks = pd.Series(p).rank().values
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _logloss(y, p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6); y = np.asarray(y, dtype=float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def _sharpe_like(r: np.ndarray) -> float | None:
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2 or r.std(ddof=1) == 0:
        return None
    return float(r.mean() / r.std(ddof=1))


def purged_walk_forward(samples: list[dict], n_folds: int = N_FOLDS, embargo: int = EMBARGO_DAYS,
                        l2: float = 1.0, size_cfg: dict | None = None, kind: str = "logit") -> dict:
    """
    依時間位置切 n_folds+1 個等長區塊：第 f 折用區塊 <f 訓練（剔除 pos > 測試起點 − embargo 的樣本）、區塊 f 測試。
    回 OOS 合併指標 + 各折 + 閘門。
    """
    n_in = len(samples or [])
    samples = [s for s in (samples or []) if np.isfinite(s.get("r_mult", float("nan")))
               and all(np.isfinite(v) for v in s.get("x", []))]
    n_dropped = n_in - len(samples)
    if not samples:
        return {"n_oos": 0, "gate_passed": False, "reason": "無樣本", "n_dropped": n_dropped}
    pos = np.array([s["pos"] for s in samples]); X = np.array([s["x"] for s in samples], dtype=float)
    y = np.array([1.0 if s["win"] else 0.0 for s in samples]); r = np.array([s["r_mult"] for s in samples], dtype=float)
    lo_p, hi_p = int(pos.min()), int(pos.max())
    edges = np.linspace(lo_p, hi_p + 1, n_folds + 2)
    folds, oos_p, oos_y, oos_r, oos_s, best_iters = [], [], [], [], [], []
    sc = size_cfg or {}
    for f in range(1, n_folds + 1):
        t0, t1 = edges[f], edges[f + 1]
        te = (pos >= t0) & (pos < t1)
        tr = (pos < t0 - embargo)
        if te.sum() < 20 or tr.sum() < 50 or len(set(y[tr])) < 2:
            folds.append({"fold": f, "n_train": int(tr.sum()), "n_test": int(te.sum()), "skipped": True})
            continue
        if kind == "gbm":
            tr_idx = np.where(tr)[0]                                   # 訓練段按時間切最後 15% 當 early-stopping 驗證（不碰測試段）
            cut = int(len(tr_idx) * 0.85)
            m = fit_gbm(X[tr_idx[:cut]], y[tr_idx[:cut]], valid=(X[tr_idx[cut:]], y[tr_idx[cut:]]))
            best_iters.append(int(m.get("best_iter") or GBM_DEFAULT_ROUNDS))
        else:
            m = fit_logistic(X[tr], y[tr], l2); m["kind"] = "logit"
        p = np.array([predict_p(m, x) for x in X[te]])
        a = auc_score(y[te], p)
        folds.append({"fold": f, "n_train": int(tr.sum()), "n_test": int(te.sum()), "auc": a,
                      "base_rate": float(y[tr].mean()), "test_rate": float(y[te].mean()),
                      **({"best_iter": m.get("best_iter")} if kind == "gbm" else {})})
        fsc = sc if sc else size_cfg_from_base(float(y[tr].mean()))      # 尺寸門檻只用該折訓練段基準（無前視，L4）
        oos_s.append(np.array([size_from_p(pp, fsc.get("lo"), fsc.get("hi"), fsc.get("max_mult")) for pp in p]))
        oos_p.append(p); oos_y.append(y[te]); oos_r.append(r[te])
    if not oos_p:
        return {"n_oos": 0, "folds": folds, "gate_passed": False, "reason": "樣本不足以切折"}
    P, Y, Rr = np.concatenate(oos_p), np.concatenate(oos_y), np.concatenate(oos_r)
    sizes = np.concatenate(oos_s)
    eq_s, sz_s = _sharpe_like(Rr), _sharpe_like(sizes * Rr)
    base = float(np.concatenate([np.full(len(yy), yy.mean()) for yy in oos_y]).mean())
    q = pd.qcut(pd.Series(P).rank(method="first"), 5, labels=False)
    calib = []
    for b in range(5):
        mask = (q.values == b)
        if mask.any():
            calib.append({"bin": b + 1, "p_mean": round(float(P[mask].mean()), 3), "hit": round(float(Y[mask].mean()), 3),
                          "r_mean": round(float(Rr[mask].mean()), 3), "n": int(mask.sum())})
    out = {"n_oos": int(len(P)), "auc": auc_score(Y, P), "logloss": _logloss(Y, P),
           "logloss_base": _logloss(Y, np.full(len(Y), max(1e-6, min(1 - 1e-6, base)))),
           "equal_sharpe": eq_s, "sized_sharpe": sz_s,
           "equal_mean_r": float(Rr.mean()), "sized_mean_r": float((sizes * Rr).mean()),
           "skip_rate": float((sizes == 0).mean()), "folds": folds, "calib": calib, "kind": kind, "n_dropped": n_dropped,
           "gbm_rounds": (int(np.median(best_iters)) if best_iters else None)}
    ok, why = gate(out)
    out["gate_passed"], out["reason"] = ok, why
    return out


def _fin(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def gate(oos: dict) -> tuple[bool, str]:
    """任何比較值非有限（None/NaN/inf）一律不通過——NaN 比較恆 False 會讓「不通過」條件被跳過而誤放行（#58）。"""
    for k in ("skip_rate", "auc", "sized_sharpe", "equal_sharpe"):
        if not _fin(oos.get(k)):
            return False, f"{k} 非有限值（{oos.get(k)}）"
    if (oos.get("n_oos") or 0) < GATE["n_oos_min"]:
        return False, f"OOS 樣本 {oos.get('n_oos', 0)} < {GATE['n_oos_min']}"
    if (oos.get("skip_rate") or 0) > GATE_MAX_SKIP:
        return False, f"跳過率 {oos.get('skip_rate'):.0%} > {GATE_MAX_SKIP:.0%}（靠不做事取勝不算）"
    if oos.get("auc") is None or oos["auc"] < GATE["auc_min"]:
        return False, f"OOS AUC {oos.get('auc')} < {GATE['auc_min']}"
    if oos.get("sized_sharpe") is None or oos.get("equal_sharpe") is None or oos["sized_sharpe"] <= oos["equal_sharpe"]:
        return False, f"尺寸化 Sharpe {oos.get('sized_sharpe')} 未勝等額 {oos.get('equal_sharpe')}"
    return True, "通過"


# ── 5. 訓練進入點 / 存取 / 文字 ───────────────────────────────────────────────

def train(samples: list[dict], as_of: str, l2: float = 1.0, kinds: tuple | None = None) -> dict:
    """
    候選模型（logit；lightgbm 可用時再加 gbm）各自走同一套 purged walk-forward → 以 OOS log-loss 擇優
    （同分取 logit）→ 全樣本重訓勝者 → meta.json 內容（含兩者 OOS 對照 candidates）。
    """
    n_raw = len(samples or [])
    samples = [s for s in (samples or []) if np.isfinite(s.get("r_mult", float("nan")))
               and all(np.isfinite(v) for v in s.get("x", []))]
    base = (float(np.mean([s["win"] for s in samples])) if samples else None)
    size_cfg = size_cfg_from_base(base)
    kinds = tuple(kinds) if kinds else (("logit", "gbm") if gbm_available() else ("logit",))
    cands = {}
    for k in kinds:
        try:
            cands[k] = purged_walk_forward(samples, l2=l2, kind=k)     # 各折尺寸門檻用訓練段基準
        except Exception as e:
            cands[k] = {"n_oos": 0, "gate_passed": False, "reason": f"{type(e).__name__}", "kind": k}
    def _key(k):
        o = cands[k]
        ll = o.get("logloss")
        return (0 if o.get("gate_passed") else 1, ll if ll is not None else 9.0, 0 if k == "logit" else 1)
    sel = sorted(cands, key=_key)[0]
    oos = cands[sel]
    model = {"version": 2, "as_of": as_of, "features": list(FEATURES), "n": len(samples), "n_dropped": n_raw - len(samples),
             "base_rate": base, "kind": sel,
             "candidates": {k: {kk: o.get(kk) for kk in ("auc", "logloss", "logloss_base", "equal_sharpe", "sized_sharpe",
                                                          "skip_rate", "n_oos", "gate_passed", "reason", "gbm_rounds")}
                            for k, o in cands.items()},
             "label_cfg": dict(LABEL_CFG), "size": size_cfg, "oos": oos,
             "gate_passed": bool(oos.get("gate_passed")), "reason": oos.get("reason")}
    if samples and len({s["win"] for s in samples}) == 2:
        X = np.array([s["x"] for s in samples], dtype=float); y = np.array([1.0 if s["win"] else 0.0 for s in samples])
        if sel == "gbm":
            model.update(fit_gbm(X, y, rounds=int(oos.get("gbm_rounds") or GBM_DEFAULT_ROUNDS)))
        else:
            model.update(fit_logistic(X, y, l2)); model["kind"] = "logit"
        mech = {}
        for s in samples:
            m = mech.setdefault(s["mech"], {"n": 0, "wins": 0, "r": 0.0})
            m["n"] += 1; m["wins"] += int(s["win"]); m["r"] += s["r_mult"]
        model["by_mech"] = {k: {"n": v["n"], "win_rate": round(v["wins"] / v["n"], 3), "avg_r": round(v["r"] / v["n"], 3)} for k, v in mech.items()}
    return model


def load_model(path: Path | None = None) -> dict | None:
    try:
        d = json.loads(Path(path or META_FILE).read_text(encoding="utf-8"))
        return d if isinstance(d, dict) and (d.get("coef") or d.get("trees")) else None
    except Exception:
        return None


def save_model(model: dict, path: Path | None = None) -> Path:
    p = Path(path or META_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    slim = {k: v for k, v in model.items() if not str(k).startswith("_")}      # _booster 等執行期物件不落檔
    tmp.write_text(json.dumps(slim, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    tmp.replace(p)
    return p


def usable(model: dict | None) -> bool:
    """過閘門、有參數、且特徵版本與現行 features() 一致（特徵改版後舊 meta.json 不得靜默失效）。"""
    return bool(model and (model.get("coef") or model.get("trees")) and model.get("gate_passed")
                and list(model.get("features") or []) == list(FEATURES))


def score_rows(model: dict, rows: list[dict], ctx: dict) -> list[dict]:
    """線上：對每列附 meta_p / meta_mult（不改原列；缺特徵照 features() 規則補 0＋旗標）。"""
    out = []
    for r in rows:
        r2 = dict(r)
        try:
            p = predict_p(model, features(r, ctx))
            r2["meta_p"] = round(p, 3)
            r2["meta_mult"] = size_from_p(p, (model.get("size") or {}).get("lo"), (model.get("size") or {}).get("hi"),
                                          (model.get("size") or {}).get("max_mult"))
        except Exception:
            pass
        out.append(r2)
    return out


def _f(x, nd=2):
    return "—" if x is None else f"{x:.{nd}f}"


def meta_text(model: dict | None) -> str:
    if not model:
        return "🧪 meta-labeling 尚未訓練（夜間工作流還沒跑，或 data/alpha/meta.json 缺）"
    o = model.get("oos") or {}
    kind_lab = {"logit": "邏輯迴歸", "gbm": "LightGBM"}.get(model.get("kind", "logit"), model.get("kind"))
    lines = [f"🧪 *交易層 meta-labeling*（{model.get('as_of')}｜樣本 {model.get('n')}｜基準勝率 {_f(model.get('base_rate'))}｜模型 {kind_lab}"
             + (f"（{model.get('n_trees')} 棵）" if model.get("kind") == "gbm" else "") + "）",
             f"OOS（purged walk-forward {len([x for x in o.get('folds', []) if not x.get('skipped')])} 折、embargo {EMBARGO_DAYS} 日）："
             f"n {o.get('n_oos', 0)}｜AUC {_f(o.get('auc'), 3)}｜log-loss {_f(o.get('logloss'), 3)} vs 基準 {_f(o.get('logloss_base'), 3)}",
             f"尺寸化測試：等額 Sharpe {_f(o.get('equal_sharpe'), 3)} → 勝率尺寸化 {_f(o.get('sized_sharpe'), 3)}｜"
             f"均 R {_f(o.get('equal_mean_r'), 3)} → {_f(o.get('sized_mean_r'), 3)}｜跳過率 {_f((o.get('skip_rate') or 0) * 100, 0)}%"]
    if o.get("calib"):
        lines.append("校準（預測勝率五分位 → 實際）：" + "、".join(f"{c['p_mean']:.2f}→{c['hit']:.2f}" for c in o["calib"]))
    cands = model.get("candidates") or {}
    if len(cands) >= 2:
        lines.append("候選對照：" + "｜".join(
            f"{'邏輯迴歸' if k == 'logit' else 'LightGBM'} AUC {_f(v.get('auc'), 3)}／log-loss {_f(v.get('logloss'), 3)}"
            f"／{'過' if v.get('gate_passed') else '未過'}" for k, v in cands.items()))
    if model.get("kind") == "logit" and model.get("coef"):
        pairs = sorted(zip(model["features"], model["coef"]), key=lambda kv: -abs(kv[1]))[:5]
        lines.append("最大係數（標準化）：" + "、".join(f"{k.replace('_', '·')} {v:+.2f}" for k, v in pairs))
    if model.get("by_mech"):
        lines.append("出場機制：" + "、".join(f"{k.replace('_', '·')} {v['n']} 筆 勝 {v['win_rate']:.0%} 均 {v['avg_r']:+.2f}R"
                                          for k, v in model["by_mech"].items()))
    lines.append("閘門：" + ("✅ 通過（`/set meta_enabled on` 讓部位吃勝率倍數）" if model.get("gate_passed")
                          else f"➖ 未通過（{model.get('reason')}）→ 線上不用"))
    lines.append("_標籤＝引擎規則單筆出場模擬（次日開盤進、收盤觸發出、含成本）；特徵訓練與線上同函數。非投資建議_")
    return "\n".join(lines)


# ── 自我測試（合成：植入「延伸度高 → 勝率低」）────────────────────────────────

if __name__ == "__main__":
    import tempfile
    import alpha_spine as asp
    rng = np.random.default_rng(9)
    # 1) simulate_exit：停損 / 追蹤 / 時間 / 截斷
    cl = np.array([100, 100, 98, 96, 95.5, 97, 99, 100], dtype=float)
    op = cl * 1.0
    lab = simulate_exit(cl, op, 0, rps=3.0)                     # entry=open[1]=100，停損 97 → 第 3 根 96 觸發
    assert lab["mech"] == "stop_loss" and lab["exit"] == 96 and lab["r_mult"] < -1.0, lab
    cl2 = np.array([100, 100, 102, 104, 106, 104, 100, 99, 98], dtype=float)
    lab2 = simulate_exit(cl2, cl2, 0, rps=2.0)                  # +3R 峰 106 → 收緊 5% → 100.7 → 第 6 根 100 觸發（地板 100）
    assert lab2["mech"] == "trailing_stop" and lab2["exit"] == 100 and abs(lab2["r_mult"] - (0 - 2 * 0.0005 * 100 / 2)) < 1e-9, lab2
    cl3 = np.array([100.0] * 60)
    lab3 = simulate_exit(cl3, cl3, 0, rps=2.0, cfg={"max_days": 45})
    assert lab3["mech"] == "time" and lab3["days"] == 45 and not lab3["truncated"]
    lab4 = simulate_exit(np.array([100.0] * 10), None, 0, rps=2.0)
    assert lab4["truncated"]
    # #55：時間柵欄那天收盤缺值 → 以區間內最後有效收盤結算（不得 NaN）；進場後全缺 → None
    cl8 = np.array([100.0] * 60); cl8[46] = np.nan
    lab8 = simulate_exit(cl8, cl8, 0, rps=2.0, cfg={"max_days": 45})
    assert lab8["mech"] == "data_end" and np.isfinite(lab8["r_mult"]) and lab8["exit"] == 100.0 and lab8["days"] == 44, lab8
    cl9 = np.array([100.0] + [np.nan] * 59); op9 = np.array([100.0] * 60)          # 開盤有價、進場後收盤全缺
    assert simulate_exit(cl9, op9, 0, rps=2.0) is None
    cl10 = np.array([100.0] * 30 + [np.nan] * 30)                                     # 尾段下市：最後有效價結算
    lab10 = simulate_exit(cl10, cl10, 0, rps=2.0, cfg={"max_days": 45, "dead_money_days": 99})
    assert lab10 is not None and np.isfinite(lab10["r_mult"])
    # OOS 統計 NaN 安全：混入 NaN 標籤/特徵的樣本被丟棄並計數
    assert _sharpe_like(np.array([1.0, np.nan, -0.5, 0.3])) is not None
    assert simulate_exit(cl, op, 6, 3.0) is None and simulate_exit(cl, op, 0, 0.0) is None
    # 訊號轉弱只在獲利中了結；死錢釋放（≥30 日、<+2%、評分 < 門檻）；虧損中轉弱不賣（等停損）
    cl5 = np.array([100, 100, 101, 102, 101.5] + [101.5] * 50, dtype=float)
    sc5 = np.array([0.6, 0.6, 0.6, 0.6, -0.3] + [0.1] * 50)
    lab5 = simulate_exit(cl5, cl5, 0, rps=2.0, score=sc5)
    assert lab5["mech"] == "signal_exit" and lab5["exit"] == 101.5 and lab5["days"] == 3, lab5
    cl6 = np.array([100.0] * 60); sc6 = np.array([0.6] * 3 + [0.1] * 57)
    lab6 = simulate_exit(cl6, cl6, 0, rps=2.0, score=sc6)
    assert lab6["mech"] == "dead_money" and lab6["days"] == 30, lab6
    lab6b = simulate_exit(cl6, cl6, 0, rps=2.0, score=np.array([0.6] * 60))
    assert lab6b["mech"] == "time"                                                      # 評分仍強 → 不算死錢
    cl7 = np.array([100, 100, 99, 98.5, 98.4] + [98.4] * 50, dtype=float)
    lab7 = simulate_exit(cl7, cl7, 0, rps=2.0, score=sc5)
    assert lab7["mech"] in ("dead_money", "time")                                       # 虧損中轉弱不賣
    print("✅ 1 出場模擬（停損/追蹤地板/訊號了結/死錢/時間柵欄/截斷）")

    # 2) signal_days 去重與濾網
    sc = np.array([0.6, 0.6, 0.6, 0.4, 0.6] + [0.6] * 12)
    ex = np.zeros(len(sc)); r5 = np.zeros(len(sc))
    d = [f"2026-01-{i + 1:02d}" for i in range(len(sc))]
    sd = signal_days(sc, ex, r5, d, {"resample_days": 10})
    assert sd == [0, 10], sd                                      # 同檔兩筆至少隔 10 日（門檻閃爍 0.6/0.4 不灌水）
    flick = np.array([0.6, 0.4] * 10)
    assert signal_days(flick, np.zeros(20), np.zeros(20), [f"2026-02-{i + 1:02d}" for i in range(20)]) == [0, 10]
    ex2 = ex.copy(); ex2[0] = 3.0; r52 = r5.copy(); r52[0] = 0.10
    assert signal_days(sc, ex2, r52, d)[0] == 1                   # 延伸且急拉 → 首日被濾，次日取
    r53 = r5.copy(); r53[0] = 0.01
    assert signal_days(sc, ex2, r53, d)[0] == 0                   # 延伸但沒急拉 → 放行
    assert signal_days(sc, ex, r5, d, blackout_fn=lambda dd: dd == "2026-01-01")[0] == 1
    print("✅ 2 訊號時點去重／延伸濾網／靜默日")

    # 3) 特徵：訓練與線上同函數、缺值旗標、regime one-hot
    x = features({"ext_atr": 1.2, "ret_5d": 0.03, "score": 0.6, "vol_60": 0.02, "mom_12_1": 0.3, "ret_1m": 0.05},
                 {"regime": "risk_on", "breadth_pct": 0.6})
    assert len(x) == len(FEATURES) and x[5] == -0.05 and x[6] == 1.0 and x[7] == 0.0 and x[10] == 1.0 and x[12] == 1.0
    x2 = features({"quality": 0.4, "rev": None, "vol_60": float("nan")}, {"regime": "risk_off"})
    assert x2[9] == 0.4 and x2[10] == 0.0 and x2[12] == 1.0 and x2[7] == 1.0 and x2[3] == 0.0 and x2[8] == 0.5
    assert x[13] == 0.0 and x2[13] == 1.0 and features({"mom_12_1": float("nan")})[13] == 1.0   # mom 缺值旗標
    print("✅ 3 特徵函數（缺值旗標、regime、rev_1m 由 ret_1m 推）")

    # 4) 邏輯迴歸與 AUC：植入效應可學到；打亂標籤學不到
    n = 3000
    Xs = rng.normal(0, 1, (n, len(FEATURES)))
    logit = -1.5 * Xs[:, 0] + 0.8 * Xs[:, 3] + 0.2
    ys = (rng.uniform(0, 1, n) < _sigmoid(logit)).astype(float)
    m = fit_logistic(Xs, ys, 1.0)
    assert m["coef"][0] < -1.0 and m["coef"][3] > 0.5, m["coef"][:4]
    pp = np.array([predict_p(m, x_) for x_ in Xs])
    assert auc_score(ys, pp) > 0.75
    assert size_from_p(0.3) == 0.0 and size_from_p(0.65) == 1.25 and size_from_p(None) == 1.0 and 0.5 <= size_from_p(0.5) < 1.25
    sc_ = size_cfg_from_base(0.45); assert sc_["lo"] == 0.37 and sc_["hi"] == 0.57 and size_cfg_from_base(None) == SIZE
    assert not gate({"n_oos": 500, "auc": 0.6, "skip_rate": 0.95, "sized_sharpe": 0.1, "equal_sharpe": -0.5})[0]
    _nan = float("nan")
    for bad in ({"sized_sharpe": _nan, "equal_sharpe": _nan}, {"auc": _nan}, {"skip_rate": _nan}, {"sized_sharpe": float("inf")}):
        g_ = {"n_oos": 500, "auc": 0.6, "skip_rate": 0.1, "sized_sharpe": 0.2, "equal_sharpe": 0.1, **bad}
        assert not gate(g_)[0], g_                                                     # #58：非有限值不得放行
    assert gate({"n_oos": 500, "auc": 0.6, "skip_rate": 0.1, "sized_sharpe": 0.2, "equal_sharpe": 0.1})[0]
    print("✅ 4 邏輯迴歸／AUC／尺寸映射")

    # 4b) LightGBM 候選：精簡樹表的純 Python 推論與 lightgbm 原生預測逐位一致（有裝才測）
    if gbm_available():
        g = fit_gbm(Xs, ys, rounds=60, keep_booster=True)
        pg = np.array([predict_gbm(g, x_) for x_ in Xs])
        ref = g["_booster"].predict(Xs, num_iteration=60)
        assert np.max(np.abs(pg - ref)) < 1e-9, np.max(np.abs(pg - ref))
        assert auc_score(ys, pg) > 0.75 and g["n_trees"] == 60 and all(isinstance(t, list) for t in g["trees"])
        g2 = fit_gbm(Xs[:2400], ys[:2400], valid=(Xs[2400:], ys[2400:]), keep_booster=True)
        assert 1 <= g2["best_iter"] <= GBM_MAX_ROUNDS and abs(predict_gbm(g2, Xs[0]) - float(g2["_booster"].predict(Xs[:1], num_iteration=g2["best_iter"])[0])) < 1e-9
        xn = Xs[:1].copy(); xn[0, 0] = float("nan")
        assert abs(predict_gbm(g, xn[0]) - float(g["_booster"].predict(xn, num_iteration=60)[0])) < 1e-9   # 訓練無 NaN：NaN 視為 0.0（與 lightgbm 同）
        Xm = Xs.copy(); Xm[::7, 1] = float("nan")                                   # 訓練含 NaN → default_left 分支
        gm = fit_gbm(Xm, ys, rounds=40, keep_booster=True)
        assert max(abs(predict_gbm(gm, x_) - float(pp_)) for x_, pp_ in zip(Xm[:200], gm["_booster"].predict(Xm[:200], num_iteration=40))) < 1e-9
        print(f"✅ 4b LightGBM 純 Python 推論逐位一致（{g['n_trees']} 棵；early-stop {g2['best_iter']} 棵）")
    else:
        print("➖ 4b lightgbm 未安裝，略過 GBM 平價測試")

    # 5) 端到端：合成宇宙（40 檔 × 700 日）植入「品質高 → 漂移高」（PIT 基本面特徵），
    #    走 purged walk-forward → 閘門通過、quality 係數為正；打亂標籤 → 不過
    n_sym, n_day = 40, 700
    idx = pd.bdate_range("2023-11-01", periods=n_day)
    data, qual = {}, {}
    for k in range(n_sym):
        qk = float(rng.uniform(-1, 1)); qual[f"T{k:02d}"] = qk
        c = 100 * np.cumprod(1 + rng.normal(0.0006 + 0.0035 * qk, 0.02, n_day))
        data[f"T{k:02d}"] = pd.DataFrame({"Open": c * (1 + rng.normal(0, 0.002, n_day)), "High": c * 1.012, "Low": c * 0.988,
                                          "Close": c, "Volume": rng.integers(1e5, 1e6, n_day).astype(float)}, index=idx)
    frames = asp.wide_frames(data)
    frames["open"] = pd.DataFrame({k: v["Open"] for k, v in data.items()})
    import indicators as ind
    scores = pd.DataFrame({k: ind.composite_series(v["Close"], v["High"], v["Low"], v["Volume"]) for k, v in data.items()})
    br = asp.breadth_series(frames["close"])
    ctx = {str(dd.date()): {"regime": "risk_on", "breadth_pct": (float(b) if b == b else None)} for dd, b in br.items()}
    samples = build_samples(frames, scores, ctx, fund_lookup=lambda s_, d_: {"quality": qual[s_], "rev": None})
    assert len(samples) > 600, len(samples)
    assert all(len(s["x"]) == len(FEATURES) for s in samples) and all(s["days"] >= 1 for s in samples)
    model = train(samples, "2026-09-19", kinds=("logit",))          # 閘門斷言用邏輯迴歸（決定性）
    o = model["oos"]
    print(f"   樣本 {len(samples)}、基準勝率 {model['base_rate']:.2f}、OOS n {o['n_oos']}、AUC {o.get('auc')}、"
          f"等額 {o.get('equal_sharpe')} → 尺寸化 {o.get('sized_sharpe')}、跳過率 {o.get('skip_rate')}、閘門 {model['gate_passed']}（{model['reason']}）")
    assert o["n_oos"] >= 300 and o["auc"] is not None and o["auc"] > 0.55, o
    assert model["gate_passed"], model["reason"]
    assert model["coef"][FEATURES.index("quality")] > 0                      # 植入效應：品質係數為正
    assert model["size"]["lo"] < model["base_rate"] < model["size"]["hi"]     # 尺寸門檻相對基準勝率
    if gbm_available():
        m_both = train(samples, "2026-09-19")
        assert set(m_both["candidates"]) == {"logit", "gbm"} and m_both["kind"] in ("logit", "gbm")
        assert usable(m_both) or not m_both["gate_passed"]
        pb = predict_p(m_both, samples[0]["x"]); assert 0.0 <= pb <= 1.0
        print(f"   候選對照：{ {k: (round(v['auc'] or 0, 3), round(v['logloss'] or 0, 3)) for k, v in m_both['candidates'].items()} } → 選 {m_both['kind']}")
    dirty = samples + [dict(samples[0], r_mult=float("nan")), dict(samples[1], x=[float("nan")] * len(FEATURES))]
    m_dirty = train(dirty, "2026-09-19", kinds=("logit",))
    assert m_dirty["n_dropped"] == 2 and m_dirty["oos"]["n_dropped"] == 0 and m_dirty["oos"]["equal_sharpe"] is not None
    assert all(v["avg_r"] == v["avg_r"] for v in m_dirty["by_mech"].values())         # 無 NaN
    shuffled = [dict(s, win=bool(w), r_mult=(abs(s["r_mult"]) if w else -abs(s["r_mult"]))) for s, w in zip(samples, rng.permutation([s["win"] for s in samples]))]
    m_sh = train(shuffled, "2026-09-19", kinds=("logit",))
    assert not m_sh["gate_passed"], m_sh["reason"]
    # purge：各折訓練樣本數單調不減（擴張視窗），且至少 3 折有效
    fl = [f for f in o["folds"] if not f.get("skipped")]
    assert len(fl) >= 3 and all(fl[i]["n_train"] <= fl[i + 1]["n_train"] for i in range(len(fl) - 1))
    # 靜默日與延伸濾網在樣本層生效
    s_bo = build_samples(frames, scores, ctx, blackout_fn=lambda d_: True)
    assert s_bo == []
    # 6) 線上打分 + 存取 + 文字
    with tempfile.TemporaryDirectory() as td:
        p = save_model(model, Path(td) / "meta.json")
        back = load_model(p)
        assert usable(back) and back["features"] == list(FEATURES) and "_booster" not in back
        assert not usable({**back, "features": back["features"][:-1]})              # 特徵版本不符 → 不可用
        rows = score_rows(back, [{"ticker": "A", "ext_atr": 0.5, "ret_5d": 0.01, "score": 0.7, "vol_60": 0.02, "quality": -0.9},
                                 {"ticker": "B", "ext_atr": 0.5, "ret_5d": 0.01, "score": 0.7, "vol_60": 0.02, "quality": 0.9}], {"regime": "risk_on", "breadth_pct": 0.5})
        assert rows[0]["meta_p"] < rows[1]["meta_p"] and rows[1]["meta_mult"] >= rows[0]["meta_mult"]
        assert "ticker" in rows[0] and "meta_mult" in rows[0]
    txt = meta_text(model); assert "**" not in txt and "AUC" in txt
    print(txt)
    assert "尚未" in meta_text(None)
    print("\nmeta_label selftest OK ✅")
