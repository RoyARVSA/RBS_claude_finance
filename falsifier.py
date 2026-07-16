"""
falsifier.py — Hypothesis Falsifier：投資故事反駁引擎（Batch 1：核心測試）
RBS Finance Dashboard

⚠️ 本工具只能告訴你故事何時是假的，永遠不能告訴你它是真的。
「存活」只代表此樣本內未被拒絕；歷史上通過同樣測試後死掉的故事，
沒有籃子、沒有資料、不在任何資料庫裡——基準率缺席，通過率就沒有意義。

輸入一個假設規格（spec）：
  {"statement": "AI 資本支出增加 → 記憶體供應商未來六個月跑贏大盤",
   "basket": ["MU","WDC","STX","SNDK"], "benchmark": "SPY",
   "horizon_days": 126, "direction": "outperform"}
對它跑反駁測試（Batch 1 實作 5 項）：
  T1 漂移顯著性（circular block bootstrap——重疊視窗的正確處理：
     六個月持有逐日進場≠上千獨立樣本，有效樣本 ≈ n/視窗長）
  T2 日期穩健性（rolling 進場勝率 + 逐年分解——抓「只有特定起訖日成立」）
  T3 晚進場測試（故事已見報——trailing 超額前 1/3 的日子——才進場還成立嗎）
  T4 交易成本後存活
  T5 事件日進場（使用者提供公告日時的 CAR-lite：+lag 進場超額）
統計方法依正統文獻：block bootstrap（Politis-Romano）、事後漂移檢定（CAR/PEAD
精神）、AHM 研究日誌紀律。Batch 2/3 加 regime 切分、動能對照、跨市場、DSR 帳本。

已知且無法修復的限制（報告中強制揭露）：
- 籃子成分是「今天的名單」——回填/存活者偏誤無法用歷史數據消除
- 下市公司在免費資料源缺席——墳墓對照組天然缺席
純邏輯離線可測；fetch 層走 yfinance。教育用途，非投資建議。
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

DEFAULT_HORIZON = 126        # 六個月交易日
DEFAULT_COST_RT = 0.001      # 來回成本（與 backtest.py 一致）
BOOT_N = 1000
BOOT_BLOCK = 21              # 月級區塊（處理報酬自相關）
MIN_OBS = 252                # 至少一年日資料才給判定


# ── 純邏輯：籃子與超額報酬 ────────────────────────────────────────────────────

def basket_series(closes: dict[str, pd.Series]) -> pd.Series | None:
    """等權籃子指數（每日再平衡＝正規化後平均）。<2 檔回 None。"""
    ser = [s.dropna() for s in closes.values() if s is not None and len(s.dropna()) > 1]
    if len(ser) < 2:
        return None
    df = pd.concat(ser, axis=1).dropna()
    if len(df) < 2:
        return None
    norm = df / df.iloc[0]
    return norm.mean(axis=1)


def _align(basket: pd.Series, bench: pd.Series) -> pd.DataFrame:
    df = pd.concat({"b": basket, "m": bench}, axis=1).dropna()
    return df


def daily_rel_log(basket: pd.Series, bench: pd.Series) -> pd.Series:
    """每日相對對數報酬 d_t = Δln(basket) − Δln(bench)。"""
    df = _align(basket, bench)
    return (np.log(df["b"]).diff() - np.log(df["m"]).diff()).dropna()


def rolling_excess(basket: pd.Series, bench: pd.Series,
                   horizon: int = DEFAULT_HORIZON, lag: int = 1) -> pd.Series:
    """
    逐日訊號、+lag 日收盤進場（無前視）、持有 horizon 交易日的
    籃子−基準 簡單超額報酬序列（索引=進場日）。**視窗高度重疊**——
    顯著性判定必須走 block bootstrap（T1），不可對本序列直接 t 檢定。
    """
    df = _align(basket, bench)
    b, m = df["b"].to_numpy(float), df["m"].to_numpy(float)
    n = len(df)
    if n < lag + horizon + 1:      # 至少要能開出一個完整視窗
        return pd.Series(dtype=float)
    e0 = lag                                  # 第一個進場位置
    last = n - horizon                        # 進場位置上限（exclusive）
    idx = np.arange(e0, last)
    exc = (b[idx + horizon] / b[idx]) - (m[idx + horizon] / m[idx])
    return pd.Series(exc, index=df.index[idx])


# ── T1 漂移顯著性（circular block bootstrap）─────────────────────────────────

def block_bootstrap_test(basket: pd.Series, bench: pd.Series,
                         horizon: int = DEFAULT_HORIZON,
                         direction: str = "outperform",
                         n_boot: int = BOOT_N, block: int = BOOT_BLOCK,
                         seed: int = 7) -> dict:
    """
    H0：籃子相對基準無漂移。統計量 = 每日相對對數報酬均值（重疊視窗的
    超額報酬本質上就是它的視窗和）。circular block bootstrap 對「去均值後」
    的序列重抽（保留自相關），單尾 p 值。有效樣本數誠實回報 ≈ n/block。
    """
    d = daily_rel_log(basket, bench).to_numpy(float)
    n = len(d)
    if n < MIN_OBS:
        return {"test": "T1 漂移顯著性", "survived": None, "p_value": None,
                "detail": f"資料不足（{n} 日 < {MIN_OBS}）"}
    obs = float(d.mean())
    dd = d - obs                              # 去均值 → 建 H0 分佈
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(n / block)
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    # circular blocks：每列串起 n_blocks 段長 block 的環狀切片，截到 n
    offs = np.arange(block)
    pos = (starts[:, :, None] + offs[None, None, :]) % n     # (B, nb, block)
    samples = dd[pos].reshape(n_boot, -1)[:, :n]
    boot_means = samples.mean(axis=1)
    if direction == "underperform":
        p = float((boot_means <= obs).mean())
    else:
        p = float((boot_means >= obs).mean())
    p = max(p, 1.0 / n_boot)
    eff_n = n / block
    ann = obs * 252
    survived = bool(p < 0.05)
    detail = (f"日均相對報酬 {obs * 1e4:+.1f}bp（年化 {ann:+.1%}），"
              f"單尾 p={p:.3f}，有效樣本 ≈{eff_n:.0f} 區塊"
              f"（{n} 日 ÷ {block} 日區塊——別被重疊視窗的假樣本數騙了）")
    if not survived:
        detail = "無顯著漂移＝故事沒有價格證據——" + detail
    return {"test": "T1 漂移顯著性（block bootstrap）", "survived": survived,
            "p_value": round(p, 4), "detail": detail}


# ── T2 日期穩健性 ─────────────────────────────────────────────────────────────

def date_robustness(excess: pd.Series, direction: str = "outperform") -> dict:
    """rolling 進場勝率 + 逐年均值——抓「只有特定起訖日期成立」。"""
    if len(excess) < 60:
        return {"test": "T2 日期穩健性", "survived": None,
                "detail": f"視窗數不足（{len(excess)}）"}
    sign = -1.0 if direction == "underperform" else 1.0
    ex = excess * sign
    win = float((ex > 0).mean())
    yearly = ex.groupby(ex.index.year).mean()
    neg_years = [str(y) for y, v in yearly.items() if v < 0]
    frac_neg = len(neg_years) / len(yearly)
    survived = bool(win >= 0.5 and frac_neg <= 0.4)
    yr_s = "、".join(f"{y}:{v:+.1%}" for y, v in yearly.items())
    detail = f"進場日勝率 {win:.0%}；逐年均超額 {yr_s}"
    if neg_years:
        detail += f"——{'、'.join(neg_years)} 年為負：故事有明顯的時段依賴"
    return {"test": "T2 日期穩健性（rolling 進場）", "survived": survived,
            "detail": detail}


# ── T3 晚進場測試（故事已見報才進場）─────────────────────────────────────────

def late_entry_test(basket: pd.Series, bench: pd.Series,
                    horizon: int = DEFAULT_HORIZON, trailing: int = 63,
                    direction: str = "outperform") -> dict:
    """
    「等到故事已經跑出來（trailing 3 個月超額位於前 1/3 的日子）才進場」
    的前瞻超額 vs 全期平均——顯著縮水甚至轉負 ⇒ 你看到報導時已太晚。
    （無事件日資料時對「公告後才買是否太晚」的價格空間近似。）
    """
    df = _align(basket, bench)
    if len(df) < trailing + horizon + 60:
        return {"test": "T3 晚進場", "survived": None, "detail": "資料不足"}
    b, m = df["b"], df["m"]
    trail = (b / b.shift(trailing) - 1) - (m / m.shift(trailing) - 1)
    fwd = (b.shift(-horizon) / b - 1) - (m.shift(-horizon) / m - 1)
    ok = pd.concat({"t": trail, "f": fwd}, axis=1).dropna()
    if len(ok) < 60:
        return {"test": "T3 晚進場", "survived": None, "detail": "重疊樣本不足"}
    sign = -1.0 if direction == "underperform" else 1.0
    hot = ok[ok["t"] * sign >= ok["t"].mul(sign).quantile(2 / 3)]
    all_mean = float(ok["f"].mean() * sign)
    hot_mean = float(hot["f"].mean() * sign)
    decay = (hot_mean - all_mean)
    survived = bool(hot_mean > 0)
    detail = (f"任何時點進場平均前瞻超額 {all_mean:+.1%}；"
              f"故事已跑出來後才進場 {hot_mean:+.1%}"
              f"（衰減 {decay:+.1%}，樣本 {len(hot)} 個重疊日）")
    if hot_mean <= 0 < all_mean:
        detail += "——動能兌現後進場的人接的是別人的獲利了結"
    return {"test": "T3 晚進場（故事見報後）", "survived": survived, "detail": detail}


# ── T4 交易成本後存活 ─────────────────────────────────────────────────────────

def cost_survival(excess: pd.Series, cost_rt: float = DEFAULT_COST_RT,
                  direction: str = "outperform") -> dict:
    """單次進出來回成本後，平均視窗超額是否仍為正。"""
    if not len(excess):
        return {"test": "T4 成本後存活", "survived": None, "detail": "無視窗"}
    sign = -1.0 if direction == "underperform" else 1.0
    gross = float(excess.mean() * sign)
    net = gross - cost_rt
    return {"test": "T4 交易成本後存活", "survived": bool(net > 0),
            "detail": (f"視窗平均超額 毛 {gross:+.2%} → 扣 {cost_rt:.2%} 來回成本後 "
                       f"淨 {net:+.2%}")}


# ── T5 事件日進場（CAR-lite，使用者提供公告日時）─────────────────────────────

def event_entry_excess(basket: pd.Series, bench: pd.Series,
                       events: list[str], horizon: int = DEFAULT_HORIZON,
                       lag: int = 1, direction: str = "outperform") -> dict:
    """
    事件日 +lag 收盤進場、持有 horizon 的平均超額（market-adjusted CAR 精神）。
    事件 <4 個只報數字不判定（橫斷面檢定力不足——誠實優先）。
    """
    df = _align(basket, bench)
    sign = -1.0 if direction == "underperform" else 1.0
    vals = []
    for ev in events or []:
        try:
            ts = pd.Timestamp(ev)
        except ValueError:
            continue
        if len(df) and ts < df.index[0]:       # 早於樣本起點 → 略過（勿灌假 CAR）
            continue
        pos = df.index.searchsorted(ts)        # 事件日或其後第一個交易日
        e = pos + lag
        x = e + horizon
        if x >= len(df):
            continue
        b, m = df["b"], df["m"]
        vals.append(((b.iloc[x] / b.iloc[e] - 1) - (m.iloc[x] / m.iloc[e] - 1)) * sign)
    if not vals:
        return {"test": "T5 事件日進場", "survived": None,
                "detail": "未提供事件日（spec 加 events:[YYYY-MM-DD,…] 可啟用）"}
    mean_v = float(np.mean(vals))
    if len(vals) < 4:
        return {"test": "T5 事件日進場", "survived": None,
                "detail": (f"僅 {len(vals)} 個事件（<4，不足以判定）："
                           f"平均事後超額 {mean_v:+.1%}——當線索看，別當結論")}
    t_stat = mean_v / (float(np.std(vals, ddof=1)) / math.sqrt(len(vals)) + 1e-12)
    survived = bool(mean_v > 0 and t_stat > 1.0)
    return {"test": "T5 事件日進場（CAR-lite）", "survived": survived,
            "detail": (f"{len(vals)} 個事件、+{lag} 日進場：平均事後超額 {mean_v:+.1%}"
                       f"（t≈{t_stat:.1f}）——t<2 就別太興奮")}


# ── Batch 2：regime 切分 / 動能混淆 / 跨市場泛化 ─────────────────────────────

# 跨市場類比籃子（放這裡而非 stock_db，避免污染各市場分組的其他使用者）
ANALOG_BASKETS = {
    "HBM/記憶儲存": {"韓國記憶體": ["000660.KS", "005930.KS"],
                     "台灣記憶體": ["2408.TW", "2344.TW"]},
}


def regime_series(bench_close: pd.Series) -> pd.Series:
    """
    逐日市場狀態標籤（無前視：全用截至當日的 rolling 值）。
    規則與 scan_signals.market_regime 一致：
    價>MA50 且 22 日動量>0 → risk_on；價<MA50 且動量<-3% → risk_off；其餘 neutral。
    """
    c = bench_close.dropna()
    ma50 = c.rolling(50).mean()
    mom = c / c.shift(22) - 1
    lab = pd.Series("neutral", index=c.index)
    lab[(c > ma50) & (mom > 0)] = "risk_on"
    lab[(c < ma50) & (mom < -0.03)] = "risk_off"
    lab[ma50.isna() | mom.isna()] = None
    return lab.dropna()


def rate_cycle_series(fedfunds: pd.Series, window: int = 6,
                      thresh: float = 0.25) -> pd.Series:
    """FEDFUNDS 月序列 → 升息/降息/持平標籤（6 個月變化 ±0.25% 門檻）。"""
    f = fedfunds.dropna()
    chg = f - f.shift(window)
    lab = pd.Series("持平", index=f.index)
    lab[chg > thresh] = "升息"
    lab[chg < -thresh] = "降息"
    lab[chg.isna()] = None
    return lab.dropna()


def regime_split_test(basket: pd.Series, bench: pd.Series, labels: pd.Series,
                      horizon: int = DEFAULT_HORIZON,
                      direction: str = "outperform",
                      name: str = "T6 市場狀態切分") -> dict:
    """
    依「進場日」的狀態標籤分組看前瞻超額。story 若只在單一 regime 成立，
    等於押注 regime 延續——那是另一個故事。組樣本 <60 不判定該組。
    """
    excess = rolling_excess(basket, bench, horizon)
    if len(excess) < 120:
        return {"test": name, "survived": None, "detail": "視窗不足"}
    sign = -1.0 if direction == "underperform" else 1.0
    lab = labels.reindex(excess.index, method="ffill")
    df = pd.DataFrame({"e": excess * sign, "g": lab}).dropna()
    groups = {g: sub["e"] for g, sub in df.groupby("g") if len(sub) >= 60}
    if len(groups) < 2:
        return {"test": name, "survived": None,
                "detail": "有效狀態組 <2（樣本期間狀態太單一）"}
    means = {g: float(v.mean()) for g, v in groups.items()}
    seg = "；".join(f"{g}:{m:+.1%}（{len(groups[g])} 窗）" for g, m in means.items())
    neg = [g for g, m in means.items() if m < 0]
    if neg and any(m > 0 for m in means.values()):
        return {"test": name, "survived": False,
                "detail": f"{seg}——只在部分狀態成立（{'、'.join(neg)} 為負）：{name.split()[0]}"
                          f" 依賴 regime 延續，那是另一個賭注"}
    return {"test": name, "survived": bool(all(m > 0 for m in means.values())),
            "detail": seg}


def momentum_confound_test(basket: pd.Series, bench: pd.Series,
                           mom_close: pd.Series,
                           direction: str = "outperform") -> dict:
    """
    「是否其實只是動能因子？」——兩因子時間序列回歸：
    r_basket = α + β1·r_mkt + β2·r_mom + ε，比較加入動能代理（MTUM 或自建）
    前後的年化 α。α 縮水殆盡 ⇒ 故事只是動能的別名。（普通 OLS t 值，
    未做 Newey-West——已在輸出揭露；動能代理為近似，非官方 UMD 因子。）
    """
    df = pd.concat({"b": basket, "m": bench, "f": mom_close}, axis=1).dropna()
    if len(df) < MIN_OBS:
        return {"test": "T7 動能混淆", "survived": None, "detail": "資料不足"}
    r = np.log(df).diff().dropna()
    y = r["b"].to_numpy()
    n = len(y)
    X1 = np.column_stack([np.ones(n), r["m"].to_numpy()])
    X2 = np.column_stack([np.ones(n), r["m"].to_numpy(), r["f"].to_numpy()])

    def _ols_alpha(X):
        beta, res, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        dof = max(n - X.shape[1], 1)
        s2 = float(resid @ resid) / dof
        cov = s2 * np.linalg.inv(X.T @ X)
        return float(beta[0]), float(beta[0] / math.sqrt(max(cov[0, 0], 1e-18)))

    a1, t1 = _ols_alpha(X1)
    a2, t2 = _ols_alpha(X2)
    sign = -1.0 if direction == "underperform" else 1.0
    a1_ann, a2_ann = a1 * 252 * sign, a2 * 252 * sign
    shrink = (1 - a2_ann / a1_ann) if a1_ann > 0 else None
    survived = bool(a2_ann > 0 and t2 * sign > 1.0)
    detail = (f"單因子 α {a1_ann:+.1%}/年（t={t1 * sign:.1f}）→ "
              f"加動能因子後 α {a2_ann:+.1%}/年（t={t2 * sign:.1f}）")
    if shrink is not None and shrink > 0.6:
        detail += f"——α 縮水 {shrink:.0%}：大部分只是動能的別名"
    detail += "（OLS 未修自相關、動能代理非官方 UMD——保守解讀 t 值）"
    return {"test": "T7 動能混淆（兩因子回歸）", "survived": survived, "detail": detail}


def generalization_tests(alts: list[tuple[str, dict, pd.Series]],
                         horizon: int = DEFAULT_HORIZON,
                         direction: str = "outperform") -> list[dict]:
    """
    跨市場/類比籃子泛化：對每個 (標籤, closes, bench) 跑 T1。
    同一機制若只在原市場成立，故事更可能是「選出來的」而非「機制性的」。
    """
    out = []
    for label, closes, bench in alts:
        bk = basket_series(closes)
        if bk is None or bench is None or bench.dropna().empty:
            out.append({"test": f"T8 泛化：{label}", "survived": None,
                        "detail": "價格資料不足"})
            continue
        r = block_bootstrap_test(bk, bench, horizon, direction)
        out.append({"test": f"T8 泛化：{label}", "survived": r["survived"],
                    "p_value": r.get("p_value"), "detail": r["detail"]})
    return out


# ── Batch 3：DSR 多重假設檢定帳本（Bailey & López de Prado 2014）─────────────

LEDGER_CAP = 50
EULER_GAMMA = 0.5772156649


def sharpe_daily(basket: pd.Series, bench: pd.Series) -> tuple[float, int, float, float]:
    """相對日報酬的 (SR_daily, T, skew, kurt)——DSR 的輸入。"""
    d = daily_rel_log(basket, bench)
    T = len(d)
    if T < 60 or float(d.std(ddof=1)) == 0:
        return 0.0, T, 0.0, 3.0
    sr = float(d.mean() / d.std(ddof=1))
    return sr, T, float(d.skew()), float(d.kurt() + 3.0)   # pandas kurt 是超額峰態


def deflated_sharpe(sr: float, T: int, n_trials: int,
                    trial_srs: list[float] | None = None,
                    skew: float = 0.0, kurt: float = 3.0) -> dict:
    """
    DSR = PSR(SR*)：在 N 次「零技巧」嘗試下，期望最大 SR 是多少——你的 SR
    要贏過那個幸運兒才算數。N 恆被低估（腦中試過的不進帳本）⇒ DSR 恆偏樂觀，
    輸出永遠掛此警語。trial_srs 給了就用其變異數，否則用 SR 估計誤差近似。
    """
    from scipy.stats import norm
    if T < 30:
        return {"dsr": None, "sr_star": None, "note": "樣本太短"}
    if n_trials <= 1:
        sr_star = 0.0
    else:
        if trial_srs and len(trial_srs) >= 2:
            v = float(np.var(trial_srs, ddof=1))
        else:
            v = (1 + 0.5 * sr * sr) / max(T, 2)          # SR 估計變異近似
        v = max(v, 1e-12)
        e_inv = 1.0 / (n_trials * math.e)
        sr_star = math.sqrt(v) * ((1 - EULER_GAMMA) * norm.ppf(1 - 1 / n_trials)
                                  + EULER_GAMMA * norm.ppf(1 - e_inv))
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr * sr
    if denom <= 0:
        return {"dsr": None, "sr_star": round(sr_star, 4), "note": "高階動差異常"}
    z = (sr - sr_star) * math.sqrt(T - 1) / math.sqrt(denom)
    return {"dsr": round(float(norm.cdf(z)), 4), "sr_star": round(sr_star, 4),
            "note": None}


def ledger_add(state: dict, spec: dict, out: dict,
               sr_t: tuple | None = None) -> dict:
    """把這次嘗試記進假設帳本（state['hypotheses']）。回本次紀錄。"""
    led = state.setdefault("hypotheses", {"entries": [], "extra_trials": 0})
    n_ref = sum(1 for r in out.get("results", []) if r.get("survived") is False)
    rec = {"date": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"),
           "statement": str(spec.get("statement", ""))[:150],
           "basket": list(spec.get("basket", []))[:15],
           "n_refuted": n_ref,
           "sr_daily": round(sr_t[0], 4) if sr_t else None,
           "T": sr_t[1] if sr_t else None}
    led["entries"] = (led["entries"] + [rec])[-LEDGER_CAP:]
    return rec


def ledger_dsr_line(state: dict, sr_t: tuple | None) -> str:
    """帳本狀態 + 這個假設的 DSR 一行（含誠實警語）。"""
    led = state.get("hypotheses") or {"entries": [], "extra_trials": 0}
    n = len(led["entries"]) + int(led.get("extra_trials", 0))
    if not sr_t or n == 0:
        return f"🧾 帳本：這是進本系統的第 {max(n, 1)} 個假設"
    trial_srs = [e["sr_daily"] for e in led["entries"] if e.get("sr_daily") is not None]
    d = deflated_sharpe(sr_t[0], sr_t[1], max(n, 1), trial_srs, sr_t[2], sr_t[3])
    line = f"🧾 帳本：第 {n} 個假設（含自報 {led.get('extra_trials', 0)} 次場外嘗試）"
    if d["dsr"] is not None:
        verdict = "通過" if d["dsr"] > 0.95 else "未達 0.95 門檻"
        line += (f"｜DSR={d['dsr']:.2f}（{verdict}；已扣「{n} 次嘗試的幸運上限 "
                 f"SR*={d['sr_star']:.3f}」）")
    line += ("\n　帳本只數得到進系統的嘗試——看圖放棄的、腦中否決的都沒算，"
             "N 恆低估、DSR 恆偏樂觀。`/falsify trials +K` 自報場外次數。")
    return line


# ── Batch 3：LLM 故事 → 規格 規劃器（assistant_tools 白名單模式）──────────────

def build_spec_prompt(story: str) -> str:
    return (
        "你是量化研究設計師。把使用者的投資故事轉成可反駁的測試規格。"
        "只回傳 JSON（不要圍欄、不要解釋），格式：\n"
        '{"statement": "一句可否證的主張", "basket": ["TICKER", ...], '
        '"benchmark": "SPY", "horizon_days": 126, "direction": "outperform"}\n'
        "規則：basket 選 3-8 檔最直接受故事影響的股票（yfinance 代碼，"
        "台股加 .TW）；benchmark 選故事對照的大盤（美股 SPY、台股 0050.TW、"
        "半導體故事可用 SMH）；horizon_days 取故事聲稱的時間（月×21，"
        "未聲稱用 126）；direction：跑贏=outperform、跑輸=underperform。"
        f"\n\n投資故事：{story}")


def parse_spec(text: str) -> dict | None:
    """解析並驗證 LLM 產出的規格（白名單式：格式不對就拒絕，不猜）。"""
    import json as _json
    import re
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        j = _json.loads(m.group(0))
    except _json.JSONDecodeError:
        return None
    basket = [str(t).upper().strip() for t in (j.get("basket") or [])
              if re.fullmatch(r"[A-Z0-9\.\-]{1,12}", str(t).upper().strip())]
    basket = list(dict.fromkeys(basket))
    if not (2 <= len(basket) <= 15):
        return None
    events = [str(e) for e in (j.get("events") or [])
              if re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(e))][:24]
    bench = str(j.get("benchmark", "SPY")).upper().strip()
    if not re.fullmatch(r"[A-Z0-9\.\-\^]{1,12}", bench):
        bench = "SPY"
    try:
        horizon = int(j.get("horizon_days", DEFAULT_HORIZON))
    except (TypeError, ValueError):
        horizon = DEFAULT_HORIZON
    horizon = max(21, min(horizon, 504))
    direction = j.get("direction", "outperform")
    if direction not in ("outperform", "underperform"):
        direction = "outperform"
    stmt = str(j.get("statement", ""))[:300]
    if not stmt:
        return None
    out = {"statement": stmt, "basket": basket, "benchmark": bench,
           "horizon_days": horizon, "direction": direction}
    if events:
        out["events"] = events
    return out


def parse_manual_spec(args: list[str]) -> dict | None:
    """
    手動速記：`MU,WDC,STX [vs SMH] [126] [空] [ev:2024-01-25,2024-04-25] 故事…`。
    第一段必須是逗號分隔 ≥2 檔代碼。防誤路由（「AI,ML 都在漲」這種口語開頭）：
    若帶了故事文字但**沒有任何明確標記**（vs/天數/方向/ev:），視為口語 → 回 None
    交給 LLM 規劃器。
    """
    import re
    if not args:
        return None
    tks = [t.strip().upper() for t in args[0].split(",") if t.strip()]
    if len(tks) < 2 or not all(re.fullmatch(r"[A-Z0-9\.\-]{1,12}", t) for t in tks):
        return None
    bench, horizon, direction = "SPY", DEFAULT_HORIZON, "outperform"
    events: list[str] = []
    story_parts: list[str] = []
    explicit = False
    i = 1
    while i < len(args):
        a = args[i]
        if a.lower() == "vs" and i + 1 < len(args):
            bench = args[i + 1].upper()
            explicit = True
            i += 2
            continue
        if a.isdigit() and 21 <= int(a) <= 504:
            horizon = int(a)
            explicit = True
        elif a in ("空", "跑輸"):
            direction = "underperform"
            explicit = True
        elif a in ("多", "跑贏"):
            direction = "outperform"
            explicit = True
        elif a.lower().startswith("ev:"):
            events = [e for e in a[3:].split(",")
                      if re.fullmatch(r"\d{4}-\d{2}-\d{2}", e)][:24]
            explicit = True
        else:
            story_parts.append(a)
        i += 1
    if story_parts and not explicit:
        return None                     # 「AI,ML 都在漲」→ 口語，交給 LLM
    stmt = " ".join(story_parts) or f"{'、'.join(tks)} 未來 {horizon} 交易日"
    stmt += f"{'跑輸' if direction == 'underperform' else '跑贏'} {bench}"
    out = {"statement": stmt, "basket": tks, "benchmark": bench,
           "horizon_days": horizon, "direction": direction}
    if events:
        out["events"] = events
    return out


# ── 組裝與報告 ────────────────────────────────────────────────────────────────

def run_tests(closes: dict[str, pd.Series], bench_close: pd.Series,
              spec: dict, extras: dict | None = None) -> dict:
    """
    對已抓好的價格跑全部測試。extras（皆可選）：
      {"mom_close": Series, "rate_labels": Series,
       "alts": [(label, closes_dict, bench_series), ...]}
    回 {basket_n, n_windows, results:[...]}。
    """
    horizon = int(spec.get("horizon_days", DEFAULT_HORIZON))
    direction = spec.get("direction", "outperform")
    bk = basket_series(closes)
    if bk is None or bench_close is None or bench_close.dropna().empty:
        return {"basket_n": 0, "results": [], "error": "籃子或基準價格不足（需 ≥2 檔）"}
    excess = rolling_excess(bk, bench_close, horizon)
    results = [
        block_bootstrap_test(bk, bench_close, horizon, direction),
        date_robustness(excess, direction),
        late_entry_test(bk, bench_close, horizon, direction=direction),
        cost_survival(excess, spec.get("cost_rt", DEFAULT_COST_RT), direction),
        event_entry_excess(bk, bench_close, spec.get("events"), horizon,
                           direction=direction),
    ]
    ex = extras or {}
    results.append(regime_split_test(bk, bench_close,
                                     regime_series(bench_close), horizon, direction))
    if ex.get("rate_labels") is not None and len(ex["rate_labels"]):
        results.append(regime_split_test(bk, bench_close, ex["rate_labels"],
                                         horizon, direction,
                                         name="T6b 利率週期切分"))
    if ex.get("mom_close") is not None:
        results.append(momentum_confound_test(bk, bench_close,
                                              ex["mom_close"], direction))
    if ex.get("alts"):
        results.extend(generalization_tests(ex["alts"], horizon, direction))
    return {"basket_n": len([s for s in closes.values()
                             if s is not None and len(s.dropna()) > 1]),
            "n_windows": len(excess), "results": results}


HEADER_WARNING = "⚠️ 本工具只能告訴你故事何時是假的，永遠不能告訴你它是真的。"


def falsify_text(spec: dict, out: dict) -> str:
    lines = ["🔨 *假設反駁報告*", HEADER_WARNING, "",
             f"📌 假設：「{spec.get('statement', '?')}」",
             f"籃子：{'、'.join(spec.get('basket', []))}（{out.get('basket_n', 0)} 檔）"
             f" vs {spec.get('benchmark', '?')}｜持有 "
             f"{spec.get('horizon_days', DEFAULT_HORIZON)} 交易日"]
    if out.get("error"):
        lines.append(f"❌ {out['error']}")
        return "\n".join(lines)
    lines.append("")
    n_surv = n_ref = 0
    for r in out["results"]:
        if r["survived"] is True:
            icon = "🟢 存活"
            n_surv += 1
        elif r["survived"] is False:
            icon = "🔴 被推翻"
            n_ref += 1
        else:
            icon = "⚪ 無法判定"
        lines.append(f"{icon}｜{r['test']}")
        lines.append(f"　{r['detail']}")
    lines.append("")
    if n_ref:
        lines.append(f"⚖️ {n_ref} 項被推翻——故事至少在這些角度站不住；先解釋得了它們再談加碼。")
    elif n_surv:
        lines.append(f"⚖️ {n_surv} 項存活、0 項推翻——注意：這些測試餵的是*同一段歷史*，"
                     "是一份證據拍了幾個角度，不是幾份獨立證據。")
    lines.append("_已知限制：籃子成分是今天的名單（回填偏誤）；下市公司缺席（存活偏誤）；"
                 "免費日線數據。教育用途，非投資建議。_")
    return "\n".join(lines)


# ── 抓取層（需網路）───────────────────────────────────────────────────────────

def fetch_and_run(spec: dict, period: str = "5y") -> tuple[dict, str] | None:
    """抓價（籃子+基準+MTUM+類比籃子）→ 跑全部測試 → (out, report_text)。"""
    try:
        import os as _os

        from sector_scan import _batch_closes
        bench = spec.get("benchmark", "SPY")
        # 基準不可同時當籃子成員（會被 pop 走）；MTUM 在籃內就不當動能代理
        tks = [t for t in dict.fromkeys(spec.get("basket", [])) if t != bench]

        # 類比籃子：spec 自帶優先，否則查 ANALOG_BASKETS（以 theme 鍵）
        alt_map: dict = dict(spec.get("alt_baskets") or {})
        theme = spec.get("theme")
        if not alt_map and theme and theme in ANALOG_BASKETS:
            alt_map = ANALOG_BASKETS[theme]
        alt_tks = [t for v in alt_map.values() for t in v]

        closes = _batch_closes(tks + [bench, "MTUM"] + alt_tks, period,
                               min_len=MIN_OBS)
        bench_s = closes.pop(bench, None)
        mom_s = closes.get("MTUM") if "MTUM" in tks else closes.pop("MTUM", None)
        extras: dict = {"mom_close": None if "MTUM" in tks else mom_s}
        alts = []
        for label, atks in alt_map.items():
            ac = {t: closes.pop(t) for t in atks if t in closes}
            if len(ac) >= 2 and bench_s is not None:
                # 類比市場仍以原基準衡量「跑贏大盤」主張的機制泛化
                alts.append((label, ac, bench_s))
        if alts:
            extras["alts"] = alts
        try:            # 利率週期（有 FRED key 才做）
            _fk = _os.environ.get("FRED_API_KEY", "")
            if _fk:
                from macro import _fred_get
                obs = _fred_get("FEDFUNDS", _fk, limit=600)
                if obs:
                    ff = pd.Series({pd.Timestamp(d): float(v) for d, v in obs
                                    if v is not None}).sort_index()
                    extras["rate_labels"] = rate_cycle_series(ff)
        except Exception:
            pass
        out = run_tests(closes, bench_s, spec, extras)
        sr_t = None
        bk = basket_series(closes)
        if bk is not None and bench_s is not None:
            sr_t = sharpe_daily(bk, bench_s)
        return out, falsify_text(spec, out), sr_t
    except Exception:
        return None


# ── CLI 自我測試（離線純邏輯，工程化合成資料）─────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 1500                                              # ~6 年日資料
    idx = pd.bdate_range("2020-06-01", periods=n)
    m_ret = rng.normal(0.0003, 0.01, n)
    bench = pd.Series(100 * np.cumprod(1 + m_ret), index=idx)

    # 情境 A：真有優勢（每日 +6bp 相對漂移）→ T1 應存活、p 小
    edge = 0.0006
    bA = pd.Series(100 * np.cumprod(1 + m_ret + edge + rng.normal(0, 0.004, n)), index=idx)
    closesA = {"X1": bA * 1.0, "X2": bA * rng.uniform(0.9, 1.1)}
    bkA = basket_series(closesA)
    assert bkA is not None and abs(bkA.iloc[0] - 1.0) < 1e-9

    t1 = block_bootstrap_test(bkA, bench, 126)
    assert t1["survived"] is True and t1["p_value"] < 0.05, t1

    # 情境 B：純噪音（無漂移）→ T1 不應宣稱顯著
    bB = pd.Series(100 * np.cumprod(1 + m_ret + rng.normal(0, 0.004, n)), index=idx)
    t1b = block_bootstrap_test(bB, bench, 126)
    assert t1b["p_value"] > 0.01, t1b            # 固定 seed 下不得誤判為極顯著
    # 方向反轉：underperform 主張對上正漂移 → 必不顯著
    t1c = block_bootstrap_test(bkA, bench, 126, direction="underperform")
    assert t1c["survived"] is False, t1c

    # rolling_excess 手算對照：造 5 點玩具序列
    tb = pd.Series([100, 110, 121, 133.1, 146.41],
                   index=pd.bdate_range("2026-01-05", periods=5))   # +10%/日
    tm = pd.Series([100.0] * 5, index=tb.index)                      # 基準不動
    ex = rolling_excess(tb, tm, horizon=2, lag=1)
    # 進場位置 1、2：b[3]/b[1]-1 = 21%、b[4]/b[2]-1 = 21%
    assert len(ex) == 2 and abs(ex.iloc[0] - 0.21) < 1e-9, ex
    # 無前視：改動最後一根之後的資料不影響先前視窗（玩具上等價於截斷不變）
    ex_trunc = rolling_excess(tb.iloc[:4], tm.iloc[:4], horizon=2, lag=1)
    assert abs(ex_trunc.iloc[0] - ex.iloc[0]) < 1e-12

    # T2 日期穩健性：優勢籃子勝率應高；再造「只有前半段成立」的故事 → 抓出時段依賴
    exA = rolling_excess(bkA, bench, 126)
    t2 = date_robustness(exA)
    assert t2["survived"] is True and "勝率" in t2["detail"]
    half = n // 2
    m2 = rng.normal(0.0003, 0.01, n)
    bench2 = pd.Series(100 * np.cumprod(1 + m2), index=idx)
    r_half = np.where(np.arange(n) < half, m2 + 0.0012, m2 - 0.0012)   # 泡沫：漲完全跌回
    bH = pd.Series(100 * np.cumprod(1 + r_half + rng.normal(0, 0.003, n)), index=idx)
    t2h = date_robustness(rolling_excess(bH, bench2, 126))
    assert t2h["survived"] is False and "為負" in t2h["detail"], t2h

    # T3 晚進場——兩種故事該給出相反判定：
    # (a) 持續型趨勢（bkA）：故事見報後進場仍賺 → 存活（動能持續）
    t3a = late_entry_test(bkA, bench, 126)
    assert t3a["survived"] is True, t3a
    # (b) 曇花一現：30 天暴走（短於 63 天偵測窗）後緩慢回吐——
    #     等 trailing 動能訊號亮起時行情已結束 → 晚進場必接刀
    r_spike = m2.copy()
    r_spike[100:130] += 0.02
    r_spike[130:] -= 0.0006
    bS = pd.Series(100 * np.cumprod(1 + r_spike + rng.normal(0, 0.003, n)), index=idx)
    t3s = late_entry_test(bS, bench2, 126)
    assert "衰減" in t3s["detail"]
    assert t3s["survived"] is False, t3s

    # T4 成本：毛 0.30% 視窗超額、成本 0.1% → 淨 0.2% 存活；毛 0.05% → 被推翻
    t4a = cost_survival(pd.Series([0.003] * 100))
    assert t4a["survived"] is True
    t4b = cost_survival(pd.Series([0.0005] * 100))
    assert t4b["survived"] is False

    # T5 事件日：4 個事件手工對照
    evs = [str(idx[i].date()) for i in (100, 400, 700, 1000)]
    t5 = event_entry_excess(bkA, bench, evs, horizon=126, lag=1)
    assert t5["survived"] is not None and "4 個事件" in t5["detail"], t5
    t5few = event_entry_excess(bkA, bench, evs[:2], horizon=126)
    assert t5few["survived"] is None and "不足以判定" in t5few["detail"]
    t5none = event_entry_excess(bkA, bench, None)
    assert t5none["survived"] is None

    # ── Batch 2 測試 ──────────────────────────────────────────────────
    # regime_series：牛市段多 risk_on、崩盤段出現 risk_off；無前視（截斷不變）
    crash = m2.copy()
    crash[800:1100] -= 0.004                     # 300 天熊市段（讓視窗留在同 regime）
    bench_cr = pd.Series(100 * np.cumprod(1 + crash), index=idx)
    labs = regime_series(bench_cr)
    assert set(labs.unique()) <= {"risk_on", "risk_off", "neutral"}
    assert (labs == "risk_off").sum() > 0
    labs_trunc = regime_series(bench_cr.iloc[:1000])
    common = labs_trunc.index.intersection(labs.index)
    assert (labs.loc[common] == labs_trunc.loc[common]).all()   # 標籤不吃未來資料

    # rate_cycle_series：升→平→降 玩具路徑
    ffidx = pd.date_range("2022-01-31", periods=30, freq="ME")
    ff = pd.Series(np.r_[np.linspace(0.25, 5.25, 12), np.full(8, 5.25),
                         np.linspace(5.25, 3.0, 10)], index=ffidx)
    rl = rate_cycle_series(ff)
    assert (rl == "升息").sum() > 0 and (rl == "降息").sum() > 0 and (rl == "持平").sum() > 0

    # regime_split_test：優勢只開在 risk_on 日 → 應被抓「只在部分狀態成立」
    lab_cr = regime_series(bench_cr).reindex(idx).ffill()
    r_cond = crash + np.where((lab_cr == "risk_on").to_numpy(), 0.0020, -0.0020)
    bC = pd.Series(100 * np.cumprod(1 + r_cond + rng.normal(0, 0.003, n)), index=idx)
    t6 = regime_split_test(bC, bench_cr, regime_series(bench_cr), 126)
    assert t6["survived"] is False and "只在部分狀態成立" in t6["detail"], t6

    # momentum_confound_test：
    # (a) 籃子＝動能因子的線性映射 → α 應縮水殆盡、不存活
    momf = pd.Series(100 * np.cumprod(1 + m_ret + 0.0007
                                      + rng.normal(0, 0.002, n)), index=idx)
    r_mom_driven = m_ret + 0.9 * (np.log(momf).diff().fillna(0).to_numpy()
                                  - m_ret) + rng.normal(0, 0.001, n)
    bM = pd.Series(100 * np.cumprod(1 + r_mom_driven), index=idx)
    t7a = momentum_confound_test(bM, bench, momf)
    assert t7a["survived"] is False and "縮水" in t7a["detail"], t7a
    # (b) 與動能因子無關的獨立優勢 → α 應保留、存活
    t7b = momentum_confound_test(bkA, bench, momf)
    assert t7b["survived"] is True, t7b

    # generalization_tests：一個真機制市場 + 一個純噪音市場
    alts = [("類比A-真", closesA, bench), ("類比B-噪音", {"N1": bB, "N2": bB * 1.01}, bench)]
    g = generalization_tests(alts, 126)
    assert len(g) == 2 and g[0]["survived"] is True and g[1]["survived"] is not True, g

    # run_tests with extras：結果數應含 T6 + T7 + 兩個 T8
    outX = run_tests(closesA, bench, {"basket": ["X1", "X2"], "horizon_days": 126},
                     extras={"mom_close": momf, "alts": alts})
    names = [r["test"] for r in outX["results"]]
    assert any("T6" in s for s in names) and any("T7" in s for s in names)
    assert sum("T8" in s for s in names) == 2, names

    # 端到端組裝 + 報告
    outA = run_tests(closesA, bench, {"statement": "測試故事", "basket": ["X1", "X2"],
                                      "benchmark": "SPY", "horizon_days": 126,
                                      "events": evs})
    txt = falsify_text({"statement": "測試故事", "basket": ["X1", "X2"],
                        "benchmark": "SPY", "horizon_days": 126}, outA)
    assert HEADER_WARNING in txt and "回填偏誤" in txt and "非投資建議" in txt
    assert ("同一段歷史" in txt) or ("被推翻" in txt)
    out_bad = run_tests({"X1": bA}, bench, {"basket": ["X1"]})
    assert out_bad.get("error")

    # ── Batch 3 測試 ──────────────────────────────────────────────────
    # DSR：N 越大扣越多（單調遞減）；N=1 不扣（=PSR）
    srA, TA, skA, kuA = sharpe_daily(bkA, bench)
    assert srA > 0 and TA > 1000
    d1 = deflated_sharpe(srA, TA, 1)
    d5 = deflated_sharpe(srA, TA, 5, trial_srs=[0.01, -0.02, 0.03, srA])
    d20 = deflated_sharpe(srA, TA, 20, trial_srs=[0.01, -0.02, 0.03, srA])
    assert d1["sr_star"] == 0.0 and d1["dsr"] >= d5["dsr"] >= d20["dsr"], (d1, d5, d20)
    assert deflated_sharpe(0.0, 10, 5)["dsr"] is None          # 樣本太短

    # 帳本：新增 + DSR 行 + cap
    stF: dict = {}
    for k in range(3):
        ledger_add(stF, {"statement": f"假設{k}", "basket": ["A", "B"]},
                   {"results": [{"survived": False}]}, sr_t=(0.02, 1400, 0.0, 3.0))
    assert len(stF["hypotheses"]["entries"]) == 3
    stF["hypotheses"]["extra_trials"] = 4
    line = ledger_dsr_line(stF, (srA, TA, skA, kuA))
    assert "第 7 個假設" in line and "N 恆低估" in line, line

    # LLM 規格解析：合法/非法
    good = parse_spec('前置廢話 {"statement":"記憶體六個月跑贏","basket":["MU","wdc","000660.KS"],'
                      '"benchmark":"smh","horizon_days":126,"direction":"outperform"} 後置')
    assert good and good["basket"] == ["MU", "WDC", "000660.KS"] and good["benchmark"] == "SMH"
    assert parse_spec('{"statement":"x","basket":["MU"]}') is None      # <2 檔
    assert parse_spec("不是 JSON") is None
    bad_h = parse_spec('{"statement":"x","basket":["MU","WDC"],"horizon_days":9999}')
    assert bad_h["horizon_days"] == 504                                  # 夾範圍

    # 手動速記
    ms = parse_manual_spec(["MU,WDC,STX", "vs", "SMH", "126", "AI", "記憶體", "故事"])
    assert ms and ms["benchmark"] == "SMH" and ms["horizon_days"] == 126
    assert "跑贏 SMH" in ms["statement"]
    ms2 = parse_manual_spec(["TSLA,RIVN", "空"])
    assert ms2["direction"] == "underperform" and "跑輸" in ms2["statement"]
    assert parse_manual_spec(["只有一檔"]) is None
    # 防誤路由（M2 迴歸）：口語開頭帶逗號、無任何明確標記 → None（交給 LLM）
    assert parse_manual_spec(["AI,ML", "都在漲", "所以買"]) is None
    assert parse_manual_spec(["MU,WDC"]) is not None            # 無故事無標記 → 合法
    # 事件日 token（M1 迴歸）：手動與 LLM 兩路都能帶 events
    ms3 = parse_manual_spec(["MU,WDC", "ev:2024-01-25,2024-04-25,bad", "126"])
    assert ms3 and ms3["events"] == ["2024-01-25", "2024-04-25"], ms3
    gs2 = parse_spec('{"statement":"x","basket":["MU","WDC"],'
                     '"events":["2024-01-25","junk","2024-04-25"]}')
    assert gs2["events"] == ["2024-01-25", "2024-04-25"]
    # T1 不顯著時的措辭（防「沒證據」被誤讀成「有反證」以外的東西）
    assert "無顯著漂移" in t1b["detail"]
    # 基準/MTUM 不得被 pop 出籃子（fetch 層防呆——純邏輯側驗證 spec 清洗）
    # （fetch_and_run 需網路，此處驗證 tks 過濾規則等價邏輯）
    _tks_clean = [t for t in dict.fromkeys(["MU", "SPY", "MU"]) if t != "SPY"]
    assert _tks_clean == ["MU"]

    print(f"✅ falsifier 離線自我測試通過（T1 優勢 p={t1['p_value']}、"
          f"噪音 p={t1b['p_value']}、DSR N=1:{d1['dsr']:.2f}→N=20:{d20['dsr']:.2f}）")
