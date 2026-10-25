"""
timing_research.py – 擇時研究第 1 步（研究計畫 v5 §2a；預先登記 #95；擇時帳 #1–#3）

問題：線上用的「市場氣象台」（market_weather 五因子）到底能不能幫我們決定何時進、何時退？
  T1 氣象台接進引擎：2008–2025 九個兩年窗、每窗從**當時** S&P 500 成分重抽 40 檔×5 組種子，
     A 氣象台／B 現行 MA50 退路／C 不用濾網／D 氣象台但不收緊追蹤，比「曝險調整後報酬」
  T2 同上，A − D：大盤偏空時「收緊追蹤停損」這個作用（抱多久）值不值得
  T3 氣象台當 SPY 持股比例（偏多 100%／中性 50%／偏空 0%）：主要看「擇時多賺」＝ vs 同樣持股、不預測的固定配置
規則寫死在 TIMING_PREREG（自測鎖住）；engine_research 的 `timing` 子指令不接受參數、只跑一次。

紀律：氣象台逐日重放只看「前 6 個月」資料（＝線上 period="6mo"）；成分不足 → MA50 退路且不更新遲滯；
t 收盤決定 → t+1 開盤成交；現金收 13 週國庫券（^IRX 貼現率 → 投資報酬率、用前一日）；成本 SPY 0.03%、個股 0.05%。
引擎與評分一律用程式預設參數（不讀個人設定）；FOMC 靜默日期表只有 2024 起 → 各組一律關閉。
教育用途，非投資建議。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

TIMING_PREREG = {
    "issue": 95,
    "windows": (("2008-01-01", "2009-12-31"), ("2010-01-01", "2011-12-31"), ("2012-01-01", "2013-12-31"),
                ("2014-01-01", "2015-12-31"), ("2016-01-01", "2017-12-31"),           # 證據窗 5 段
                ("2018-01-01", "2019-12-31"), ("2020-01-01", "2021-12-31"), ("2022-01-01", "2023-12-31"),
                ("2024-01-01", "2025-12-31")),                                        # 保留段 A 4 段
    "n_evidence_windows": 5,
    "k": 40, "seeds": (1, 2, 3, 4, 5),
    "warmup_days": 450,                       # 每窗起點往前抓的日曆天（指標暖機）
    "t1_min_evidence_wins": 4,                # 證據窗 5 段中 ≥ 4 段 A − B > 0
    "t1_holdout_max_bad": 1,                  # 保留段 4 段中「A − B < 0 且 A 回撤較大」最多 1 段
    "t3_evidence": ("2007-07-02", "2018-12-31"),
    "t3_holdout_start": "2019-01-01",
    "t3_map": {"risk_on": 1.0, "neutral": 0.5, "risk_off": 0.0},
    "spy_cost": 0.0003, "event_dd": 0.15, "event_max_days": 252,
    "placebo_min_shift": 126, "placebo_pct": 0.90,
    "event_help_frac": 0.75, "holdout_max_harm": 1,
}

WEATHER_SECTOR = None   # 由 market_weather.SECTOR_ETFS 提供（延遲匯入，避免離線測試時強制載入）


# ── 1. 基本序列（純函式）────────────────────────────────────────────────────

def irx_daily(irx: pd.Series) -> pd.Series:
    """^IRX（13 週國庫券貼現率，％）→ 每日現金報酬：投資報酬率 y＝365d／(360 − 91d)，÷ 252；用前一日的值。"""
    d = (irx.astype(float) / 100.0).clip(lower=0.0)
    y = 365.0 * d / (360.0 - 91.0 * d)
    return (y / 252.0).shift(1).fillna(0.0)


def open_returns(open_px: pd.Series) -> pd.Series:
    """r_t ＝ O_{t+1}／O_t − 1：t 日開盤持有到 t+1 日開盤（t−1 收盤決定的曝險，在 t 開盤成交後吃這段報酬）。"""
    return (open_px.shift(-1) / open_px - 1.0).fillna(0.0)


def strat_returns(expo: pd.Series, r_open: pd.Series, cash: pd.Series, cost: float) -> pd.Series:
    """曝險 e_t（t 日收盤決定）→ 下一段開盤到開盤的報酬：e_{t−1}·r_t +（1 − e_{t−1}）·cash_t − cost·|e_{t−1} − e_{t−2}|。"""
    e = expo.reindex(r_open.index).ffill().fillna(0.0)
    e1 = e.shift(1).fillna(0.0)
    turn = (e1 - e1.shift(1).fillna(0.0)).abs()
    return e1 * r_open + (1.0 - e1) * cash.reindex(r_open.index).fillna(0.0) - cost * turn


def const_mix_returns(w: float, r_open: pd.Series, cash: pd.Series, cost: float) -> pd.Series:
    """固定持股 w、每月第一個交易日再平衡（含成本）。月內權重隨漲跌漂移。"""
    idx = r_open.index
    cash = cash.reindex(idx).fillna(0.0)
    out = np.zeros(len(idx))
    ws = w
    prev_m = None
    for i, d in enumerate(idx):
        c = 0.0
        if prev_m is None or d.month != prev_m:
            c = cost * abs(ws - w) if prev_m is not None else cost * abs(w)
            ws = w
            prev_m = d.month
        r = ws * r_open.iloc[i] + (1 - ws) * cash.iloc[i]
        out[i] = r - c
        tot = 1 + r
        ws = ws * (1 + r_open.iloc[i]) / tot if tot > 0 else w
    return pd.Series(out, index=idx)


def perf(r: pd.Series) -> dict:
    """日報酬 → 年化報酬、波動、最大回撤、最後的錢（1 → x）。"""
    r = r.dropna()
    if not len(r):
        return {"cagr": None, "vol": None, "mdd": None, "final": None, "n": 0}
    w = (1 + r).cumprod()
    yrs = len(r) / 252.0
    cagr = float(w.iloc[-1] ** (1 / yrs) - 1) if yrs > 0 and w.iloc[-1] > 0 else None
    return {"cagr": cagr, "vol": float(r.std(ddof=1) * math.sqrt(252)) if len(r) > 2 else None,
            "mdd": float(-(w / w.cummax() - 1).min()), "final": float(w.iloc[-1]), "n": int(len(r))}


def placebo_percentile(expo: pd.Series, r_open: pd.Series, cash: pd.Series, cost: float, min_shift: int) -> dict:
    """日曝險序列循環平移 k＝min_shift…T−min_shift 的**每一個**值（報酬不動）→ 年化報酬分佈；真策略的百分位。"""
    e = expo.reindex(r_open.index).ffill().fillna(0.0).values
    T = len(e)
    real = perf(strat_returns(pd.Series(e, index=r_open.index), r_open, cash, cost))["cagr"]
    vals = []
    for k in range(int(min_shift), T - int(min_shift) + 1):
        vals.append(perf(strat_returns(pd.Series(np.roll(e, k), index=r_open.index), r_open, cash, cost))["cagr"])
    vals = [v for v in vals if v is not None]
    if real is None or not vals:
        return {"pct": None, "n": len(vals), "real": real}
    return {"pct": float(np.mean([v < real for v in vals])), "n": len(vals), "real": real,
            "p50": float(np.median(vals))}


def find_events(close: pd.Series, dd: float, max_days: int) -> list[dict]:
    """局部高點回撤 ≥ dd 的事件（互不重疊）：區間＝高點日 → 「收復高點日」與「谷底後 max_days 個交易日」取較早者；
    下一個事件從上一個區間結束後重新找高點。"""
    c = close.dropna()
    v, idx = c.values, c.index
    N = len(v)
    out, i = [], 0
    while i < N:
        peak = i
        t = i
        found = None
        while t < N:
            if v[t] > v[peak]:
                peak = t
            if v[t] <= v[peak] * (1 - dd):
                found = t
                break
            t += 1
        if found is None:
            break
        j, trough = found, found
        while j < N and v[j] < v[peak]:
            if v[j] < v[trough]:
                trough = j
            j += 1
        rec = j if j < N else None
        end = min(rec if rec is not None else N - 1, trough + int(max_days), N - 1)
        out.append({"peak": idx[peak], "trough": idx[trough], "end": idx[end],
                    "recovered": idx[rec] if rec is not None else None, "depth": float(v[trough] / v[peak] - 1)})
        i = end + 1
    return out


def _cum(r: pd.Series, a, b) -> float:
    s = r[(r.index >= a) & (r.index <= b)]
    return float((1 + s).prod() - 1) if len(s) else 0.0


def month_end_states(state: pd.Series) -> pd.Series:
    s = state.dropna()
    return s.groupby([s.index.year, s.index.month]).tail(1)


def prediction_diag(state: pd.Series, spy_close: pd.Series) -> dict:
    """月底狀態 → SPY 下個月（月底收盤到下個月底收盤）報酬：各狀態的平均與上漲機率；偏多／偏空方向命中率。"""
    me = month_end_states(state)
    cl = spy_close.reindex(me.index).astype(float)
    nxt = cl.shift(-1) / cl - 1
    df = pd.DataFrame({"st": me.values, "nxt": nxt.values}, index=me.index).dropna()
    out = {}
    for k in ("risk_on", "neutral", "risk_off"):
        x = df[df["st"] == k]["nxt"]
        out[k] = {"n": int(len(x)), "mean": float(x.mean()) if len(x) else None,
                  "p_up": float((x > 0).mean()) if len(x) else None}
    dirn = df[df["st"].isin(["risk_on", "risk_off"])]
    hits = ((dirn["st"] == "risk_on") & (dirn["nxt"] > 0)) | ((dirn["st"] == "risk_off") & (dirn["nxt"] < 0))
    out["hit"] = float(hits.mean()) if len(dirn) else None
    out["n_dir"] = int(len(dirn))
    return out


def block_bootstrap_pos(diff_m: pd.Series, block: int = 12, n: int = 2000, seed: int = 7) -> float | None:
    """月度差額的區塊自助法：平均 > 0 的比例（只報告）。"""
    x = diff_m.dropna().values
    if len(x) < block * 2:
        return None
    rng = np.random.default_rng(seed)
    nb = int(math.ceil(len(x) / block))
    pos = 0
    for _ in range(n):
        st = rng.integers(0, len(x) - block + 1, nb)
        smp = np.concatenate([x[s:s + block] for s in st])[:len(x)]
        pos += smp.mean() > 0
    return pos / n


# ── 2. T3：氣象台當 SPY 持股比例 ───────────────────────────────────────────

def t3_evaluate(state: pd.Series, ma50_state: pd.Series, spy_open: pd.Series, spy_close_tr: pd.Series,
                spy_close_raw: pd.Series, irx: pd.Series, prereg: dict = TIMING_PREREG) -> dict:
    """state／ma50_state：每日三態（t 收盤）；spy_open、spy_close_tr：含息還原；spy_close_raw：只還原拆股（200 日線用）。"""
    P = prereg
    cost = float(P["spy_cost"])
    a, b = (pd.Timestamp(x) for x in P["t3_evidence"])
    ho = pd.Timestamp(P["t3_holdout_start"])
    mp = P["t3_map"]
    r_open_all = open_returns(spy_open)
    cash_all = irx_daily(irx).reindex(r_open_all.index).ffill().fillna(0.0)
    e_all = state.map(lambda x: mp.get(x, 0.5) if isinstance(x, str) else np.nan).reindex(r_open_all.index).ffill().fillna(0.5)
    e_ma = ma50_state.map(lambda x: mp.get(x, 0.5) if isinstance(x, str) else np.nan).reindex(r_open_all.index).ffill().fillna(0.5)
    sma200 = spy_close_raw.rolling(200).mean()
    e_200 = (spy_close_raw > sma200).astype(float).where(sma200.notna()).reindex(r_open_all.index).ffill().fillna(1.0)

    def seg(x, lo, hi):
        return x[(x.index >= lo) & (x.index <= hi)]
    r_open, cash = seg(r_open_all, a, b), seg(cash_all, a, b)
    e, e_ma_s, e_200_s = seg(e_all, a, b), seg(e_ma, a, b), seg(e_200, a, b)
    avg_e = float(e.mean())
    s_ret = strat_returns(e, r_open, cash, cost)
    mix = const_mix_returns(avg_e, r_open, cash, cost)
    vol_s = float(s_ret.std(ddof=1)) if len(s_ret) > 2 else 0.0
    vol_spy = float(r_open.std(ddof=1)) if len(r_open) > 2 else 0.0
    mix_vol = const_mix_returns(min(vol_s / vol_spy, 1.5) if vol_spy > 0 else avg_e, r_open, cash, cost)
    b4 = strat_returns(e_200_s * (avg_e / float(e_200_s.mean())) if e_200_s.mean() > 0 else e_200_s, r_open, cash, cost)
    b5 = strat_returns(e_ma_s * (avg_e / float(e_ma_s.mean())) if e_ma_s.mean() > 0 else e_ma_s, r_open, cash, cost)
    b200 = strat_returns(e_200_s, r_open, cash, cost)
    spy = r_open
    P_s, P_mix, P_4b, P_5b, P_spy = perf(s_ret), perf(mix), perf(b4), perf(b5), perf(spy)
    plc = placebo_percentile(e, r_open, cash, cost, int(P["placebo_min_shift"]))
    ev = find_events(seg(spy_close_tr, a, b), float(P["event_dd"]), int(P["event_max_days"]))
    ev_rows, helped, caps, caps200 = [], 0, [], []
    for x in ev:
        rs, rm = _cum(s_ret, x["peak"], x["end"]), _cum(mix, x["peak"], x["end"])
        cap_s = _cum(s_ret, x["trough"], x["end"])
        cap_200 = _cum(b200, x["trough"], x["end"])
        cap_spy = _cum(spy, x["trough"], x["end"])
        c1 = cap_s / cap_spy if cap_spy > 0 else None
        c2 = cap_200 / cap_spy if cap_spy > 0 else None
        helped += rs > rm
        if c1 is not None:
            caps.append(c1)
        if c2 is not None:
            caps200.append(c2)
        ev_rows.append({**x, "strat": rs, "mix": rm, "capture": c1, "capture_200": c2})
    # 保留段 A
    r_h, cash_h, e_h = r_open_all[r_open_all.index >= ho], cash_all[cash_all.index >= ho], e_all[e_all.index >= ho]
    s_h = strat_returns(e_h, r_h, cash_h, cost)
    mix_h = const_mix_returns(float(e_h.mean()) if len(e_h) else avg_e, r_h, cash_h, cost)
    ev_h = find_events(spy_close_tr[spy_close_tr.index >= ho], float(P["event_dd"]), int(P["event_max_days"]))
    harm = sum(1 for x in ev_h if _cum(s_h, x["peak"], x["end"]) < _cum(mix_h, x["peak"], x["end"]))
    diff_m = (s_ret - mix).groupby([s_ret.index.year, s_ret.index.month]).sum()
    c = {
        "c1": plc["pct"] is not None and plc["pct"] >= float(P["placebo_pct"]),
        "c2": all(v is not None for v in (P_s["cagr"], P_mix["cagr"], P_4b["cagr"], P_5b["cagr"]))
              and P_s["cagr"] > P_mix["cagr"] and P_s["cagr"] > P_4b["cagr"] and P_s["cagr"] > P_5b["cagr"],
        "c3": P_s["mdd"] is not None and P_mix["mdd"] is not None and P_s["mdd"] <= P_mix["mdd"],
        "c4": bool(ev) and helped / len(ev) >= float(P["event_help_frac"]),
        "c5": bool(caps) and bool(caps200) and float(np.mean(caps)) >= float(np.mean(caps200)),
        "c6": harm <= int(P["holdout_max_harm"]),
    }
    return {"avg_exposure": avg_e, "strat": P_s, "mix": P_mix, "mix_vol": perf(mix_vol), "b200": perf(b200),
            "b4": P_4b, "b5": P_5b, "spy": P_spy, "timing_gain": (P_s["cagr"] - P_mix["cagr"])
            if P_s["cagr"] is not None and P_mix["cagr"] is not None else None,
            "placebo": plc, "events": ev_rows, "helped": helped, "capture": float(np.mean(caps)) if caps else None,
            "capture_200": float(np.mean(caps200)) if caps200 else None,
            "holdout": {"strat": perf(s_h), "mix": perf(mix_h), "spy": perf(r_h), "events": len(ev_h), "harm": harm},
            "diag": prediction_diag(seg(state, a, b), spy_close_tr), "boot_pos": block_bootstrap_pos(diff_m),
            "checks": c, "passed": all(c.values()),
            "state_share": {k: float((seg(state, a, b) == k).mean()) for k in ("risk_on", "neutral", "risk_off")}}


# ── 3. T1／T2：氣象台接進引擎（長窗重放） ───────────────────────────────────

def ew_daily(pre: dict) -> pd.Series:
    """樣本等權日報酬（收盤到收盤；兩天都有收盤的檔才算；SPY 除外）。"""
    dates = pre["dates"]
    out, prev = [], None
    for d in dates:
        cur = {s: v["close"] for s, v in (pre["by_date"].get(d) or {}).items() if s != "SPY" and v.get("close")}
        if prev is None:
            out.append(0.0)
        else:
            rs = [cur[s] / prev[s] - 1 for s in cur if s in prev and prev[s] > 0]
            out.append(float(np.mean(rs)) if rs else 0.0)
        prev = cur
    return pd.Series(out, index=pd.to_datetime(dates))


def adj_return(rep: dict, ew: pd.Series) -> float | None:
    """曝險調整後報酬 ＝ Σ（策略日報酬 − 前一日曝險 × 等權日報酬）。"""
    eq, ex = rep.get("equity"), rep.get("exposure")
    if eq is None or ex is None or len(eq) < 3:
        return None
    r = eq.pct_change().fillna(0.0)
    e1 = ex.shift(1).fillna(0.0)
    w = ew.reindex(r.index).fillna(0.0)
    return float((r - e1 * w).sum())


def window_pre(data: dict, start: str, end: str, thresholds: dict | None = None) -> dict | None:
    """截到 end、重放天數＝SPY 在 [start, end] 的交易日數 → precompute（錨定 SPY 倒數 days 根＝窗起點）。"""
    import engine_backtest as ebt
    e = pd.Timestamp(end)
    sub = {s: df[df.index <= e] for s, df in data.items()}
    spy = sub.get("SPY")
    if spy is None or not len(spy):
        return None
    days = int(((spy.index >= pd.Timestamp(start)) & (spy.index <= e)).sum())
    if days < 40:
        return None
    return ebt.precompute(sub, days, thresholds, None)


ARMS = ("A", "B", "C", "D")


def run_arms(pre: dict, weather: dict) -> dict:
    """四組：A 氣象台（停新倉＋收緊追蹤）、B MA50 退路（pre 原本的 regime）、C 不用濾網、D 氣象台但不收緊追蹤。"""
    import engine_backtest as ebt
    base = {"event_blackout": False}
    pre_w = dict(pre, regime={d: weather.get(d) for d in pre["dates"]})
    return {"A": ebt.replay(pre_w, dict(base)),
            "B": ebt.replay(pre, dict(base)),
            "C": ebt.replay(pre, dict(base, regime_filter=False)),
            "D": ebt.replay(pre_w, dict(base, regime_tighten=False))}


def t12_windows(data: dict, periods: dict, weather: dict, prereg: dict = TIMING_PREREG,
                thresholds: dict | None = None) -> list[dict]:
    """每窗 × 每種子：當時成分抽樣 → 截斷資料 → precompute（成分遮罩）→ 四組重放 → 曝險調整後報酬、回撤。"""
    import engine_backtest as ebt
    rows = []
    for (a, b) in prereg["windows"]:
        per_seed = []
        for sd in prereg["seeds"]:
            smp = ebt.pit_sample(periods, a, int(prereg["k"]), int(sd))
            sub = {s: data[s] for s in smp if s in data}
            if len(sub) < max(3, len(smp) // 2) or "SPY" not in data:
                per_seed.append({"seed": sd, "skip": True, "n": len(sub), "k": len(smp)})
                continue
            sub["SPY"] = data["SPY"]
            pre = window_pre(sub, a, b, thresholds)
            if pre is None:
                per_seed.append({"seed": sd, "skip": True, "n": len(sub), "k": len(smp)})
                continue
            pre["pit"] = {s: periods.get(s, []) for s in sub if s != "SPY"}
            ew = ew_daily(pre)
            reps = run_arms(pre, weather)
            r = {"seed": sd, "n": len(sub) - 1, "k": len(smp), "dates": (pre["dates"][0], pre["dates"][-1])}
            for k, rp in reps.items():
                m = rp["metrics"]
                r[k] = {"adj": adj_return(rp, ew), "ret": m["total_ret"], "mdd": m["max_dd"],
                        "expo": m.get("exposure"), "trades": m["n_trades"]}
            r["spy"] = ebt.bench_return(pre, pre["dates"])
            per_seed.append(r)
        ok = [x for x in per_seed if not x.get("skip")]

        def avg(arm, key):
            v = [x[arm][key] for x in ok if x[arm][key] is not None]
            return float(np.mean(v)) if v else None
        w = {"window": (a, b), "n_ok": len(ok), "seeds": per_seed,
             "spy": float(np.mean([x["spy"] for x in ok if x.get("spy") is not None])) if ok else None}
        for arm in ARMS:
            w[arm] = {k: avg(arm, k) for k in ("adj", "ret", "mdd", "expo", "trades")}
        rows.append(w)
    return rows


def t12_verdict(rows: list[dict], other: str, prereg: dict = TIMING_PREREG) -> dict:
    """A 對 other（T1：B；T2：D）。證據窗 ≥ 4/5 段 A−other > 0、2008–09 窗 A 回撤 ≤ other、保留段最多 1 段「較差且回撤較大」。"""
    ne = int(prereg["n_evidence_windows"])
    ev, ho = rows[:ne], rows[ne:]

    def diff(w):
        a_, o_ = w["A"]["adj"], w[other]["adj"]
        return (a_ - o_) if a_ is not None and o_ is not None else None
    wins = sum(1 for w in ev if (diff(w) or 0) > 0)
    first = ev[0] if ev else None
    mdd_ok = bool(first and first["A"]["mdd"] is not None and first[other]["mdd"] is not None
                  and first["A"]["mdd"] <= first[other]["mdd"])
    bad = sum(1 for w in ho if diff(w) is not None and diff(w) < 0 and (w["A"]["mdd"] or 0) > (w[other]["mdd"] or 0))
    c = {"wins": wins >= int(prereg["t1_min_evidence_wins"]), "mdd_2008": mdd_ok,
         "holdout": bad <= int(prereg["t1_holdout_max_bad"])}
    return {"wins": wins, "n_ev": len(ev), "bad": bad, "n_ho": len(ho), "checks": c, "passed": all(c.values()),
            "diffs": [diff(w) for w in rows]}


# ── 4. 文字 ──────────────────────────────────────────────────────────────

def _p(x, nd=1, sign=True):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"{x * 100:+.{nd}f}%" if sign else f"{x * 100:.{nd}f}%"


def timing_text(t12: list[dict] | None, v1: dict | None, v2: dict | None, t3: dict | None,
                meta: dict, prereg: dict = TIMING_PREREG) -> str:
    L = [f"⏱ *擇時第 1 步：市場氣象台*（預先登記 #{prereg['issue']}；擇時帳 #1–#3；程式預設參數；非投資建議）"]
    if t3:
        s, m, spy = t3["strat"], t3["mix"], t3["spy"]
        L += ["", f"*T3 氣象台當 SPY 持股比例*（偏多 100%／中性 50%／偏空 0%；證據窗 {prereg['t3_evidence'][0]}→{prereg['t3_evidence'][1]}）",
              f"・平均持股 {t3['avg_exposure']:.0%}（偏多 {t3['state_share']['risk_on']:.0%}／中性 {t3['state_share']['neutral']:.0%}／偏空 {t3['state_share']['risk_off']:.0%} 的時間）",
              f"・*擇時多賺*（vs 同樣持股、不預測）：{_p(t3['timing_gain'], 2)}／年｜策略年化 {_p(s['cagr'])}、同曝險配置 {_p(m['cagr'])}",
              f"・vs SPY：策略年化 {_p(s['cagr'])} vs SPY {_p(spy['cagr'])}｜最後的錢 1 → {s['final']:.2f} vs {spy['final']:.2f}",
              f"・最大回撤：策略 {_p(s['mdd'], 0, False)}｜同曝險配置 {_p(m['mdd'], 0, False)}｜SPY {_p(spy['mdd'], 0, False)}",
              f"・同持股的簡單規則：200 日線（縮放）{_p(t3['b4']['cagr'])}｜MA50 退路（縮放）{_p(t3['b5']['cagr'])}｜原版 200 日線 {_p(t3['b200']['cagr'])}",
              f"・安慰劑（時機打亂 {t3['placebo']['n']} 次）：真策略排在 {_p(t3['placebo']['pct'], 0, False)} 位置（≥ 90% 才算真的會抓時機）",
              f"・大跌事件 {len(t3['events'])} 次，贏過同曝險配置 {t3['helped']} 次｜回場捕獲率 {_p(t3['capture'], 0, False)}（200 日線 {_p(t3['capture_200'], 0, False)}）"]
        for x in t3["events"]:
            L.append(f"  - {str(x['peak'].date())} 高點 → 谷底 {str(x['trough'].date())}（{_p(x['depth'], 0)}）：策略 {_p(x['strat'], 0)}、同曝險 {_p(x['mix'], 0)}")
        dg = t3["diag"]
        L.append("・預測力：月底偏多 → SPY 下月平均 " + _p(dg["risk_on"]["mean"]) + f"（上漲機率 {_p(dg['risk_on']['p_up'], 0, False)}，{dg['risk_on']['n']} 個月）｜"
                 f"中性 {_p(dg['neutral']['mean'])}（{dg['neutral']['n']}）｜偏空 {_p(dg['risk_off']['mean'])}（{dg['risk_off']['n']}）｜方向命中率 {_p(dg['hit'], 0, False)}")
        h = t3["holdout"]
        L.append(f"・保留段 A（2019 起）：策略 {_p(h['strat']['cagr'])}、同曝險 {_p(h['mix']['cagr'])}、SPY {_p(h['spy']['cagr'])}｜事件 {h['events']} 次、輸給同曝險 {h['harm']} 次")
        ck = t3["checks"]
        names = {"c1": "安慰劑 ≥ 90%", "c2": "多賺且贏簡單規則", "c3": "回撤不較大", "c4": "大跌事件 ≥ 3/4", "c5": "回場不比 200 日線晚", "c6": "保留段不出事"}
        L.append("・判定：" + ("✅ 通過" if t3["passed"] else "❌ 沒通過") + "（" + "、".join(f"{names[k]}{'✓' if v else '✗'}" for k, v in ck.items()) + "）")
    if t12:
        L += ["", "*T1／T2 氣象台接進引擎*（每窗 40 檔當時成分 × 5 組種子；曝險調整後報酬，正＝贏）"]
        for w in t12:
            a, b = w["window"]
            L.append(f"・{a[:4]}–{b[:4]}（{w['n_ok']} 組）：A−B {_p(None if w['A']['adj'] is None or w['B']['adj'] is None else w['A']['adj'] - w['B']['adj'])}｜"
                     f"A−D {_p(None if w['A']['adj'] is None or w['D']['adj'] is None else w['A']['adj'] - w['D']['adj'])}｜"
                     f"A−C {_p(None if w['A']['adj'] is None or w['C']['adj'] is None else w['A']['adj'] - w['C']['adj'])}｜"
                     f"回撤 A {_p(w['A']['mdd'], 0, False)}／B {_p(w['B']['mdd'], 0, False)}｜曝險 A {_p(w['A']['expo'], 0, False)}／B {_p(w['B']['expo'], 0, False)}｜"
                     f"原始報酬 A {_p(w['A']['ret'], 0)}、SPY {_p(w['spy'], 0)}")
        for nm, v in (("T1 氣象台 vs MA50", v1), ("T2 收緊追蹤（A vs D）", v2)):
            if v:
                L.append(f"・{nm}：證據窗 {v['wins']}/{v['n_ev']} 段較好、2008–09 回撤{'不較大' if v['checks']['mdd_2008'] else '較大'}、"
                         f"保留段較差且回撤大 {v['bad']}/{v['n_ho']} 段 → " + ("✅ 通過" if v["passed"] else "❌ 沒通過"))
    L += ["", "⚠️ 同一段美國歷史；Yahoo 沒有已下市股票（每窗內偏倖存者）；氣象台 2000–07 屬降級模式不在此測；"
          "重放用日收盤、線上每小時更新（近似）；廣度否決是 2026-09 的案例調出來的。非投資建議"]
    if meta.get("note"):
        L.append(meta["note"])
    return "\n".join(L)


# ── 5. 進入點（需網路；Actions 上跑） ─────────────────────────────────────

def _chunked_fetch(tickers: list[str], start: str, end: str, chunk: int = 120) -> dict:
    import engine_backtest as ebt
    out: dict = {}
    tk = sorted(set(tickers))
    for i in range(0, len(tk), chunk):
        out.update(ebt.fetch_history(tk[i:i + chunk], start=start, end=end))
    return out


def _closes(tickers: list[str], start: str, end: str, auto_adjust: bool = True, field: str = "Close") -> pd.DataFrame:
    import yfinance as yf
    raw = yf.download(tickers, start=start, end=(pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                      auto_adjust=auto_adjust, progress=False, group_by="column")
    if raw is None or raw.empty:
        return pd.DataFrame()
    px = raw[field] if field in raw else raw
    if isinstance(px, pd.Series):
        px = px.to_frame(tickers[0])
    px.index = pd.to_datetime(px.index).tz_localize(None).normalize()
    return px


def run_timing(today: str | None = None, periods: dict | None = None, fetch=None, prereg: dict = TIMING_PREREG) -> dict:
    """預先登記 #95 的唯一正式執行：T3（SPY 層）＋ T1／T2（引擎長窗）。fetch 可注入（自測用）。"""
    import engine_backtest as ebt
    import market_weather as mw
    end = (today or pd.Timestamp.today().strftime("%Y-%m-%d"))
    F = fetch or {}
    get_closes = F.get("closes") or _closes
    get_hist = F.get("hist") or _chunked_fetch
    start_w = "2006-06-01"
    ps = get_closes(list(mw.SECTOR_ETFS), start_w, end)
    pc = get_closes(list(mw.CROSS_TICKERS), start_w, end)
    spy_tr = get_closes(["SPY"], "2005-01-01", end, True, "Close")["SPY"].dropna()
    spy_open = get_closes(["SPY"], "2005-01-01", end, True, "Open")["SPY"].dropna()
    spy_raw = get_closes(["SPY"], "2005-01-01", end, False, "Close")["SPY"].dropna()
    irx = get_closes(["^IRX"], "2005-01-01", end, True, "Close")["^IRX"].dropna()
    ma50 = ebt.regime_series(spy_tr)
    dates_all = [str(d.date()) for d in spy_tr.index if d >= pd.Timestamp("2007-01-02")]
    fb = {d: (v if isinstance(v, str) else None) for d, v in ((str(k.date()), v) for k, v in ma50.items())}
    wr = mw.replay_regimes(ps, pc, dates_all, fb)
    weather = {d: x["regime"] for d, x in wr.items()}
    state = pd.Series({pd.Timestamp(d): r for d, r in weather.items()})
    t3 = t3_evaluate(state, ma50, spy_open, spy_tr, spy_raw, irx, prereg)
    t3["source_share"] = {k: float(np.mean([x["source"] == k for x in wr.values()])) for k in ("weather", "ma50", "none")}
    # T1／T2
    if periods is None:
        import universe as un
        periods = un.fetch_sp500_periods()
    samples = set()
    for (a, _b) in prereg["windows"]:
        for sd in prereg["seeds"]:
            samples |= set(ebt.pit_sample(periods, a, int(prereg["k"]), int(sd)))
    ren = ebt.PIT_RENAMES
    raw = get_hist(sorted({ren.get(s, s) for s in samples}), "2006-06-01", prereg["windows"][-1][1])
    data = {s: raw[ren.get(s, s)] for s in samples if ren.get(s, s) in raw}
    if "SPY" in raw:
        data["SPY"] = raw["SPY"]
    per_eff = dict(periods)
    for old_t, new_t in ren.items():
        if old_t in per_eff and new_t in periods:
            per_eff[old_t] = sorted(set(periods[old_t]) | set(periods[new_t]))
    t12 = t12_windows(data, per_eff, weather, prereg)
    v1, v2 = t12_verdict(t12, "B", prereg), t12_verdict(t12, "D", prereg)
    meta = {"note": f"（氣象台重放：{_p(t3['source_share']['weather'], 0, False)} 的交易日五因子可算、"
                    f"{_p(t3['source_share']['ma50'], 0, False)} 退回 MA50；引擎樣本抓到價格 {len(data) - 1}/{len(samples)} 檔）"}
    return {"t3": t3, "t12": t12, "v1": v1, "v2": v2, "text": timing_text(t12, v1, v2, t3, meta, prereg)}


# ── 6. 自我測試（合成；離線） ───────────────────────────────────────────

if __name__ == "__main__":
    # 1) 現金換算、開盤報酬對齊、策略報酬（含成本）
    irx = pd.Series([5.0, 5.0], index=pd.bdate_range("2020-01-01", periods=2))
    cd = irx_daily(irx)
    y = 365 * 0.05 / (360 - 91 * 0.05)
    assert cd.iloc[0] == 0.0 and abs(cd.iloc[1] - y / 252) < 1e-12
    idx = pd.bdate_range("2020-01-01", periods=5)
    op = pd.Series([100, 110, 99, 99, 108.9], index=idx, dtype=float)
    ro = open_returns(op)
    assert abs(ro.iloc[0] - 0.10) < 1e-12 and ro.iloc[-1] == 0.0
    e = pd.Series([1.0, 0.0, 0.0, 1.0, 1.0], index=idx)
    sr = strat_returns(e, ro, pd.Series(0.0, index=idx), 0.001)
    assert abs(sr.iloc[1] - (1.0 * ro.iloc[1] - 0.001 * 1.0)) < 1e-12        # e_0=1 在 t=1 吃報酬、首次上車成本
    assert abs(sr.iloc[2] - (0.0 - 0.001)) < 1e-12                            # e_1=0：空手、下車成本
    print("✅ 1 現金換算、開盤到開盤報酬、策略報酬與成本對齊（t 收盤決定 → t+1 開盤）")

    # 2) 事件偵測：局部高點回撤 ≥15%、收復或 252 日取早者、互不重疊
    px = pd.Series(np.r_[np.linspace(100, 120, 50), np.linspace(120, 90, 30), np.linspace(90, 125, 60),
                         np.linspace(125, 104, 20), np.linspace(104, 130, 40)], index=pd.bdate_range("2010-01-01", periods=200))
    evs = find_events(px, 0.15, 252)
    assert len(evs) == 2 and abs(evs[0]["depth"] + 0.25) < 1e-9 and evs[0]["recovered"] is not None, evs
    assert abs(evs[1]["depth"] - (104 / 125 - 1)) < 1e-9 and evs[1]["peak"] > evs[0]["end"]
    print("✅ 2 大跌事件（局部高點、收復、互不重疊）")

    # 3) 安慰劑：植入「跌之前空手」的真本事 → 高百分位；隨機曝險 → 不高
    rng = np.random.default_rng(3)
    n = 1500
    idx3 = pd.bdate_range("2005-01-03", periods=n)
    ret = rng.normal(0.0004, 0.01, n)
    crash = np.zeros(n, bool)
    for c0 in range(100, n - 60, 250):
        crash[c0:c0 + 20] = True
        ret[c0:c0 + 20] -= 0.01
    op3 = pd.Series(100 * np.cumprod(1 + ret), index=idx3)
    ro3 = open_returns(op3)
    good = pd.Series(np.where(np.roll(crash, -1), 0.0, 1.0), index=idx3)   # 提前一天知道要跌 → 空手
    rnd = pd.Series(rng.choice([0.0, 1.0], n, p=[0.1, 0.9]), index=idx3)
    zero = pd.Series(0.0, index=idx3)
    pg, pr = placebo_percentile(good, ro3, zero, 0.0003, 126), placebo_percentile(rnd, ro3, zero, 0.0003, 126)
    assert pg["pct"] > 0.95 and pr["pct"] < 0.9, (pg, pr)
    print(f"✅ 3 安慰劑（有本事 {pg['pct']:.0%}、隨機 {pr['pct']:.0%}；平移 {pg['n']} 次）")

    # 4) 同曝險固定配置（每月再平衡）＋ T3 全流程與判定（合成：狀態在大跌前轉偏空）
    st = pd.Series(np.where(np.roll(crash, -1) | crash, "risk_off", "risk_on"), index=idx3)
    ma = pd.Series("risk_on", index=idx3)
    P4 = dict(TIMING_PREREG, t3_evidence=(str(idx3[0].date()), str(idx3[1100].date())), t3_holdout_start=str(idx3[1101].date()))
    cl3 = op3.copy()
    t3 = t3_evaluate(st, ma, op3, cl3, cl3, pd.Series(2.0, index=idx3), P4)
    assert t3["timing_gain"] > 0 and t3["checks"]["c1"] and t3["checks"]["c3"], (t3["timing_gain"], t3["checks"])
    mix = const_mix_returns(0.5, ro3, zero, 0.0)
    assert abs(perf(mix)["cagr"]) < abs(perf(ro3)["cagr"]) + 0.05
    t3b = t3_evaluate(pd.Series(np.where(rng.random(n) < 0.15, "risk_off", "risk_on"), index=idx3), ma, op3, cl3, cl3,
                      pd.Series(2.0, index=idx3), P4)
    assert not t3b["passed"]                                                 # 隨機狀態不該通過
    dg = t3["diag"]
    assert dg["n_dir"] > 0 and dg["risk_on"]["n"] > 0
    print(f"✅ 4 T3：有本事 → 擇時多賺 {t3['timing_gain'] * 100:+.1f}%／年、安慰劑通過；隨機狀態判沒通過")

    # 5) T1／T2：合成引擎長窗（兩窗 × 2 種子）→ 四組都跑、判定邏輯
    import engine_backtest as ebt
    syn = {f"S{i:02d}": ebt._synthetic(n=1300, seed=20 + i, drift=0.0003 + 0.0002 * (i % 4), start="2008-01-01")
           for i in range(10)}
    syn["SPY"] = ebt._synthetic(n=1300, seed=99, drift=0.0004, vol=0.01, start="2008-01-01")
    per = {s: [("2000-01-01", "9999-12-31")] for s in syn if s != "SPY"}
    days = [str(d.date()) for d in syn["SPY"].index]
    wx = {d: ("risk_off" if (i // 60) % 4 == 3 else "risk_on") for i, d in enumerate(days)}
    P5 = dict(TIMING_PREREG, windows=((days[500], days[800]), (days[801], days[1100]), (days[1101], days[1299])),
              n_evidence_windows=2, k=6, seeds=(1, 2), t1_min_evidence_wins=1)
    rows = t12_windows(syn, per, wx, P5)
    assert len(rows) == 3 and all(w["n_ok"] == 2 for w in rows), [w["n_ok"] for w in rows]
    for w in rows:
        for arm in ARMS:
            assert w[arm]["adj"] is not None and w[arm]["expo"] is not None
    v1 = t12_verdict(rows, "B", P5)
    assert set(v1["checks"]) == {"wins", "mdd_2008", "holdout"} and len(v1["diffs"]) == 3
    assert any(abs((w["A"]["ret"] or 0) - (w["C"]["ret"] or 0)) > 1e-9 for w in rows)   # 濾網真的有作用
    print("✅ 5 T1／T2：每窗重抽成分、四組重放、曝險調整後報酬、判定")

    # 6) 規則常數鎖住、文字 Markdown 安全
    assert TIMING_PREREG["windows"][0] == ("2008-01-01", "2009-12-31") and len(TIMING_PREREG["windows"]) == 9
    assert TIMING_PREREG["n_evidence_windows"] == 5 and TIMING_PREREG["k"] == 40 and TIMING_PREREG["seeds"] == (1, 2, 3, 4, 5)
    assert TIMING_PREREG["t3_evidence"] == ("2007-07-02", "2018-12-31") and TIMING_PREREG["t3_map"] == {"risk_on": 1.0, "neutral": 0.5, "risk_off": 0.0}
    assert TIMING_PREREG["placebo_pct"] == 0.90 and TIMING_PREREG["event_dd"] == 0.15 and TIMING_PREREG["spy_cost"] == 0.0003
    t3["source_share"] = {"weather": 0.9, "ma50": 0.1, "none": 0.0}
    txt = timing_text(rows, v1, t12_verdict(rows, "D", P5), t3, {"note": "（測試）"}, P5)
    assert txt.count("*") % 2 == 0 and "_" not in txt and "擇時多賺" in txt and "判定" in txt, txt[:400]
    print("✅ 6 規則常數鎖住、報告文字（Markdown 安全）")
    print("\ntiming_research selftest OK ✅")
