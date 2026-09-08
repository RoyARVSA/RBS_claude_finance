"""
factor_eval.py – 因子評估（alphalens 式，純 pandas/numpy、離線可測；估值層 P2）

用來回答「MoS / 修正動能 / 品質分 這些慢因子到底有沒有預測力」——進配置前的把關
（VALUATION_LANDSCAPE §7.4 門檻：21 日 Rank IC > 0.03 且 ICIR > 0.3；宇宙小時只求方向一致）。

輸入慣例：factor = {date: {ticker: value}}（各期橫截面快照，date 為**可得知日**——PIT 由呼叫端保證）；
closes = {ticker: pd.Series}（日收盤，DatetimeIndex）。前瞻報酬取 date 之後第 1 個交易日起算 h 日
（避免同日前視）。
輸出：逐期 Rank IC、IC 均值 / ICIR / t 值、分位數平均前瞻報酬與 top−bottom spread、因子自相關（換手代理）。
教育用途，非投資建議。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _rank(vals: dict) -> pd.Series:
    s = pd.Series({k: v for k, v in vals.items() if v is not None and math.isfinite(float(v))}, dtype=float)
    return s.rank() if len(s) else s


def forward_returns(closes: dict, date: str, horizon: int) -> dict:
    """date 之後第 1 個交易日收盤 → 再 horizon 個交易日收盤的報酬（無同日前視）。"""
    out = {}
    ts = pd.Timestamp(date)
    for t, s in (closes or {}).items():
        try:
            s = s.dropna()
            idx = s.index.searchsorted(ts, side="right")      # 嚴格在 date 之後
            if idx + horizon < len(s):
                p0, p1 = float(s.iloc[idx]), float(s.iloc[idx + horizon])
                if p0 > 0:
                    out[t] = p1 / p0 - 1
        except Exception:
            continue
    return out


def rank_ic(factor: dict, fwd: dict, min_n: int = 5) -> float | None:
    """Spearman 相關（兩邊都取 rank 再算 Pearson）。"""
    common = [t for t in factor if t in fwd and factor[t] is not None and fwd[t] is not None]
    if len(common) < min_n:
        return None
    a = _rank({t: factor[t] for t in common}); b = _rank({t: fwd[t] for t in common})
    a, b = a.align(b, join="inner")
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a.values, b.values)[0, 1])


def ic_series(factor_by_date: dict, closes: dict, horizon: int = 21, min_n: int = 5) -> pd.Series:
    """逐期 Rank IC（index=date）。"""
    rows = {}
    for d in sorted(factor_by_date):
        fwd = forward_returns(closes, d, horizon)
        ic = rank_ic(factor_by_date[d], fwd, min_n)
        if ic is not None:
            rows[pd.Timestamp(d)] = ic
    return pd.Series(rows, dtype=float)


def ic_summary(ics: pd.Series) -> dict:
    n = int(len(ics.dropna()))
    if n == 0:
        return {"n": 0, "mean": None, "icir": None, "t": None, "hit": None}
    m, sd = float(ics.mean()), float(ics.std(ddof=1)) if n > 1 else 0.0
    return {"n": n, "mean": m, "icir": (m / sd) if sd > 0 else None,
            "t": (m / sd * math.sqrt(n)) if sd > 0 else None, "hit": float((ics > 0).mean())}


def quantile_returns(factor_by_date: dict, closes: dict, horizon: int = 21, q: int = 3, min_n: int = 6) -> dict:
    """各分位數（1=最低因子值 … q=最高）的平均前瞻報酬與 top−bottom spread。"""
    buckets = {i: [] for i in range(1, q + 1)}
    for d in sorted(factor_by_date):
        f = factor_by_date[d]
        fwd = forward_returns(closes, d, horizon)
        common = [t for t in f if t in fwd and f[t] is not None]
        if len(common) < min_n:
            continue
        s = pd.Series({t: float(f[t]) for t in common}).rank(method="first")
        try:
            lab = pd.qcut(s, q, labels=False) + 1
        except ValueError:
            continue
        for t, b in lab.items():
            buckets[int(b)].append(fwd[t])
    mean = {i: (float(np.mean(v)) if v else None) for i, v in buckets.items()}
    spread = (mean[q] - mean[1]) if (mean[q] is not None and mean[1] is not None) else None
    return {"mean_by_q": mean, "n_by_q": {i: len(v) for i, v in buckets.items()}, "spread": spread}


def factor_autocorr(factor_by_date: dict, lag: int = 1) -> float | None:
    """相鄰快照的橫截面 rank 相關（高=穩定低換手；MoS 應 >0.9、修正動能會低）。"""
    dates = sorted(factor_by_date)
    if len(dates) <= lag:
        return None
    vals = []
    for a, b in zip(dates[:-lag], dates[lag:]):
        ic = rank_ic(factor_by_date[a], factor_by_date[b], min_n=5)
        if ic is not None:
            vals.append(ic)
    return float(np.mean(vals)) if vals else None


def evaluate(factor_by_date: dict, closes: dict, horizons=(21, 63), q: int = 3) -> dict:
    out = {"horizons": {}, "autocorr": factor_autocorr(factor_by_date), "n_dates": len(factor_by_date)}
    for h in horizons:
        ics = ic_series(factor_by_date, closes, h)
        out["horizons"][h] = {"ic": ic_summary(ics), "quantiles": quantile_returns(factor_by_date, closes, h, q)}
    return out


def passes_gate(ev: dict, horizon: int = 21, ic_min: float = 0.03, icir_min: float = 0.3, min_dates: int = 12) -> tuple[bool, str]:
    """進配置門檻：IC 均值 > ic_min 且 ICIR > icir_min 且樣本期數足夠。"""
    h = (ev.get("horizons") or {}).get(horizon) or {}
    ic = h.get("ic") or {}
    if ic.get("n", 0) < min_dates:
        return False, f"樣本期數 {ic.get('n', 0)} < {min_dates}"
    if ic.get("mean") is None or ic["mean"] <= ic_min:
        return False, f"IC 均值 {ic.get('mean')}"
    if ic.get("icir") is None or ic["icir"] <= icir_min:
        return False, f"ICIR {ic.get('icir')}"
    return True, "通過"


def eval_text(name: str, ev: dict) -> str:
    lines = [f"📏 *因子評估：{name}*（{ev.get('n_dates', 0)} 期快照）"]
    for h, r in sorted((ev.get("horizons") or {}).items()):
        ic, qr = r["ic"], r["quantiles"]
        if ic.get("n"):
            lines.append(f"{h} 日：Rank IC {ic['mean']:+.3f}（ICIR {ic['icir'] if ic['icir'] is None else round(ic['icir'], 2)}，"
                         f"命中 {ic['hit']:.0%}，n={ic['n']}）"
                         + (f"｜Q高−Q低 {qr['spread']:+.1%}" if qr.get("spread") is not None else ""))
    if ev.get("autocorr") is not None:
        lines.append(f"因子自相關 {ev['autocorr']:.2f}（越高越穩定、換手越低）")
    ok, why = passes_gate(ev)
    lines.append(("✅ 通過配置門檻" if ok else f"➖ 未達門檻（{why}）") + "；非投資建議")
    return "\n".join(lines)


# ── 自我測試（合成資料：因子=未來報酬+雜訊 → IC 應顯著為正；純雜訊 → 約 0）────

if __name__ == "__main__":
    rng = np.random.default_rng(11)
    tickers = [f"T{i:02d}" for i in range(20)]
    idx = pd.bdate_range("2025-01-01", periods=400)
    rets = {t: rng.normal(0.0003, 0.015, len(idx)) for t in tickers}
    closes = {t: pd.Series(100 * np.cumprod(1 + rets[t]), index=idx) for t in tickers}
    dates = [str(d.date()) for d in idx[50:300:10]]                 # 25 期快照，每 10 個交易日
    # 有效因子：與未來 21 日報酬相關（偷看未來只為造測試資料）
    good, noise = {}, {}
    for d in dates:
        fwd = forward_returns(closes, d, 21)
        good[d] = {t: fwd[t] + rng.normal(0, 0.03) for t in tickers if t in fwd}
        noise[d] = {t: rng.normal() for t in tickers}
    ev_g, ev_n = evaluate(good, closes), evaluate(noise, closes)
    icg, icn = ev_g["horizons"][21]["ic"], ev_n["horizons"][21]["ic"]
    assert icg["mean"] > 0.3 and icg["icir"] > 1.0 and icg["n"] == 25, icg
    assert abs(icn["mean"]) < 0.15, icn
    assert ev_g["horizons"][21]["quantiles"]["spread"] > 0
    assert passes_gate(ev_g)[0] and not passes_gate(ev_n)[0]
    print(f"✅ 1 有效因子 IC {icg['mean']:+.3f}（ICIR {icg['icir']:.2f}）；雜訊因子 IC {icn['mean']:+.3f}")

    # 2) 無同日前視：forward_returns 從 date 之後第一個交易日起算
    d0 = dates[0]; fr = forward_returns(closes, d0, 5)
    s = closes["T00"]; i0 = s.index.searchsorted(pd.Timestamp(d0), side="right")
    assert abs(fr["T00"] - (s.iloc[i0 + 5] / s.iloc[i0] - 1)) < 1e-12 and s.index[i0] > pd.Timestamp(d0)
    assert forward_returns(closes, str(idx[-2].date()), 21) == {}         # 尾端不足 → 空
    print("✅ 2 前瞻報酬無同日前視、尾端截斷")

    # 3) 自相關：持續因子 ≈ 1、雜訊 ≈ 0；門檻文字
    persist = {d: {t: i for i, t in enumerate(tickers)} for d in dates}
    assert factor_autocorr(persist) > 0.99 and abs(factor_autocorr(noise)) < 0.3
    assert rank_ic({"a": 1, "b": 2}, {"a": 1, "b": 2}) is None                # n < min_n
    assert rank_ic({"a": 1, "b": 1, "c": 1, "d": 1, "e": 1}, {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}) is None   # 零變異
    txt = eval_text("MoS", ev_g)
    assert txt.count("*") % 2 == 0 and "_" not in txt
    print(txt)
    print("\nfactor_eval selftest OK ✅")
