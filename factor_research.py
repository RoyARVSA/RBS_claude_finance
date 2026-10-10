"""
factor_research.py – 長歷史、當時成分（PIT）的選股因子研究 ＋ 產業 ETF 輪動（2026-10 衛星改為「選股」路線的第 1–3 步）

為什麼：既有研究只量到「技術評分擇時引擎」，而且只有 2 年、單一多頭路徑；Alpha 脊椎的 IC 閘門只看 300 天
（有效期數 ~14，動能等級的訊號統計上過不了）。這裡用 S&P 500 **當時**成分（fja05680，1996 起）＋多年歷史回答：
  1. 面板：每個月底的成分股，有多少抓得到價格（存活偏誤有多大）
  2. 價格因子能不能**挑股**（橫斷面 Rank IC、Newey-West t、分時段）——含現行技術綜合評分當排名因子（從沒測過）
     ＋ 前 20% 等權持有 vs SPY／vs 成分等權的預覽（扣換手成本）
  3. 產業 ETF 輪動（SPDR XL*，1998-12 起、ETF 不下市＝無存活偏誤）：每月持有過去 L 個月最強的 N 檔，
     L×N 網格 walk-forward 選參（每年 1 月只用之前的資料挑一次）、PBO、DSR，對照 SPY

紀律：月底收盤算訊號 → 次一交易日開盤成交 → 持有到下個月底的次一交易日開盤（無前視）；單邊成本 0.05%×換手。
誠實邊界：yfinance 沒有已下市股的價格 → 舊年份只剩「活到今天」的成分，結果偏樂觀（每段揭露覆蓋率）；
fja05680 期間表本身也可能有誤差；分時段只是同一段歷史切片，不是獨立樣本。教育用途，非投資建議。

純邏輯（寬表進、統計出）與抓取層分離；`python3 factor_research.py` 離線合成資料自測。
"""

from __future__ import annotations

import math
import random
import time

import numpy as np
import pandas as pd

COST_SIDE = 0.0005
MIN_CROSS = 30                     # 每期至少幾檔才算 IC
SECTOR_ETFS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "XLRE", "XLC"]
SUBPERIODS = [("1999-01-01", "2003-12-31", "1999–2003 網路泡沫"), ("2004-01-01", "2007-12-31", "2004–07 多頭"),
              ("2008-01-01", "2009-12-31", "2008–09 金融海嘯"), ("2010-01-01", "2019-12-31", "2010–19 長多"),
              ("2020-01-01", "2099-12-31", "2020– 疫情後")]
FACTOR_LABELS = {"mom_12_1": "12-1 動能", "ret_1m": "1 個月報酬", "vol_60": "60 日波動", "ma200": "距 200 日均線",
                 "ext_atr": "延伸度（距 MA20／ATR）", "hi_252": "距 52 週高點", "composite": "現行技術綜合評分"}
LOOKBACKS = (1, 3, 6, 9, 12)        # 產業輪動：過去 L 個月報酬
TOPNS = (2, 3, 4)                   # 產業輪動：持有前 N 檔


# ── 1. 日曆與報酬 ─────────────────────────────────────────────────────────────

def month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """每個月最後一個交易日（以傳入的交易日曆為準）。"""
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby([index.year, index.month]).max().values)


def period_returns(open_w: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """持有期報酬寬表：列＝訊號日 d_i，值＝d_i 次一交易日開盤 → d_{i+1} 次一交易日開盤。
    最後一個訊號日沒有完整持有期 → 不列。任一端缺價 → NaN（不硬補）。"""
    idx = open_w.index
    pos = idx.searchsorted(dates, side="right")          # 第一個 > d 的交易日
    rows = {}
    for i in range(len(dates) - 1):
        a, b = int(pos[i]), int(pos[i + 1])
        if b >= len(idx) or a >= b:
            break
        rows[dates[i]] = open_w.iloc[b] / open_w.iloc[a] - 1
    out = pd.DataFrame(rows).T if rows else pd.DataFrame(columns=open_w.columns)
    return out.replace([np.inf, -np.inf], np.nan)


def membership_matrix(periods: dict, dates: pd.DatetimeIndex, tickers: list[str]) -> pd.DataFrame:
    """布林寬表：當天是否為成分（期間表為閉區間，universe.parse_period_csv 已把移除日轉成最後在籍日）。"""
    ds = [str(d.date()) for d in dates]
    M = pd.DataFrame(False, index=dates, columns=tickers)
    for t in tickers:
        for a, b in periods.get(t, []):
            M.loc[[a <= d <= b for d in ds], t] = True
    return M


# ── 2. 因子 ──────────────────────────────────────────────────────────────────

def price_factor_frames(c: pd.DataFrame, h: pd.DataFrame | None = None, l: pd.DataFrame | None = None) -> dict:
    """原始方向的價格因子寬表（不翻號：IC 正負直接告訴你方向）。只用 t 以前含 t 的資料。"""
    if h is None or l is None or h.empty or l.empty:
        h, l = c, c
    h, l = h.reindex_like(c), l.reindex_like(c)
    prev = c.shift(1)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()]).groupby(level=0).max().reindex(c.index)
    # min_periods ≈ 90%：單根缺 K 棒不該讓滾動因子 NaN 長達一整個視窗（驗證 Low）
    atr = tr.rolling(14, min_periods=12).mean()
    out = {
        "mom_12_1": c.shift(21) / c.shift(252) - 1,
        "ret_1m": c / c.shift(21) - 1,
        "vol_60": c.pct_change(fill_method=None).rolling(60, min_periods=54).std(),
        "ma200": c / c.rolling(200, min_periods=180).mean() - 1,
        "ext_atr": (c - c.rolling(20, min_periods=18).mean()) / atr.replace(0, np.nan),
        "hi_252": c / c.rolling(252, min_periods=227).max() - 1,
    }
    return {k: v.replace([np.inf, -np.inf], np.nan) for k, v in out.items()}


def composite_frame(data: dict, dates: pd.DatetimeIndex, mtf: bool = True) -> pd.DataFrame:
    """現行技術綜合評分（indicators.composite_series，與正式掃描同一函數）在訊號日的值。
    收盤含 NaN 的檔（composite_series 前提不成立）略過。"""
    import indicators as ind
    cols = {}
    for sym, df in data.items():
        if sym == "SPY" or df is None or "Close" not in df or len(df) < 260:
            continue
        df = df.dropna(subset=["Close"])
        try:
            s = ind.composite_series(df["Close"], df.get("High"), df.get("Low"), df.get("Volume"), mtf=mtf)
        except Exception:
            continue
        cols[sym] = s.reindex(dates, method="ffill", limit=3)
    return pd.DataFrame(cols, index=dates)


# ── 3. IC 與分位組合 ─────────────────────────────────────────────────────────

def rank_ic_rows(F: pd.DataFrame, R: pd.DataFrame, M: pd.DataFrame | None = None, min_n: int = MIN_CROSS) -> pd.Series:
    """逐期橫斷面 Spearman IC（只在「當期成分 ∩ 兩邊都有值」上算）。"""
    F, R = F.align(R, join="inner", axis=None)
    ok = F.notna() & R.notna()
    if M is not None:
        ok &= M.reindex_like(F).fillna(False).astype(bool)
    Fr = F.where(ok).rank(axis=1)
    Rr = R.where(ok).rank(axis=1)
    n = ok.sum(axis=1)
    fc = Fr.sub(Fr.mean(axis=1), axis=0)
    rc = Rr.sub(Rr.mean(axis=1), axis=0)
    num = (fc * rc).sum(axis=1)
    den = np.sqrt((fc ** 2).sum(axis=1) * (rc ** 2).sum(axis=1))
    ic = num / den.replace(0, np.nan)
    return ic[n >= min_n].dropna()


def quantile_portfolio(F: pd.DataFrame, R: pd.DataFrame, M: pd.DataFrame | None = None, q: float = 0.2,
                       top: bool = True, min_n: int = MIN_CROSS) -> pd.DataFrame:
    """每期挑因子前（或後）q 比例等權持有：回 {gross, turnover, net, n}（逐期）。換手＝新進比例（單邊）。"""
    F, R = F.align(R, join="inner", axis=None)
    ok = F.notna()                                       # 選股只看訊號日可得的資訊（不能偷看「下期報酬存不存在」）
    if M is not None:
        ok &= M.reindex_like(F).fillna(False).astype(bool)
    pct = F.where(ok).rank(axis=1, pct=True)
    sel = (pct > 1 - q) if top else (pct <= q)
    rows, prev = [], set()
    for d in F.index:
        if int(ok.loc[d].sum()) < min_n:
            continue
        names = set(sel.columns[sel.loc[d].fillna(False).values])
        if not names:
            continue
        rr = R.loc[d, list(names)]
        if rr.notna().sum() == 0:
            continue
        g = float(rr.mean())                             # 持有期中資料中斷（下市等）的檔：缺值不計入（殘留存活偏誤，計數揭露）
        to = 1.0 if not prev else len(names - prev) / len(names)
        rows.append({"date": d, "gross": g, "turnover": to, "net": g - 2 * COST_SIDE * to, "n": len(names),
                     "missing": int(rr.isna().sum())})
        prev = names
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame(columns=["gross", "turnover", "net", "n"])


def perf(r: pd.Series, periods_per_year: int = 12) -> dict:
    """逐期報酬序列 → CAGR／年化波動／Sharpe（無風險 0）／最大回撤。"""
    r = r.dropna()
    if len(r) < 3:
        return {"n": len(r), "cagr": None, "vol": None, "sharpe": None, "max_dd": None}
    eq = (1 + r).cumprod()
    yrs = len(r) / periods_per_year
    sd = float(r.std(ddof=1))
    return {"n": int(len(r)), "cagr": float(eq.iloc[-1] ** (1 / yrs) - 1) if yrs > 0 else None,
            "vol": sd * math.sqrt(periods_per_year),
            "sharpe": float(r.mean()) / sd * math.sqrt(periods_per_year) if sd > 0 else None,
            "max_dd": abs(float((eq / eq.cummax() - 1).min()))}


def _sub(s: pd.Series, a: str, b: str) -> pd.Series:
    return s[(s.index >= pd.Timestamp(a)) & (s.index <= pd.Timestamp(b))]


def factor_study(F_all: dict, R: pd.DataFrame, M: pd.DataFrame, spy_r: pd.Series, ew_r: pd.Series,
                 q: float = 0.2) -> dict:
    """每個因子：全期與分時段 IC 摘要、前 q 等權（扣成本）vs SPY／vs 成分等權、後 q 對照。"""
    import factor_eval as fe
    out = {}
    for name, F in F_all.items():
        Fd = F.reindex(R.index)
        ics = rank_ic_rows(Fd, R, M)

        def _summ(x):
            sm = fe.ic_summary(x, horizon=21)
            if len(x) >= 3:                              # 月頻不重疊但 IC 仍可能自相關 → Newey-West（Andrews 落差規則）
                sm["t_nw"] = fe._nw_t(x, max(1, int(4 * (len(x) / 100) ** (2 / 9))))
            return sm
        summ = _summ(ics)
        subs = {}
        for a, b, lab in SUBPERIODS:
            s_ = _sub(ics, a, b)
            if len(s_) >= 12:
                subs[lab] = _summ(s_)
        top = quantile_portfolio(Fd, R, M, q, top=True)
        bot = quantile_portfolio(Fd, R, M, q, top=False)
        res = {"ic": summ, "sub": subs, "n_cross": float(M.reindex(R.index).sum(axis=1).mean())}
        if len(top):
            t_net = top["net"]
            res["top"] = perf(t_net)
            res["top_turnover"] = float(top["turnover"].mean())
            res["top_missing"] = float(top["missing"].sum() / max(1, top["n"].sum()))
            res["top_vs_spy"] = _excess(t_net, spy_r)
            res["top_vs_ew"] = _excess(t_net, ew_r)
            res["top_sub"] = {lab: _excess(_sub(t_net, a, b), spy_r) for a, b, lab in SUBPERIODS
                              if len(_sub(t_net, a, b)) >= 12}
        if len(bot):
            res["bot"] = perf(bot["net"])
        out[name] = res
    return out


def _excess(r: pd.Series, bench: pd.Series) -> dict:
    """主動報酬：年化超額（CAGR 差）、資訊比率、逐年勝率。"""
    r, b = r.dropna().align(bench.dropna(), join="inner")
    if len(r) < 12:
        return {"n": len(r), "cagr_diff": None, "ir": None, "win_years": None}
    pr, pb = perf(r), perf(b)
    act = r - b
    sd = float(act.std(ddof=1))
    yr = pd.DataFrame({"r": (1 + r).groupby(r.index.year).prod() - 1, "b": (1 + b).groupby(b.index.year).prod() - 1})
    return {"n": int(len(r)), "cagr_diff": (pr["cagr"] - pb["cagr"]) if pr["cagr"] is not None and pb["cagr"] is not None else None,
            "ir": float(act.mean()) / sd * math.sqrt(12) if sd > 0 else None,
            "win_years": float((yr["r"] > yr["b"]).mean()) if len(yr) else None, "years": int(len(yr))}


# ── 4. 產業 ETF 輪動 ─────────────────────────────────────────────────────────

def sector_rotation_returns(close: pd.DataFrame, open_: pd.DataFrame, dates: pd.DatetimeIndex,
                            lookback_m: int, top_n: int) -> pd.DataFrame:
    """每個訊號日：過去 lookback_m 個月（21 交易日/月）報酬排名，持有前 top_n 檔等權；扣換手成本。
    訊號只用訊號日以前含當日的收盤；可用 ETF＝當天訊號有值者（XLRE/XLC 上市且滿回看期才納入）。"""
    sig = close / close.shift(21 * lookback_m) - 1
    R = period_returns(open_, dates)
    rows, prev = [], set()
    for d in R.index:
        s = sig.loc[d].dropna() if d in sig.index else pd.Series(dtype=float)
        s = s[[c for c in s.index if pd.notna(R.loc[d].get(c))]]
        if len(s) < top_n + 1:
            continue
        names = set(s.sort_values(ascending=False).index[:top_n])
        g = float(R.loc[d, list(names)].mean())
        to = 1.0 if not prev else len(names - prev) / len(names)
        rows.append({"date": d, "gross": g, "net": g - 2 * COST_SIDE * to, "turnover": to, "hold": ",".join(sorted(names))})
        prev = names
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame(columns=["gross", "net", "turnover", "hold"])


def walk_forward_select(grid: pd.DataFrame, min_train: int = 36, window: int = 60) -> tuple[pd.Series, list]:
    """每年 1 月（及第一個可選月）只用之前 window 個月（不足時用全部、至少 min_train）挑 Sharpe 最高的設定，
    持有到下次重選；回 (樣本外月報酬, [(生效月, 設定)])。"""
    grid = grid.dropna(how="all")
    out, picks, cur = {}, [], None
    for i, d in enumerate(grid.index):
        if i - 1 < min_train:
            continue
        if cur is None or d.month == 1:
            # 第 i-1 列的報酬要到「第 i 列的執行開盤」才實現 → 選參只能用到第 i-2 列（驗證 Low：一夜前視）
            hist = grid.iloc[max(0, i - 1 - window):i - 1]
            sr = hist.mean() / hist.std(ddof=1).replace(0, np.nan)
            sr = sr.dropna()
            if len(sr):
                cur = str(sr.idxmax())
                picks.append((str(d.date()), cur))
        if cur is not None and pd.notna(grid.loc[d, cur]):
            out[d] = float(grid.loc[d, cur])
    return pd.Series(out, dtype=float), picks


def sector_study(close: pd.DataFrame, open_: pd.DataFrame, spy_close: pd.Series, spy_open: pd.Series,
                 start: str | None = None) -> dict:
    import falsifier as fz
    dates = month_ends(close.index)
    if start:
        dates = dates[dates >= pd.Timestamp(start)]        # 訊號仍用完整序列算（回看期在 start 之前），研究期從 start 起
    grid = {}
    for L in LOOKBACKS:
        for N in TOPNS:
            r = sector_rotation_returns(close, open_, dates, L, N)
            if len(r):
                grid[f"L{L}_N{N}"] = r["net"]
    G = pd.DataFrame(grid)
    spy_r = period_returns(spy_open.to_frame("SPY"), dates)["SPY"]
    ew_r = period_returns(open_, dates).mean(axis=1)                       # 可用產業 ETF 等權（每月再平衡）
    oos, picks = walk_forward_select(G)
    res = {"grid": {k: {**perf(G[k]), **{"vs_spy": _excess(G[k], spy_r)}} for k in G.columns},
           "wf": perf(oos), "wf_vs_spy": _excess(oos, spy_r), "wf_vs_ew": _excess(oos, ew_r), "picks": picks,
           "spy": perf(spy_r.reindex(oos.index)), "ew": perf(ew_r.reindex(oos.index)),
           "wf_sub": {lab: _excess(_sub(oos, a, b), spy_r) for a, b, lab in SUBPERIODS if len(_sub(oos, a, b)) >= 12},
           "start": str(G.index[0].date()) if len(G) else None, "end": str(G.index[-1].date()) if len(G) else None}
    try:
        res["pbo"] = fz.pbo_cscv(G.dropna().values)
    except Exception as e:
        res["pbo"] = {"pbo": None, "note": type(e).__name__}
    try:
        act = (oos - spy_r.reindex(oos.index)).dropna()
        trial = [float((G[k] - spy_r).dropna().mean() / (G[k] - spy_r).dropna().std(ddof=1)) for k in G.columns]
        res["dsr"] = fz.deflated_sharpe(float(act.mean() / act.std(ddof=1)), len(act), len(G.columns), trial,
                                        skew=float(act.skew()), kurt=float(act.kurt()) + 3.0)
    except Exception as e:
        res["dsr"] = {"dsr": None, "note": type(e).__name__}
    return res


# ── 5. 文字報告（繁中；無底線、單 * 成對）────────────────────────────────────

def _p(x, nd=1, sign=True):
    return "—" if x is None else (f"{x:+.{nd}%}" if sign else f"{x:.{nd}%}")


def _f(x, nd=2):
    return "—" if x is None else f"{x:+.{nd}f}"


def coverage_text(cov: pd.DataFrame) -> list[str]:
    lines = ["*資料覆蓋（存活偏誤量尺）*：每期「當時成分」中抓得到價格的比例"]
    for a, b, lab in SUBPERIODS:
        s = cov[(cov.index >= pd.Timestamp(a)) & (cov.index <= pd.Timestamp(b))]
        if len(s):
            lines.append(f"・{lab}：平均 {s['members'].mean():.0f} 檔成分、有價格 {s['covered'].mean():.0f} 檔（{(s['covered'] / s['members']).mean():.0%}）")
    return lines


def factor_text(study: dict, cov: pd.DataFrame | None, meta: dict) -> str:
    lines = [f"🔬 *選股因子研究*（當時 S&P 500 成分、{meta.get('start')}→{meta.get('end')}、每月底訊號、次日開盤成交、"
             f"前 {meta.get('q', 0.2):.0%} 等權、單邊成本 0.05%）"]
    if cov is not None and len(cov):
        lines += coverage_text(cov)
    lines.append("\n*橫斷面 Rank IC*（正＝數值高的股票下個月表現較好；|t| ≥ 2 才算有訊號）")
    for name, r in study.items():
        ic = r["ic"]
        lines.append(f"・{FACTOR_LABELS.get(name, name)}：IC {_f(ic.get('mean'), 3)}｜ICIR {_f(ic.get('icir'))}｜"
                     f"t {_f(ic.get('t_nw'), 1)}｜{ic.get('n', 0)} 期｜命中 {_p(ic.get('hit'), 0, False)}")
        segs = [f"{lab.split(' ')[0]} {_f(s.get('mean'), 3)}" for lab, s in r.get("sub", {}).items()]
        if segs:
            lines.append("  分時段 IC：" + "｜".join(segs))
    qq = meta.get("q", 0.2)
    lines.append(f"\n*前 {qq:.0%} 等權（扣成本）vs SPY*（預覽，不是最終組合；正＝贏）")
    for name, r in study.items():
        if "top" not in r:
            continue
        vs, ve = r["top_vs_spy"], r["top_vs_ew"]
        lines.append(f"・{FACTOR_LABELS.get(name, name)}：年化 {_p(r['top'].get('cagr'))}｜vs SPY {_p(vs.get('cagr_diff'))}"
                     f"（IR {_f(vs.get('ir'))}、贏的年份 {_p(vs.get('win_years'), 0, False)}）｜vs 成分等權 {_p(ve.get('cagr_diff'))}｜"
                     f"回撤 {_p(r['top'].get('max_dd'), 0, False)}｜月換手 {r.get('top_turnover', 0):.0%}"
                     + (f"｜後 {qq:.0%} 年化 {_p(r['bot'].get('cagr'))}" if "bot" in r else "")
                     + (f"｜持有期資料中斷 {r['top_missing']:.1%}" if r.get("top_missing") else ""))
        segs = [f"{lab.split(' ')[0]} {_p(s.get('cagr_diff'))}" for lab, s in r.get("top_sub", {}).items()]
        if segs:
            lines.append("  分時段 vs SPY：" + "｜".join(segs))
    if meta.get("spy") and meta.get("ew"):
        lines.append(f"同期 SPY 年化 {_p(meta['spy'].get('cagr'))}（回撤 {_p(meta['spy'].get('max_dd'), 0, False)}）｜"
                     f"成分等權 年化 {_p(meta['ew'].get('cagr'))}")
    lines.append("⚠️ 舊年份只剩活到今天的成分（見覆蓋率）→ 結果偏樂觀；同時測了多個因子，單一因子的 t≈2 要打折；"
                 "IC 是排名能力、不等於扣成本後能贏大盤。非投資建議")
    return "\n".join(lines)


def sector_text(res: dict) -> str:
    lines = [f"🔄 *產業 ETF 輪動*（SPDR 產業 ETF、{res.get('start')}→{res.get('end')}、每月底訊號、次日開盤成交、單邊成本 0.05%）"]
    wf, vs, ve = res.get("wf") or {}, res.get("wf_vs_spy") or {}, res.get("wf_vs_ew") or {}
    sp, ew = res.get("spy") or {}, res.get("ew") or {}
    lines.append(f"*walk-forward 樣本外*（每年 1 月只用之前資料挑 L×N）：年化 {_p(wf.get('cagr'))}｜Sharpe {_f(wf.get('sharpe'))}｜"
                 f"回撤 {_p(wf.get('max_dd'), 0, False)}｜{wf.get('n', 0)} 個月")
    lines.append(f"・vs SPY：年化 {_p(vs.get('cagr_diff'))}｜IR {_f(vs.get('ir'))}｜贏的年份 {_p(vs.get('win_years'), 0, False)}（{vs.get('years', 0)} 年）")
    lines.append(f"・vs 產業等權：年化 {_p(ve.get('cagr_diff'))}｜IR {_f(ve.get('ir'))}")
    lines.append(f"・同期 SPY 年化 {_p(sp.get('cagr'))}（Sharpe {_f(sp.get('sharpe'))}、回撤 {_p(sp.get('max_dd'), 0, False)}）｜"
                 f"產業等權 年化 {_p(ew.get('cagr'))}")
    segs = [f"{lab.split(' ')[0]} {_p(s.get('cagr_diff'))}" for lab, s in (res.get("wf_sub") or {}).items()]
    if segs:
        lines.append("・分時段 vs SPY：" + "｜".join(segs))
    pb, ds = res.get("pbo") or {}, res.get("dsr") or {}
    if pb.get("pbo") is not None:
        lines.append(f"・PBO {pb['pbo']:.0%}（{len(res.get('grid', {}))} 組設定的網格排名在樣本外有沒有用；≥50% ＝沒有）")
    if ds.get("dsr") is not None:
        lines.append(f"・DSR {ds['dsr']:.2f}（主動報酬扣掉 {len(res.get('grid', {}))} 組嘗試的幸運上限；> 0.95 才算可信）")
    picks = res.get("picks") or []
    if picks:
        lines.append("・挑過的設定：" + "、".join(f"{d[:4]} {k.replace('_', '/')}" for d, k in picks[-6:]) + ("…" if len(picks) > 6 else ""))
    best = sorted(res.get("grid", {}).items(), key=lambda kv: -(kv[1].get("sharpe") or -9))[:3]
    if best:
        lines.append("*全期最佳 3 組*（事後挑的＝樣本內，只當參考）：" + "｜".join(
            f"{k.replace('_', '/')} 年化 {_p(v.get('cagr'))}、vs SPY {_p((v.get('vs_spy') or {}).get('cagr_diff'))}" for k, v in best))
    lines.append("⚠️ 只有一段美股歷史；PBO／DSR 已扣掉網格搜尋的運氣，但扣不掉「同一段歷史」的運氣。非投資建議")
    return "\n".join(lines)


# ── 6. 抓取層（Actions；本地 proxy 擋 Yahoo）──────────────────────────────────

def fetch_ohlcv(tickers: list[str], start: str, chunk: int = 120, pause: float = 1.0) -> dict:
    """分批 yf.download（auto_adjust）。回 {sym: OHLCV DataFrame}；抓不到的直接缺。"""
    import yfinance as yf
    out = {}
    tickers = sorted(set(tickers))
    for i in range(0, len(tickers), chunk):
        part = tickers[i:i + chunk]
        try:
            raw = yf.download(part, start=start, auto_adjust=True, progress=False, group_by="column", threads=True)
        except Exception as e:
            print(f"factor_research: 抓價失敗（{i}）{type(e).__name__}")
            continue
        if raw is None or raw.empty:
            continue
        multi = isinstance(raw.columns, pd.MultiIndex)
        for s in part:
            try:
                if multi:
                    df = pd.DataFrame({f: raw[(f, s)] for f in ("Open", "High", "Low", "Close", "Volume") if (f, s) in raw.columns})
                else:
                    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
                df = df.dropna(subset=["Close"])
                df = df[df["Close"] > 0]
                # 壞開盤價（0／與收盤差兩倍以上）→ 當缺值，否則持有期報酬變 inf、汙染等權基準與分位組合（驗證 Med）
                df["Open"] = df["Open"].where((df["Open"] > 0) & (df["Open"] / df["Close"]).between(0.5, 2.0))
                if len(df) >= 260:
                    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
                    out[s] = df
            except Exception:
                continue
        time.sleep(pause)
    return out


def _wide(data: dict, col: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({s: df[col] for s, df in data.items() if col in df}).reindex(index)


def run_factors(start: str = "2004-01-01", end: str | None = None, q: float = 0.2, composite: bool = True,
                sample: int | None = None, seed: int = 1, periods: dict | None = None, fetch_fn=None) -> dict:
    """第 1–2 步進入點：當時成分面板 → 價格因子（＋技術評分）→ IC／分位組合 → 文字。"""
    if periods is None:
        import universe as un
        periods = un.fetch_sp500_periods()
    if not periods:
        return {"text": "❌ S&P 500 成分歷史抓取失敗，稍後再試"}
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    tickers = sorted(t for t, ps in periods.items() if any(a <= end and b >= start for a, b in ps))
    if sample and sample < len(tickers):
        tickers = sorted(random.Random(seed).sample(tickers, sample))
    warm = (pd.Timestamp(start) - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
    raw = (fetch_fn or fetch_ohlcv)(tickers + ["SPY"], warm)
    if "SPY" not in raw:
        return {"text": "❌ SPY 抓不到，無法建立交易日曆"}
    cal = raw["SPY"].index
    cal = cal[cal <= pd.Timestamp(end)]
    data = {s: raw[s] for s in tickers if s in raw}
    c, o = _wide(data, "Close", cal), _wide(data, "Open", cal)
    h, l = _wide(data, "High", cal), _wide(data, "Low", cal)
    dates = month_ends(cal)
    dates = dates[dates >= pd.Timestamp(start)]
    R = period_returns(o, dates)
    if R.empty:
        return {"text": f"❌ {start}→{end} 期間內沒有完整的月份可研究"}
    M = membership_matrix(periods, R.index, list(c.columns))
    allM = membership_matrix(periods, R.index, tickers)
    cov = pd.DataFrame({"members": allM.sum(axis=1), "covered": (M & R.notna()).sum(axis=1)})
    F = price_factor_frames(c, h, l)
    if composite:
        F["composite"] = composite_frame(data, R.index)
    spy_r = period_returns(raw["SPY"][["Open"]].reindex(cal).rename(columns={"Open": "SPY"}), dates)["SPY"]
    ew_r = R.where(M).mean(axis=1)
    study = factor_study({k: v.reindex(R.index) for k, v in F.items()}, R, M, spy_r, ew_r, q)
    meta = {"start": str(R.index[0].date()) if len(R) else start, "end": str(R.index[-1].date()) if len(R) else end,
            "q": q, "spy": perf(spy_r), "ew": perf(ew_r), "n_tickers": len(tickers), "n_fetched": len(data)}
    return {"study": study, "coverage": cov, "meta": meta, "text": factor_text(study, cov, meta)}


def run_sectors(start: str = "1999-01-01", fetch_fn=None) -> dict:
    """第 3 步進入點：SPDR 產業 ETF 輪動 vs SPY。"""
    warm = (pd.Timestamp(start) - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
    raw = (fetch_fn or fetch_ohlcv)(SECTOR_ETFS + ["SPY"], warm)
    if "SPY" not in raw:
        return {"text": "❌ SPY 抓不到"}
    cal = raw["SPY"].index
    etfs = {s: raw[s] for s in SECTOR_ETFS if s in raw}
    if len(etfs) < 5:
        return {"text": f"❌ 產業 ETF 只抓到 {len(etfs)} 檔"}
    close, open_ = _wide(etfs, "Close", cal), _wide(etfs, "Open", cal)
    res = sector_study(close, open_, raw["SPY"]["Close"], raw["SPY"]["Open"], start=start)
    res["text"] = sector_text(res)
    return res


# ── 7. 自我測試（合成資料；離線）──────────────────────────────────────────────

def _synthetic_panel(n_stocks: int = 80, n_days: int = 1600, seed: int = 3, edge: float = 0.0) -> tuple[dict, dict]:
    """合成個股：每檔有固定「品質」q，若 edge>0，日報酬帶 edge×q 的漂移 → 動能因子應有正 IC。"""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2012-01-02", periods=n_days)
    data, periods = {}, {}
    for i in range(n_stocks):
        q_ = rng.normal()
        r = rng.normal(0.0003 + edge * q_, 0.018, n_days)
        c = 50 * np.cumprod(1 + r)
        o = c * (1 + rng.normal(0, 0.002, n_days))
        hi, lo = np.maximum(c, o) * 1.004, np.minimum(c, o) * 0.996
        data[f"S{i:03d}"] = pd.DataFrame({"Open": o, "High": hi, "Low": lo, "Close": c,
                                          "Volume": rng.integers(1e5, 1e6, n_days).astype(float)}, index=idx)
        periods[f"S{i:03d}"] = [("2000-01-01", "9999-12-31")]
    spy = 300 * np.cumprod(1 + rng.normal(0.0004, 0.01, n_days))
    data["SPY"] = pd.DataFrame({"Open": spy, "High": spy * 1.003, "Low": spy * 0.997, "Close": spy,
                                "Volume": np.full(n_days, 1e7)}, index=idx)
    return data, periods


if __name__ == "__main__":
    # 1) 日曆與持有期報酬：次日開盤進、下期次日開盤出；最後一期不列
    idx = pd.bdate_range("2024-01-01", "2024-04-30")
    op = pd.DataFrame({"A": np.arange(1, len(idx) + 1, dtype=float)}, index=idx)
    me = month_ends(idx)
    assert list(me.month) == [1, 2, 3, 4] and me[0] == pd.Timestamp("2024-01-31")
    pr = period_returns(op, me)
    a, b = idx.searchsorted(me[0], side="right"), idx.searchsorted(me[1], side="right")
    assert len(pr) == 2 and abs(pr.iloc[0]["A"] - (op.iloc[b]["A"] / op.iloc[a]["A"] - 1)) < 1e-12   # 4/30 之後無交易日 → 3 月那期不完整
    op0 = op.copy(); op0.iloc[5, 0] = 0.0                       # 壞開盤價 0 → inf 不得外流
    assert np.isfinite(period_returns(op0, me).fillna(0).values).all()
    print("✅ 1 月底訊號、次日開盤持有期報酬（無前視、最後一期不列、壞開盤價不產生 inf）")

    # 2) 成分矩陣（閉區間）與 IC：完美排名 IC=1、反向 −1、成分外不算
    d3 = pd.DatetimeIndex(["2024-01-31", "2024-02-29"])
    Mx = membership_matrix({"A": [("2024-01-01", "2024-01-31")], "B": [("2000-01-01", "9999-12-31")]}, d3, ["A", "B"])
    assert Mx.loc["2024-01-31", "A"] and not Mx.loc["2024-02-29", "A"] and Mx["B"].all()
    cols = [f"T{i}" for i in range(40)]
    Fp = pd.DataFrame([np.arange(40.0)] * 2, index=d3, columns=cols)
    assert np.allclose(rank_ic_rows(Fp, Fp * 2, min_n=30).values, 1.0)
    assert np.allclose(rank_ic_rows(Fp, -Fp, min_n=30).values, -1.0)
    Mh = pd.DataFrame(True, index=d3, columns=cols)
    Mh.iloc[:, :15] = False
    assert rank_ic_rows(Fp, Fp, Mh, min_n=30).empty                         # 只剩 25 檔 < 30 → 不算
    print("✅ 2 成分矩陣閉區間、IC 計算（完美／反向／成分外排除）")

    # 3) 合成面板：有 edge 時動能 IC 顯著為正、無 edge 時在雜訊範圍；技術評分因子算得出
    for edge, expect_pos in ((0.0012, True), (0.0, False)):
        dat, per = _synthetic_panel(edge=edge)
        fetch = lambda syms, start, _d=dat: {s: _d[s] for s in syms if s in _d}   # noqa: E731
        out = run_factors("2013-06-01", "2018-03-01", periods=per, fetch_fn=fetch, composite=(edge > 0))
        mom = out["study"]["mom_12_1"]["ic"]
        if expect_pos:
            assert mom["mean"] > 0.05 and mom["t_nw"] > 3, mom
            assert out["study"]["mom_12_1"]["top"]["cagr"] > out["study"]["mom_12_1"]["bot"]["cagr"]
            assert "composite" in out["study"] and out["study"]["composite"]["ic"]["n"] > 20
        else:
            assert abs(mom["mean"]) < 0.05, mom
        assert out["coverage"]["covered"].min() > 0
    t_ = out["text"]
    assert "選股因子研究" in t_ and "資料覆蓋" in t_ and t_.count("*") % 2 == 0 and "_" not in t_, t_
    print("✅ 3 合成面板：植入 edge 時動能 IC 顯著為正、無 edge 時在雜訊範圍；技術評分因子可算")

    # 4) 產業輪動：植入「強者恆強」時樣本外勝過等權；PBO/DSR/文字可算
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2004-01-01", periods=3000)
    drifts = np.linspace(-0.0004, 0.0012, 9)
    regime = np.repeat(rng.permutation(9), 3000 // 9 + 1)[:3000]
    cl = {}
    for j, s in enumerate(SECTOR_ETFS[:9]):
        mu = np.where(regime == j, 0.002, drifts[j] * 0.2)        # 輪流當主升段，持續數月 → 動能有效
        cl[s] = 30 * np.cumprod(1 + rng.normal(mu, 0.012, 3000))
    C = pd.DataFrame(cl, index=idx)
    O = C * (1 + rng.normal(0, 0.001, C.shape))
    spy_c = C.mean(axis=1)
    res = sector_study(C, O, spy_c, spy_c)
    assert res["wf"]["n"] > 60 and res["wf_vs_ew"]["cagr_diff"] > 0, res["wf_vs_ew"]
    assert res["pbo"]["pbo"] is not None and res["dsr"]["dsr"] is not None
    st = sector_text(res)
    assert "產業 ETF 輪動" in st and st.count("*") % 2 == 0 and "_" not in st, st
    # 可用 ETF 逐步加入：XLRE 只在有資料後才進排名
    C2 = C.copy(); C2["XLRE"] = np.where(np.arange(3000) > 2000, C["XLK"], np.nan)
    r2 = sector_rotation_returns(C2, C2, month_ends(idx), 3, 3)
    early = r2[r2.index < idx[2000]]
    assert not any("XLRE" in h for h in early["hold"]), early["hold"].head()
    # walk-forward 不得用到「選參當月的前一列」（該列報酬在執行開盤才實現）：把那一列改成極端值，選擇不變
    gi = pd.date_range("2010-01-31", periods=60, freq="ME")
    gg = pd.DataFrame({"A": 0.0, "B": 0.0}, index=gi)
    gg.iloc[44:48, 0] = [0.010, 0.012, 0.011, 0.010]          # A：穩定小正報酬
    gg.iloc[44:48, 1] = [0.000, 0.020, 0.021, 0.022]          # B：只有「含 2013-12 那列」的視窗 Sharpe 才高於 A
    _o, _pk = walk_forward_select(gg, min_train=2, window=3)
    assert dict(_pk).get("2014-01-31") == "A", _pk            # 用到第 47 列（一夜前視）就會選 B
    _leak = gg.iloc[45:48]                                    # 反證：若偷看該列，B 確實勝出（測試不空轉）
    assert (_leak.mean() / _leak.std()).idxmax() == "B"
    print(f"✅ 4 產業輪動：植入持續強勢時 walk-forward 勝過等權（{res['wf_vs_ew']['cagr_diff']:+.1%}/年）、PBO {res['pbo']['pbo']:.0%}、晚上市 ETF 不提前入選")
    print("\nfactor_research selftest OK ✅")
