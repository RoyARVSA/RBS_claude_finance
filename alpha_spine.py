"""
alpha_spine.py – 橫斷面 Alpha 脊椎（A 段；規劃見 ALPHA_SPINE.md §1）

把「該看哪些股票」從手選 watchlist 變成每日對整個選股池的排名：
  價格因子（12-1 動能／1 月反轉／延伸度／波動，向量化）
  + 基本面因子（品質 PIT、預估修正 PIT）
  + C 段核准的 DSL 因子
  → factor_eval 週頻 Rank IC / ICIR / Newey-West t 閘門
  → ICIR 加權合成 → 今日排名 → data/alpha/rank.json（公開資料衍生、明文）

純邏輯（離線可測、可注入合成資料）；抓取與落檔由 alpha_nightly.py 負責。
方向寫死（不依 IC 符號翻轉）；沒有任何因子通過閘門 → gate_passed=False，引擎不用候選池。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ALPHA_DIR = Path(__file__).parent / "data" / "alpha"
RANK_FILE = ALPHA_DIR / "rank.json"
LOOKBACK_DAYS = 300          # IC 評估回看交易日
SAMPLE_STEP = 5              # 快照間隔（交易日）——減少重疊視窗
HORIZON = 21                 # 前瞻報酬視窗（交易日）
TOP_N = 50                   # rank.json 保留名次
MIN_COVERAGE = 30            # 因子在單一快照日至少要有幾檔有值才參與 IC

# 因子方向固定：全部轉成「值越高越好」後再評估（不允許依 IC 符號自動翻轉＝資料探勘）
PRICE_FACTORS = ("mom_12_1", "rev_1m", "ext_atr", "vol_60")
FUND_FACTORS = ("quality", "rev")
FACTOR_LABELS = {"mom_12_1": "12-1 動能", "rev_1m": "1 月反轉", "ext_atr": "延伸度(負)", "vol_60": "低波動",
                 "quality": "品質分", "rev": "預估修正"}


# ── 1. 價格因子（寬表向量化；只用當日以前含當日）─────────────────────────────

def wide_frames(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """{sym: OHLCV DataFrame} → {"close","high","low","volume"} 寬表（columns=sym，index 對齊聯集）。"""
    out = {}
    for col, key in (("Close", "close"), ("High", "high"), ("Low", "low"), ("Volume", "volume")):
        cols = {}
        for s, df in (data or {}).items():
            if df is None or col not in df:
                continue
            ser = df[col]
            if getattr(ser.index, "tz", None) is not None:
                ser = ser.copy(); ser.index = ser.index.tz_localize(None)
            cols[s] = ser.astype(float)
        out[key] = pd.DataFrame(cols).sort_index() if cols else pd.DataFrame()
    return out


def atr14(close: pd.DataFrame, high: pd.DataFrame | None, low: pd.DataFrame | None, n: int = 14) -> pd.DataFrame:
    if high is None or low is None or high.empty or low.empty:
        high, low = close, close
    high, low = high.reindex_like(close), low.reindex_like(close)
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()]).groupby(level=0).max()
    return tr.rolling(n).mean()


def price_factors(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """回 {因子名: 寬表}，全部已轉成「高＝好」方向。"""
    c = frames["close"]
    if c.empty:
        return {}
    a = atr14(c, frames.get("high"), frames.get("low"))
    out = {
        "mom_12_1": c.shift(21) / c.shift(252) - 1,
        "rev_1m": -(c / c.shift(21) - 1),
        "ext_atr": -((c - c.rolling(20).mean()) / a.replace(0, np.nan)),
        "vol_60": -c.pct_change().rolling(60).std(),
    }
    return {k: v.replace([np.inf, -np.inf], np.nan) for k, v in out.items()}


def sample_dates(index: pd.DatetimeIndex, lookback: int = LOOKBACK_DAYS, step: int = SAMPLE_STEP,
                 horizon: int = HORIZON) -> list[str]:
    """快照日：最後一個「還有 horizon 日前瞻報酬」的交易日往回每 step 日一個，最多 lookback 日。"""
    idx = list(index)
    if len(idx) <= horizon + 1:
        return []
    last = len(idx) - horizon - 2                      # 需要 t+1 … t+1+horizon 存在（forward_returns：pos+1+h < len）
    first = max(0, last - lookback)
    return [str(idx[i].date()) for i in range(last, first - 1, -step)][::-1]


def factor_by_date(wide: pd.DataFrame, dates: list[str]) -> dict[str, dict[str, float]]:
    out = {}
    for d in dates:
        ts = pd.Timestamp(d)
        if ts not in wide.index:
            continue
        row = wide.loc[ts].dropna()
        out[d] = {str(k): float(v) for k, v in row.items() if math.isfinite(float(v))}
    return out


# ── 2. 基本面因子（PIT 重建）───────────────────────────────────────────────────

def quality_by_date(stores: dict[str, dict], dates: list[str]) -> dict[str, dict[str, float]]:
    """{sym: fin_data store} → {date: {sym: quality score}}，每日只用 available_at ≤ date 的期別。
    同一檔在「可得期別數」不變的日子重用結果（品質分只在新財報可得時變）。"""
    try:
        import fin_data as fd
        import quality as ql
    except Exception:
        return {}
    out: dict[str, dict[str, float]] = {d: {} for d in dates}
    for sym, store in (stores or {}).items():
        cache: dict[int, float | None] = {}
        for d in dates:
            try:
                periods = fd.pit_view(store, d, "A")
            except Exception:
                continue
            k = len(periods)
            if k < 2:
                continue
            if k not in cache:
                try:
                    q = ql.quality_summary(periods, None, None)
                    cache[k] = q.get("score")
                except Exception:
                    cache[k] = None
            v = cache[k]
            if v is not None and math.isfinite(float(v)):
                out[d][sym] = float(v)
    return {d: v for d, v in out.items() if v}


def rev_score_pit(rows: list, as_of: str) -> float | None:
    """預估帳本列（週頻 [d, eps0y, eps1y, rev0y, rev1y, up30, down30, n, …]）只用 d ≤ as_of 的列，
    重建 revision_momentum 的 score（0.6×90 日 EPS 變動 + 0.4×上下修廣度；缺一項就只用另一項）。"""
    try:
        import estimates_ledger as el
    except Exception:
        return None
    rs = [r for r in (rows or []) if isinstance(r, list) and r and str(r[0])[:10] <= as_of[:10]]
    if not rs:
        return None
    last = rs[-1]
    chg90 = None
    old = el._row_about_days_ago(rs, 90) if len(rs) >= 2 else None
    if old is not None and last[1] is not None and old[1] is not None and el._same_fy(last, old):
        chg90 = el._chg(last[1], old[1])
    n = last[7] if len(last) > 7 else None
    up, down = (last[5] if len(last) > 5 else None), (last[6] if len(last) > 6 else None)
    breadth = ((float(up or 0) - float(down or 0)) / float(n)) if n else None
    parts, w = [], []
    if chg90 is not None and math.isfinite(chg90):
        parts.append(max(-1.0, min(1.0, chg90 / 0.10))); w.append(0.6)
    if breadth is not None and math.isfinite(breadth):
        parts.append(max(-1.0, min(1.0, breadth))); w.append(0.4)
    return (sum(p * ww for p, ww in zip(parts, w)) / sum(w)) if w else None


def rev_by_date(ledgers: list[dict], dates: list[str]) -> dict[str, dict[str, float]]:
    """多本帳本（主帳本 watchlist + 夜間宇宙帳本）合併 → {date: {sym: rev score}}（PIT）。"""
    rows_by_sym: dict[str, list] = {}
    for led in ledgers or []:
        for sym, ent in ((led or {}).get("tickers") or {}).items():
            rows = (ent or {}).get("rows") or []
            if rows and (sym not in rows_by_sym or len(rows) > len(rows_by_sym[sym])):
                rows_by_sym[sym] = rows
    out: dict[str, dict[str, float]] = {d: {} for d in dates}
    for sym, rows in rows_by_sym.items():
        for d in dates:
            v = rev_score_pit(rows, d)
            if v is not None:
                out[d][sym] = float(v)
    return {d: v for d, v in out.items() if v}


# ── 3. 評估 / 閘門 / 權重 ───────────────────────────────────────────────────────

def evaluate_factors(fbd: dict[str, dict], closes: dict[str, pd.Series], horizon: int = HORIZON,
                     min_coverage: int = MIN_COVERAGE) -> dict[str, dict]:
    """{name: factor_by_date} → {name: {ic, icir, t_nw, n_eff, hit, spread, pass, reason, coverage, ev}}。"""
    import factor_eval as fe
    out = {}
    for name, series in (fbd or {}).items():
        usable = {d: v for d, v in (series or {}).items() if len(v) >= min_coverage}
        cov = int(np.median([len(v) for v in series.values()])) if series else 0
        if len(usable) < 3:
            out[name] = {"ic": None, "icir": None, "t_nw": None, "n_eff": 0, "hit": None, "spread": None,
                         "pass": False, "reason": f"覆蓋不足（中位 {cov} 檔 < {min_coverage}）", "coverage": cov, "weight": 0.0}
            continue
        ev = fe.evaluate(usable, closes, horizons=(horizon,))
        ok, why = fe.passes_gate(ev, horizon)
        h = ev["horizons"][horizon]
        ic = h["ic"]
        out[name] = {"ic": ic.get("mean"), "icir": ic.get("icir"), "t_nw": ic.get("t_nw"),
                     "n_eff": ic.get("n_eff"), "hit": ic.get("hit"), "spread": h["quantiles"].get("spread"),
                     "autocorr": ev.get("autocorr"), "n_dates": len(usable),
                     "pass": bool(ok), "reason": why, "coverage": cov, "weight": 0.0}
    return out


def assign_weights(evals: dict[str, dict]) -> dict[str, dict]:
    """通過閘門者依 ICIR 正部歸一化；未通過 0。就地更新並回傳。"""
    pos = {k: max(float(v.get("icir") or 0.0), 0.0) for k, v in evals.items() if v.get("pass")}
    tot = sum(pos.values())
    for k, v in evals.items():
        v["weight"] = (pos[k] / tot) if (k in pos and tot > 0) else 0.0
    return evals


def composite_scores(fbd: dict[str, dict], weights: dict[str, float], date: str) -> dict[str, dict]:
    """單一快照日：Σ w × 橫截面 rank-pct（中心化 −0.5..0.5）／可得權重和；conf = 可得權重和 / 總權重。"""
    tot_w = sum(w for w in weights.values() if w > 0)
    if tot_w <= 0:
        return {}
    ranks: dict[str, pd.Series] = {}
    for name, w in weights.items():
        if w <= 0:
            continue
        vals = (fbd.get(name) or {}).get(date) or {}
        if len(vals) < 2:
            continue
        s = pd.Series(vals, dtype=float)
        ranks[name] = s.rank(pct=True) - 0.5
    syms = set().union(*[set(r.index) for r in ranks.values()]) if ranks else set()
    out = {}
    for sym in syms:
        num, den, parts = 0.0, 0.0, {}
        for name, r in ranks.items():
            if sym in r.index:
                num += weights[name] * float(r[sym]); den += weights[name]
                parts[name] = round(float((fbd[name][date])[sym]), 4)
        if den > 0:
            out[sym] = {"score": round(num / den, 4), "conf": round(den / tot_w, 3), "f": parts}
    return out


def composite_by_date(fbd: dict[str, dict], weights: dict[str, float], dates: list[str]) -> dict[str, dict[str, float]]:
    out = {}
    for d in dates:
        cs = composite_scores(fbd, weights, d)
        if cs:
            out[d] = {s: v["score"] for s, v in cs.items()}
    return out


# ── 4. 市場背景（給 B 段特徵與顯示）───────────────────────────────────────────

def market_context(close: pd.DataFrame, spy: pd.Series | None) -> dict:
    """廣度＝選股池站上 MA50 比例（最後一日）；SPY MA50 三態（engine_backtest.regime_series 規則）。"""
    out = {"breadth_pct": None, "spy_regime": None, "as_of": None}
    try:
        if close is not None and not close.empty:
            ma50 = close.rolling(50).mean()
            last = close.index[-1]
            above = (close.loc[last] > ma50.loc[last]).sum()
            total = int(ma50.loc[last].notna().sum())
            out["breadth_pct"] = round(float(above) / total, 3) if total else None
            out["as_of"] = str(last.date())
    except Exception:
        pass
    try:
        if spy is not None and len(spy) >= 50:
            import engine_backtest as eb
            rs = eb.regime_series(spy.dropna())
            v = rs.iloc[-1]
            out["spy_regime"] = v if isinstance(v, str) else None
    except Exception:
        pass
    return out


def breadth_series(close: pd.DataFrame) -> pd.Series:
    """逐日廣度（站上 MA50 比例）——B 段訓練特徵用（只用當日以前資料）。"""
    if close is None or close.empty:
        return pd.Series(dtype=float)
    ma50 = close.rolling(50).mean()
    above = (close > ma50).sum(axis=1)
    total = ma50.notna().sum(axis=1).replace(0, np.nan)
    return (above / total).astype(float)


# ── 5. 組裝 rank.json ──────────────────────────────────────────────────────────

def build_rank(frames: dict[str, pd.DataFrame], dates: list[str], fund_fbd: dict[str, dict] | None = None,
               extra_fbd: dict[str, dict] | None = None, spy: pd.Series | None = None,
               horizon: int = HORIZON, top_n: int = TOP_N, as_of: str | None = None) -> dict:
    """
    frames：wide_frames 輸出；dates：快照日；fund_fbd：{quality/rev: factor_by_date}；extra_fbd：C 段核准因子。
    回 rank.json 內容（純 dict，可 json.dumps）。
    """
    close = frames["close"]
    pf = price_factors(frames)
    fbd = {name: factor_by_date(w, dates) for name, w in pf.items()}
    for src in (fund_fbd or {}, extra_fbd or {}):
        for name, series in src.items():
            fbd[name] = {d: v for d, v in (series or {}).items() if d in set(dates)} or series
    closes = {str(c): close[c].dropna() for c in close.columns}
    evals = assign_weights(evaluate_factors(fbd, closes, horizon))
    weights = {k: v["weight"] for k, v in evals.items()}
    gate_passed = any(v["pass"] for v in evals.values())
    last_date = as_of or str(close.index[-1].date())
    # 今日排名：用最後一個交易日的因子值（不需前瞻報酬）
    fbd_today = {name: {last_date: (factor_by_date(w, [last_date]).get(last_date) or {})} for name, w in pf.items()}
    for src in (fund_fbd or {}, extra_fbd or {}):
        for name, series in src.items():
            if series:
                latest_d = max(series)          # 基本面因子最新快照（PIT 已保證 ≤ 今日）
                fbd_today[name] = {last_date: series.get(last_date) or series[latest_d]}
    today_scores = composite_scores(fbd_today, weights, last_date) if gate_passed else {}
    ranked = sorted(today_scores.items(), key=lambda kv: -kv[1]["score"])
    top = [{"t": s, "rank": i + 1, **v} for i, (s, v) in enumerate(ranked[:top_n])]
    comp_ev = {}
    if gate_passed:
        cbd = composite_by_date(fbd, weights, dates)
        if cbd:
            ce = evaluate_factors({"composite": cbd}, closes, horizon).get("composite") or {}
            comp_ev = {k: ce.get(k) for k in ("ic", "icir", "t_nw", "n_eff", "spread", "hit")}
    ctx = market_context(close, spy)
    return {"version": 1, "as_of": last_date, "n_universe": int(close.shape[1]), "lookback_days": LOOKBACK_DAYS,
            "horizon": horizon, "n_dates": len(dates), "gate_passed": bool(gate_passed),
            "factors": {k: {kk: vv for kk, vv in v.items() if kk != "ev"} for k, v in evals.items()},
            "composite": comp_ev, "top": top, "market": ctx,
            "note": "權重由過去 IC 決定、合成 IC 為描述性（輕微樣本內）；候選池只在 gate_passed 時啟用"}


def load_rank(path: Path | None = None) -> dict | None:
    p = Path(path or RANK_FILE)
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) and d.get("as_of") else None
    except Exception:
        return None


def save_rank(rank: dict, path: Path | None = None) -> Path:
    p = Path(path or RANK_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rank, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    tmp.replace(p)
    return p


def pool_symbols(rank: dict | None, k: int = 20, exclude: list[str] | None = None) -> list[str]:
    """候選池：gate 通過時的前 k 名（排除 exclude）；否則空。"""
    if not rank or not rank.get("gate_passed"):
        return []
    ex = set(exclude or [])
    return [r["t"] for r in (rank.get("top") or []) if r.get("t") and r["t"] not in ex][:max(0, int(k))]


# ── 6. 文字輸出（Telegram legacy Markdown：只用單 *、不用底線）────────────────

def _f3(x, nd=3):
    return "—" if x is None else f"{x:+.{nd}f}" if nd else f"{x}"


def _pc(x, nd=1, sign=True):
    """百分比或 —（py3.11：f-string 內不能再巢狀同引號 f-string）。"""
    if x is None:
        return "—"
    return f"{x:+.{nd}%}" if sign else f"{x:.{nd}%}"


def factors_text(rank: dict | None) -> str:
    if not rank:
        return "🧬 Alpha 脊椎尚未產出（夜間工作流還沒跑，或 data/alpha/rank.json 缺）"
    lines = [f"🧬 *Alpha 脊椎因子評估*（{rank.get('as_of')}｜{rank.get('n_universe')} 檔｜{rank.get('horizon')} 日 IC｜{rank.get('n_dates')} 個快照）"]
    for name, v in (rank.get("factors") or {}).items():
        lab = FACTOR_LABELS.get(name, name).replace("_", "·")
        mark = "✅" if v.get("pass") else "➖"
        if v.get("ic") is None:
            lines.append(f"{mark} {lab}：{v.get('reason')}")
        else:
            lines.append(f"{mark} {lab}：IC {v['ic']:+.3f}｜ICIR {_f3(v.get('icir'), 2)}｜NW t {_f3(v.get('t_nw'), 1)}｜"
                         f"價差 {_pc(v.get('spread'))}｜權重 {v.get('weight', 0):.0%}"
                         + ("" if v.get("pass") else f"（{v.get('reason')}）"))
    c = rank.get("composite") or {}
    if c.get("ic") is not None:
        lines.append(f"合成：IC {c['ic']:+.3f}｜ICIR {_f3(c.get('icir'), 2)}｜價差 {_pc(c.get('spread'))}（描述性）")
    m = rank.get("market") or {}
    lines.append(f"市場：廣度 {_pc(m.get('breadth_pct'), 0, False)} 站上 MA50｜SPY {m.get('spy_regime') or '—'}")
    lines.append("閘門：" + ("✅ 有因子通過，候選池可用（`/set alpha_pool_enabled on`）" if rank.get("gate_passed")
                          else "➖ 無因子通過 IC 閘門，候選池停用（退回 watchlist）"))
    lines.append("_IC>0.03、ICIR>0.3、NW t≥2、有效期數≥12 才配權；方向寫死不翻轉；非投資建議_")
    return "\n".join(lines)


def rank_text(rank: dict | None, n: int = 15, held: list[str] | None = None) -> str:
    if not rank:
        return factors_text(rank)
    if not rank.get("gate_passed"):
        return factors_text(rank)
    hs = set(held or [])
    lines = [f"🧬 *Alpha 脊椎排名*（{rank.get('as_of')}｜前 {n}／{rank.get('n_universe')} 檔）"]
    for r in (rank.get("top") or [])[:n]:
        f = r.get("f") or {}
        bits = []
        for k in ("mom_12_1", "rev_1m", "ext_atr", "vol_60", "quality", "rev"):
            if k in f:
                bits.append(f"{FACTOR_LABELS.get(k, k).replace('_', '·')} {f[k]:+.2f}")
        tag = " 📌" if r["t"] in hs else ""
        lines.append(f"{r['rank']:>2}. *{r['t']}* {r['score']:+.2f}（信心 {r.get('conf', 0):.0%}）{tag}"
                     + (f"\n　　{'｜'.join(bits[:4])}" if bits else ""))
    w = {k: v["weight"] for k, v in (rank.get("factors") or {}).items() if v.get("weight")}
    lines.append("權重：" + "、".join(f"{FACTOR_LABELS.get(k, k).replace('_', '·')} {v:.0%}" for k, v in w.items()))
    lines.append("_排名＝配置候選順序，不是買進訊號；進場仍走引擎評分與濾網。非投資建議_")
    return "\n".join(lines)


# ── 自我測試（合成宇宙：植入「vol_60 有預測力」，其餘雜訊）────────────────────

if __name__ == "__main__":
    import tempfile
    rng = np.random.default_rng(5)
    n_sym, n_day = 60, 420
    idx = pd.bdate_range("2024-09-02", periods=n_day)
    data = {}
    sigmas = rng.uniform(0.008, 0.03, n_sym)
    # 低波動股植入正漂移（→ vol_60 因子「高＝好」應有正 IC）；動能因子無植入
    for i in range(n_sym):
        drift = 0.0012 * (0.03 - sigmas[i]) / 0.022 - 0.0002
        r = rng.normal(drift, sigmas[i], n_day)
        c = 100 * np.cumprod(1 + r)
        df = pd.DataFrame({"Open": c * (1 + rng.normal(0, 0.002, n_day)), "High": c * 1.01, "Low": c * 0.99,
                           "Close": c, "Volume": rng.integers(1e5, 1e6, n_day).astype(float)}, index=idx)
        data[f"S{i:02d}"] = df
    frames = wide_frames(data)
    assert frames["close"].shape == (n_day, n_sym)
    pf = price_factors(frames)
    assert set(pf) == set(PRICE_FACTORS) and pf["mom_12_1"].iloc[-1].notna().sum() == n_sym
    print("✅ 1 寬表與價格因子形狀")

    # 2) 無前視：竄改最後 30 根 K 棒不改變之前日期的因子值
    data2 = {k: v.copy() for k, v in data.items()}
    for k in data2:
        data2[k].iloc[-30:, :4] *= 1.3
    pf2 = price_factors(wide_frames(data2))
    cut = idx[-31]
    for name in PRICE_FACTORS:
        a, b = pf[name].loc[:cut], pf2[name].loc[:cut]
        assert np.allclose(a.fillna(-9).values, b.fillna(-9).values), name
    print("✅ 2 價格因子無前視")

    dates = sample_dates(frames["close"].index)
    assert dates and max(dates) == str(idx[-HORIZON - 2].date())
    import factor_eval as _fe
    assert _fe.forward_returns({"S00": frames["close"]["S00"]}, max(dates), HORIZON)          # 最後快照有前瞻報酬
    assert len(dates) >= 40
    rank = build_rank(frames, dates)
    assert rank["gate_passed"] and rank["factors"]["vol_60"]["pass"], rank["factors"]["vol_60"]
    assert rank["factors"]["vol_60"]["ic"] > 0.03
    assert rank["top"] and rank["top"][0]["rank"] == 1 and 0 < rank["top"][0]["conf"] <= 1
    assert abs(sum(v["weight"] for v in rank["factors"].values()) - 1.0) < 1e-9
    assert rank["market"]["breadth_pct"] is not None
    print(f"✅ 3 閘門與合成（vol_60 IC {rank['factors']['vol_60']['ic']:+.3f}、通過 {[k for k, v in rank['factors'].items() if v['pass']]}、"
          f"合成 IC {rank['composite'].get('ic')}）")

    # 4) 基本面 PIT：quality_by_date 只用 available_at ≤ 日期的期別；rev_score_pit 只用 d ≤ as_of 的列
    store = {"ticker": "S00", "periods": [
        {"period_end": "2024-12-31", "available_at": "2025-03-15", "freq": "A", "revenue": 100, "net_income": 10, "cfo": 12,
         "total_assets": 200, "total_equity": 100, "total_debt": 20, "cash_and_sti": 10, "operating_income": 15, "ebit": 15,
         "shares_out": 10, "diluted_shares": 10, "receivables": 10, "inventory": 5, "payables": 8, "pretax_income": 14, "tax_provision": 3,
         "gross_profit": 40, "capex": -5, "da": 4},
        {"period_end": "2023-12-31", "available_at": "2024-03-15", "freq": "A", "revenue": 90, "net_income": 8, "cfo": 10,
         "total_assets": 180, "total_equity": 90, "total_debt": 25, "cash_and_sti": 8, "operating_income": 12, "ebit": 12,
         "shares_out": 10, "diluted_shares": 10, "receivables": 9, "inventory": 5, "payables": 7, "pretax_income": 11, "tax_provision": 2,
         "gross_profit": 35, "capex": -5, "da": 4},
        {"period_end": "2022-12-31", "available_at": "2023-03-15", "freq": "A", "revenue": 80, "net_income": 6, "cfo": 8,
         "total_assets": 170, "total_equity": 80, "total_debt": 30, "cash_and_sti": 6, "operating_income": 10, "ebit": 10,
         "shares_out": 10, "diluted_shares": 10, "receivables": 9, "inventory": 5, "payables": 7, "pretax_income": 9, "tax_provision": 2,
         "gross_profit": 30, "capex": -5, "da": 4}]}
    q = quality_by_date({"S00": store}, ["2025-03-01", "2025-04-01"])
    assert "S00" in q.get("2025-04-01", {}) and "S00" in q.get("2025-03-01", {})        # 3/1 只有兩期仍可算
    q1 = quality_by_date({"S00": {"ticker": "S00", "periods": store["periods"][:1]}}, ["2025-04-01"])
    assert not q1                                                                        # 只有 1 期 → 不算
    rows = [["2025-01-05", 5.0, 6.0, None, None, 2, 1, 10], ["2025-02-02", 5.1, 6.1, None, None, 3, 1, 10],
            ["2025-04-06", 5.5, 6.2, None, None, 4, 0, 10], ["2025-05-04", 5.6, 6.3, None, None, 4, 0, 10]]
    r_early = rev_score_pit(rows, "2025-02-15")
    r_late = rev_score_pit(rows, "2025-05-10")
    assert r_early is not None and r_late is not None and r_late > r_early                # 上修後分數上升
    assert rev_score_pit(rows, "2024-12-01") is None                                     # 尚無列
    print("✅ 4 基本面 PIT（quality 期別可得性、rev 列日期）")

    # 5) 覆蓋不足的因子不配權；無因子通過 → gate False、top 空、候選池空
    fund = {"quality": {d: {"S00": 0.5, "S01": -0.2} for d in dates}}
    rank2 = build_rank(frames, dates, fund_fbd=fund)
    assert rank2["factors"]["quality"]["pass"] is False and rank2["factors"]["quality"]["weight"] == 0.0
    assert "覆蓋不足" in rank2["factors"]["quality"]["reason"]
    noise = {k: v.copy() for k, v in data.items()}
    for k in noise:
        r = rng.normal(0, 0.02, n_day); c = 100 * np.cumprod(1 + r)
        noise[k]["Close"] = c; noise[k]["High"] = c * 1.01; noise[k]["Low"] = c * 0.99
    rank3 = build_rank(wide_frames(noise), dates)
    assert pool_symbols(rank3, 10) == [] or rank3["gate_passed"]                        # 雜訊：閘門多半不過 → 候選池空
    assert pool_symbols(rank, 5, exclude=[rank["top"][0]["t"]])[0] == rank["top"][1]["t"]
    print("✅ 5 覆蓋不足/雜訊 → 不配權、候選池為空；exclude 生效")

    # 6) 落檔/讀回 + 文字輸出（Markdown 安全）
    with tempfile.TemporaryDirectory() as td:
        p = save_rank(rank, Path(td) / "rank.json")
        back = load_rank(p)
        assert back and back["as_of"] == rank["as_of"] and len(back["top"]) == len(rank["top"])
    for txt in (factors_text(rank), rank_text(rank, 5), factors_text(None), rank_text(rank3)):
        assert "**" not in txt and txt
    print(rank_text(rank, 5))
    print("\nalpha_spine selftest OK ✅")
