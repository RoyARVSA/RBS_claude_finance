"""
value_research.py – VRT 方法（company_model）的長歷史、當時可得回測（衛星「選股」路線：基本面那一塊）

問題：「上次分析 VRT 那套，其他股票也能用吧？能不能贏大盤？」——company_model 已泛化到任何代碼，但只在
觀察清單建模、估值歷史從 2026-09 才開始，從沒被測過。這裡用 SEC EDGAR 年報的「當時可得」版本（edgar_pit）
重建過去每一份 10-K 公布當下的機械化模型，回答：
  1. MoS（安全邊際＝模型合理價／市價 − 1）能不能挑出下個月表現較好的股票（橫斷面 Rank IC、分時段）
  2. MoS 前 20% 等權（扣成本）vs SPY／vs 成分等權；模型判定「accumulate」的股票組合 vs SPY
  3. 對照組：經典價值／品質因子（盈餘殖利率、自由現金流殖利率、淨值市價比、ROIC、毛利／資產、應計、資產成長）
     ——同一份 PIT 財報算，分辨「是 VRT 模型有用」還是「只是便宜股票」

機械化版本與正式 /model 的差別（誠實揭露）：沒有歷史分析師預估（驅動只用歷史財報推導）、不跑蒙地卡羅、
無人工覆蓋；β 用模型日前 2 年週報酬、無風險利率用當時 10 年期公債殖利率（^TNX）、產業路由用現行 GICS
（金融→RIM、不動產→DDM；舊年份改過分類的公司有些許前視）。
紀律：模型日＝10-K 申報日的次一交易日（申報當天收盤前不可見）；每月底用「最新一份已公布模型」算 MoS；
次一交易日開盤成交（含息報酬）、單邊成本 0.05%×換手（沿用 factor_research）。
存活偏誤：已下市公司沒有價格、多數也對不到 CIK → 舊年份樣本偏向倖存者（報告揭露覆蓋率）。教育用途，非投資建議。
"""

from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd

import factor_research as fr

VALUE_LABELS = {"mos": "VRT 模型 MoS（安全邊際）", "ey": "盈餘殖利率 E/P", "fcfy": "自由現金流殖利率", "bm": "淨值市價比 B/M",
                "roic": "ROIC", "gpa": "毛利／總資產", "accruals": "應計（淨利−營業現金流）／資產", "asset_growth": "總資產成長"}
SECTOR_ROUTE = {40: "Financial Services", 60: "Real Estate"}       # GICS 前兩碼 → company_model 產業路由名


# ── 1. 價格與拆股（auto_adjust=False 才拿得到拆股欄位）──────────────────────────

def fetch_prices(tickers: list[str], start: str, chunk: int = 120, pause: float = 1.0) -> dict:
    """{sym: DataFrame[Open, Close, AdjClose, Splits]}：Close 已還原拆股、未還原股息；AdjClose 含息。"""
    import yfinance as yf
    out = {}
    tickers = sorted(set(tickers))
    for i in range(0, len(tickers), chunk):
        part = tickers[i:i + chunk]
        try:
            raw = yf.download(part, start=start, auto_adjust=False, actions=True, progress=False,
                              group_by="column", threads=True)
        except Exception as e:
            print(f"value_research: 抓價失敗（{i}）{type(e).__name__}")
            continue
        if raw is None or raw.empty:
            continue
        multi = isinstance(raw.columns, pd.MultiIndex)
        for s in part:
            try:
                g = (lambda f: raw[(f, s)]) if multi else (lambda f: raw[f])
                df = pd.DataFrame({"Open": g("Open"), "Close": g("Close"), "AdjClose": g("Adj Close"),
                                   "Splits": g("Stock Splits") if (("Stock Splits", s) in raw.columns if multi else "Stock Splits" in raw) else 0.0})
                df = df.dropna(subset=["Close"])
                df = df[df["Close"] > 0]
                df["Open"] = df["Open"].where((df["Open"] > 0) & (df["Open"] / df["Close"]).between(0.5, 2.0))
                if len(df) >= 260:
                    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
                    out[s] = df
            except Exception:
                continue
        time.sleep(pause)
    return out


def split_after(splits: pd.Series, t: pd.Timestamp) -> float:
    """t 之後（不含）所有拆股比率的乘積 F：還原後價格＝原始價格／F；今天基準的股數＝原始股數×F。"""
    s = splits[(splits.index > t) & (splits > 0)]
    return float(np.prod(s.values)) if len(s) else 1.0


def total_return_open(df: pd.DataFrame) -> pd.Series:
    """含息的開盤價序列（Open × AdjClose／Close）：持有期報酬含股息。"""
    return df["Open"] * (df["AdjClose"] / df["Close"])


def weekly_returns(close: pd.Series) -> pd.Series:
    """週五標記的週報酬（整段算一次；beta_at 只取「標記日 ≤ t」＝ t 以前已收完的週）。"""
    return close.resample("W-FRI").last().pct_change(fill_method=None)


def beta_at(stock: pd.Series, spy: pd.Series, t: pd.Timestamp, weeks: int = 104,
            wa: pd.Series | None = None, wb: pd.Series | None = None) -> float | None:
    """t 以前已收完的 weeks 週週報酬 β（樣本 < 52 週回 None）。wa/wb 可傳預先算好的 weekly_returns（快）。"""
    a = weekly_returns(stock) if wa is None else wa
    b = weekly_returns(spy) if wb is None else wb
    df = pd.concat([a[a.index <= t], b[b.index <= t]], axis=1).dropna().iloc[-weeks:]
    if len(df) < 52 or float(df.iloc[:, 1].var()) <= 0:
        return None
    return float(df.iloc[:, 0].cov(df.iloc[:, 1]) / df.iloc[:, 1].var())


# ── 2. 每份 10-K 公布後重建模型 ───────────────────────────────────────────────

SHARE_FIELDS = ("diluted_shares", "shares_out", "cover_shares")
MODEL_MAX_AGE = 460              # 天：超過 15 個月沒有新年報 → 模型失效


def to_basis(periods: list[dict], splits: pd.Series, t: pd.Timestamp) -> tuple[list[dict], float]:
    """股數類科目換到 t 的拆股基準：原始股數 × F(該科目申報日)／F(t)。
    逐科目合併時股數可能取自較舊的申報（之後沒重報）→ 不能一律用最新申報日的 F（#89 Med-1）。"""
    Ft = split_after(splits, t)
    out = []
    for p in periods:
        q = dict(p)
        fo = p.get("_filed") or {}
        for f in SHARE_FIELDS:
            v, fl = q.get(f), fo.get(f)
            if v and fl:
                q[f] = v * split_after(splits, pd.Timestamp(fl)) / Ft
        out.append(q)
    return out, Ft


def _scaled(x, F):
    return float(x) / F if x is not None and math.isfinite(float(x)) else None


def build_models(company: dict, px: pd.DataFrame, spy_close: pd.Series, rf_series: pd.Series | None,
                 sector: str | None, ticker: str, start: str, spy_wk: pd.Series | None = None) -> list[dict]:
    """每份新年報一筆：{date（模型日）, pe（最新期末）, bear/base/bull（今天拆股基準的每股價值）, audit, …}。
    判定（accumulate…）不在這裡定案——每個訊號日用當天價格重跑 company_model.signal（#89 Med-4）。"""
    import company_model as cm
    import edgar_pit as ep
    idx = px.index
    sp = px["Splits"][px["Splits"] > 0]
    wa, wb = weekly_returns(px["Close"]), (spy_wk if spy_wk is not None else weekly_returns(spy_close))
    filed = sorted({r[4] for r in company.get("rows") or []})
    out = []
    for f in filed:
        if f < (pd.Timestamp(start) - pd.Timedelta(days=400)).strftime("%Y-%m-%d"):
            continue
        pos = idx.searchsorted(pd.Timestamp(f), side="right")      # 申報日之後第一個交易日
        if pos >= len(idx):
            break
        md = idx[pos]
        periods = ep.pit_periods(company, f)
        if not periods:
            continue
        periods, F = to_basis(periods, sp, md)
        price_raw = float(px["Close"].iloc[pos]) * F
        shares = periods[0].get("diluted_shares") or periods[0].get("shares_out")
        if not shares or shares <= 0 or price_raw <= 0:
            continue
        rf = None
        if rf_series is not None and len(rf_series):
            r_ = rf_series[rf_series.index <= md]
            rf = float(r_.iloc[-1]) / 100 if len(r_) else None
        prof = {"ticker": ticker, "price": price_raw, "mkt_cap": price_raw * shares, "shares": shares,
                "beta": beta_at(px["Close"], spy_close, md, wa=wa, wb=wb) or 1.0, "sector": sector}
        try:
            res = cm.run_model(periods, prof, cfg=({"rf": rf} if rf else None), mc=False)
        except Exception:
            continue
        sc, sig = res.get("scenarios") or {}, res.get("signal") or {}
        base = sc.get("base")
        if base is None or not math.isfinite(float(base)) or base <= 0:
            continue
        out.append({"date": md, "pe": periods[0].get("period_end"), "base": float(base) / F,
                    "bear": _scaled(sc.get("bear"), F), "bull": _scaled(sc.get("bull"), F),
                    "fair_adj": float(base) / F, "audit": bool((res.get("audit") or {}).get("passed", True)),
                    "method": res.get("method"), "verdict0": sig.get("verdict"), "mos0": sig.get("mos")})
    return out


def _broken(pe, d: pd.Timestamp, brk: list[pd.Timestamp]) -> bool:
    """最新期末在斷點之前、且斷點已到（pe < b ≤ d）→ 這份財報不代表現況（分拆／代碼重複使用，#89 Med-2/3）。
    用期末不用申報日：分拆後才申報、但期末在分拆前的年報仍含被分拆事業。"""
    if not brk or not pe:
        return False
    p = pd.Timestamp(pe)
    return any(p < b <= d for b in brk)


def mos_frame(models: dict, close_adj: pd.DataFrame, dates: pd.DatetimeIndex,
              breaks: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """每個訊號日：最新一份「模型日 ≤ 訊號日」的模型 → MoS＝base／收盤 − 1；
    accumulate 用當天收盤重跑 company_model.signal（不是凍結在模型日的判定）。
    模型超過 MODEL_MAX_AGE 天、或遇到斷點 → NaN／False。"""
    import company_model as cm
    mos, acc = {}, {}
    for sym, recs in models.items():
        if not recs or sym not in close_adj:
            continue
        md = pd.DatetimeIndex([r["date"] for r in recs])
        pos = md.searchsorted(dates, side="right") - 1
        c = close_adj[sym].reindex(dates).values
        brk = [pd.Timestamp(b) for b in (breaks or {}).get(sym, ())]
        m = np.full(len(dates), np.nan)
        a = np.zeros(len(dates), dtype=bool)
        for i, d in enumerate(dates):
            j = pos[i]
            if j < 0 or not (np.isfinite(c[i]) and c[i] > 0):
                continue
            r = recs[j]
            if (d - r["date"]).days > MODEL_MAX_AGE or _broken(r.get("pe"), d, brk):
                continue
            m[i] = r["base"] / c[i] - 1
            sig = cm.signal({"bear": r.get("bear"), "base": r["base"], "bull": r.get("bull")}, float(c[i]),
                            None, r.get("audit", True))
            a[i] = sig.get("verdict") == "accumulate"
        mos[sym], acc[sym] = m, a
    return pd.DataFrame(mos, index=dates), pd.DataFrame(acc, index=dates).astype(bool)


# ── 3. 經典價值／品質因子（同一份 PIT 財報）────────────────────────────────────

def classic_factors(companies: dict, prices: dict, dates: pd.DatetimeIndex, close_adj: pd.DataFrame | None = None,
                    breaks: dict | None = None) -> dict:
    """每個訊號日、每檔：最新一份「申報日 < 訊號日」的年報（與 MoS 的模型日規則一致）→ E/P、FCF/P、B/M、ROIC、
    GP/A、應計、資產成長。市值＝訊號日還原收盤 × 原始股數 × F(股數那筆的申報日)（拆股前後一致）。
    訊號日沒有價格、年報超過 MODEL_MAX_AGE 天、遇到斷點 → NaN（不沿用舊價、不沿用失效財報）。"""
    import bisect
    import edgar_pit as ep
    keys = list(VALUE_LABELS)[1:]
    out = {k: {} for k in keys}
    for sym, comp in companies.items():
        px = prices.get(sym)
        if px is None:
            continue
        sp = px["Splits"][px["Splits"] > 0]
        c = (close_adj[sym] if close_adj is not None and sym in close_adj else px["Close"]).reindex(dates).values
        fl_sorted = sorted({r[4] for r in comp.get("rows") or []})
        brk = [pd.Timestamp(b) for b in (breaks or {}).get(sym, ())]
        vals = {k: [np.nan] * len(dates) for k in keys}
        cache: dict[str, list] = {}
        for i, d in enumerate(dates):
            j = bisect.bisect_left(fl_sorted, d.strftime("%Y-%m-%d")) - 1     # 申報日 < 訊號日
            if j < 0 or not (np.isfinite(c[i]) and c[i] > 0):
                continue
            lf = fl_sorted[j]
            if (d - pd.Timestamp(lf)).days > MODEL_MAX_AGE:
                continue
            if lf not in cache:
                cache[lf] = ep.pit_periods(comp, lf, max_periods=2)
            ps = cache[lf]
            if not ps or _broken(ps[0].get("period_end"), d, brk):
                continue
            p0 = ps[0]
            shf = "diluted_shares" if p0.get("diluted_shares") else "shares_out"
            sh = p0.get(shf)
            if not sh or sh <= 0:
                continue
            fsh = (p0.get("_filed") or {}).get(shf) or lf
            mcap = float(c[i]) * sh * split_after(sp, pd.Timestamp(fsh))
            ni, cfo, fcf = p0.get("net_income"), p0.get("cfo"), p0.get("fcf")
            eq, ta = p0.get("total_equity"), p0.get("total_assets")
            ebit, tax, pti = p0.get("operating_income"), p0.get("tax_provision"), p0.get("pretax_income")
            debt, cash = p0.get("total_debt") or 0.0, p0.get("cash_and_sti") or p0.get("cash") or 0.0
            tr = min(max(tax / pti, 0.0), 0.4) if (tax is not None and pti and pti > 0) else 0.21
            ic = (eq or 0.0) + debt - cash
            ta_prev = ps[1].get("total_assets") if len(ps) > 1 else None

            def _r(a, b):
                return (a / b) if (a is not None and b and b > 0) else np.nan
            vals["ey"][i] = _r(ni, mcap)
            vals["fcfy"][i] = _r(fcf, mcap)
            vals["bm"][i] = _r(eq, mcap)
            vals["roic"][i] = _r(ebit * (1 - tr) if ebit is not None else None, ic)
            vals["gpa"][i] = _r(p0.get("gross_profit"), ta)
            vals["accruals"][i] = _r((ni - cfo) if (ni is not None and cfo is not None) else None, ta)
            vals["asset_growth"][i] = (ta / ta_prev - 1) if (ta and ta_prev and ta_prev > 0) else np.nan
        for k in keys:
            out[k][sym] = vals[k]
    return {k: pd.DataFrame(v, index=dates) for k, v in out.items()}


# ── 4. 文字 ──────────────────────────────────────────────────────────────────

def value_text(study: dict, acc_res: dict, cov: pd.DataFrame, meta: dict) -> str:
    labels = dict(fr.FACTOR_LABELS)
    fr.FACTOR_LABELS.update(VALUE_LABELS)                  # 共用 factor_text 的排版
    try:
        body = fr.factor_text(study, None, meta)
    finally:
        fr.FACTOR_LABELS.clear()
        fr.FACTOR_LABELS.update(labels)
    body = body.replace("🔬 *選股因子研究*", "🏛 *VRT 方法（公司模型）與價值／品質因子回測*", 1)
    lines = body.split("\n")
    head, rest = lines[:1], lines[1:]
    cov_lines = ["*資料覆蓋*：每期「當時成分」中有價格且有可用模型（近 15 個月內有年報）的比例"]
    for a, b, lab in fr.SUBPERIODS:
        s = cov[(cov.index >= pd.Timestamp(a)) & (cov.index <= pd.Timestamp(b))]
        if len(s):
            cov_lines.append(f"・{lab}：平均 {s['members'].mean():.0f} 檔成分、有模型 {s['modeled'].mean():.0f} 檔"
                             f"（{(s['modeled'] / s['members']).mean():.0%}）")
    acc_lines = []
    if acc_res.get("perf"):
        vs = acc_res["vs_spy"]
        acc_lines = [f"\n*模型判定 accumulate 的股票等權持有*（每月用當月價格重判；平均 {acc_res.get('avg_n', 0):.0f} 檔、"
                     f"{acc_res.get('cash_months', 0)} 個月不足 5 檔空手）：年化 {fr._p(acc_res['perf'].get('cagr'))}｜"
                     f"vs SPY {fr._p(vs.get('cagr_diff'))}（IR {fr._f(vs.get('ir'))}、贏的年份 {fr._p(vs.get('win_years'), 0, False)}）｜"
                     f"回撤 {fr._p(acc_res['perf'].get('max_dd'), 0, False)}"]
    tail = ["⚠️ 機械化版本：沒有歷史分析師預估、不跑蒙地卡羅、無人工覆蓋；產業路由用現行 GICS。"
            "分拆與代碼重複使用：期末在斷點前的年報於斷點後失效；改名代碼併入現行代碼。"
            "已下市公司缺價格與 CIK → 舊年份偏向倖存者。非投資建議"]
    return "\n".join(head + cov_lines + rest[:-1] + acc_lines + tail)


def accumulate_portfolio(A: pd.DataFrame, R: pd.DataFrame, M: pd.DataFrame, spy_r: pd.Series) -> dict:
    """每期持有「當月判定 accumulate 且為當期成分」的股票等權（扣換手成本）。
    不足 5 檔的月份＝空手（報酬 0、照樣和 SPY 比），之後重新進場算全額換手（#89 Low）。"""
    A, R = A.align(R, join="inner", axis=None)
    ok = A.fillna(False).astype(bool) & M.reindex_like(A).fillna(False).astype(bool)
    rows, prev, cash = [], set(), 0
    for d in A.index:
        names = set(A.columns[ok.loc[d].values])
        rr = R.loc[d, list(names)] if names else pd.Series(dtype=float)
        if len(names) < 5 or rr.notna().sum() == 0:
            rows.append({"date": d, "net": -fr.COST_SIDE if prev else 0.0, "n": 0})   # 出清的單邊成本
            prev = set()
            cash += 1
            continue
        to = 1.0 if not prev else len(names - prev) / len(names)
        rows.append({"date": d, "net": float(rr.mean()) - 2 * fr.COST_SIDE * to, "n": len(names)})
        prev = names
    if not rows or not any(r["n"] for r in rows):
        return {}
    df = pd.DataFrame(rows).set_index("date")
    inv = df[df["n"] > 0]
    return {"perf": fr.perf(df["net"]), "vs_spy": fr._excess(df["net"], spy_r), "avg_n": float(inv["n"].mean()),
            "cash_months": int(cash), "net": df["net"]}


# ── 5. 進入點 ────────────────────────────────────────────────────────────────

def sector_map() -> dict:
    """現行 GICS（data/gics/members.json）→ company_model 產業路由名（金融／不動產）。"""
    import json
    from pathlib import Path
    p = Path(__file__).resolve().parent / "data" / "gics" / "members.json"
    try:
        tk = json.loads(p.read_text()).get("tickers") or {}
        return {t: SECTOR_ROUTE.get(int(str(c)[:2])) for t, c in tk.items() if str(c)[:2].isdigit()}
    except Exception:
        return {}


def merge_renames(periods: dict) -> dict:
    """改名的舊代碼（FB、HCP…）成分期間併入現行代碼：Yahoo 的歷史跟著證券走到新代碼，舊代碼多半抓不到、
    或已被別家公司拿去用（#89 Low）。"""
    import edgar_pit as ep
    out: dict[str, list] = {}
    for t, ps in (periods or {}).items():
        out.setdefault(ep.canonical(t), []).extend(ps)
    return {t: sorted(ps) for t, ps in out.items()}


def run_value(start: str = "2011-01-01", end: str | None = None, q: float = 0.2, store: dict | None = None,
              periods: dict | None = None, price_fn=None, sectors: dict | None = None) -> dict:
    import edgar_pit as ep
    store = store if store is not None else ep.load()
    if not store or not store.get("companies"):
        return {"text": "❌ 還沒有 EDGAR 歷史財報檔（data/edgar/annual_pit.json.gz）——請先跑 Colab Cell 4 並上傳"}
    if periods is None:
        import universe as un
        periods = un.fetch_sp500_periods()
    if not periods:
        return {"text": "❌ S&P 500 成分歷史抓取失敗，稍後再試"}
    periods = merge_renames(periods)
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    t2c = ep.ticker_to_company(store)
    members = sorted(t for t, ps in periods.items() if any(a <= end and b >= start for a, b in ps))
    tickers = [t for t in members if t in t2c]
    warm = (pd.Timestamp(start) - pd.Timedelta(days=800)).strftime("%Y-%m-%d")
    prices = (price_fn or fetch_prices)(tickers + ["SPY", "^TNX"], warm)
    if "SPY" not in prices:
        return {"text": "❌ SPY 抓不到，無法建立交易日曆"}
    cal = prices["SPY"].index
    cal = cal[cal <= pd.Timestamp(end)]
    spy_close = prices["SPY"]["Close"]
    spy_wk = weekly_returns(spy_close)
    rf = prices["^TNX"]["Close"] if "^TNX" in prices else None
    sectors = sectors if sectors is not None else sector_map()
    have = [t for t in tickers if t in prices]
    breaks = {t: ep.break_dates(t) for t in have}
    models = {t: build_models(t2c[t], prices[t], spy_close, rf, sectors.get(t), t, start, spy_wk=spy_wk) for t in have}
    dates = fr.month_ends(cal)
    dates = dates[dates >= pd.Timestamp(start)]
    close_adj = pd.DataFrame({t: prices[t]["Close"] for t in have}).reindex(cal)
    open_tr = pd.DataFrame({t: total_return_open(prices[t]) for t in have}).reindex(cal)
    R = fr.period_returns(open_tr, dates)
    if R.empty:
        return {"text": f"❌ {start}→{end} 期間內沒有完整的月份可研究"}
    M = fr.membership_matrix(periods, R.index, have)
    for t in have:                                         # 代碼重複使用：早期期間是別家公司 → 遮掉
        bad = [not ep.reuse_ok(t, str(d.date())) for d in M.index]
        if any(bad):
            M.loc[M.index[bad], t] = False
    MOS, ACC = mos_frame(models, close_adj, R.index, breaks)
    F = {"mos": MOS}
    F.update(classic_factors({t: t2c[t] for t in have}, prices, R.index, close_adj, breaks))
    spy_r = fr.period_returns(total_return_open(prices["SPY"]).reindex(cal).to_frame("SPY"), dates)["SPY"]
    ew_r = R.where(M).mean(axis=1)
    study = fr.factor_study({k: v.reindex(R.index) for k, v in F.items()}, R, M, spy_r, ew_r, q)
    allM = fr.membership_matrix(periods, R.index, members)
    cov = pd.DataFrame({"members": allM.sum(axis=1), "modeled": (M & MOS.reindex_like(M).notna()).sum(axis=1)})
    acc_res = accumulate_portfolio(ACC, R, M, spy_r)
    meta = {"start": str(R.index[0].date()), "end": str(R.index[-1].date()), "q": q, "spy": fr.perf(spy_r), "ew": fr.perf(ew_r)}
    n_models = sum(len(v) for v in models.values())
    n_brk = sum(1 for t in have if breaks.get(t))
    text = (value_text(study, acc_res, cov, meta) + f"\n（共重建 {n_models:,} 份年報模型、{sum(1 for v in models.values() if v)} 家公司"
            + (f"；{n_brk} 檔有分拆／代碼重複使用斷點" if n_brk else "") + "）")
    return {"study": study, "coverage": cov, "acc": acc_res, "meta": meta, "models": models, "members": M,
            "factors": F, "text": text}


# ── 6. 自我測試（合成 EDGAR＋價格；離線）────────────────────────────────────────

def _synthetic(n_co: int = 60, seed: int = 4, edge: bool = True):
    """合成：每家公司有固定「真實價值」成長；價格圍繞價值隨機遊走並均值回歸 → 便宜（MoS 高）的下月表現較好。"""
    import edgar_pit as ep
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2008-01-02", "2019-12-31")
    tpos = {t: i for i, t in enumerate(ep.TAGS)}
    companies, prices, per = {}, {}, {}
    for i in range(n_co):
        tk = f"C{i:03d}"
        g = rng.uniform(0.0, 0.08)
        rev0 = rng.uniform(1e9, 5e9)
        shares = 1e8
        rows = []
        for y in range(2008, 2020):
            rev = rev0 * (1 + g) ** (y - 2008)
            end, start = f"{y}-12-31", f"{y}-01-01"
            filed = f"{y + 1}-02-20"
            vals = {"Revenues": rev, "OperatingIncomeLoss": rev * 0.18, "IncomeTaxExpenseBenefit": rev * 0.035,
                    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest": rev * 0.17,
                    "NetIncomeLoss": rev * 0.13, "NetCashProvidedByUsedInOperatingActivities": rev * 0.2,
                    "PaymentsToAcquirePropertyPlantAndEquipment": rev * 0.05, "DepreciationDepletionAndAmortization": rev * 0.04,
                    "ShareBasedCompensation": rev * 0.01, "GrossProfit": rev * 0.4,
                    "WeightedAverageNumberOfDilutedSharesOutstanding": shares}
            for tag, v in vals.items():
                rows.append([tpos[tag], end, start, v, filed, "10-K"])
            bs = {"Assets": rev * 1.2, "AssetsCurrent": rev * 0.5, "LiabilitiesCurrent": rev * 0.3, "StockholdersEquity": rev * 0.6,
                  "CashAndCashEquivalentsAtCarryingValue": rev * 0.1, "LongTermDebtNoncurrent": rev * 0.2,
                  "AccountsReceivableNetCurrent": rev * 0.12, "InventoryNet": rev * 0.1, "AccountsPayableCurrent": rev * 0.08}
            for tag, v in bs.items():
                rows.append([tpos[tag], end, None, v, filed, "10-K"])
        companies[str(i)] = {"cik": i, "name": tk, "tickers": [tk], "rows": rows}
        val = rev0 * 0.13 * 15 / shares * (1 + g) ** (np.arange(len(idx)) / 252)
        dev = np.zeros(len(idx))
        for k in range(1, len(idx)):
            dev[k] = dev[k - 1] * (0.995 if edge else 1.0) + rng.normal(0, 0.012)
        close = val * np.exp(dev)
        op = close * (1 + rng.normal(0, 0.002, len(idx)))
        prices[tk] = pd.DataFrame({"Open": op, "Close": close, "AdjClose": close, "Splits": 0.0}, index=idx)
        per[tk] = [("2000-01-01", "9999-12-31")]
    spy = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, len(idx)))
    prices["SPY"] = pd.DataFrame({"Open": spy, "Close": spy, "AdjClose": spy, "Splits": 0.0}, index=idx)
    prices["^TNX"] = pd.DataFrame({"Open": 3.0, "Close": 3.0, "AdjClose": 3.0, "Splits": 0.0}, index=idx)
    return {"version": 1, "tags": ep.TAGS, "companies": companies}, prices, per


if __name__ == "__main__":
    # 1) 拆股換算：拆股前後市值一致、合理價換到今天基準
    sp = pd.Series([0.0, 2.0, 0.0], index=pd.DatetimeIndex(["2020-01-02", "2021-06-01", "2022-01-03"]))
    assert split_after(sp, pd.Timestamp("2020-06-01")) == 2.0 and split_after(sp, pd.Timestamp("2021-06-01")) == 1.0
    print("✅ 1 拆股因子（只計 t 之後的拆股）")

    # 2) 合成：價格對價值均值回歸 → MoS IC 顯著為正、前 20% 勝後 20%；文字可產出
    store, prices, per = _synthetic(edge=True)
    pf = lambda syms, start: {s: prices[s] for s in syms if s in prices}   # noqa: E731
    out = run_value("2011-01-01", "2019-06-30", store=store, periods=per, price_fn=pf, sectors={})
    ic = out["study"]["mos"]["ic"]
    assert ic["mean"] > 0.05 and ic["t_nw"] > 3, ic
    assert out["study"]["mos"]["top"]["cagr"] > out["study"]["mos"]["bot"]["cagr"]
    assert out["coverage"]["modeled"].min() > 30 and sum(len(v) for v in out["models"].values()) > 300
    t = out["text"]
    assert "VRT 方法" in t and "資料覆蓋" in t and "MoS" in t and t.count("*") % 2 == 0 and "_" not in t, t[:400]
    print(f"✅ 2 合成（價格均值回歸到價值）：MoS IC {ic['mean']:+.3f}（t {ic['t_nw']:+.1f}）、前 20% 勝後 20%")

    # 3) 無前視：模型日 = 申報日次一交易日；申報前的月份沒有 MoS
    m0 = out["models"]["C000"][0]
    ix0 = prices["C000"].index
    filed0 = pd.Timestamp(f"{m0['date'].year}-02-20")
    assert m0["date"] == ix0[ix0.searchsorted(filed0, side="right")], m0      # 模型日＝申報日的次一交易日
    before = ix0[ix0.searchsorted(m0["date"]) - 1]
    mos_c0 = mos_frame({"C000": out["models"]["C000"]}, pd.DataFrame({"C000": prices["C000"]["Close"]}),
                       pd.DatetimeIndex([before, m0["date"]]))[0]["C000"]
    assert np.isnan(mos_c0.iloc[0]) and np.isfinite(mos_c0.iloc[1]), mos_c0  # 模型日前一天看不到這份模型
    # 沒有 EDGAR 檔 → 明確訊息
    assert run_value(store={"companies": {}})["text"].startswith("❌")
    print("✅ 3 無前視（申報前無模型）、缺檔訊息")

    # 4) 代碼重複使用：TICKER_REUSE 日期前的成分期間遮掉（別家公司的價格不能接這家公司的財報）
    c5 = store["companies"]["5"]
    c5["tickers"] = ["JCI"]
    prices["JCI"] = prices.pop("C005")
    per["JCI"] = per.pop("C005")
    out4 = run_value("2011-01-01", "2019-06-30", store=store, periods=per, price_fn=pf, sectors={})
    mj = out4["members"]["JCI"]
    assert not mj[mj.index < pd.Timestamp("2016-09-02")].any() and mj[mj.index >= pd.Timestamp("2016-09-02")].all(), mj
    assert out4["members"]["C006"].all()
    # 斷點：重複使用日前的期末 → 斷點後 MoS／經典因子都失效，等下一份年報（#89 Med-2）
    mj4 = out4["factors"]["mos"]["JCI"]
    ey4 = out4["factors"]["ey"]["JCI"]
    win = (mj4.index >= pd.Timestamp("2016-09-02")) & (mj4.index < pd.Timestamp("2017-02-21"))
    assert mj4[win].isna().all() and ey4[win].isna().all(), mj4[win]
    assert mj4[mj4.index >= pd.Timestamp("2017-02-28")].notna().all()
    assert out["factors"]["mos"]["C005"][win].notna().all()           # 同一家公司沒有斷點時那段有值（非空測）
    print("✅ 4 代碼重複使用：JCI 2016-09 前遮掉、斷點後舊年報失效到下一份 10-K、其他不受影響")

    # 5) 拆股＋10-K/A（#89 Med-1）：2015-06 拆股 2:1、2015-08 的 10-K/A 只改營收 → 股數仍取自 2015-02 那份
    #    （拆股前單位）；換算要用「股數那筆的申報日」→ MoS 前後連續。舊做法（一律用最新申報日）會翻倍
    import edgar_pit as ep
    store5, prices5, per5 = _synthetic(n_co=12, seed=9)
    c10 = store5["companies"]["10"]
    tpos = {t: i for i, t in enumerate(ep.TAGS)}
    shi, rvi = tpos["WeightedAverageNumberOfDilutedSharesOutstanding"], tpos["Revenues"]
    for r in c10["rows"]:
        if r[0] == shi and r[4] <= "2015-05-31":
            r[3] = r[3] / 2                                            # 拆股前申報＝拆股前單位
    rev14 = next(r[3] for r in c10["rows"] if r[0] == rvi and r[1] == "2014-12-31")
    c10["rows"].append([rvi, "2014-12-31", "2014-01-01", rev14 * 1.001, "2015-08-14", "10-K/A"])
    prices5["C010"].loc[pd.Timestamp("2015-06-01"), "Splits"] = 2.0
    mods = build_models(c10, prices5["C010"], prices5["SPY"]["Close"], None, None, "C010", "2014-01-01")
    d5 = pd.DatetimeIndex([pd.Timestamp("2015-07-31"), pd.Timestamp("2015-08-31")])
    assert any(m["date"] == pd.Timestamp("2015-08-17") for m in mods), [m["date"] for m in mods]
    mm = mos_frame({"C010": mods}, pd.DataFrame({"C010": prices5["C010"]["Close"]}), d5)[0]["C010"]
    assert abs(mm.iloc[1] - mm.iloc[0]) < 0.15, mm                  # 舊做法差約 +1.0
    cf5 = classic_factors({"C010": c10}, prices5, d5)
    assert abs(cf5["ey"]["C010"].iloc[1] / cf5["ey"]["C010"].iloc[0] - 1) < 0.15, cf5["ey"]
    # 經典因子數值：E/P＝淨利／（訊號日收盤 × 今天基準股數）
    p0 = ep.pit_periods(c10, "2015-08-14")[0]
    exp_ey = p0["net_income"] / (float(prices5["C010"]["Close"].loc["2015-08-31"]) * 1e8)
    assert abs(cf5["ey"]["C010"].iloc[1] / exp_ey - 1) < 1e-9, (cf5["ey"]["C010"].iloc[1], exp_ey)
    print("✅ 5 拆股後的 10-K/A：股數依自己的申報日換算（MoS／E/P 連續）、E/P 數值正確")

    # 6) 判定每月重判（#89 Med-4）：base 80／bull 95——價格 50 → accumulate；漲到 100 → 不再 accumulate；
    #    模型超過 460 天失效；斷點（期末 < 斷點 ≤ 訊號日）失效
    d6 = pd.DatetimeIndex(["2020-01-31", "2020-06-30", "2021-06-30"])
    rec = [{"date": pd.Timestamp("2020-01-02"), "pe": "2019-12-31", "base": 80.0, "bear": 60.0, "bull": 95.0, "audit": True}]
    cl6 = pd.DataFrame({"X": [50.0, 100.0, 50.0]}, index=d6)
    M6, A6 = mos_frame({"X": rec}, cl6, d6)
    assert list(A6["X"]) == [True, False, False] and abs(M6["X"].iloc[0] - 0.6) < 1e-12 and np.isnan(M6["X"].iloc[2])
    M6b, _ = mos_frame({"X": rec}, cl6, d6, {"X": ["2020-03-02"]})
    assert np.isfinite(M6b["X"].iloc[0]) and np.isnan(M6b["X"].iloc[1])
    print("✅ 6 accumulate 每月用當月價格重判、460 天失效、分拆斷點失效")

    # 7) 負對照：價格與價值無關（純隨機漫步）→ MoS 不該顯著（防前視／資料洩漏讓 IC 灌水）
    store7, prices7, per7 = _synthetic(edge=False, seed=11)
    pf7 = lambda syms, start: {s: prices7[s] for s in syms if s in prices7}   # noqa: E731
    out7 = run_value("2011-01-01", "2019-06-30", store=store7, periods=per7, price_fn=pf7, sectors={})
    ic7 = out7["study"]["mos"]["ic"]
    assert abs(ic7["t_nw"]) < 2.0 and abs(ic7["mean"]) < 0.03, ic7
    print(f"✅ 7 負對照（無訊號）：MoS IC {ic7['mean']:+.3f}（t {ic7['t_nw']:+.1f}）")

    # 8) accumulate 組合：不足 5 檔＝空手（報酬 0、出清單邊成本），重新進場算全額換手
    d8 = pd.DatetimeIndex(pd.date_range("2020-01-31", periods=4, freq="ME"))
    cols = [f"S{i}" for i in range(6)]
    A8 = pd.DataFrame(True, index=d8, columns=cols)
    A8.iloc[2, 1:] = False                                            # 第 3 個月只剩 1 檔 → 空手
    R8 = pd.DataFrame(0.01, index=d8, columns=cols)
    M8 = pd.DataFrame(True, index=d8, columns=cols)
    acc8 = accumulate_portfolio(A8, R8, M8, pd.Series(0.0, index=d8))
    n8 = acc8["net"]
    assert acc8["cash_months"] == 1 and abs(n8.iloc[2] + fr.COST_SIDE) < 1e-12
    assert abs(n8.iloc[1] - 0.01) < 1e-12 and abs(n8.iloc[3] - (0.01 - 2 * fr.COST_SIDE)) < 1e-12, n8
    print("✅ 8 accumulate 空手月與重新進場成本")

    # 9) 改名併入現行代碼（FB→META、PEAK→DOC 鏈）
    mr = merge_renames({"FB": [("2013-12-23", "2022-06-08")], "META": [("2022-06-09", "9999-12-31")],
                        "HCP": [("2008-03-31", "2019-11-04")], "PEAK": [("2019-11-05", "2024-02-29")], "AAPL": [("1982-11-30", "9999-12-31")]})
    assert set(mr) == {"META", "DOC", "AAPL"} and len(mr["META"]) == 2 and len(mr["DOC"]) == 2
    assert ep.break_dates("JCI") == ["2016-09-02"] and ep.break_dates("AAPL") == []
    print("✅ 9 改名代碼併入現行代碼、斷點表")
    print("\nvalue_research selftest OK ✅")
