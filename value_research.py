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


def beta_at(stock: pd.Series, spy: pd.Series, t: pd.Timestamp, weeks: int = 104) -> float | None:
    """t 以前（含）weeks 週的週報酬 β（樣本 < 52 週回 None）。"""
    a = stock[stock.index <= t].resample("W-FRI").last().pct_change(fill_method=None)
    b = spy[spy.index <= t].resample("W-FRI").last().pct_change(fill_method=None)
    df = pd.concat([a, b], axis=1).dropna().iloc[-weeks:]
    if len(df) < 52 or float(df.iloc[:, 1].var()) <= 0:
        return None
    return float(df.iloc[:, 0].cov(df.iloc[:, 1]) / df.iloc[:, 1].var())


# ── 2. 每份 10-K 公布後重建模型 ───────────────────────────────────────────────

def build_models(company: dict, px: pd.DataFrame, spy_close: pd.Series, rf_series: pd.Series | None,
                 sector: str | None, ticker: str, start: str) -> list[dict]:
    """回 [{date（模型日）, fair_adj（今天拆股基準的每股合理價）, verdict, mos0}]——每份新年報一筆。"""
    import company_model as cm
    import edgar_pit as ep
    idx = px.index
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
        F = split_after(px["Splits"], md)
        price_raw = float(px["Close"].iloc[pos]) * F
        shares = periods[0].get("diluted_shares") or periods[0].get("shares_out")
        if not shares or shares <= 0 or price_raw <= 0:
            continue
        rf = None
        if rf_series is not None and len(rf_series):
            r_ = rf_series[rf_series.index <= md]
            rf = float(r_.iloc[-1]) / 100 if len(r_) else None
        prof = {"ticker": ticker, "price": price_raw, "mkt_cap": price_raw * shares, "shares": shares,
                "beta": beta_at(px["Close"], spy_close, md) or 1.0, "sector": sector}
        try:
            res = cm.run_model(periods, prof, cfg=({"rf": rf} if rf else None), mc=False)
        except Exception:
            continue
        sig = res.get("signal") or {}
        fv = sig.get("fair_value")
        if fv is None or not math.isfinite(float(fv)) or fv <= 0:
            continue
        out.append({"date": md, "fair_adj": float(fv) / F, "verdict": sig.get("verdict"), "mos0": sig.get("mos")})
    return out


def mos_frame(models: dict, close_adj: pd.DataFrame, dates: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    """每個訊號日：最新一份「模型日 ≤ 訊號日」的模型 → MoS＝合理價／收盤 − 1；另回 accumulate 布林寬表。"""
    mos, acc = {}, {}
    for sym, recs in models.items():
        if not recs or sym not in close_adj:
            continue
        md = pd.DatetimeIndex([r["date"] for r in recs])
        fv = np.array([r["fair_adj"] for r in recs])
        vd = np.array([r.get("verdict") == "accumulate" for r in recs])
        pos = md.searchsorted(dates, side="right") - 1
        c = close_adj[sym].reindex(dates)
        m = np.where(pos >= 0, fv[np.clip(pos, 0, None)] / c.values - 1, np.nan)
        a = np.where(pos >= 0, vd[np.clip(pos, 0, None)], False)
        # 模型太舊（> 15 個月沒有新年報）視為失效
        age = np.where(pos >= 0, (dates - md[np.clip(pos, 0, None)]).days, 99999)
        m = np.where(age <= 460, m, np.nan)
        a = np.where(age <= 460, a, False)
        mos[sym], acc[sym] = m, a
    return pd.DataFrame(mos, index=dates), pd.DataFrame(acc, index=dates).astype(bool)


# ── 3. 經典價值／品質因子（同一份 PIT 財報）────────────────────────────────────

def classic_factors(companies: dict, prices: dict, dates: pd.DatetimeIndex) -> dict:
    """每個訊號日、每檔：最新可見年報 → E/P、FCF/P、B/M、ROIC、GP/A、應計、資產成長。
    市值＝還原收盤 × 原始股數 × F(年報申報日)（拆股前後一致）。"""
    import edgar_pit as ep
    keys = list(VALUE_LABELS)[1:]
    out = {k: {} for k in keys}
    for sym, comp in companies.items():
        px = prices.get(sym)
        if px is None:
            continue
        cl = px["Close"]
        vals = {k: [] for k in keys}
        cache: dict[str, tuple] = {}
        for d in dates:
            lf = ep.latest_filed(comp, str(d.date()))
            if lf is None:
                for k in keys:
                    vals[k].append(np.nan)
                continue
            if lf not in cache:
                ps = ep.pit_periods(comp, lf, max_periods=2)
                cache[lf] = (ps, split_after(px["Splits"], pd.Timestamp(lf)))
            ps, F = cache[lf]
            c_ = cl[cl.index <= d]
            p0 = ps[0] if ps else {}
            sh = p0.get("diluted_shares") or p0.get("shares_out")
            if not ps or not sh or not len(c_):
                for k in keys:
                    vals[k].append(np.nan)
                continue
            mcap = float(c_.iloc[-1]) * sh * F
            ni, cfo, fcf = p0.get("net_income"), p0.get("cfo"), p0.get("fcf")
            eq, ta = p0.get("total_equity"), p0.get("total_assets")
            ebit, tax, pti = p0.get("operating_income"), p0.get("tax_provision"), p0.get("pretax_income")
            debt, cash = p0.get("total_debt") or 0.0, p0.get("cash_and_sti") or p0.get("cash") or 0.0
            tr = min(max(tax / pti, 0.0), 0.4) if (tax is not None and pti and pti > 0) else 0.21
            ic = (eq or 0.0) + debt - cash
            ta_prev = ps[1].get("total_assets") if len(ps) > 1 else None

            def _r(a, b):
                return (a / b) if (a is not None and b and b > 0) else np.nan
            vals["ey"].append(_r(ni, mcap))
            vals["fcfy"].append(_r(fcf, mcap))
            vals["bm"].append(_r(eq, mcap))
            vals["roic"].append(_r(ebit * (1 - tr) if ebit is not None else None, ic))
            vals["gpa"].append(_r(p0.get("gross_profit"), ta))
            vals["accruals"].append(_r((ni - cfo) if (ni is not None and cfo is not None) else None, ta))
            vals["asset_growth"].append((ta / ta_prev - 1) if (ta and ta_prev and ta_prev > 0) else np.nan)
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
        acc_lines = [f"\n*模型判定 accumulate 的股票等權持有*（平均 {acc_res.get('avg_n', 0):.0f} 檔）：年化 {fr._p(acc_res['perf'].get('cagr'))}｜"
                     f"vs SPY {fr._p(vs.get('cagr_diff'))}（IR {fr._f(vs.get('ir'))}、贏的年份 {fr._p(vs.get('win_years'), 0, False)}）｜"
                     f"回撤 {fr._p(acc_res['perf'].get('max_dd'), 0, False)}"]
    tail = ["⚠️ 機械化版本：沒有歷史分析師預估、不跑蒙地卡羅、無人工覆蓋；產業路由用現行 GICS。"
            "已下市公司缺價格與 CIK → 舊年份偏向倖存者。非投資建議"]
    return "\n".join(head + cov_lines + rest[:-1] + acc_lines + tail)


def accumulate_portfolio(A: pd.DataFrame, R: pd.DataFrame, M: pd.DataFrame, spy_r: pd.Series) -> dict:
    """每期持有「最新模型判定 accumulate 且為當期成分」的股票等權（扣換手成本）。"""
    A, R = A.align(R, join="inner", axis=None)
    ok = A.fillna(False).astype(bool) & M.reindex_like(A).fillna(False).astype(bool)
    rows, prev = [], set()
    for d in A.index:
        names = set(A.columns[ok.loc[d].values])
        if len(names) < 5:
            continue
        rr = R.loc[d, list(names)]
        if rr.notna().sum() == 0:
            continue
        to = 1.0 if not prev else len(names - prev) / len(names)
        rows.append({"date": d, "net": float(rr.mean()) - 2 * fr.COST_SIDE * to, "n": len(names)})
        prev = names
    if not rows:
        return {}
    df = pd.DataFrame(rows).set_index("date")
    return {"perf": fr.perf(df["net"]), "vs_spy": fr._excess(df["net"], spy_r), "avg_n": float(df["n"].mean())}


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
    rf = prices["^TNX"]["Close"] if "^TNX" in prices else None
    sectors = sectors if sectors is not None else sector_map()
    have = [t for t in tickers if t in prices]
    models = {t: build_models(t2c[t], prices[t], spy_close, rf, sectors.get(t), t, start) for t in have}
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
    MOS, ACC = mos_frame(models, close_adj, R.index)
    F = {"mos": MOS}
    F.update(classic_factors({t: t2c[t] for t in have}, prices, R.index))
    spy_r = fr.period_returns(total_return_open(prices["SPY"]).reindex(cal).to_frame("SPY"), dates)["SPY"]
    ew_r = R.where(M).mean(axis=1)
    study = fr.factor_study({k: v.reindex(R.index) for k, v in F.items()}, R, M, spy_r, ew_r, q)
    allM = fr.membership_matrix(periods, R.index, members)
    cov = pd.DataFrame({"members": allM.sum(axis=1), "modeled": (M & MOS.reindex_like(M).notna()).sum(axis=1)})
    acc_res = accumulate_portfolio(ACC, R, M, spy_r)
    meta = {"start": str(R.index[0].date()), "end": str(R.index[-1].date()), "q": q, "spy": fr.perf(spy_r), "ew": fr.perf(ew_r)}
    n_models = sum(len(v) for v in models.values())
    text = value_text(study, acc_res, cov, meta) + f"\n（共重建 {n_models:,} 份年報模型、{sum(1 for v in models.values() if v)} 家公司）"
    return {"study": study, "coverage": cov, "acc": acc_res, "meta": meta, "models": models, "members": M, "text": text}


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
        close = val * np.exp(dev) * (1 if edge else np.exp(rng.normal(0, 0.0, len(idx))))
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
    print("✅ 4 代碼重複使用：JCI 2016-09 前遮掉、其他不受影響")
    print("\nvalue_research selftest OK ✅")
