"""
fund_eval.py — 基金/ETF 評估：追蹤誤差、滾動 α/β、上下行捕獲、持股重疊、費用率
RBS Finance Dashboard

回答「這檔基金值不值得持有」的四個量化角度：
- 追蹤誤差（TE）：對指數型基金是品質指標（越低越好）；對主動基金是主動度
- 滾動 α/β：付管理費到底有沒有換到超額（α），還是只是槓桿化的大盤（β）
- 上/下行捕獲率：漲的時候跟多少、跌的時候躲多少（理想：上行 >100%、下行 <100%）
- 持股重疊度：兩檔基金的權重交集——買三檔科技 ETF 可能等於一檔買三次
費用率/週轉率取自 yfinance funds_data（欄名已對 1.5.1 原始碼逐一驗證：
top_holdings 是以 Symbol 為索引、含 "Holding Percent" 欄的 DataFrame；
fund_operations 的 Attributes 含 "Annual Report Expense Ratio"）。

適用：有交易代碼的 ETF / 美股共同基金；台灣未上市投信基金無資料來源、不支援。
純邏輯離線可測；fetch 層需網路。教育用途，非投資建議。
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_DAYS = 252
ROLL_WINDOW = 252            # 滾動 α/β 視窗（一年）
MIN_OBS = 126                # 至少半年日資料才評


# ── 純邏輯 ────────────────────────────────────────────────────────────────────

def _rets(px: pd.Series) -> pd.Series:
    # fill_method=None：資料缺口（停牌/缺日）跨段報酬會是 NaN 而被丟棄——
    # 先 dropna 再 pct_change 會把 60 天缺口變成一根幻影「單日」報酬，污染 TE/β
    return px.pct_change(fill_method=None).dropna()


def _align_rets(fund_px: pd.Series, bench_px: pd.Series) -> pd.DataFrame:
    df = pd.concat({"f": _rets(fund_px), "b": _rets(bench_px)}, axis=1).dropna()
    return df


def tracking_error(fund_px: pd.Series, bench_px: pd.Series) -> float | None:
    """年化追蹤誤差 = std(日超額) × √252。"""
    df = _align_rets(fund_px, bench_px)
    if len(df) < MIN_OBS:
        return None
    return float((df["f"] - df["b"]).std(ddof=1) * math.sqrt(TRADING_DAYS))


def alpha_beta(fund_px: pd.Series, bench_px: pd.Series,
               window: int | None = None) -> dict | None:
    """
    β = cov/var、α = 年化截距（CAPM 單因子）。window=None 用全期；
    給 window 則取最近 window 日（滾動視窗的「當前值」）。
    """
    df = _align_rets(fund_px, bench_px)
    if window:
        df = df.tail(window)
    if len(df) < MIN_OBS:
        return None
    f, b = df["f"].to_numpy(), df["b"].to_numpy()
    vb = float(np.var(b, ddof=1))
    if vb <= 0:
        return None
    beta = float(np.cov(f, b, ddof=1)[0, 1] / vb)
    alpha_d = float(f.mean() - beta * b.mean())
    return {"beta": round(beta, 3), "alpha_ann": round(alpha_d * TRADING_DAYS, 4),
            "n": len(df)}


def capture_ratios(fund_px: pd.Series, bench_px: pd.Series) -> dict | None:
    """上/下行捕獲率：基準上漲日與下跌日，基金平均日報酬 ÷ 基準平均日報酬。"""
    df = _align_rets(fund_px, bench_px)
    if len(df) < MIN_OBS:
        return None
    up = df[df["b"] > 0]
    dn = df[df["b"] < 0]
    out = {}
    if len(up) >= 30 and float(up["b"].mean()) > 0:
        out["up"] = round(float(up["f"].mean() / up["b"].mean()), 3)
    if len(dn) >= 30 and float(dn["b"].mean()) < 0:
        out["down"] = round(float(dn["f"].mean() / dn["b"].mean()), 3)
    return out or None


def holdings_overlap(h1: dict[str, float], h2: dict[str, float]) -> dict | None:
    """
    權重重疊 = Σ min(w1, w2)（僅就雙方公布的前十大等可得持股計，屬下限估計）。
    輸入 {symbol: weight(0-1)}。
    """
    if not h1 or not h2:
        return None

    def _norm(h: dict) -> dict:
        # 防呆：權重若以「百分點」制傳入（總和 >3），正規化回小數制
        tot = sum(float(v) for v in h.values())
        return {k: float(v) / 100 for k, v in h.items()} if tot > 3 else \
            {k: float(v) for k, v in h.items()}

    h1, h2 = _norm(h1), _norm(h2)
    common = set(h1) & set(h2)
    ov = sum(min(h1[s], h2[s]) for s in common)
    return {"overlap": round(ov, 4), "n_common": len(common),
            "cov1": round(sum(map(float, h1.values())), 4),
            "cov2": round(sum(map(float, h2.values())), 4)}


def parse_operations(df: pd.DataFrame | None) -> dict:
    """fund_operations DataFrame → {expense_ratio, turnover, net_assets, cat_expense}。"""
    out: dict = {}
    if df is None or not hasattr(df, "iterrows") or df.empty:
        return out
    d = df.copy()
    if "Attributes" in d.columns:
        d = d.set_index("Attributes")
    cols = list(d.columns)
    # 精確比對「Category …」開頭，避免代碼含 CAT 的基金（如 CATH）被誤認成類別欄
    cat_col = next((c for c in cols if str(c).startswith("Category")), None)
    fund_col = next((c for c in cols if c != cat_col), None)
    for idx, _row in d.iterrows():
        key = str(idx)
        val = _num(d.loc[idx, fund_col]) if fund_col else None
        if "Expense Ratio" in key:
            out["expense_ratio"] = val
            if cat_col is not None:
                out["cat_expense"] = _num(d.loc[idx, cat_col])
        elif "Turnover" in key:
            out["turnover"] = val
        elif "Net Assets" in key:
            out["net_assets"] = val
    return out


def _num(x):
    try:
        if isinstance(x, str):
            x = x.strip()
            if x.endswith("%"):                 # "0.09%" 這類字串型費用率
                return float(x[:-1].replace(",", "")) / 100
            x = x.replace(",", "")
        v = float(x)
        return v if not math.isnan(v) else None
    except (TypeError, ValueError):
        return None


def evaluate(fund_px: pd.Series, bench_px: pd.Series,
             ops: dict | None = None) -> dict | None:
    """組裝單檔評估（純邏輯部分）。"""
    te = tracking_error(fund_px, bench_px)
    if te is None:
        return None
    full = alpha_beta(fund_px, bench_px)
    recent = alpha_beta(fund_px, bench_px, window=ROLL_WINDOW)
    cap = capture_ratios(fund_px, bench_px)
    out = {"te": round(te, 4), "full": full, "recent": recent, "capture": cap}
    if ops:
        out["ops"] = ops
    return out


def fund_text(symbol: str, bench: str, ev: dict,
              sectors: dict | None = None) -> str:
    lines = [f"🎯 *{symbol} 基金評估*（基準 {bench}）"]
    ops = ev.get("ops") or {}
    if ops.get("expense_ratio") is not None:
        seg = f"💸 費用率 {ops['expense_ratio']:.2%}"
        if ops.get("cat_expense") is not None:
            seg += f"（同類平均 {ops['cat_expense']:.2%}）"
        if ops.get("turnover") is not None:
            seg += f"｜週轉率 {ops['turnover']:.0%}"
        lines.append(seg)
    te = ev["te"]
    te_note = ("被動追蹤緊密" if te < 0.02 else
               ("中度主動" if te < 0.06 else "高度主動/偏離基準"))
    lines.append(f"📏 追蹤誤差 {te:.1%}/年（{te_note}）")
    if ev.get("full"):
        f = ev["full"]
        lines.append(f"📐 全期 β {f['beta']:.2f}｜α {f['alpha_ann']:+.1%}/年"
                     f"（{f['n']} 日）")
    if ev.get("recent") and ev.get("full") and ev["recent"]["n"] < ev["full"]["n"]:
        r = ev["recent"]
        lines.append(f"　近一年 β {r['beta']:.2f}｜α {r['alpha_ann']:+.1%}/年")
    cap = ev.get("capture") or {}
    if cap:
        seg = "⚖️ 捕獲率："
        if "up" in cap:
            seg += f"上行 {cap['up']:.0%}"
        if "down" in cap:
            seg += f"｜下行 {cap['down']:.0%}"
            if cap.get("up") and cap["up"] > cap["down"]:
                seg += "（跟漲多於跟跌 👍）"
            elif cap.get("up") and cap["up"] < cap["down"]:
                seg += "（跌時跟好跟滿、漲時縮水 ⚠️）"
        lines.append(seg)
    if sectors:
        top3 = sorted(sectors.items(), key=lambda x: -x[1])[:3]
        lines.append("🏭 前三產業：" + "、".join(f"{k} {v:.0%}" for k, v in top3))
    if (symbol.endswith((".TW", ".TWO", ".HK", ".T", ".KS"))
            != bench.endswith((".TW", ".TWO", ".HK", ".T", ".KS"))):
        lines.append("⚠️ 基金與基準屬不同市場/幣別——TE 與 β 混入匯率效果，僅供粗略參考")
    lines.append("_α 為 CAPM 單因子、未扣顯著性檢定；費用率是長期報酬的確定損耗，"
                 "α 則不保證持續。非投資建議。_")
    return "\n".join(lines)


def overlap_text(a: str, b: str, ov: dict | None,
                 top_common: list[tuple[str, float, float]] | None = None) -> str:
    if not ov:
        return f"❌ {a} / {b} 至少一檔拿不到持股資料（債券/貨幣型基金常見）"
    lines = [f"🔍 *{a} × {b} 持股重疊*",
             f"權重重疊 ≥ {ov['overlap']:.0%}（僅就公布的前十大持股計，屬下限）",
             f"共同持股 {ov['n_common']} 檔"
             f"（兩檔公布覆蓋率 {ov['cov1']:.0%} / {ov['cov2']:.0%}）"]
    for s, w1, w2 in (top_common or [])[:5]:
        lines.append(f"　• {s}：{w1:.1%} vs {w2:.1%}")
    if ov["overlap"] >= 0.3:
        lines.append("⚠️ 重疊偏高——同時持有兩檔的分散效果有限，等於加倍押同一批股票")
    lines.append("_非投資建議。_")
    return "\n".join(lines)


# ── 抓取層（需網路）───────────────────────────────────────────────────────────

def fetch_fund_data(symbol: str) -> dict:
    """funds_data 各件（逐項容錯）：holdings dict、sectors、ops。"""
    out: dict = {"holdings": {}, "sectors": None, "ops": {}}
    try:
        import yfinance as yf
        fd = yf.Ticker(symbol).funds_data
        try:
            th = fd.top_holdings
            if th is not None and len(th) and "Holding Percent" in th.columns:
                out["holdings"] = {str(i): float(v) for i, v
                                   in th["Holding Percent"].items()
                                   if _num(v) is not None}
        except Exception:
            pass
        try:
            sw = fd.sector_weightings
            if isinstance(sw, dict) and sw:
                out["sectors"] = {str(k): float(v) for k, v in sw.items()}
        except Exception:
            pass
        try:
            out["ops"] = parse_operations(fd.fund_operations)
        except Exception:
            pass
    except Exception:
        pass
    return out


def run_fund_eval(symbol: str, bench: str = "SPY",
                  period: str = "5y") -> tuple[dict, str] | None:
    """端到端單檔評估。"""
    try:
        from sector_scan import _batch_closes
        symbol, bench = symbol.upper(), bench.upper()
        if symbol == bench:
            bench = "SPY" if symbol != "SPY" else "QQQ"
        closes = _batch_closes([symbol, bench], period, min_len=MIN_OBS)
        if symbol not in closes or bench not in closes:
            return None
        fd = fetch_fund_data(symbol)
        ev = evaluate(closes[symbol], closes[bench], fd.get("ops"))
        if not ev:
            return None
        return ev, fund_text(symbol, bench, ev, fd.get("sectors"))
    except Exception:
        return None


def run_overlap(a: str, b: str) -> str | None:
    """端到端重疊比較。"""
    try:
        a, b = a.upper(), b.upper()
        h1 = fetch_fund_data(a)["holdings"]
        h2 = fetch_fund_data(b)["holdings"]
        ov = holdings_overlap(h1, h2)
        top_common = sorted(((s, h1[s], h2[s]) for s in set(h1) & set(h2)),
                            key=lambda x: -min(x[1], x[2]))
        return overlap_text(a, b, ov, top_common)
    except Exception:
        return None


# ── CLI 自我測試（離線純邏輯，工程化合成資料）─────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(3)
    n = 1000
    idx = pd.bdate_range("2022-06-01", periods=n)
    b_ret = rng.normal(0.0004, 0.010, n)
    bench = pd.Series(100 * np.cumprod(1 + b_ret), index=idx)

    # 工程化：β=1.2、日 α=+2bp、追蹤噪音 σ=0.4%
    noise = rng.normal(0, 0.004, n)
    f_ret = 1.2 * b_ret + 0.0002 + noise
    fund = pd.Series(50 * np.cumprod(1 + f_ret), index=idx)

    te = tracking_error(fund, bench)
    # TE ≈ std(0.2·b + noise)·√252（β≠1 的部分也算偏離）
    te_expect = float(np.std(0.2 * b_ret + noise, ddof=1) * math.sqrt(252))
    assert te is not None and abs(te - te_expect) < 0.01, (te, te_expect)

    ab = alpha_beta(fund, bench)
    assert ab and abs(ab["beta"] - 1.2) < 0.05, ab
    assert 0 < ab["alpha_ann"] < 0.12                       # 年化 α ≈ +5%（2bp×252）

    # 純追蹤型（β=1、無α、微噪音）→ TE 極小
    g_ret = b_ret + rng.normal(0, 0.0005, n)
    tracker = pd.Series(100 * np.cumprod(1 + g_ret), index=idx)
    assert tracking_error(tracker, bench) < 0.02

    cap = capture_ratios(fund, bench)
    assert cap and cap["up"] > 1.0 and cap["down"] > 1.0    # β>1 → 雙向都放大
    cap_t = capture_ratios(tracker, bench)
    assert abs(cap_t["up"] - 1.0) < 0.1 and abs(cap_t["down"] - 1.0) < 0.1

    # 重疊：手算 min-sum
    ov = holdings_overlap({"AAPL": 0.30, "MSFT": 0.20, "NVDA": 0.10},
                          {"AAPL": 0.10, "MSFT": 0.40, "GOOG": 0.20})
    assert ov and abs(ov["overlap"] - 0.30) < 1e-9 and ov["n_common"] == 2, ov
    assert holdings_overlap({}, {"A": 1}) is None

    # fund_operations 解析（照 yfinance 1.5.1 實際形狀：Attributes 欄 + 基金/類別欄）
    ops_df = pd.DataFrame({
        "Attributes": ["Annual Report Expense Ratio", "Annual Holdings Turnover",
                       "Total Net Assets"],
        "SPY": [0.0009, 0.02, 5.5e11], "Category Average": [0.0035, 0.4, None]})
    ops = parse_operations(ops_df)
    assert abs(ops["expense_ratio"] - 0.0009) < 1e-12 and \
        abs(ops["cat_expense"] - 0.0035) < 1e-12 and ops["turnover"] == 0.02, ops
    assert parse_operations(None) == {} and parse_operations(pd.DataFrame()) == {}

    ev = evaluate(fund, bench, ops)
    txt = fund_text("TESTFUND", "SPY", ev, {"technology": 0.5, "financial": 0.2})
    assert "追蹤誤差" in txt and "費用率 0.09%" in txt and "非投資建議" in txt
    assert "technology 50%" in txt

    ot = overlap_text("QQQ", "VGT", ov, [("AAPL", 0.3, 0.1), ("MSFT", 0.2, 0.4)])
    assert "權重重疊 ≥ 30%" in ot and "AAPL" in ot
    assert "拿不到持股" in overlap_text("A", "B", None)

    # 短資料 → 不評
    assert evaluate(fund.iloc[:100], bench.iloc[:100]) is None

    print(f"✅ fund_eval 離線自我測試通過（TE {te:.1%} vs 預期 {te_expect:.1%}、"
          f"β {ab['beta']}、重疊 {ov['overlap']:.0%}）")
