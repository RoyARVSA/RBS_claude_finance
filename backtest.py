"""
backtest.py – 訊號回測引擎（Triple-Barrier 三重關卡法）

學習自：
  • intelligent-trading-bot — 用「未來 N 天內價格是否突破門檻」當 label
  • López de Prado《Advances in Financial ML》— Triple-Barrier labeling
  • QuantifiedStrategies / StratBase 實證 — Profit Factor + 三層確認過濾假訊號

核心方法（Triple-Barrier）：
  每當訊號觸發，從進場價往後看 horizon 天，設三道關卡：
    1. 停利線  entry × (1 + tp)
    2. 停損線  entry × (1 - sl)
    3. 時間線  horizon 天到期
  先碰到哪一道就以那個結果結算 → 勝 / 敗 / 到期平倉。

回測指標：
  勝率 win_rate、獲利因子 profit_factor（總獲利/總虧損）、
  期望值 expectancy（每筆平均報酬）、平均持有天數、交易次數。

此模組無 Streamlit 依賴，dashboard 與 bot 都可 import。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── yfinance 單檔下載正規化 ───────────────────────────────────────────────────

def normalize_ohlc(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    把單檔 yfinance 下載結果正規化成「平面欄位」DataFrame（Close/High/Low/Volume）。
    新版 yfinance 單檔也回傳 MultiIndex (field,ticker) 或 (ticker,field)，
    若不處理，df["Close"] 會得到 (N,1) DataFrame 而非 Series，導致指標計算錯誤。
    """
    if not isinstance(raw.columns, pd.MultiIndex):
        return raw
    lvl0 = set(raw.columns.get_level_values(0))
    if "Close" in lvl0:                       # (field, ticker)
        return raw.xs(ticker, axis=1, level=1) if ticker in raw.columns.get_level_values(1) \
            else raw.droplevel(1, axis=1)
    # (ticker, field)
    return raw.xs(ticker, axis=1, level=0) if ticker in lvl0 \
        else raw.droplevel(0, axis=1)


# ── 指標計算（向量化）──────────────────────────────────────────────────────────

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, min_periods=period).mean()
    loss = (-delta).clip(lower=0).ewm(alpha=1 / period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    line = ema12 - ema26
    sig = line.ewm(span=9, adjust=False).mean()
    return line, sig, line - sig


def bollinger(close: pd.Series, period: int = 20, k: float = 2.0):
    ma = close.rolling(period).mean()
    sd = close.rolling(period).std()
    return ma + k * sd, ma, ma - k * sd


def weekly_bias_daily(close: pd.Series) -> pd.Series:
    """
    週線偏向（-2~+2）映射回每日索引，供回測 MTF 過濾用。

    無前視保證：週線用 resample("W")（週日結尾標記），每日以 ffill 對齊時
    只會取到「上一個已收完的週」——當週的週日標籤在未來，不會被選到；
    再 shift(1) 多一層保險，確保進場當日只用到已完成週的資訊。
    回傳與 close 對齊、值為 {-2,-1,0,1,2} 的 Series（資料不足回全 0）。
    """
    if not isinstance(close.index, pd.DatetimeIndex):
        return pd.Series(0, index=close.index)
    wk = close.resample("W").last().dropna()
    if len(wk) < 12:
        return pd.Series(0, index=close.index)
    ma10 = wk.rolling(10).mean()
    ema12 = wk.ewm(span=12, adjust=False).mean()
    ema26 = wk.ewm(span=26, adjust=False).mean()
    hist = (ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()
    bias = np.sign(wk - ma10).fillna(0) + np.sign(hist).fillna(0)   # -2~+2（週）
    bias = bias.shift(1)                                            # 只用已完成週
    daily = bias.reindex(close.index, method="ffill").fillna(0)
    return daily.astype(float)


# ── 訊號規則（回傳布林 Series，True = 該日觸發進場）──────────────────────────────

def _cross_up(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift() <= b.shift())


def _cross_down(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a < b) & (a.shift() >= b.shift())


def signal_rules(df: pd.DataFrame) -> dict[str, pd.Series]:
    """
    回傳 {規則名稱: 進場布林Series}。
    df 需含 'Close'（可選 'High','Low','Volume'）。
    多數規則偏「做多進場」；空方規則名稱以 (空) 標示。
    """
    close = df["Close"]
    r = rsi(close)
    line, sig, hist = macd(close)
    ub, mb, lb = bollinger(close)
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()

    rules: dict[str, pd.Series] = {}

    # 1. RSI 超賣反彈（經典均值回歸）
    rules["RSI<30 超賣反彈"] = _cross_down(r, pd.Series(30, index=r.index))

    # 2. RSI 超買回落（做空）
    rules["RSI>70 超買回落(空)"] = _cross_up(r, pd.Series(70, index=r.index))

    # 3. MACD 金叉
    rules["MACD 金叉"] = _cross_up(line, sig)

    # 4. MACD 死叉（做空）
    rules["MACD 死叉(空)"] = _cross_down(line, sig)

    # 5. MA20/50 黃金交叉
    rules["MA20/50 黃金交叉"] = _cross_up(ma20, ma50)

    # 6. 布林下軌觸及（均值回歸買）
    rules["布林下軌反彈"] = _cross_down(close, lb)

    # 7. 布林上軌突破（動量買）
    rules["布林上軌突破"] = _cross_up(close, ub)

    # 8. ⭐三層確認（實證最佳）：MACD金叉 + RSI<60 + 價>200MA
    rules["⭐三層確認(MACD+RSI+趨勢)"] = _cross_up(line, sig) & (r < 60) & (close > ma200)

    # 9. 黃金交叉 + 多頭排列（趨勢確認）
    rules["黃金交叉+站上200MA"] = _cross_up(ma20, ma50) & (close > ma200)

    return rules


# ── Triple-Barrier 評估 ───────────────────────────────────────────────────────

def triple_barrier(
    close: pd.Series,
    entries: pd.Series,
    tp: float = 0.05,
    sl: float = 0.03,
    horizon: int = 10,
    short: bool = False,
    cost: float = 0.001,
) -> list[dict]:
    """
    對每個進場點，往後 horizon 天評估三道關卡。
    tp/sl 為正數比例（如 0.05 = 5%）。short=True 時方向反轉（做空）。

    避免前視偏誤：訊號在第 i 根收盤才確認，因此「下一根」第 i+1 根才進場
    （以 i+1 收盤為進場價），關卡從 i+2 起算 — 不偷看產生訊號當根的未來。
    cost：來回交易成本（手續費+滑價，預設 0.001 = 0.1%），自每筆報酬扣除。
    """
    px = close.values
    idx = np.where(entries.fillna(False).values)[0]
    trades = []

    for i in idx:
        entry_i = i + 1                     # 下一根才進場（消除同棒前視）
        if entry_i + 1 >= len(px):
            continue
        entry_px = px[entry_i]
        if entry_px <= 0 or np.isnan(entry_px):
            continue

        end = min(entry_i + horizon, len(px) - 1)
        tp_line = entry_px * (1 + tp) if not short else entry_px * (1 - tp)
        sl_line = entry_px * (1 - sl) if not short else entry_px * (1 + sl)

        outcome, exit_px, held = "time", px[end], end - entry_i
        for j in range(entry_i + 1, end + 1):
            p = px[j]
            if np.isnan(p):
                continue
            if not short:
                if p >= tp_line:
                    outcome, exit_px, held = "win", tp_line, j - entry_i; break
                if p <= sl_line:
                    outcome, exit_px, held = "loss", sl_line, j - entry_i; break
            else:
                if p <= tp_line:
                    outcome, exit_px, held = "win", tp_line, j - entry_i; break
                if p >= sl_line:
                    outcome, exit_px, held = "loss", sl_line, j - entry_i; break

        gross = (exit_px / entry_px - 1) * (-1 if short else 1)
        ret = gross - cost                  # 扣除來回交易成本
        trades.append({
            "entry_idx": int(entry_i),
            "entry_date": close.index[entry_i],
            "ret": float(ret),
            "outcome": outcome,
            "held": int(held),
        })

    return trades


# ── 績效指標 ──────────────────────────────────────────────────────────────────

def evaluate(trades: list[dict]) -> dict:
    """從交易清單計算回測指標。"""
    n = len(trades)
    if n == 0:
        return {
            "trades": 0, "win_rate": np.nan, "profit_factor": np.nan,
            "expectancy": np.nan, "avg_win": np.nan, "avg_loss": np.nan,
            "avg_held": np.nan, "total_ret": np.nan,
        }

    rets = np.array([t["ret"] for t in trades])
    wins = rets[rets > 0]
    losses = rets[rets < 0]

    gross_win = wins.sum() if len(wins) else 0.0
    gross_loss = -losses.sum() if len(losses) else 0.0
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else np.inf

    return {
        "trades": n,
        "win_rate": len(wins) / n,
        "profit_factor": profit_factor,
        "expectancy": float(rets.mean()),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "avg_held": float(np.mean([t["held"] for t in trades])),
        "total_ret": float(rets.sum()),
    }


def backtest_all(
    df: pd.DataFrame,
    tp: float = 0.05,
    sl: float = 0.03,
    horizon: int = 10,
    cost: float = 0.001,
    mtf_filter: bool = False,
) -> pd.DataFrame:
    """
    對所有訊號規則跑回測，回傳排序後的績效比較表。
    名稱含「(空)」的規則自動以做空方向評估。
    cost：來回交易成本（預設 0.1%）。
    mtf_filter=True：只保留「週線同向」的進場（做多→週偏多、做空→週偏空），
                     用來檢驗多時間框架確認是否提升績效（無前視，見 weekly_bias_daily）。
    """
    close = df["Close"].dropna()
    rules = signal_rules(df)
    wbias = weekly_bias_daily(close) if mtf_filter else None
    rows = []
    for name, entries in rules.items():
        is_short = "(空)" in name
        ent = entries.reindex(close.index).fillna(False).astype(bool)
        if mtf_filter and wbias is not None:
            aligned = (wbias < 0) if is_short else (wbias > 0)   # 週線同向
            ent = ent & aligned.reindex(ent.index).fillna(False).astype(bool)
        trades = triple_barrier(close, ent,
                                tp=tp, sl=sl, horizon=horizon, short=is_short, cost=cost)
        m = evaluate(trades)
        m["rule"] = name
        rows.append(m)

    out = pd.DataFrame(rows).set_index("rule")
    cols = ["trades", "win_rate", "profit_factor", "expectancy",
            "avg_win", "avg_loss", "avg_held", "total_ret"]
    out = out[cols]
    # 排序：先看獲利因子，再看期望值（過濾交易數過少者放後面）
    out["_sort"] = out.apply(
        lambda r: (r["profit_factor"] if r["trades"] >= 5 and np.isfinite(r["profit_factor"]) else -1),
        axis=1,
    )
    out = out.sort_values("_sort", ascending=False).drop(columns="_sort")
    return out


# ── 規則勝率 → 回饋給評分系統的權重 ───────────────────────────────────────────

def rule_edge_scores(df: pd.DataFrame, robust: bool = True, **kw) -> dict[str, float]:
    """
    回傳每個規則的「edge」分數 = (勝率-0.5)×2 × min(profit_factor,3)/3，
    範圍約 -1~+1，可用來動態加權訊號（勝率高的訊號權重大）。

    robust=True：用樣本外一致性折減 edge（過擬合的規則被打折），
                 避免自我優化迴圈追逐只在歷史成立、未來無效的雜訊。
    """
    bt = backtest_all(df, **kw)
    edges = {}
    for name, row in bt.iterrows():
        if row["trades"] < 5 or not np.isfinite(row["profit_factor"]):
            edges[name] = 0.0
            continue
        wr_edge = (row["win_rate"] - 0.5) * 2
        pf_factor = min(row["profit_factor"], 3.0) / 3.0
        edges[name] = float(np.clip(wr_edge * pf_factor, -1, 1))

    if robust:
        consistency = walk_forward(df, **kw)   # {rule: 0~1 一致性}
        for name in edges:
            edges[name] *= consistency.get(name, 0.5)  # 不穩定者折減
    return edges


def walk_forward(df: pd.DataFrame, n_splits: int = 4, **kw) -> dict[str, float]:
    """
    Walk-forward 樣本外一致性檢測（防過擬合）。
    把資料切成 n_splits 段，逐段獨立回測，看每個規則「賺錢方向是否穩定」。

    回傳 {rule: consistency 0~1}：
      = (期望值為正的分段數 / 有效分段數)，分段越一致分數越高。
    一致性低 → 該規則只在某些時期有效，不該重押。
    """
    close = df["Close"].dropna()
    if len(close) < n_splits * 40:
        return {}   # 資料太短，無法切段

    bounds = np.linspace(0, len(df), n_splits + 1).astype(int)
    rules = list(signal_rules(df).keys())
    pos_count = {r: 0 for r in rules}
    valid_count = {r: 0 for r in rules}

    for s in range(n_splits):
        seg = df.iloc[bounds[s]:bounds[s + 1]]
        if len(seg) < 40:
            continue
        seg_close = seg["Close"].dropna()
        seg_rules = signal_rules(seg)
        for name, entries in seg_rules.items():
            is_short = "(空)" in name
            trades = triple_barrier(seg_close, entries.reindex(seg_close.index),
                                    short=is_short,
                                    tp=kw.get("tp", 0.05), sl=kw.get("sl", 0.03),
                                    horizon=kw.get("horizon", 10), cost=kw.get("cost", 0.001))
            if len(trades) >= 3:
                valid_count[name] += 1
                if evaluate(trades)["expectancy"] > 0:
                    pos_count[name] += 1

    consistency = {}
    for r in rules:
        consistency[r] = (pos_count[r] / valid_count[r]) if valid_count[r] >= 2 else 0.5
    return consistency


def walk_forward_details(df: pd.DataFrame, rule: str, n_splits: int = 4,
                         **kw) -> list[dict]:
    """
    單一規則的逐段 walk-forward 明細（給視覺化用）：
    [{fold, start, end, trades, win_rate, expectancy}, ...]。資料太短回 []。
    """
    close = df["Close"].dropna()
    if len(close) < n_splits * 40:
        return []
    bounds = np.linspace(0, len(df), n_splits + 1).astype(int)
    out = []
    for s in range(n_splits):
        seg = df.iloc[bounds[s]:bounds[s + 1]]
        if len(seg) < 40:
            continue
        seg_close = seg["Close"].dropna()
        entries = signal_rules(seg).get(rule)
        if entries is None:
            continue
        trades = triple_barrier(seg_close, entries.reindex(seg_close.index),
                                short=("(空)" in rule),
                                tp=kw.get("tp", 0.05), sl=kw.get("sl", 0.03),
                                horizon=kw.get("horizon", 10), cost=kw.get("cost", 0.001))
        m = evaluate(trades) if len(trades) else {"trades": 0, "win_rate": np.nan,
                                                  "expectancy": np.nan}
        out.append({"fold": s + 1,
                    "start": str(seg.index[0].date()) if hasattr(seg.index[0], "date") else str(seg.index[0]),
                    "end": str(seg.index[-1].date()) if hasattr(seg.index[-1], "date") else str(seg.index[-1]),
                    "trades": int(m.get("trades", 0)),
                    "win_rate": float(m["win_rate"]) if np.isfinite(m.get("win_rate", np.nan)) else None,
                    "expectancy": float(m["expectancy"]) if np.isfinite(m.get("expectancy", np.nan)) else None})
    return out


# ── 參數最佳化（mini-hyperopt）─────────────────────────────────────────────────

def optimize_params(
    df: pd.DataFrame,
    rule: str | None = None,
    tp_grid: list[float] | None = None,
    sl_grid: list[float] | None = None,
    horizon_grid: list[int] | None = None,
    cost: float = 0.001,
    min_trades: int = 8,
) -> pd.DataFrame:
    """
    網格搜尋最佳 (停利, 停損, 持有天數)。學習自 freqtrade hyperopt，
    但目標函數加入「樣本外一致性」以抑制過擬合：

      objective = expectancy × profit_factor_capped × consistency

    rule=None 時，每組參數取「當組表現最好的單一規則」當代表（找最適合此標的的玩法）；
    指定 rule 時只最佳化該規則。回傳依 objective 排序的網格結果表。
    """
    tp_grid = tp_grid or [0.03, 0.05, 0.08]
    sl_grid = sl_grid or [0.02, 0.03, 0.05]
    horizon_grid = horizon_grid or [5, 10, 20]

    rows = []
    for tp in tp_grid:
        for sl in sl_grid:
            for h in horizon_grid:
                bt = backtest_all(df, tp=tp, sl=sl, horizon=h, cost=cost)
                cons = walk_forward(df, tp=tp, sl=sl, horizon=h, cost=cost)

                if rule is not None:
                    if rule not in bt.index:
                        continue
                    candidates = [rule]
                else:
                    candidates = list(bt.index)

                best_obj, best = -np.inf, None
                for rname in candidates:
                    row = bt.loc[rname]
                    if row["trades"] < min_trades or not np.isfinite(row["profit_factor"]):
                        continue
                    pf_cap = min(row["profit_factor"], 3.0)
                    c = cons.get(rname, 0.5)
                    obj = row["expectancy"] * pf_cap * c
                    if obj > best_obj:
                        best_obj, best = obj, rname

                if best is None:
                    continue
                row = bt.loc[best]
                rows.append({
                    "tp": tp, "sl": sl, "horizon": h,
                    "best_rule": best,
                    "trades": int(row["trades"]),
                    "win_rate": float(row["win_rate"]),
                    "profit_factor": float(row["profit_factor"]),
                    "expectancy": float(row["expectancy"]),
                    "consistency": float(cons.get(best, 0.5)),
                    "objective": float(best_obj),
                })

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values("objective", ascending=False).reset_index(drop=True)
    return out


# ── CLI 自我測試 ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    try:
        import yfinance as yf
    except ImportError:
        print("需要 yfinance 才能跑 CLI 測試"); sys.exit(1)

    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    print(f"下載 {ticker} 2 年資料…")
    raw = yf.download(ticker, period="2y", auto_adjust=True, progress=False)
    if raw.empty:
        print("無資料"); sys.exit(1)
    df = normalize_ohlc(raw, ticker)

    print(f"\n=== {ticker} 訊號回測（停利5% / 停損3% / 持有10天）===\n")
    bt = backtest_all(df, tp=0.05, sl=0.03, horizon=10)
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print(bt.to_string())
