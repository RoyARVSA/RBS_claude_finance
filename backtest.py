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
) -> list[dict]:
    """
    對每個進場點，往後 horizon 天評估三道關卡。
    tp/sl 為正數比例（如 0.05 = 5%）。short=True 時方向反轉（做空）。
    回傳每筆交易的結果 list。
    """
    px = close.values
    idx = np.where(entries.fillna(False).values)[0]
    trades = []

    for i in idx:
        if i + 1 >= len(px):
            continue
        entry_px = px[i]
        if entry_px <= 0 or np.isnan(entry_px):
            continue

        end = min(i + horizon, len(px) - 1)
        tp_line = entry_px * (1 + tp) if not short else entry_px * (1 - tp)
        sl_line = entry_px * (1 - sl) if not short else entry_px * (1 + sl)

        outcome, exit_px, held = "time", px[end], end - i
        for j in range(i + 1, end + 1):
            p = px[j]
            if np.isnan(p):
                continue
            if not short:
                if p >= tp_line:
                    outcome, exit_px, held = "win", tp_line, j - i; break
                if p <= sl_line:
                    outcome, exit_px, held = "loss", sl_line, j - i; break
            else:
                if p <= tp_line:
                    outcome, exit_px, held = "win", tp_line, j - i; break
                if p >= sl_line:
                    outcome, exit_px, held = "loss", sl_line, j - i; break

        ret = (exit_px / entry_px - 1) * (-1 if short else 1)
        trades.append({
            "entry_idx": int(i),
            "entry_date": close.index[i],
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
) -> pd.DataFrame:
    """
    對所有訊號規則跑回測，回傳排序後的績效比較表。
    名稱含「(空)」的規則自動以做空方向評估。
    """
    close = df["Close"].dropna()
    rules = signal_rules(df)
    rows = []
    for name, entries in rules.items():
        is_short = "(空)" in name
        trades = triple_barrier(close, entries.reindex(close.index),
                                tp=tp, sl=sl, horizon=horizon, short=is_short)
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

def rule_edge_scores(df: pd.DataFrame, **kw) -> dict[str, float]:
    """
    回傳每個規則的「edge」分數 = (勝率-0.5)×2 × min(profit_factor,3)/3，
    範圍約 -1~+1，可用來動態加權訊號（勝率高的訊號權重大）。
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
    return edges


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
    df = raw if "Close" in raw.columns else raw.xs(ticker, axis=1, level=1)

    print(f"\n=== {ticker} 訊號回測（停利5% / 停損3% / 持有10天）===\n")
    bt = backtest_all(df, tp=0.05, sl=0.03, horizon=10)
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print(bt.to_string())
