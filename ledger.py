"""
ledger.py – 交易帳本式投組追蹤（Ghostfolio 精神，純邏輯離線可測）

輸入真實買賣紀錄 [{date, ticker, side(buy/sell), qty, price, fee}]，輸出：
平均成本/已實現/未實現損益、每日估值權益曲線、TWR（現金流調整）、XIRR（資金加權）、
股息收入。價格/股息序列由呼叫端傳入（網頁層抓），本模組零網路依賴。
慣例：金額同一幣別（美股=USD）；賣出數量不得超過持有（超賣列會被拒絕並回報）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── 驗證與成本基礎 ─────────────────────────────────────────────────────────────

def validate_trades(trades: list[dict]) -> tuple[list[dict], list[str]]:
    """清洗交易列。回 (合法交易・依日期排序, 錯誤訊息清單)。"""
    ok, errs = [], []
    for i, t in enumerate(trades or []):
        try:
            d = pd.Timestamp(t["date"]).normalize()
            side = str(t["side"]).strip().lower()
            qty = float(t["qty"])
            price = float(t["price"])
            fee = float(t.get("fee") or 0)
            tk = str(t["ticker"]).strip().upper()
            if side not in ("buy", "sell") or qty <= 0 or price < 0 or not tk:
                raise ValueError("欄位值不合法")
            ok.append({"date": d, "ticker": tk, "side": side,
                       "qty": qty, "price": price, "fee": fee})
        except Exception as e:
            errs.append(f"第 {i + 1} 列略過：{e}")
    ok.sort(key=lambda t: (t["date"], t["side"]))   # 同日先 buy 後 sell
    return ok, errs


def positions_and_realized(trades: list[dict]) -> tuple[dict, list[str], list[dict]]:
    """
    平均成本法逐筆結轉。
    回 ({ticker: {qty, avg_cost, cost, realized}}, 錯誤清單, 已入帳交易清單)。
    賣出超過持有的交易列會被拒絕（記錯誤、不入帳）——**權益曲線請用回傳的
    已入帳清單**，否則被拒的列會汙染持股計算。
    """
    pos: dict = {}
    errs: list[str] = []
    accepted: list[dict] = []
    for t in trades:
        p = pos.setdefault(t["ticker"], {"qty": 0.0, "avg_cost": 0.0,
                                         "cost": 0.0, "realized": 0.0})
        if t["side"] == "buy":
            new_cost = p["cost"] + t["qty"] * t["price"] + t["fee"]
            p["qty"] += t["qty"]
            p["cost"] = new_cost
            p["avg_cost"] = new_cost / p["qty"] if p["qty"] else 0.0
        else:
            if t["qty"] > p["qty"] + 1e-9:
                errs.append(f"{t['date'].date()} 賣出 {t['ticker']} {t['qty']:g} 股"
                            f" > 持有 {p['qty']:g} 股，該列不入帳")
                continue
            p["realized"] += (t["price"] - p["avg_cost"]) * t["qty"] - t["fee"]
            p["qty"] -= t["qty"]
            p["cost"] = p["avg_cost"] * p["qty"]
            if p["qty"] < 1e-9:
                p["qty"], p["cost"], p["avg_cost"] = 0.0, 0.0, 0.0   # 清倉即歸零，避免顯示過期均價
        accepted.append(t)
    return pos, errs, accepted


# ── 權益曲線 / TWR ─────────────────────────────────────────────────────────────

def equity_curve(trades: list[dict], prices: pd.DataFrame) -> pd.DataFrame:
    """
    每日估值。prices：index=日期、columns=ticker 的收盤價（覆蓋交易期間）。
    回 DataFrame(value, flow)：value=當日持倉市值、flow=當日淨外部現金流
    （買入金額+費用 − 賣出淨額；視為投入/取回）。
    """
    if not trades or prices is None or prices.empty:
        return pd.DataFrame(columns=["value", "flow"])
    idx = prices.index
    start = min(t["date"] for t in trades)
    idx = idx[idx >= start]
    qty = pd.DataFrame(0.0, index=idx, columns=sorted({t["ticker"] for t in trades}))
    flow = pd.Series(0.0, index=idx)
    for t in trades:
        if t["ticker"] not in prices.columns:
            continue    # 沒價格的標的：qty 與 flow 一起跳過，否則 flow 入帳但市值為零會扭曲 TWR
        # 交易日若非交易所開盤日，落到下一個可用日
        pos_dates = idx[idx >= t["date"]]
        if len(pos_dates) == 0:
            continue
        d0 = pos_dates[0]
        sign = 1 if t["side"] == "buy" else -1
        qty.loc[d0:, t["ticker"]] += sign * t["qty"]
        amt = t["qty"] * t["price"]
        flow.loc[d0] += (amt + t["fee"]) if t["side"] == "buy" else -(amt - t["fee"])
    px = prices.reindex(idx).ffill()
    common = [c for c in qty.columns if c in px.columns]
    value = (qty[common] * px[common]).sum(axis=1)
    return pd.DataFrame({"value": value, "flow": flow})


def twr_returns(curve: pd.DataFrame) -> pd.Series:
    """
    現金流調整的每日報酬（true TWR 子期間法）：
      r_t = (V_t − F_t) / V_{t−1} − 1，V_{t−1}=0 的日子略過。
    """
    if curve is None or curve.empty:
        return pd.Series(dtype=float)
    v, f = curve["value"], curve["flow"]
    prev = v.shift(1)
    r = (v - f) / prev - 1
    r = r[(prev > 0) & r.notna()]
    return r


def xirr(cashflows: list[tuple], guess: float = 0.1) -> float | None:
    """
    資金加權年化報酬（XIRR）。cashflows：[(date, amount)]，投入為負、取回/終值為正。
    牛頓法 + 二分備援；無解回 None。
    """
    if len(cashflows) < 2:
        return None
    cfs = sorted(((pd.Timestamp(d), float(a)) for d, a in cashflows), key=lambda x: x[0])
    if not (any(a < 0 for _, a in cfs) and any(a > 0 for _, a in cfs)):
        return None
    t0 = cfs[0][0]
    years = np.array([(d - t0).days / 365.25 for d, _ in cfs])
    amts = np.array([a for _, a in cfs])

    def npv(rate):
        return float(np.sum(amts / np.power(1 + rate, years)))

    r = guess
    for _ in range(60):                     # 牛頓法
        f0 = npv(r)
        df = (npv(r + 1e-6) - f0) / 1e-6
        if abs(df) < 1e-12:
            break
        r2 = r - f0 / df
        if not np.isfinite(r2) or r2 <= -0.999:
            break
        if abs(r2 - r) < 1e-9:
            return float(r2)
        r = r2
    lo, hi = -0.999, 10.0                   # 二分備援
    if npv(lo) * npv(hi) > 0:
        return None
    for _ in range(200):
        mid = (lo + hi) / 2
        if npv(lo) * npv(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return float((lo + hi) / 2)


def dividend_income(trades: list[dict], dividends: dict) -> dict:
    """
    股息收入（純函數）。dividends：{ticker: pd.Series(除息日→每股股息)}。
    以除息日當下持有股數計。回 {ticker: 金額, "_total": 總額}。
    """
    out: dict = {}
    total = 0.0
    for tk, div in (dividends or {}).items():
        if div is None or len(div) == 0:
            continue
        tk_trades = [t for t in trades if t["ticker"] == tk.upper()]
        if not tk_trades:
            continue
        amt = 0.0
        for ex_date, dps in div.items():
            ex = pd.Timestamp(ex_date).tz_localize(None).normalize()
            held = sum((1 if t["side"] == "buy" else -1) * t["qty"]
                       for t in tk_trades if t["date"] < ex)
            if held > 0:
                amt += held * float(dps)
        if amt > 0:
            out[tk.upper()] = amt
            total += amt
    out["_total"] = total
    return out


# ── CLI 自我測試 ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    trades_raw = [
        {"date": "2025-01-02", "ticker": "aapl", "side": "BUY", "qty": 10, "price": 100, "fee": 1},
        {"date": "2025-06-02", "ticker": "AAPL", "side": "buy", "qty": 10, "price": 120, "fee": 1},
        {"date": "2025-09-01", "ticker": "AAPL", "side": "sell", "qty": 5, "price": 150, "fee": 1},
        {"date": "2025-09-02", "ticker": "AAPL", "side": "sell", "qty": 999, "price": 150, "fee": 0},  # 超賣
        {"date": "bad-date", "ticker": "X", "side": "buy", "qty": 1, "price": 1},                      # 壞列
    ]
    trades, verrs = validate_trades(trades_raw)
    assert len(trades) == 4 and len(verrs) == 1
    pos, perrs, trades = positions_and_realized(trades)    # trades ← 已入帳清單
    assert len(perrs) == 1 and len(trades) == 3            # 超賣被拒且不在入帳清單
    p = pos["AAPL"]
    avg = (10 * 100 + 1 + 10 * 120 + 1) / 20               # 110.1
    assert abs(p["avg_cost"] - avg) < 1e-9
    assert abs(p["realized"] - ((150 - avg) * 5 - 1)) < 1e-9
    assert abs(p["qty"] - 15) < 1e-9
    print(f"positions OK: qty={p['qty']}, avg={p['avg_cost']:.2f}, realized={p['realized']:.2f}")

    # 權益曲線 + TWR：價格不變的日子報酬必為 0（現金流不該汙染 TWR）
    idx = pd.bdate_range("2025-01-02", "2025-10-01")
    px = pd.DataFrame({"AAPL": np.linspace(100, 150, len(idx))}, index=idx)
    curve = equity_curve(trades, px)
    assert abs(curve["value"].iloc[-1] - 15 * px["AAPL"].iloc[-1]) < 1e-6
    r = twr_returns(curve)
    flat = pd.DataFrame({"AAPL": [100.0] * 10},
                        index=pd.bdate_range("2025-01-02", periods=10))
    c2 = equity_curve(trades[:2], flat)
    r2 = twr_returns(c2)
    assert (r2.abs() < 1e-12).all()                        # 價格平 → TWR 全 0
    print(f"TWR OK: {len(r)} 日報酬，累積 {(1 + r).prod() - 1:.2%}")

    # XIRR：投入 1000、一年後拿回 1100 → 10%
    x = xirr([("2025-01-01", -1000), ("2026-01-01", 1100)])
    assert x is not None and abs(x - 0.10) < 0.002, x
    assert xirr([("2025-01-01", -1000)]) is None
    assert xirr([("2025-01-01", -1000), ("2026-01-01", -50)]) is None   # 無正流
    print(f"XIRR OK: {x:.2%}")

    # 股息：除息日前持有 20 股、每股 0.5 → 10
    div = {"AAPL": pd.Series([0.5], index=[pd.Timestamp("2025-07-01")])}
    inc = dividend_income(trades, div)
    assert abs(inc["AAPL"] - 10.0) < 1e-9 and abs(inc["_total"] - 10.0) < 1e-9
    print(f"dividends OK: {inc}")

    print("\n✅ ledger 純邏輯測試通過")
