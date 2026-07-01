"""
alpaca_trader.py – Alpaca 紙上交易整合（可重用，無 Streamlit 依賴）

用免費的 Alpaca paper trading 帳戶，把訊號評分轉成模擬下單，追蹤策略真實績效。
純模擬、不涉真錢。API key 申請：https://alpaca.markets/ → Paper Trading。

設計：
  • HTTP 層（帳戶/持倉/下單/訂單/權益曲線）以 requests 實作（cron 無 SDK）
  • 決策層 decide_orders() 為純函數，離線可完整測試
  • Long-only（v1）；預設安全，實際下單由 bot 端的 autotrade 開關控制

環境變數：ALPACA_KEY_ID、ALPACA_SECRET_KEY（可選 ALPACA_BASE_URL）
"""

from __future__ import annotations

import os

PAPER_BASE = "https://paper-api.alpaca.markets"

# 風控/進出場預設（bot 端可覆蓋）
ALPACA_DEFAULTS = {
    "buy_threshold":     0.5,    # 評分 ≥ 此值 → 開多
    "exit_threshold":   -0.2,    # 評分 ≤ 此值 → 平倉
    "max_positions":     10,     # 最多持倉檔數
    "max_position_pct":  0.15,   # 每檔最多佔淨值比例
    "risk_pct":          0.01,   # ATR 部位單筆風險
}


# ── 純決策邏輯（離線可測）──────────────────────────────────────────────────────

def decide_orders(scored: list[dict], positions: dict, equity: float,
                  buying_power: float, config: dict | None = None) -> list[dict]:
    """
    給定評分、目前持倉、帳戶淨值與購買力，決定要下哪些單（純函數）。

    scored:    [{"ticker","score","price","risk_per_share"(選填)}, ...]
    positions: {symbol: {"qty": float, ...}}  目前 Alpaca 持倉
    回傳:      [{"symbol","side","qty","reason"}, ...]（side: buy/sell）

    規則（long-only）：
      • 持倉中且評分 ≤ exit_threshold → 賣出平倉
      • 未持倉且評分 ≥ buy_threshold → 買進，ATR 風險部位，
        受「每檔上限 max_position_pct×淨值」「可用購買力」「max_positions」三重約束
      • 依評分由高到低分配買進，額度或檔數用盡即止
    """
    cfg = {**ALPACA_DEFAULTS, **(config or {})}
    buy_th   = float(cfg["buy_threshold"])
    exit_th  = float(cfg["exit_threshold"])
    max_pos  = int(cfg["max_positions"])
    max_pct  = float(cfg["max_position_pct"])
    risk_pct = float(cfg["risk_pct"])

    score_map = {s["ticker"]: s for s in scored}
    held = set(positions.keys())
    orders: list[dict] = []

    # 1) 出場：持倉評分 ≤ 出場門檻 → 平倉
    for sym, pos in positions.items():
        s = score_map.get(sym)
        if s is not None and float(s.get("score", 0)) <= exit_th:
            qty = abs(float(pos.get("qty", 0)))
            if qty > 0:
                orders.append({"symbol": sym, "side": "sell", "qty": qty,
                               "reason": f"評分 {s['score']:+.2f} ≤ 出場門檻 {exit_th}"})
                held.discard(sym)   # 釋出一個名額

    # 2) 進場：評分 ≥ 買進門檻且未持倉，依分數高→低分配
    slots = max_pos - len(held)
    if slots > 0 and equity > 0:
        cands = sorted(
            [s for s in scored
             if s["ticker"] not in held
             and float(s.get("score", 0)) >= buy_th
             and float(s.get("price", 0) or 0) > 0],
            key=lambda s: -float(s["score"]),
        )
        bp = float(buying_power)
        for s in cands:
            if slots <= 0:
                break
            price = float(s["price"])
            rps = s.get("risk_per_share")
            rps = float(rps) if rps and float(rps) > 0 else price * 0.05   # 缺 ATR → 退回 5% 停損
            qty_risk = (equity * risk_pct) / rps            # ATR 風險部位
            qty_cap  = (equity * max_pct) / price           # 每檔上限
            qty_bp   = bp / price if price > 0 else 0        # 購買力上限
            qty = int(min(qty_risk, qty_cap, qty_bp))
            if qty >= 1:
                orders.append({"symbol": s["ticker"], "side": "buy", "qty": qty,
                               "reason": f"評分 {s['score']:+.2f} ≥ 買進門檻 {buy_th}"})
                bp -= qty * price
                slots -= 1
    return orders


def account_return(account: dict) -> dict:
    """從帳戶資料算報酬（純函數）。回 {equity, last_equity, day_change, day_pct}。"""
    eq = _f(account.get("equity"))
    last = _f(account.get("last_equity"))
    day_chg = (eq - last) if (eq is not None and last is not None) else None
    day_pct = (day_chg / last) if (day_chg is not None and last) else None
    return {"equity": eq, "last_equity": last, "day_change": day_chg, "day_pct": day_pct}


def _f(x):
    try:
        return float(x) if x is not None else None
    except (TypeError, ValueError):
        return None


# ── HTTP 層（需網路 + key；此環境代理擋 Alpaca，需部署後實測）──────────────────

def _base() -> str:
    return os.environ.get("ALPACA_BASE_URL", PAPER_BASE).rstrip("/")


def _headers(key: str, secret: str) -> dict:
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


def get_account(key: str, secret: str) -> dict | None:
    import requests
    try:
        r = requests.get(f"{_base()}/v2/account", headers=_headers(key, secret), timeout=20)
        return r.json() if r.ok else None
    except Exception:
        return None


def get_positions(key: str, secret: str) -> dict:
    """回 {symbol: {qty, avg_entry_price, market_value, unrealized_pl, unrealized_plpc}}。"""
    import requests
    try:
        r = requests.get(f"{_base()}/v2/positions", headers=_headers(key, secret), timeout=20)
        if not r.ok:
            return {}
        out = {}
        for p in r.json():
            out[p["symbol"]] = {
                "qty": _f(p.get("qty")),
                "avg_entry_price": _f(p.get("avg_entry_price")),
                "market_value": _f(p.get("market_value")),
                "unrealized_pl": _f(p.get("unrealized_pl")),
                "unrealized_plpc": _f(p.get("unrealized_plpc")),
            }
        return out
    except Exception:
        return {}


def submit_order(key: str, secret: str, symbol: str, qty: float, side: str) -> tuple[bool, str]:
    """市價單、time_in_force=day。回 (成功, 訊息)。"""
    import requests
    try:
        payload = {"symbol": symbol, "qty": str(int(qty)), "side": side,
                   "type": "market", "time_in_force": "day"}
        r = requests.post(f"{_base()}/v2/orders", headers=_headers(key, secret),
                          json=payload, timeout=20)
        if r.ok:
            return True, "OK"
        return False, f"{r.status_code}: {r.text[:150]}"
    except Exception as e:
        return False, str(e)


def get_orders(key: str, secret: str, limit: int = 50) -> list:
    import requests
    try:
        r = requests.get(f"{_base()}/v2/orders",
                         headers=_headers(key, secret),
                         params={"status": "all", "limit": limit, "direction": "desc"},
                         timeout=20)
        return r.json() if r.ok else []
    except Exception:
        return []


def portfolio_history(key: str, secret: str, period: str = "1M") -> dict:
    """權益曲線。回 {timestamp: [...], equity: [...]}（失敗回空）。"""
    import requests
    try:
        r = requests.get(f"{_base()}/v2/account/portfolio/history",
                         headers=_headers(key, secret),
                         params={"period": period, "timeframe": "1D"}, timeout=20)
        if not r.ok:
            return {}
        j = r.json()
        return {"timestamp": j.get("timestamp", []), "equity": j.get("equity", [])}
    except Exception:
        return {}


def close_all(key: str, secret: str) -> tuple[bool, str]:
    import requests
    try:
        r = requests.delete(f"{_base()}/v2/positions", headers=_headers(key, secret), timeout=20)
        return (r.ok, "OK" if r.ok else f"{r.status_code}: {r.text[:150]}")
    except Exception as e:
        return False, str(e)


# ── CLI 自我測試（純決策邏輯）──────────────────────────────────────────────────

if __name__ == "__main__":
    scored = [
        {"ticker": "NVDA", "score": 0.7, "price": 900, "risk_per_share": 30},
        {"ticker": "AAPL", "score": 0.55, "price": 190, "risk_per_share": 6},
        {"ticker": "TSLA", "score": -0.4, "price": 200, "risk_per_share": 10},  # held → exit
        {"ticker": "MSFT", "score": 0.1, "price": 420, "risk_per_share": 8},    # neutral → no action
    ]
    positions = {"TSLA": {"qty": 20}, "GOOG": {"qty": 5}}   # GOOG held, no score → hold
    print("=== decide_orders ===")
    orders = decide_orders(scored, positions, equity=100000, buying_power=100000)
    for o in orders:
        print(f"  {o['side'].upper():4} {o['symbol']} x{o['qty']}  ← {o['reason']}")

    print("\n=== 邊界：購買力不足 ===")
    o2 = decide_orders(scored, {}, equity=100000, buying_power=500)
    print("  ", o2 or "無單（購買力不足）")

    print("\n=== 邊界：已達最大持倉 ===")
    full = {f"S{i}": {"qty": 1} for i in range(10)}
    o3 = decide_orders(scored, full, equity=100000, buying_power=100000)
    print("  buys:", [o for o in o3 if o["side"] == "buy"] or "無（已滿倉）")

    print("\n=== account_return ===")
    print("  ", account_return({"equity": "102500", "last_equity": "100000"}))
