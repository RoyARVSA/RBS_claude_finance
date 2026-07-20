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

JOURNAL_CAP = 500   # 交易日誌保留最近筆數


# ── 交易日誌（純檔案 I/O，離線可測；bot 寫、網頁讀）─────────────────────────────

def load_journal(path) -> list:
    """讀取交易日誌（list of dict）。不存在或壞檔回 []。"""
    import json
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def append_journal(path, entries: list[dict], cap: int = JOURNAL_CAP) -> None:
    """把新紀錄附加到日誌，保留最近 cap 筆後寫回。entries 為 dict list。"""
    import json
    from pathlib import Path
    log = load_journal(path)
    log.extend(entries)
    if len(log) > cap:
        log = log[-cap:]
    try:
        Path(path).write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"journal write error: {e}")


# ── 交易日誌績效統計（純邏輯，離線可測）────────────────────────────────────────

def journal_win_stats(journal: list[dict], price_at, price_now) -> dict:
    """
    從交易日誌的「買進」訊號回算實際前進報酬（mark-to-market），對照回測。

    journal:   [{time, symbol, side, qty, score, submitted, ...}]
    price_at(symbol, time_iso) -> 進場價 or None（呼叫端提供，通常取當日/次日收盤）
    price_now(symbol) -> 最新價 or None

    只計 side=='buy' 且 submitted==True 的訊號。回:
      {trades, win_rate, avg_return, best, worst, per_trade:[...]}
    win = 前進報酬 > 0。這是「訊號發出後有沒有上漲」的實測，非精確已實現損益
    （未配對賣出、未計費用），但可直接與回測期望對照。
    """
    per = []
    for e in journal:
        if e.get("side") != "buy" or not e.get("submitted"):
            continue
        sym, t = e.get("symbol"), e.get("time")
        entry = price_at(sym, t) if sym and t else None
        now = price_now(sym) if sym else None
        if entry and now and entry > 0:
            ret = now / entry - 1
            per.append({"symbol": sym, "time": t, "entry": round(entry, 2),
                        "now": round(now, 2), "return": ret,
                        "score": e.get("score"), "win": ret > 0})
    n = len(per)
    if n == 0:
        return {"trades": 0, "win_rate": None, "avg_return": None,
                "best": None, "worst": None, "per_trade": []}
    rets = [p["return"] for p in per]
    wins = sum(1 for r in rets if r > 0)
    return {
        "trades": n,
        "win_rate": wins / n,
        "avg_return": sum(rets) / n,
        "best": max(per, key=lambda p: p["return"]),
        "worst": min(per, key=lambda p: p["return"]),
        "per_trade": per,
    }


# ── 純決策邏輯（離線可測）──────────────────────────────────────────────────────

def decide_orders(scored: list[dict], positions: dict, equity: float,
                  buying_power: float, config: dict | None = None) -> list[dict]:
    """
    給定評分、目前持倉、帳戶淨值與購買力，決定要下哪些單（純函數）。

    ⚠️ Legacy：bot 端已改用 trade_engine.decide（分層引擎：出場與訊號脫鉤、
    追蹤停損、保險絲），此函數僅作為引擎故障時的退回路徑保留。

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

    equity = float(equity)                    # 防呆：Alpaca JSON 可能是字串
    score_map = {s["ticker"]: s for s in scored}
    held = set(positions.keys())
    orders: list[dict] = []

    # 1) 出場：持倉評分 ≤ 出場門檻 → 平倉
    for sym, pos in positions.items():
        s = score_map.get(sym)
        if s is not None:
            sc = float(s.get("score", 0))
            if sc <= exit_th:
                qty = abs(float(pos.get("qty", 0)))
                if qty > 0:
                    orders.append({"symbol": sym, "side": "sell", "qty": qty,
                                   "reason": f"評分 {sc:+.2f} ≤ 出場門檻 {exit_th}"})
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
                               "reason": f"評分 {float(s['score']):+.2f} ≥ 買進門檻 {buy_th}"})
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
    # 安全鎖：只接受 paper API。若有人把 ALPACA_BASE_URL 設成正式盤（真錢），
    # 靜默退回 paper——本專案的 /autotrade、/closeall 絕不允許碰真實帳戶。
    b = os.environ.get("ALPACA_BASE_URL", PAPER_BASE).rstrip("/")
    if "paper-api" not in b:
        print(f"ALPACA_BASE_URL '{b}' 非 paper API，已強制退回 {PAPER_BASE}")
        return PAPER_BASE
    return b


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


def submit_bracket(key: str, secret: str, symbol: str, qty: int,
                   limit_price: float, stop_price: float,
                   target_price: float) -> tuple[bool, str]:
    """Bracket 買單（限價進場 + 自動掛停損/停利），time_in_force=day。
    僅 paper API（_base() 安全鎖）。回 (成功, 訂單 id 或錯誤訊息)。
    價格規則：≥$1 標的限價最多 2 位小數；bracket 不允許碎股 → qty 取整。"""
    import requests
    qty = int(qty)

    def _px(p: float) -> float:
        # sub-penny 規則：≥$1 最多 2 位小數、<$1 最多 4 位（違反會 422）
        return round(float(p), 2 if p >= 1 else 4)

    lp, sp, tp = _px(limit_price), _px(stop_price), _px(target_price)
    # Alpaca 要求 TP ≥ 進場+0.01、SL ≤ 進場−0.01；用「分」比較避免浮點誤拒
    lp_c, sp_c, tp_c = round(lp * 100), round(sp * 100), round(tp * 100)
    if qty <= 0 or not (sp_c <= lp_c - 1 and tp_c >= lp_c + 1):
        return False, (f"參數不合法：qty={qty}，需 stop≤limit-0.01≤target-0.02"
                       f"（rounding 後 {sp}/{lp}/{tp}）")
    try:
        payload = {
            "symbol": symbol.upper(), "qty": str(qty), "side": "buy",
            "type": "limit", "limit_price": str(lp),
            "time_in_force": "day", "order_class": "bracket",
            "take_profit": {"limit_price": str(tp)},
            "stop_loss": {"stop_price": str(sp)},
        }
        r = requests.post(f"{_base()}/v2/orders", headers=_headers(key, secret),
                          json=payload, timeout=20)
        if r.ok:
            return True, (r.json() or {}).get("id", "OK")
        return False, f"{r.status_code}: {r.text[:200]}"
    except Exception as e:
        return False, str(e)


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

    print("\n=== submit_bracket 參數驗證（不打網路）===")
    ok1, _m1 = submit_bracket("k", "s", "AAPL", 0, 100.0, 98.0, 104.0)      # qty=0
    ok2, _m2 = submit_bracket("k", "s", "AAPL", 10, 100.0, 99.995, 100.005)  # 間距不足
    ok3, _m3 = submit_bracket("k", "s", "AAPL", 10, 100.0, 104.0, 98.0)     # stop/target 顛倒
    assert not ok1 and not ok2 and not ok3, (ok1, ok2, ok3)
    print("  不合法參數全數擋下 OK（qty=0 / 0.01 間距 / 顛倒）")
