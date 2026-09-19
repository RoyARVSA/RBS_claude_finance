"""
lanes.py – 多線平行帳（策略車道）：同一輪訊號下，讓「舊跑法」與「新跑法」各自記帳、並排比較

鏡像帳（mirror_book）回答「引擎接管我的實倉會怎樣」；車道回答「候選池／meta 尺寸該不該開」——
在旗標還關著的時候，就用同一台引擎、同一輪 scored/config/regime，各跑一本 10 萬起始的虛擬帳：

  base       watchlist 候選、等額部位（＝真帳現行跑法）
  pool       + Alpha 脊椎候選池（rank.json 前 k 名；閘門沒過時 = base）
  pool_meta  + meta-labeling 勝率倍數（模型沒過閘門時 = pool）

每輪 cron：decide → 以掃描價成交（單邊 0.05% 成本）→ 每日淨值一點；殭屍倉防護同鏡像帳。
state["lanes"] 加密（虛擬持倉 ≈ 引擎當下會持有什麼，與真帳高度相關）。/lanes 看比較表；不推播每筆單。
純邏輯、離線可測。
"""

from __future__ import annotations

START_EQUITY = 100_000.0
COST_SIDE = 0.0005
HISTORY_CAP = 400
LANE_DEFS = {                      # name → (用候選池, 用 meta 倍數)
    "base": (False, False),
    "pool": (True, False),
    "pool_meta": (True, True),
}
LANE_LABELS = {"base": "現行（watchlist）", "pool": "＋候選池", "pool_meta": "＋候選池＋meta 部位"}


def new_lanes(today: str) -> dict:
    return {"started": str(today)[:10], "start_equity": START_EQUITY, "spy_start": None,
            "lanes": {name: {"cash": START_EQUITY, "positions": {}, "last_px": {}, "engine": None,
                             "history": [], "n_trades": 0, "last": None} for name in LANE_DEFS}}


def lane_scored(scored: list[dict], use_pool: bool, use_meta: bool) -> list[dict]:
    """依車道定義過濾/剝除欄位：不含候選池列、或不帶 meta 倍數（純函數，回新列）。"""
    out = []
    for s in scored:
        if not use_pool and s.get("pool"):
            continue
        s2 = dict(s)
        if not use_meta:
            s2.pop("meta_mult", None)
            s2.pop("meta_p", None)
        out.append(s2)
    return out


def _apply(book: dict, orders: list[dict], prices: dict) -> int:
    """以掃描價成交、含單邊成本；回成交筆數。"""
    import shadow_book as sb
    n = 0
    pos = book.setdefault("positions", {})
    for o in orders:
        px = prices.get(o["symbol"])
        if not px or px <= 0:
            continue
        before_cash, before_pos = book["cash"], dict(pos.get(o["symbol"]) or {})
        sb.apply_orders(book, [o], prices)
        if book["cash"] != before_cash or (pos.get(o["symbol"]) or {}) != before_pos:
            traded = abs(book["cash"] - before_cash)
            book["cash"] -= traded * COST_SIDE
            n += 1
    return n


def run_lanes(state: dict, scored: list[dict], config: dict, regime: str | None, today: str,
              meta_usable: bool = False, pool_available: bool = False, real_equity: float | None = None) -> dict:
    """
    每輪呼叫。scored：完整列（含候選池列 pool=True、meta_mult/meta_p 已附）；config：與真帳同。
    就地更新 state["lanes"]；回 {name: n_filled}。任何車道炸掉不影響其他車道。
    """
    import shadow_book as sb
    import trade_engine as te
    from shadow_book import STALE_CLOSE_DAYS, _days_between

    L = state.get("lanes")
    if not isinstance(L, dict) or "lanes" not in L:
        L = new_lanes(today)
        state["lanes"] = L
    prices = {s["ticker"]: float(s["price"]) for s in scored if s.get("ticker") and s.get("price")}
    if L.get("spy_start") is None and prices.get("SPY"):
        L["spy_start"] = prices["SPY"]
    if prices.get("SPY"):
        L["spy_last"] = prices["SPY"]
    if real_equity:
        L.setdefault("real_start_equity", float(real_equity))       # 真帳同期基準（車道起算當下的 Alpaca 淨值）
        L["real_last_equity"] = float(real_equity)
    L["flags"] = {"pool_available": bool(pool_available), "meta_usable": bool(meta_usable)}
    filled: dict[str, int] = {}
    for name, (use_pool, use_meta) in LANE_DEFS.items():
        book = L["lanes"].setdefault(name, {"cash": START_EQUITY, "positions": {}, "last_px": {}, "engine": None,
                                            "history": [], "n_trades": 0, "last": None})
        try:
            rows = lane_scored(scored, use_pool, use_meta)
            pos_book = book.setdefault("positions", {})
            for sym, p in list(pos_book.items()):                     # 殭屍倉：連續 N 天無報價 → 凍結價強平
                if sym in prices:
                    p.pop("stale_since", None)
                    continue
                p.setdefault("stale_since", str(today)[:10])
                if _days_between(p["stale_since"], today) >= STALE_CLOSE_DAYS:
                    px = (book.get("last_px") or {}).get(sym) or p.get("entry") or 0
                    book["cash"] += float(p.get("qty", 0)) * float(px)
                    del pos_book[sym]
                    (book.get("last_px") or {}).pop(sym, None)
            pos_view = {}
            for sym, p in pos_book.items():
                px = prices.get(sym) or (book.get("last_px") or {}).get(sym) or p["entry"]
                pos_view[sym] = {"qty": p["qty"], "avg_entry_price": p["entry"], "market_value": p["qty"] * px,
                                 "unrealized_pl": p["qty"] * (px - p["entry"]),
                                 "unrealized_plpc": (px / p["entry"] - 1) if p["entry"] else 0}
            equity = sb.book_equity(book, prices)
            orders, eng, _notes = te.decide(rows, pos_view, equity, book["cash"], book.get("engine"), regime, config, today)
            book["engine"] = eng
            n = _apply(book, [o for o in orders if prices.get(o["symbol"])], prices)
            book["n_trades"] = int(book.get("n_trades", 0)) + n
            lp = book.setdefault("last_px", {})
            for sym in list(pos_book):
                if sym in prices:
                    lp[sym] = prices[sym]
            for sym in [s for s in lp if s not in pos_book]:
                del lp[sym]
            eq = round(sb.book_equity(book, prices), 2)
            book["last"] = {"date": str(today)[:10], "equity": eq}
            hist = book.setdefault("history", [])
            if hist and hist[-1].get("date") == book["last"]["date"]:
                hist[-1] = dict(book["last"])
            else:
                hist.append(dict(book["last"]))
            if len(hist) > HISTORY_CAP:
                del hist[:-HISTORY_CAP]
            filled[name] = n
            book.pop("error", None)
        except Exception as e:            # 單一車道失敗不影響其他車道
            book["error"] = f"{type(e).__name__}"
            filled[name] = 0
    return filled


def lane_stats(book: dict, start_equity: float) -> dict:
    hist = book.get("history") or []
    eq = (book.get("last") or {}).get("equity")
    ret = (eq / start_equity - 1) if eq else None
    peak, dd = start_equity, 0.0
    for h in hist:
        v = float(h.get("equity") or 0)
        peak = max(peak, v)
        if peak > 0:
            dd = min(dd, v / peak - 1)
    return {"equity": eq, "ret": ret, "max_dd": dd, "n_pos": len(book.get("positions") or {}),
            "n_trades": int(book.get("n_trades", 0)), "days": len(hist), "error": book.get("error")}


def lanes_text(state: dict, real_equity: float | None = None, real_start: float | None = None) -> str:
    L = state.get("lanes")
    if not isinstance(L, dict) or "lanes" not in L:
        return "🛣 多線平行帳尚未開始（autotrade 第一輪後自動建立；`/set lanes_enabled on`）"
    st = float(L.get("start_equity") or START_EQUITY)
    lines = [f"🛣 *多線平行帳*（{L.get('started')} 起、各 {st:,.0f} 起始；同一台引擎、同一輪訊號）"]
    fl = L.get("flags") or {}
    for name in LANE_DEFS:
        b = L["lanes"].get(name) or {}
        s = lane_stats(b, st)
        tag = ""
        if name == "pool" and not fl.get("pool_available"):
            tag = "（候選池閘門未過＝現行）"
        if name == "pool_meta" and not fl.get("meta_usable"):
            tag = "（meta 未過閘門＝候選池）"
        if s["equity"] is None:
            lines.append(f"• {LANE_LABELS[name]}{tag}：尚無資料")
            continue
        lines.append(f"• {LANE_LABELS[name]}{tag}：{s['ret']:+.2%}｜回撤 {s['max_dd']:.1%}｜"
                     f"持倉 {s['n_pos']}｜成交 {s['n_trades']}｜{s['days']} 日" + (f"｜⚠️ 上輪錯誤 {s['error']}" if s.get("error") else ""))
    if L.get("spy_start") and L.get("spy_last"):
        lines.append(f"• SPY 同期：{L['spy_last'] / L['spy_start'] - 1:+.2%}")
    real_equity = real_equity or L.get("real_last_equity")
    real_start = real_start or L.get("real_start_equity")
    if real_equity and real_start:
        lines.append(f"• 真帳同期（Alpaca）：{real_equity / real_start - 1:+.2%}")
    lines.append("_虛擬帳、掃描價成交含 0.05% 成本；候選池/meta 旗標未開時，車道就是「如果開了會怎樣」。非投資建議_")
    return "\n".join(lines)


def reset(state: dict, today: str) -> None:
    state["lanes"] = new_lanes(today)


# ── 自我測試 ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    T = "2026-09-21"
    base = [{"ticker": "AAA", "score": 0.8, "price": 100.0, "risk_per_share": 10.0},     # rps 10 → 風險預算綁定（100 股 < 15% 上限）
            {"ticker": "SPY", "score": 0.1, "price": 500.0, "risk_per_share": 5.0}]
    pool = [{"ticker": "PPP", "score": 0.9, "price": 50.0, "risk_per_share": 2.0, "pool": True, "meta_mult": 0.0, "meta_p": 0.2},
            {"ticker": "QQQ2", "score": 0.7, "price": 20.0, "risk_per_share": 1.0, "pool": True, "meta_mult": 1.25, "meta_p": 0.7}]
    scored = [dict(base[0], meta_mult=0.5, meta_p=0.45), base[1]] + pool
    # 1) 車道過濾
    assert [s["ticker"] for s in lane_scored(scored, False, False)] == ["AAA", "SPY"]
    assert all("meta_mult" not in s for s in lane_scored(scored, True, False))
    assert any(s.get("meta_mult") == 0.0 for s in lane_scored(scored, True, True))
    print("✅ 1 車道過濾（候選池列／meta 欄）")

    # 2) 三條車道各自成交：base 只買 AAA；pool 多買 PPP/QQQ2；pool_meta 跳過 PPP（meta 0）且 AAA 半倉
    st: dict = {}
    filled = run_lanes(st, scored, {"max_positions": 10}, "risk_on", T, meta_usable=True, pool_available=True, real_equity=95_000)
    L = st["lanes"]["lanes"]
    assert set(L["base"]["positions"]) == {"AAA"} and filled["base"] == 1
    assert set(L["pool"]["positions"]) == {"AAA", "PPP", "QQQ2"}
    assert set(L["pool_meta"]["positions"]) == {"AAA", "QQQ2"}                     # PPP meta_mult 0 → 跳過
    assert L["pool_meta"]["positions"]["AAA"]["qty"] == int(L["base"]["positions"]["AAA"]["qty"] * 0.5)
    assert L["base"]["cash"] < START_EQUITY and all(b["last"]["date"] == T for b in L.values())
    assert st["lanes"]["spy_start"] == 500.0
    print("✅ 2 三車道各自記帳（base/pool/pool_meta 持倉不同、meta 半倉/跳過）")

    # 3) 第二輪：價格上漲 → 淨值歷史新增一點；SPY 同期；殭屍倉（PPP 消失 5 天）強平
    up = [dict(s, price=s["price"] * 1.05) for s in scored if s["ticker"] != "PPP"]
    run_lanes(st, up, {"max_positions": 10}, "risk_on", "2026-09-22", True, True)
    assert L["base"]["last"]["equity"] > START_EQUITY and len(L["base"]["history"]) == 2
    assert "stale_since" in L["pool"]["positions"]["PPP"]
    run_lanes(st, up, {"max_positions": 10}, "risk_on", "2026-09-28", True, True)
    assert "PPP" not in L["pool"]["positions"]                                        # 5 天無報價 → 凍結價強平
    run_lanes(st, up, {"max_positions": 10}, "risk_on", "2026-09-28", True, True, real_equity=96_900)
    txt = lanes_text(st)
    assert "多線平行帳" in txt and "SPY 同期" in txt and "真帳同期（Alpaca）：+2.00%" in txt and "**" not in txt
    print(txt)
    s2 = lane_stats(L["base"], START_EQUITY)
    assert s2["ret"] > 0 and s2["max_dd"] <= 0 and s2["n_trades"] >= 1
    # 4) 旗標未開的標籤；reset；壞列不炸
    st2: dict = {}
    run_lanes(st2, scored, {}, None, T, meta_usable=False, pool_available=False)
    t2 = lanes_text(st2); assert "閘門未過" in t2 and "meta 未過" in t2
    reset(st2, T); assert st2["lanes"]["lanes"]["base"]["positions"] == {}
    run_lanes(st2, [{"ticker": "BAD", "score": "x", "price": None}], {}, None, T)     # 壞列：decide 過濾或例外都不炸
    assert "lanes" in st2
    print("✅ 3 淨值歷史／殭屍倉／文字／reset／壞列")
    print("\nlanes selftest OK ✅")
