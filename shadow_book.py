"""
shadow_book.py – Shadow 對照帳本（純邏輯、離線可測）

讓「舊版決策邏輯」（alpaca_trader.decide_orders，單一分數進出場、無 Alpha 疊加、
無相關性控制）以虛擬帳本平行記帳——不真下單，只記錄「如果沒升級，現在會怎樣」。

比較口徑刻意乾淨：
  • 影子帳吃「疊加前的原始評分」——舊系統當年就是這樣看世界的
  • 影子帳與真帳在啟動日對齊起始淨值，之後各走各的
  • 成交一律以當輪掃描價成交（無滑價；兩邊同樣樂觀，比較仍公平）
  • 每輪 autotrade 內更新（開盤時段、與真單同節奏）

這回答一個任何回測都答不了的問題：**引擎重製到底值多少錢**——
因為兩邊吃的是完全相同的即時訊號流。/shadow 查看對照。

state["shadow"] 結構：
  {started, start_real_equity, cash, positions: {sym: {qty, entry}},
   last_px: {sym: px}, last: {date, equity, real_equity}}
"""

from __future__ import annotations


def init_book(real_equity: float, today: str) -> dict:
    return {"started": str(today)[:10],
            "start_real_equity": float(real_equity),
            "cash": float(real_equity),
            "positions": {}, "last_px": {},
            "last": None}


def book_equity(book: dict, prices: dict | None = None) -> float:
    """cash + Σ qty×價。價優先用本輪 prices，缺的用 last_px，再缺用進場價。"""
    prices = prices or {}
    eq = float(book.get("cash", 0))
    for sym, p in (book.get("positions") or {}).items():
        px = prices.get(sym) or (book.get("last_px") or {}).get(sym) or p.get("entry") or 0
        eq += float(p.get("qty", 0)) * float(px)
    return eq


def apply_orders(book: dict, orders: list[dict], prices: dict) -> list[str]:
    """把 decide_orders 的訂單記進帳本（成交價=當輪掃描價）。回執行備註。"""
    notes = []
    pos = book.setdefault("positions", {})
    for o in orders:
        sym, side = o["symbol"], o["side"]
        px = prices.get(sym)
        if not px or px <= 0:
            continue
        qty = float(o.get("qty", 0))
        if qty < 1:
            continue
        if side == "buy":
            cost = qty * px
            if cost > book["cash"] + 1e-6:       # 防呆（decide_orders 已用 cash 當 bp）
                qty = int(book["cash"] / px)
                cost = qty * px
            if qty < 1:
                continue
            book["cash"] -= cost
            cur = pos.get(sym)
            if cur:                               # 加碼 → 均價
                tot = cur["qty"] + qty
                cur["entry"] = (cur["entry"] * cur["qty"] + px * qty) / tot
                cur["qty"] = tot
            else:
                pos[sym] = {"qty": qty, "entry": px}
            notes.append(f"買 {sym} x{int(qty)} @ {px:.2f}")
        else:
            cur = pos.get(sym)
            if not cur:
                continue
            sell_qty = min(qty, cur["qty"])
            book["cash"] += sell_qty * px
            cur["qty"] -= sell_qty
            if cur["qty"] < 1:
                del pos[sym]
            notes.append(f"賣 {sym} x{int(sell_qty)} @ {px:.2f}")
    return notes


def run_shadow(state: dict, scored_raw: list[dict], config: dict,
               real_equity: float, today: str, decide_fn=None) -> None:
    """
    每輪 autotrade 呼叫：用舊決策邏輯對影子帳本記帳。就地更新 state["shadow"]。
    decide_fn 可注入（測試用）；預設 alpaca_trader.decide_orders。
    """
    if decide_fn is None:
        import alpaca_trader as at
        decide_fn = at.decide_orders
    book = state.get("shadow")
    if not isinstance(book, dict) or "cash" not in book:
        book = init_book(real_equity, today)
        state["shadow"] = book

    prices = {s["ticker"]: float(s["price"]) for s in scored_raw
              if s.get("ticker") and s.get("price")}
    pos_view = {sym: {"qty": p["qty"]} for sym, p in
                (book.get("positions") or {}).items()}
    eq = book_equity(book, prices)
    orders = decide_fn(scored_raw, pos_view, eq, book["cash"], config) or []
    apply_orders(book, orders, prices)

    lp = book.setdefault("last_px", {})
    for sym in list(book.get("positions") or {}):
        if sym in prices:
            lp[sym] = prices[sym]
    for sym in [s for s in lp if s not in (book.get("positions") or {})]:
        del lp[sym]
    book["last"] = {"date": str(today)[:10],
                    "equity": round(book_equity(book, prices), 2),
                    "real_equity": round(float(real_equity), 2)}


def shadow_text(book: dict | None, real_equity: float | None = None) -> str:
    """/shadow 指令文字（Telegram legacy Markdown：單 *、避免底線）。"""
    if not book or "cash" not in book:
        return ("👥 *Shadow 對照* 尚未啟動\n"
                "開 /autotrade on 後，舊版決策邏輯會以虛擬帳本平行記帳，"
                "量化新引擎的增量價值")
    start = float(book.get("start_real_equity") or 0)
    last = book.get("last") or {}
    sh_eq = last.get("equity") or book_equity(book)
    re_eq = real_equity if real_equity is not None else last.get("real_equity")
    lines = [f"👥 *Shadow 對照*（自 {book.get('started')} 起）",
             "新引擎＝真帳；影子帳＝舊邏輯（同一訊號流、無疊加層）"]
    if start > 0 and sh_eq:
        sh_ret = sh_eq / start - 1
        lines.append(f"👻 影子帳（舊邏輯）：${sh_eq:,.0f}（{sh_ret:+.2%}）")
        if re_eq:
            re_ret = float(re_eq) / start - 1
            edge = re_ret - sh_ret
            icon = "✅" if edge >= 0 else "⚠️"
            lines.append(f"🤖 真帳（新引擎）：${float(re_eq):,.0f}（{re_ret:+.2%}）")
            lines.append(f"{icon} 引擎增量：{edge:+.2%}"
                         + ("（新引擎領先）" if edge >= 0 else "（舊邏輯領先——值得檢討）"))
    npos = len(book.get("positions") or {})
    lines.append(f"影子持倉 {npos} 檔"
                 + ("：" + "、".join(sorted((book.get("positions") or {}).keys()))
                    if npos else ""))
    lines.append("同輪同價成交、無滑價；樣本 < 1 個月時勿下結論")
    return "\n".join(lines)


# ── 自我測試 ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    T = "2026-07-20"

    # 1) 初始化 + 買進記帳
    st = {}
    scored = [{"ticker": "AAA", "score": 0.9, "price": 100.0},
              {"ticker": "BBB", "score": 0.1, "price": 50.0}]

    def fake_decide(sc, pos, eq, bp, cfg):
        return [{"symbol": "AAA", "side": "buy", "qty": 100, "reason": "x"}]

    run_shadow(st, scored, {}, 100000.0, T, decide_fn=fake_decide)
    b = st["shadow"]
    assert b["cash"] == 90000.0 and b["positions"]["AAA"]["qty"] == 100
    assert b["last"]["equity"] == 100000.0        # 現金 9 萬 + 持股 1 萬
    print("✅ 1 初始化+買進")

    # 2) mark-to-market：價格上漲反映在 equity；缺價用 last_px
    scored2 = [{"ticker": "AAA", "score": 0.5, "price": 110.0}]
    run_shadow(st, scored2, {}, 101000.0, T, decide_fn=lambda *a: [])
    assert st["shadow"]["last"]["equity"] == 101000.0     # 9 萬 + 100×110
    run_shadow(st, [], {}, 101000.0, T, decide_fn=lambda *a: [])
    assert st["shadow"]["last"]["equity"] == 101000.0     # 缺價 → last_px 110
    print("✅ 2 mark-to-market + last_px")

    # 3) 賣出：回收現金、清持倉、last_px 跟著清
    def fake_sell(sc, pos, eq, bp, cfg):
        return [{"symbol": "AAA", "side": "sell", "qty": 100, "reason": "x"}]

    run_shadow(st, [{"ticker": "AAA", "score": -0.5, "price": 120.0}],
               {}, 102000.0, T, decide_fn=fake_sell)
    b = st["shadow"]
    assert b["positions"] == {} and b["cash"] == 102000.0 and b["last_px"] == {}
    print("✅ 3 賣出清帳")

    # 4) 現金防呆：買超過現金 → 縮量
    st2 = {}
    run_shadow(st2, [{"ticker": "XXX", "score": 0.9, "price": 100.0}],
               {}, 1000.0, T,
               decide_fn=lambda *a: [{"symbol": "XXX", "side": "buy", "qty": 999}])
    assert st2["shadow"]["positions"]["XXX"]["qty"] == 10
    assert st2["shadow"]["cash"] == 0.0
    print("✅ 4 現金防呆縮量")

    # 5) 加碼均價；賣出部位不存在 → 忽略
    st3 = {}
    run_shadow(st3, [{"ticker": "Y", "score": 0.9, "price": 100.0}], {}, 10000.0, T,
               decide_fn=lambda *a: [{"symbol": "Y", "side": "buy", "qty": 10}])
    run_shadow(st3, [{"ticker": "Y", "score": 0.9, "price": 120.0}], {}, 10000.0, T,
               decide_fn=lambda *a: [{"symbol": "Y", "side": "buy", "qty": 10},
                                     {"symbol": "Z", "side": "sell", "qty": 5}])
    p = st3["shadow"]["positions"]["Y"]
    assert p["qty"] == 20 and abs(p["entry"] - 110.0) < 1e-9
    print("✅ 5 加碼均價 + 幽靈賣單忽略")

    # 6) 真 decide_orders 串接（舊邏輯出場：分數 ≤ -0.2 全平）
    import alpaca_trader as at
    st4 = {}
    run_shadow(st4, [{"ticker": "W", "score": 0.9, "price": 100.0,
                      "risk_per_share": 5.0}], {}, 100000.0, T,
               decide_fn=at.decide_orders)
    assert st4["shadow"]["positions"].get("W"), st4["shadow"]
    run_shadow(st4, [{"ticker": "W", "score": -0.5, "price": 95.0}],
               {}, 100000.0, T, decide_fn=at.decide_orders)
    assert "W" not in st4["shadow"]["positions"]          # 舊邏輯：訊號衰減即平倉
    print("✅ 6 舊邏輯端到端（衰減即平——正是要對照的行為）")

    # 7) 文字輸出（新引擎領先/落後兩態；未啟動）
    txt = shadow_text(st4["shadow"], real_equity=105000.0)
    assert "Shadow 對照" in txt and "引擎增量" in txt, txt
    assert shadow_text(None).count("*") % 2 == 0
    assert txt.count("_") == 0, txt                        # TG legacy Markdown 安全
    print("✅ 7 shadow_text")
    print()
    print(txt)
    print("\nshadow_book selftest OK ✅")
