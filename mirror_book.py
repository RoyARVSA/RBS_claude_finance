"""
mirror_book.py – 鏡像帳本（模式 A 接管模擬；純邏輯、離線可測）

回答一個 Alpaca 真帳答不了的問題：**「如果把我目前的實際布建與資金量
交給引擎管，它會怎麼操作、績效如何」**。

用法（Telegram）：
  /mirror init 25000 AAPL:10:180.5 TSLA:5:250   ← 現金 + 各檔 代碼:股數:成本
  /mirror                                        ← 查看淨值/持倉/報酬
  /mirror reset                                  ← 刪除帳本（重建前先 reset）

設計：
  • 初始化後引擎**完全自主**（模式 A）——你的真實買賣不會同步進來；
    對照的意義正是「你操作的真帳 vs 引擎操作的鏡像帳」
  • 完整引擎堆疊：與 Alpaca 真帳吃同一份 scored（含 Alpha 疊加/相關性
    縮量）、同 config、同 regime——差別只在起點是你的實倉
  • 獨立引擎簿記（peak/停損事件/保險絲自成一格，state["mirror"]["engine"]）；
    你的原持倉會被引擎「收養」：entry=你的成本、停損距離=5% 預設
  • 記帳復用 shadow_book（apply_orders/book_equity——已對抗驗證過的機件）
  • 掃描清單自動聯集 mirror 持倉（引擎對你的持股不失明——殭屍倉教訓）
  • state["mirror"] 屬加密區（STATE_ENC_KEY）：實倉與資金量不落公開明文
  • start_equity = 現金 + Σ(股數×成本)——報酬含你原持倉的後續帳面變動

誠實邊界：成交=當輪掃描價、無滑價（與 shadow 同口徑）；美股 only。
"""

from __future__ import annotations

JOURNAL_CAP = 200


def parse_holdings(tokens: list[str]) -> tuple[list, list]:
    """解析 ["AAPL:10:180.5", ...] → ([(sym, qty, cost)], 錯誤清單)。
    股數 <1 拒收（引擎不管零股，收了會靜默凍結佔名額——對抗驗證 Med-3）；
    重複 symbol 合併為加權均價（覆蓋會虛報虧損——Med-2）。"""
    out, bad = [], []
    for t in tokens:
        try:
            sym, qty, cost = t.split(":")
            sym = sym.strip().upper()
            qty_f, cost_f = float(qty), float(cost)
            if not sym or not sym.isalnum() or qty_f < 1 or cost_f <= 0:
                raise ValueError
            out.append((sym, qty_f, cost_f))
        except ValueError:
            bad.append(t)
    merged: dict = {}
    order = []
    for sym, q, c in out:
        if sym in merged:
            q0, c0 = merged[sym]
            merged[sym] = (q0 + q, (q0 * c0 + q * c) / (q0 + q))
        else:
            merged[sym] = (q, c)
            order.append(sym)
    return [(s, merged[s][0], round(merged[s][1], 4)) for s in order], bad


def init_book(cash: float, holdings: list, today: str) -> dict:
    """holdings: [(sym, qty, cost)]。start_equity = 現金 + 成本市值。"""
    positions = {s: {"qty": float(q), "entry": float(c)} for s, q, c in holdings}
    start = float(cash) + sum(q * c for _, q, c in holdings)
    return {"started": str(today)[:10], "start_equity": round(start, 2),
            "cash": float(cash), "positions": positions,
            "last_px": {}, "last": None, "engine": None, "journal": []}


def holdings_symbols(state: dict) -> list[str]:
    """掃描清單聯集用：mirror 持倉的標的（未初始化回空）。"""
    m = state.get("mirror")
    if not isinstance(m, dict):
        return []
    return sorted((m.get("positions") or {}).keys())


def run_mirror(state: dict, scored: list[dict], config: dict,
               regime: str | None, today: str) -> list[str]:
    """
    每輪 cron 呼叫（與真帳同 scored/config/regime）。就地更新 state["mirror"]。
    回傳給 Telegram 的訊息行（無帳本/無動作 → 空 list）。
    """
    import shadow_book as sb
    import trade_engine as te

    m = state.get("mirror")
    if not isinstance(m, dict) or "cash" not in m:
        return []

    prices = {s["ticker"]: float(s["price"]) for s in scored
              if s.get("ticker") and s.get("price")}

    # 殭屍倉防護（對抗驗證 Med-1，同 shadow_book 設計）：持倉連續 5 天無報價
    # →凍結價強平。不然 decide 每輪用凍結價產生幽靈賣單、apply 因無價 skip、
    # 迴圈永不終止（production=每 15 分鐘推播騷擾一次）
    from shadow_book import STALE_CLOSE_DAYS, _days_between
    pos_book = m.setdefault("positions", {})
    stale_notes = []
    for sym, p in list(pos_book.items()):
        if sym in prices:
            p.pop("stale_since", None)
            continue
        p.setdefault("stale_since", str(today)[:10])
        if _days_between(p["stale_since"], today) >= STALE_CLOSE_DAYS:
            px = (m.get("last_px") or {}).get(sym) or p.get("entry") or 0
            m["cash"] += float(p.get("qty", 0)) * float(px)
            del pos_book[sym]
            (m.get("last_px") or {}).pop(sym, None)
            stale_notes.append(f"🪞 {sym} 連續 {STALE_CLOSE_DAYS} 天無報價 → "
                               f"凍結價 {px:.2f} 強制平倉")

    # 引擎視角的「broker 持倉」＝鏡像帳本身（含 market_value 供 price_of 退路）
    pos_view = {}
    for sym, p in (m.get("positions") or {}).items():
        px = prices.get(sym) or (m.get("last_px") or {}).get(sym) or p["entry"]
        pos_view[sym] = {"qty": p["qty"], "avg_entry_price": p["entry"],
                         "market_value": p["qty"] * px,
                         "unrealized_pl": p["qty"] * (px - p["entry"]),
                         "unrealized_plpc": (px / p["entry"] - 1) if p["entry"] else 0}

    equity = sb.book_equity(m, prices)
    orders, eng, notes = te.decide(scored, pos_view, equity, m["cash"],
                                   m.get("engine"), regime, config, today)
    m["engine"] = eng
    # 只認「有報價、實際會成交」的單——無價幽靈單不入帳/不入 journal/不推播
    # （對抗驗證 Med-1：apply_orders 對無價單 skip，記意圖會與帳本脫節）
    filled = [o for o in orders if prices.get(o["symbol"])]
    sb.apply_orders(m, filled, prices)

    lp = m.setdefault("last_px", {})
    for sym in list(m.get("positions") or {}):
        if sym in prices:
            lp[sym] = prices[sym]
    for sym in [s for s in lp if s not in (m.get("positions") or {})]:
        del lp[sym]
    m["last"] = {"date": str(today)[:10],
                 "equity": round(sb.book_equity(m, prices), 2)}

    lines = []
    if filled or stale_notes:
        j = m.setdefault("journal", [])
        for o in filled:
            j.append({"date": str(today)[:10], "symbol": o["symbol"],
                      "side": o["side"], "qty": o["qty"],
                      "price": prices.get(o["symbol"]),
                      "mechanism": o.get("mechanism"), "reason": o["reason"]})
        if len(j) > JOURNAL_CAP:
            m["journal"] = j[-JOURNAL_CAP:]
        lines.append("🪞 *鏡像帳*（引擎管理你的實倉起點）")
        for o in filled:
            lines.append(f"・{o['side'].upper()} {o['symbol']} x{int(o['qty'])} — {o['reason']}")
        lines.extend(stale_notes)
        # 保險絲/曝險級事件值得推播（對抗驗證 Low-4）
        lines.extend(n for n in notes if n.startswith("🚨"))
    return lines


def to_std_journal(m: dict | None) -> list[dict]:
    """mirror journal → attribution/behavior_check 相容格式（純函數）。
    mirror 條目用 date 且必然成交；標準格式要 time/submitted。"""
    if not isinstance(m, dict):
        return []
    out = []
    for e in (m.get("journal") or []):
        if not isinstance(e, dict) or not e.get("symbol"):
            continue
        out.append({"time": f"{e.get('date', '')}T15:00:00Z",
                    "symbol": e["symbol"], "side": e.get("side"),
                    "qty": e.get("qty"), "price": e.get("price"),
                    "mechanism": e.get("mechanism"), "reason": e.get("reason"),
                    "submitted": True})
    return out


def mirror_text(state: dict) -> str:
    """/mirror 顯示（Telegram legacy Markdown：單 *、無底線）。"""
    m = state.get("mirror")
    if not isinstance(m, dict) or "cash" not in m:
        return ("🪞 *鏡像帳未初始化*\n"
                "用你目前的實際布建啟動：\n"
                "`/mirror init 現金 代碼:股數:成本 ...`\n"
                "例：`/mirror init 25000 AAPL:10:180.5 NVDA:20:150`\n"
                "初始化後引擎完全自主操作（你的真實買賣不會同步進來），"
                "資料存加密區")
    import shadow_book as sb
    last = m.get("last") or {}
    eq = last.get("equity") or sb.book_equity(m)
    start = float(m.get("start_equity") or 0)
    lines = [f"🪞 *鏡像帳*（{m.get('started')} 起由引擎自主管理）"]
    if start > 0:
        lines.append(f"淨值 ${eq:,.0f}（{eq / start - 1:+.2%}，起始 ${start:,.0f}）"
                     f"　現金 ${m.get('cash', 0):,.0f}")
    pos = m.get("positions") or {}
    if pos:
        for sym in sorted(pos):
            p = pos[sym]
            px = (m.get("last_px") or {}).get(sym) or p["entry"]
            lines.append(f"・{sym} x{p['qty']:g} @ {p['entry']:.2f}"
                         f"（現 {px:.2f}，{px / p['entry'] - 1:+.1%}）")
    else:
        lines.append("（無持倉，全現金）")
    eng = m.get("engine") or {}
    if eng.get("halted_until"):
        lines.append(f"🚨 保險絲冷卻至 {eng['halted_until']}")
    nj = len(m.get("journal") or [])
    lines.append(f"引擎已執行 {nj} 筆虛擬交易　`/mirror reset` 可重建")
    lines.append("模式 A：真實買賣不同步；成交=掃描價無滑價；非投資建議")
    return "\n".join(lines)


# ── 自我測試（離線；假 scored 餵真 trade_engine）──────────────────────────

if __name__ == "__main__":
    T = "2026-08-21"

    # 1) 解析：壞格式拒收、零股拒收（Med-3）、重複 symbol 加權合併（Med-2）
    hold, bad = parse_holdings(["AAPL:10:180.5", "NVDA:20:150", "壞的", "X:-1:5",
                                "Y:1:0", "Z:0.5:100"])
    assert hold == [("AAPL", 10.0, 180.5), ("NVDA", 20.0, 150.0)] and len(bad) == 4
    dup, bad2 = parse_holdings(["AAPL:10:100", "AAPL:5:200"])
    assert dup == [("AAPL", 15.0, round((10 * 100 + 5 * 200) / 15, 4))] and not bad2
    b_dup = init_book(1000, dup, T)
    assert abs(b_dup["start_equity"] - (1000 + 10 * 100 + 5 * 200)) < 1e-6  # 不再虛報
    print("✅ 1 快照解析（壞格式/零股拒收/重複合併不虛報）")

    # 2) init：start_equity = 現金 + 成本市值
    st = {"mirror": init_book(25000, hold, T)}
    m = st["mirror"]
    assert m["start_equity"] == 25000 + 10 * 180.5 + 20 * 150
    assert m["positions"]["AAPL"]["entry"] == 180.5
    print("✅ 2 初始化")

    # 3) 端到端：引擎收養原持倉（entry=成本）＋高分新標的買進
    scored = [{"ticker": "AAPL", "score": 0.1, "price": 185.0},
              {"ticker": "NVDA", "score": 0.2, "price": 155.0},
              {"ticker": "MSFT", "score": 0.9, "price": 100.0, "risk_per_share": 4.0}]
    lines = run_mirror(st, scored, {}, "risk_on", T)
    eng = m["engine"]
    assert eng["pos"]["AAPL"]["entry"] == 180.5          # 收養用你的成本
    assert "MSFT" in m["positions"] and m["positions"]["MSFT"]["qty"] >= 1
    assert any("MSFT" in ln for ln in lines)
    assert m["journal"] and m["journal"][-1]["symbol"] == "MSFT"
    cash_after = m["cash"]
    assert cash_after < 25000
    print("✅ 3 端到端（收養+新倉+journal）")

    # 4) 停損出場：AAPL 跌破 成本−5% 預設停損 → 賣出回收現金
    px_stop = 180.5 * 0.94
    lines2 = run_mirror(st, [{"ticker": "AAPL", "score": 0.1, "price": px_stop},
                             {"ticker": "NVDA", "score": 0.2, "price": 155.0},
                             {"ticker": "MSFT", "score": 0.6, "price": 100.0}],
                        {}, "risk_on", T)
    assert "AAPL" not in st["mirror"]["positions"], st["mirror"]["positions"]
    assert st["mirror"]["cash"] > cash_after
    print("✅ 4 停損出場回收現金")

    # 5) 未初始化 / 顯示 / 聯集 / Markdown 安全
    assert run_mirror({}, scored, {}, "risk_on", T) == []
    assert holdings_symbols(st) == sorted(st["mirror"]["positions"].keys())
    assert holdings_symbols({}) == []
    txt = mirror_text(st)
    assert "鏡像帳" in txt and "_" not in txt and txt.count("*") % 2 == 0, txt
    assert "未初始化" in mirror_text({})
    print("✅ 5 邊界 + 顯示（Markdown 安全）")

    # 5b) 幽靈賣單防護（Med-1）：持倉不在 scored → decide 用凍結價產生賣單，
    #     但不得入帳/入 journal/推播；連續 5 天無報價 → 凍結價強平
    stg = {"mirror": init_book(1000, [("DEAD", 10, 100.0)], "2026-08-21")}
    stg["mirror"]["positions"]["DEAD"]["entry"] = 100.0
    ln_a = run_mirror(stg, [{"ticker": "GLD2", "score": 0.0, "price": 50.0}],
                      {}, "risk_on", "2026-08-21")
    assert "DEAD" in stg["mirror"]["positions"]           # 無價 → 沒被幽靈單平掉
    assert not any(e["symbol"] == "DEAD" for e in stg["mirror"].get("journal", []))
    assert not any("DEAD" in ln for ln in ln_a), ln_a     # 不推播幽靈單
    assert stg["mirror"]["positions"]["DEAD"]["stale_since"] == "2026-08-21"
    ln_b = run_mirror(stg, [{"ticker": "GLD2", "score": 0.0, "price": 50.0}],
                      {}, "risk_on", "2026-08-26")        # 5 天後
    assert "DEAD" not in stg["mirror"]["positions"]       # 凍結價強平
    assert stg["mirror"]["cash"] >= 1000 + 10 * 100 - 1e-6
    assert any("強制平倉" in ln for ln in ln_b), ln_b
    print("✅ 5b 幽靈單不入帳 + 殭屍倉 5 天凍結強平")

    # 6) journal cap
    st["mirror"]["journal"] = [{"x": i} for i in range(JOURNAL_CAP + 50)]
    run_mirror(st, [{"ticker": "GLD", "score": 0.9, "price": 50.0,
                     "risk_per_share": 2.0}], {}, "risk_on", T)
    assert len(st["mirror"]["journal"]) <= JOURNAL_CAP + 1
    print("✅ 6 journal cap")

    # 7) to_std_journal：mirror journal → attribution/behavior_check 相容格式
    std = to_std_journal({"journal": [
        {"date": "2026-08-21", "side": "buy", "symbol": "MSFT", "qty": 3,
         "price": 100.0, "mechanism": "entry", "reason": "r"},
        {"bad": 1}, "junk"]})
    assert len(std) == 1 and std[0]["submitted"] is True
    assert std[0]["time"] == "2026-08-21T15:00:00Z"
    assert std[0]["symbol"] == "MSFT" and std[0]["mechanism"] == "entry"
    assert to_std_journal(None) == [] and to_std_journal({}) == []
    assert to_std_journal({"__enc__": True, "n": "x"}) == []  # 鎖定密文 → 空
    print("✅ 7 to_std_journal 橋接")

    print("\nmirror_book selftest OK ✅")
