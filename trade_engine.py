"""
trade_engine.py – Lean 式分層自動交易引擎（純邏輯、離線可測）

重製自開源交易機器人的成熟機制（來源已逐一查證原始碼）：
  • QuantConnect Lean —— Alpha / Portfolio / Risk / Execution 分層：
      評分只當「觀察」用（alpha），出場由價格機制驅動（risk），兩者脫鉤
  • freqtrade —— exit_profit_only（訊號轉弱只在獲利時了結，虧損時交給停損、
      不在低點殺出）、trailing_stop_positive_offset（獲利達標才啟動追蹤、
      並墊高到保本）、Protections（StoplossGuard / MaxDrawdown / CooldownPeriod）
  • nautilus_trader —— TradingState 三態 ACTIVE / REDUCING / HALTED
  • Jesse —— 分批出場（+1.5R 先賣一半、停損上移保本 = 「免費部位」）
  • 海龜式贏家加碼 —— 只對獲利部位每 +1R 加碼、最多 2 次

與舊 alpaca_trader.decide_orders 的關鍵差異：
  舊版「評分 ≤ -0.2 → 無條件全平」——訊號衰減常發生在回檔低點，
  等於系統性地在低點賣出、把好部位洗掉，績效自然貼著大盤還慢半拍。
  新版：分數弱只擋「新買進」；已持有的部位靠停損 / 追蹤停損 / 死錢釋放出場，
  虧損中的訊號衰減 → 續抱等待（這就是「耐心」的機制化）。

進入點：decide(scored, positions, equity, buying_power, engine, regime, config, today)
  回 (orders, engine, notes)。engine 為可持久化的引擎狀態（存 watchlist_state.json
  的 state["engine"]），呼叫端負責存檔。純函數風格：不打網路、時間由 today 傳入。
"""

from __future__ import annotations

from datetime import date, timedelta

ENGINE_DEFAULTS = {
    # ── Alpha 層（訊號 → 觀察）───────────────────────────────────────────
    "buy_threshold":      0.5,    # 評分 ≥ 此值 → 新多頭候選
    "exit_threshold":    -0.2,    # 評分 ≤ 此值 → 訊號轉弱（只在獲利時了結）
    # ── Portfolio 層（部位建構）─────────────────────────────────────────
    "max_positions":      10,
    "max_position_pct":   0.15,   # 每檔市值上限（占淨值）
    "risk_pct":           0.01,   # 單筆風險占淨值
    "pyramid_r":          1.0,    # 每獲利 +1R 可加碼一次
    "pyramid_max_adds":   2,      # 最多加碼次數
    "pyramid_frac":       0.5,    # 加碼股數 = 初始股數 × 此比例
    "pyramid_min_score":  0.0,    # 加碼時評分至少要 ≥ 此值
    # ── Risk 層（價格驅動出場）───────────────────────────────────────────
    "stop_mult":          1.0,    # 硬停損 = 進場價 − stop_mult × 每股風險
    "trail_activate_r":   1.0,    # 獲利 ≥ +1R 才啟動追蹤停損（同時墊高到保本）
    "trail_pct":          0.08,   # 追蹤：自 peak 回落 8% 出場
    "trail_tight_pct":    0.05,   # 收緊版（+2R 之後或非 ACTIVE 狀態）
    "trail_tighten_r":    2.0,    # 獲利 ≥ +2R → 改用收緊版追蹤
    "scale_out_r":        1.5,    # 獲利 ≥ +1.5R → 先賣一半（免費部位）
    "dead_money_days":    30,     # 持有 ≥ N 天…
    "dead_money_ret":     0.02,   # …且報酬 < +2% 且評分未達買進門檻 → 釋放名額
    # ── Protections 保險絲（帳戶層）─────────────────────────────────────
    "stoploss_guard_n":   3,      # 觀察窗內硬停損 ≥ N 次 → 全帳戶冷卻
    "stoploss_guard_days": 7,     # 觀察窗天數
    "account_cooldown_days": 3,   # 全帳戶冷卻天數（HALTED）
    "symbol_cooldown_days":  5,   # 個股停損後禁止再進場天數
    "max_dd_halt":        0.10,   # 帳戶自高點回撤 ≥ 10% → REDUCING（停新倉）
}


# ── 小工具 ────────────────────────────────────────────────────────────────

def _d(s: str) -> date:
    return date.fromisoformat(str(s)[:10])


def _days_between(a: str, b: str) -> int:
    try:
        return (_d(b) - _d(a)).days
    except Exception:
        return 0


def _plus_days(today: str, n: int) -> str:
    return (_d(today) + timedelta(days=int(n))).isoformat()


def new_engine_state() -> dict:
    return {"pos": {}, "stop_events": [], "cooldown": {},
            "halted_until": None, "equity_peak": 0.0}


# ── 狀態同步（broker 為真相）─────────────────────────────────────────────

def sync_positions(engine: dict, positions: dict, price_of, today: str) -> list[str]:
    """
    引擎內部部位簿與 broker 實際持倉對齊。回傳同步備註。
      • broker 有、引擎沒有（手動買進 / 舊版遺留）→ 以均價收養
      • 引擎有、broker 沒有（手動平倉）→ 移除紀錄
    """
    notes = []
    pos_book = engine.setdefault("pos", {})
    for sym, p in positions.items():
        if sym not in pos_book:
            entry = float(p.get("avg_entry_price") or 0) or (price_of(sym) or 0.0)
            px = price_of(sym) or entry
            pos_book[sym] = {
                "entry": entry, "rps": max(entry * 0.05, 0.01),
                "peak": max(entry, px), "opened": today,
                "init_qty": abs(float(p.get("qty") or 0)), "adds": 0,
                "scaled_out": False,
            }
            notes.append(f"收養既有持倉 {sym}（均價 {entry:.2f}）")
    for sym in [s for s in pos_book if s not in positions]:
        del pos_book[sym]
        notes.append(f"移除已不在帳上的 {sym}")
    return notes


# ── Exposure 三態（nautilus TradingState）────────────────────────────────

def exposure_state(regime: str | None, drawdown: float,
                   halted_until: str | None, today: str, cfg: dict) -> tuple[str, str]:
    """回 (state, reason)。state ∈ ACTIVE / REDUCING / HALTED。"""
    if halted_until:
        try:
            if _d(today) < _d(halted_until):
                return "HALTED", f"停損保險絲觸發，冷卻至 {halted_until}"
        except Exception:
            pass
    if drawdown >= float(cfg["max_dd_halt"]):
        return "REDUCING", f"帳戶自高點回撤 {drawdown:.1%} ≥ {cfg['max_dd_halt']:.0%}"
    if regime == "risk_off":
        return "REDUCING", "大盤風險偏空（跌破 MA50）"
    return "ACTIVE", ""


# ── 主決策（Alpha → Portfolio → Risk → Execution）────────────────────────

def decide(scored: list[dict], positions: dict, equity: float, buying_power: float,
           engine: dict | None, regime: str | None, config: dict | None,
           today: str) -> tuple[list[dict], dict, list[str]]:
    """
    回 (orders, engine, notes)。orders: [{symbol, side, qty, reason, mechanism}]。
    engine 就地更新後回傳（呼叫端持久化）。notes 為給使用者的機制說明行。
    """
    cfg = {**ENGINE_DEFAULTS, **(config or {})}
    # /set eng_* 不驗證值域 → 引擎端夾住，避免 trail_pct 5 或 guard_n 0 之類造成
    # 「保本價全平」「恆真保險絲→永久 HALTED」的靜默災難
    for k in ("trail_pct", "trail_tight_pct"):
        cfg[k] = min(max(float(cfg[k]), 0.005), 0.5)
    cfg["stoploss_guard_n"] = max(1, int(cfg["stoploss_guard_n"]))
    cfg["max_positions"] = max(1, int(cfg["max_positions"]))
    engine = engine if isinstance(engine, dict) and engine.get("pos") is not None \
        else new_engine_state()
    equity = float(equity)
    today = str(today)[:10]

    score_map = {s["ticker"]: s for s in scored}

    def price_of(sym):
        s = score_map.get(sym)
        if s and s.get("price"):
            return float(s["price"])
        p = positions.get(sym) or {}
        qty, mv = float(p.get("qty") or 0), float(p.get("market_value") or 0)
        return (mv / qty) if qty else None

    notes = sync_positions(engine, positions, price_of, today)
    pos_book = engine["pos"]

    # equity 高水位 → 回撤
    peak_eq = max(float(engine.get("equity_peak") or 0), equity)
    engine["equity_peak"] = peak_eq
    dd = (1 - equity / peak_eq) if peak_eq > 0 else 0.0

    exp_state, exp_reason = exposure_state(
        regime, dd, engine.get("halted_until"), today, cfg)
    if exp_state != "ACTIVE":
        notes.append(f"曝險狀態 {exp_state}：{exp_reason}")
    tighten = exp_state != "ACTIVE"

    orders: list[dict] = []
    exited: set[str] = set()

    def _record_stop(sym):
        """硬停損（或跳空虧損出場）記事件＋個股冷卻。同股同日去重——賣單卡住
        跨輪重試不該被保險絲當成多次獨立停損（事件格式 'YYYY-MM-DD|SYM'，
        _d 只取前 10 字元故日期數學不受影響）。"""
        ev = engine.setdefault("stop_events", [])
        key_ = f"{today}|{sym}"
        if key_ not in ev:
            ev.append(key_)
        engine.setdefault("cooldown", {})[sym] = \
            _plus_days(today, cfg["symbol_cooldown_days"])

    # ── Risk 層：逐檔出場檢查（優先序固定：硬停損 → 追蹤 → 分批 → 訊號 → 死錢）
    for sym, pos in positions.items():
        rec = pos_book.get(sym)
        px = price_of(sym)
        qty = abs(float(pos.get("qty") or 0))
        if not rec or qty <= 0:
            continue
        if not px:
            notes.append(f"⚠️ {sym} 無價格資料，本輪跳過出場檢查")
            continue
        if qty < 1:
            # 零股賣單經 int() 會變 qty=0 被 Alpaca 拒單 → 無限重試+誤觸保險絲
            notes.append(f"⚠️ {sym} 為零股部位（{qty:g} 股），引擎不處理，請手動平倉")
            continue
        entry, rps = float(rec["entry"]), max(float(rec["rps"]), 0.01)
        rec["peak"] = max(float(rec.get("peak", entry)), px)
        peak = rec["peak"]
        r_now = (px - entry) / rps
        r_peak = (peak - entry) / rps
        sc = float((score_map.get(sym) or {}).get("score") or 0)   # None 防炸

        def _sell(q, reason, mech):
            orders.append({"symbol": sym, "side": "sell", "qty": q,
                           "reason": reason, "mechanism": mech})

        # 1) 硬停損（追蹤啟動前的下檔保護）
        stop_line = entry - float(cfg["stop_mult"]) * rps
        if r_peak < float(cfg["trail_activate_r"]) and px <= stop_line:
            _sell(qty, f"硬停損 {stop_line:.2f}（進場 {entry:.2f}−{cfg['stop_mult']}×風險）",
                  "stop_loss")
            _record_stop(sym)
            exited.add(sym)
            continue

        # 2) 追蹤停損（獲利曾達 +1R 才啟動；地板 = 保本價，永不倒虧）
        if r_peak >= float(cfg["trail_activate_r"]):
            pct = float(cfg["trail_tight_pct"]) if (
                tighten or r_peak >= float(cfg["trail_tighten_r"])
            ) else float(cfg["trail_pct"])
            trail_line = max(entry, peak * (1 - pct))
            if px <= trail_line:
                _sell(qty, f"追蹤停損 {trail_line:.2f}（峰值 {peak:.2f} 回落 {pct:.0%}，保本地板 {entry:.2f}）",
                      "trailing_stop")
                if px < entry:
                    # 跳空穿越保本地板 = 實際虧損出場 → 一樣計入保險絲＋個股冷卻
                    _record_stop(sym)
                exited.add(sym)
                continue

        # 3) 分批出場：+1.5R 先賣一半 → 剩餘部位變「免費」
        if not rec.get("scaled_out") and r_now >= float(cfg["scale_out_r"]) and qty >= 2:
            half = int(qty // 2)
            if half >= 1:
                _sell(half, f"獲利 {r_now:+.1f}R ≥ {cfg['scale_out_r']}R，先賣一半鎖利（餘倉保本追蹤）",
                      "scale_out")
                rec["scaled_out"] = True
                continue

        # 4) 訊號轉弱 → 只在「獲利中」了結（freqtrade exit_profit_only）
        if sc <= float(cfg["exit_threshold"]):
            if px > entry:
                _sell(qty, f"評分 {sc:+.2f} 轉弱且獲利中（{px / entry - 1:+.1%}）→ 了結",
                      "signal_exit")
                exited.add(sym)
            else:
                notes.append(f"⏸ {sym} 評分 {sc:+.2f} 轉弱但未達停損（{px / entry - 1:+.1%}）→ 續抱等待")
            continue

        # 5) 死錢釋放：占著名額不動的部位讓位給新機會
        held_days = _days_between(rec.get("opened", today), today)
        if held_days >= int(cfg["dead_money_days"]) \
                and (px / entry - 1) < float(cfg["dead_money_ret"]) \
                and sc < float(cfg["buy_threshold"]):
            _sell(qty, f"持有 {held_days} 天報酬 {px / entry - 1:+.1%} 且評分平庸 → 死錢釋放",
                  "dead_money")
            exited.add(sym)

    # ── Protections：停損保險絲（freqtrade StoplossGuard）
    win = int(cfg["stoploss_guard_days"])

    def _ev_ok(e):
        try:
            _d(e)
            return True
        except Exception:
            return False   # 壞事件直接剔除，否則 _days_between 回 0 → 永遠算「今天」

    events = [e for e in engine.get("stop_events", [])
              if _ev_ok(e) and 0 <= _days_between(e, today) <= win * 2]
    engine["stop_events"] = events
    recent = [e for e in events if _days_between(e, today) <= win]
    # 同一檔股票在窗內只算一次（賣單卡住多日重試 ≠ 多次獨立停損）；
    # 舊格式（純日期、無代碼）各算一次
    n_recent = len({e.split("|", 1)[1] for e in recent if "|" in e}) \
        + sum(1 for e in recent if "|" not in e)
    if n_recent >= int(cfg["stoploss_guard_n"]) and exp_state != "HALTED":
        engine["halted_until"] = _plus_days(today, cfg["account_cooldown_days"])
        exp_state = "HALTED"
        # 觸發即消耗掉這批事件——否則到期日同批舊事件立刻再武裝，
        # 名目 3 天冷卻實際變 9 天、且重複推送誤導性警報
        engine["stop_events"] = [e for e in events if e not in recent]
        notes.append(f"🚨 停損保險絲：{win} 天內 {n_recent} 次硬停損 → "
                     f"全帳戶暫停新倉至 {engine['halted_until']}")
    # 冷卻期滿自動解除
    if engine.get("halted_until"):
        try:
            if _d(today) >= _d(engine["halted_until"]):
                engine["halted_until"] = None
        except Exception:
            engine["halted_until"] = None

    # 個股冷卻清理
    cd = engine.setdefault("cooldown", {})
    for sym in [s for s, until in cd.items() if _days_between(today, until) <= 0]:
        del cd[sym]

    # ── Portfolio 層：新進場（僅 ACTIVE）
    held = {s for s in positions if s not in exited}
    bp = float(buying_power)
    if exp_state == "ACTIVE" and equity > 0:
        slots = int(cfg["max_positions"]) - len(held)
        cands = sorted(
            [s for s in scored
             if s["ticker"] not in held and s["ticker"] not in exited
             and s["ticker"] not in cd
             and not s.get("no_entry")          # alpha overlay veto（如財報前）只擋新倉
             and float(s.get("score") or 0) >= float(cfg["buy_threshold"])
             and float(s.get("price", 0) or 0) > 0],
            key=lambda s: -float(s["score"]))
        for s in cands:
            if slots <= 0:
                break
            px = float(s["price"])
            rps = s.get("risk_per_share")
            rps = float(rps) if rps and float(rps) > 0 else px * 0.05
            qty = int(min((equity * float(cfg["risk_pct"])) / rps,
                          (equity * float(cfg["max_position_pct"])) / px,
                          bp / px))
            if qty >= 1:
                orders.append({"symbol": s["ticker"], "side": "buy", "qty": qty,
                               "reason": f"評分 {float(s['score']):+.2f} ≥ 門檻 {cfg['buy_threshold']}"
                                         f"（風險部位 {qty} 股）",
                               "mechanism": "entry"})
                pos_book[s["ticker"]] = {
                    "entry": px, "rps": rps, "peak": px, "opened": today,
                    "init_qty": qty, "adds": 0, "scaled_out": False,
                }
                bp -= qty * px
                slots -= 1
                held.add(s["ticker"])

        # 贏家加碼（海龜式：每 +1R 加一次、最多 2 次、只加給還在趨勢中的贏家）
        for sym in sorted(held):
            rec = pos_book.get(sym)
            if not rec or sym in exited or rec.get("scaled_out"):
                continue
            if sym not in positions:      # 本輪新開倉不加碼
                continue
            px = price_of(sym)
            if not px:
                continue
            entry, rps = float(rec["entry"]), max(float(rec["rps"]), 0.01)
            r_now = (px - entry) / rps
            adds = int(rec.get("adds", 0))
            s_rec = score_map.get(sym) or {}
            if s_rec.get("no_entry"):           # veto 也擋加碼（出場機制不受影響）
                continue
            sc = float(s_rec.get("score") or 0)
            if adds < int(cfg["pyramid_max_adds"]) \
                    and r_now >= (adds + 1) * float(cfg["pyramid_r"]) \
                    and sc >= float(cfg["pyramid_min_score"]):
                add_qty = int(max(1, rec.get("init_qty", 0) * float(cfg["pyramid_frac"])))
                mv_now = abs(float(positions[sym].get("market_value") or 0))
                if (mv_now + add_qty * px) <= equity * float(cfg["max_position_pct"]) \
                        and add_qty * px <= bp:
                    orders.append({"symbol": sym, "side": "buy", "qty": add_qty,
                                   "reason": f"贏家加碼 #{adds + 1}（獲利 {r_now:+.1f}R，評分 {sc:+.2f}）",
                                   "mechanism": "pyramid"})
                    rec["adds"] = adds + 1
                    bp -= add_qty * px

    # 平倉部位移出簿記（成交與否由下一輪 broker 同步兜底）
    for sym in exited:
        pos_book.pop(sym, None)

    return orders, engine, notes


def engine_status_text(engine: dict | None, today: str, cfg: dict | None = None) -> str:
    """給 /protections 用的引擎保險絲狀態摘要（Telegram legacy Markdown：單 *）。"""
    cfg = {**ENGINE_DEFAULTS, **(cfg or {})}
    e = engine or {}
    win = int(cfg["stoploss_guard_days"])
    recent = [ev for ev in e.get("stop_events", []) if _days_between(ev, today) <= win]
    halted = e.get("halted_until")
    try:
        halted_active = bool(halted) and _d(today) < _d(halted)
    except Exception:
        halted_active = False
    n_recent = len({ev.split("|", 1)[1] for ev in recent if "|" in ev}) \
        + sum(1 for ev in recent if "|" not in ev)
    lines = ["⚙️ *交易引擎保險絲*"]
    if halted_active:
        lines.append(f"🚨 全帳戶冷卻中（至 {halted}）——停損保險絲觸發")
    else:
        lines.append(f"🟢 保險絲正常（{win} 天內硬停損 {n_recent}/{int(cfg['stoploss_guard_n'])} 次）")
    cd = {s: u for s, u in (e.get("cooldown") or {}).items()
          if _days_between(today, u) > 0}
    if cd:
        lines.append("🧊 個股冷卻：" + "、".join(f"{s}(至{u})" for s, u in sorted(cd.items())))
    npos = len(e.get("pos") or {})
    if npos:
        lines.append(f"📒 引擎追蹤 {npos} 檔持倉（peak/加碼/分批狀態）")
    return "\n".join(lines)


# ── 自我測試 ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    T = "2026-07-20"

    def mk_pos(qty, entry, px):
        return {"qty": qty, "avg_entry_price": entry, "market_value": qty * px,
                "unrealized_pl": qty * (px - entry), "unrealized_plpc": px / entry - 1}

    def mk_eng(sym, entry, rps, peak=None, opened=T, init_qty=10, adds=0, scaled=False):
        e = new_engine_state()
        e["pos"][sym] = {"entry": entry, "rps": rps, "peak": peak or entry,
                         "opened": opened, "init_qty": init_qty, "adds": adds,
                         "scaled_out": scaled}
        return e

    # 1) 核心行為：訊號衰減 + 虧損中 → 不賣（耐心機制）
    eng = mk_eng("AAPL", 100, 3)
    orders, eng, notes = decide(
        [{"ticker": "AAPL", "score": -0.5, "price": 98.0}],
        {"AAPL": mk_pos(10, 100, 98)}, 100000, 50000, eng, "neutral", None, T)
    assert orders == [], f"訊號衰減虧損中不應賣出: {orders}"
    assert any("續抱等待" in n for n in notes), notes
    print("✅ 1 訊號衰減+虧損 → 續抱（舊版會在此低點殺出）")

    # 2) 訊號衰減 + 獲利中 → 了結（exit_profit_only）
    eng = mk_eng("AAPL", 100, 3)
    orders, _, _ = decide(
        [{"ticker": "AAPL", "score": -0.5, "price": 102.0}],
        {"AAPL": mk_pos(10, 100, 102)}, 100000, 50000, eng, "neutral", None, T)
    assert len(orders) == 1 and orders[0]["mechanism"] == "signal_exit", orders
    print("✅ 2 訊號衰減+獲利 → 了結")

    # 3) 硬停損 → 賣出 + 記事件 + 個股冷卻
    eng = mk_eng("TSLA", 100, 3)
    orders, eng, _ = decide(
        [{"ticker": "TSLA", "score": 0.1, "price": 96.5}],
        {"TSLA": mk_pos(10, 100, 96.5)}, 100000, 50000, eng, "neutral", None, T)
    assert orders and orders[0]["mechanism"] == "stop_loss", orders
    assert eng["stop_events"] == [f"{T}|TSLA"] and "TSLA" in eng["cooldown"]
    assert "TSLA" not in eng["pos"]
    print("✅ 3 硬停損 → 事件記錄 + 冷卻 + 清簿")

    # 4) 個股冷卻中 → 高分也不得再進場
    orders, _, _ = decide(
        [{"ticker": "TSLA", "score": 0.9, "price": 97.0}],
        {}, 100000, 50000, eng, "risk_on", None, T)
    assert not any(o["symbol"] == "TSLA" for o in orders), orders
    print("✅ 4 冷卻中不再進場")

    # 5) 追蹤停損：峰值 +2R 啟動、跌破回落線 → 出場；且地板保本
    eng = mk_eng("NVDA", 100, 5, peak=112)          # r_peak=2.4 → 收緊 5%
    orders, _, _ = decide(
        [{"ticker": "NVDA", "score": 0.6, "price": 106.0}],
        {"NVDA": mk_pos(10, 100, 106)}, 100000, 50000, eng, "risk_on", None, T)
    assert orders and orders[0]["mechanism"] == "trailing_stop", orders
    eng = mk_eng("NVDA", 100, 5, peak=106)          # r_peak=1.2 → 8%、地板=entry=100
    orders, _, _ = decide(
        [{"ticker": "NVDA", "score": 0.6, "price": 99.5}],
        {"NVDA": mk_pos(10, 100, 99.5)}, 100000, 50000, eng, "risk_on", None, T)
    assert orders and orders[0]["mechanism"] == "trailing_stop" \
        and "保本地板" in orders[0]["reason"], orders
    print("✅ 5 追蹤停損（含收緊 + 保本地板）")

    # 6) 分批出場：+1.5R 賣一半、標記後不重複賣
    eng = mk_eng("MSFT", 100, 4)
    orders, eng, _ = decide(
        [{"ticker": "MSFT", "score": 0.6, "price": 106.5}],
        {"MSFT": mk_pos(10, 100, 106.5)}, 100000, 50000, eng, "risk_on", None, T)
    assert orders and orders[0]["mechanism"] == "scale_out" and orders[0]["qty"] == 5, orders
    orders2, _, _ = decide(
        [{"ticker": "MSFT", "score": 0.6, "price": 106.5}],
        {"MSFT": mk_pos(5, 100, 106.5)}, 100000, 50000, eng, "risk_on", None, T)
    assert not any(o["mechanism"] == "scale_out" for o in orders2), orders2
    print("✅ 6 分批出場一次性")

    # 7) 死錢釋放：持有 40 天、報酬 +1%、評分平庸 → 釋放
    eng = mk_eng("KO", 100, 3, opened="2026-06-10")
    orders, _, _ = decide(
        [{"ticker": "KO", "score": 0.1, "price": 101.0}],
        {"KO": mk_pos(10, 100, 101)}, 100000, 50000, eng, "neutral", None, T)
    assert orders and orders[0]["mechanism"] == "dead_money", orders
    print("✅ 7 死錢釋放")

    # 8) 停損保險絲：7 天內第 3 次硬停損 → HALTED、高分新標的不進場
    eng = mk_eng("X1", 100, 2)
    eng["stop_events"] = ["2026-07-18", "2026-07-19"]
    orders, eng, notes = decide(
        [{"ticker": "X1", "score": 0.1, "price": 97.0},
         {"ticker": "GOOD", "score": 0.9, "price": 50.0}],
        {"X1": mk_pos(10, 100, 97)}, 100000, 50000, eng, "risk_on", None, T)
    assert any(o["mechanism"] == "stop_loss" for o in orders)
    assert not any(o["side"] == "buy" for o in orders), orders
    assert eng["halted_until"] == "2026-07-23", eng["halted_until"]
    assert eng["stop_events"] == [], eng["stop_events"]   # 觸發即消耗，冷卻天數才真實
    print("✅ 8 停損保險絲 → 暫停新倉 + 事件消耗（不再武裝）")

    # 9) REDUCING（risk_off）：不進新倉、不加碼，但出場照常
    orders, _, notes = decide(
        [{"ticker": "GOOD", "score": 0.9, "price": 50.0}],
        {}, 100000, 50000, new_engine_state(), "risk_off", None, T)
    assert orders == [] and any("REDUCING" in n for n in notes), (orders, notes)
    print("✅ 9 risk_off → REDUCING 停新倉")

    # 10) 回撤保險絲：equity 自高點 −12% → REDUCING
    eng = new_engine_state()
    eng["equity_peak"] = 100000
    orders, _, notes = decide(
        [{"ticker": "GOOD", "score": 0.9, "price": 50.0}],
        {}, 88000, 50000, eng, "risk_on", None, T)
    assert orders == [] and any("回撤" in n for n in notes), (orders, notes)
    print("✅ 10 最大回撤 → 停新倉")

    # 11) 贏家加碼：+1R 加一次、達上限不再加、REDUCING 不加
    eng = mk_eng("AMD", 100, 5, peak=105, init_qty=10)
    orders, eng, _ = decide(
        [{"ticker": "AMD", "score": 0.6, "price": 105.0}],
        {"AMD": mk_pos(10, 100, 105)}, 100000, 50000, eng, "risk_on", None, T)
    pyr = [o for o in orders if o["mechanism"] == "pyramid"]
    assert len(pyr) == 1 and pyr[0]["qty"] == 5, orders
    assert eng["pos"]["AMD"]["adds"] == 1
    eng2 = mk_eng("AMD", 100, 5, peak=115, adds=2)
    orders, _, _ = decide(
        [{"ticker": "AMD", "score": 0.6, "price": 103.0}],
        {"AMD": mk_pos(15, 100, 103)}, 100000, 50000, eng2, "risk_on", None, T)
    assert not any(o["mechanism"] == "pyramid" for o in orders), orders
    eng3 = mk_eng("AMD", 100, 5, peak=105)
    orders, _, _ = decide(
        [{"ticker": "AMD", "score": 0.6, "price": 105.0}],
        {"AMD": mk_pos(10, 100, 105)}, 100000, 50000, eng3, "risk_off", None, T)
    assert not any(o["mechanism"] == "pyramid" for o in orders), orders
    print("✅ 11 贏家加碼（一次 / 上限 / REDUCING 禁止）")

    # 12) 新進場：ACTIVE + 高分 → 買進且簿記建檔；上限約束仍在
    eng = new_engine_state()
    orders, eng, _ = decide(
        [{"ticker": "NEW", "score": 0.8, "price": 100.0, "risk_per_share": 4.0}],
        {}, 100000, 100000, eng, "risk_on", None, T)
    assert orders and orders[0]["side"] == "buy" and orders[0]["qty"] == 150, orders
    rec = eng["pos"]["NEW"]
    assert rec["entry"] == 100.0 and rec["rps"] == 4.0 and rec["opened"] == T
    orders2, _, _ = decide(
        [{"ticker": "NEW2", "score": 0.8, "price": 100.0}],
        {}, 100000, 500, new_engine_state(), "risk_on", None, T)
    assert orders2 and orders2[0]["qty"] == 5, orders2   # 購買力约束
    print("✅ 12 進場 sizing + 簿記")

    # 13) broker 真相同步：收養手動買的、移除手動平的
    eng = mk_eng("GONE", 100, 3)
    orders, eng, notes = decide(
        [{"ticker": "MANUAL", "score": 0.0, "price": 55.0}],
        {"MANUAL": mk_pos(20, 50, 55)}, 100000, 50000, eng, "neutral", None, T)
    assert "MANUAL" in eng["pos"] and "GONE" not in eng["pos"]
    assert eng["pos"]["MANUAL"]["entry"] == 50
    print("✅ 13 broker 同步（收養/移除）")

    # 14) max_positions：滿倉不再進場
    full_pos = {f"S{i}": mk_pos(1, 100, 100) for i in range(10)}
    orders, _, _ = decide(
        [{"ticker": "NEW", "score": 0.9, "price": 100.0}],
        full_pos, 100000, 100000, new_engine_state(), "risk_on", None, T)
    assert not any(o["side"] == "buy" for o in orders), orders
    print("✅ 14 滿倉不進場")

    # 15) 保險絲到期自動解除
    eng = new_engine_state()
    eng["halted_until"] = "2026-07-20"      # 今天到期
    orders, eng, _ = decide(
        [{"ticker": "GOOD", "score": 0.9, "price": 50.0}],
        {}, 100000, 50000, eng, "risk_on", None, T)
    # 到期日當天：halted_until 清除；HALTED 判斷用 <，今天不再受限
    assert eng["halted_until"] is None
    assert any(o["side"] == "buy" for o in orders), orders
    print("✅ 15 冷卻期滿自動解除")

    # 16) 零股部位 → 不下 qty=0 賣單（會被 Alpaca 拒單→無限重試→誤觸保險絲）
    eng = mk_eng("FRAC", 100, 3)
    orders, eng, notes = decide(
        [{"ticker": "FRAC", "score": 0.1, "price": 90.0}],
        {"FRAC": mk_pos(0.5, 100, 90)}, 100000, 50000, eng, "neutral", None, T)
    assert orders == [] and any("零股" in n for n in notes), (orders, notes)
    assert eng["stop_events"] == []
    print("✅ 16 零股部位不下單、不記停損事件")

    # 17) 賣單卡住跨輪重試 → 同股同日事件去重（保險絲不被污染）
    eng = mk_eng("STUCK", 100, 3)
    orders, eng, _ = decide(
        [{"ticker": "STUCK", "score": 0.1, "price": 94.0}],
        {"STUCK": mk_pos(10, 100, 94)}, 100000, 50000, eng, "neutral", None, T)
    assert orders and orders[0]["mechanism"] == "stop_loss"
    # 部位還在（賣單沒成交）→ 下一輪 sync 收養後再次觸發停損
    orders, eng, _ = decide(
        [{"ticker": "STUCK", "score": 0.1, "price": 94.0}],
        {"STUCK": mk_pos(10, 100, 94)}, 100000, 50000, eng, "neutral", None, T)
    assert eng["stop_events"] == [f"{T}|STUCK"], eng["stop_events"]
    print("✅ 17 卡單重試不重複計停損事件")

    # 18) 追蹤啟動後跳空穿越保本地板（實際虧損出場）→ 也計保險絲 + 冷卻
    eng = mk_eng("GAP", 100, 5, peak=110)
    orders, eng, _ = decide(
        [{"ticker": "GAP", "score": 0.3, "price": 80.0}],
        {"GAP": mk_pos(10, 100, 80)}, 100000, 50000, eng, "risk_on", None, T)
    assert orders and orders[0]["mechanism"] == "trailing_stop", orders
    assert eng["stop_events"] == [f"{T}|GAP"] and "GAP" in eng["cooldown"]
    print("✅ 18 跳空虧損出場計入保險絲")

    # 19) 值域夾制：guard_n=0 不得恆真觸發；trail_pct=5 夾到 0.5 不亂平倉
    orders, eng, notes = decide(
        [{"ticker": "GOOD", "score": 0.9, "price": 50.0}],
        {}, 100000, 50000, new_engine_state(), "risk_on",
        {"stoploss_guard_n": 0}, T)
    assert eng["halted_until"] is None and any(o["side"] == "buy" for o in orders)
    eng = mk_eng("CLMP", 100, 5, peak=110)
    orders, _, _ = decide(
        [{"ticker": "CLMP", "score": 0.6, "price": 100.5}],
        {"CLMP": mk_pos(10, 100, 100.5)}, 100000, 50000, eng, "risk_on",
        {"trail_pct": 5, "trail_tight_pct": 5}, T)
    assert not any(o["mechanism"] == "trailing_stop" for o in orders), orders
    print("✅ 19 /set 亂值被引擎夾住")

    # 20) 壞日期事件被剔除（不會永遠賴在保險絲視窗裡）
    eng = new_engine_state()
    eng["stop_events"] = ["not-a-date", "2026-07-19|OK"]
    _, eng, _ = decide([], {}, 100000, 50000, eng, "neutral", None, T)
    assert eng["stop_events"] == ["2026-07-19|OK"], eng["stop_events"]
    print("✅ 20 壞日期事件剔除")

    # 21) alpha overlay no_entry：高分也不進場/不加碼；出場照常
    orders, _, _ = decide(
        [{"ticker": "ERN", "score": 0.9, "price": 50.0, "no_entry": True}],
        {}, 100000, 50000, new_engine_state(), "risk_on", None, T)
    assert orders == [], orders
    eng = mk_eng("ERN", 100, 5, peak=105)
    orders, _, _ = decide(
        [{"ticker": "ERN", "score": 0.9, "price": 105.0, "no_entry": True}],
        {"ERN": mk_pos(10, 100, 105)}, 100000, 50000, eng, "risk_on", None, T)
    assert not any(o["mechanism"] == "pyramid" for o in orders), orders
    eng = mk_eng("ERN", 100, 3)
    orders, _, _ = decide(
        [{"ticker": "ERN", "score": 0.1, "price": 96.0, "no_entry": True}],
        {"ERN": mk_pos(10, 100, 96)}, 100000, 50000, eng, "neutral", None, T)
    assert orders and orders[0]["mechanism"] == "stop_loss", orders   # 出場不受 veto 影響
    print("✅ 21 no_entry veto 擋進場/加碼、不擋出場")

    print("\n─ engine_status_text ─")
    eng = mk_eng("AAPL", 100, 3)
    eng["stop_events"] = [T]
    eng["cooldown"] = {"TSLA": "2026-07-25"}
    print(engine_status_text(eng, T))

    print("\ntrade_engine selftest OK ✅")
