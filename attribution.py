"""
attribution.py – 機制歸因報告（/attrib；純邏輯與抓取分離、離線可測）

回答自我學習閉環的核心問題：**引擎的哪一層機制真的在賺錢？**

從 trade journal 以 FIFO 重建每筆已實現損益（復用 behavior_check 的配對），
按「出場機制」與「進場機制」分組統計：

  出場側：硬停損 / 追蹤停損 / 分批鎖利 / 訊號轉弱 / 死錢釋放 / 未標記
    → 各機制的筆數、勝率、平均報酬、總損益、平均持有天
    → 併上 behavior_check 的「賣後 10 日前瞻報酬」= 每個機制「賺不賺 + 急不急」
  進場側：首次進場 vs 贏家加碼（pyramid）→ 加碼到底有沒有加對

誠實邊界（報告內文也會揭露）：
  • 以 journal 重建，與券商實帳可能有出入（/closeall、手動平倉不經 journal）
    → 對帳：journal 未平倉 lot 若超過 broker 實際持股，視為已在 journal 外
      平倉，以 FIFO 淘汰最舊 lot（最舊的最可能先被平掉）且不計入歸因
  • 舊紀錄缺 price 欄 → 以當日收盤近似；缺 mechanism → 歸「未標記」
  • 各組 < 3 筆樣本照列但標「樣本少」
  • journal 有 500 筆上限（JOURNAL_CAP）：最舊買單被截掉而其賣單仍在時，
    該賣單會 FIFO 配到較新的 lot——持有天與損益歸戶可能錯位，屬 journal
    重建的固有極限
"""

from __future__ import annotations

from behavior_check import MECH_LABELS, _fifo_state, _normalize_tz, _px_on

MIN_GROUP_N = 3


def _mech_label(m) -> str:
    return MECH_LABELS.get(m, str(m)).replace("_", "·")


def reconcile_open_lots(open_lots: dict, broker_qty: dict | None) -> dict:
    """
    journal 未平倉 lot 與 broker 實際持股對帳。broker_qty: {sym: qty}；None = 不對帳。
    journal 數量 > broker → 超出部分在 journal 外被平掉 → FIFO 淘汰最舊 lot。
    個股 qty 為 None（解析失敗）→ 該股不對帳全保留（fail-open，與整體
    broker_qty=None 的哲學一致）；缺 symbol = 0 = broker 真的沒持有。
    """
    if broker_qty is None:
        return open_lots
    out = {}
    for sym, lots in open_lots.items():
        if sym in broker_qty and broker_qty[sym] is None:
            out[sym] = lots
            continue
        target = float(broker_qty.get(sym) or 0)
        total = sum(l[0] for l in lots)
        excess = total - target
        kept = []
        for lot in lots:                       # 由最舊往新淘汰
            if excess >= lot[0] - 1e-9:
                excess -= lot[0]
                continue
            if excess > 0:
                lot = [lot[0] - excess, lot[1], lot[2]]
                excess = 0.0
            kept.append(lot)
        if kept:
            out[sym] = kept
    return out


def _pair_pnl(p: dict, closes: dict) -> dict | None:
    """單一配對 → {ret, pnl, ...}。價格：journal price 優先、缺則當日收盤近似。"""
    s = closes.get(p["symbol"])
    buy_px = p.get("buy_price") or (_px_on(s, p["buy_day"]) if s is not None else None)
    sell_px = p.get("sell_price") or (_px_on(s, p["sell_day"]) if s is not None else None)
    if not buy_px or not sell_px or buy_px <= 0 or sell_px <= 0:
        return None
    ret = sell_px / buy_px - 1
    return {**p, "ret": ret, "pnl": (sell_px - buy_px) * float(p.get("qty") or 0),
            "approx": not (p.get("buy_price") and p.get("sell_price"))}


def _group(rows: list[dict], key: str) -> dict:
    """按 rows[i][key] 分組 → {label: {n, win_rate, avg_ret, total_pnl, avg_days}}。"""
    g: dict[str, list] = {}
    for r in rows:
        g.setdefault(r.get(key) or "未標記", []).append(r)
    out = {}
    for m, rs in sorted(g.items(), key=lambda kv: -sum(x["pnl"] for x in kv[1])):
        rets = [r["ret"] for r in rs]
        out[m] = {"n": len(rs),
                  "win_rate": sum(1 for x in rets if x > 0) / len(rets),
                  "avg_ret": sum(rets) / len(rets),
                  "total_pnl": sum(r["pnl"] for r in rs),
                  "avg_days": sum(r["days"] for r in rs) / len(rs)}
    return out


def benchmark_stats(rows: list[dict], bench) -> dict | None:
    """
    已實現配對 vs 基準（同持有期 SPY 報酬）——回答「賺的是 α 還是 β」
    （財金審查缺失 #2）。bench: pd.Series（日收盤）。
      excess = 配對報酬 − 基準同期報酬
      β = cov(配對, 基準同期)/var(基準同期)；α = 每筆平均超額（β 調整後截距）
      IR = mean(excess)/std(excess)——**per-trade 口徑，非年化**（不定期樣本，
           年化會假精確）
    樣本 < 5 回 None。
    """
    pts = []
    for r in rows:
        b0 = _px_on(bench, r.get("buy_day"))
        b1 = _px_on(bench, r.get("sell_day"))
        if not b0 or not b1 or b0 <= 0:
            continue
        pts.append((r["ret"], b1 / b0 - 1))
    if len(pts) < 5:
        return None
    rets = [p[0] for p in pts]
    brets = [p[1] for p in pts]
    n = len(pts)
    exc = [a - b for a, b in zip(rets, brets)]
    mr, mb = sum(rets) / n, sum(brets) / n
    var_b = sum((b - mb) ** 2 for b in brets) / n
    beta = (sum((r - mr) * (b - mb) for r, b in pts) / n / var_b) \
        if var_b > 1e-12 else None
    alpha = (mr - beta * mb) if beta is not None else None
    me = sum(exc) / n
    sd = (sum((e - me) ** 2 for e in exc) / n) ** 0.5
    ir = me / sd if sd > 1e-12 else None
    return {"n": n, "avg_excess": me,
            "beat_rate": sum(1 for e in exc if e > 0) / n,
            "beta": beta, "alpha_per_trade": alpha, "ir": ir}


def attribution(journal: list, closes: dict,
                broker_qty: dict | None = None, bench=None) -> dict | None:
    """
    總入口（純函數）。回 {realized_by_exit, realized_by_entry, open_by_entry,
    totals, n_pairs, n_approx, bench} 或 None（無可計算配對）。
    bench：基準日收盤 Series（如 SPY）→ 附相對基準歸因（α/β/IR）。
    """
    closes = _normalize_tz(closes)     # 防護放純函數入口——呼叫端繞不過（451e51b 同坑）
    pairs, open_lots = _fifo_state(journal)
    rows = [r for r in (_pair_pnl(p, closes) for p in pairs) if r]
    open_lots = reconcile_open_lots(open_lots, broker_qty)

    # 未平倉：以最新收盤 mark（無收盤價的 lot 跳過不猜）
    open_rows = []
    open_approx = 0
    for sym, lots in open_lots.items():
        s = closes.get(sym)
        if s is None or not len(s):
            continue
        last = float(s.iloc[-1])
        for qty, day, e in lots:
            entry = e.get("price") or _px_on(s, day.isoformat())
            if not entry or entry <= 0:
                continue
            if not e.get("price"):
                open_approx += 1
            open_rows.append({"symbol": sym, "qty": qty,
                              "days": 0, "ret": last / entry - 1,
                              "pnl": (last - entry) * qty,
                              "buy_mech": e.get("mechanism")})
    if not rows and not open_rows:
        return None
    return {
        "realized_by_exit": _group(rows, "sell_mech"),
        "realized_by_entry": _group(rows, "buy_mech"),
        "open_by_entry": _group(open_rows, "buy_mech") if open_rows else {},
        "totals": {"realized_pnl": sum(r["pnl"] for r in rows),
                   "unrealized_pnl": sum(r["pnl"] for r in open_rows),
                   "n_open_lots": len(open_rows)},
        "n_pairs": len(rows),
        "n_approx": sum(1 for r in rows if r["approx"]) + open_approx,
        "bench": benchmark_stats(rows, bench) if bench is not None else None,
    }


def attrib_text(r: dict | None, fwd: dict | None = None) -> str:
    """
    /attrib 文字（Telegram legacy Markdown：單 *、無底線）。
    fwd：behavior_check.exit_quality 的結果（各機制賣後 10 日前瞻），可 None。
    """
    if not r:
        return "🧾 尚無可歸因的配對交易（需至少一組買進→賣出；/journal 看紀錄）"
    lines = ["🧾 *機制歸因報告*（journal FIFO 重建，非券商正式損益）"]
    t = r["totals"]
    lines.append(f"已實現 {t['realized_pnl']:+,.0f}（{r['n_pairs']} 組配對）"
                 f"　未實現 {t['unrealized_pnl']:+,.0f}（{t['n_open_lots']} 筆持有中）")

    fwd_by = (fwd or {}).get("by_mech") or {}
    if r["realized_by_exit"]:
        lines.append("")
        lines.append("*出場機制*（賺不賺 + 急不急）：")
        for m, g in r["realized_by_exit"].items():
            tail = ""
            f = fwd_by.get(m)
            if f:
                verdict = "急了" if f["avg_fwd"] > 0.03 else \
                    ("賣得對" if f["avg_fwd"] < -0.005 else "中性")
                tail = f"｜賣後10日 {f['avg_fwd']:+.1%}（{verdict}）"
            small = "（樣本少）" if g["n"] < MIN_GROUP_N else ""
            lines.append(f"・{_mech_label(m)}：{g['total_pnl']:+,.0f}"
                         f"｜{g['n']} 筆{small}｜勝率 {g['win_rate']:.0%}"
                         f"｜均 {g['avg_ret']:+.1%}｜持有 {g['avg_days']:.0f} 天{tail}")

    if r["realized_by_entry"]:
        lines.append("")
        lines.append("*進場機制*（加碼有沒有加對）：")
        for m, g in r["realized_by_entry"].items():
            small = "（樣本少）" if g["n"] < MIN_GROUP_N else ""
            lines.append(f"・{_mech_label(m)}：{g['total_pnl']:+,.0f}"
                         f"｜{g['n']} 筆{small}｜勝率 {g['win_rate']:.0%}"
                         f"｜均 {g['avg_ret']:+.1%}")

    b = r.get("bench")
    if b:
        lines.append("")
        lines.append("*相對基準*（SPY 同持有期——賺的是 α 還是 β）：")
        lines.append(f"・超額報酬 均 {b['avg_excess']:+.1%}"
                     f"｜勝過大盤 {b['beat_rate']:.0%}（n={b['n']}）")
        seg = []
        if b.get("beta") is not None:
            seg.append(f"β {b['beta']:.2f}")
        if b.get("alpha_per_trade") is not None:
            seg.append(f"每筆 α {b['alpha_per_trade']:+.1%}")
        if b.get("ir") is not None:
            seg.append(f"IR {b['ir']:.2f}")
        if seg:
            lines.append("・" + "｜".join(seg) + "（per-trade 口徑，非年化）")

    if r["open_by_entry"]:
        lines.append("")
        lines.append("*持有中*（未實現，最新收盤 mark）：")
        for m, g in r["open_by_entry"].items():
            lines.append(f"・{_mech_label(m)}：{g['total_pnl']:+,.0f}"
                         f"｜{g['n']} 筆｜均 {g['avg_ret']:+.1%}")

    if r["n_approx"]:
        lines.append(f"（{r['n_approx']} 組舊紀錄缺成交價，以當日收盤近似）")
    lines.append("配對外的平倉（/closeall、手動）不計入；非投資建議")
    return "\n".join(lines)


# ── 抓取層 + Bot 進入點 ───────────────────────────────────────────────────

def run_attrib(journal_path) -> str:
    import alpaca_trader as at
    from behavior_check import fetch_closes, exit_quality
    journal = at.load_journal(journal_path)
    subs = [e for e in journal if e.get("submitted")]
    if not subs:
        return "🧾 交易日誌是空的——開 /autotrade on 累積紀錄後再來"
    syms = sorted({e.get("symbol") for e in subs if e.get("symbol")})
    # 連 SPY 一起抓（相對基準歸因用；SPY 本身也可能是持倉標的，留在 closes 內）
    closes = _normalize_tz(fetch_closes(sorted(set(syms) | {"SPY"})))
    if not closes:
        return "❌ 行情抓取失敗，稍後再試"

    broker_qty = None
    try:
        import os
        key = os.environ.get("ALPACA_KEY_ID", "")
        sec = os.environ.get("ALPACA_SECRET_KEY", "")
        if key and sec:
            pos = at.get_positions(key, sec)
            if pos is not None:
                broker_qty = {}
                for s, p in pos.items():
                    try:
                        broker_qty[s] = abs(float(p.get("qty")))
                    except (TypeError, ValueError):
                        broker_qty[s] = None    # 解析失敗 → 該股 fail-open 不對帳
    except Exception:
        pass

    r = attribution(journal, closes, broker_qty, bench=closes.get("SPY"))
    fwd = exit_quality(journal, closes)
    return attrib_text(r, fwd)


# ── 自我測試（合成資料，離線）─────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    idx = pd.date_range("2026-03-02", periods=100, freq="B")
    d = [str(x.date()) for x in idx]
    up = pd.Series(np.linspace(100, 160, 100), index=idx)
    closes = {"UP": up}

    def J(day, sym, side, qty=10, mech=None, price=None):
        return {"time": f"{day}T15:00:00Z", "symbol": sym, "side": side,
                "qty": qty, "submitted": True, "mechanism": mech, "price": price}

    # 1) 已實現按出場機制分組：帶價 vs 缺價（收盤近似）
    j = [J(d[10], "UP", "buy", 10, "entry", 100.0),
         J(d[20], "UP", "sell", 10, "trailing_stop", 130.0),     # +30%、+300
         J(d[30], "UP", "buy", 10, "entry"),                     # 缺價 → 收盤近似
         J(d[40], "UP", "sell", 10, "signal_exit")]              # 缺價
    r = attribution(j, closes)
    assert r and r["n_pairs"] == 2 and r["n_approx"] == 1, r
    ts = r["realized_by_exit"]["trailing_stop"]
    assert ts["n"] == 1 and abs(ts["total_pnl"] - 300) < 1e-6 and ts["win_rate"] == 1.0
    assert "signal_exit" in r["realized_by_exit"]
    assert r["realized_by_entry"]["entry"]["n"] == 2
    print("✅ 1 已實現分組 + 缺價近似")

    # 2) 未平倉 mark-to-market 按進場機制
    j2 = j + [J(d[50], "UP", "buy", 5, "pyramid", 140.0)]
    r2 = attribution(j2, closes)
    po = r2["open_by_entry"]["pyramid"]
    assert po["n"] == 1 and abs(po["total_pnl"] - (160 - 140) * 5) < 1e-6, po
    assert abs(r2["totals"]["unrealized_pnl"] - 100) < 1e-6
    print("✅ 2 未平倉 mark")

    # 3) broker 對帳：journal 有 15 股未平、broker 只剩 5 → 淘汰最舊 10 股 lot
    j3 = [J(d[10], "UP", "buy", 10, "entry", 100.0),
          J(d[50], "UP", "buy", 5, "pyramid", 140.0)]
    r3 = attribution(j3, closes, broker_qty={"UP": 5})
    assert r3["totals"]["n_open_lots"] == 1
    assert "pyramid" in r3["open_by_entry"] and "entry" not in r3["open_by_entry"], r3
    # 部分淘汰：broker 剩 12 → 最舊 lot 削到 7
    r3b = attribution(j3, closes, broker_qty={"UP": 12})
    assert abs(r3b["open_by_entry"]["entry"]["total_pnl"] - (160 - 100) * 7) < 1e-6
    # broker 全平 + 無已實現配對 → 整體無可歸因 → None
    r3c = attribution(j3, closes, broker_qty={})
    assert r3c is None
    print("✅ 3 broker 對帳（全淘汰/部分/清零）")

    # 4) 文字輸出：TG Markdown 安全、fwd 併入判語、空資料
    from behavior_check import exit_quality
    jj = j + [J(d[i], "UP", "sell", 1, "signal_exit") for i in range(41, 45)]
    jj = [J(d[i], "UP", "buy", 1, "entry", 100.0) for i in range(5, 10)] + jj
    fwd = exit_quality(jj, closes)
    txt = attrib_text(attribution(jj, closes), fwd)
    assert "機制歸因報告" in txt and "_" not in txt and txt.count("*") % 2 == 0, txt
    assert "賣後10日" in txt and ("急了" in txt or "中性" in txt or "賣得對" in txt)
    assert attrib_text(None).startswith("🧾 尚無")
    print("✅ 4 attrib_text（Markdown 安全 + fwd 判語）")

    # 5) tz-aware 迴歸：缺價 journal + tz-aware closes → 配對不得靜默蒸發
    #    （451e51b 同坑：防護在 analyze，run_attrib 繞過——現已下沉到 attribution 入口）
    up_tz = up.copy()
    up_tz.index = up_tz.index.tz_localize("America/New_York")
    r5 = attribution(j, {"UP": up_tz})           # j 內含缺價配對（收盤近似路徑）
    assert r5 and r5["n_pairs"] == 2 and r5["n_approx"] == 1, r5
    print("✅ 5 tz-aware 入口防護（缺價近似不蒸發）")

    # 6) 邊界：sell_price 負值擋掉；broker qty=None → 該股 fail-open
    bad = [J(d[10], "UP", "buy", 10, "entry", 100.0),
           J(d[20], "UP", "sell", 10, "stop_loss", -5.0)]
    r6 = attribution(bad, closes)
    assert r6 is None or "stop_loss" not in r6["realized_by_exit"], r6
    j6 = [J(d[10], "UP", "buy", 10, "entry", 100.0)]
    r6b = attribution(j6, closes, broker_qty={"UP": None})
    assert r6b and r6b["totals"]["n_open_lots"] == 1, r6b   # 不對帳 → 保留
    print("✅ 6 負價防護 + broker None fail-open")

    # 7) 相對基準歸因：UP 跑贏慢漲基準 → 超額為正、β 有值、IR 有值；樣本<5 → None
    bench = pd.Series(np.linspace(100, 112, 100), index=idx)     # 慢漲 12%
    jb = [J(d[10 + i * 8], "UP", "buy", 10, "entry", None) for i in range(6)] + \
         [J(d[14 + i * 8], "UP", "sell", 10, "signal_exit", None) for i in range(6)]
    rb = attribution(jb, {"UP": up}, bench=bench)
    bs = rb["bench"]
    assert bs and bs["n"] == 6, bs
    assert bs["avg_excess"] > 0 and bs["beat_rate"] == 1.0, bs   # UP 斜率遠勝基準
    assert bs["beta"] is not None and bs["ir"] is not None and bs["ir"] > 0, bs
    assert benchmark_stats([], bench) is None                     # 樣本不足
    txt_b = attrib_text(rb)
    assert "相對基準" in txt_b and "α" in txt_b and "_" not in txt_b, txt_b
    r_nb = attribution(jb, {"UP": up})                            # 無基準 → 無此段
    assert r_nb["bench"] is None and "相對基準" not in attrib_text(r_nb)
    print("✅ 7 相對基準歸因（α/β/IR + 無基準降級）")
    print()
    print(txt)
    print("\nattribution selftest OK ✅")
