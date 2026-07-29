"""
behavior_check.py – 交易行為體檢（/checkup；純邏輯與抓取分離、離線可測）

從 trade_journal 找系統（與人）的行為偏誤——概念參考 Vibe-Trading 的
交易紀錄分析，指標定義原生設計：

  1. 追高：每筆買進「進場價位於前 20 日高低區間的百分位」——
     平均 > 0.7 或超過一半的單在 0.8 以上 = 有追高傾向
  2. 過度交易：週均下單次數 + 短進短出比例（買進後 ≤3 個交易日就賣）
  3. 太早出場：各出場機制「賣出後 10 個交易日的後續報酬」——
     若某機制出場後平均還漲很多 = 該機制太急（追蹤停損太緊、訊號太敏感）
     ——這同時是「機制歸因」的雛形：用實測數據看哪一層在賺/賠
  4. 持有期：FIFO 配對的持有天數分佈與配對勝率

樣本門檻：各節 < 5 筆樣本 → 顯示「樣本不足」不下結論。
出場後續報酬只計「已有 10 個交易日後價」的單（避免右刪截偏誤）。
"""

from __future__ import annotations

MECH_LABELS = {
    "stop_loss": "硬停損", "trailing_stop": "追蹤停損", "scale_out": "分批鎖利",
    "signal_exit": "訊號轉弱", "dead_money": "死錢釋放", "entry": "進場",
    "pyramid": "加碼", None: "未標記", "": "未標記",
}

MIN_N = 5


def _day(e: dict) -> str:
    return str(e.get("time", ""))[:10]


def _px_on(s, d):
    """收盤序列 s 在日期 d（含之前最近一日）的價。無 → None。"""
    import pandas as pd
    try:
        sub = s.loc[:pd.Timestamp(d)]
        return float(sub.iloc[-1]) if len(sub) else None
    except Exception:
        return None


def chase_stats(journal: list, closes: dict) -> dict | None:
    """追高：買進日收盤價在前 20 日高低區間的百分位。"""
    import pandas as pd
    pcts = []
    for e in journal:
        if e.get("side") != "buy" or not e.get("submitted"):
            continue
        s = closes.get(e.get("symbol"))
        d = _day(e)
        if s is None or not d:
            continue
        try:
            win = s.loc[:pd.Timestamp(d)].tail(21)
        except Exception:
            continue
        if len(win) < 15:
            continue
        lo, hi, px = float(win.min()), float(win.max()), float(win.iloc[-1])
        if hi > lo:
            pcts.append((px - lo) / (hi - lo))
    if len(pcts) < MIN_N:
        return None
    high = sum(1 for p in pcts if p >= 0.8)
    return {"n": len(pcts), "avg_pct": sum(pcts) / len(pcts),
            "share_top20": high / len(pcts)}


def frequency_stats(journal: list) -> dict | None:
    """過度交易：週均單數 + 短進短出（FIFO 配對持有 ≤3 個交易日）比例。"""
    subs = [e for e in journal if e.get("submitted")]
    if len(subs) < MIN_N:
        return None
    import datetime as dt
    weeks = {}
    for e in subs:
        try:
            d = dt.date.fromisoformat(_day(e))
        except Exception:
            continue
        wk = f"{d.isocalendar()[0]}-W{d.isocalendar()[1]:02d}"
        weeks[wk] = weeks.get(wk, 0) + 1
    if not weeks:
        return None
    pairs = _fifo_pairs(journal)
    churn = [p for p in pairs if p["days"] <= 3]
    return {"n": len(subs), "weeks": len(weeks),
            "per_week": len(subs) / len(weeks),
            "max_week": max(weeks.values()),
            "churn_ratio": (len(churn) / len(pairs)) if pairs else None,
            "pairs": len(pairs)}


def _fifo_pairs(journal: list) -> list[dict]:
    """
    買賣 FIFO 配對 → [{symbol, days, qty, buy_day, sell_day, buy_price,
    sell_price, buy_mech, sell_mech}]。以日曆日近似交易日。
    注意：一筆賣單吃到多個 lot 會產生多對，統計為 lot-touch 等權（非成交量加權）。
    """
    return _fifo_state(journal)[0]


def _fifo_state(journal: list) -> tuple[list[dict], dict]:
    """FIFO 完整狀態：(配對清單, 未平倉 lots {sym: [[qty, buy_date, entry_dict], ...]})。"""
    import datetime as dt
    open_lots: dict[str, list] = {}
    pairs = []
    for e in journal:
        if not e.get("submitted"):
            continue
        sym, side, d = e.get("symbol"), e.get("side"), _day(e)
        try:
            day = dt.date.fromisoformat(d)
        except Exception:
            continue
        qty = float(e.get("qty") or 0)
        if qty <= 0:
            continue
        if side == "buy":
            open_lots.setdefault(sym, []).append([qty, day, e])
        elif side == "sell":
            lots = open_lots.get(sym) or []
            remain = qty
            while remain > 0 and lots:
                lot = lots[0]
                take = min(remain, lot[0])
                pairs.append({"symbol": sym, "days": (day - lot[1]).days,
                              "qty": take,
                              "buy_day": lot[1].isoformat(), "sell_day": d,
                              "buy_price": lot[2].get("price"),
                              "sell_price": e.get("price"),
                              "buy_mech": lot[2].get("mechanism"),
                              "sell_mech": e.get("mechanism")})
                lot[0] -= take
                remain -= take
                if lot[0] <= 0:
                    lots.pop(0)
    open_lots = {s: v for s, v in open_lots.items() if v}
    return pairs, open_lots


def exit_quality(journal: list, closes: dict, horizon: int = 10) -> dict | None:
    """
    太早出場（機制歸因雛形）：各出場機制賣出後 horizon 個交易日的後續報酬。
    只計已有完整後續資料的賣單；回 {mech: {n, avg_fwd}} + overall。
    """
    import pandas as pd
    per: dict[str, list] = {}
    for e in journal:
        if e.get("side") != "sell" or not e.get("submitted"):
            continue
        s = closes.get(e.get("symbol"))
        d = _day(e)
        if s is None or not d:
            continue
        try:
            idx = s.index.searchsorted(pd.Timestamp(d), side="right") - 1
        except Exception:
            continue
        if idx < 0 or idx + horizon >= len(s):
            continue                                   # 右刪截：後續不足不計
        px0, pxn = float(s.iloc[idx]), float(s.iloc[idx + horizon])
        if px0 <= 0:
            continue
        mech = e.get("mechanism") or "未標記"
        per.setdefault(mech, []).append(pxn / px0 - 1)
    total = [r for v in per.values() for r in v]
    if len(total) < MIN_N:
        return None
    return {"overall": {"n": len(total), "avg_fwd": sum(total) / len(total)},
            "by_mech": {m: {"n": len(v), "avg_fwd": sum(v) / len(v)}
                        for m, v in sorted(per.items())},
            "horizon": horizon}


def holding_stats(journal: list) -> dict | None:
    pairs = _fifo_pairs(journal)
    if len(pairs) < MIN_N:
        return None
    days = sorted(p["days"] for p in pairs)
    return {"n": len(pairs), "median_days": days[len(days) // 2],
            "avg_days": sum(days) / len(days), "max_days": days[-1]}


def analyze(journal: list, closes: dict) -> dict:
    """總入口（純函數）。closes: {sym: pd.Series（日期索引收盤價）}。
    tz-aware 索引統一去時區——否則 loc 比較 aware vs naive 會拋例外，
    被內層 try 吃掉後樣本靜默歸零、誤報「樣本不足」。"""
    norm = {}
    for k, v in closes.items():
        try:
            if getattr(v.index, "tz", None) is not None:
                v = v.tz_localize(None)
        except Exception:
            pass
        norm[k] = v
    closes = norm
    return {"chase": chase_stats(journal, closes),
            "freq": frequency_stats(journal),
            "exit": exit_quality(journal, closes),
            "hold": holding_stats(journal)}


def checkup_text(r: dict) -> str:
    """/checkup 文字（Telegram legacy Markdown：單 *；機制名轉中文避免底線）。"""
    lines = ["🩺 *交易行為體檢*（來自 trade journal 實測，非投資建議）"]

    c = r.get("chase")
    if c:
        flag = "⚠️ 有追高傾向" if (c["avg_pct"] > 0.7 or c["share_top20"] > 0.5) \
            else "✅ 未見明顯追高"
        lines.append(f"📈 追高：買進位於 20 日區間平均第 {c['avg_pct']:.0%} 位、"
                     f"{c['share_top20']:.0%} 的單在頂部 20% 區——{flag}（n={c['n']}）")
    else:
        lines.append("📈 追高：樣本不足（<5 筆買單）")

    f = r.get("freq")
    if f:
        churn = f"、短進短出（≤3 天）佔 {f['churn_ratio']:.0%}" \
            if f.get("churn_ratio") is not None else ""
        flag = "⚠️ 偏高" if f["per_week"] > 10 else "✅ 正常"
        lines.append(f"🔁 頻率：週均 {f['per_week']:.1f} 單（峰值 {f['max_week']}）"
                     f"{churn}——{flag}")
    else:
        lines.append("🔁 頻率：樣本不足")

    x = r.get("exit")
    if x:
        o = x["overall"]
        flag = "⚠️ 整體賣太早（賣後還在漲）" if o["avg_fwd"] > 0.02 else \
            ("✅ 出場後平均續跌——賣得對" if o["avg_fwd"] < -0.005 else "・出場時點中性")
        lines.append(f"🚪 出場品質：賣後 {x['horizon']} 日平均 {o['avg_fwd']:+.1%}"
                     f"（n={o['n']}）——{flag}")
        for m, v in x["by_mech"].items():
            lbl = MECH_LABELS.get(m, m).replace("_", "·")
            mark = "⚠️" if v["avg_fwd"] > 0.03 else "・"
            lines.append(f"  {mark} {lbl}：賣後 {v['avg_fwd']:+.1%}（n={v['n']}）")
    else:
        lines.append("🚪 出場品質：樣本不足（<5 筆已滿 10 日的賣單）")

    h = r.get("hold")
    if h:
        lines.append(f"⏳ 持有期：中位 {h['median_days']} 天、"
                     f"平均 {h['avg_days']:.0f} 天、最長 {h['max_days']} 天（n={h['n']}）")

    lines.append("機制別出場後仍大漲 = 該機制太急（如追蹤停損太緊可調 eng 參數）")
    return "\n".join(lines)


# ── 抓取層（需網路）───────────────────────────────────────────────────────

def fetch_closes(symbols: list[str], period: str = "9mo") -> dict:
    import yfinance as yf
    out = {}
    if not symbols:
        return out
    try:
        raw = yf.download(sorted(set(symbols)), period=period, auto_adjust=True,
                          progress=False, group_by="column")
        if raw is None or raw.empty:
            return out
        px = raw["Close"] if "Close" in raw else raw
        import pandas as pd
        if isinstance(px, pd.Series):                 # 單一標的
            out[symbols[0]] = px.dropna()
        else:
            for t in px.columns:
                s = px[t].dropna()
                if len(s):
                    out[str(t)] = s
    except Exception as e:
        print(f"behavior_check: 抓價失敗 {e}")
    return out


def run_checkup(journal_path) -> str:
    """Bot /checkup 進入點。"""
    import alpaca_trader as at
    journal = at.load_journal(journal_path)
    subs = [e for e in journal if e.get("submitted")]
    if len(subs) < MIN_N:
        return f"🩺 交易日誌僅 {len(subs)} 筆，累積 {MIN_N} 筆以上再來體檢"
    syms = sorted({e.get("symbol") for e in subs if e.get("symbol")})
    closes = fetch_closes(syms)
    if not closes:
        return "❌ 行情抓取失敗，稍後再試"
    return checkup_text(analyze(journal, closes))


# ── 自我測試（合成資料，離線）─────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    idx = pd.date_range("2026-03-02", periods=100, freq="B")

    up = pd.Series(np.linspace(100, 160, 100), index=idx)          # 一路漲
    vshape = pd.Series(np.r_[np.linspace(120, 90, 50),
                             np.linspace(90, 130, 50)], index=idx)  # V 型
    closes = {"UP": up, "VV": vshape}

    def J(day, sym, side, qty=10, mech=None):
        return {"time": f"{day}T15:00:00Z", "symbol": sym, "side": side,
                "qty": qty, "submitted": True, "mechanism": mech}

    d = [str(x.date()) for x in idx]

    # 1) 追高：UP 每天都創高 → 買進百分位全為 1.0
    j = [J(d[i], "UP", "buy") for i in range(30, 36)]
    c = chase_stats(j, closes)
    assert c and c["avg_pct"] > 0.95 and c["share_top20"] == 1.0, c
    # V 型低點買 → 低百分位
    j2 = [J(d[i], "VV", "buy") for i in range(45, 51)]
    c2 = chase_stats(j2, closes)
    assert c2 and c2["avg_pct"] < 0.3, c2
    print("✅ 1 追高偵測（頂部買 vs 低點買）")

    # 2) 太早出場：UP 在第 40 天賣 → 後 10 日仍大漲 → 正向前瞻報酬
    j3 = [J(d[i], "UP", "sell", mech="signal_exit") for i in range(40, 46)]
    x = exit_quality(j3, closes)
    assert x and x["overall"]["avg_fwd"] > 0.02, x
    assert x["by_mech"]["signal_exit"]["n"] == 6
    # 右刪截：最後 5 天的賣單（不足 10 日後續）不計
    j4 = j3 + [J(d[97], "UP", "sell", mech="stop_loss")]
    x2 = exit_quality(j4, closes)
    assert "stop_loss" not in x2["by_mech"], x2
    print("✅ 2 出場品質 + 右刪截防護")

    # 3) FIFO 配對 / 短進短出 / 持有期
    j5 = [J(d[10], "UP", "buy", 10), J(d[12], "UP", "sell", 10),   # 2 天（churn）
          J(d[20], "UP", "buy", 10), J(d[40], "UP", "sell", 6),    # 部分賣
          J(d[20], "VV", "buy", 5), J(d[50], "VV", "sell", 5),
          J(d[60], "VV", "buy", 5), J(d[61], "VV", "sell", 5)]     # 1 天（churn）
    pairs = _fifo_pairs(j5)
    assert len(pairs) == 4, pairs
    f = frequency_stats(j5)
    assert f and f["churn_ratio"] == 0.5, f
    h = holding_stats(j5 + [J(d[70], "UP", "buy", 1), J(d[71], "UP", "sell", 1)])
    assert h and h["n"] == 5, h
    print("✅ 3 FIFO/頻率/持有期")

    # 3b) tz-aware 索引：analyze 統一去時區，樣本不得靜默歸零
    up_tz = up.copy()
    up_tz.index = up_tz.index.tz_localize("America/New_York")
    r_tz = analyze(j, {"UP": up_tz, "VV": vshape})
    assert r_tz["chase"] and r_tz["chase"]["n"] == 6, r_tz["chase"]
    print("✅ 3b tz-aware 索引正規化")

    # 4) 樣本不足 → None；checkup_text 全路徑不炸且無底線
    assert chase_stats(j[:2], closes) is None
    r = analyze(j + j3 + j5, closes)
    txt = checkup_text(r)
    assert "交易行為體檢" in txt and "_" not in txt, txt   # TG Markdown 安全
    txt2 = checkup_text({"chase": None, "freq": None, "exit": None, "hold": None})
    assert "樣本不足" in txt2
    print("✅ 4 樣本門檻 + 文字輸出（Markdown 安全）")
    print()
    print(txt)
    print("\nbehavior_check selftest OK ✅")
