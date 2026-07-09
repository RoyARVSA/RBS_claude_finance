"""
plan_backtest.py — 當日交易計畫（trade_plan）歷史回測 + walk-forward 校準
RBS Finance Dashboard

把過去 ~60 個交易日（yfinance 5 分 K 上限）逐日「重放」：每 15 分鐘用
**當時看得到的資料**重新產生訂單票，模擬進場區成交、停損/停利/收盤出場，
統計各型態（ORB 突破 / VWAP 回踩）與信心等級的實證勝率與期望值（R 倍數），
再以 walk-forward（前 60% 訓練、後 40% 驗證）產生校準建議，回饋給 build_ticket
（負期望型態停用、不穩定型態降信心）——讓 /today 的判定吃歷史實證而非只憑規則。

回測紀律（與 backtest.py 同一套鐵律）：
- 無前視：日線只用「該交易日以前」的資料；訊號 K 棒之後的 K 棒才能成交
- 保守成交：從上方進入進場區以區間上緣成交；同根 K 棒同時觸及停損與停利算停損
- 扣來回交易成本 COST_RT（0.1%）
- 校準只降不升（停用/降級），避免用歷史樣本自我吹捧

純邏輯（simulate_session / aggregate / walk_forward_calibrate / stats_text）離線可測；
fetch_history / run 需網路。非投資建議。
"""
from __future__ import annotations

import pandas as pd

from trade_plan import (ORB_MINUTES, STOP_ATR_MULT, TARGET_RR,
                        build_ticket, daily_gate, intraday_metrics)

COST_RT = 0.001          # 來回交易成本（與 backtest.py 一致）
STEP_BARS = 3            # 每 3 根 5 分 K（=15 分鐘）評估一次，貼近 bot cron 節奏
MIN_TRADES = 5           # 校準判斷的最小樣本數（低於此不動作）
TRAIN_FRAC = 0.6         # walk-forward 訓練段比例
CONF_BUCKETS = ((2, "2"), (3, "3"), (4, "4+"))


# ── 純邏輯：單日重放 ──────────────────────────────────────────────────────────

def simulate_session(sess: pd.DataFrame, daily_upto: pd.DataFrame,
                     ticker: str = "?", session_minutes: float = 390.0,
                     params: dict | None = None,
                     cache: dict | None = None) -> dict | None:
    """
    重放一個交易日。sess：該日全部盤中 K（OHLCV，時間索引）；
    daily_upto：**只含該日以前**的日線（呼叫端負責切片——這是無前視的關鍵）。
    params：{orb_minutes, stop_atr_mult, target_rr}（缺省用 trade_plan 預設）。
    cache：尋優跨參數組重用（daily_gate 不隨參數變、intraday_metrics 只隨
    orb_minutes 變——27 組網格可省 ~9 倍計算）。
    回一筆交易紀錄 {ticker,date,setup,conf,filled,fill,exit,exit_kind,r,win}
    或 None（整天無可執行訊號）。每個 ticker-日最多一筆（取第一個訊號）。
    """
    if sess is None or len(sess) < 8 or daily_upto is None or len(daily_upto) < 60:
        return None
    prm = params or {}
    orb_m = int(prm.get("orb_minutes", ORB_MINUTES))
    stop_m = float(prm.get("stop_atr_mult", STOP_ATR_MULT))
    tgt_rr = float(prm.get("target_rr", TARGET_RR))
    day_key = str(sess.index[-1].date())
    if cache is not None:
        gk = ("gate", ticker, day_key)
        gate = cache.get(gk)
        if gate is None:
            gate = daily_gate(daily_upto)
            cache[gk] = gate
    else:
        gate = daily_gate(daily_upto)

    n = len(sess)
    # 每 STEP_BARS 根評估一次（開盤 15 分後開始；ORB 未滿時與實盤一樣用部分區間）
    for i in range(STEP_BARS, n - 1, STEP_BARS):
        if cache is not None:
            mk = ("m", ticker, day_key, i, orb_m)
            m = cache.get(mk, False)
            if m is False:
                m = intraday_metrics(sess.iloc[:i + 1], daily_upto, orb_minutes=orb_m,
                                     session_minutes=session_minutes)
                cache[mk] = m
        else:
            m = intraday_metrics(sess.iloc[:i + 1], daily_upto, orb_minutes=orb_m,
                                 session_minutes=session_minutes)
        if not m:
            continue
        t = build_ticket(ticker, m, gate, stop_atr_mult=stop_m, target_rr=tgt_rr)
        if t["action"] not in ("買進", "小量試單"):
            continue
        entry_lo, entry_hi, stop, target = (t["entry_lo"], t["entry_hi"],
                                            t["stop"], t["target"])
        # ── 訊號之後的 K 棒才可成交（無前視）──
        fill = None
        for j in range(i + 1, n):
            o = float(sess["Open"].iloc[j])
            hi_j, lo_j = float(sess["High"].iloc[j]), float(sess["Low"].iloc[j])
            if entry_lo <= o <= entry_hi:
                fill = o                       # 開在區間內：以開盤價成交
            elif o > entry_hi and lo_j <= entry_hi:
                fill = entry_hi                # 從上方回落進區：區間上緣（保守）
            elif o < entry_lo and hi_j >= entry_lo:
                fill = entry_lo                # 從下方觸及：限價單掛區間下緣
            if fill is not None:
                fill_j = j
                break
        rec = {"ticker": ticker, "date": str(sess.index[-1].date()),
               "setup": t["setup"], "conf": int(t["confidence"]),
               "filled": fill is not None}
        if fill is None:
            return rec                          # 有訊號但整天沒成交
        risk_ps = fill - stop
        if risk_ps <= 0:
            return rec
        # ── 出場：停損優先（同根 K 同時觸及算最壞情況）──
        exit_px, exit_kind = None, None
        for j in range(fill_j, n):
            hi_j, lo_j = float(sess["High"].iloc[j]), float(sess["Low"].iloc[j])
            if lo_j <= stop:
                exit_px, exit_kind = stop, "stop"
                break
            if hi_j >= target:
                exit_px, exit_kind = target, "target"
                break
        if exit_px is None:
            exit_px, exit_kind = float(sess["Close"].iloc[-1]), "eod"
        r = (exit_px - fill) / risk_ps - COST_RT * fill / risk_ps   # 成本換算成 R
        rec.update({"fill": round(fill, 4), "exit": round(exit_px, 4),
                    "exit_kind": exit_kind, "r": round(r, 3), "win": r > 0})
        return rec
    return None


# ── 純邏輯：彙總與校準 ────────────────────────────────────────────────────────

def _agg(trades: list[dict]) -> dict:
    fills = [t for t in trades if t.get("filled") and "r" in t]
    n = len(fills)
    if n == 0:
        return {"n": 0, "signals": len(trades), "win_rate": None,
                "avg_r": None, "total_r": None}
    wins = sum(1 for t in fills if t["win"])
    rs = [t["r"] for t in fills]
    return {"n": n, "signals": len(trades), "win_rate": wins / n,
            "avg_r": sum(rs) / n, "total_r": sum(rs)}


def aggregate(trades: list[dict]) -> dict:
    """整體 + 各 setup + 各信心桶的統計。"""
    out = {"overall": _agg(trades), "by_setup": {}, "by_conf": {}}
    for s in sorted({t["setup"] for t in trades if t.get("setup")}):
        out["by_setup"][s] = _agg([t for t in trades if t.get("setup") == s])
    for lo, label in CONF_BUCKETS:
        hi = next((l2 for l2, _ in CONF_BUCKETS if l2 > lo), 99)
        out["by_conf"][label] = _agg([t for t in trades
                                      if lo <= t.get("conf", 0) < hi])
    return out


def walk_forward_calibrate(trades: list[dict]) -> dict:
    """
    walk-forward 校準：依日期排序，前 TRAIN_FRAC 訓練、其餘驗證。
    只降不升：兩段皆負期望（樣本足夠）→ 停用該 setup；
    訓練正但驗證負（不穩定）→ conf_delta = -1；其餘不動。
    回 {"sessions": n_days, "split_date": d, "setups": {setup: {...}}}。
    """
    dated = sorted([t for t in trades if t.get("date")], key=lambda t: t["date"])
    days = sorted({t["date"] for t in dated})
    calib = {"sessions": len(days), "split_date": None, "setups": {}}
    if len(days) < 10:
        return calib                            # 樣本太少，不校準
    split = days[max(1, int(len(days) * TRAIN_FRAC)) - 1]
    calib["split_date"] = split
    train = [t for t in dated if t["date"] <= split]
    val = [t for t in dated if t["date"] > split]
    for s in sorted({t["setup"] for t in dated if t.get("setup")}):
        tr = _agg([t for t in train if t["setup"] == s])
        va = _agg([t for t in val if t["setup"] == s])
        enabled, delta, why = True, 0, "樣本不足或表現可接受，維持原狀"
        if tr["n"] >= MIN_TRADES and va["n"] >= max(2, MIN_TRADES // 2):
            if tr["avg_r"] < 0 and va["avg_r"] < 0:
                enabled, why = False, "訓練與驗證段皆負期望——停用"
            elif tr["avg_r"] > 0 and va["avg_r"] < 0:
                delta, why = -1, "訓練正、驗證負（不穩定）——信心降一級"
        calib["setups"][s] = {"enabled": enabled, "conf_delta": delta,
                              "why": why,
                              "train": {k: (round(v, 3) if isinstance(v, float) else v)
                                        for k, v in tr.items()},
                              "val": {k: (round(v, 3) if isinstance(v, float) else v)
                                      for k, v in va.items()}}
    return calib


def stats_text(agg: dict, calib: dict | None = None, n_tickers: int = 0) -> str:
    """統計 → 文字（bot / 下載共用）。"""
    ov = agg["overall"]
    lines = [f"📜 當日計畫回測（{n_tickers} 檔 × {calib.get('sessions', '?') if calib else '?'} 個交易日、15 分鐘節奏重放）"]
    if ov["n"] == 0:
        lines.append("期間內無成交訊號。")
        return "\n".join(lines)
    lines.append(f"成交 {ov['n']} 筆（訊號 {ov['signals']}）｜勝率 {ov['win_rate']:.0%}｜"
                 f"平均 {ov['avg_r']:+.2f}R｜累計 {ov['total_r']:+.1f}R")
    for s, g in agg["by_setup"].items():
        if g["n"]:
            lines.append(f"・{s}：{g['n']} 筆、勝率 {g['win_rate']:.0%}、平均 {g['avg_r']:+.2f}R")
    for c, g in agg["by_conf"].items():
        if g["n"]:
            lines.append(f"・信心 {c}：{g['n']} 筆、勝率 {g['win_rate']:.0%}、平均 {g['avg_r']:+.2f}R")
    if calib and calib.get("setups"):
        lines.append(f"\n🔧 walk-forward 校準（訓練 ≤{calib['split_date']}、其後驗證）：")
        for s, c in calib["setups"].items():
            state = "⛔ 停用" if not c["enabled"] else ("⬇️ 信心-1" if c["conf_delta"] else "✅ 維持")
            lines.append(f"・{s}：{state}——{c['why']}")
    lines.append("\n⚠️ 歷史模擬（含 0.1% 成本、停損優先保守假設、已剔除財報 ±1 日；"
                 "模型假設＝開盤起每 15 分看一次並執行第一張票），非投資建議。")
    return "\n".join(lines)


# ── 抓取層（需網路）───────────────────────────────────────────────────────────

def fetch_history(tickers: list[str], period: str = "60d") -> dict[str, dict]:
    """每檔抓 60 天 5 分 K + 1 年日線。回 {ticker: {bars, daily}}；失敗的略過。"""
    import yfinance as yf
    out = {}
    for tk in tickers:
        try:
            bars = yf.download(tk, period=period, interval="5m",
                               auto_adjust=True, progress=False, prepost=False)
            daily = yf.download(tk, period="1y", interval="1d",
                                auto_adjust=True, progress=False)
            for df in (bars, daily):
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
            earn: set = set()
            try:            # 歷史財報日：重放時略過 ±1 日（實盤 /today 本就迴避）
                edf = yf.Ticker(tk).get_earnings_dates(limit=16)
                if edf is not None and len(edf):
                    earn = {ts.date() if hasattr(ts, "date") else ts
                            for ts in edf.index}
            except Exception:
                pass
            if len(bars) > 50 and len(daily) > 60:
                out[tk] = {"bars": bars, "daily": daily, "earnings": earn}
        except Exception:
            continue
    return out


def _replay(data: dict, params: dict | None = None,
            cache: dict | None = None) -> list[dict]:
    """逐檔逐日重放（data 來自 fetch_history 或測試注入）。"""
    trades: list[dict] = []
    for tk, dd in data.items():
        bars, daily = dd["bars"], dd["daily"]
        sess_min = 270.0 if tk.upper().endswith((".TW", ".TWO")) else 390.0
        earn = dd.get("earnings") or set()
        days = sorted({d.date() for d in bars.index})
        for day in days:
            if any(abs((day - e).days) <= 1 for e in earn):
                continue        # 財報 ±1 日：實盤一律迴避，重放同樣剔除（母體一致）
            sess = bars[[d.date() == day for d in bars.index]]
            daily_upto = daily[daily.index.date < day]        # 無前視：只用該日以前
            rec = simulate_session(sess, daily_upto, ticker=tk,
                                   session_minutes=sess_min, params=params,
                                   cache=cache)
            if rec:
                trades.append(rec)
    return trades


def run(tickers: list[str], max_tickers: int = 12,
        params: dict | None = None,
        data: dict | None = None) -> tuple[dict, dict, list[dict]]:
    """
    整條流程：抓資料 → 逐檔逐日重放 → (aggregate, calib, trades)。
    台股（.TW/.TWO）session_minutes=270。data 可注入（尋優/測試重用免重抓）。
    """
    tickers = list(dict.fromkeys(tickers))[:max_tickers]
    if data is None:
        data = fetch_history(tickers)
    trades = _replay(data, params)
    return aggregate(trades), walk_forward_calibrate(trades), trades


# ── 參數尋優（walk-forward：訓練段排序、驗證段把關）──────────────────────────

GRID = {"orb_minutes": (15, 30, 45),
        "stop_atr_mult": (1.0, 1.5, 2.0),
        "target_rr": (1.5, 2.0, 2.5)}

_BASELINE = {"orb_minutes": ORB_MINUTES, "stop_atr_mult": STOP_ATR_MULT,
             "target_rr": TARGET_RR}


def _split_stats(trades: list[dict]) -> dict | None:
    """依日期切 walk-forward（TRAIN_FRAC），回 train/val 統計；天數 <10 回 None。"""
    dated = sorted([t for t in trades if t.get("date")], key=lambda t: t["date"])
    days = sorted({t["date"] for t in dated})
    if len(days) < 10:
        return None
    split = days[max(1, int(len(days) * TRAIN_FRAC)) - 1]
    return {"split": split, "sessions": len(days),
            "train": _agg([t for t in dated if t["date"] <= split]),
            "val": _agg([t for t in dated if t["date"] > split])}


def optimize(tickers: list[str], grid: dict | None = None,
             max_tickers: int = 8, data: dict | None = None) -> dict:
    """
    網格掃參數（ORB 分鐘 × 停損 ATR 倍數 × 目標 R:R），每組完整重放。
    穩健挑選：訓練段 avg_R 排序（樣本 ≥ MIN_TRADES）→ 取第一個「驗證段也正期望」
    的組合為 best；再與現行預設比驗證段——**沒有明確勝過預設就不推薦**
    （recommend=None＝維持預設，避免為調而調的過擬合）。
    回 {"results", "best", "baseline", "recommend", "n_tickers"}；
    results 各項含 params/train/val/trades。
    """
    from itertools import product
    grid = grid or GRID
    tickers = list(dict.fromkeys(tickers))[:max_tickers]
    if data is None:
        data = fetch_history(tickers)
    keys = list(grid)
    results = []
    cache: dict = {}                 # 跨參數組共用 gate/metrics（省 ~9 倍計算）
    for combo in product(*(grid[k] for k in keys)):
        prm = dict(zip(keys, combo))
        trades = _replay(data, prm, cache=cache)
        ss = _split_stats(trades)
        if ss:
            results.append({"params": prm, "trades": trades, **ss})
    out = {"results": results, "best": None, "baseline": None,
           "recommend": None, "n_tickers": len(data)}
    if not results:
        return out
    base_prm = {k: _BASELINE[k] for k in keys if k in _BASELINE}
    baseline = next((r for r in results if r["params"] == base_prm), None)
    if baseline is None:
        tr_b = _replay(data, base_prm, cache=cache)
        ss_b = _split_stats(tr_b)
        if ss_b:
            baseline = {"params": base_prm, "trades": tr_b, **ss_b}
    out["baseline"] = baseline
    eligible = [r for r in results if r["train"]["n"] >= MIN_TRADES]
    eligible.sort(key=lambda r: r["train"]["avg_r"], reverse=True)
    best = next((r for r in eligible
                 if r["val"]["n"] >= max(2, MIN_TRADES // 2)
                 and r["val"]["avg_r"] > 0), None)
    out["best"] = best
    if best and best["params"] != (baseline or {}).get("params"):
        base_val = (baseline or {}).get("val") or {}
        # 推薦門檻：驗證段 avg_R 至少比預設好 0.05R（預設無資料時只要驗證為正）
        if base_val.get("avg_r") is None or \
           best["val"]["avg_r"] >= base_val["avg_r"] + 0.05:
            out["recommend"] = best
    return out


def opt_text(opt: dict, top_n: int = 5) -> str:
    """尋優結果 → 文字（bot / 下載共用）。"""
    lines = [f"🔧 當日計畫參數尋優（{opt['n_tickers']} 檔、{len(opt['results'])} 組參數、"
             "walk-forward 前 60% 訓練/後 40% 驗證）"]
    if not opt["results"]:
        lines.append("資料不足（需 ≥10 個交易日），未產生結果。")
        return "\n".join(lines)

    def _fmt(r):
        p = r["params"]
        tr, va = r["train"], r["val"]
        return (f"ORB{p.get('orb_minutes', '?')}分/停損{p.get('stop_atr_mult', '?')}×ATR/"
                f"目標{p.get('target_rr', '?')}R → 訓練 {tr['n']} 筆 "
                f"{(tr['avg_r'] if tr['avg_r'] is not None else 0):+.2f}R｜"
                f"驗證 {va['n']} 筆 "
                f"{(va['avg_r'] if va['avg_r'] is not None else 0):+.2f}R")

    if opt["baseline"]:
        lines.append(f"基準（現行預設）：{_fmt(opt['baseline'])}")
    ranked = sorted([r for r in opt["results"] if r["train"]["n"] >= MIN_TRADES],
                    key=lambda r: r["train"]["avg_r"], reverse=True)
    for i, r in enumerate(ranked[:top_n], 1):
        lines.append(f"{i}. {_fmt(r)}")
    if opt["recommend"]:
        p = {**_BASELINE, **opt["recommend"]["params"]}   # partial grid 用預設補齊
        lines.append(f"\n✅ 推薦：ORB {p['orb_minutes']} 分、停損 {p['stop_atr_mult']}×ATR、"
                     f"目標 {p['target_rr']}R（驗證段明確勝過預設）")
    else:
        lines.append("\n➖ 無組合在驗證段明確勝過現行預設——維持預設（不為調而調）")
    lines.append("\n⚠️ 歷史尋優極易過擬合；只信「訓練與驗證都好」的組合。非投資建議。")
    return "\n".join(lines)


# ── CLI 自我測試（離線純邏輯，合成 K 棒）─────────────────────────────────────

if __name__ == "__main__":
    import numpy as np

    def _daily(n=120, up=True, seed=1):
        rng = np.random.default_rng(seed)
        drift = 0.004 if up else -0.004
        px = 100 * np.cumprod(1 + rng.normal(drift, 0.008, n))
        idx = pd.bdate_range("2026-01-01", periods=n)
        return pd.DataFrame({"Open": px, "High": px * 1.01, "Low": px * 0.99,
                             "Close": px, "Volume": np.full(n, 1e6)}, index=idx)

    def _sess(day: str, path: list[tuple], vol=3e5):
        """path：[(o,h,l,c), ...] 每根 5 分 K。"""
        idx = pd.date_range(f"{day} 09:30", periods=len(path), freq="5min")
        df = pd.DataFrame(path, columns=["Open", "High", "Low", "Close"], index=idx)
        df["Volume"] = vol
        return df

    daily = _daily(up=True)
    base = float(daily["Close"].iloc[-1])
    d0 = str((daily.index[-1] + pd.Timedelta(days=1)).date())

    # 情境 1：ORB 突破後一路漲 → 應成交且獲利出場（target 或 eod，r > 0）
    p = base
    path = [(p, p * 1.002, p * 0.999, p * 1.001)] * 6                 # 開盤區間（30 分）
    path += [(p * 1.001, p * 1.02, p * 1.000, p * 1.018)] * 3         # 放量突破
    path += [(p * 1.018, p * 1.06, p * 1.015, p * 1.055)] * 12        # 續漲
    rec1 = simulate_session(_sess(d0, path), daily, "UPUP")
    assert rec1 and rec1["filled"] and rec1["r"] > 0, rec1
    assert rec1["setup"] in ("ORB 突破", "VWAP 回踩"), rec1

    # 情境 2：突破後崩跌 → 應打到停損，r ≈ -1（含成本略低於 -1）
    path2 = [(p, p * 1.002, p * 0.999, p * 1.001)] * 6
    path2 += [(p * 1.001, p * 1.02, p * 1.000, p * 1.018)] * 3
    path2 += [(p * 1.018, p * 1.018, p * 0.90, p * 0.905)] * 6        # 崩跌破停損
    rec2 = simulate_session(_sess(d0, path2), daily, "CRASH")
    assert rec2 and rec2["filled"] and rec2["exit_kind"] == "stop" and rec2["r"] < 0, rec2
    assert rec2["r"] >= -1.4, rec2            # 停損出場的 R 應在 -1 附近（含成本）

    # 情境 3：日線空頭 → 全日無訊號（K 棒基價須取自空頭日線，避免假跳空）
    idx_dn = pd.bdate_range("2026-01-01", periods=120)
    # 前段緩跌、後段加速下跌（讓 MACD 柱明確轉負，閘門必為 bearish）
    steps = np.r_[np.full(80, 0.999), np.full(40, 0.99)]
    px_dn = pd.Series(100 * np.cumprod(steps), index=idx_dn)
    daily_dn = pd.DataFrame({"Open": px_dn, "High": px_dn * 1.01, "Low": px_dn * 0.99,
                             "Close": px_dn, "Volume": np.full(120, 1e6)}, index=idx_dn)
    assert daily_gate(daily_dn)["bias"] == "bearish", daily_gate(daily_dn)
    pdn = float(daily_dn["Close"].iloc[-1])
    path3 = [(pdn, pdn * 1.002, pdn * 0.999, pdn * 1.001)] * 6
    path3 += [(pdn * 1.001, pdn * 1.02, pdn * 1.000, pdn * 1.018)] * 15
    rec3 = simulate_session(_sess(d0, path3), daily_dn, "BEAR")
    assert rec3 is None, rec3

    # 情境 4：同根 K 同時觸及停損與停利 → 算停損（保守）
    #   人工構造：成交後下一根 K 高低都跨過 target/stop
    path4 = [(p, p * 1.002, p * 0.999, p * 1.001)] * 6
    path4 += [(p * 1.001, p * 1.02, p * 1.000, p * 1.018)] * 3
    path4 += [(p * 1.018, p * 1.30, p * 0.80, p * 1.0)] * 2           # 巨幅震盪
    rec4 = simulate_session(_sess(d0, path4), daily, "WILD")
    assert rec4 and rec4["exit_kind"] == "stop", rec4

    # 彙總與校準
    trades = []
    for k in range(30):
        day = f"2026-03-{(k % 28) + 1:02d}"
        # ORB：前段賺後段賠（不穩定→降級）；VWAP：全程賠（→停用）
        early = k < 18
        trades.append({"ticker": "A", "date": day, "setup": "ORB 突破", "conf": 3,
                       "filled": True, "r": 0.5 if early else -0.4,
                       "win": early})
        trades.append({"ticker": "B", "date": day, "setup": "VWAP 回踩", "conf": 2,
                       "filled": True, "r": -0.3, "win": False})
    agg = aggregate(trades)
    assert agg["overall"]["n"] == 60
    assert agg["by_setup"]["VWAP 回踩"]["win_rate"] == 0.0
    cal = walk_forward_calibrate(trades)
    assert cal["setups"]["VWAP 回踩"]["enabled"] is False, cal
    assert cal["setups"]["ORB 突破"]["enabled"] and \
        cal["setups"]["ORB 突破"]["conf_delta"] == -1, cal
    txt = stats_text(agg, cal, n_tickers=2)
    assert "停用" in txt and "非投資建議" in txt

    # 樣本 <10 天 → 不校準
    assert walk_forward_calibrate(trades[:6])["setups"] == {}

    # 參數化重放：target_rr=0.5 → 情境 1 應改為快速停利出場（而非 eod）
    rec1b = simulate_session(_sess(d0, path), daily, "UPUP",
                             params={"target_rr": 0.5})
    assert rec1b and rec1b["exit_kind"] == "target" and rec1b["r"] > 0, rec1b

    # 參數尋優（注入 12 個合成上漲日；離線）
    sessions = []
    for k in range(12):
        dk = str((daily.index[-1] + pd.Timedelta(days=k + 1)).date())
        sessions.append(_sess(dk, path))
    data_syn = {"SYN": {"bars": pd.concat(sessions), "daily": daily}}
    opt = optimize(["SYN"], grid={"target_rr": (0.5, 8.0)}, data=data_syn)
    assert len(opt["results"]) == 2 and opt["best"] is not None, opt["results"]
    assert opt["baseline"] is not None            # 預設 2.0R 不在 grid → 另行計算
    assert opt["baseline"]["params"]["target_rr"] == 2.0
    ot = opt_text(opt)
    assert ("推薦" in ot or "維持預設" in ot) and "過擬合" in ot
    # run() 可注入 data + params（免網路）
    agg_i, cal_i, tr_i = run(["SYN"], data=data_syn, params={"target_rr": 0.5})
    assert agg_i["overall"]["n"] == 12 and agg_i["overall"]["win_rate"] == 1.0, agg_i

    print("✅ plan_backtest 離線自我測試通過"
          f"（情境1 {rec1['exit_kind']} {rec1['r']:+.2f}R、情境2 {rec2['r']:+.2f}R）")
