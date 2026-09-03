"""
engine_backtest.py – trade_engine 整台引擎的歷史重放 + 參數學習（walk-forward）

回答使用者的問題：**「如果過去這段時間用不同的買進/賣出參數，引擎會賺多少？
能不能從歷史裡學出更好的規則？」**

與既有回測的分工：
  • backtest.py      → 單一訊號的 triple-barrier（訊號有沒有 edge）
  • plan_backtest.py → 當沖計畫（ORB/VWAP）的 60 日重放
  • 本模組          → **整台波段引擎**（進場門檻 × 停損/追蹤/分批/死錢 × 保險絲）
                      逐日重放，所有出場機制、部位上限、regime 三態一起跑

方法（誠實版的「強化學習」：參數搜索 + walk-forward，不是深度 RL——
日 K 樣本太少，真 RL 必然學到雜訊）：
  1. 逐日評分：每檔每日只用 **當日以前** 的 K 棒算 composite_score（無前視）
  2. 訊號在 t 日收盤決策 → **t+1 日開盤成交**（鐵律：下一根 K 棒進場），
     單邊 0.05% 成本（來回 0.1%，與 backtest/plan_backtest 一致）
  3. 參數網格逐組完整重放（引擎狀態、保險絲、追蹤停損全部重算）
  4. 三段 walk-forward：50% 訓練（排序）/ 25% 驗證（挑選）/ 25% holdout
     （只看一次做最終把關）——與 plan_backtest.optimize 同一套防污染設計
  5. DSR（falsifier.deflated_sharpe）：扣掉「試了 N 組才挑到一個好看的」
     幸運上限；未達 0.95 一律標示

誠實邊界：成交=次日開盤價無滑價；regime 用 SPY vs MA50 近似（歷史氣象台
五因子不可得）；Alpha 疊加層（內部人/選擇權/財報 veto）歷史不可重建、不含；
校準權重用現值（輕微前視，已知偏樂觀）。輸出永遠掛「非投資建議」。
"""

from __future__ import annotations

import math
from itertools import product

import numpy as np
import pandas as pd

COST_SIDE = 0.0005            # 單邊成本（來回 0.1%）
START_EQUITY = 100_000.0
MIN_TRADES = 8                # 訓練段最少出場筆數（低於此不參與排序）
VAL_MIN_TRADES = 5            # 驗證段挑選門檻
HOLDOUT_MIN_TRADES = 4        # holdout 把關門檻
HOLDOUT_MARGIN = 0.01         # best 須勝過 baseline holdout 報酬 +1 個百分點
PERIOD_DAYS = {"3m": 63, "6m": 126, "1y": 252, "2y": 504}
FETCH_PERIOD = {"3m": "2y", "6m": "2y", "1y": "2y", "2y": "3y"}   # 含 ~1 年暖機

# 網格：進場門檻 × 停損倍數 × 追蹤回落 × 分批 R × 死錢天數 = 108 組
GRID = {
    "buy_threshold":   (0.4, 0.5, 0.6),
    "stop_mult":       (1.0, 1.5),
    "trail_pct":       (0.05, 0.08, 0.12),
    "scale_out_r":     (1.5, 2.5),
    "dead_money_days": (20, 30, 45),
}
PARAM_LABELS = {          # 顯示用（Telegram Markdown 不能有底線）
    "buy_threshold": "進場門檻", "stop_mult": "停損倍數", "trail_pct": "追蹤回落",
    "scale_out_r": "分批R", "dead_money_days": "死錢天數", "trail_tight_pct": "收緊追蹤",
    "exit_threshold": "轉弱門檻", "max_positions": "最大檔數", "risk_pct": "單筆風險",
}


# ── 1. 資料 ───────────────────────────────────────────────────────────────

def fetch_history(tickers: list[str], period: str = "2y") -> dict[str, pd.DataFrame]:
    """批次抓日線 OHLCV（含 SPY 作 regime/基準）。回 {sym: DataFrame}。"""
    import yfinance as yf
    syms = sorted(set(t.upper() for t in tickers) | {"SPY"})
    out: dict[str, pd.DataFrame] = {}
    try:
        raw = yf.download(syms, period=period, auto_adjust=True, progress=False,
                          group_by="column", threads=True)
    except Exception as e:
        print(f"engine_backtest: 抓價失敗 {e}")
        return out
    if raw is None or raw.empty:
        return out
    multi = isinstance(raw.columns, pd.MultiIndex)
    for s in syms:
        try:
            if multi:
                df = pd.DataFrame({f: raw[(f, s)] for f in ("Open", "High", "Low", "Close", "Volume")
                                   if (f, s) in raw.columns})
            else:
                df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
            df = df.dropna(subset=["Close"])
            if len(df) >= 120:
                df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
                out[s] = df
        except Exception:
            continue
    return out


def regime_series(spy_close: pd.Series) -> pd.Series:
    """scan_signals.market_regime 的 MA50 退路規則，逐日向量化（只用當日以前資料）。"""
    ma50 = spy_close.rolling(50).mean()
    ret_1m = spy_close / spy_close.shift(22) - 1
    reg = pd.Series("neutral", index=spy_close.index, dtype=object)
    reg[(spy_close > ma50) & (ret_1m > 0)] = "risk_on"
    reg[(spy_close < ma50) & (ret_1m < -0.03)] = "risk_off"
    reg[ma50.isna()] = None
    return reg


def precompute(data: dict[str, pd.DataFrame], days: int = 252,
               thresholds: dict | None = None, calibration: dict | None = None) -> dict:
    """
    逐檔逐日算評分（**只用該日以前含當日的 K 棒**）與每股風險（ATR×atr_mult）。
    回 {"dates": [...], "by_date": {date: {sym: {score, close, open, rps}}},
        "regime": {date: str|None}, "spy": Series, "n_syms": int}
    """
    import indicators as ind
    th = thresholds or {}
    atr_mult = float(th.get("atr_mult", 1.5))
    mtf = bool(th.get("mtf_enabled", True))
    by_date: dict[str, dict] = {}
    spy = data.get("SPY")
    for sym, df in data.items():
        n = len(df)
        start = max(60, n - int(days))
        close, high, low = df["Close"], df.get("High"), df.get("Low")
        vol = df.get("Volume")
        edge_w = (calibration or {}).get(sym)
        for i in range(start, n):
            c = close.iloc[:i + 1]
            h = high.iloc[:i + 1] if high is not None else None
            lo = low.iloc[:i + 1] if low is not None else None
            v = vol.iloc[:i + 1] if vol is not None else None
            try:
                sc = float(ind._composite_score(c, h, lo, v, edge_weights=edge_w, mtf=mtf)["score"])
            except Exception:
                continue
            atr = ind._atr_value(c, h, lo)
            d = str(df.index[i].date())
            by_date.setdefault(d, {})[sym] = {
                "score": sc, "close": float(c.iloc[-1]),
                "open": float(df["Open"].iloc[i]) if "Open" in df else float(c.iloc[-1]),
                "rps": (atr * atr_mult) if atr > 0 else None,
            }
    dates = sorted(by_date)
    reg: dict[str, str | None] = {}
    if spy is not None and len(spy) >= 50:
        rs = regime_series(spy["Close"])
        for d in dates:
            try:
                v = rs.get(pd.Timestamp(d))
                reg[d] = v if isinstance(v, str) else None
            except Exception:
                reg[d] = None
    return {"dates": dates, "by_date": by_date, "regime": reg,
            "spy": (spy["Close"] if spy is not None else None),
            "n_syms": len([s for s in data if s != "SPY"])}


# ── 2. 重放 ───────────────────────────────────────────────────────────────

def _fill(book: dict, orders: list[dict], day: dict, date: str,
          lots: dict, trades: list[dict]) -> list[dict]:
    """前一日決策在今日開盤成交（含單邊成本）。就地更新 book/lots，回實際成交單。"""
    import shadow_book as sb
    filled = []
    for o in orders:
        sym = o["symbol"]
        px0 = (day.get(sym) or {}).get("open")
        if not px0 or px0 <= 0:
            continue                                   # 今日無報價 → 放棄，引擎會再決策
        px = px0 * (1 + COST_SIDE) if o["side"] == "buy" else px0 * (1 - COST_SIDE)
        before_qty = float((book.get("positions") or {}).get(sym, {}).get("qty") or 0)
        before_cash = book["cash"]
        sb.apply_orders(book, [o], {sym: px})
        after_qty = float((book.get("positions") or {}).get(sym, {}).get("qty") or 0)
        dq = after_qty - before_qty
        if abs(dq) < 1e-9:
            continue
        rec = {**o, "qty": abs(dq), "price": px, "date": date}
        if dq > 0:                                     # 買：建/加 lot（均價）
            L = lots.get(sym)
            if L:
                tot = L["qty"] + dq
                L["entry"] = (L["entry"] * L["qty"] + px * dq) / tot
                L["qty"] = tot
            else:
                lots[sym] = {"qty": dq, "entry": px, "opened": date}
        else:                                          # 賣：實現損益
            L = lots.get(sym)
            if L:
                q = min(-dq, L["qty"])
                pnl = q * (px - L["entry"])
                trades.append({"symbol": sym, "date": date, "opened": L["opened"],
                               "qty": q, "entry": L["entry"], "exit": px, "pnl": pnl,
                               "ret": px / L["entry"] - 1 if L["entry"] else 0.0,
                               "hold_days": max(0, (pd.Timestamp(date) - pd.Timestamp(L["opened"])).days),
                               "mechanism": o.get("mechanism") or "exit"})
                L["qty"] -= q
                if L["qty"] < 1e-9:
                    del lots[sym]
        rec["cash_delta"] = book["cash"] - before_cash
        filled.append(rec)
    return filled


def replay(pre: dict, params: dict | None = None, dates: list[str] | None = None,
           equity0: float = START_EQUITY) -> dict:
    """
    用一組引擎參數在 dates（預設全部）上完整重放。
    t 日收盤決策 → t+1 日開盤成交；淨值以收盤 mark。
    回 {"equity": Series, "trades": [...], "metrics": {...}, "journal": [...]}
    """
    import shadow_book as sb
    import trade_engine as te
    cfg = dict(params or {})
    dates = list(dates if dates is not None else pre["dates"])
    book = {"cash": float(equity0), "positions": {}, "last_px": {}}
    engine = None
    lots: dict = {}
    trades: list[dict] = []
    journal: list[dict] = []
    pending: list[dict] = []
    eq_curve: list[tuple[str, float]] = []
    expo: list[float] = []
    for d in dates:
        day = pre["by_date"].get(d) or {}
        if pending:
            journal.extend(_fill(book, pending, day, d, lots, trades))
            pending = []
        closes = {s: v["close"] for s, v in day.items() if v.get("close")}
        for s in list(book["positions"]):
            if s in closes:
                book["last_px"][s] = closes[s]
        equity = sb.book_equity(book, closes)
        eq_curve.append((d, equity))
        expo.append(1 - book["cash"] / equity if equity > 0 else 0.0)
        scored = [{"ticker": s, "score": v["score"], "price": v["close"],
                   "risk_per_share": v.get("rps")} for s, v in day.items() if s != "SPY"]
        pos_view = {}
        for s, p in book["positions"].items():
            px = closes.get(s) or book["last_px"].get(s) or p["entry"]
            pos_view[s] = {"qty": p["qty"], "avg_entry_price": p["entry"],
                           "market_value": p["qty"] * px}
        try:
            orders, engine, _notes = te.decide(scored, pos_view, equity, book["cash"],
                                               engine, pre["regime"].get(d), cfg, d)
        except Exception as e:                        # 單日炸掉不毀整段（記錄即可）
            orders = []
            journal.append({"date": d, "error": str(e)[:80]})
        pending = [o for o in orders if o.get("symbol") in day or o["side"] == "sell"]
    eq = pd.Series([v for _, v in eq_curve], index=pd.to_datetime([d for d, _ in eq_curve]))
    m = _metrics(eq, trades, equity0)
    m["exposure"] = float(np.mean(expo)) if expo else 0.0
    m["open_positions"] = len(book["positions"])
    return {"equity": eq, "trades": trades, "metrics": m, "journal": journal,
            "book": book, "engine": engine}


def _metrics(eq: pd.Series, trades: list[dict], equity0: float) -> dict:
    if eq is None or len(eq) == 0:
        return {"n_days": 0, "total_ret": 0.0, "max_dd": 0.0, "sr_d": 0.0, "sharpe": 0.0,
                "n_trades": 0, "win_rate": None, "avg_ret": None, "by_mech": {},
                "skew": 0.0, "kurt": 3.0}
    r = eq.pct_change().dropna()
    sd = float(r.std(ddof=1)) if len(r) > 2 else 0.0
    sr_d = float(r.mean() / sd) if sd > 0 else 0.0
    dd = float((eq / eq.cummax() - 1).min()) if len(eq) else 0.0
    by: dict[str, dict] = {}
    for t in trades:
        b = by.setdefault(t["mechanism"], {"n": 0, "pnl": 0.0, "wins": 0})
        b["n"] += 1
        b["pnl"] += t["pnl"]
        b["wins"] += 1 if t["pnl"] > 0 else 0
    n = len(trades)
    return {
        "n_days": int(len(eq)),
        "total_ret": float(eq.iloc[-1] / equity0 - 1),
        "max_dd": abs(dd),
        "sr_d": sr_d,
        "sharpe": sr_d * math.sqrt(252),
        "n_trades": n,
        "win_rate": (sum(1 for t in trades if t["pnl"] > 0) / n) if n else None,
        "avg_ret": (sum(t["ret"] for t in trades) / n) if n else None,
        "by_mech": by,
        "skew": float(r.skew()) if len(r) > 3 else 0.0,
        "kurt": float(r.kurt() + 3.0) if len(r) > 3 else 3.0,
    }


def bench_return(pre: dict, dates: list[str]) -> float | None:
    """SPY 買進持有同期報酬（同一段 dates，首日收盤→末日收盤）。"""
    spy = pre.get("spy")
    if spy is None or not dates:
        return None
    try:
        a = spy.get(pd.Timestamp(dates[0]))
        b = spy.get(pd.Timestamp(dates[-1]))
        if a and b:
            return float(b / a - 1)
    except Exception:
        pass
    return None


# ── 3. 參數學習（walk-forward + DSR）─────────────────────────────────────

def split_dates(dates: list[str]) -> dict | None:
    """50/25/25 三段（各段最少 20 個交易日，否則 None）。"""
    n = len(dates)
    if n < 80:
        return None
    i1, i2 = int(n * 0.5), int(n * 0.75)
    return {"train": dates[:i1], "val": dates[i1:i2], "holdout": dates[i2:]}


def optimize(pre: dict, grid: dict | None = None, baseline: dict | None = None,
             progress=None) -> dict:
    """
    網格逐組在三段各自完整重放（每段獨立起跑，段間不漏資訊）。
    挑選：train Sharpe 排序（n≥MIN_TRADES）→ 第一個 val 合格者（n≥VAL_MIN_TRADES
    且 val 報酬>0 且 Sharpe>0）= best → holdout **只看一次**把關：best 與 baseline
    皆 n≥HOLDOUT_MIN_TRADES 時須勝 +HOLDOUT_MARGIN，否則 best holdout 須為正。
    DSR：best 的 holdout 日 Sharpe 對「N 組嘗試」的幸運上限（未達 0.95 標示）。
    回 {"results","baseline","best","recommend","split","dsr","n_trials"}。
    """
    grid = grid or GRID
    keys = list(grid)
    sp = split_dates(pre["dates"])
    out = {"results": [], "baseline": None, "best": None, "recommend": None,
           "split": sp, "dsr": None, "n_trials": 0, "n_syms": pre.get("n_syms", 0)}
    if not sp:
        return out
    import trade_engine as te
    base_prm = {k: te.ENGINE_DEFAULTS[k] for k in keys}
    if baseline:
        base_prm.update({k: baseline[k] for k in keys if k in baseline})

    def _run(prm):
        segs = {seg: replay(pre, prm, sp[seg]) for seg in ("train", "val", "holdout")}
        return {"params": prm,
                **{seg: segs[seg]["metrics"] for seg in segs},
                "_holdout_eq": segs["holdout"]["equity"]}

    combos = [dict(zip(keys, c)) for c in product(*(grid[k] for k in keys))]
    if base_prm not in combos:
        combos.append(base_prm)
    results = []
    for i, prm in enumerate(combos):
        results.append(_run(prm))
        if progress and i % 10 == 0:
            progress(i + 1, len(combos))
    out["results"] = results
    out["n_trials"] = len(results)
    baseline_r = next((r for r in results if r["params"] == base_prm), None)
    out["baseline"] = baseline_r
    eligible = [r for r in results if r["train"]["n_trades"] >= MIN_TRADES]
    eligible.sort(key=lambda r: r["train"]["sharpe"], reverse=True)
    best = next((r for r in eligible
                 if r["val"]["n_trades"] >= VAL_MIN_TRADES
                 and r["val"]["total_ret"] > 0 and r["val"]["sharpe"] > 0), None)
    out["best"] = best
    if best and baseline_r and best["params"] != base_prm:
        bh, ah = best["holdout"], baseline_r["holdout"]
        if bh["n_trades"] >= HOLDOUT_MIN_TRADES:
            if ah["n_trades"] >= HOLDOUT_MIN_TRADES:
                ok = bh["total_ret"] >= ah["total_ret"] + HOLDOUT_MARGIN
            else:
                ok = bh["total_ret"] > 0
            if ok:
                out["recommend"] = best
    if best:
        try:
            import falsifier as fz
            trial_srs = [r["train"]["sr_d"] for r in results]
            out["dsr"] = fz.deflated_sharpe(best["holdout"]["sr_d"], best["holdout"]["n_days"],
                                            len(results), trial_srs,
                                            best["holdout"]["skew"], best["holdout"]["kurt"])
        except Exception as e:
            out["dsr"] = {"dsr": None, "sr_star": None, "note": f"DSR 不可用（{e}）"[:60]}
    for r in results:
        r.pop("_holdout_eq", None)
    return out


def apply_params(state: dict, params: dict, meta: dict | None = None) -> list[str]:
    """把推薦參數寫進 thresholds 的 eng_* 鍵（引擎 config 覆蓋路徑），並記錄
    state["eng_opt"] 供 /engtest clear 還原。回寫入的鍵名。"""
    th = state.setdefault("thresholds", {})
    prev = {}
    written = []
    for k, v in params.items():
        key = f"eng_{k}"
        prev[key] = th.get(key)
        th[key] = v
        written.append(key)
    state["eng_opt"] = {"params": dict(params), "prev": prev, **(meta or {})}
    return written


def clear_params(state: dict) -> list[str]:
    """還原 apply_params 寫入的鍵（有舊值還舊值、原本沒有就刪）。回還原的鍵名。"""
    rec = state.pop("eng_opt", None) or {}
    th = state.get("thresholds") or {}
    done = []
    for key, old in (rec.get("prev") or {}).items():
        if old is None:
            th.pop(key, None)
        else:
            th[key] = old
        done.append(key)
    return done


# ── 4. 文字輸出（Telegram legacy Markdown：單 *、無底線）────────────────────

def _pct(x) -> str:
    return "—" if x is None else f"{x:+.1%}"


def _params_text(p: dict) -> str:
    return "、".join(f"{PARAM_LABELS.get(k, k).replace('_', '·')} {v:g}" for k, v in p.items())


def run_text(rep: dict, params: dict | None, dates: list[str], bench: float | None,
             period_label: str = "") -> str:
    m = rep["metrics"]
    import trade_engine as te
    p = {k: (params or {}).get(k, te.ENGINE_DEFAULTS[k]) for k in GRID}
    lines = [f"🧪 *引擎歷史重放*（{period_label or '期間'} {dates[0]}→{dates[-1]}，"
             f"{m['n_days']} 個交易日）",
             f"參數：{_params_text(p)}",
             f"報酬 {_pct(m['total_ret'])}｜最大回撤 {m['max_dd']:.1%}｜"
             f"Sharpe {m['sharpe']:.2f}｜平均曝險 {m['exposure']:.0%}",
             f"出場 {m['n_trades']} 筆｜勝率 {(m['win_rate'] or 0):.0%}｜"
             f"均報酬 {_pct(m['avg_ret'])}"]
    if bench is not None:
        lines.append(f"SPY 買進持有同期 {_pct(bench)}（超額 {_pct(m['total_ret'] - bench)}）")
    if m["by_mech"]:
        from behavior_check import MECH_LABELS
        lines.append("*出場機制*：")
        for mech, b in sorted(m["by_mech"].items(), key=lambda kv: -kv[1]["n"]):
            lab = MECH_LABELS.get(mech, str(mech)).replace("_", "·")
            wr = b["wins"] / b["n"] if b["n"] else 0
            lines.append(f"・{lab}：{b['n']} 筆｜損益 {b['pnl']:+,.0f}｜勝率 {wr:.0%}")
    lines.append("成交=次日開盤、單邊 0.05% 成本、regime 用 SPY/MA50 近似、"
                 "不含 Alpha 疊加層；`/engtest opt` 跑參數學習。非投資建議")
    return "\n".join(lines)


def opt_text(opt: dict, top_n: int = 5) -> str:
    sp = opt.get("split")
    lines = [f"🔧 *引擎參數學習*（{opt.get('n_syms', 0)} 檔、{opt['n_trials']} 組參數、"
             "三段 walk-forward：50% 訓練排序/25% 驗證挑選/25% holdout 把關）"]
    if not sp or not opt["results"]:
        lines.append("資料不足（三段切分需 ≥80 個交易日），未產生結果。")
        return "\n".join(lines)
    lines.append(f"訓練 {sp['train'][0]}→{sp['train'][-1]}｜驗證 →{sp['val'][-1]}｜"
                 f"holdout →{sp['holdout'][-1]}")

    def _fmt(r):
        tr, va, ho = r["train"], r["val"], r["holdout"]
        return (f"{_params_text(r['params'])} → 訓練 {_pct(tr['total_ret'])}"
                f"(S{tr['sharpe']:.1f},{tr['n_trades']}筆)｜驗證 {_pct(va['total_ret'])}"
                f"({va['n_trades']}筆)｜holdout {_pct(ho['total_ret'])}"
                f"({ho['n_trades']}筆，回撤 {ho['max_dd']:.0%})")

    if opt["baseline"]:
        lines.append(f"基準（現行）：{_fmt(opt['baseline'])}")
    ranked = sorted([r for r in opt["results"] if r["train"]["n_trades"] >= MIN_TRADES],
                    key=lambda r: r["train"]["sharpe"], reverse=True)
    for i, r in enumerate(ranked[:top_n], 1):
        lines.append(f"{i}. {_fmt(r)}")
    d = opt.get("dsr") or {}
    if d.get("dsr") is not None:
        verdict = "通過" if d["dsr"] > 0.95 else "未達 0.95"
        lines.append(f"DSR {d['dsr']:.2f}（{verdict}；已扣 {opt['n_trials']} 組嘗試的幸運上限，"
                     "腦中試過的不算在內 → 恆偏樂觀）")
    elif d.get("note"):
        lines.append(f"DSR：{d['note']}")
    if opt["recommend"]:
        lines.append(f"\n✅ 推薦：{_params_text(opt['recommend']['params'])}"
                     "（holdout 段明確勝過現行——把關與挑選分離）")
    elif opt["best"]:
        lines.append("\n➖ 驗證段最佳組合未通過 holdout 把關——維持現行（不為調而調）")
    else:
        lines.append("\n➖ 無組合同時滿足訓練樣本與驗證正期望——維持現行")
    lines.append("\n⚠️ 歷史尋優極易過擬合；可信的是 holdout 欄與 DSR。"
                 "成交=次日開盤含成本、不含 Alpha 疊加。非投資建議")
    return "\n".join(lines)


# ── 5. 網路進入點（Bot / 網頁共用）──────────────────────────────────────────

def run(tickers: list[str], period: str = "1y", params: dict | None = None,
        thresholds: dict | None = None, calibration: dict | None = None) -> dict:
    """單組參數重放。回 {"rep","pre","dates","bench","text"}；資料不足 rep=None。"""
    period = period if period in PERIOD_DAYS else "1y"
    data = fetch_history(tickers, FETCH_PERIOD[period])
    if len([s for s in data if s != "SPY"]) == 0:
        return {"rep": None, "text": "❌ 行情抓取失敗或資料不足，稍後再試"}
    pre = precompute(data, PERIOD_DAYS[period], thresholds, calibration)
    if len(pre["dates"]) < 40:
        return {"rep": None, "pre": pre, "text": "❌ 可重放的交易日不足 40 天"}
    rep = replay(pre, params)
    bench = bench_return(pre, pre["dates"])
    return {"rep": rep, "pre": pre, "dates": pre["dates"], "bench": bench,
            "text": run_text(rep, params, pre["dates"], bench, period)}


def run_optimize(tickers: list[str], period: str = "1y", baseline: dict | None = None,
                 thresholds: dict | None = None, calibration: dict | None = None,
                 grid: dict | None = None) -> dict:
    """參數學習進入點。回 optimize() 結果 + "text"。"""
    period = period if period in PERIOD_DAYS else "1y"
    data = fetch_history(tickers, FETCH_PERIOD[period])
    if len([s for s in data if s != "SPY"]) == 0:
        return {"results": [], "recommend": None, "text": "❌ 行情抓取失敗或資料不足，稍後再試"}
    pre = precompute(data, PERIOD_DAYS[period], thresholds, calibration)
    opt = optimize(pre, grid, baseline)
    opt["text"] = opt_text(opt)
    return opt


# ── 6. 自我測試（合成 K 線；離線）─────────────────────────────────────────

def _synthetic(n: int = 420, seed: int = 7, drift: float = 0.0006, vol: float = 0.018,
               start: str = "2024-06-03") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    r = rng.normal(drift, vol, n)
    close = 100 * np.cumprod(1 + r)
    op = close * (1 + rng.normal(0, 0.003, n))
    hi = np.maximum(close, op) * (1 + np.abs(rng.normal(0, 0.006, n)))
    lo = np.minimum(close, op) * (1 - np.abs(rng.normal(0, 0.006, n)))
    v = rng.integers(1_000_000, 5_000_000, n).astype(float)
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame({"Open": op, "High": hi, "Low": lo, "Close": close, "Volume": v}, index=idx)


if __name__ == "__main__":
    import time
    data = {"AAA": _synthetic(seed=1, drift=0.0012), "BBB": _synthetic(seed=2, drift=0.0004),
            "CCC": _synthetic(seed=3, drift=-0.0008), "DDD": _synthetic(seed=4, drift=0.0009, vol=0.03),
            "SPY": _synthetic(seed=9, drift=0.0004, vol=0.01)}
    t0 = time.time()
    pre = precompute(data, days=200, thresholds={"mtf_enabled": True})
    dt = time.time() - t0
    assert len(pre["dates"]) == 200, len(pre["dates"])
    assert all(len(pre["by_date"][d]) == 5 for d in pre["dates"])
    assert set(pre["regime"].values()) <= {"risk_on", "risk_off", "neutral", None}
    print(f"✅ 1 precompute（5 檔 × 200 日 {dt:.1f}s，每日評分 {dt / 1000 * 1000:.1f}ms）")

    # 2) 無前視：竄改最後 40 根 K 棒，之前日期的評分必須逐位相同
    data2 = {k: v.copy() for k, v in data.items()}
    for k in data2:
        data2[k].iloc[-40:, :4] *= 1.5
    pre2 = precompute(data2, days=200, thresholds={"mtf_enabled": True})
    cut = pre["dates"][-41]
    for d in pre["dates"]:
        if d <= cut:
            for s in pre["by_date"][d]:
                assert pre["by_date"][d][s]["score"] == pre2["by_date"][d][s]["score"], (d, s)
    print("✅ 2 無前視（未來 K 棒竄改不影響過去評分）")

    # 3) 成交成本：買後立刻賣、價格不變 → 現金少 2×COST_SIDE×名目
    bk = {"cash": 10_000.0, "positions": {}, "last_px": {}}
    lots, tr = {}, []
    day = {"AAA": {"open": 100.0}}
    _fill(bk, [{"symbol": "AAA", "side": "buy", "qty": 10, "mechanism": "entry", "reason": ""}],
          day, "2025-01-02", lots, tr)
    _fill(bk, [{"symbol": "AAA", "side": "sell", "qty": 10, "mechanism": "stop_loss", "reason": ""}],
          day, "2025-01-03", lots, tr)
    assert abs((10_000.0 - bk["cash"]) - 2 * COST_SIDE * 1000.0) < 1e-6, bk["cash"]
    assert tr and tr[0]["mechanism"] == "stop_loss" and tr[0]["pnl"] < 0 and not lots
    print("✅ 3 成交成本與 lot 實現損益")

    # 4) 重放：基準參數有交易、淨值有限、曝險介於 0-1；決策 t 日→成交 t+1 日
    rep = replay(pre, {"buy_threshold": 0.3})
    m = rep["metrics"]
    assert m["n_days"] == 200 and math.isfinite(m["total_ret"]) and 0 <= m["exposure"] <= 1
    assert rep["journal"], "合成多頭資料下應有成交"
    first = next(j for j in rep["journal"] if j.get("symbol"))
    assert first["date"] > pre["dates"][0], "首筆成交不可在第一天（t+1 才成交）"
    assert all(t["hold_days"] >= 0 for t in rep["trades"])
    print(f"✅ 4 重放（{m['n_trades']} 筆出場、報酬 {m['total_ret']:+.1%}、回撤 {m['max_dd']:.1%}、"
          f"機制 {sorted(m['by_mech'])}）")

    # 5) 參數確實影響結果（極緊追蹤 vs 極鬆）
    tight = replay(pre, {"buy_threshold": 0.3, "trail_pct": 0.01})["metrics"]
    loose = replay(pre, {"buy_threshold": 0.3, "trail_pct": 0.3})["metrics"]
    assert tight["by_mech"].get("trailing_stop", {}).get("n", 0) >= \
        loose["by_mech"].get("trailing_stop", {}).get("n", 0)
    print("✅ 5 參數敏感度（緊追蹤出場次數 ≥ 鬆追蹤）")

    # 6) 三段切分：不重疊、有序、涵蓋全部
    sp = split_dates(pre["dates"])
    assert sp and sp["train"][-1] < sp["val"][0] < sp["val"][-1] < sp["holdout"][0]
    assert sp["train"] + sp["val"] + sp["holdout"] == pre["dates"]
    assert split_dates(pre["dates"][:50]) is None
    print("✅ 6 walk-forward 三段切分")

    # 7) optimize 小網格：結構完整、baseline 在內、DSR 可算或優雅降級、
    #    recommend 若有必過 holdout 門檻
    small = {"buy_threshold": (0.3, 0.5), "trail_pct": (0.05, 0.12)}
    t1 = time.time()
    opt = optimize(pre, small)
    assert opt["n_trials"] >= 4 and opt["baseline"] is not None
    assert all(set(r) >= {"params", "train", "val", "holdout"} for r in opt["results"])
    assert not any("_holdout_eq" in r for r in opt["results"])
    if opt["recommend"]:
        bh, ah = opt["recommend"]["holdout"], opt["baseline"]["holdout"]
        assert bh["n_trades"] >= HOLDOUT_MIN_TRADES
        assert bh["total_ret"] > 0 or bh["total_ret"] >= ah["total_ret"] + HOLDOUT_MARGIN
    d = opt.get("dsr")
    assert d is None or "dsr" in d
    print(f"✅ 7 optimize（{opt['n_trials']} 組 {time.time() - t1:.1f}s，"
          f"best={'有' if opt['best'] else '無'}、recommend={'有' if opt['recommend'] else '無'}、"
          f"DSR={(d or {}).get('dsr')}）")

    # 8) apply/clear 往返：thresholds 還原到原狀
    st = {"thresholds": {"eng_trail_pct": 0.07}}
    keys = apply_params(st, {"trail_pct": 0.05, "buy_threshold": 0.6}, {"as_of": "2026-09-03"})
    assert st["thresholds"]["eng_trail_pct"] == 0.05 and st["thresholds"]["eng_buy_threshold"] == 0.6
    assert st["eng_opt"]["as_of"] == "2026-09-03" and set(keys) == {"eng_trail_pct", "eng_buy_threshold"}
    clear_params(st)
    assert st["thresholds"] == {"eng_trail_pct": 0.07} and "eng_opt" not in st
    print("✅ 8 apply/clear 往返")

    # 9) 文字輸出 Markdown 安全（單 * 成對、無底線）
    txt = run_text(rep, {"buy_threshold": 0.3}, pre["dates"], bench_return(pre, pre["dates"]), "1y")
    txt2 = opt_text(opt)
    for t in (txt, txt2):
        assert t.count("*") % 2 == 0 and "_" not in t.replace("`/engtest opt`", ""), t
    assert "引擎歷史重放" in txt and "引擎參數學習" in txt2
    print("✅ 9 文字輸出（Markdown 安全）")
    print("\nengine_backtest selftest OK ✅")
