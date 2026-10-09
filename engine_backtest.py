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
     幸運上限；未達 0.95 一律標示。PBO（falsifier.pbo_cscv，CSCV）：挑選流程本身
     有沒有資訊，≥ 50% 取消推薦
  6. /engtest pit：當時 S&P 500 成分隨機抽樣多組（無事後選股偏誤）；/engtest try：單組試算不寫入

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

# 出場網格（預設 /engtest opt）：進場門檻 × 停損倍數 × 追蹤回落 × 分批 R × 死錢天數 = 108 組
GRID = {
    "buy_threshold":   (0.4, 0.5, 0.6),
    "stop_mult":       (1.0, 1.5),
    "trail_pct":       (0.05, 0.08, 0.12),
    "scale_out_r":     (1.5, 2.5),
    "dead_money_days": (20, 30, 45),
}
# 進場品質網格（/engtest opt entry；2026-09 診斷：追高進場、加碼在頂、中性 regime 未縮量）= 32 組
GRID_ENTRY = {
    "buy_threshold":     (0.5, 0.6),
    "entry_max_ext_atr": (1.5, 2.0, 3.0, 0),   # 0 = 關閉追高濾網
    "pyramid_r":         (1.0, 2.0),           # 預設分批 1.5R 下 2.0 = 實質關閉加碼；若已 apply scale_out_r 2.5 則語意為「晚加碼」
    "neutral_risk_mult": (0.5, 1.0),
}
# 放寬出場網格（/engtest opt loose；#56：動能行情中 +1.5R 先賣一半、5–8% 追蹤是否系統性賣掉贏家）= 36 組
# trail_tighten_r 99 = 不再在 +2R 後改用 5% 收緊版——否則 trail_pct 只在 +1R~+2R 間生效、又常被保本地板蓋過，
# 各 trail_pct 結果幾乎相同（對抗驗證 H1）。不放 trail_activate_r：調高它會讓加碼部位失去保本保護（#57）
GRID_LOOSE = {
    "trail_pct":       (0.08, 0.12, 0.20),
    "trail_tighten_r": (2.0, 99.0),         # 99 = 不收緊
    "scale_out_r":     (1.5, 3.0, 99.0),    # 99 = 關閉分批鎖利
    "stop_mult":       (1.0, 1.5),
}
OFF_VALUE_KEYS = ("scale_out_r", "trail_tighten_r")    # 值 ≥ 50 代表「關閉」的參數
GRIDS = {"exit": GRID, "entry": GRID_ENTRY, "loose": GRID_LOOSE}
PARAM_LABELS = {          # 顯示用（Telegram Markdown 不能有底線）
    "val_enabled": "估值層",
    "buy_threshold": "進場門檻", "stop_mult": "停損倍數", "trail_pct": "追蹤回落",
    "scale_out_r": "分批R", "dead_money_days": "死錢天數", "trail_tight_pct": "收緊追蹤",
    "exit_threshold": "轉弱門檻", "max_positions": "最大檔數", "risk_pct": "單筆風險",
    "entry_max_ext_atr": "追高上限ATR", "pyramid_max_ext_atr": "加碼延伸上限", "pyramid_r": "加碼R",
    "pyramid_min_score": "加碼最低分", "neutral_risk_mult": "中性風險倍數", "event_blackout": "事件靜默",
    "trail_activate_r": "追蹤啟動R", "trail_tighten_r": "收緊門檻R", "legacy": "舊邏輯",
    # /engtest try 可指定任何引擎鍵 → 全部給中文標籤（避免底線進 Telegram Markdown）
    "max_position_pct": "單檔上限", "pyramid_max_adds": "加碼次數上限", "pyramid_frac": "加碼比例",
    "entry_max_ret5d": "急拉門檻5日", "neutral_pyramid": "中性可加碼", "pyramid_headroom": "加碼空間倍數",
    "dead_money_ret": "死錢報酬門檻", "stoploss_guard_n": "保險絲次數", "stoploss_guard_days": "保險絲窗口天",
    "account_cooldown_days": "全帳戶冷卻天", "symbol_cooldown_days": "個股冷卻天", "max_dd_halt": "回撤鎖門檻",
    "dd_reset_flat_days": "空手重置天數", "regime_filter": "大盤濾網",
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
            df = df[df["Close"] > 0]                   # 壞報價（0/負）：快慢兩條評分路徑處理不同、ret_5d 會無窮大
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


def _row(sc, c_i, open_i, atr, ma20, r5_raw, n_bars, atr_mult) -> dict:
    """單日一檔的重放輸入列（快慢兩條路徑共用，保證欄位與捨入一致）。"""
    ext = round((c_i - ma20) / atr, 2) if (n_bars >= 20 and ma20 is not None and atr > 0) else None
    r5 = round(float(r5_raw), 4) if (n_bars >= 6 and r5_raw is not None) else None
    return {"score": float(sc), "close": c_i, "open": open_i,
            "rps": (atr * atr_mult) if atr > 0 else None,
            "ext_atr": ext, "ret_5d": r5}         # ret_5d：追高濾網第二條件，缺它濾網會比正式嚴（#72）


def _sym_rows_slow(df: pd.DataFrame, start: int, atr_mult: float, mtf: bool, edge_w) -> list[tuple[str, dict]]:
    """逐日切片版（原始定義；自測拿來對照向量化版逐位相等，也是收盤含 NaN 時的退路）。"""
    import indicators as ind
    close, high, low, vol = df["Close"], df.get("High"), df.get("Low"), df.get("Volume")
    out = []
    for i in range(start, len(df)):
        c = close.iloc[:i + 1]
        h = high.iloc[:i + 1] if high is not None else None
        lo = low.iloc[:i + 1] if low is not None else None
        v = vol.iloc[:i + 1] if vol is not None else None
        try:
            sc = float(ind._composite_score(c, h, lo, v, edge_weights=edge_w, mtf=mtf)["score"])
        except Exception:
            continue
        atr = ind._atr_value(c, h, lo)
        try:
            ma20 = float(c.rolling(20).mean().iloc[-1]) if len(c) >= 20 else None
        except Exception:
            ma20 = None
        r5 = float(c.iloc[-1] / c.iloc[-6] - 1) if len(c) >= 6 else None
        c_i = float(c.iloc[-1])
        out.append((str(df.index[i].date()),
                    _row(sc, c_i, float(df["Open"].iloc[i]) if "Open" in df else c_i, atr, ma20, r5, len(c), atr_mult)))
    return out


def _sym_rows_fast(df: pd.DataFrame, start: int, atr_mult: float, mtf: bool, edge_w) -> list[tuple[str, dict]]:
    """向量化版：indicators.composite_series + 滾動 ATR/MA20/5 日報酬一次算完（~100×）。
    rolling/ewm 在全序列第 i 位與切片 [:i+1] 末位的運算序列相同 → 結果逐位相等（自測驗證）。
    收盤含 NaN 時 composite_series 語意不同 → 退回逐日版。"""
    import indicators as ind
    close, high, low, vol = df["Close"], df.get("High"), df.get("Low"), df.get("Volume")
    if close.isna().any():
        return _sym_rows_slow(df, start, atr_mult, mtf, edge_w)
    sc_s = ind.composite_series(close, high, low, vol, edge_weights=edge_w, mtf=mtf)
    hh, ll = (high, low) if (high is not None and low is not None) else (close, close)   # 同 _atr_value
    tr = pd.concat([hh - ll, (hh - close.shift()).abs(), (ll - close.shift()).abs()], axis=1).max(axis=1)
    atr_s = tr.rolling(14).mean()
    ma20_s = close.rolling(20).mean()
    r5_s = close / close.shift(5) - 1
    opens = df["Open"] if "Open" in df else close
    out = []
    for i in range(start, len(df)):
        sc = sc_s.iloc[i]
        if sc is None or not math.isfinite(float(sc)):
            continue
        a = atr_s.iloc[i]
        atr = float(a) if not pd.isna(a) else 0.0
        m = ma20_s.iloc[i]
        r5 = r5_s.iloc[i]
        c_i = float(close.iloc[i])
        out.append((str(df.index[i].date()),
                    _row(sc, c_i, float(opens.iloc[i]), atr, None if pd.isna(m) else float(m),
                         None if pd.isna(r5) else float(r5), i + 1, atr_mult)))
    return out


def precompute(data: dict[str, pd.DataFrame], days: int = 252,
               thresholds: dict | None = None, calibration: dict | None = None,
               fast: bool = True) -> dict:
    """
    逐檔逐日算評分（**只用該日以前含當日的 K 棒**）與每股風險（ATR×atr_mult）。
    fast=True 走向量化（與逐日切片逐位相等）；回
    {"dates": [...], "by_date": {date: {sym: {score, close, open, rps, ext_atr, ret_5d}}},
     "regime": {date: str|None}, "spy": Series, "n_syms": int}
    """
    th = thresholds or {}
    atr_mult = float(th.get("atr_mult", 1.5))
    mtf = bool(th.get("mtf_enabled", True))
    by_date: dict[str, dict] = {}
    spy = data.get("SPY")
    rows_fn = _sym_rows_fast if fast else _sym_rows_slow
    # 重放起點錨定 SPY 的倒數第 days 根（共同日曆）：若各檔用自己的 K 棒數倒推，歷史中斷的股票
    # （下市/改名，PIT 抽樣常見）會把整段重放起點拉早好幾個月（對抗驗證 Med）
    anchor = spy.index[max(0, len(spy) - int(days))] if spy is not None and len(spy) else None
    last_date: dict[str, str] = {}
    for sym, df in data.items():
        start = max(60, int(df.index.searchsorted(anchor)) if anchor is not None else len(df) - int(days))
        for d, row in rows_fn(df, start, atr_mult, mtf, (calibration or {}).get(sym)):
            by_date.setdefault(d, {})[sym] = row
            last_date[sym] = d
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
            "n_syms": len([s for s in data if s != "SPY"]), "last_date": last_date}


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


def val_ctx_from_hist(val_hist: dict, date: str, price_by_sym: dict, max_age_days: int | None = 45) -> dict:
    """
    估值層 PIT 上下文：對每檔取「日期 ≤ date 的最新 val_hist 列」（列日期就是可得知日），
    算 val_mult（0.5–1.25，MoS 線性）、val_no_add（市價 > bull）、val_early（MoS>30%）、val_tilt（±0.1）。
    列超過 max_age_days 視為過期不給（實盤與回測同規則，對抗驗證 C-2）；無列 → 不給欄位（引擎＝現狀）。
    """
    out = {}
    from datetime import datetime as _dt
    for sym, rows in (val_hist or {}).items():
        rs = [r for r in (rows or []) if str(r.get("d", "9999")) <= date and r.get("base")]
        if not rs:
            continue
        r = max(rs, key=lambda x: str(x.get("d", "")))
        if max_age_days is not None:
            try:
                if (_dt.strptime(date[:10], "%Y-%m-%d") - _dt.strptime(str(r["d"])[:10], "%Y-%m-%d")).days > max_age_days:
                    continue
            except Exception:
                continue
        px = price_by_sym.get(sym)
        try:
            px = float(px)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(px) or px <= 0:
            continue
        mos = float(r["base"]) / px - 1
        bull = r.get("bull")
        out[sym] = {"val_mult": min(max(1 + 0.5 * mos, 0.5), 1.25),
                    "val_no_add": bool(bull and px > float(bull)),
                    "val_early": mos > 0.30,                       # 提早加碼：MoS>30%（上修動能由 playbook 顯示、不進引擎）
                    "val_tilt": min(max(0.6 * min(max(mos / 0.5, -1), 1) * 0.1, -0.1), 0.1)}
    return out


def val_hist_coverage(val_hist: dict) -> str | None:
    """最早的估值列日期（A/B 是否有意義：須早於訓練段結束）。"""
    ds = [str(r.get("d")) for rows in (val_hist or {}).values() for r in (rows or []) if r.get("d") and r.get("base")]
    return min(ds) if ds else None


def _blackout_day(d: str, cfg: dict) -> bool:
    """FOMC 靜默日判定（macro.event_blackout；模組缺失回 False，不毀重放）。"""
    try:
        import macro as _mc
        return bool(_mc.event_blackout(d, int(cfg.get("event_blackout_days", 1)))[0])
    except Exception:
        return False


def replay(pre: dict, params: dict | None = None, dates: list[str] | None = None,
           equity0: float = START_EQUITY, val_hist: dict | None = None) -> dict:
    """
    用一組引擎參數在 dates（預設全部）上完整重放。
    t 日收盤決策 → t+1 日開盤成交；淨值以收盤 mark。
    回 {"equity": Series, "trades": [...], "metrics": {...}, "journal": [...]}
    """
    import shadow_book as sb
    import trade_engine as te
    cfg = dict(params or {})
    cfg.setdefault("val_enabled", False)
    dates = list(dates if dates is not None else pre["dates"])
    book = {"cash": float(equity0), "positions": {}, "last_px": {}}
    engine = None
    lots: dict = {}
    trades: list[dict] = []
    journal: list[dict] = []
    pending: list[dict] = []
    eq_curve: list[tuple[str, float]] = []
    expo: list[float] = []
    lock_days = {"dd": 0, "regime": 0, "halt": 0}
    # 沒進場的原因（診斷低曝險）：無達標訊號的日數、追高濾網擋下次數、中性 regime 縮量日數
    blocks = {"no_signal": 0, "ext": 0, "neutral": 0, "blackout": 0}
    _bt = float(cfg.get("buy_threshold", te.ENGINE_DEFAULTS["buy_threshold"]))
    last_date = pre.get("last_date") or {}
    all_dates = pre["dates"]
    pos_of = {d_: i for i, d_ in enumerate(all_dates)}
    for d in dates:
        day = pre["by_date"].get(d) or {}
        if pending:
            journal.extend(_fill(book, pending, day, d, lots, trades))
            pending = []
        # 資料已中斷超過 5 個交易日（下市/併購/改名）的持倉 → 以最後收盤價結清（含成本），釋出名額與現金；
        # 只缺最新一兩根（資料源偶發）不算，避免誤賣
        i_d = pos_of.get(d, 0)
        for s_ in list(book["positions"]):
            ld = last_date.get(s_)
            if ld and i_d >= 5 and ld < all_dates[i_d - 5] and book["last_px"].get(s_):
                q_ = float(book["positions"][s_]["qty"])
                journal.extend(_fill(book, [{"symbol": s_, "side": "sell", "qty": q_, "mechanism": "data_end",
                                             "reason": f"資料於 {ld} 中斷 → 以最後收盤結清"}],
                                     {s_: {"open": book["last_px"][s_]}}, d, lots, trades))
        closes = {s: v["close"] for s, v in day.items() if v.get("close")}
        for s in list(book["positions"]):
            if s in closes:
                book["last_px"][s] = closes[s]
        equity = sb.book_equity(book, closes)
        eq_curve.append((d, equity))
        expo.append(1 - book["cash"] / equity if equity > 0 else 0.0)
        scored = [{"ticker": s, "score": v["score"], "price": v["close"],
                   "risk_per_share": v.get("rps"), "ext_atr": v.get("ext_atr"), "ret_5d": v.get("ret_5d")}
                  for s, v in day.items() if s != "SPY"]
        if val_hist and cfg.get("val_enabled"):
            vc = val_ctx_from_hist(val_hist, d, closes)      # PIT：只用列日期 ≤ d 的估值
            for sc_ in scored:
                sc_.update(vc.get(sc_["ticker"], {}))
        pit = pre.get("pit")
        if pit:                                             # 成分遮罩（/engtest pit）：離開指數（或尚未加入）的日子不開新倉
            for sc_ in scored:
                ps = pit.get(sc_["ticker"])
                if ps is not None and not _is_member(ps, d):
                    sc_["no_entry"] = True
                    sc_["pit_out"] = True
        # 有無達標訊號要在事件靜默標 no_entry 之前判斷（只排除成分遮罩）——否則靜默日被誤算成「沒訊號」
        has_sig = any(float(r.get("score") or 0) >= _bt and not r.get("pit_out") for r in scored)
        if cfg.get("event_blackout", True) and _blackout_day(d, cfg):
            if not cfg.get("legacy"):
                blocks["blackout"] += 1
            for sc_ in scored:                              # 與正式 run_autotrade 同語意：FOMC 會期不開新倉/不加碼
                sc_["no_entry"] = True
        pos_view = {}
        for s, p in book["positions"].items():
            px = closes.get(s) or book["last_px"].get(s) or p["entry"]
            pos_view[s] = {"qty": p["qty"], "avg_entry_price": p["entry"],
                           "market_value": p["qty"] * px}
        try:
            if cfg.get("legacy"):
                # 舊邏輯（與正式 Shadow 同語意：評分 ≥ 門檻買、≤ 出場門檻賣；無停損/追蹤/分批；
                # 正式 Shadow 在事件靜默之前呼叫 → 不受靜默影響，這裡同樣移除 no_entry；參數用 legacy_cfg（at_* 鍵）
                import alpaca_trader as at
                lg_rows = [{k: v for k, v in r.items() if k != "no_entry"} for r in scored]
                for r in lg_rows:                           # 成分遮罩對舊邏輯同樣有效（只擋進場；decide_orders 認 no_entry）
                    if r.get("pit_out"):
                        r["no_entry"] = True
                lg_cfg = dict(cfg.get("legacy_cfg") or cfg)
                orders = [dict(o, mechanism=o.get("mechanism") or ("legacy_exit" if o["side"] == "sell" else "legacy_entry"))
                          for o in at.decide_orders(lg_rows, pos_view, equity, book["cash"], lg_cfg)]
            else:
                # regime_filter=False 對應正式的 /set regime_filter_enabled off（regime 傳 None）
                _rg = pre["regime"].get(d) if cfg.get("regime_filter", True) else None
                orders, engine, _notes = te.decide(scored, pos_view, equity, book["cash"],
                                                   engine, _rg, cfg, d)
                _code = ((engine or {}).get("last_exposure") or {}).get("code")
                if _code in lock_days:                        # 曝險鎖定天數（#71 可見性）
                    lock_days[_code] += 1
                if not has_sig:
                    blocks["no_signal"] += 1
                blocks["ext"] += sum(1 for n_ in _notes if n_.startswith("⏳") and "等回檔再進" in n_)
                blocks["neutral"] += 1 if any(n_.startswith("🟡") for n_ in _notes) else 0
        except Exception as e:                        # 單日炸掉不毀整段（記錄即可）
            orders = []
            journal.append({"date": d, "error": str(e)[:80]})
        pending = [o for o in orders if o.get("symbol") in day or o["side"] == "sell"]
    eq = pd.Series([v for _, v in eq_curve], index=pd.to_datetime([d for d, _ in eq_curve]))
    m = _metrics(eq, trades, equity0)
    m["exposure"] = float(np.mean(expo)) if expo else 0.0
    m["open_positions"] = len(book["positions"])
    m["lock_days"] = lock_days
    m["blocks"] = blocks
    # 持有天數按部位（symbol, 開倉日）計：分批出場的半倉與餘倉算同一筆、取最後出場日；
    # 期末未平倉以段末為設限值納入（抱最久的贏家常在這裡，排除會低估持有期——審查 Low）
    by_lot: dict[tuple, int] = {}
    for t in trades:
        k_ = (t["symbol"], t["opened"])
        by_lot[k_] = max(by_lot.get(k_, 0), int(t["hold_days"]))
    if dates:
        for sym, L in lots.items():
            k_ = (sym, L["opened"])
            by_lot[k_] = max(by_lot.get(k_, 0), max(0, (pd.Timestamp(dates[-1]) - pd.Timestamp(L["opened"])).days))
    m["hold_med"] = float(np.median(list(by_lot.values()))) if by_lot else None
    m["beta"] = _beta(eq, pre.get("spy"))
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


def ew_return(pre: dict, dates: list[str], detail: bool = False):
    """宇宙等權買進持有同期報酬（SPY 除外）：每檔各取段內**第一個與最後一個有收盤的日子**再平均——
    不用首日／末日交集（各檔暖機起點不同、偶發缺 K 棒時交集會只剩一兩檔，審查 Med）。
    與策略報酬對照：觀察清單是事後挑的，等權持有常常就贏過任何擇時（選擇偏誤的量尺）。
    detail=True 回 (報酬, 納入檔數, 宇宙檔數)。"""
    firsts: dict[str, float] = {}
    lasts: dict[str, float] = {}
    for d in dates or []:
        for sym, v in (pre["by_date"].get(d) or {}).items():
            c = v.get("close")
            if sym == "SPY" or not c or c <= 0:
                continue
            firsts.setdefault(sym, float(c))
            lasts[sym] = float(c)
    rets = [lasts[s_] / firsts[s_] - 1 for s_ in firsts]
    val = float(np.mean(rets)) if rets else None
    return (val, len(rets), int(pre.get("n_syms") or len(rets))) if detail else val


def _beta(eq: pd.Series, spy: pd.Series | None) -> float | None:
    """策略日報酬對 SPY 的 beta（同日對齊；樣本 < 20 回 None）。"""
    if spy is None or eq is None or len(eq) < 21:
        return None
    try:
        r = eq.pct_change()
        b = spy.reindex(eq.index).pct_change()
        df = pd.concat([r, b], axis=1).dropna()
        if len(df) < 20 or float(df.iloc[:, 1].var()) <= 0:
            return None
        return float(df.iloc[:, 0].cov(df.iloc[:, 1]) / df.iloc[:, 1].var())
    except Exception:
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
             progress=None, val_hist: dict | None = None, legacy_cfg: dict | None = None) -> dict:
    """
    網格逐組在三段各自完整重放（每段獨立起跑，段間不漏資訊）。
    挑選：train Sharpe 排序（n≥MIN_TRADES）→ 第一個 val 合格者（n≥VAL_MIN_TRADES
    且 val 報酬>0 且 Sharpe>0）= best → holdout **只看一次**把關：best 與 baseline
    皆 n≥HOLDOUT_MIN_TRADES 時須勝 +HOLDOUT_MARGIN，否則 best holdout 須為正。
    DSR：best 的 holdout 日 Sharpe 對「N 組嘗試」的幸運上限（未達 0.95 標示）。
    PBO（falsifier.pbo_cscv）：可選組合在訓練＋驗證段連續重放的日報酬做 CSCV；≥ 50% 取消推薦（pbo_blocked）。
    回 {"results","baseline","best","recommend","split","dsr","pbo","pbo_blocked"?,"n_trials","bench","legacy"}。
    """
    grid = grid or GRID
    keys = list(grid)
    sp = split_dates(pre["dates"])
    out = {"results": [], "baseline": None, "best": None, "recommend": None,
           "split": sp, "dsr": None, "n_trials": 0, "n_syms": pre.get("n_syms", 0)}
    if not sp:
        return out
    import trade_engine as te
    base_prm = {k: te.ENGINE_DEFAULTS.get(k, False) for k in keys}     # val_enabled 不在引擎預設 → False
    if baseline:
        base_prm.update({k: baseline[k] for k in keys if k in baseline})
    # 非網格的現行覆蓋（max_positions/risk_pct/…）固定帶入每組——否則「基準（現行）」
    # 不是現行、推薦參數沒在使用者實際設定下測過（對抗驗證 Med-1）
    fixed = {k: v for k, v in (baseline or {}).items() if k not in keys}
    out["fixed"] = fixed

    tv_dates = sp["train"] + sp["val"]

    def _run(prm, with_tv=True):
        segs = {seg: replay(pre, {**fixed, **prm}, sp[seg], val_hist=val_hist) for seg in ("train", "val", "holdout")}
        r_ = {"params": prm,
              **{seg: segs[seg]["metrics"] for seg in segs},
              "_holdout_eq": segs["holdout"]["equity"]}
        if with_tv:                                   # PBO 用：訓練＋驗證段連續重放的日報酬（holdout 不碰）
            eq_tv = replay(pre, {**fixed, **prm}, tv_dates, val_hist=val_hist)["equity"]
            r_["_tv_ret"] = eq_tv.pct_change().dropna()
        return r_

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
    # 分段對照（顯示用、不參與挑選）：SPY 與宇宙等權買進持有——分辨「贏在擇時」還是「贏在曝險／選股偏誤」
    out["bench"] = {}
    for seg in ("train", "val", "holdout"):
        _ew, _n, _N = ew_return(pre, sp[seg], detail=True)
        out["bench"][seg] = {"spy": bench_return(pre, sp[seg]), "ew": _ew, "ew_n": _n, "ew_N": _N}
    try:                                              # 舊邏輯基準（不參與挑選、不計入嘗試數；#56）
        out["legacy"] = _run({"legacy": True, **({"legacy_cfg": dict(legacy_cfg)} if legacy_cfg else {})}, with_tv=False)
        out["legacy"].pop("_holdout_eq", None)
    except Exception as e:
        out["legacy"] = None
        out["legacy_error"] = type(e).__name__
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
    # PBO（CSCV）：挑選流程本身有沒有資訊——沒有 best 也算；≥ 50% 時取消推薦（排名不比亂猜好）
    try:
        import falsifier as fz
        # 只放挑選流程「選得到」的組（訓練段筆數達標）；完全相同的序列（參數沒作用）只留一份
        elig = [r for r in results if r["train"]["n_trades"] >= MIN_TRADES]
        mat = pd.concat([r["_tv_ret"].rename(i) for i, r in enumerate(elig)], axis=1).dropna() if elig else pd.DataFrame()
        M = np.unique(mat.values, axis=1) if mat.shape[1] else mat.values
        out["pbo"] = fz.pbo_cscv(M) if M.ndim == 2 and M.shape[1] >= 2 else \
            {"pbo": None, "note": f"可區分的候選組合不足（{M.shape[1] if M.ndim == 2 else 0} 組）"}
    except Exception as e:
        out["pbo"] = {"pbo": None, "note": f"PBO 不可用（{type(e).__name__}）"}
    _apply_pbo_gate(out)
    if best:
        try:
            import falsifier as fz
            # 幸運上限的跨組變異須與 sr 同樣本長度：用各組 holdout sr_d（只借變異、
            # 不做挑選；train 段長 2 倍會讓 sr_star 低估→DSR 偏樂觀，對抗驗證 Med-2）
            trial_srs = [r["holdout"]["sr_d"] for r in results]
            out["dsr"] = fz.deflated_sharpe(best["holdout"]["sr_d"], best["holdout"]["n_days"],
                                            len(results), trial_srs,
                                            best["holdout"]["skew"], best["holdout"]["kurt"])
        except Exception as e:
            out["dsr"] = {"dsr": None, "sr_star": None, "note": f"DSR 不可用（{e}）"[:60]}
    for r in results:
        r.pop("_holdout_eq", None)
        r.pop("_tv_ret", None)
    return out


def _apply_pbo_gate(out: dict) -> dict:
    """PBO ≥ 50%（網格排名不比亂猜好）→ 取消推薦、標 pbo_blocked。就地修改並回傳。"""
    v = (out.get("pbo") or {}).get("pbo")
    if out.get("recommend") and v is not None and v >= 0.5:
        out["recommend"] = None
        out["pbo_blocked"] = True
    return out


def apply_params(state: dict, params: dict, meta: dict | None = None) -> list[str]:
    """把推薦參數寫進 thresholds 的 eng_* 鍵（引擎 config 覆蓋路徑），並記錄
    state["eng_opt"] 供 /engtest clear 還原。回寫入的鍵名。"""
    th = state.setdefault("thresholds", {})
    old_rec = state.get("eng_opt") if isinstance(state.get("eng_opt"), dict) else {}
    prev = dict(old_rec.get("prev") or {})       # 連續 apply：保留最早的原值
    applied = dict(old_rec.get("applied") or {})
    written = []
    for k, v in params.items():
        key = k if k == "val_enabled" else f"eng_{k}"     # 估值層開關是頂層鍵，不是 eng_*（對抗驗證 C-1）
        if key not in prev:
            prev[key] = th.get(key)
        th[key] = v
        applied[key] = v
        written.append(key)
    state["eng_opt"] = {"params": dict(params), "prev": prev, "applied": applied,
                        **(meta or {})}
    return written


def clear_params(state: dict) -> list[str]:
    """還原 apply_params 寫入的鍵（有舊值還舊值、原本沒有就刪）。
    使用者事後手動 /set 過的鍵（現值 ≠ 套用值）保留不動（對抗驗證 Med-3）。
    回還原的鍵名（被保留的鍵以 "鍵(手動值保留)" 標示）。"""
    rec = state.pop("eng_opt", None) or {}
    th = state.get("thresholds") or {}
    done = []
    applied = rec.get("applied") or {}
    for key, old in (rec.get("prev") or {}).items():
        if key in applied and key in th and th[key] != applied[key]:
            done.append(f"{key}(手動值保留)")
            continue
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
    def _v(v):
        if isinstance(v, bool):
            return "開" if v else "關"
        return f"{v:g}"
    return "、".join(f"{PARAM_LABELS.get(k, k).replace('_', '·')} "
                    f"{'關' if (k in OFF_VALUE_KEYS and isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 50) else _v(v)}"
                    for k, v in p.items())


def _num(x, fmt: str = "{:.2f}") -> str:
    return "—" if x is None else fmt.format(x)


def _lock_text(m: dict) -> str:
    """鎖定天數（#71）：回撤鎖／大盤偏空／停損保險絲；全 0 回空字串。"""
    ld = m.get("lock_days") or {}
    parts = [f"{lab} {ld[k]} 交易日" for k, lab in (("dd", "回撤鎖"), ("regime", "偏空停新倉"), ("halt", "保險絲"))
             if ld.get(k)]                              # 重放逐交易日計數（引擎的空手重置門檻是日曆日）
    return "、".join(parts)


def run_text(rep: dict, params: dict | None, dates: list[str], bench: float | None,
             period_label: str = "", ew: float | None = None) -> str:
    m = rep["metrics"]
    import trade_engine as te
    p = {k: (params or {}).get(k, te.ENGINE_DEFAULTS[k]) for k in GRID}
    p.update({k: params[k] for k in list(GRID_ENTRY) + list(GRID_LOOSE) + ["trail_activate_r"]
              if k in (params or {}) and k not in GRID})   # 進場品質／放寬出場參數（有覆蓋才顯示）
    if (params or {}).get("val_enabled"):
        p["val_enabled"] = True
    lines = [f"🧪 *引擎歷史重放*（{period_label or '期間'} {dates[0]}→{dates[-1]}，"
             f"{m['n_days']} 個交易日）",
             f"參數：{_params_text(p)}",
             f"報酬 {_pct(m['total_ret'])}｜最大回撤 {m['max_dd']:.1%}｜"
             f"Sharpe {m['sharpe']:.2f}｜平均曝險 {m['exposure']:.0%}",
             f"出場 {m['n_trades']} 筆｜勝率 {(m['win_rate'] or 0):.0%}｜"
             f"均報酬 {_pct(m['avg_ret'])}"]
    if bench is not None:
        lines.append(f"SPY 買進持有同期 {_pct(bench)}（超額 {_pct(m['total_ret'] - bench)}）")
    if ew is not None:
        lines.append(f"清單等權買進持有 {_pct(ew)}（觀察清單是事後挑的——贏不過它代表擇時沒有加值）")
    lines.append(f"beta {_num(m.get('beta'))}｜持有天數中位 {_num(m.get('hold_med'), '{:.0f}')} 日曆日"
                 + (f"｜鎖定：{_lock_text(m)}" if _lock_text(m) else ""))
    bk = m.get("blocks") or {}
    if bk and not (params or {}).get("legacy"):
        lines.append(f"沒進場的原因：無達標訊號 {bk.get('no_signal', 0)}/{m['n_days']} 交易日｜"
                     f"追高濾網擋下 {bk.get('ext', 0)} 次｜中性縮量 {bk.get('neutral', 0)} 交易日｜"
                     f"事件靜默 {bk.get('blackout', 0)} 交易日")
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
        _lk = (ho.get("lock_days") or {}).get("dd")
        return (f"{_params_text(r['params'])} → 訓練 {_pct(tr['total_ret'])}"
                f"(S{tr['sharpe']:.1f},{tr['n_trades']}筆)｜驗證 {_pct(va['total_ret'])}"
                f"({va['n_trades']}筆)｜holdout {_pct(ho['total_ret'])}"
                f"({ho['n_trades']}筆，回撤 {ho['max_dd']:.0%}{f'，回撤鎖 {_lk} 交易日' if _lk else ''})")

    if opt["baseline"]:
        lines.append(f"基準（現行）：{_fmt(opt['baseline'])}")
    if opt.get("legacy"):
        lg = opt["legacy"]
        lines.append(f"舊邏輯（Shadow 同款）：訓練 {_pct(lg['train']['total_ret'])}｜驗證 {_pct(lg['val']['total_ret'])}｜"
                     f"holdout {_pct(lg['holdout']['total_ret'])}（{lg['holdout']['n_trades']}筆，回撤 {lg['holdout']['max_dd']:.0%}）"
                     "——只當參照、不參與挑選")
    bm = opt.get("bench") or {}
    if bm:
        lines.append("📊 *分段對照*（曝險/beta/持有天數：現行／舊邏輯）")
        for seg, lab in (("train", "訓練"), ("val", "驗證"), ("holdout", "holdout")):
            b_ = bm.get(seg) or {}
            ba = (opt.get("baseline") or {}).get(seg) or {}
            lg_ = (opt.get("legacy") or {}).get(seg) or {}
            _cov = (f"（{b_['ew_n']}/{b_['ew_N']} 檔）" if b_.get("ew_n") is not None
                    and b_.get("ew_N") and b_["ew_n"] < b_["ew_N"] else "")
            parts = [f"SPY {_pct(b_.get('spy'))}", f"等權持有 {_pct(b_.get('ew'))}{_cov}",
                     f"曝險 {_num(ba.get('exposure'), '{:.0%}')}／{_num(lg_.get('exposure'), '{:.0%}')}",
                     f"beta {_num(ba.get('beta'))}／{_num(lg_.get('beta'))}",
                     f"持有中位 {_num(ba.get('hold_med'), '{:.0f}')}／{_num(lg_.get('hold_med'), '{:.0f}')} 日曆日"]
            if _lock_text(ba):
                parts.append("現行鎖定：" + _lock_text(ba))
            lines.append(f"・{lab}：" + "｜".join(parts))
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
    pb = opt.get("pbo") or {}
    if pb.get("pbo") is not None:
        v = pb["pbo"]
        _vd = ("排名不比亂猜好——任何推薦都別套用" if v >= 0.5 else
               "排名有部分持續性" if v >= 0.2 else "排名在樣本外大致維持")
        lines.append(f"PBO {v:.0%}（過擬合機率：{pb['n_splits']} 次切分中，訓練＋驗證段樣本內最佳組在另一半"
                     f"排到中位數以下的比例；{_vd}）")
    elif pb.get("note"):
        lines.append(f"PBO：{pb['note']}")
    if opt.get("pbo_blocked"):
        lines.append("\n➖ holdout 雖過關，但 PBO ≥ 50%（排名沒有資訊）→ 取消推薦、維持現行")
    elif opt["recommend"]:
        lines.append(f"\n✅ 推薦：{_params_text(opt['recommend']['params'])}"
                     "（holdout 段明確勝過現行——把關與挑選分離）")
    elif opt["best"]:
        lines.append("\n➖ 驗證段最佳組合未通過 holdout 把關——維持現行（不為調而調）")
    else:
        lines.append("\n➖ 無組合同時滿足訓練樣本與驗證正期望——維持現行")
    lines.append("\n⚠️ 歷史尋優極易過擬合；可信的是 holdout 欄與 DSR。"
                 "成交=次日開盤含成本、不含 Alpha 疊加；regime 用 SPY/MA50 三態近似（無氣象台廣度否決）。非投資建議")
    return "\n".join(lines)


# ── 4b. 無事後偏誤股票池（/engtest pit）＋ 單組參數試算（/engtest try）─────────

PIT_K_RANGE = (5, 40)          # 每組抽樣檔數
PIT_SEEDS_MAX = 5              # 最多幾組隨機種子（每組一次完整重放 ×3 策略）
TRY_EXTRA_KEYS = ("regime_filter",)     # 非 ENGINE_DEFAULTS 但重放認得的鍵
# 成分歷史裡的「改名」（同一家公司換代碼，fja05680 記成舊代碼結束＋新代碼開始）：yfinance 的歷史掛在新代碼下，
# 用舊代碼抓不到 → 被誤算成下市。抓價改用新代碼、成分期間合併新舊兩段（2026-10 實查 sp500_ticker_start_end.csv）
PIT_RENAMES = {"FI": "FISV", "BK": "BNY", "FB": "META"}


def _is_member(periods_of: list, d: str) -> bool:
    d = str(d)[:10]
    return any(a <= d <= b for a, b in periods_of)


def pit_sample(periods: dict, as_of: str, k: int, seed: int) -> list[str]:
    """as_of 當天的 S&P 500 成分中，以 seed 決定性抽 k 檔（成分排序後抽 → 同輸入同結果）。
    只用「當時」成分 → 沒有「今天回頭挑贏家」的事後偏誤；之後下市/改名的抓不到價（殘留存活偏誤，呼叫端揭露）。"""
    import random
    pool = sorted(t for t, ps in periods.items() if t != "SPY" and _is_member(ps, as_of))
    if not pool:
        return []
    return sorted(random.Random(int(seed)).sample(pool, min(int(k), len(pool))))


def parse_try_params(tokens: list[str]) -> tuple[dict, list[str]]:
    """`/engtest try` 的 k=v 參數 → (候選參數, 錯誤訊息)。鍵須在 ENGINE_DEFAULTS（可加 eng_ 前綴）或
    TRY_EXTRA_KEYS；布林鍵吃 on/off；OFF_VALUE_KEYS 吃 off（=99，與網格同語意）；非有限數一律拒絕。"""
    import trade_engine as te
    out: dict = {}
    errs: list[str] = []
    for tok in tokens:
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        k, v = k.strip().lower(), v.strip().lower()
        if not k or not v:
            errs.append(f"`{tok}` 格式不對（參數=值 中間不要空格）")
            continue
        if k.startswith("eng_"):
            k = k[4:]
        if k == "seed":
            continue
        if k not in te.ENGINE_DEFAULTS and k not in TRY_EXTRA_KEYS:
            errs.append(f"未知參數 `{k}`")
            continue
        dv = te.ENGINE_DEFAULTS.get(k, True)
        if isinstance(dv, bool):
            if v in ("on", "true", "1", "yes", "開"):
                out[k] = True
            elif v in ("off", "false", "0", "no", "關"):
                out[k] = False
            else:
                errs.append(f"`{k}` 只接受 on/off")
            continue
        if v in ("off", "關") and k in OFF_VALUE_KEYS:
            out[k] = 99.0
            continue
        try:
            f = float(v)
        except ValueError:
            errs.append(f"`{k}` 的值 `{v}` 不是數字")
            continue
        if not math.isfinite(f):
            errs.append(f"`{k}` 的值必須是有限數")
            continue
        out[k] = int(round(f)) if isinstance(dv, int) else f
    return out, errs


def parse_engtest_args(args: list[str]) -> dict:
    """/engtest 子指令解析（Bot 與 engine_research 共用，避免兩份規則漂移）。
    回 {"kind": run|opt|pit|try|clear, "period", "grid", "apply", "k", "n_seeds", "seed0", "cand", "errs", "use_pit"}。
    參數=值 放錯位置（非 opt/clear）也當 try；數字只收十進位（全形可、上標拒）。"""
    args = [str(a) for a in (args or [])]
    sub = args[0].lower() if args else ""
    if sub not in ("opt", "clear", "pit", "try") and any("=" in a for a in args):
        args, sub = ["try"] + args, "try"
    kind = sub if sub in ("opt", "clear", "pit", "try") else "run"
    rest = args[1:] if kind != "run" else args
    low = [a.lower() for a in rest]
    default_p = "2y" if kind in ("pit", "try") else "1y"
    period = next((a for a in low if a in PERIOD_DAYS), default_p)
    if kind in ("opt", "pit", "try") and PERIOD_DAYS.get(period, 0) < 80:
        period = "6m"                                   # 三段切分／有意義的跨組比較需 ≥80 交易日
    cand, errs = parse_try_params(rest) if kind in ("pit", "try") else ({}, [])
    if kind == "try" and not cand and not errs:
        errs = ["沒有指定要試的參數"]
    nums = [int(a) for a in low if a.isdecimal()]
    seed0 = next((int(a.split("=", 1)[1]) for a in low
                  if a.startswith("seed=") and a.split("=", 1)[1].isdecimal()), 1)
    return {"kind": kind, "period": period,
            "grid": ("loose" if "loose" in low else "entry" if "entry" in low else "exit"),
            "apply": kind == "opt" and "apply" in low,
            "k": min(max(nums[0] if nums else 20, PIT_K_RANGE[0]), PIT_K_RANGE[1]),
            "n_seeds": min(max(nums[1] if len(nums) > 1 else 3, 1), PIT_SEEDS_MAX),
            "seed0": seed0, "cand": cand, "errs": errs, "use_pit": kind == "pit" or "pit" in low}


def _seg_metrics(pre: dict, prm: dict, sp: dict, val_hist: dict | None = None, err: list | None = None) -> dict:
    """三段各自獨立重放的 metrics；err 給 list 時累加各段 journal 的錯誤日數。"""
    out = {}
    for seg in ("train", "val", "holdout"):
        rp = replay(pre, prm, sp[seg], val_hist=val_hist)
        out[seg] = rp["metrics"]
        if err is not None:
            err.append(sum(1 for j in rp["journal"] if j.get("error")))
    return out


def try_compare(pre: dict, baseline: dict, candidate: dict, legacy_cfg: dict | None = None,
                val_hist: dict | None = None) -> dict:
    """現行 vs 候選（現行 + 覆蓋）vs 舊邏輯，三段各自獨立重放。只比較、不挑選、不寫入。"""
    sp = split_dates(pre["dates"])
    out = {"split": sp, "candidate": dict(candidate), "n_syms": pre.get("n_syms", 0)}
    if not sp:
        return out
    base = dict(baseline or {})
    out["base"] = _seg_metrics(pre, base, sp, val_hist)
    _err: list = []
    out["cand"] = _seg_metrics(pre, {**base, **candidate}, sp, val_hist, _err)
    out["errors"] = sum(_err)
    out["legacy"] = _seg_metrics(pre, {"legacy": True, "legacy_cfg": dict(legacy_cfg or {})}, sp)
    out["bench"] = {seg: {"spy": bench_return(pre, sp[seg]), "ew": ew_return(pre, sp[seg])}
                    for seg in ("train", "val", "holdout")}
    return out


def try_text(res: dict) -> str:
    sp = res.get("split")
    lines = [f"🧪 *單組參數試算*（{res.get('n_syms', 0)} 檔，三段各自獨立重放；不寫入設定）",
             f"候選：{_params_text(res.get('candidate') or {})}"]
    if not sp:
        lines.append("資料不足（三段切分需 ≥80 個交易日）。")
        return "\n".join(lines)
    lines.append(f"訓練 {sp['train'][0]}→{sp['train'][-1]}｜驗證 →{sp['val'][-1]}｜holdout →{sp['holdout'][-1]}")

    def _m(m):
        return f"{_pct(m['total_ret'])}（回撤 {m['max_dd']:.0%}、曝險 {m['exposure']:.0%}、{m['n_trades']}筆）"

    for seg, lab in (("train", "訓練"), ("val", "驗證"), ("holdout", "holdout")):
        b_ = (res.get("bench") or {}).get(seg) or {}
        lines.append(f"・{lab}：現行 {_m(res['base'][seg])}｜候選 {_m(res['cand'][seg])}｜"
                     f"舊邏輯 {_pct(res['legacy'][seg]['total_ret'])}｜等權持有 {_pct(b_.get('ew'))}｜SPY {_pct(b_.get('spy'))}")
    if res.get("errors"):
        lines.append(f"⚠️ 候選設定重放時有 {res['errors']} 日出錯（參數值可能超出合理範圍）——結果不可信")
    lines.append("（引擎會把超出安全範圍的值夾回，例如單筆風險 0.05%–5%、追蹤 0.5%–50%、停損倍數 0.25–5）")
    lines.append("⚠️ 這是 1 組額外嘗試、沒有經過挑選保護（腦中試過的組數不會被扣掉）——holdout 好看也不能直接套用；"
                 "觀察清單是事後挑的，請再用 `/engtest try … pit` 在隨機股票池驗一次。非投資建議")
    return "\n".join(lines)


_TCRIT = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57}   # 雙尾 5% t 臨界值（自由度 → 值）；組數少時 |t|≥2 誤報率 12–30%


def _stats(xs: list[float]) -> tuple[float | None, float | None, float | None, int]:
    """平均、標準差、t 值、樣本數（n<2 或零離散時標準差/t 為 None）。"""
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    if not xs:
        return None, None, None, 0
    m = float(np.mean(xs))
    if len(xs) < 2:
        return m, None, None, len(xs)
    sd = float(np.std(xs, ddof=1))
    if sd < 1e-12:
        return m, None, None, len(xs)
    return m, sd, m / (sd / math.sqrt(len(xs))), len(xs)


def _sig(xs: list[float]) -> tuple[str, float | None]:
    """跨組差異的顯著性判讀：('pos'|'neg'|'noise'|'few', 平均)。自由度校正的 5% 臨界值，n<3 不判讀。"""
    m, _, t, n = _stats(xs)
    if m is None or n < 3 or t is None:
        return "few", m
    crit = _TCRIT.get(n - 1, 2.0)
    return ("pos" if t >= crit else "neg" if t <= -crit else "noise"), m


def _same_sign(xs: list[float]) -> bool:
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return bool(xs) and (all(x > 0 for x in xs) or all(x < 0 for x in xs))


def pit_compare(data: dict, periods: dict, samples: list[list[str]], seeds: list[int], days: int,
                baseline: dict, legacy_cfg: dict | None = None, candidate: dict | None = None,
                thresholds: dict | None = None) -> list[dict]:
    """每組抽樣各自 precompute + 重放（現行／舊邏輯／候選），附等權持有與 SPY。純邏輯（data 由呼叫端給）。"""
    rows = []
    for seed, smp in zip(seeds, samples):
        sub = {s_: data[s_] for s_ in smp if s_ in data}
        r = {"seed": seed, "k": len(smp), "n": len(sub), "miss": len(smp) - len(sub)}
        if len(sub) < max(3, len(smp) // 2) or "SPY" not in data:
            r["skip"] = "可用檔數不足" if "SPY" in data else "SPY 抓不到"
            rows.append(r)
            continue
        sub["SPY"] = data["SPY"]
        pre = precompute(sub, days, thresholds, None)
        pre["pit"] = {s_: periods.get(s_, []) for s_ in sub if s_ != "SPY"}
        if len(pre["dates"]) < 40:
            r["skip"] = "可重放交易日不足"
            rows.append(r)
            continue
        base = dict(baseline or {})
        r["dates"] = (pre["dates"][0], pre["dates"][-1])
        reps = {"base": replay(pre, base),
                "legacy": replay(pre, {"legacy": True, "legacy_cfg": dict(legacy_cfg or {})})}
        if candidate:
            reps["cand"] = replay(pre, {**base, **candidate})
        for k_, rp in reps.items():
            r[k_] = rp["metrics"]
        r["errors"] = sum(1 for rp in reps.values() for j in rp["journal"] if j.get("error"))
        r["ew"] = ew_return(pre, pre["dates"])
        r["spy"] = bench_return(pre, pre["dates"])
        rows.append(r)
    return rows


def pit_text(rows: list[dict], meta: dict) -> str:
    cand = meta.get("candidate") or {}
    lines = [f"🎲 *無事後偏誤回測*（{meta['as_of']} 當時的 S&P 500 成分 {meta['n_members']} 檔中"
             f"隨機抽 {meta['k']} 檔 × {len(rows)} 組，{meta['period']}）"]
    if cand:
        lines.append(f"候選：{_params_text(cand)}")
    tot = sum(r["k"] for r in rows)
    miss = sum(r["miss"] for r in rows)
    spy_bad = any(r.get("skip") == "SPY 抓不到" for r in rows)
    if spy_bad:
        lines.append(f"⚠️ 基準 SPY 抓不到價（個股缺 {miss}/{tot} 檔）——行情抓取可能失敗，稍後再試")
    elif tot and miss > tot / 2:
        lines.append(f"⚠️ 抓不到價 {miss}/{tot} 檔——行情抓取可能失敗，結果不可靠，稍後再試")
    else:
        lines.append(f"成交＝次日開盤含成本；離開指數後不開新倉；資料中斷的持倉以最後收盤結清。抓不到價 {miss}/{tot} 檔"
                     "（多為之後下市/併購/改名——殘留存活偏誤，方向不確定：併購多溢價、破產為負）")
    eo = meta.get("eng_opt") or {}
    if eo:
        if str(eo.get("as_of", "")) >= str(meta["as_of"]):
            lines.append(f"⚠️ 現行含 /engtest opt 於 {eo.get('as_of', '?')} 以 {eo.get('period', '?')} 視窗套用的參數——"
                         "調參視窗與本回測期間重疊（時間上屬樣本內，現行 vs 舊邏輯偏向現行）")
        else:
            lines.append(f"ℹ️ 現行含 /engtest opt 於 {eo.get('as_of', '?')} 套用的參數（調參在本回測起點之前）")

    def _m(m):
        return f"{_pct(m['total_ret'])}（S{m['sharpe']:.1f}、曝險 {m['exposure']:.0%}、回撤 {m['max_dd']:.0%}）"

    ok = [r for r in rows if "base" in r]
    for r in rows:
        if "skip" in r:
            lines.append(f"・種子 {r['seed']}：略過（{r['skip']}）")
            continue
        parts = [f"現行 {_m(r['base'])}", f"舊邏輯 {_m(r['legacy'])}"]
        if "cand" in r:
            parts.append(f"候選 {_m(r['cand'])}")
        parts += [f"等權持有 {_pct(r['ew'])}", f"SPY {_pct(r['spy'])}"]
        err = f"｜⚠️ 重放錯誤 {r['errors']} 日" if r.get("errors") else ""
        lines.append(f"・種子 {r['seed']}（{r['n']} 檔，{r['dates'][0]}→{r['dates'][1]}）：" + "｜".join(parts) + err)
    if not ok:
        lines.append("\n❌ 沒有可用的組（行情抓取失敗？），稍後再試")
        return "\n".join(lines)
    lines.append("（S＝年化 Sharpe；曝險＝平均持股占淨值比例）")

    ew_ok = [r for r in ok if r["ew"] is not None]
    # 曝險調整後擇時值：報酬 − 平均曝險 × 等權持有（現金報酬 0 的近似）——只比原始報酬時，
    # 七成持股的策略在上漲段「必然」輸滿倉的等權持有，那不是擇時好壞（對抗驗證 Med）
    adj_b = [r["base"]["total_ret"] - r["base"]["exposure"] * r["ew"] for r in ew_ok]
    adj_l = [r["legacy"]["total_ret"] - r["legacy"]["exposure"] * r["ew"] for r in ew_ok]
    series = [("現行 − 等權持有", [r["base"]["total_ret"] - r["ew"] for r in ew_ok]),
              ("現行擇時值（曝險調整）", adj_b),
              ("舊邏輯擇時值（曝險調整）", adj_l),
              ("舊邏輯 − 現行", [r["legacy"]["total_ret"] - r["base"]["total_ret"] for r in ok])]
    if cand:
        series.append(("候選 − 現行", [r["cand"]["total_ret"] - r["base"]["total_ret"] for r in ok if "cand" in r]))
    lines.append(f"\n*跨組平均*（± 標準差，{len(ok)} 組）")
    for lab, xs in series:
        m, sd, t, _n = _stats(xs)
        if m is None:
            continue
        lines.append(f"{lab} {m:+.1%}" + (f" ± {sd:.1%}" if sd is not None else "") + (f"（t {t:+.1f}）" if t is not None else ""))
    m, sd, t, _n = _stats([r["legacy"]["sharpe"] - r["base"]["sharpe"] for r in ok])
    if m is not None:
        lines.append(f"Sharpe 差（舊邏輯 − 現行）{m:+.2f}" + (f" ± {sd:.2f}" if sd is not None else "")
                     + (f"（t {t:+.1f}）" if t is not None else ""))
    ex_b = float(np.mean([r["base"]["exposure"] for r in ok]))
    ex_l = float(np.mean([r["legacy"]["exposure"] for r in ok]))
    lines.append(f"平均曝險：現行 {ex_b:.0%}／舊邏輯 {ex_l:.0%}")

    # 判讀（規則式、保守）：自由度校正的 5% 臨界值；少於 3 組不判讀
    if len(ok) < 3:
        lines.append(f"判讀：只有 {len(ok)} 組，無法判讀（至少 3 組，建議 `/engtest pit 20 5`）")
    else:
        verdict = []
        def _vd(xs, pos, neg, noise, few):
            sg_, _mm = _sig(xs)
            tail = "（各組同向）" if sg_ in ("pos", "neg") and _same_sign(xs) else \
                   "（但有組別方向相反）" if sg_ in ("pos", "neg") else ""
            return {"pos": pos + tail, "neg": neg + tail, "noise": noise, "few": few}[sg_]

        verdict.append(_vd(adj_b, "現行擇時（曝險調整後）跨組平均顯著為正——有加值的跡象",
                           "現行擇時（曝險調整後）跨組平均顯著為負——擇時還沒證明有加值",
                           "現行擇時（曝險調整後）在雜訊範圍內", "現行擇時：無法判讀"))
        ex_note = (f"；曝險差 {ex_l - ex_b:+.0%}，曝險調整後 舊邏輯 − 現行 "
                   f"{(_stats([a - b for a, b in zip(adj_l, adj_b)])[0] or 0):+.1%}" if abs(ex_l - ex_b) >= 0.05 else "")
        verdict.append(_vd(series[3][1], "舊邏輯跨組平均顯著領先現行", "舊邏輯跨組平均顯著落後現行",
                           "舊邏輯 vs 現行在雜訊範圍內", "舊邏輯 vs 現行：無法判讀") + ex_note)
        if cand:
            verdict.append(_vd(series[4][1], "候選跨組平均顯著優於現行", "候選跨組平均顯著劣於現行",
                               "候選 vs 現行在雜訊範圍內", "候選：無法判讀"))
        lines.append("判讀：" + "；".join(verdict))
    lines.append("⚠️ t 值只反映「抽到哪些股票」的差異，不含「哪段行情」——各組共用同一段市場路徑，加組數也補不了；"
                 "看方向、不看精確數字。非投資建議")
    return "\n".join(lines)


def run_pit(period: str = "2y", k: int = 20, n_seeds: int = 3, seed0: int = 1,
            baseline: dict | None = None, legacy_cfg: dict | None = None, candidate: dict | None = None,
            thresholds: dict | None = None, periods: dict | None = None, fetch_fn=None,
            today: str | None = None, eng_opt: dict | None = None) -> dict:
    """/engtest pit 進入點：抓成分歷史 → 各組抽樣 → 一次批次抓價 → pit_compare → 文字。"""
    from datetime import date, timedelta
    period = period if period in PERIOD_DAYS else "2y"
    k = int(min(max(int(k), PIT_K_RANGE[0]), PIT_K_RANGE[1]))
    n_seeds = int(min(max(int(n_seeds), 1), PIT_SEEDS_MAX))
    if periods is None:
        import universe as un
        periods = un.fetch_sp500_periods()
    if not periods:
        return {"rows": [], "text": "❌ S&P 500 成分歷史抓取失敗（GitHub raw），稍後再試"}
    t0 = date.fromisoformat(str(today)[:10]) if today else date.today()
    as_of = (t0 - timedelta(days=int(PERIOD_DAYS[period] * 365 / 252) + 3)).isoformat()   # ≈ 重放首日
    seeds = [int(seed0) + i for i in range(n_seeds)]
    samples = [pit_sample(periods, as_of, k, sd) for sd in seeds]
    allsyms = sorted(set().union(*samples)) if samples else []
    if not allsyms:
        return {"rows": [], "text": f"❌ {as_of} 查無 S&P 500 成分（期間表異常）"}
    fetch_syms = sorted({PIT_RENAMES.get(s_, s_) for s_ in allsyms})
    raw = (fetch_fn or fetch_history)(fetch_syms, FETCH_PERIOD[period])
    data = {s_: raw[PIT_RENAMES.get(s_, s_)] for s_ in allsyms if PIT_RENAMES.get(s_, s_) in raw}
    if "SPY" in raw:
        data["SPY"] = raw["SPY"]
    per_eff = dict(periods)
    for old_t, new_t in PIT_RENAMES.items():
        if old_t in per_eff and new_t in periods:
            per_eff[old_t] = sorted(set(periods[old_t]) | set(periods[new_t]))
    rows = pit_compare(data, per_eff, samples, seeds, PERIOD_DAYS[period], baseline or {},
                       legacy_cfg, candidate, thresholds)
    meta = {"as_of": as_of, "n_members": sum(1 for ps in periods.values() if _is_member(ps, as_of)),
            "k": k, "period": period, "candidate": candidate,
            "eng_opt": eng_opt if isinstance(eng_opt, dict) else None}
    return {"rows": rows, "meta": meta, "samples": samples, "text": pit_text(rows, meta)}


def run_try(tickers: list[str], period: str = "2y", baseline: dict | None = None, candidate: dict | None = None,
            thresholds: dict | None = None, calibration: dict | None = None, val_hist: dict | None = None,
            legacy_cfg: dict | None = None, fetch_fn=None) -> dict:
    """/engtest try（觀察清單）進入點。"""
    period = period if period in PERIOD_DAYS else "2y"
    data = (fetch_fn or fetch_history)(tickers, FETCH_PERIOD[period])
    if len([s_ for s_ in data if s_ != "SPY"]) == 0:
        return {"text": "❌ 行情抓取失敗或資料不足，稍後再試"}
    pre = precompute(data, PERIOD_DAYS[period], thresholds, calibration)
    res = try_compare(pre, baseline or {}, candidate or {}, legacy_cfg, val_hist)
    res["text"] = try_text(res)
    return res


# ── 5. 網路進入點（Bot / 網頁共用）──────────────────────────────────────────

def run(tickers: list[str], period: str = "1y", params: dict | None = None,
        thresholds: dict | None = None, calibration: dict | None = None, val_hist: dict | None = None,
        fetch_fn=None) -> dict:
    """單組參數重放。回 {"rep","pre","dates","bench","text"}；資料不足 rep=None。"""
    period = period if period in PERIOD_DAYS else "1y"
    data = (fetch_fn or fetch_history)(tickers, FETCH_PERIOD[period])
    if len([s for s in data if s != "SPY"]) == 0:
        return {"rep": None, "text": "❌ 行情抓取失敗或資料不足，稍後再試"}
    pre = precompute(data, PERIOD_DAYS[period], thresholds, calibration)
    if len(pre["dates"]) < 40:
        return {"rep": None, "pre": pre, "text": "❌ 可重放的交易日不足 40 天"}
    rep = replay(pre, params, val_hist=val_hist)
    bench = bench_return(pre, pre["dates"])
    ew = ew_return(pre, pre["dates"])
    return {"rep": rep, "pre": pre, "dates": pre["dates"], "bench": bench, "ew": ew,
            "text": run_text(rep, params, pre["dates"], bench, period, ew=ew)}


def run_optimize(tickers: list[str], period: str = "1y", baseline: dict | None = None,
                 thresholds: dict | None = None, calibration: dict | None = None,
                 grid: dict | None = None, val_hist: dict | None = None, legacy_cfg: dict | None = None,
                 fetch_fn=None) -> dict:
    """參數學習進入點。回 optimize() 結果 + "text"。"""
    period = period if period in PERIOD_DAYS else "1y"
    data = (fetch_fn or fetch_history)(tickers, FETCH_PERIOD[period])
    if len([s for s in data if s != "SPY"]) == 0:
        return {"results": [], "recommend": None, "text": "❌ 行情抓取失敗或資料不足，稍後再試"}
    pre = precompute(data, PERIOD_DAYS[period], thresholds, calibration)
    opt = optimize(pre, grid, baseline, val_hist=val_hist, legacy_cfg=legacy_cfg)
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

    # 7b) 估值層 PIT 上下文：只用列日期 ≤ 當日的估值；val_enabled=False 完全等於現狀
    vh = {"AAA": [{"d": pre["dates"][100], "base": 999.0, "bear": 1.0, "bull": 9999.0}]}
    rep_off = replay(pre, {"buy_threshold": 0.3}, val_hist=vh)
    rep_on = replay(pre, {"buy_threshold": 0.3, "val_enabled": True}, val_hist=vh)
    assert rep_off["metrics"]["total_ret"] == rep["metrics"]["total_ret"]          # 預設關閉＝現狀
    early = val_ctx_from_hist(vh, pre["dates"][50], {"AAA": 100.0})
    late = val_ctx_from_hist(vh, pre["dates"][115], {"AAA": 100.0})          # 15 個交易日後（≤45 天）
    assert early == {} and late["AAA"]["val_mult"] == 1.25 and late["AAA"]["val_early"] and not late["AAA"]["val_no_add"]
    assert val_ctx_from_hist({"AAA": [{"d": "2020-01-01", "base": 50.0, "bull": 60.0}]}, "2026-01-01", {"AAA": 100.0}, max_age_days=None)["AAA"]["val_no_add"]
    assert val_ctx_from_hist({"AAA": [{"d": "2020-01-01", "base": 50.0, "bull": 60.0}]}, "2026-01-01", {"AAA": 100.0}) == {}   # 過期不給（C-2）
    assert val_ctx_from_hist({"AAA": [{"d": "2026-01-01", "base": 50.0}]}, "2026-01-10", {"AAA": float("nan")}) == {}
    assert val_hist_coverage(vh) == pre["dates"][100] and val_hist_coverage({}) is None
    st_v = {"thresholds": {}}
    apply_params(st_v, {"val_enabled": True, "trail_pct": 0.05})
    assert st_v["thresholds"]["val_enabled"] is True and st_v["thresholds"]["eng_trail_pct"] == 0.05     # C-1：頂層鍵
    clear_params(st_v); assert "val_enabled" not in st_v["thresholds"]
    assert math.isfinite(rep_on["metrics"]["total_ret"])
    print("✅ 7b 估值層 val_ctx（PIT 列日期、預設關閉＝現狀、乘數/加碼閘）")

    # 8) apply/clear 往返：thresholds 還原到原狀
    st = {"thresholds": {"eng_trail_pct": 0.07}}
    keys = apply_params(st, {"trail_pct": 0.05, "buy_threshold": 0.6}, {"as_of": "2026-09-03"})
    assert st["thresholds"]["eng_trail_pct"] == 0.05 and st["thresholds"]["eng_buy_threshold"] == 0.6
    assert st["eng_opt"]["as_of"] == "2026-09-03" and set(keys) == {"eng_trail_pct", "eng_buy_threshold"}
    clear_params(st)
    assert st["thresholds"] == {"eng_trail_pct": 0.07} and "eng_opt" not in st
    # 連續兩次 apply → clear 必須回到最初原值（不是第一次 apply 的值）
    st2 = {"thresholds": {"eng_trail_pct": 0.07}}
    apply_params(st2, {"trail_pct": 0.05})
    apply_params(st2, {"trail_pct": 0.12, "stop_mult": 1.5})
    clear_params(st2)
    assert st2["thresholds"] == {"eng_trail_pct": 0.07}, st2["thresholds"]
    # 使用者事後手動 /set 過的鍵 → clear 保留手動值
    st3 = {"thresholds": {}}
    apply_params(st3, {"trail_pct": 0.05, "stop_mult": 1.5})
    st3["thresholds"]["eng_trail_pct"] = 0.1          # 手動覆蓋
    done3 = clear_params(st3)
    assert st3["thresholds"] == {"eng_trail_pct": 0.1}, st3["thresholds"]
    assert any("手動值保留" in d for d in done3)
    print("✅ 8 apply/clear 往返（連續 apply / 手動值保留）")

    # 8b) opt 的非網格現行覆蓋固定帶入：max_positions=1 必須讓結果不同於不帶
    o_a = optimize(pre, {"buy_threshold": (0.3,)}, baseline={"max_positions": 1})
    o_b = optimize(pre, {"buy_threshold": (0.3,)})
    assert o_a["fixed"] == {"max_positions": 1}
    assert o_a["baseline"]["train"]["n_trades"] != o_b["baseline"]["train"]["n_trades"] or \
        o_a["baseline"]["train"]["total_ret"] != o_b["baseline"]["train"]["total_ret"]
    print("✅ 8b optimize 帶入非網格現行參數")

    # 9) 文字輸出 Markdown 安全（單 * 成對、無底線）
    txt = run_text(rep, {"buy_threshold": 0.3}, pre["dates"], bench_return(pre, pre["dates"]), "1y")
    txt2 = opt_text(opt)
    for t in (txt, txt2):
        assert t.count("*") % 2 == 0 and "_" not in t.replace("`/engtest opt`", ""), t
    assert "引擎歷史重放" in txt and "引擎參數學習" in txt2
    print("✅ 9 文字輸出（Markdown 安全）")

    # 10) 事件靜默：FOMC 決議日（及前一日）不開新倉/不加碼——重放與正式同語意；關閉時可買
    bo_days = [d for d in pre["dates"] if _blackout_day(d, {})]
    assert bo_days and _blackout_day("2025-03-19", {}) and not _blackout_day("2025-03-20", {})
    nxt = {d: pre["dates"][i + 1] for i, d in enumerate(pre["dates"][:-1])}
    fill_after_bo = {nxt[d] for d in bo_days if d in nxt}
    rep_bo = replay(pre, {"buy_threshold": 0.3, "entry_max_ext_atr": 0, "event_blackout": True})
    buys_bo = [j for j in rep_bo["journal"] if j.get("side") == "buy" and j.get("date") in fill_after_bo]
    assert buys_bo == [], buys_bo
    rep_nb = replay(pre, {"buy_threshold": 0.3, "entry_max_ext_atr": 0, "event_blackout": False})
    assert len([j for j in rep_nb["journal"] if j.get("side") == "buy"]) >= len([j for j in rep_bo["journal"] if j.get("side") == "buy"])
    # 進場品質網格：ext_atr 進 scored、apply/clear 鍵名帶 eng_ 前綴並可還原
    assert all("ext_atr" in v for d in pre["dates"][-5:] for v in pre["by_date"][d].values())
    st_g = {"thresholds": {"eng_trail_pct": 0.08}}
    wr = apply_params(st_g, {"entry_max_ext_atr": 1.5, "neutral_risk_mult": 1.0, "pyramid_r": 2.0})
    assert set(wr) == {"eng_entry_max_ext_atr", "eng_neutral_risk_mult", "eng_pyramid_r"} and st_g["thresholds"]["eng_pyramid_r"] == 2.0
    clear_params(st_g)
    assert st_g["thresholds"] == {"eng_trail_pct": 0.08}
    strict = replay(pre, {"buy_threshold": 0.3, "entry_max_ext_atr": 0.5, "entry_max_ret5d": 0})["metrics"]
    loose_e = replay(pre, {"buy_threshold": 0.3, "entry_max_ext_atr": 0})["metrics"]
    assert strict["n_trades"] <= loose_e["n_trades"]                       # 追高濾網越嚴交易越少
    print(f"✅ 10 事件靜默重放（{len(bo_days)} 個靜默日無買單）、進場網格 apply/clear、追高濾網敏感度")

    # 11) 舊邏輯重放：只有 legacy 機制、無停損/追蹤；放寬出場網格可跑、分批關閉時無 scale_out；optimize 附舊邏輯基準
    rep_lg = replay(pre, {"legacy": True, "buy_threshold": 0.3})
    mechs = set(rep_lg["metrics"]["by_mech"])
    assert rep_lg["journal"] and mechs <= {"legacy_exit", "exit"} and "stop_loss" not in mechs, mechs
    rep_noso = replay(pre, {"buy_threshold": 0.3, "entry_max_ext_atr": 0, "scale_out_r": 99.0})
    assert "scale_out" not in rep_noso["metrics"]["by_mech"]
    assert len(list(product(*GRID_LOOSE.values()))) == 36 and all(k in __import__("trade_engine").ENGINE_DEFAULTS for k in GRID_LOOSE)
    # H1：不收緊時不同 trail_pct 必須產生不同結果（網格不得退化）
    outs = {tp: replay(pre, {"buy_threshold": 0.3, "entry_max_ext_atr": 0, "trail_pct": tp, "trail_tighten_r": 99.0,
                             "scale_out_r": 99.0})["metrics"]["total_ret"] for tp in (0.08, 0.20)}
    assert outs[0.08] != outs[0.20], outs
    # M2：舊邏輯忽略事件靜默（同正式 Shadow）、吃 legacy_cfg
    rep_bo_lg = replay(pre, {"legacy": True, "buy_threshold": 0.3, "event_blackout": True})
    rep_nb_lg = replay(pre, {"legacy": True, "buy_threshold": 0.3, "event_blackout": False})
    assert rep_bo_lg["metrics"]["total_ret"] == rep_nb_lg["metrics"]["total_ret"]
    rep_hi = replay(pre, {"legacy": True, "legacy_cfg": {"buy_threshold": 0.95, "exit_threshold": -0.2}})
    assert rep_hi["metrics"]["n_trades"] <= rep_lg["metrics"]["n_trades"]
    assert "死錢天數 60" in _params_text({"dead_money_days": 60}) and "收緊門檻R 關" in _params_text({"trail_tighten_r": 99.0})
    opt_l = optimize(pre, {"trail_pct": (0.08, 0.2), "scale_out_r": (1.5, 99.0)})
    assert opt_l.get("legacy") and set(opt_l["legacy"]) >= {"train", "val", "holdout"}
    assert "_holdout_eq" not in opt_l["legacy"] and opt_l["n_trials"] >= 4
    txt_l = opt_text(opt_l); assert "舊邏輯" in txt_l and "**" not in txt_l
    assert "分批R 關" in _params_text({"scale_out_r": 99.0})
    print("✅ 11 舊邏輯重放／放寬出場網格（分批可關閉）／optimize 附舊邏輯基準")

    # 12) #72 ret_5d 進重放：與 indicators._extension 同定義；有 ret_5d 時「延伸但平穩」放行 → 進場不少於缺欄版
    import indicators as _ind
    _n12 = 0
    for d_ in pre["dates"][::20]:                       # 與正式 scan 的 indicators._extension 逐欄對照（同定義同捨入）
        for s_, v in pre["by_date"][d_].items():
            _x = _ind._extension(data[s_]["Close"].loc[:d_], data[s_]["High"].loc[:d_], data[s_]["Low"].loc[:d_])
            assert v["ret_5d"] == _x["ret_5d"] and v["ext_atr"] == _x["ext_atr"], (d_, s_, v, _x)
            _n12 += 1
    pre_no5 = {**pre, "by_date": {d: {s_: {k: x for k, x in v.items() if k != "ret_5d"} for s_, v in day.items()}
                                  for d, day in pre["by_date"].items()}}
    _p12 = {"buy_threshold": 0.3, "entry_max_ext_atr": 0.5}
    buys_5 = len([j for j in replay(pre, _p12)["journal"] if j.get("side") == "buy"])
    buys_no5 = len([j for j in replay(pre_no5, _p12)["journal"] if j.get("side") == "buy"])
    assert buys_5 > buys_no5, (buys_5, buys_no5)        # 嚴格大於：scored rows 漏帶 ret_5d 的回歸會被抓到
    print(f"✅ 12 ret_5d／ext_atr 與 indicators._extension 一致（{_n12} 筆）、進重放（買單 {buys_no5} → {buys_5}，#72）")

    # 13) #71 回撤鎖重置 + 鎖定天數統計：崩跌→盤整的合成行情，N=0（永不重置）鎖得比 N=10 久
    crash = {f"C{i}": _synthetic(seed=40 + i, drift=-0.004, vol=0.03, n=300) for i in range(4)}
    for i in range(4):                                   # 後半段轉為溫和上漲
        df_ = crash[f"C{i}"]
        tail = _synthetic(seed=60 + i, drift=0.0015, vol=0.015, n=150, start=str(df_.index[-1].date()))
        k_ = float(df_["Close"].iloc[-1]) / float(tail["Close"].iloc[0])
        crash[f"C{i}"] = pd.concat([df_, (tail.iloc[1:] * [k_, k_, k_, k_, 1])])
    crash["SPY"] = _synthetic(seed=99, drift=0.0004, vol=0.01, n=449)
    pre_c = precompute(crash, days=380, thresholds={"mtf_enabled": False})
    _pc = {"buy_threshold": 0.2, "entry_max_ext_atr": 0, "max_dd_halt": 0.05}
    lk0 = replay(pre_c, {**_pc, "dd_reset_flat_days": 0})["metrics"]
    lk10 = replay(pre_c, {**_pc, "dd_reset_flat_days": 10})["metrics"]
    assert set(lk0["lock_days"]) == {"dd", "regime", "halt"}
    assert lk0["lock_days"]["dd"] > 0 and lk10["lock_days"]["dd"] < lk0["lock_days"]["dd"], (lk0["lock_days"], lk10["lock_days"])
    print(f"✅ 13 回撤鎖：永不重置鎖 {lk0['lock_days']['dd']} 交易日 → 空手 10 日曆日重置後 {lk10['lock_days']['dd']} 交易日（#71）")

    # 14) 分段對照：bench/等權持有/beta/持有天數，文字含對照列且 Markdown 安全
    assert set(opt["bench"]) == {"train", "val", "holdout"} and opt["bench"]["holdout"]["ew"] is not None
    assert opt["baseline"]["holdout"].get("beta") is not None and "hold_med" in opt["baseline"]["holdout"]
    t14 = opt_text(opt)
    assert "分段對照" in t14 and "等權持有" in t14 and t14.count("*") % 2 == 0 and "_" not in t14, t14
    r14 = run_text(rep, {"buy_threshold": 0.3}, pre["dates"], 0.05, "1y", ew=0.1)
    assert "等權買進持有" in r14 and "beta" in r14 and "_" not in r14.replace("`/engtest opt`", "")
    _pre14 = {"n_syms": 2, "by_date": {"d1": {"A": {"close": 10.0}}, "d2": {"A": {"close": 11.0}, "B": {"close": 20.0}},
                                       "d3": {"B": {"close": 22.0}}}}
    _e14 = ew_return(_pre14, ["d1", "d2", "d3"], detail=True)            # A 末日缺、B 首日缺 → 各取自己的首末日
    assert abs(_e14[0] - 0.10) < 1e-9 and _e14[1:] == (2, 2), _e14
    _rf_on = replay(pre_c, {**_pc, "dd_reset_flat_days": 0})["metrics"]["lock_days"]["regime"]
    _rf_off = replay(pre_c, {**_pc, "dd_reset_flat_days": 0, "regime_filter": False})["metrics"]["lock_days"]["regime"]
    assert _rf_off == 0 and _rf_on >= _rf_off, (_rf_on, _rf_off)          # 大盤濾網關閉 → 不因偏空停新倉
    print("✅ 14 分段對照（SPY／等權持有 各檔自取首末日／曝險／beta／持有天數按部位／鎖定交易日）")
    # 15) 向量化 precompute 與逐日切片版逐位相等（含 NaN 成交量、mtf）
    d15 = {"P": _synthetic(seed=21, n=330), "Q": _synthetic(seed=22, n=330, drift=-0.0005), "SPY": _synthetic(seed=99, n=330)}
    d15["P"].iloc[100:104, 4] = float("nan")
    for _mtf in (True, False):
        _a = precompute(d15, 200, {"mtf_enabled": _mtf}, fast=False)
        _b = precompute(d15, 200, {"mtf_enabled": _mtf}, fast=True)
        assert _a["dates"] == _b["dates"] and _a["by_date"] == _b["by_date"]
    print("✅ 15 向量化 precompute 與逐日版逐位相等")

    # 16) PIT 抽樣（決定性、只抽當時成分）、成分遮罩（離開指數後不開新倉，舊邏輯同樣受限）、/engtest pit 文字
    per16 = {"AAA": [("2020-01-01", "9999-12-31")], "BBB": [("2020-01-01", "2024-12-31")],
             "CCC": [("2025-06-01", "9999-12-31")], "DDD": [("2019-01-01", "9999-12-31")]}
    assert pit_sample(per16, "2025-01-15", 2, 7) == pit_sample(per16, "2025-01-15", 2, 7)
    assert set(pit_sample(per16, "2025-01-15", 9, 1)) == {"AAA", "DDD"}             # BBB 已離開、CCC 尚未加入
    pre16 = precompute({k_: data[k_] for k_ in ("AAA", "BBB", "SPY")}, days=200, thresholds={"mtf_enabled": True})
    cut16 = pre16["dates"][len(pre16["dates"]) // 2]
    pre16["pit"] = {"AAA": [("2000-01-01", "9999-12-31")], "BBB": [("2000-01-01", cut16)]}
    pre16n = {k_: v for k_, v in pre16.items() if k_ != "pit"}
    for prm16 in ({"buy_threshold": 0.2, "entry_max_ext_atr": 0}, {"legacy": True, "legacy_cfg": {"buy_threshold": 0.2}}):
        _late = lambda pp: [j for j in replay(pp, prm16)["journal"]
                            if j.get("side") == "buy" and j.get("symbol") == "BBB" and j.get("date") > cut16]
        assert _late(pre16n) and not _late(pre16), prm16              # 無遮罩時確實會買 → 遮罩真的擋下
    rows16 = pit_compare({k_: data[k_] for k_ in ("AAA", "BBB", "CCC", "SPY")}, per16, [["AAA", "BBB", "CCC"], ["AAA", "ZZZ"]],
                         [1, 2], 200, {"buy_threshold": 0.3}, {"buy_threshold": 0.3}, {"stop_mult": 1.5})
    assert rows16[0]["n"] == 3 and "cand" in rows16[0] and rows16[1].get("skip"), rows16
    t16 = pit_text(rows16, {"as_of": "2024-06-01", "n_members": 3, "k": 3, "period": "1y", "candidate": {"stop_mult": 1.5}})
    assert "無事後偏誤回測" in t16 and "跨組平均" in t16 and "略過" in t16
    assert t16.count("*") % 2 == 0 and "_" not in t16 and "**" not in t16, t16
    fake_fetch = lambda syms, period: {k_: data[k_] for k_ in list(syms) + ["SPY"] if k_ in data}
    r16 = run_pit("1y", 5, 2, 1, {"buy_threshold": 0.3}, {"buy_threshold": 0.3},
                  periods={k_: [("2000-01-01", "9999-12-31")] for k_ in ("AAA", "BBB", "CCC", "DDD")},
                  fetch_fn=fake_fetch, today="2026-01-10")
    assert len(r16["rows"]) == 2 and all(r.get("n", 0) >= 3 for r in r16["rows"]), r16["rows"]
    assert run_pit("1y", 5, 1, periods={})["text"].startswith("❌")
    print("✅ 16 PIT 抽樣（決定性／當時成分）、成分遮罩（引擎與舊邏輯皆不在離開後進場）、pit 文字")

    # 17) /engtest try：參數解析（eng_ 前綴、off、布林、未知鍵、非有限數）＋三段並排、不寫入
    p17, e17 = parse_try_params(["trail_tighten_r=off", "eng_stop_mult=1.5", "neutral_pyramid=on",
                                 "max_positions=8", "regime_filter=off", "bogus=1", "risk_pct=nan", "seed=3", "2y"])
    assert p17 == {"trail_tighten_r": 99.0, "stop_mult": 1.5, "neutral_pyramid": True, "max_positions": 8,
                   "regime_filter": False}, p17
    assert len(e17) == 2 and all("`" in e for e in e17), e17
    res17 = try_compare(pre, {"buy_threshold": 0.3}, {"buy_threshold": 0.3, "scale_out_r": 99.0}, {"buy_threshold": 0.3})
    assert set(res17["base"]) == {"train", "val", "holdout"} and res17["cand"] != res17["base"]
    t17 = try_text(res17)
    assert "單組參數試算" in t17 and "不能直接套用" in t17 and t17.count("*") % 2 == 0
    assert "_" not in t17.replace("`/engtest try … pit`", ""), t17
    print("✅ 17 /engtest try 參數解析與三段並排（不寫入）")
    # 18) PBO 接進 optimize：無 best 也有 PBO、≥50% 取消推薦、文字含 PBO 行、暫存序列不外洩
    o18 = optimize(pre, {"trail_pct": (0.08, 0.2), "scale_out_r": (1.5, 99.0), "buy_threshold": (0.3, 0.5)})
    assert o18["pbo"].get("pbo") is not None and 2 <= o18["pbo"]["n_strats"] <= o18["n_trials"], o18["pbo"]   # 只含可選、去重後的組
    assert all("_tv_ret" not in r for r in o18["results"]) and "_tv_ret" not in (o18.get("legacy") or {})
    assert "PBO" in opt_text(o18) and "_" not in opt_text(o18)
    if o18.get("pbo_blocked"):
        assert o18["recommend"] is None
    assert _apply_pbo_gate({"recommend": {"x": 1}, "pbo": {"pbo": 0.6}}) == {"recommend": None, "pbo": {"pbo": 0.6}, "pbo_blocked": True}
    assert _apply_pbo_gate({"recommend": {"x": 1}, "pbo": {"pbo": 0.3}})["recommend"] == {"x": 1}
    assert _apply_pbo_gate({"recommend": {"x": 1}, "pbo": {"pbo": None}})["recommend"] == {"x": 1}
    _blk = opt_text({**o18, "recommend": None, "pbo_blocked": True})
    assert "取消推薦" in _blk and "_" not in _blk
    print(f"✅ 18 PBO 接進 optimize（{o18['pbo']['pbo']:.0%}，{o18['pbo']['n_splits']} 次切分；≥50% 取消推薦）")

    # 19) 重放起點錨定 SPY（歷史中斷的股票不得把起點拉早）＋資料中斷 ≥5 日的持倉以最後收盤結清
    d19 = {"LONG": _synthetic(seed=31, n=420, drift=0.002, vol=0.01), "SPY": _synthetic(seed=99, n=420)}
    d19["SHORT"] = _synthetic(seed=32, n=420, drift=0.003, vol=0.01).iloc[:300]          # 第 300 根後無資料（下市）
    p19 = precompute(d19, days=200, thresholds={"mtf_enabled": False})
    assert p19["dates"][0] == str(d19["SPY"].index[-200].date()) and len(p19["dates"]) == 200, p19["dates"][:2]
    assert p19["last_date"]["SHORT"] < p19["dates"][-1]
    rep19 = replay(p19, {"buy_threshold": -1.0, "entry_max_ext_atr": 0, "stop_mult": 5.0, "trail_activate_r": 50.0,
                         "scale_out_r": 99.0, "dead_money_days": 999, "exit_threshold": -9.0})
    ends = [t for t in rep19["trades"] if t["mechanism"] == "data_end"]
    if any(j.get("symbol") == "SHORT" and j.get("side") == "buy" for j in rep19["journal"]):
        assert ends and ends[0]["symbol"] == "SHORT" and "SHORT" not in rep19["book"]["positions"], ends
    print(f"✅ 19 重放起點錨定 SPY、資料中斷結清（{len(ends)} 筆）")

    # 20) pit 判讀：<3 組不判讀；自由度校正（3 組 t≈3 屬雜訊）；曝險註記不留空括號
    def _mk(br, lr, ew_, be=0.6, le=0.9):
        return {"seed": 1, "k": 5, "n": 5, "miss": 0, "dates": ("2024-01-02", "2026-01-02"), "ew": ew_, "spy": 0.1,
                "base": {"total_ret": br, "sharpe": 1.0, "exposure": be, "max_dd": 0.1},
                "legacy": {"total_ret": lr, "sharpe": 1.2, "exposure": le, "max_dd": 0.1}}
    m20 = {"as_of": "2024-01-01", "n_members": 500, "k": 5, "period": "2y"}
    t20a = pit_text([_mk(0.1, 0.2, 0.15), _mk(0.12, 0.25, 0.18)], m20)
    assert "無法判讀" in t20a and "一致" not in t20a.split("判讀：")[1], t20a
    t20b = pit_text([_mk(0.10, 0.20, 0.15), _mk(0.10, 0.23, 0.15), _mk(0.10, 0.24, 0.15)], m20)    # 舊−現 ≈ +12%、t≈7.6 → 顯著
    assert "舊邏輯跨組平均顯著領先現行（各組同向）；曝險差" in t20b and "（）" not in t20b, t20b
    t20e = pit_text([_mk(0.1, 0.2, 0.15)] * 4 + [_mk(0.1, 0.099, 0.15)], m20)      # 5 組中 1 組反向、t 仍 ≥2.78
    assert "方向相反" in t20e and "各組同向" not in t20e, t20e
    t20c = pit_text([_mk(0.10, 0.12, 0.15), _mk(0.10, 0.10, 0.15), _mk(0.10, 0.15, 0.15)], m20)    # t≈2.6 < 4.30 → 雜訊
    assert "舊邏輯 vs 現行在雜訊範圍內" in t20c, t20c
    for t_ in (t20a, t20b, t20c):
        assert t_.count("*") % 2 == 0 and "_" not in t_.replace("`/engtest pit 20 5`", ""), t_
    t20d = pit_text([dict(_mk(0.1, 0.2, 0.15), miss=4), {"seed": 2, "k": 5, "n": 0, "miss": 5, "skip": "SPY 抓不到"}], m20)
    assert "行情抓取可能失敗" in t20d
    print("✅ 20 pit 判讀（<3 組不判讀、自由度校正、曝險調整、抓取失敗警示）")

    # 21) 沒進場原因統計＋共用參數解析（Bot 與 engine_research 同一套規則）
    bk21 = replay(pre, {"buy_threshold": 0.95})["metrics"]["blocks"]
    assert bk21["no_signal"] > 0 and set(bk21) == {"no_signal", "ext", "neutral", "blackout"}, bk21
    _on = replay(pre, {"buy_threshold": 0.2, "event_blackout": True})["metrics"]["blocks"]
    _off = replay(pre, {"buy_threshold": 0.2, "event_blackout": False})["metrics"]["blocks"]
    assert _on["no_signal"] == _off["no_signal"] and _on["blackout"] > 0 == _off["blackout"], (_on, _off)   # 靜默不污染無訊號
    bk21b = replay(pre, {"buy_threshold": 0.2, "entry_max_ext_atr": 0.3, "entry_max_ret5d": 0})["metrics"]["blocks"]
    assert bk21b["ext"] > 0, bk21b
    assert "沒進場的原因" in run_text(rep, {"buy_threshold": 0.3}, pre["dates"], None, "1y")
    a21 = parse_engtest_args(["1y", "stop_mult=1.5"])
    assert a21["kind"] == "try" and a21["cand"] == {"stop_mult": 1.5} and a21["period"] == "1y"
    assert parse_engtest_args(["pit", "999", "99", "seed=7"])[ "k"] == PIT_K_RANGE[1]
    assert parse_engtest_args(["pit", "８", "2"])["k"] == PIT_K_RANGE[0] + 3 and parse_engtest_args(["try"])["errs"]
    assert parse_engtest_args(["opt", "loose", "apply"])["grid"] == "loose" and parse_engtest_args([])["kind"] == "run"
    print(f"✅ 21 沒進場原因統計（無訊號 {bk21['no_signal']} 日、追高擋 {bk21b['ext']} 次）＋共用參數解析")
    print("\nengine_backtest selftest OK ✅")
