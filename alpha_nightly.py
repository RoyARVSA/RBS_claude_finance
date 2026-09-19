"""
alpha_nightly.py – 夜間工作流進入點（A/B/C 段落地；規劃見 ALPHA_SPINE.md §0）

每個交易日收盤後跑一次（.github/workflows/alpha_nightly.yml）：
  1. 選股池 broad ∪ watchlist → yf.download 3y OHLCV（一次批次）
  2. A：價格因子 + 基本面 PIT 因子 + C 段核准因子 → IC 閘門 → 排名 → data/alpha/rank.json
  3. B：indicators.composite_series（＝引擎評分）→ 訊號樣本 → 引擎規則出場標籤 → purged walk-forward
        → data/alpha/meta.json
  4. 基本面覆蓋輪替：每晚 N 檔 fin_data（data/fin/）+ N 檔預估快照（data/alpha/estimates_universe.json）
  5. 寫「本輪觸碰檔案清單」給 workflow 只 add 這些檔（與 15 分鐘 cron 的 commit 不互相覆蓋）

安全：只讀 state 的明文鍵（watchlist、factor_lab）；不寫 state；不 print 持倉。
離線：`python3 alpha_nightly.py --offline` 用合成資料走完整流程（CI 自測）。
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
ALPHA_DIR = ROOT / "data" / "alpha"
UNI_LEDGER = ALPHA_DIR / "estimates_universe.json"
DEFAULTS = {
    "period": "3y",              # 2y 下 12-1 動能（252 日回看）可用快照不足、有效期數 <12 永遠過不了閘門（對抗驗證 M2）
    "max_symbols": 450,          # 保護：宇宙 + watchlist 上限
    "fin_per_night": 12,         # 基本面覆蓋輪替（yfinance 三表 3 次呼叫/檔）
    "est_per_night": 12,         # 預估快照輪替（quoteSummary 1 次/檔）
    "min_days": 260,             # 檔案至少要有的 K 棒數
}


# ── 1. 輸入 ───────────────────────────────────────────────────────────────────

def read_plain_state(path: Path) -> dict:
    """只取明文鍵（watchlist / factor_lab）；密文塊原樣不解。"""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    out = {"watchlist": [x for x in (raw.get("watchlist") or []) if isinstance(x, str)]}
    fl = raw.get("factor_lab")
    out["factor_lab"] = fl if isinstance(fl, dict) and "__enc__" not in fl else {}
    return out


def universe_symbols(snap: dict | None, watchlist: list[str], cap: int) -> list[str]:
    syms = [r.get("t") for r in ((snap or {}).get("broad") or []) if isinstance(r, dict) and r.get("t")]
    syms = [s for s in syms if isinstance(s, str) and "." not in s]      # yfinance 用連字號（BRK-B）可留；點號不行
    out = list(dict.fromkeys([s.upper() for s in (watchlist or []) if isinstance(s, str)] + [s.upper() for s in syms]))
    return out[:cap]


def fetch_ohlcv(tickers: list[str], period: str = "2y", min_days: int = 260) -> dict[str, pd.DataFrame]:
    """批次抓日線（含 SPY）。回 {sym: DataFrame(Open/High/Low/Close/Volume)}。"""
    import engine_backtest as eb
    data = eb.fetch_history(tickers, period)
    return {s: df for s, df in data.items() if len(df) >= min_days or s == "SPY"}


def load_fin_stores(symbols: list[str], fin_dir: Path) -> dict[str, dict]:
    import fin_data as fd
    out = {}
    for s in symbols:
        p = Path(fin_dir) / f"{s}.json"
        if p.exists():
            try:
                st = fd.load_store(s, fin_dir)
                if st.get("periods"):
                    out[s] = st
            except Exception:
                continue
    return out


def load_ledgers(paths: list[Path]) -> list[dict]:
    import estimates_ledger as el
    out = []
    for p in paths:
        try:
            if Path(p).exists():
                out.append(el.load_ledger(p))
        except Exception:
            continue
    return out


# ── 2. 基本面 PIT 查詢（B 段樣本用；與 A 段同函數）─────────────────────────────

class FundLookup:
    def __init__(self, stores: dict[str, dict], ledgers: list[dict]):
        import fin_data as fd
        self._fd = fd
        self.stores = stores
        self.rows = {}
        for led in ledgers:
            for sym, ent in ((led or {}).get("tickers") or {}).items():
                rows = (ent or {}).get("rows") or []
                if rows and (sym not in self.rows or len(rows) > len(self.rows[sym])):
                    self.rows[sym] = rows
        self._qcache: dict[tuple, float | None] = {}

    def __call__(self, sym: str, date: str) -> dict:
        import alpha_spine as asp
        import quality as ql
        q = None
        st = self.stores.get(sym)
        if st:
            periods = self._fd.pit_view(st, date, "A")
            k = len(periods)
            if k >= 2:
                key = (sym, k)
                if key not in self._qcache:
                    try:
                        self._qcache[key] = ql.quality_summary(periods, None, None).get("score")
                    except Exception:
                        self._qcache[key] = None
                q = self._qcache[key]
        r = asp.rev_score_pit(self.rows.get(sym, []), date) if sym in self.rows else None
        return {"quality": q, "rev": r}


# ── 3. 主流程（純邏輯；資料可注入）────────────────────────────────────────────

def run(today: str, data: dict[str, pd.DataFrame], stores: dict[str, dict], ledgers: list[dict],
        approved: dict[str, str] | None = None, out_dir: Path | None = None, blackout_fn=None) -> dict:
    """回 {"rank", "meta", "files": [寫出的檔案], "notes": [...]}。"""
    import alpha_spine as asp
    import indicators as ind
    import meta_label as ml
    out_dir = Path(out_dir or ALPHA_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    notes, files = [], []
    t0 = time.time()
    spy = data.get("SPY")
    uni = {s: df for s, df in data.items() if s != "SPY"}
    frames = asp.wide_frames(uni)
    frames["open"] = pd.DataFrame({s: df["Open"] for s, df in uni.items() if "Open" in df}).sort_index()
    if frames["close"].empty:
        raise RuntimeError("no price data")
    dates = asp.sample_dates(frames["close"].index)
    last_date = str(frames["close"].index[-1].date())
    all_dates = dates + ([last_date] if last_date not in dates else [])    # 今日排名也要「今日」的因子值（M1）
    # 基本面 PIT 因子（覆蓋不足時 evaluate_factors 自動不配權）
    fund = {}
    q = asp.quality_by_date(stores, all_dates)
    if q:
        fund["quality"] = q
    r = asp.rev_by_date(ledgers, all_dates)
    if r:
        fund["rev"] = r
    # C 段核准因子
    extra = {}
    if approved:
        import factor_lab as fl
        for name, expr in approved.items():
            try:
                extra[name] = asp.factor_by_date(fl.evaluate_expr(expr, frames), all_dates)
            except Exception as e:
                notes.append(f"因子 {name} 求值失敗：{type(e).__name__}")
    # as_of 由 build_rank 取最後 K 棒日（不用 today：假日／週末手動觸發時 today 不在索引會得到空榜，H1）
    rank = asp.build_rank(frames, dates, fund_fbd=fund, extra_fbd=extra, spy=spy["Close"] if spy is not None else None)
    rank["generated"] = today
    files.append(asp.save_rank(rank, out_dir / "rank.json"))
    notes.append(f"A 段：{rank['n_universe']} 檔、{rank['n_dates']} 快照、通過因子 "
                 f"{[k for k, v in rank['factors'].items() if v['pass']]}、{time.time() - t0:.0f}s")
    # B 段：引擎評分（向量化＝逐日評分）→ 樣本 → 訓練
    t1 = time.time()
    scores = {}
    for s, df in uni.items():
        try:
            scores[s] = ind.composite_series(df["Close"], df.get("High"), df.get("Low"), df.get("Volume"), None, True)
        except Exception:
            continue
    scores = pd.DataFrame(scores).reindex(frames["close"].index)
    br = asp.breadth_series(frames["close"])
    reg = None
    if spy is not None and len(spy) >= 50:
        import engine_backtest as eb
        reg = eb.regime_series(spy["Close"].dropna())
    ctx = {}
    for d in frames["close"].index:
        rg = None
        if reg is not None:
            try:
                v = reg.get(pd.Timestamp(d))
                rg = v if isinstance(v, str) else None
            except Exception:
                rg = None
        b = br.get(d)
        ctx[str(d.date())] = {"regime": rg, "breadth_pct": (float(b) if b == b else None)}
    lookup = FundLookup(stores, ledgers)
    samples = ml.build_samples(frames, scores, ctx, fund_lookup=lookup, blackout_fn=blackout_fn)
    meta = ml.train(samples, rank["as_of"])
    meta["generated"] = today
    files.append(ml.save_model(meta, out_dir / "meta.json"))
    o = meta.get("oos") or {}
    notes.append(f"B 段：樣本 {len(samples)}、OOS n {o.get('n_oos', 0)}、AUC {o.get('auc')}、閘門 {meta.get('gate_passed')}"
                 f"（{meta.get('reason')}）、{time.time() - t1:.0f}s")
    return {"rank": rank, "meta": meta, "files": files, "notes": notes}


# ── 4. 基本面覆蓋輪替（需網路；fetch 可注入）────────────────────────────────────

def expand_coverage(symbols: list[str], watchlist: list[str], today: str, fin_dir: Path, uni_ledger: Path,
                    n_fin: int = 12, n_est: int = 12, fetch_fin=None, fetch_snap=None, sleep_s: float = 1.0) -> list[Path]:
    """宇宙（扣 watchlist——那是 15 分鐘 cron 的責任）最舊優先補三表與預估快照。回觸碰的檔案。"""
    import estimates_ledger as el
    import fin_data as fd
    touched: list[Path] = []
    wl = set(watchlist or [])
    pool = [s for s in symbols if s not in wl and s != "SPY"]
    # 三表：沒有 store 或 updated 最舊者優先
    def _age(s):
        p = Path(fin_dir) / f"{s}.json"
        if not p.exists():
            return "0000-00-00"
        try:
            return str(json.loads(p.read_text(encoding="utf-8")).get("updated") or "0000-00-00")
        except Exception:
            return "0000-00-00"
    picked = sorted(pool, key=_age)[:max(0, int(n_fin))]
    ff = fetch_fin or (lambda s: fd.get_financials(s, today, base_dir=fin_dir))
    for i, s in enumerate(picked):
        try:
            ff(s)
            touched.append(Path(fin_dir) / f"{s}.json")
        except Exception:
            pass
        if sleep_s and i < len(picked) - 1:
            time.sleep(sleep_s)
    # 預估快照：獨立帳本（不與主帳本共檔，避免兩個 workflow 互相覆蓋）
    try:
        led = el.load_ledger(uni_ledger)
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        el.refresh(led, pool, now_iso, {"est_refresh_per_run": int(n_est), "est_ttl_hours": 24 * 7},
                   fetch_snap=fetch_snap, sleep_s=sleep_s)
        if el.save_ledger(uni_ledger, led):
            touched.append(Path(uni_ledger))
    except Exception as e:
        print(f"alpha_nightly: 預估快照輪替失敗 {type(e).__name__}")
    return touched


# ── 5. CLI ────────────────────────────────────────────────────────────────────

def _synthetic(n_sym: int = 30, n_day: int = 600, seed: int = 4) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-04-01", periods=n_day)
    data = {}
    for k in range(n_sym + 1):
        c = 100 * np.cumprod(1 + rng.normal(0.0004, 0.018, n_day))
        name = "SPY" if k == n_sym else f"Z{k:02d}"
        data[name] = pd.DataFrame({"Open": c * (1 + rng.normal(0, 0.002, n_day)), "High": c * 1.01, "Low": c * 0.99,
                                   "Close": c, "Volume": rng.integers(1e5, 1e6, n_day).astype(float)}, index=idx)
    return data


def main(argv: list[str]) -> int:
    offline = "--offline" in argv
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if offline:
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            res = run(today, _synthetic(), {}, [], approved={"lowvol": "-vol(60)"}, out_dir=Path(td))
            assert (Path(td) / "rank.json").exists() and (Path(td) / "meta.json").exists()
            assert res["rank"]["n_universe"] == 30 and "lowvol" in res["rank"]["factors"]
            assert isinstance(res["meta"].get("gate_passed"), bool)
            # 覆蓋輪替（注入假抓取；不碰網路）
            fin_dir = Path(td) / "fin"; fin_dir.mkdir()
            calls = []
            touched = expand_coverage([f"Z{k:02d}" for k in range(30)], ["Z00", "Z01"], today, fin_dir, Path(td) / "eu.json",
                                      n_fin=3, n_est=2, fetch_fin=lambda s: calls.append(s), fetch_snap=lambda s: None, sleep_s=0)
            assert calls == ["Z02", "Z03", "Z04"] and any(str(p).endswith("eu.json") for p in touched)
            st = read_plain_state(Path(td) / "nope.json")
            assert st == {}
            (Path(td) / "ws.json").write_text(json.dumps({"watchlist": ["AAPL"], "factor_lab": {"approved": {"a": {"expr": "ret(5)"}}},
                                                          "engine": {"__enc__": True}}), encoding="utf-8")
            st = read_plain_state(Path(td) / "ws.json")
            assert st["watchlist"] == ["AAPL"] and st["factor_lab"]["approved"]["a"]["expr"] == "ret(5)"
            assert universe_symbols({"broad": [{"t": "MSFT"}, {"t": "BRK.B"}, {"t": "BRK-B"}, {"t": "AAPL"}]}, ["AAPL"], 10) == ["AAPL", "MSFT", "BRK-B"]
            # H1/M1：假日觸發（today 不在索引）rank 仍有榜；核准因子今日值＝最後 K 棒當日值
            r_top = res["rank"]["top"]
            assert r_top or not res["rank"]["gate_passed"]
            if "lowvol" in res["rank"]["factors"] and res["rank"]["factors"]["lowvol"]["weight"] > 0:
                import factor_lab as _fl, alpha_spine as _asp
                fr = _asp.wide_frames({k: v for k, v in _synthetic().items() if k != "SPY"})
                w = _fl.evaluate_expr("-vol(60)", fr)
                t0 = r_top[0]
                assert abs(t0["f"]["lowvol"] - float(w[t0["t"]].iloc[-1])) < 1e-6
        for n in res["notes"]:
            print(n)
        print("\nalpha_nightly selftest OK ✅")
        return 0

    import macro
    import universe as un
    st = read_plain_state(ROOT / "watchlist_state.json")
    snap = un.load_latest_snapshot()
    syms = universe_symbols(snap, st.get("watchlist") or [], int(DEFAULTS["max_symbols"]))
    if not syms:
        print("alpha_nightly: 無選股池快照也無 watchlist，結束")
        return 0
    print(f"alpha_nightly: 宇宙 {len(syms)} 檔，抓價中…")
    data = fetch_ohlcv(syms, DEFAULTS["period"], int(DEFAULTS["min_days"]))
    if len(data) < 20:
        print("alpha_nightly: 行情不足，結束（不覆寫舊輸出）")
        return 0
    fin_dir = ROOT / "data" / "fin"
    stores = load_fin_stores(list(data), fin_dir)
    ledgers = load_ledgers([ROOT / "estimates_ledger.json", UNI_LEDGER])
    import factor_lab as fl
    approved = fl.approved_factors(st)
    res = run(today, data, stores, ledgers, approved=approved,
              blackout_fn=lambda d: macro.event_blackout(d, 1)[0])
    for n in res["notes"]:
        print(n)
    touched = list(res["files"])
    try:
        touched += expand_coverage(list(data), st.get("watchlist") or [], today, fin_dir, UNI_LEDGER,
                                   int(DEFAULTS["fin_per_night"]), int(DEFAULTS["est_per_night"]))
    except Exception as e:
        print(f"alpha_nightly: 覆蓋輪替錯誤 {type(e).__name__}")
    tf = os.environ.get("ALPHA_TOUCHED_FILE")
    if tf:
        Path(tf).write_text("\n".join(str(Path(p).relative_to(ROOT)) for p in touched), encoding="utf-8")
    print(f"alpha_nightly: 完成，觸碰 {len(touched)} 個檔案")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
