"""
universe.py – 選股池（Universe）層：Stage 1 寬宇宙 → Stage 2 品質/流動性/動能篩 → 快照與成分歷史

背景（VALUATION_LANDSCAPE §2、§7）：Lean / Zipline / Qlib 的核心不是因子，是「宇宙管線 +
point-in-time 紀律」。本專案原本只有固定 watchlist：(1) 引擎只看得到人工挑的名單；
(2) 回測用今日名單 → 倖存者偏差。本模組把宇宙變成**月頻重建、可回溯的資料**：

  Layer 3 Broad   ：yf.screen（美股、市值 ≥ 20 億、3 月均量 ≥ 100 萬股、價 ≥ $5）→ 500–750 檔，3 次呼叫
  Layer 2 Theme   ：stock_db 主題（AI 光學/連接/化合物半導體/HBM/電力散熱 …）
  Layer 1 Core    ：watchlist（∪ 持倉——持倉永遠保留，出場只走價格機制）
  Stage 2 Screen  ：品質（ROE、流動比、負債比、淨利率）+ 流動性（ADV$）+ 12-1 動能 → 候選前 N
  成分歷史        ：S&P 500（fja05680，1996 起）、NDX（yfiua 月快照）期間表 → 回測任一天用「當天成分」

規則：宇宙只決定「誰能被引擎看到」與（P3 之後）「配多少」；進場仍由技術評分過門檻、
出場仍由價格機制；**持倉不因掉出宇宙而被賣**。P0 只做快照 + 顯示，不接引擎。

PIT 欄位：每份快照帶 `as_of`（資料日）與 `available_at`（as_of + 1 交易日，Lean T+1 規則）；
快照 append-only 落在 data/universe/YYYY-MM.json（公開資料衍生，明文）。
純邏輯離線可測；抓取層需網路。教育用途，非投資建議。
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime, timedelta
from pathlib import Path

DEFAULTS = {
    "uni_enabled":       True,
    "uni_min_mcap":      2e9,      # 市值下限（USD）
    "uni_min_avgvol":    1e6,      # 3 月日均量（股）
    "uni_min_price":     5.0,
    "uni_min_roe":       0.08,     # Stage 2 品質：ROE ≥ 8%
    "uni_min_current":   1.0,      # 流動比 ≥ 1
    "uni_max_de":        2.0,      # 負債/權益 ≤ 200%
    "uni_min_margin":    0.0,      # 淨利率 > 0
    "uni_top_n":         60,       # 動能前 N 進候選池
    "uni_pages":         3,        # yf.screen 翻頁數（每頁 250）
    "uni_rebuild_day":   1,        # 每月幾號後的第一輪重建
}
EXCLUDE_SECTORS = ("Financial Services", "Real Estate")   # EV 類指標無意義 → 另路由（RIM/AFFO），P1
UNIVERSE_DIR = Path(__file__).parent / "data" / "universe"
SP500_PERIODS_URL = "https://raw.githubusercontent.com/fja05680/sp500/master/sp500_ticker_start_end.csv"
NDX_MONTHLY_URL = "https://yfiua.github.io/index-constituents/{y}/{m:02d}/constituents-ndx.csv"


# ── 小工具 ────────────────────────────────────────────────────────────────

def _f(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def next_trading_day(d: str) -> str:
    """as_of → available_at（T+1 交易日；週末跳過，假日不判——保守足夠）。"""
    dt = datetime.strptime(d[:10], "%Y-%m-%d").date() + timedelta(days=1)
    while dt.weekday() >= 5:
        dt += timedelta(days=1)
    return dt.isoformat()


# ── 1. Stage 1：寬宇宙（解析為純邏輯）───────────────────────────────────────

def parse_screen_quotes(payload: dict | None) -> list[dict]:
    """yf.screen 回應（finance.result[0]）→ 標準列。缺欄一律 None。"""
    if not isinstance(payload, dict):
        return []
    out = []
    for q in payload.get("quotes") or []:
        if not isinstance(q, dict) or not q.get("symbol"):
            continue
        out.append({
            "ticker": str(q["symbol"]).upper(),
            "name": q.get("shortName") or q.get("longName"),
            "mcap": _f(q.get("marketCap")),
            "avgvol": _f(q.get("averageDailyVolume3Month")),
            "price": _f(q.get("regularMarketPrice")),
            "sector": q.get("sector"),
            "industry": q.get("industry"),
            "exchange": q.get("fullExchangeName") or q.get("exchange"),
        })
    return out


def stage1_filter(rows: list[dict], cfg: dict | None = None) -> list[dict]:
    """在本地再套一次門檻（screener 端可能鬆動）+ 去重 + 排除非普通股樣式（權證/單位）。"""
    c = {**DEFAULTS, **(cfg or {})}
    seen, out = set(), []
    for r in rows:
        t = r.get("ticker") or ""
        if not t or t in seen or "." in t or "-" in t and t.endswith(("-WT", "-U", "-R")):
            continue
        if (r.get("mcap") or 0) < float(c["uni_min_mcap"]):
            continue
        if (r.get("avgvol") or 0) < float(c["uni_min_avgvol"]):
            continue
        if (r.get("price") or 0) < float(c["uni_min_price"]):
            continue
        seen.add(t)
        out.append(r)
    return out


# ── 2. Stage 2：品質 / 流動性 / 動能（純邏輯）─────────────────────────────

def quality_gate(row: dict, cfg: dict | None = None) -> tuple[bool, str]:
    """品質門檻（缺值不擋——資料缺席不是負面證據；金融/地產另路由）。"""
    c = {**DEFAULTS, **(cfg or {})}
    if row.get("sector") in EXCLUDE_SECTORS:
        return False, "sector_route"
    roe, cr, de, nm = row.get("roe"), row.get("current_ratio"), row.get("de"), row.get("net_margin")
    if roe is not None and roe < float(c["uni_min_roe"]):
        return False, "roe"
    if cr is not None and cr < float(c["uni_min_current"]):
        return False, "current_ratio"
    if de is not None and de > float(c["uni_max_de"]):
        return False, "de"
    if nm is not None and nm <= float(c["uni_min_margin"]):
        return False, "margin"
    return True, ""


def momentum_metrics(closes: dict) -> dict:
    """{ticker: close Series（≥ 130 日）} → {ticker: {mom_12_1, adv20, off_high, days}}。
    12-1 動能 = 252 日前→21 日前報酬（缺 252 日用最早可得，但至少 130 日）；
    off_high = 現價 / 52 週高 − 1。全部只用序列內資料（無前視）。"""
    out = {}
    for t, s in (closes or {}).items():
        try:
            s = s.dropna()
            n = len(s)
            if n < 130:
                continue
            p_now = float(s.iloc[-21]) if n >= 21 else float(s.iloc[-1])
            p_then = float(s.iloc[-252]) if n >= 252 else float(s.iloc[0])
            hi = float(s.iloc[-252:].max()) if n >= 252 else float(s.max())
            out[t] = {"mom_12_1": (p_now / p_then - 1) if p_then > 0 else None,
                      "off_high": (float(s.iloc[-1]) / hi - 1) if hi > 0 else None,
                      "days": n}
        except Exception:
            continue
    return out


def stage2_rank(rows: list[dict], mom: dict, cfg: dict | None = None) -> tuple[list[dict], dict]:
    """品質門檻 → 動能排序 → 前 N。回 (候選列, 統計)。"""
    c = {**DEFAULTS, **(cfg or {})}
    passed, reasons = [], {}
    for r in rows:
        ok, why = quality_gate(r, c)
        if not ok:
            reasons[why] = reasons.get(why, 0) + 1
            continue
        m = mom.get(r["ticker"])
        if not m or m.get("mom_12_1") is None:
            reasons["no_price_history"] = reasons.get("no_price_history", 0) + 1
            continue
        passed.append({**r, **m, "adv_usd": (r.get("avgvol") or 0) * (r.get("price") or 0)})
    passed.sort(key=lambda x: -(x["mom_12_1"] or -9))
    top = passed[: int(c["uni_top_n"])]
    for i, r in enumerate(top, 1):
        r["rank"] = i
    return top, {"passed_quality": len(passed), "rejected": reasons}


# ── 3. 成分歷史（純邏輯）─────────────────────────────────────────────────────

def parse_period_csv(text: str) -> dict[str, list[tuple[str, str]]]:
    """fja05680 `sp500_ticker_start_end.csv`（ticker,start_date,end_date；end 空=至今）→ 期間表。
    容忍欄名大小寫與多段期間（同代碼多列）。"""
    import csv
    import io
    out: dict[str, list[tuple[str, str]]] = {}
    rd = csv.DictReader(io.StringIO(text))
    if not rd.fieldnames:
        return out
    cols = {c.lower().strip(): c for c in rd.fieldnames}
    tk = cols.get("ticker") or cols.get("symbol")
    st = cols.get("start_date") or cols.get("start")
    en = cols.get("end_date") or cols.get("end")
    if not (tk and st):
        return out
    for row in rd:
        t = (row.get(tk) or "").strip().upper().replace(".", "-")
        s = (row.get(st) or "").strip()[:10]
        e = (row.get(en) or "").strip()[:10] if en else ""
        if t and s:
            out.setdefault(t, []).append((s, e or "9999-12-31"))
    return out


def parse_daily_list_csv(text: str) -> dict[str, list[tuple[str, str]]]:
    """fja05680 每日成分格式（date,tickers 逗號串）→ 期間表（連續出現合併）。"""
    import csv
    import io
    rows = list(csv.reader(io.StringIO(text)))
    if len(rows) < 2:
        return {}
    body = rows[1:] if rows[0] and rows[0][0].lower() in ("date",) else rows
    body = sorted((r for r in body if len(r) >= 2 and r[0]), key=lambda r: r[0])
    active: dict[str, str] = {}
    out: dict[str, list[tuple[str, str]]] = {}
    prev_d = None
    for r in body:
        d = r[0][:10]
        cur = {t.strip().upper().replace(".", "-") for t in r[1].split(",") if t.strip()}
        for t in list(active):
            if t not in cur:
                out.setdefault(t, []).append((active.pop(t), prev_d or d))
        for t in cur:
            active.setdefault(t, d)
        prev_d = d
    for t, s in active.items():
        out.setdefault(t, []).append((s, "9999-12-31"))
    return out


def members_on(periods: dict[str, list[tuple[str, str]]], d: str) -> list[str]:
    """某日的成分（start ≤ d ≤ end）。"""
    d = d[:10]
    return sorted(t for t, ps in periods.items() if any(s <= d <= e for s, e in ps))


# ── 4. 快照（純邏輯）─────────────────────────────────────────────────────────

def build_snapshot(as_of: str, broad: list[dict], top: list[dict], stats: dict,
                   themes: dict | None = None, core: list[str] | None = None) -> dict:
    return {
        "version": 1, "as_of": as_of[:10], "available_at": next_trading_day(as_of),
        "counts": {"broad": len(broad), "top": len(top), **{k: v for k, v in stats.items() if k != "rejected"}},
        "rejected": stats.get("rejected", {}),
        "core": sorted(core or []),
        "themes": {k: sorted(v) for k, v in (themes or {}).items()},
        "broad": [{"t": r["ticker"], "mc": r.get("mcap"), "v": r.get("avgvol"), "p": r.get("price"),
                   "s": r.get("sector")} for r in broad],
        "top": [{"t": r["ticker"], "rank": r.get("rank"), "mom": r.get("mom_12_1"), "oh": r.get("off_high"),
                 "adv": r.get("adv_usd"), "s": r.get("sector")} for r in top],
    }


def save_snapshot(snap: dict, base_dir: Path | None = None) -> Path:
    """data/universe/YYYY-MM.json（同月覆寫——月頻重建的最新版）。"""
    base = Path(base_dir or UNIVERSE_DIR)
    base.mkdir(parents=True, exist_ok=True)
    p = base / f"{snap['as_of'][:7]}.json"
    p.write_text(json.dumps(snap, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return p


def load_latest_snapshot(base_dir: Path | None = None) -> dict | None:
    base = Path(base_dir or UNIVERSE_DIR)
    if not base.exists():
        return None
    files = sorted(base.glob("*.json"))
    for p in reversed(files):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(d, dict) and d.get("as_of"):
                return d
        except Exception:
            continue
    return None


def should_rebuild(state: dict, today: str, cfg: dict | None = None) -> bool:
    """每月一次：今天 ≥ 重建日 且 本月尚未重建。"""
    c = {**DEFAULTS, **(cfg or {})}
    if not c.get("uni_enabled", True):
        return False
    last = str((state.get("universe") or {}).get("as_of") or "")
    d = datetime.strptime(today[:10], "%Y-%m-%d").date()
    return d.day >= int(c["uni_rebuild_day"]) and last[:7] != today[:7]


def universe_text(snap: dict | None, state_summary: dict | None = None) -> str:
    """Telegram legacy Markdown（單 *、無底線）。"""
    if not snap:
        return ("🌐 *選股池未建立*\n每月自動重建（或 `/universe rebuild`）：yf.screen 寬宇宙 → 品質/流動性/動能篩 → 候選前 N。"
                "P0 只快照與顯示，不接引擎")
    c = snap.get("counts", {})
    rej = snap.get("rejected", {})
    lines = [f"🌐 *選股池快照*（資料日 {snap['as_of']}，回測可用日 {snap.get('available_at')}）",
             f"寬宇宙 {c.get('broad', 0)} 檔 → 品質通過 {c.get('passed_quality', 0)} → 動能前 {c.get('top', 0)}"]
    if rej:
        lines.append("淘汰：" + "、".join(f"{k.replace('_', '·')} {v}" for k, v in sorted(rej.items(), key=lambda kv: -kv[1])))
    top = snap.get("top") or []
    if top:
        lines.append("*候選前 10（12-1 動能）*：")
        for r in top[:10]:
            mom = r.get("mom")
            lines.append(f"・{r['t']} {'' if mom is None else f'{mom:+.0%}'}"
                         f"（距 52 週高 {(r.get('oh') or 0):+.0%}，{(r.get('s') or '—')}）")
    core = snap.get("core") or []
    if core:
        in_top = sum(1 for r in top if r["t"] in core)
        lines.append(f"watchlist {len(core)} 檔中 {in_top} 檔在候選池內")
    lines.append("宇宙只決定引擎看得到誰；進場仍看技術評分、出場仍走價格機制；非投資建議")
    return "\n".join(lines)


# ── 5. 抓取層（需網路）────────────────────────────────────────────────────────

def fetch_broad(cfg: dict | None = None) -> list[dict]:
    """yf.screen 三次翻頁（每頁 250）。任何一頁失敗就用已拿到的。"""
    import yfinance as yf
    c = {**DEFAULTS, **(cfg or {})}
    q = yf.EquityQuery("and", [
        yf.EquityQuery("eq", ["region", "us"]),
        yf.EquityQuery("gt", ["intradaymarketcap", float(c["uni_min_mcap"])]),
        yf.EquityQuery("gt", ["avgdailyvol3m", float(c["uni_min_avgvol"])]),
        yf.EquityQuery("gt", ["intradayprice", float(c["uni_min_price"])]),
    ])
    rows = []
    for page in range(int(c["uni_pages"])):
        try:
            resp = yf.screen(q, offset=page * 250, size=250, sortField="intradaymarketcap", sortAsc=False)
        except Exception as e:
            print(f"universe: screen 第 {page + 1} 頁失敗 {e}")
            break
        got = parse_screen_quotes(resp)
        rows.extend(got)
        if len(got) < 250:
            break
    return stage1_filter(rows, c)


def fetch_closes_batch(tickers: list[str], period: str = "1y") -> dict:
    from behavior_check import fetch_closes
    out = {}
    for i in range(0, len(tickers), 200):          # 分批，避免單請求過大
        out.update(fetch_closes(tickers[i:i + 200], period=period) or {})
    return out


def fetch_sp500_periods() -> dict:
    """fja05680 期間表 CSV（GitHub raw）；失敗回 {}。"""
    import requests
    try:
        r = requests.get(SP500_PERIODS_URL, timeout=20)
        if r.ok and r.text:
            return parse_period_csv(r.text)
    except Exception as e:
        print(f"universe: S&P 500 成分歷史抓取失敗 {e}")
    return {}


def rebuild(state: dict, today: str, cfg: dict | None = None, themes: dict | None = None,
            fetch_broad_fn=None, fetch_closes_fn=None, base_dir: Path | None = None) -> dict:
    """月頻重建：Stage 1 → Stage 2 → 快照落檔 + state["universe"] 摘要（輕量）。"""
    c = {**DEFAULTS, **(cfg or {})}
    broad = (fetch_broad_fn or fetch_broad)(c)
    core = list(state.get("watchlist") or [])
    tickers = [r["ticker"] for r in broad]
    closes = (fetch_closes_fn or fetch_closes_batch)(tickers) if tickers else {}
    mom = momentum_metrics(closes)
    top, stats = stage2_rank(broad, mom, c)
    snap = build_snapshot(today, broad, top, stats, themes, core)
    path = save_snapshot(snap, base_dir)
    state["universe"] = {"as_of": snap["as_of"], "available_at": snap["available_at"],
                         "counts": snap["counts"], "top": [r["ticker"] for r in top], "file": path.name}
    return snap


THEME_KEYS = ("AI光學/CPO/矽光子", "AI連接/Serdes", "化合物半導體/基板", "HBM/記憶儲存", "AI電力/散熱")


def theme_map() -> dict[str, list[str]]:
    """stock_db 美股 AI 供應鏈瓶頸主題 → {主題: [代碼]}；stock_db 缺席回 {}。"""
    try:
        import stock_db
        us = (getattr(stock_db, "ADB", {}) or {}).get("US", {})
        return {k: list(v.get("tickers") or []) for k, v in us.items()
                if k in THEME_KEYS or k.startswith("AI")}
    except Exception:
        return {}


# ── 6. 自我測試（離線）─────────────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import tempfile

    # 1) screen 解析 + Stage 1 門檻/去重/樣式
    payload = {"quotes": [
        {"symbol": "AAA", "marketCap": 5e9, "averageDailyVolume3Month": 2e6, "regularMarketPrice": 50, "sector": "Technology"},
        {"symbol": "AAA", "marketCap": 5e9, "averageDailyVolume3Month": 2e6, "regularMarketPrice": 50},          # 重複
        {"symbol": "SMALL", "marketCap": 5e8, "averageDailyVolume3Month": 2e6, "regularMarketPrice": 50},        # 市值不足
        {"symbol": "THIN", "marketCap": 5e9, "averageDailyVolume3Month": 1e5, "regularMarketPrice": 50},         # 量不足
        {"symbol": "PENNY", "marketCap": 5e9, "averageDailyVolume3Month": 2e6, "regularMarketPrice": 2},         # 價太低
        {"symbol": "BRK.B", "marketCap": 9e11, "averageDailyVolume3Month": 3e6, "regularMarketPrice": 400},      # 含點（screener 格式）
        {"symbol": "BANK", "marketCap": 9e10, "averageDailyVolume3Month": 3e6, "regularMarketPrice": 40, "sector": "Financial Services"},
        {"symbol": "NOMC", "averageDailyVolume3Month": 3e6, "regularMarketPrice": 40},
        {"notsymbol": 1}]}
    rows = parse_screen_quotes(payload)
    assert len(rows) == 8 and rows[0]["mcap"] == 5e9 and rows[-1]["mcap"] is None
    s1 = stage1_filter(rows)
    assert [r["ticker"] for r in s1] == ["AAA", "BANK"], [r["ticker"] for r in s1]
    assert parse_screen_quotes(None) == [] and parse_screen_quotes({"quotes": "x"}) == []
    print("✅ 1 Stage 1 解析與門檻（去重/市值/量/價/缺值）")

    # 2) 動能：無前視（竄改最後 5 日不改 12-1 動能）、樣本不足跳過
    idx = pd.bdate_range("2025-06-02", periods=300)
    rng = np.random.default_rng(3)
    up = pd.Series(100 * np.cumprod(1 + rng.normal(0.001, 0.01, 300)), index=idx)
    dn = pd.Series(100 * np.cumprod(1 + rng.normal(-0.001, 0.01, 300)), index=idx)
    short = up.iloc[-100:]
    mom = momentum_metrics({"UP": up, "DN": dn, "SHORT": short})
    assert "SHORT" not in mom and mom["UP"]["mom_12_1"] > mom["DN"]["mom_12_1"]
    up2 = up.copy(); up2.iloc[-5:] *= 0.5        # 壓低最近 5 日：現價變、52 週高不變
    assert abs(momentum_metrics({"UP": up2})["UP"]["mom_12_1"] - mom["UP"]["mom_12_1"]) < 1e-12   # 最後 21 日不進 12-1
    assert momentum_metrics({"UP": up2})["UP"]["off_high"] != mom["UP"]["off_high"]              # off_high 才會變
    print("✅ 2 動能指標（12-1 無近月前視、樣本門檻）")

    # 3) Stage 2：品質門檻（缺值不擋、金融另路由）、排序、top_n、統計
    base = {"mcap": 5e9, "avgvol": 2e6, "price": 50, "sector": "Technology"}
    cand = [{**base, "ticker": "UP", "roe": 0.2, "current_ratio": 1.5, "de": 0.5, "net_margin": 0.1},
            {**base, "ticker": "DN", "roe": 0.2},                                    # 缺值不擋
            {**base, "ticker": "LOWROE", "roe": 0.02},
            {**base, "ticker": "BANK", "sector": "Financial Services"},
            {**base, "ticker": "NOPX"}]
    top, stats = stage2_rank(cand, mom, {"uni_top_n": 1})
    assert [r["ticker"] for r in top] == ["UP"] and top[0]["rank"] == 1
    assert stats["passed_quality"] == 2 and stats["rejected"] == {"roe": 1, "sector_route": 1, "no_price_history": 1}, stats
    print("✅ 3 Stage 2 品質/排序/top_n")

    # 4) 成分歷史：期間表 + 每日清單兩種格式；members_on 邊界
    per = parse_period_csv("ticker,start_date,end_date\nAAA,2000-01-01,2010-06-30\nAAA,2015-01-01,\nBBB,1996-01-02,\nBRK.B,2010-02-16,\n")
    assert per["AAA"] == [("2000-01-01", "2010-06-30"), ("2015-01-01", "9999-12-31")]
    assert members_on(per, "2012-01-01") == ["BBB", "BRK-B"] and "AAA" in members_on(per, "2010-06-30")
    assert "AAA" in members_on(per, "2026-09-08") and members_on(per, "1990-01-01") == []
    daily = "date,tickers\n2020-01-02,\"AAA,BBB\"\n2020-01-03,\"AAA,BBB\"\n2020-01-06,\"AAA,CCC\"\n"
    per2 = parse_daily_list_csv(daily)
    assert per2["BBB"] == [("2020-01-02", "2020-01-03")] and per2["CCC"] == [("2020-01-06", "9999-12-31")]
    assert members_on(per2, "2020-01-06") == ["AAA", "CCC"]
    assert parse_period_csv("") == {} and parse_period_csv("x,y\n1,2\n") == {}
    print("✅ 4 成分歷史（期間表/每日清單/邊界）")

    # 5) 快照與 rebuild（假抓取器）：PIT 欄位、落檔、state 摘要、should_rebuild 月頻
    tmpd = Path(tempfile.mkdtemp())
    st = {"watchlist": ["UP", "ZZZ"], "thresholds": {}}
    snap = rebuild(st, "2026-09-04", {"uni_top_n": 5}, themes={"AI電力/散熱": ["VRT", "GEV"]},
                   fetch_broad_fn=lambda c: cand, fetch_closes_fn=lambda t: {"UP": up, "DN": dn}, base_dir=tmpd)
    assert snap["available_at"] == "2026-09-07" and snap["counts"]["broad"] == 5           # 週五 → 下週一
    assert (tmpd / "2026-09.json").exists() and st["universe"]["top"] == ["UP", "DN"]
    assert load_latest_snapshot(tmpd)["as_of"] == "2026-09-04"
    assert should_rebuild(st, "2026-09-20") is False and should_rebuild(st, "2026-10-01") is True
    assert should_rebuild({"universe": {}}, "2026-10-01", {"uni_enabled": False}) is False
    assert next_trading_day("2026-09-04") == "2026-09-07"    # 五→一
    print("✅ 5 快照/落檔/月頻重建判斷")

    # 6) 文字輸出 Markdown 安全
    t1, t2 = universe_text(snap), universe_text(None)
    for t in (t1, t2):
        assert t.count("*") % 2 == 0 and "_" not in t.replace("`/universe rebuild`", ""), t
    assert "候選前 10" in t1 and "未建立" in t2
    tm = theme_map(); assert isinstance(tm, dict)
    print(f"✅ 6 文字輸出（Markdown 安全）；stock_db AI 主題 {len(tm)} 組")
    print("\nuniverse selftest OK ✅")
