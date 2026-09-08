"""
estimates_ledger.py – 分析師預估快照帳本（append-only）+ 修正動能 + SUE 歷史回填

背景（VALUATION_LANDSCAPE §4.1）：免費資料拿得到「現在的共識」與「財報日當下的共識」，
拿不到「逐日共識歷史」——修正動能（有文獻支持的中期領先因子）只能**從今天起自建**。
本模組每天把 yfinance 的 eps_trend / eps_revisions / 共識 / 目標價與 Finnhub 的評等
家數壓成一列存進 `estimates_ledger.json`（週頻列 + 最新完整快照），
並用 Alpha Vantage `EARNINGS`（免費 key、25 次/日）回填每季「公告日當下共識 vs 實際」
的 SUE 歷史（長壽股回溯至 1996）。

設計：
  • 檔案獨立於 state（公開資料衍生，明文；不進 SENSITIVE_KEYS），只在內容變動時寫
  • 列格式緊湊（list），每檔上限 ROW_CAP 週 ≈ 1.5 年；`latest` 存完整快照供顯示
  • 刷新走限額輪替（同 alpha_overlay）：每輪最多 refresh_per_run 檔、TTL 小時、最舊優先
  • Alpha Vantage 配額由程式計數（`av_quota`），無 key 或用罄一律略過、絕不報錯
  • 冷啟動：eps_trend 自帶 7/30/60/90 天回看 → 帳本累積前就能算 30/90 日修正
  • 修正動能只做「顯示 + 論點監測」；進部位要等 ≥ 6 個月自有歷史 + walk-forward

純邏輯（解析/追加/動能/文字）離線可測；抓取層需網路。教育用途，非投資建議。
"""

from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

ROW_CAP = 80              # 每檔週頻列上限（~1.5 年）
SUE_CAP = 40              # 每檔 SUE 季數上限（10 年）
AV_DAILY_QUOTA = 25       # Alpha Vantage 免費層
FAIL_TTL_HOURS = 3        # 抓取失敗檔的重試間隔（對抗驗證 M5）
DEFAULTS = {
    "est_enabled":          True,
    "est_refresh_per_run":  4,      # 每輪最多刷新幾檔（yfinance quoteSummary 1 檔 1 請求）
    "est_ttl_hours":        20,     # 快照壽命（每日一次）
    "est_sue_per_run":      3,      # 每輪最多幾次 Alpha Vantage 呼叫（配額 25/日）
    "est_sue_refresh_days": 100,    # SUE 歷史多久重抓一次（一季一次）
}
ROW_KEYS = ("d", "eps0y", "eps1y", "rev0y", "rev1y", "up30", "down30", "n", "tgt", "buy", "hold", "sell", "ya0y")


# ── 小工具 ────────────────────────────────────────────────────────────────

def _f(x):
    """安全轉 float（None/NaN/字串 'None' → None）。"""
    try:
        if x is None:
            return None
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _cell(df, row: str, col: str):
    """DataFrame 取格（列/欄缺 → None）。"""
    try:
        if df is None or getattr(df, "empty", True):
            return None
        if row not in df.index or col not in df.columns:
            return None
        return _f(df.loc[row, col])
    except Exception:
        return None


def _week_key(date_s: str) -> str:
    y, w, _ = datetime.strptime(date_s[:10], "%Y-%m-%d").isocalendar()
    return f"{y}-W{w:02d}"


# ── 1. 解析（純邏輯）────────────────────────────────────────────────────────

def snapshot_from_frames(eps_trend=None, eps_revisions=None, earnings_est=None,
                         revenue_est=None, targets: dict | None = None,
                         rec=None) -> dict:
    """
    yfinance 各 DataFrame / dict → 緊湊快照 dict（缺欄一律 None，不拋例外）。
    index 慣例：0q / +1q / 0y / +1y；rec 取 period=='0m' 或第一列。
    """
    snap = {"eps": {}, "rev": {}, "est": {}, "tgt": {}, "rec": {}}
    for p in ("0q", "+1q", "0y", "+1y"):
        snap["eps"][p] = {"cur": _cell(eps_trend, p, "current"),
                          "d7": _cell(eps_trend, p, "7daysAgo"),
                          "d30": _cell(eps_trend, p, "30daysAgo"),
                          "d60": _cell(eps_trend, p, "60daysAgo"),
                          "d90": _cell(eps_trend, p, "90daysAgo")}
        snap["rev"][p] = {"up7": _cell(eps_revisions, p, "upLast7days"),
                          "up30": _cell(eps_revisions, p, "upLast30days"),
                          "down7": _cell(eps_revisions, p, "downLast7days"),
                          "down30": _cell(eps_revisions, p, "downLast30days")}
        snap["est"][p] = {"eps_avg": _cell(earnings_est, p, "avg"),
                          "eps_n": _cell(earnings_est, p, "numberOfAnalysts"),
                          "eps_growth": _cell(earnings_est, p, "growth"),
                          "eps_year_ago": _cell(earnings_est, p, "yearAgoEps"),   # FY 滾動偵測
                          "rev_avg": _cell(revenue_est, p, "avg"),
                          "rev_n": _cell(revenue_est, p, "numberOfAnalysts"),
                          "rev_growth": _cell(revenue_est, p, "growth")}
    t = targets if isinstance(targets, dict) else {}
    snap["tgt"] = {k: _f(t.get(k)) for k in ("current", "low", "high", "mean", "median")}
    try:
        if rec is not None and not getattr(rec, "empty", True):
            r0 = rec.iloc[0]
            if "period" in rec.columns and (rec["period"] == "0m").any():
                r0 = rec[rec["period"] == "0m"].iloc[0]
            snap["rec"] = {k: _f(r0.get(k)) for k in ("strongBuy", "buy", "hold", "sell", "strongSell")}
    except Exception:
        snap["rec"] = {}
    return snap


def snapshot_row(snap: dict, date_s: str) -> list:
    """完整快照 → 週頻緊湊列（ROW_KEYS 順序）。"""
    e, r, s, t, c = snap.get("eps", {}), snap.get("rev", {}), snap.get("est", {}), snap.get("tgt", {}), snap.get("rec", {})
    g = lambda d, p, k: (d.get(p) or {}).get(k)
    return [date_s[:10],
            g(e, "0y", "cur") if g(e, "0y", "cur") is not None else g(s, "0y", "eps_avg"),
            g(e, "+1y", "cur") if g(e, "+1y", "cur") is not None else g(s, "+1y", "eps_avg"),
            g(s, "0y", "rev_avg"), g(s, "+1y", "rev_avg"),
            g(r, "0y", "up30"), g(r, "0y", "down30"), g(s, "0y", "eps_n"),
            t.get("mean"),
            (c.get("strongBuy") or 0) + (c.get("buy") or 0) if c else None,
            c.get("hold") if c else None,
            (c.get("sell") or 0) + (c.get("strongSell") or 0) if c else None,
            g(s, "0y", "eps_year_ago")]


# ── 2. 帳本（純邏輯）────────────────────────────────────────────────────────

def new_ledger() -> dict:
    return {"version": 1, "tickers": {}, "sue": {}, "av_quota": {"date": "", "used": 0}}


def load_ledger(path) -> dict:
    p = Path(path)
    if not p.exists():
        return new_ledger()
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(d, dict) or not isinstance(d.get("tickers"), dict):
            return new_ledger()
        d.setdefault("sue", {})
        d.setdefault("av_quota", {"date": "", "used": 0})
        return d
    except Exception:
        return new_ledger()


def save_ledger(path, ledger: dict) -> bool:
    """內容有變才寫（避免每輪產生無意義 commit）。回是否寫入。"""
    p = Path(path)
    new = json.dumps(ledger, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    try:
        if p.exists() and p.read_text(encoding="utf-8") == new:
            return False
    except Exception:
        pass
    p.write_text(new, encoding="utf-8")
    return True


def append_snapshot(ledger: dict, ticker: str, snap: dict, now_iso: str) -> None:
    """同一 ISO 週覆寫、跨週追加、超過 ROW_CAP 截舊；latest 永遠是最新完整快照。"""
    date_s = now_iso[:10]
    ent = ledger["tickers"].setdefault(ticker, {"rows": [], "latest": None, "ts": None})
    row = snapshot_row(snap, date_s)
    rows = ent["rows"]
    if rows and _week_key(rows[-1][0]) == _week_key(date_s):
        rows[-1] = row
    else:
        rows.append(row)
    if len(rows) > ROW_CAP:
        del rows[:-ROW_CAP]
    ent["latest"] = snap
    ent["ts"] = now_iso


def prune(ledger: dict, keep: list[str]) -> None:
    """移除已不在清單的標的（空清單不剪——一輪失敗不該炸掉帳本）。"""
    if not keep:
        return
    ks = set(keep)
    for k in [k for k in ledger["tickers"] if k not in ks]:
        del ledger["tickers"][k]
    for k in [k for k in ledger.get("sue", {}) if k not in ks]:
        del ledger["sue"][k]


# ── 3. 修正動能（純邏輯）─────────────────────────────────────────────────────

def _chg(cur, old):
    """共識變動 %：(cur − old)/|old|。雙負值時方向正確（-4→-2 為 +50%，對抗驗證 M1）；
    正負翻轉無意義回 None；cur 為 0 視為 −100%。"""
    if cur is None or old is None or abs(old) < 0.01:
        return None
    if cur != 0 and (cur > 0) != (old > 0):
        return None
    return (cur - old) / abs(old)


def _row_about_days_ago(rows: list, days: int, lo: int = 70, hi: int = 110):
    """挑「距最新列 days 天」的列：跨度須在 [lo, hi] 天內（缺口太大不硬算，對抗驗證 M3）。"""
    try:
        last = datetime.strptime(rows[-1][0], "%Y-%m-%d")
        best, best_gap = None, None
        for r in rows[:-1]:
            span = (last - datetime.strptime(r[0], "%Y-%m-%d")).days
            if lo <= span <= hi:
                gap = abs(span - days)
                if best is None or gap < best_gap:
                    best, best_gap = r, gap
        return best
    except Exception:
        return None


def _same_fy(a: list, b: list) -> bool:
    """兩列的本年度 yearAgoEps 相同 → 同一會計年度（FY 滾動後 0y 會換期別，對抗驗證 M2）。
    舊列缺此欄（升級前）視為不可比。"""
    ia = ROW_KEYS.index("ya0y")
    if len(a) <= ia or len(b) <= ia or a[ia] is None or b[ia] is None:
        return False
    return abs(a[ia] - b[ia]) < 1e-9


def revision_momentum(ent: dict | None, price: float | None = None) -> dict | None:
    """
    由 latest 快照（eps_trend 自帶 30/90 天回看）與帳本列（自有歷史）算：
      eps_chg_30d / eps_chg_90d（本年度 EPS 共識變動 %）、breadth_30d=(上修−下修)/分析師數、
      tgt_upside（目標價中值 vs 現價）、score ∈ [-1, 1]。
    自有歷史 ≥ 13 週時 90 日變動改用帳本（source=ledger），否則用 yfinance 回看。
    """
    if not isinstance(ent, dict) or not ent.get("latest"):
        return None
    snap = ent["latest"]
    e0 = (snap.get("eps") or {}).get("0y") or {}
    r0 = (snap.get("rev") or {}).get("0y") or {}
    s0 = (snap.get("est") or {}).get("0y") or {}
    cur = e0.get("cur") if e0.get("cur") is not None else s0.get("eps_avg")
    chg30, chg90, src = _chg(cur, e0.get("d30")), _chg(cur, e0.get("d90")), "yf_trend"
    rows = ent.get("rows") or []
    old_row = _row_about_days_ago(rows, 90) if len(rows) >= 2 else None
    if old_row is not None and rows[-1][1] is not None and old_row[1] is not None \
            and _same_fy(rows[-1], old_row):
        c90 = _chg(rows[-1][1], old_row[1])
        if c90 is not None:
            chg90, src = c90, "ledger"
    n = s0.get("eps_n") or 0
    up, down = r0.get("up30") or 0, r0.get("down30") or 0
    breadth = ((up - down) / n) if n else None
    tgt = (snap.get("tgt") or {}).get("median") or (snap.get("tgt") or {}).get("mean")
    px = price if price else (snap.get("tgt") or {}).get("current")
    upside = (tgt / px - 1) if (tgt and px) else None
    parts, w = [], []
    if chg90 is not None:
        parts.append(max(-1.0, min(1.0, chg90 / 0.10))); w.append(0.6)
    if breadth is not None:
        parts.append(max(-1.0, min(1.0, breadth))); w.append(0.4)
    score = (sum(p * ww for p, ww in zip(parts, w)) / sum(w)) if w else None
    return {"eps_chg_30d": chg30, "eps_chg_90d": chg90, "breadth_30d": breadth,
            "n": int(n) if n else 0, "up30": int(up), "down30": int(down),
            "tgt_upside": upside, "score": score, "source": src,
            "weeks": len(rows), "as_of": (ent.get("ts") or "")[:10]}


# ── 4. SUE 歷史（純邏輯）─────────────────────────────────────────────────────

def parse_av_earnings(payload: dict | None) -> list[list]:
    """Alpha Vantage EARNINGS → [[fiscalEnd, reportedDate, actual, estimate, surprise%], ...]
    最新在前；缺估計值的季度保留但 estimate=None。配額/錯誤訊息回 []。"""
    if not isinstance(payload, dict):
        return []
    q = payload.get("quarterlyEarnings")
    if not isinstance(q, list):
        return []
    out = []
    for r in q:
        if not isinstance(r, dict):
            continue
        out.append([str(r.get("fiscalDateEnding", ""))[:10], str(r.get("reportedDate", ""))[:10],
                    _f(r.get("reportedEPS")), _f(r.get("estimatedEPS")), _f(r.get("surprisePercentage"))])
    return out[:SUE_CAP]


def sue_stats(rows: list[list]) -> dict | None:
    """最近 8 季：beat 率、平均驚奇 %、最新 SUE（驚奇 / 過去驚奇 % 標準差）。"""
    good = [r for r in rows if r[2] is not None and r[3] is not None]
    if len(good) < 4:
        return None
    recent = good[:8]
    surp = [(r[2] - r[3]) for r in recent]
    pct = [r[4] for r in recent if r[4] is not None]
    hist = [(r[2] - r[3]) for r in good[1:13]]
    sd = (sum((x - sum(hist) / len(hist)) ** 2 for x in hist) / (len(hist) - 1)) ** 0.5 if len(hist) >= 4 else None
    sue = (surp[0] / sd) if (sd and sd > 1e-9) else None
    return {"n": len(recent), "beat_rate": sum(1 for x in surp if x > 0) / len(recent),
            "avg_surprise_pct": (sum(pct) / len(pct)) if pct else None,
            "last_surprise": surp[0], "last_sue": sue,
            "last_report": recent[0][1], "history_from": rows[-1][0]}   # 含無估計的早期季


def av_quota_left(ledger: dict, today: str) -> int:
    q = ledger.setdefault("av_quota", {"date": "", "used": 0})
    if q.get("date") != today:
        q["date"], q["used"] = today, 0
    return max(0, AV_DAILY_QUOTA - int(q.get("used") or 0))


def av_quota_use(ledger: dict, today: str, n: int = 1) -> None:
    q = ledger.setdefault("av_quota", {"date": today, "used": 0})
    if q.get("date") != today:
        q["date"], q["used"] = today, 0
    q["used"] = int(q.get("used") or 0) + n


# ── 5. 抓取層（需網路；全部失敗回 None）────────────────────────────────────

def fetch_yf_snapshot(ticker: str) -> dict | None:
    import yfinance as yf
    tk = yf.Ticker(ticker)

    def _get(name):
        try:
            v = getattr(tk, name)
            return v
        except Exception:
            return None

    frames = {n: _get(n) for n in ("eps_trend", "eps_revisions", "earnings_estimate",
                                    "revenue_estimate", "analyst_price_targets", "recommendations")}
    if all(v is None or (hasattr(v, "empty") and v.empty) for v in frames.values()):
        return None
    return snapshot_from_frames(frames["eps_trend"], frames["eps_revisions"],
                                frames["earnings_estimate"], frames["revenue_estimate"],
                                frames["analyst_price_targets"], frames["recommendations"])


def fetch_finnhub_rec(ticker: str, key: str) -> dict | None:
    """Finnhub /stock/recommendation（免費）：最新月份的評等家數。"""
    if not key:
        return None
    import requests
    try:
        r = requests.get("https://finnhub.io/api/v1/stock/recommendation",
                         params={"symbol": ticker, "token": key}, timeout=15)
        data = r.json() if r.ok else None
        if isinstance(data, list) and data:
            d0 = data[0]
            return {k: _f(d0.get(k)) for k in ("strongBuy", "buy", "hold", "sell", "strongSell")}
    except Exception:
        pass
    return None


def fetch_av_earnings(ticker: str, key: str) -> dict | None:
    """Alpha Vantage EARNINGS（免費 key、25 次/日）。回原始 payload（含配額提示時亦回，由呼叫端判讀）。"""
    if not key:
        return None
    import requests
    try:
        r = requests.get("https://www.alphavantage.co/query",
                         params={"function": "EARNINGS", "symbol": ticker, "apikey": key}, timeout=20)
        return r.json() if r.ok else None
    except Exception:
        return None


def refresh(ledger: dict, symbols: list[str], now_iso: str, cfg: dict | None = None,
            fetch_snap=None, fetch_rec=None, sleep_s: float = 1.2) -> list[str]:
    """
    限額輪替刷新：缺快照或過期者最舊優先，每輪最多 est_refresh_per_run 檔。
    fetch_snap / fetch_rec 可注入（測試）。回備註行（不含敏感內容）。
    """
    c = {**DEFAULTS, **(cfg or {})}
    symbols = list(dict.fromkeys(s for s in symbols if s))
    prune(ledger, symbols)
    now = datetime.fromisoformat(now_iso.replace("Z", "+00:00"))
    stale = []
    for s in symbols:
        ent = ledger["tickers"].get(s)
        ts = ent.get("ts") if ent else None
        try:
            age = (now - datetime.fromisoformat(str(ts).replace("Z", "+00:00"))).total_seconds() / 3600 if ts else 1e9
        except Exception:
            age = 1e9
        ttl = float(c["est_ttl_hours"])
        if ent and ent.get("latest") is None:
            ttl = min(ttl, FAIL_TTL_HOURS)          # 上次抓取失敗：短 TTL 提早重試（對抗驗證 M5）
        if age >= ttl:
            stale.append((s, age))
    stale.sort(key=lambda x: -x[1])
    picked = [s for s, _ in stale[: max(0, int(c["est_refresh_per_run"]))]]
    notes = []
    fs = fetch_snap or fetch_yf_snapshot
    fkey = os.environ.get("FINNHUB_API_KEY", "").strip()
    fr = fetch_rec or (lambda t: fetch_finnhub_rec(t, fkey))
    for i, sym in enumerate(picked):
        try:
            snap = fs(sym)
        except Exception as e:
            snap = None
            notes.append(f"{sym} 快照失敗 {type(e).__name__}")
        if snap is None:
            # 記一個空快照的時間戳（短 TTL 由呼叫端決定）避免同檔每輪重試佔名額
            ent = ledger["tickers"].setdefault(sym, {"rows": [], "latest": None, "ts": None})
            ent["ts"] = now_iso
            continue
        if not snap.get("rec"):
            try:
                rec = fr(sym)
                if rec:
                    snap["rec"] = rec
            except Exception:
                pass
        append_snapshot(ledger, sym, snap, now_iso)
        notes.append(f"{sym} 快照更新")
        if i < len(picked) - 1 and sleep_s > 0:
            time.sleep(sleep_s)
    return notes


def backfill_sue(ledger: dict, symbols: list[str], today: str, cfg: dict | None = None,
                 key: str | None = None, fetch=None) -> list[str]:
    """Alpha Vantage 配額內回填 SUE 歷史：無資料或超過 est_sue_refresh_days 者優先。"""
    c = {**DEFAULTS, **(cfg or {})}
    key = key if key is not None else os.environ.get("ALPHA_VANTAGE_KEY", "").strip()
    if not key:
        return []
    left = av_quota_left(ledger, today)
    budget = min(left, int(c["est_sue_per_run"]))
    if budget <= 0:
        return []
    sue = ledger.setdefault("sue", {})
    cand = []
    for s in dict.fromkeys(symbols):
        ent = sue.get(s)
        if not ent:
            cand.append((s, 1e9))
            continue
        try:
            age = (datetime.strptime(today, "%Y-%m-%d") - datetime.strptime(str(ent.get("ts", ""))[:10], "%Y-%m-%d")).days
        except Exception:
            age = 1e9
        if age >= int(c["est_sue_refresh_days"]):
            cand.append((s, age))
    cand.sort(key=lambda x: -x[1])
    f = fetch or (lambda t: fetch_av_earnings(t, key))
    notes = []
    for sym, _ in cand[:budget]:
        av_quota_use(ledger, today, 1)
        payload = f(sym)
        rows = parse_av_earnings(payload)
        if not rows:
            if isinstance(payload, dict) and any(k in payload for k in ("Note", "Information")):
                notes.append(f"{sym} SUE 配額提示，本輪停止")
                break                                   # 已達每日上限：不記 ts，明天再試
            # ETF/無效代碼/暫時失敗 → 負快取（記 ts、空 q），否則每天 25 次配額會被
            # watchlist 開頭的 SPY/QQQ 永遠吃光（對抗驗證 H1）；到期（est_sue_refresh_days）自動重試
            sue[sym] = {"ts": today, "q": []}
            notes.append(f"{sym} SUE 無資料（負快取）")
            continue
        sue[sym] = {"ts": today, "q": rows}
        notes.append(f"{sym} SUE {len(rows)} 季")
    return notes


# ── 6. 文字輸出（Telegram legacy Markdown：單 *、無底線）──────────────────────

def _pct(x, d=1):
    return "—" if x is None else f"{x:+.{d}%}"


def est_text(ticker: str, ledger: dict, price: float | None = None) -> str:
    ticker = str(ticker or "").strip().lstrip("$").upper().replace("_", "-")   # Markdown 安全 + $aapl
    ent = (ledger.get("tickers") or {}).get(ticker)
    if not ent or not ent.get("latest"):
        return (f"📐 *{ticker} 預估快照*\n尚無快照（每輪自動輪替刷新，或該代碼無分析師覆蓋）")
    snap, m = ent["latest"], revision_momentum(ent, price)
    s0, s1 = (snap.get("est") or {}).get("0y") or {}, (snap.get("est") or {}).get("+1y") or {}
    t, c = snap.get("tgt") or {}, snap.get("rec") or {}
    lines = [f"📐 *{ticker} 分析師預估*（{(ent.get('ts') or '')[:10]}，自有歷史 {len(ent.get('rows') or [])} 週）"]
    if s0.get("eps_avg") is not None:
        lines.append(f"本年 EPS 共識 {s0['eps_avg']:.2f}（{int(s0.get('eps_n') or 0)} 位，"
                     f"成長 {_pct(s0.get('eps_growth'))}）｜明年 {s1.get('eps_avg') or 0:.2f}"
                     f"（成長 {_pct(s1.get('eps_growth'))}）")
    if s0.get("rev_avg") is not None:
        lines.append(f"本年營收共識 {s0['rev_avg'] / 1e9:.2f}B（成長 {_pct(s0.get('rev_growth'))}）｜"
                     f"明年 {(s1.get('rev_avg') or 0) / 1e9:.2f}B")
    if m:
        lines.append(f"修正動能：30 日 {_pct(m['eps_chg_30d'])}｜90 日 {_pct(m['eps_chg_90d'])}"
                     f"（{'自有帳本' if m['source'] == 'ledger' else 'yfinance 回看'}）｜"
                     f"30 日上修 {m['up30']} / 下修 {m['down30']}"
                     + (f"（廣度 {m['breadth_30d']:+.2f}）" if m["breadth_30d"] is not None else ""))
        if m["score"] is not None:
            tag = "🟢 上修中" if m["score"] > 0.2 else ("🔴 下修中" if m["score"] < -0.2 else "🟡 持平")
            lines.append(f"動能分 {m['score']:+.2f} {tag}")
    if t.get("mean"):
        rng = (f"（{t['low']:.0f}–{t['high']:.0f}）" if t.get("low") and t.get("high") else "")
        lines.append(f"目標價 均 {t['mean']:.0f}／中 {(t.get('median') or t['mean']):.0f}{rng}"
                     + (f"｜距現價 {_pct(m['tgt_upside'])}" if m and m.get("tgt_upside") is not None else ""))
    if c:
        lines.append(f"評等：買 {int((c.get('strongBuy') or 0) + (c.get('buy') or 0))}｜"
                     f"持有 {int(c.get('hold') or 0)}｜賣 {int((c.get('sell') or 0) + (c.get('strongSell') or 0))}")
    st = sue_stats(((ledger.get("sue") or {}).get(ticker) or {}).get("q") or [])
    if st:
        lines.append(f"財報驚奇（近 {st['n']} 季）：beat {st['beat_rate']:.0%}｜均 {_pct((st['avg_surprise_pct'] or 0) / 100)}"
                     + (f"｜最新 SUE {st['last_sue']:+.1f}" if st.get("last_sue") is not None else "")
                     + f"｜歷史自 {st['history_from'][:4]}")
    lines.append("修正動能僅供顯示與論點監測，未進部位；非投資建議")
    return "\n".join(lines)


def movers_text(ledger: dict, symbols: list[str], top: int = 5) -> str:
    """watchlist 修正動能排行（上修/下修各 top）。"""
    rows = []
    for s in dict.fromkeys(symbols):
        m = revision_momentum((ledger.get("tickers") or {}).get(s))
        if m and m.get("score") is not None:
            rows.append((s, m))
    if not rows:
        return "📐 尚無足夠快照可排行（每輪輪替刷新中）"
    rows.sort(key=lambda x: -x[1]["score"])
    lines = [f"📐 *預估修正動能*（{len(rows)} 檔有覆蓋）", "*上修：*"]
    lines += [f"・{s} {m['score']:+.2f}（90 日 {_pct(m['eps_chg_90d'])}，上修 {m['up30']}/下修 {m['down30']}）"
              for s, m in rows[:top] if m["score"] > 0]
    lines.append("*下修：*")
    lines += [f"・{s} {m['score']:+.2f}（90 日 {_pct(m['eps_chg_90d'])}，上修 {m['up30']}/下修 {m['down30']}）"
              for s, m in rows[::-1][:top] if m["score"] < 0]
    lines.append("`/est TICKER` 看單檔；非投資建議")
    return "\n".join(lines)


# ── 7. 自我測試（離線）───────────────────────────────────────────────────────

if __name__ == "__main__":
    import pandas as pd
    idx = ["0q", "+1q", "0y", "+1y"]
    eps_trend = pd.DataFrame({"current": [1.0, 1.1, 5.0, 6.0], "7daysAgo": [1.0, 1.1, 4.95, 5.9],
                              "30daysAgo": [0.98, 1.08, 4.8, 5.7], "60daysAgo": [0.95, 1.0, 4.6, 5.5],
                              "90daysAgo": [0.9, 1.0, 4.5, 5.4]}, index=idx)
    eps_rev = pd.DataFrame({"upLast7days": [1, 0, 3, 2], "upLast30days": [4, 2, 12, 9],
                            "downLast7days": [0, 0, 1, 0], "downLast30days": [1, 1, 2, 1]}, index=idx)
    earn = pd.DataFrame({"numberOfAnalysts": [20, 20, 30, 28], "avg": [1.0, 1.1, 5.0, 6.0],
                         "low": [0.9, 1.0, 4.5, 5.2], "high": [1.1, 1.2, 5.5, 6.8],
                         "yearAgoEps": [0.8, 0.9, 4.0, 5.0], "growth": [0.25, 0.22, 0.25, 0.2]}, index=idx)
    rev = pd.DataFrame({"numberOfAnalysts": [18, 18, 25, 24], "avg": [1e9, 1.1e9, 4.4e9, 5.2e9],
                        "growth": [0.3, 0.28, 0.3, 0.18]}, index=idx)
    tgt = {"current": 100.0, "low": 90.0, "high": 150.0, "mean": 125.0, "median": 122.0}
    rec = pd.DataFrame({"period": ["0m", "-1m"], "strongBuy": [10, 9], "buy": [15, 14],
                        "hold": [5, 6], "sell": [1, 1], "strongSell": [0, 0]})

    # 1) 解析：完整、缺欄、全空
    snap = snapshot_from_frames(eps_trend, eps_rev, earn, rev, tgt, rec)
    assert snap["eps"]["0y"]["cur"] == 5.0 and snap["rev"]["0y"]["up30"] == 12
    assert snap["est"]["+1y"]["rev_avg"] == 5.2e9 and snap["tgt"]["median"] == 122.0
    assert snap["rec"]["strongBuy"] == 10
    empty = snapshot_from_frames(None, None, None, None, None, None)
    assert empty["eps"]["0y"]["cur"] is None and empty["rec"] == {} and empty["tgt"]["mean"] is None
    partial = snapshot_from_frames(eps_trend.drop(columns=["90daysAgo"]), None, earn.drop(index=["+1y"]), None, {"mean": "x"}, None)
    assert partial["eps"]["0y"]["d90"] is None and partial["est"]["+1y"]["eps_avg"] is None and partial["tgt"]["mean"] is None
    print("✅ 1 快照解析（完整 / 缺欄 / 全空 不拋例外）")

    # 2) 帳本：同週覆寫、跨週追加、cap、prune、save 只在變動時寫
    L = new_ledger()
    append_snapshot(L, "AAA", snap, "2026-09-07T13:00:00Z")
    append_snapshot(L, "AAA", snap, "2026-09-08T13:00:00Z")      # 同一 ISO 週 → 覆寫
    assert len(L["tickers"]["AAA"]["rows"]) == 1 and L["tickers"]["AAA"]["rows"][0][0] == "2026-09-08"
    append_snapshot(L, "AAA", snap, "2026-09-14T13:00:00Z")      # 下一週
    assert len(L["tickers"]["AAA"]["rows"]) == 2
    row = L["tickers"]["AAA"]["rows"][-1]
    assert len(row) == len(ROW_KEYS) and row[1] == 5.0 and row[5] == 12 and row[9] == 25
    for d in pd.date_range("2020-01-06", periods=ROW_CAP + 10, freq="7D"):
        append_snapshot(L, "BBB", snap, d.strftime("%Y-%m-%dT00:00:00Z"))
    assert len(L["tickers"]["BBB"]["rows"]) <= ROW_CAP
    prune(L, ["AAA"]); assert "BBB" not in L["tickers"]
    prune(L, []); assert "AAA" in L["tickers"]
    import tempfile
    tmp = Path(tempfile.mkdtemp()) / "led.json"
    assert save_ledger(tmp, L) is True and save_ledger(tmp, L) is False
    L2 = load_ledger(tmp); assert L2["tickers"]["AAA"]["latest"]["eps"]["0y"]["cur"] == 5.0
    assert load_ledger(tmp.parent / "nope.json")["tickers"] == {}
    tmp.write_text("{bad json", encoding="utf-8"); assert load_ledger(tmp)["tickers"] == {}
    print("✅ 2 帳本（同週覆寫 / 跨週 / cap / prune / 變動才寫 / 壞檔）")

    # 3) 修正動能：冷啟動用 yfinance 回看；≥13 週改用帳本；符號翻轉不算
    m = revision_momentum(L["tickers"]["AAA"], price=100.0)
    assert m["source"] == "yf_trend" and abs(m["eps_chg_90d"] - (5.0 / 4.5 - 1)) < 1e-9
    assert abs(m["breadth_30d"] - (12 - 2) / 30) < 1e-9 and abs(m["tgt_upside"] - 0.22) < 1e-9
    assert 0 < m["score"] <= 1
    L3 = new_ledger()
    for i, d in enumerate(pd.date_range("2026-01-05", periods=14, freq="7D")):
        s2 = json.loads(json.dumps(snap)); s2["eps"]["0y"]["cur"] = 4.0 + 0.1 * i
        append_snapshot(L3, "CCC", s2, d.strftime("%Y-%m-%dT00:00:00Z"))
    m3 = revision_momentum(L3["tickers"]["CCC"])
    assert m3["source"] == "ledger" and m3["eps_chg_90d"] > 0
    assert abs(_chg(-2.0, -4.0) - 0.5) < 1e-12 and abs(_chg(-4.0, -2.0) + 1.0) < 1e-12   # 雙負值方向（M1）
    assert _chg(0.0, 1.0) == -1.0 and _chg(1.0, -1.0) is None
    # FY 滾動（yearAgoEps 變）→ 不用帳本，退回 yf 回看（M2）；列缺口太大 → 不用帳本（M3）
    L4 = new_ledger()
    for i, d in enumerate(pd.date_range("2026-01-05", periods=14, freq="7D")):
        s4 = json.loads(json.dumps(snap)); s4["eps"]["0y"]["cur"] = 4.0 + 0.1 * i
        if i >= 10:
            s4["est"]["0y"]["eps_year_ago"] = 9.9              # 第 11 週起換會計年度
        append_snapshot(L4, "FY", s4, d.strftime("%Y-%m-%dT00:00:00Z"))
    assert revision_momentum(L4["tickers"]["FY"])["source"] == "yf_trend"
    L5 = new_ledger()
    append_snapshot(L5, "GAP", snap, "2026-01-05T00:00:00Z")
    append_snapshot(L5, "GAP", snap, "2026-09-07T00:00:00Z")   # 只有兩列、相距 8 個月
    assert revision_momentum(L5["tickers"]["GAP"])["source"] == "yf_trend"
    flip = json.loads(json.dumps(snap)); flip["eps"]["0y"]["d90"] = -0.5
    Lf = new_ledger(); append_snapshot(Lf, "F", flip, "2026-09-08T00:00:00Z")
    assert revision_momentum(Lf["tickers"]["F"])["eps_chg_90d"] is None
    assert revision_momentum(None) is None and revision_momentum({"latest": None}) is None
    print("✅ 3 修正動能（冷啟動 / 自有帳本 / 符號翻轉防護）")

    # 4) SUE：解析 + 統計 + 配額
    payload = {"symbol": "AAA", "quarterlyEarnings": [
        {"fiscalDateEnding": f"2026-{m_:02d}-30", "reportedDate": f"2026-{m_ + 1:02d}-01",
         "reportedEPS": str(1.0 + 0.05 * k), "estimatedEPS": str(0.95 + 0.05 * k),
         "surprisePercentage": "5.2"} for k, m_ in enumerate((9, 6, 3, 1))] + [
        {"fiscalDateEnding": "1996-03-31", "reportedDate": "1996-04-17", "reportedEPS": "0.1", "estimatedEPS": "None"}]}
    rows = parse_av_earnings(payload)
    assert len(rows) == 5 and rows[-1][3] is None and rows[0][2] == 1.0
    st = sue_stats(rows)
    assert st and st["n"] == 4 and st["beat_rate"] == 1.0 and st["history_from"] == "1996-03-31"
    assert parse_av_earnings({"Note": "quota"}) == [] and parse_av_earnings(None) == []
    Lq = new_ledger()
    assert av_quota_left(Lq, "2026-09-08") == AV_DAILY_QUOTA
    calls = {"n": 0}

    def fake_av(t):
        calls["n"] += 1
        return payload if t != "QUOTA" else {"Note": "limit"}
    notes = backfill_sue(Lq, ["AAA", "BBB", "CCC", "DDD"], "2026-09-08", {"est_sue_per_run": 2}, key="k", fetch=fake_av)
    assert calls["n"] == 2 and Lq["av_quota"]["used"] == 2 and "AAA" in Lq["sue"] and "CCC" not in Lq["sue"]
    notes2 = backfill_sue(Lq, ["AAA", "BBB", "CCC"], "2026-09-08", {"est_sue_per_run": 5}, key="k", fetch=fake_av)
    assert "CCC" in Lq["sue"] and calls["n"] == 3               # 已有的不重抓
    Lq["av_quota"] = {"date": "2026-09-08", "used": AV_DAILY_QUOTA}
    assert backfill_sue(Lq, ["ZZZ"], "2026-09-08", key="k", fetch=fake_av) == [] and calls["n"] == 3
    assert backfill_sue(Lq, ["ZZZ"], "2026-09-09", key="", fetch=fake_av) == []      # 無 key 略過
    n0 = calls["n"]; backfill_sue(Lq, ["QUOTA", "YYY"], "2026-09-10", {"est_sue_per_run": 5}, key="k", fetch=fake_av)
    assert calls["n"] == n0 + 1 and "QUOTA" not in Lq["sue"]        # 配額提示 → 本輪停止、不記 ts
    # H1：ETF/無效代碼負快取——不再每輪吃配額
    Lh = new_ledger(); seq = []

    def fake_bad(t):
        seq.append(t)
        return {"quarterlyEarnings": []} if t == "SPY" else ({"Error Message": "x"} if t == "BAD" else payload)
    for day in ("2026-09-11", "2026-09-12"):
        backfill_sue(Lh, ["SPY", "BAD", "OK1", "OK2"], day, {"est_sue_per_run": 3}, key="k", fetch=fake_bad)
    assert seq == ["SPY", "BAD", "OK1", "OK2"], seq                  # 第二天輪到 OK2，不重打 SPY/BAD
    assert Lh["sue"]["SPY"]["q"] == [] and "OK2" in Lh["sue"]
    print("✅ 4 SUE 解析/統計/配額（無 key 略過、用罄停止、配額提示中止）")

    # 5) refresh 輪替：TTL、最舊優先、限額、失敗不佔名額重試、rec 補抓
    Lr = new_ledger()
    got = []

    def fake_snap(t):
        got.append(t)
        if t == "BAD":
            raise RuntimeError("boom")
        s = json.loads(json.dumps(snap)); s["rec"] = {}
        return s
    fake_rec = lambda t: {"strongBuy": 1, "buy": 2, "hold": 3, "sell": 0, "strongSell": 0}
    n1 = refresh(Lr, ["AAA", "BAD", "CCC", "DDD", "EEE"], "2026-09-08T13:00:00Z",
                 {"est_refresh_per_run": 3}, fetch_snap=fake_snap, fetch_rec=fake_rec, sleep_s=0)
    assert got == ["AAA", "BAD", "CCC"] and Lr["tickers"]["AAA"]["latest"]["rec"]["hold"] == 3
    assert Lr["tickers"]["BAD"]["latest"] is None and Lr["tickers"]["BAD"]["ts"]
    got.clear()
    refresh(Lr, ["AAA", "BAD", "CCC", "DDD", "EEE"], "2026-09-08T13:05:00Z",
            {"est_refresh_per_run": 3}, fetch_snap=fake_snap, fetch_rec=fake_rec, sleep_s=0)
    assert got == ["DDD", "EEE"]                                   # 已刷新者在 TTL 內不重抓
    got.clear()
    refresh(Lr, ["AAA", "BAD", "CCC", "DDD", "EEE"], "2026-09-08T17:00:00Z",
            {"est_refresh_per_run": 3}, fetch_snap=fake_snap, fetch_rec=fake_rec, sleep_s=0)
    assert got == ["BAD"], got                                     # 失敗檔 3h 後重試（M5），成功檔仍在 TTL 內
    got.clear()
    refresh(Lr, ["AAA", "BAD"], "2026-09-09T13:00:00Z", {"est_refresh_per_run": 3},
            fetch_snap=fake_snap, fetch_rec=fake_rec, sleep_s=0)
    assert got == ["AAA", "BAD"] and "CCC" not in Lr["tickers"]   # 過期重抓 + prune
    print("✅ 5 輪替刷新（TTL / 限額 / 失敗記時間戳 / prune）")

    # 6) 文字輸出 Markdown 安全
    Lr["sue"]["AAA"] = {"ts": "2026-09-08", "q": rows}
    t1 = est_text("$aaa", Lr, price=100.0); t2 = movers_text(Lr, ["AAA", "DDD", "EEE", "ZZZ"]); t3 = est_text("foo_bar", Lr)
    for t in (t1, t2, t3):
        assert t.count("*") % 2 == 0 and "_" not in t.replace("`/est TICKER`", ""), t
    assert "修正動能" in t1 and "財報驚奇" in t1 and "尚無快照" in t3
    print("✅ 6 文字輸出（Markdown 安全）")
    print("\nestimates_ledger selftest OK ✅")
