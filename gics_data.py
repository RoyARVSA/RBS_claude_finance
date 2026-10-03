"""gics_data.py – GICS 輪動用的資料擷取 + 成分股 GICS 標籤（移植自使用者 gics_nn 專案）。

- 成分股與 GICS Sub-Industry 標籤：Wikipedia 上的 S&P 500 (/400/600) 清單，或你自己的 CSV
- 價格：yfinance 日收盤 (auto-adjusted)
- 流通股數 / 公司描述 / Yahoo sector-industry：yfinance .info (有快取，第二次跑很快)
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

import gics_taxonomy as tx

WIKI = {
    "sp500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "sp400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    "sp600": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
}
UA = {"User-Agent": "Mozilla/5.0 (gics-nn research script)"}


def yahoo_symbol(sym: str) -> str:
    return sym.strip().upper().replace(".", "-")


# ------------------------------------------------------------------ universe
def _wiki_table(url: str) -> pd.DataFrame:
    import requests
    html = requests.get(url, headers=UA, timeout=30).text
    for t in pd.read_html(StringIO(html)):
        cols = {str(c).lower(): c for c in t.columns}
        sym = next((cols[c] for c in cols if c in ("symbol", "ticker symbol", "ticker")), None)
        sub = next((cols[c] for c in cols if "sub-industry" in c or "sub industry" in c), None)
        name = next((cols[c] for c in cols if c in ("security", "company")), None)
        if sym is not None and sub is not None:
            return pd.DataFrame({"ticker": t[sym].astype(str), "name": t[name] if name is not None else t[sym],
                                 "sub_industry": t[sub].astype(str)})
    raise RuntimeError(f"no constituent table found at {url}")


def load_universe(which: str = "sp500", csv: str | None = None) -> pd.DataFrame:
    """Return DataFrame[ticker, name, sub_industry, code8, index]. code8 may be NaN if unmatched."""
    if csv:
        df = pd.read_csv(csv)
        df.columns = [c.lower() for c in df.columns]
        if "code8" not in df.columns:
            df["code8"] = df.get("sub_industry", pd.Series([""] * len(df))).map(tx.match_sub_industry)
        df["code8"] = df["code8"].map(lambda c: str(int(c)) if pd.notna(c) and str(c).strip() else None)
        df["index"] = "custom"
    else:
        parts = []
        for key in (["sp500", "sp400", "sp600"] if which == "sp1500" else [which]):
            d = _wiki_table(WIKI[key])
            d["index"] = key
            parts.append(d)
        df = pd.concat(parts, ignore_index=True)
        df["code8"] = df["sub_industry"].map(tx.match_sub_industry)
    df["ticker"] = df["ticker"].map(yahoo_symbol)
    df = df.drop_duplicates("ticker").reset_index(drop=True)
    miss = df[df["code8"].isna()]
    if len(miss):
        print(f"[universe] {len(miss)} names with unmatched sub-industry label "
              f"(will be classified by the NN if a model exists): "
              f"{sorted(set(miss['sub_industry']))[:8]}")
    return df


# -------------------------------------------------------------------- prices
def download_prices(tickers: list[str], period: str = "2y", chunk: int = 100, adjust: bool = True) -> pd.DataFrame:
    import yfinance as yf
    frames = []
    for i in range(0, len(tickers), chunk):
        part = tickers[i:i + chunk]
        print(f"[prices] {i + len(part)}/{len(tickers)}")
        d = yf.download(part, period=period, auto_adjust=adjust, progress=False, threads=True, group_by="column")
        c = d["Close"] if isinstance(d.columns, pd.MultiIndex) else d[["Close"]].rename(columns={"Close": part[0]})
        frames.append(c)
        time.sleep(0.5)
    px = pd.concat(frames, axis=1)
    px.index = pd.to_datetime(px.index).tz_localize(None)
    return px.sort_index()


# ---------------------------------------------------------------------- meta
def _is_empty(m: dict | None) -> bool:
    return not m or (not m.get("summary") and not m.get("shares"))


def fetch_meta(tickers: list[str], cache_path: Path, refresh: bool = False, workers: int = 3,
               passes: int = 3, budget_s: float | None = None) -> dict:
    """Fetch shares / description / Yahoo sector per ticker, with a JSON cache.

    Failed or empty responses (401 Invalid Crumb, 429 rate limit, 404 delisted) are NOT cached,
    so re-running the command simply resumes with the missing names. Empty entries left by an
    older version of this script are treated as missing too.
    """
    import yfinance as yf
    cache = json.loads(cache_path.read_text()) if cache_path.exists() and not refresh else {}
    cache = {k: v for k, v in cache.items() if not _is_empty(v)}

    def one(t):
        for attempt in range(3):
            try:
                info = yf.Ticker(t).info or {}
                m = {
                    "name": info.get("shortName") or info.get("longName") or t,
                    "shares": info.get("sharesOutstanding") or info.get("impliedSharesOutstanding"),
                    "market_cap": info.get("marketCap"),
                    "summary": info.get("longBusinessSummary", ""),
                    "yf_sector": info.get("sector", ""),
                    "yf_industry": info.get("industry", ""),
                }
                if not _is_empty(m):
                    return t, m
            except Exception:
                pass
            time.sleep(2.0 * (attempt + 1))
        return t, None

    t_start = time.monotonic()
    for p in range(passes):
        todo = [t for t in tickers if t not in cache]
        if not todo:
            break
        if budget_s is not None and time.monotonic() - t_start > budget_s:
            print(f"[meta] time budget reached; {len(todo)} tickers left for next run")
            break
        if p:
            wait = 60 * p
            print(f"[meta] {len(todo)} tickers failed (Yahoo rate limit / crumb). waiting {wait}s, retry pass {p + 1}/{passes}")
            time.sleep(wait)
        print(f"[meta] fetching {len(todo)} tickers (cached: {len(cache)})")
        fails_in_row = 0
        if budget_s is not None:                         # 本輪也受預算限制：只送出估計做得完的量（每檔約 1–2 秒）
            left = max(0.0, budget_s - (time.monotonic() - t_start))
            todo = todo[:max(1, int(left / 1.5 * workers))]
        with ThreadPoolExecutor(workers) as ex:
            for n, fut in enumerate(as_completed([ex.submit(one, t) for t in todo]), 1):
                t, m = fut.result()
                if m is None:
                    fails_in_row += 1
                else:
                    cache[t] = m
                    fails_in_row = 0
                if fails_in_row == 15:
                    print("[meta] many consecutive failures — Yahoo is throttling; pausing 45s")
                    time.sleep(45)
                if n % 50 == 0:
                    print(f"[meta] {n}/{len(todo)}")
                    cache_path.write_text(json.dumps(cache))
        cache_path.write_text(json.dumps(cache))
    left = [t for t in tickers if t not in cache]
    if left:
        print(f"[meta] {len(left)} tickers still without data (delisted/renamed or throttled): {left[:10]}"
              f"{' …' if len(left) > 10 else ''}  — re-run later to fill them in; the run continues without them.")
    return cache


def sector_etf_corr(prices: pd.DataFrame, etf_px: pd.DataFrame, tickers: list[str], window: int = 252) -> np.ndarray:
    """Correlation of each stock's daily return with the 11 SPDR sector ETFs (price-behaviour features)."""
    from gics_model import FeatureBuilder
    r = prices.pct_change().iloc[-window:]
    e = etf_px.reindex(columns=FeatureBuilder.SECTOR_ETFS).pct_change().iloc[-window:]
    out = np.zeros((len(tickers), len(FeatureBuilder.SECTOR_ETFS)))
    for i, t in enumerate(tickers):
        if t in r.columns:
            s = r[t]
            out[i] = [s.corr(e[c]) if c in e else 0.0 for c in e.columns]
    return np.nan_to_num(out)


# ------------------------------------------------------- SPY official holdings
SPY_HOLDINGS_URL = ("https://www.ssga.com/us/en/intermediary/library-content/products/"
                    "fund-data/etfs/us/holdings-daily-us-en-spy.xlsx")


def spy_holdings_table(cache_path: Path, quiet: bool = False):
    """State Street daily SPY holdings → DataFrame[ticker, shares, weight, sector], or None.

    SPY holds each stock in proportion to its float-adjusted S&P 500 index weight, so
    'Shares Held' × price gives the same relative weights as the official index — including the
    correct split between dual-class lines (GOOGL/GOOG, FOXA/FOX, NWSA/NWS).
    The 'Sector' column is the official GICS sector, used by `verify` to cross-check labels.
    """
    import requests
    from datetime import date
    fresh = cache_path.exists() and date.fromtimestamp(cache_path.stat().st_mtime) == date.today()
    try:
        if fresh:
            raise StopIteration                       # already downloaded today
        r = requests.get(SPY_HOLDINGS_URL, headers=UA, timeout=30)
        r.raise_for_status()
        cache_path.write_bytes(r.content)
    except StopIteration:
        pass
    except Exception as e:
        if not cache_path.exists():
            print(f"[spy] could not download SPY holdings ({e}); using Yahoo shares outstanding")
            return None
        print(f"[spy] download failed ({e}); using cached {cache_path.name}")
    try:
        raw = pd.read_excel(cache_path, header=None)
        hdr = next(i for i in range(min(15, len(raw))) if "Ticker" in raw.iloc[i].astype(str).tolist())
        df = raw.iloc[hdr + 1:].copy()
        df.columns = raw.iloc[hdr].astype(str).str.strip()
        df["shares"] = pd.to_numeric(df.get("Shares Held"), errors="coerce")
        df = df[df["Ticker"].notna() & df["shares"].notna() & (df["shares"] > 0)]
        df = df[~df["Ticker"].astype(str).str.strip().isin(["", "-", "nan"])]
        out = pd.DataFrame({
            "ticker": df["Ticker"].astype(str).map(yahoo_symbol),
            "shares": df["shares"].astype(float),
            "weight": pd.to_numeric(df.get("Weight"), errors="coerce") if "Weight" in df else np.nan,
            "sector": df["Sector"].astype(str).str.strip() if "Sector" in df else "",
        }).drop_duplicates("ticker").reset_index(drop=True)
    except ImportError:
        print("[spy] openpyxl is not installed → using Yahoo shares outstanding. "
              "Fix: python -m pip install openpyxl")
        return None
    except Exception as e:                       # file format changed etc. — never block the build
        print(f"[spy] could not read SPY holdings ({type(e).__name__}: {e}); using Yahoo shares outstanding")
        return None
    if not quiet:
        print(f"[spy] official SPY holdings: {len(out)} names")
    return out


def spy_holdings_shares(cache_path: Path) -> dict:
    t = spy_holdings_table(cache_path)
    return {} if t is None else dict(zip(t["ticker"], t["shares"]))


# ------------------------------------------------------------ corporate actions
# Spin-offs: Yahoo does not adjust the parent's price history, so on the ex-date the parent
# shows a huge one-day "loss" that shareholders never had (they received the new shares).
# (parent, ex-date, spun-off ticker, new shares per parent share)
KNOWN_SPINOFFS = [
    ("CTVA", "2026-10-01", "VYLR", 1.0),     # Corteva → Vylor (seed business), 1:1
]
AUTO_DROP = -0.50     # any other one-day drop worse than this is treated as an unlisted corporate action


def load_spinoffs(cache_dir: Path) -> list[tuple]:
    """Built-in list + optional data/corporate_actions.csv (columns: parent,date,spinco,ratio)."""
    ev = list(KNOWN_SPINOFFS)
    f = Path(cache_dir) / "corporate_actions.csv"
    if f.exists():
        d = pd.read_csv(f)
        ev += [(yahoo_symbol(r.parent), str(r.date), yahoo_symbol(str(r.spinco)) if pd.notna(r.spinco) else "",
                float(r.ratio) if pd.notna(r.ratio) else 0.0) for r in d.itertuples()]
    return ev


def apply_corporate_actions(px: pd.DataFrame, members: list[str], spinoffs: list[tuple]):
    """Back-adjust parent prices for spin-offs so the ex-date return = parent + spun-off shares.

    factor = P_parent(ex) / (P_parent(ex) + ratio × P_spinco(ex)) multiplies all earlier prices,
    the same method index providers use. Unknown drops worse than AUTO_DROP are neutralised
    (return set to 0). Returns (adjusted prices, list of log lines).
    """
    px = px.copy()
    log, done = [], set()
    for parent, day, spin, ratio in spinoffs:
        if parent not in px.columns:
            continue
        day = pd.Timestamp(day)
        if day not in px.index:
            continue
        i = px.index.get_loc(day)
        p_prev, p_ex = px[parent].iloc[i - 1], px[parent].iloc[i]
        if not (p_prev > 0 and p_ex > 0) or p_ex / p_prev - 1 > -0.25:
            continue                                           # nothing to fix (or already adjusted upstream)
        s_ex = px[spin].iloc[i] if spin in px.columns else np.nan
        if ratio > 0 and s_ex > 0:
            f = p_ex / (p_ex + ratio * s_ex)
            how = f"spin-off {spin} {ratio:g}:1"
        else:
            f = p_ex / p_prev
            how = f"spin-off {spin or '?'} (no price → return set to 0)"
        px.loc[px.index < day, parent] *= f
        done.add((parent, day))
        log.append(f"{parent} {day:%Y-%m-%d}: raw {p_ex / p_prev - 1:+.0%} → {px[parent].iloc[i] / px[parent].iloc[i - 1] - 1:+.1%} ({how})")
    r = px[[m for m in members if m in px.columns]].pct_change()
    for t in r.columns:
        for day in r.index[r[t] < AUTO_DROP]:
            if (t, day) in done:
                continue
            i = px.index.get_loc(day)
            raw = px[t].iloc[i] / px[t].iloc[i - 1] - 1
            px.loc[px.index < day, t] *= px[t].iloc[i] / px[t].iloc[i - 1]
            log.append(f"{t} {day:%Y-%m-%d}: raw {raw:+.0%} → +0.0% (unlisted corporate action? check news; "
                       f"add it to data/corporate_actions.csv if it is a spin-off)")
    return px, log


# ------------------------------------------------------------ point-in-time membership
def membership_mask(index: pd.DatetimeIndex, tickers: list[str], periods: dict | None) -> pd.DataFrame | None:
    """S&P 500 成分期間表（universe.parse_period_csv 格式）→ 布林遮罩（True = 當天在指數內）。
    期間表沒有的代碼一律 True（不知道就不排除）。periods 為空 → None（不套用）。
    用途：把「加入指數之前」的日子排除在群組指數外，降低「今天的成分套回過去」的存活者偏差。"""
    if not periods:
        return None
    m = pd.DataFrame(True, index=index, columns=tickers)
    for t in tickers:
        ps = periods.get(t)
        if not ps:
            continue
        inside = pd.Series(False, index=index)
        for s, e in ps:
            inside |= (index >= pd.Timestamp(s)) & (index <= pd.Timestamp(e))
        m[t] = inside.values
    return m


# ------------------------------------------------------------------ selftest
if __name__ == "__main__":
    import tempfile
    # 1) 代碼正規化、CSV 宇宙（sub_industry 名稱 → code8）
    assert yahoo_symbol(" brk.b ") == "BRK-B"
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "u.csv"
        p.write_text("ticker,name,sub_industry\nNVDA,NVIDIA,Semiconductors\nXOM,Exxon,Integrated Oil & Gas\nZZZ,Unknown,Widgets\n")
        u = load_universe(csv=str(p))
        assert u.set_index("ticker").loc["NVDA", "code8"] == "45301020" and u.set_index("ticker").loc["XOM", "code8"] == "10102010"
        assert u.set_index("ticker")["code8"].isna().sum() == 1
        # 2) SPY 持股檔解析（合成 xlsx：前幾列是標頭說明，之後才是表頭）
        x = Path(td) / "spy.xlsx"
        raw = pd.DataFrame([["Fund Name:", "SPDR"], ["", ""], ["Name", "Ticker", "Identifier", "SEDOL", "Weight", "Sector", "Shares Held"]], dtype=object)
        rows = pd.DataFrame([["NVIDIA", "NVDA", "", "", 7.1, "Information Technology", 1000.0],
                             ["Berkshire", "BRK.B", "", "", 1.6, "Financials", 50.0],
                             ["Cash", "-", "", "", 0.1, "", 1.0]], dtype=object)
        pd.concat([raw, rows], ignore_index=True).to_excel(x, header=False, index=False)
        import os
        os.utime(x, None)                                       # 今天的檔 → 不重新下載
        tb = spy_holdings_table(x, quiet=True)
        assert tb is not None and set(tb["ticker"]) == {"NVDA", "BRK-B"} and float(tb.set_index("ticker").loc["NVDA", "shares"]) == 1000.0
        assert spy_holdings_shares(x)["BRK-B"] == 50.0
    # 3) 分拆修正：母公司 ex-date 假暴跌 → 回溯調整後該日報酬 ≈ 母+子；未列名的 >50% 暴跌中性化
    idx = pd.bdate_range("2026-09-28", periods=6)
    px = pd.DataFrame({"CTVA": [100, 101, 102, 40, 41, 42.0], "VYLR": [np.nan, np.nan, np.nan, 61, 60, 61.0],
                       "BAD": [50, 50, 50, 10, 10, 10.0], "OK": [10, 11, 12, 13, 14, 15.0]}, index=idx)
    adj, log = apply_corporate_actions(px, ["CTVA", "BAD", "OK"], [("CTVA", str(idx[3].date()), "VYLR", 1.0)])
    r_ex = adj["CTVA"].iloc[3] / adj["CTVA"].iloc[2] - 1
    assert abs(r_ex - ((40 + 61) / 102 - 1)) < 1e-9, r_ex                     # ex 日報酬 = (母+子)/前收 − 1
    assert abs(adj["BAD"].iloc[3] / adj["BAD"].iloc[2] - 1) < 1e-12             # 未列名暴跌 → 0
    assert (adj["OK"] == px["OK"]).all() and len(log) == 2
    # 4) 成分期間遮罩
    mm = membership_mask(idx, ["A", "B"], {"A": [(str(idx[2].date()), "9999-12-31")]})
    assert mm["A"].tolist() == [False, False, True, True, True, True] and mm["B"].all()
    assert membership_mask(idx, ["A"], {}) is None
    print("gics_data selftest OK ✅")
