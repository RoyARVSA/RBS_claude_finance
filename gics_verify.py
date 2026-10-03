"""gics_verify.py – GICS 輪動數字對帳（移植自使用者 gics_nn 專案的 verify.py；由每週工作流執行，結果存 data/gics/verify.json）。

Checks (each prints PASS / WARN with the numbers behind it):

  1. Data integrity     every name has a price on the latest date; no impossible one-day jumps
  2. GICS labels        our sector for each stock vs the official GICS sector in State Street's SPY file
  3. Sector weights     our cap weights vs State Street's published SPY weights
  4. Whole index        all 503 members rebuilt cap-weighted vs the real S&P 500 (^GSPC)
  5. Sector returns     each rebuilt sector vs S&P's own sector index (^SP500-45 for IT, …)
  6. Spot checks        a few closes + Yahoo links so you can compare by eye

Returns in 4/5 are PRICE returns (no dividends) because ^GSPC and ^SP500-xx are price indices;
the dashboard itself uses dividend-adjusted prices for both sides, so its RS is unaffected.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

import gics_taxonomy as tx

SECTOR_INDEX = {"10": "^SP500-10", "15": "^SP500-15", "20": "^SP500-20", "25": "^SP500-25",
                "30": "^SP500-30", "35": "^SP500-35", "40": "^SP500-40", "45": "^SP500-45",
                "50": "^SP500-50", "55": "^SP500-55", "60": "^SP500-60"}
# used when Yahoo has no ^SP500-xx series for a sector (first one with data wins)
SECTOR_FALLBACK = {"10": ["^GSPE", "XLE"], "15": ["XLB"], "20": ["XLI"], "25": ["XLY"], "30": ["XLP"],
                   "35": ["XLV"], "40": ["XLF"], "45": ["XLK"], "50": ["XLC"], "55": ["XLU"], "60": ["XLRE"]}
WINDOWS = {"1M": 21, "3M": 63, "6M": 126, "12M": 252}
TOL_PP = {"1M": 0.5, "3M": 1.0, "6M": 1.5, "12M": 2.5}     # allowed gap in percentage points

OK, WARN, NOTE, SKIP = "PASS", "WARN", "NOTE", "SKIP"   # only WARN counts as a problem


# ------------------------------------------------------------- pure helpers
def cap_index(px: pd.DataFrame, shares: pd.Series, members: list[str]) -> pd.Series:
    """Same construction as the dashboard's Cap-wtd mode: daily return weighted by shares × previous close."""
    m = [t for t in members if t in px.columns]
    p = px[m]
    r = p.pct_change()
    w = p.shift(1).mul(shares.reindex(m), axis=1)
    w = w.where(r.notna())
    ret = (r * w).sum(axis=1) / w.sum(axis=1)
    return (1 + ret.fillna(0)).cumprod() * 100


def window_ret(s: pd.Series, n: int) -> float:
    s = s.dropna()
    if len(s) <= n:
        return np.nan
    return float(s.iloc[-1] / s.iloc[-1 - n] - 1)


def _norm_sector(x: str) -> str:
    return re.sub(r"[^a-z]", "", str(x).lower().replace("&", "and"))


_SECTOR_BY_NAME = {_norm_sector(tx.NAMES[c]): c for c in tx.codes_at(1)}
_SECTOR_BY_NAME.update({_norm_sector("Telecommunication Services"): "50", _norm_sector("Information Tech"): "45"})


def check_integrity(px: pd.DataFrame, tickers: list[str]):
    last = px.index.max()
    have = [t for t in tickers if t in px.columns]
    missing = [t for t in tickers if t not in px.columns or px[t].dropna().empty]
    stale = [t for t in have if t not in missing and px[t].last_valid_index() < last]
    real, bad_ticks = [], []
    r = px[have].pct_change().iloc[-260:]
    for t in have:
        for day in r.index[r[t].abs() > 0.5]:
            i = r.index.get_loc(day)
            nxt = r[t].iloc[i + 1] if i + 1 < len(r) else np.nan
            txt = f"{t} {day:%Y-%m-%d} {r[t].iloc[i]:+.0%}"
            # a price that jumps and comes straight back the next day is a bad print, not news
            (bad_ticks if np.isfinite(nxt) and abs((1 + r[t].iloc[i]) * (1 + nxt) - 1) < 0.1 else real).append(txt)
    rows = [
        ("prices present", OK if not missing else WARN, f"{len(tickers) - len(missing)}/{len(tickers)}",
         ", ".join(missing[:10])),
        (f"close on {last:%Y-%m-%d}", OK if not stale else WARN, f"{len(have) - len(missing) - len(stale)} up to date",
         ", ".join(stale[:10])),
        ("bad prints (spike & revert)", OK if not bad_ticks else WARN, f"{len(bad_ticks)} found", "; ".join(bad_ticks[:6])),
        ("one-day moves > 50%", OK if not real else NOTE, f"{len(real)} found",
         "; ".join(real[:6]) + (" — real moves stay in the data; check the news if unsure" if real else "")),
    ]
    return rows


def check_labels(uni: pd.DataFrame, spy: pd.DataFrame | None):
    if spy is None:
        return [("GICS sector vs SPY file", SKIP, "skipped", "SPY holdings file not available")], pd.DataFrame()
    ref = {t: _SECTOR_BY_NAME.get(_norm_sector(s)) for t, s in zip(spy["ticker"], spy["sector"])}
    if not any(ref.values()):
        sample = sorted(set(map(str, spy["sector"])))[:5]
        return [("GICS sector vs SPY file", SKIP, "skipped",
                 f"the SPY file has no GICS sector names (column values: {sample})")], pd.DataFrame()
    rows = []
    for t, c8 in zip(uni["ticker"], uni["code8"]):
        if t in ref and ref[t] and c8:
            rows.append((t, c8[:2], ref[t]))
    df = pd.DataFrame(rows, columns=["ticker", "ours", "official"])
    bad = df[df["ours"] != df["official"]]
    detail = ", ".join(f"{r.ticker} ours={tx.NAMES[r.ours]} / official={tx.NAMES[r.official]}" for r in bad.head(6).itertuples())
    status = OK if len(df) and bad.empty else WARN
    return [("GICS sector vs SPY file", status, f"{len(df) - len(bad)}/{len(df)} match", detail)], bad


def check_weights(uni: pd.DataFrame, shares: pd.Series, last_px: pd.Series, spy: pd.DataFrame | None):
    cap = (shares * last_px).dropna()
    sec = uni.set_index("ticker")["code8"].str[:2]
    ours = cap.groupby(sec.reindex(cap.index)).sum()
    ours = 100 * ours / ours.sum()
    if spy is None or spy["weight"].isna().all():
        return [("sector weights vs SPY", SKIP, "skipped", "SPY holdings file not available")], ours.to_frame("ours")
    w = spy.set_index("ticker")["weight"].reindex(cap.index).dropna()
    off = w.groupby(sec.reindex(w.index)).sum()
    off = 100 * off / off.sum()
    t = pd.DataFrame({"ours": ours, "official": off}).fillna(0)
    t["gap_pp"] = t["ours"] - t["official"]
    worst = t["gap_pp"].abs().idxmax()
    status = OK if t["gap_pp"].abs().max() <= 0.5 else WARN
    return [("sector weights vs SPY", status, f"max gap {t['gap_pp'].abs().max():.2f}pp",
             f"largest: {tx.NAMES[worst]} {t.loc[worst, 'ours']:.1f}% vs {t.loc[worst, 'official']:.1f}%")], t


def check_returns(px: pd.DataFrame, shares: pd.Series, uni: pd.DataFrame, bench: dict[str, pd.Series]):
    """bench: name -> (members, reference series). Returns table of our vs reference returns."""
    out = []
    for name, (members, ref, loose) in bench.items():
        if ref is None or ref.dropna().empty:
            out.append({"group": name, **{f"{k} ours": np.nan for k in WINDOWS}, "status": "no reference"})
            continue
        ours = cap_index(px, shares, members)
        row = {"group": name}
        worst = 0.0
        for k, n in WINDOWS.items():
            a, b = window_ret(ours, n), window_ret(ref.reindex(ours.index).ffill(), n)
            row[f"{k} ours"], row[f"{k} ref"] = a, b
            if np.isfinite(a) and np.isfinite(b):
                worst = max(worst, abs(a - b) * 100 / TOL_PP[k])
        row["status"] = OK if worst <= 1 else (NOTE if loose else WARN)
        out.append(row)
    return pd.DataFrame(out)


# ------------------------------------------------------------------- report
def _print_rows(title, rows):
    print(f"\n{title}")
    for name, st, val, det in rows:
        mark = {OK: "✓", WARN: "!", NOTE: "i", SKIP: "-"}[st]
        print(f"  {mark} {st:4s}  {name:26s} {val:22s} {det}")


def _print_returns(df: pd.DataFrame):
    print("\n4–5. Returns: rebuilt from members vs official index (price return, cap-weighted)")
    print(f"  {'':4s}  {'group':24s}" + "".join(f"{k:>17s}" for k in WINDOWS))
    for d in df.to_dict("records"):
        if d["status"] == "no reference":
            print(f"  ?       {d['group']:24s}  (reference index not available on Yahoo)")
            continue
        cells = "".join(f"{d[f'{k} ours'] * 100:+7.1f} vs {d[f'{k} ref'] * 100:+6.1f}" if np.isfinite(d[f'{k} ours'])
                        and np.isfinite(d[f'{k} ref']) else f"{'—':>17s}" for k in WINDOWS)
        mark = {OK: "✓", WARN: "!", NOTE: "i"}[d["status"]]
        print(f"  {mark} {d['status']:4s}  {d['group'][:24]:24s}{cells}")
    print("  tolerance: " + ", ".join(f"{k} ±{v}pp" for k, v in TOL_PP.items()) +
          ". 12M gaps also come from index adds/deletes during the year (dashboard uses today's members).\n"
          "  [XLx] = sector ETF used because Yahoo has no S&P sector index for it; ETFs cap big weights, "
          "so gaps there are NOTE, not WARN.")


def run(universe="sp500", csv=None, cache_dir=None, meta=None):
    from pathlib import Path
    from gics_data import (apply_corporate_actions, download_prices, fetch_meta, load_spinoffs, load_universe,
                           spy_holdings_table)
    cache = Path(cache_dir or "data")
    cache.mkdir(exist_ok=True)
    uni = load_universe(universe, csv)
    uni = uni[uni["code8"].notna()].reset_index(drop=True)
    tickers = uni["ticker"].tolist()

    spy = spy_holdings_table(cache / "spy_holdings.xlsx")
    if meta is None:                                   # 每週工作流會傳入已抓好的公司資料（避免重抓、避免寫回未精簡欄位）
        meta = fetch_meta(tickers, cache / "meta_cache.json")
    sh = {t: (meta.get(t) or {}).get("shares") for t in tickers}
    if spy is not None:
        sh.update({t: s for t, s in zip(spy["ticker"], spy["shares"]) if t in sh})
        src = "SPY official holdings"
    else:
        src = "Yahoo shares outstanding"
    shares = pd.Series(sh, dtype=float)

    refs = list(SECTOR_INDEX.values()) + ["^GSPC"]
    refs += [x for v in SECTOR_FALLBACK.values() for x in v]
    spinoffs = load_spinoffs(cache)
    spincos = [s for p, _, s, _ in spinoffs if p in tickers and s and s not in tickers]
    px = download_prices(tickers + spincos + refs, period="2y", adjust=False)
    px, ca_log = apply_corporate_actions(px, tickers, spinoffs)

    print("\n" + "=" * 100)
    print(f"VERIFY · {len(tickers)} names · weights from {src} · last close {px.index.max():%Y-%m-%d}")
    print("=" * 100)
    int_rows = check_integrity(px, tickers)
    int_rows += [("corporate action adjusted", NOTE, "", line) for line in ca_log]
    _print_rows("1. Data integrity", int_rows)
    lab_rows, bad = check_labels(uni, spy)
    _print_rows("2. GICS labels (Wikipedia list vs State Street)", lab_rows)
    w_rows, wtab = check_weights(uni, shares, px[tickers].ffill().iloc[-1], spy)
    _print_rows("3. Sector weights", w_rows)

    def has(sym):
        return sym in px.columns and px[sym].notna().sum() > 260

    bench = {"S&P 500 (all members)": (tickers, px.get("^GSPC"), False)}
    for c, sym in SECTOR_INDEX.items():
        members = uni.loc[uni["code8"].str[:2] == c, "ticker"].tolist()
        use = next((x for x in [sym] + SECTOR_FALLBACK.get(c, []) if has(x)), None)
        loose = use is not None and not use.startswith("^")
        label = tx.NAMES[c] + ("" if use in (sym, None) else f" [{use}]")
        bench[label] = (members, px.get(use) if use else None, loose)
    rets = check_returns(px, shares, uni, bench)
    _print_returns(rets)

    print("\n6. Spot checks — open the link, compare the last close")
    sample = ["NVDA", "AAPL", "JPM", "XOM"] + uni["ticker"].sample(2, random_state=None).tolist()
    for t in dict.fromkeys(sample):
        if t in px.columns and px[t].notna().any():
            s = px[t].dropna()
            print(f"  {t:6s} {s.index[-1]:%Y-%m-%d} close {s.iloc[-1]:>10.2f}   https://finance.yahoo.com/quote/{t}/history")

    out = cache / "verify_report.csv"
    rets.to_csv(out, index=False)
    if len(bad):
        bad.to_csv(cache / "verify_label_mismatches.csv", index=False)
    print(f"\nsaved → {out.resolve()}")
    warns = [f"{name}: {val}  {det}".rstrip() for name, st, val, det in int_rows + lab_rows + w_rows if st == WARN]
    for d in rets[rets["status"] == WARN].to_dict("records"):
        gaps = ", ".join(f"{k} {d[f'{k} ours'] * 100:+.1f} vs {d[f'{k} ref'] * 100:+.1f}"
                         for k in WINDOWS if np.isfinite(d[f"{k} ours"]) and np.isfinite(d[f"{k} ref"])
                         and abs(d[f"{k} ours"] - d[f"{k} ref"]) * 100 > TOL_PP[k])
        warns.append(f"returns {d['group']}: {gaps}")
    print("RESULT:", "all checks passed" if not warns else f"{len(warns)} warning(s):")
    for w in warns:
        print("  ! " + w)
    summary = summarize(int_rows + lab_rows + w_rows, rets, px.index.max())
    import json as _json
    (cache / "verify.json").write_text(_json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    return summary


def summarize(rows: list, rets: pd.DataFrame, last_date) -> dict:
    """對帳結果 → 小型 JSON（網頁角落與 /gics verify 顯示用；不含持倉）。"""
    checks = [{"name": n, "status": s, "value": v, "detail": d[:200]} for n, s, v, d in rows]
    ret_rows = []
    for d in rets.to_dict("records"):
        r = {"group": d["group"], "status": d["status"]}
        for k in WINDOWS:
            a, b = d.get(f"{k} ours"), d.get(f"{k} ref")
            r[k] = None if a is None or b is None or not (np.isfinite(a) and np.isfinite(b)) else [round(a, 4), round(b, 4)]
        ret_rows.append(r)
    n_warn = sum(1 for c in checks if c["status"] == WARN) + sum(1 for r in ret_rows if r["status"] == WARN)
    return {"as_of": str(pd.Timestamp(last_date).date()) if last_date is not None else None,
            "passed": n_warn == 0, "n_warn": n_warn, "checks": checks, "returns": ret_rows}


if __name__ == "__main__":
    # 純邏輯自測（不連網）：cap_index 與模板 Cap-wtd 同義、window_ret、check_integrity 的假尖刺判定、summarize
    idx = pd.bdate_range("2025-01-02", periods=300)
    px = pd.DataFrame({"A": np.linspace(10, 20, 300), "B": np.linspace(30, 15, 300)}, index=idx)
    sh = pd.Series({"A": 1e6, "B": 2e6})
    ci = cap_index(px, sh, ["A", "B"])
    r = px.pct_change(); w = px.shift(1).mul(sh, axis=1)
    exp = (1 + ((r * w).sum(axis=1) / w.sum(axis=1)).fillna(0)).cumprod() * 100
    assert np.allclose(ci.values, exp.values) and abs(window_ret(px["A"], 21) - (px["A"].iloc[-1] / px["A"].iloc[-22] - 1)) < 1e-12
    px2 = px.copy(); px2.iloc[200, 0] *= 3                      # 單日尖刺又回來 → bad print
    px2.iloc[250:, 1] *= 0.3                                     # 真的暴跌 → NOTE
    rows = check_integrity(px2, ["A", "B", "C"])
    st = {n: s for n, s, *_ in rows}
    assert st["prices present"] == WARN and st["bad prints (spike & revert)"] == WARN and st["one-day moves > 50%"] == NOTE
    uni = pd.DataFrame({"ticker": ["A", "B"], "code8": ["45301020", "10102010"]})
    spy = pd.DataFrame({"ticker": ["A", "B"], "shares": [1, 1], "weight": [60.0, 40.0],
                        "sector": ["Information Technology", "Health Care"]})
    lab, bad = check_labels(uni, spy)
    assert lab[0][1] == WARN and list(bad["ticker"]) == ["B"]
    rets = check_returns(px, sh, uni, {"All": (["A", "B"], ci, False)})
    sm = summarize(rows + lab, rets, idx[-1])
    assert sm["n_warn"] >= 3 and not sm["passed"] and sm["returns"][0]["status"] == OK
    print("gics_verify selftest OK ✅")
