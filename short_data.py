"""
short_data.py – 做空籌碼數據（FINRA 日做空量 + yfinance 短倉 + SEC 失券 FTD）

免 key 官方資料源（端點取自 OpenBB provider 原始碼）：
  · FINRA 日做空量  https://cdn.finra.org/equity/regsho/daily/CNMSshvolYYYYMMDD.txt
    （pipe 分隔：Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market）
  · SEC 失券資料    https://www.sec.gov/files/data/fails-deliver-data/cnsfails{YYYYMM}{a|b}.zip
  · 短倉餘額（雙週）走 yfinance .info（sharesShort / shortPercentOfFloat / shortRatio）
解析層為純函數（離線可測）；抓取層需網路。註：日做空量為場外(TRF/ADF/ORF)成交的
做空「量」，不是短倉餘額——高比率＝當日賣壓中做空占比高，解讀需搭配趨勢。非投資建議。
"""

from __future__ import annotations

import io
import os
import zipfile

FINRA_DAILY = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{ymd}.txt"
SEC_FTD = "https://www.sec.gov/files/data/fails-deliver-data/cnsfails{ym}{half}.zip"


# ── 純解析 ─────────────────────────────────────────────────────────────────────

def parse_short_volume(text: str, tickers: set | None = None) -> dict:
    """
    解析 FINRA 日做空量檔 → {symbol: {"short_vol", "total_vol", "ratio"}}。
    tickers 給定時只留那些代碼（省記憶體）。壞行直接跳過。
    """
    out: dict = {}
    for line in (text or "").splitlines():
        parts = line.split("|")
        if len(parts) < 5 or parts[0] == "Date":
            continue
        sym = parts[1].strip().upper()
        if tickers and sym not in tickers:
            continue
        try:
            sv, tv = float(parts[2]), float(parts[4])
        except ValueError:
            continue
        if tv <= 0:
            continue
        prev = out.get(sym)
        if prev:                       # 同代碼多列（不同市場）→ 加總
            sv += prev["short_vol"]
            tv += prev["total_vol"]
        out[sym] = {"short_vol": sv, "total_vol": tv, "ratio": sv / tv}
    return out


def parse_ftd(zip_bytes: bytes, tickers: set | None = None) -> dict:
    """
    解析 SEC 失券 zip（內含 pipe 分隔 txt：SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY|DESCRIPTION|PRICE）
    → {symbol: [{"date","qty","price"}...]}（每檔留最近 10 筆）。
    """
    out: dict = {}
    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
        name = zf.namelist()[0]
        text = zf.read(name).decode("utf-8", errors="replace")
    except Exception:
        return {}
    for line in text.splitlines():
        parts = line.split("|")
        if len(parts) < 6 or not parts[0][:1].isdigit():
            continue
        sym = parts[2].strip().upper()
        if tickers and sym not in tickers:
            continue
        try:
            qty = float(parts[3])
        except ValueError:
            continue
        try:
            price = float(parts[5])
        except ValueError:
            price = None
        out.setdefault(sym, []).append(
            {"date": parts[0].strip(), "qty": qty, "price": price})
    for sym in out:
        out[sym] = sorted(out[sym], key=lambda r: r["date"])[-10:]
    return out


def short_summary(sv: dict | None, info: dict | None, ftd: list | None) -> dict:
    """
    彙總單一標的的做空面（純函數）。
      sv:   parse_short_volume 的單檔值（或 None）
      info: yfinance .info（用 sharesShort/shortPercentOfFloat/shortRatio/sharesShortPriorMonth）
      ftd:  parse_ftd 的單檔清單（或 None）
    """
    info = info or {}

    def _f(x):
        try:
            v = float(x)
            return v if v == v else None      # NaN guard
        except (TypeError, ValueError):
            return None

    spf = _f(info.get("shortPercentOfFloat"))
    if spf is not None and spf > 1:            # 有些版本給百分比數字（3.2=3.2%）
        # 取捨：GME 2021 級的 >100% 短倉（小數慣例下 >1）會被誤除——
        # 但那是十年一遇；百分比慣例值 >1 天天都有。選擇讓常見情況正確。
        spf /= 100
    cur, prior = _f(info.get("sharesShort")), _f(info.get("sharesShortPriorMonth"))
    chg = (cur / prior - 1) if (cur and prior) else None

    notes = []
    ratio = (sv or {}).get("ratio")
    if ratio is not None:
        tag = "偏高" if ratio >= 0.5 else ("中性" if ratio >= 0.35 else "偏低")
        notes.append(f"日做空量占比 {ratio:.0%}（{tag}；>50% 表示當日賣壓中做空活躍）")
    if spf is not None:
        tag = "高" if spf >= 0.15 else ("中" if spf >= 0.05 else "低")
        notes.append(f"短倉占流通 {spf:.1%}（{tag}；>15% 有軋空題材但也代表強烈看空）")
    if _f(info.get("shortRatio")) is not None:
        notes.append(f"回補天數 {float(info['shortRatio']):.1f} 天（days-to-cover）")
    if chg is not None:
        notes.append(f"短倉月變化 {chg:+.0%}")
    if ftd:
        last = ftd[-1]
        notes.append(f"最近失券 {last['qty']:,.0f} 股（{last['date']}；持續大量 FTD 值得留意）")

    return {"short_vol_ratio": ratio, "short_pct_float": spf,
            "days_to_cover": _f(info.get("shortRatio")),
            "shares_short": cur, "short_chg_1m": chg,
            "ftd_recent": (ftd or [])[-3:], "notes": notes}


# ── 抓取層（需網路；官方檔案下載，免 key）───────────────────────────────────────

def _ua() -> dict:
    ua = os.environ.get("SEC_USER_AGENT",
                        "RBS-Finance-Dashboard/1.0 (contact via GitHub repo)")
    return {"User-Agent": ua}


def fetch_short_volume(tickers: list, lookback_days: int = 6) -> dict:
    """抓最近一個交易日的 FINRA 做空量（從今天往回試最多 lookback_days 天）。"""
    import datetime as _dt

    import requests
    tset = {t.upper() for t in tickers}
    d = _dt.date.today()
    for _ in range(lookback_days):
        url = FINRA_DAILY.format(ymd=d.strftime("%Y%m%d"))
        try:
            r = requests.get(url, headers=_ua(), timeout=20)
            if r.ok and r.text and "|" in r.text[:200]:
                parsed = parse_short_volume(r.text, tset)
                if parsed:
                    for v in parsed.values():
                        v["as_of"] = d.isoformat()
                    return parsed
        except Exception:
            pass
        d -= _dt.timedelta(days=1)
    return {}


def fetch_ftd(tickers: list, months_back: int = 2) -> dict:
    """抓最近可得的 SEC 失券檔（半月一檔，發布延遲 2-4 週 → 從上月開始往回試）。"""
    import datetime as _dt

    import requests
    tset = {t.upper() for t in tickers}
    today = _dt.date.today()
    candidates = []
    y, m = today.year, today.month
    for _ in range(months_back + 1):
        candidates += [(f"{y}{m:02d}", "b"), (f"{y}{m:02d}", "a")]
        m -= 1
        if m == 0:
            y, m = y - 1, 12
    for ym, half in candidates:
        url = SEC_FTD.format(ym=ym, half=half)
        try:
            r = requests.get(url, headers=_ua(), timeout=30)
            if r.ok and r.content[:2] == b"PK":          # 是 zip
                parsed = parse_ftd(r.content, tset)
                if parsed:
                    return parsed
        except Exception:
            pass
    return {}


def fetch_short_overview(ticker: str, info: dict | None = None) -> dict:
    """單一標的完整做空面。info 可傳入既有的 yfinance .info 避免重抓。"""
    if info is None:
        try:
            import yfinance as yf
            info = dict(yf.Ticker(ticker).info or {})
        except Exception:
            info = {}
    sv = fetch_short_volume([ticker]).get(ticker.upper())
    ftd = fetch_ftd([ticker]).get(ticker.upper())
    out = short_summary(sv, info, ftd)
    out["ticker"] = ticker.upper()
    if sv:
        out["as_of"] = sv.get("as_of")
    return out


# ── CLI 自我測試（純解析）─────────────────────────────────────────────────────

if __name__ == "__main__":
    SAMPLE = ("Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n"
              "20260702|AAPL|3000000|10000|5000000|B\n"
              "20260702|AAPL|1000000|5000|3000000|Q\n"      # 同檔第二市場 → 加總
              "20260702|TSLA|900000|0|1000000|B\n"
              "20260702|BAD|x|0|y|B\n")
    sv = parse_short_volume(SAMPLE, {"AAPL", "TSLA"})
    print("short_volume:", {k: round(v["ratio"], 3) for k, v in sv.items()})
    assert abs(sv["AAPL"]["ratio"] - 4000000 / 8000000) < 1e-9
    assert abs(sv["TSLA"]["ratio"] - 0.9) < 1e-9
    assert "BAD" not in sv

    # FTD zip（合成）
    ftd_txt = ("SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE\n"
               "20260610|037833100|AAPL|150000|APPLE INC|195.30\n"
               "20260611|037833100|AAPL|80000|APPLE INC|196.10\n"
               "20260611|88160R101|TSLA|50000|TESLA INC|.\n")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("cnsfails202606b.txt", ftd_txt)
    ftd = parse_ftd(buf.getvalue(), {"AAPL", "TSLA"})
    print("ftd:", {k: len(v) for k, v in ftd.items()})
    assert len(ftd["AAPL"]) == 2 and ftd["AAPL"][-1]["date"] == "20260611"
    assert ftd["TSLA"][0]["price"] is None                  # "." 價格安全

    s = short_summary(sv["AAPL"],
                      {"shortPercentOfFloat": 0.032, "shortRatio": 1.8,
                       "sharesShort": 120e6, "sharesShortPriorMonth": 100e6},
                      ftd["AAPL"])
    print("summary notes:")
    for n in s["notes"]:
        print("  ·", n)
    assert s["short_pct_float"] == 0.032 and abs(s["short_chg_1m"] - 0.2) < 1e-9
    # 百分比慣例輸入（3.2 = 3.2%）
    s2 = short_summary(None, {"shortPercentOfFloat": 3.2}, None)
    assert abs(s2["short_pct_float"] - 0.032) < 1e-9
    assert short_summary(None, {}, None)["notes"] == []     # 全空安全

    print("\n✅ short_data 純解析測試通過")
