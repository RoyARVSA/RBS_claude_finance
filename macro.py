"""
macro.py – 總體經濟數據（FRED，可重用，無 Streamlit 依賴）

用免費的 FRED API（需 API key，https://fred.stlouisfed.org/ → My Account → API Keys）
抓關鍵總經序列，供市場總覽與 AI 分析使用。

抓取層（fetch_macro）與純解析/判讀層（_parse_observations / macro_regime）分離，
純邏輯可用合成資料離線測試。
"""

from __future__ import annotations

# 關鍵序列：Fed 利率、10Y/2Y 殖利率、殖利率曲線、CPI、失業率
FRED_SERIES = {
    "fed_funds":  ("FEDFUNDS",  "Fed 基準利率", "%"),
    "y10":        ("DGS10",     "10年期殖利率", "%"),
    "y2":         ("DGS2",      "2年期殖利率",  "%"),
    "curve":      ("T10Y2Y",    "殖利率曲線(10Y-2Y)", "%"),
    "unemploy":   ("UNRATE",    "失業率",       "%"),
}
CPI_SERIES = ("CPIAUCSL", "CPI 年增率", "%")


# ── 純解析（可離線測試）──────────────────────────────────────────────────────

def _parse_observations(payload: dict) -> list:
    """
    把 FRED observations JSON 解析成 [(date_str, float), ...]（新→舊），略過缺值 "."。
    payload = {"observations": [{"date": "...", "value": "..."}, ...]}
    """
    obs = (payload or {}).get("observations", [])
    out = []
    for o in obs:
        v = o.get("value")
        if v is None or v == ".":
            continue
        try:
            out.append((o.get("date"), float(v)))
        except (TypeError, ValueError):
            continue
    return out


def _latest_change(series: list) -> dict | None:
    """series 為新→舊的 [(date, val)]；回最新值、前值、變動。"""
    if not series:
        return None
    latest_date, latest = series[0]
    prev = series[1][1] if len(series) > 1 else None
    chg = (latest - prev) if prev is not None else None
    return {"value": latest, "prev": prev, "chg": chg, "date": latest_date}


def _cpi_yoy(series: list) -> dict | None:
    """
    CPI 年增率：series 為新→舊的 [(date, index)]，取最新 vs 12 期前。
    回 {value(%), prev(%), chg, date}。需至少 13 期。
    """
    if len(series) < 13:
        return None
    latest_date, latest = series[0]
    year_ago = series[12][1]
    yoy = (latest / year_ago - 1) * 100 if year_ago else None
    # 前一個月的年增率（用第 2 與第 14 期）
    prev = None
    if len(series) >= 14 and series[13][1]:
        prev = (series[1][1] / series[13][1] - 1) * 100
    chg = (yoy - prev) if (yoy is not None and prev is not None) else None
    return {"value": yoy, "prev": prev, "chg": chg, "date": latest_date}


def macro_regime(macro: dict) -> dict:
    """
    從總經數據做簡單判讀。回 {signals: [...], risk: 'caution'|'neutral'|'ok'}。
    - 殖利率曲線倒掛（<0）→ 衰退風險
    - 失業率上升 → 轉弱
    - CPI 仍高（>3%）→ 通膨壓力
    """
    signals = []
    risk = "neutral"
    curve = macro.get("curve", {})
    if curve and curve.get("value") is not None:
        if curve["value"] < 0:
            signals.append("⚠️ 殖利率曲線倒掛（衰退領先訊號）")
            risk = "caution"
        elif curve["value"] < 0.2:
            signals.append("殖利率曲線接近倒掛")
    un = macro.get("unemploy", {})
    if un and un.get("chg") is not None and un["chg"] > 0.1:
        signals.append("失業率上升，勞動市場轉弱")
        risk = "caution"
    cpi = macro.get("cpi", {})
    if cpi and cpi.get("value") is not None and cpi["value"] > 3.0:
        signals.append(f"通膨仍偏高（CPI {cpi['value']:.1f}%）")
    if not signals:
        signals.append("總經數據無明顯警訊")
        risk = "ok"
    return {"signals": signals, "risk": risk}


def macro_summary_text(macro: dict) -> str:
    """組給 AI 或訊息用的一行摘要。"""
    parts = []
    for key, label in [("fed_funds", "Fed利率"), ("y10", "10Y"), ("curve", "殖利率曲線"),
                       ("cpi", "CPI年增"), ("unemploy", "失業率")]:
        d = macro.get(key)
        if d and d.get("value") is not None:
            parts.append(f"{label} {d['value']:.2f}%")
    return "、".join(parts)


# ── 抓取層（需網路 + API key）────────────────────────────────────────────────

def _fred_get(series_id: str, api_key: str, limit: int = 14) -> list:
    """呼叫 FRED observations API，回 [(date, val)]（新→舊）。失敗回 []。"""
    import requests
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id, "api_key": api_key, "file_type": "json",
        "sort_order": "desc", "limit": limit,
    }
    try:
        r = requests.get(url, params=params, timeout=20)
        if not r.ok:
            return []
        return _parse_observations(r.json())
    except Exception:
        return []


# 重要數據發布（FRED release id → 短名）；FOMC 不在 FRED releases，需另行留意
# id 經網查核實：10=CPI、50=Employment Situation、53=GDP、54=PCE、
# 9=Advance Retail Sales、13=G.17 Industrial Production、46=PPI
KEY_RELEASES = {10: "CPI 通膨", 50: "非農就業", 53: "GDP", 54: "PCE 物價",
                9: "零售銷售", 13: "工業生產", 46: "PPI 生產者物價"}


def filter_release_dates(rows: list, today: str, days_ahead: int = 7) -> list:
    """
    純函數：從 FRED releases/dates 列（[{release_id, release_name, date}]）
    篩出 today ~ today+days_ahead 內的重要發布。回 [(date, 短名)] 依日期排序去重。
    """
    import datetime as _dt
    try:
        t0 = _dt.date.fromisoformat(today)
    except ValueError:
        return []
    t1 = t0 + _dt.timedelta(days=days_ahead)
    seen, out = set(), []
    for r in rows or []:
        rid = r.get("release_id")
        if rid not in KEY_RELEASES:
            continue
        try:
            d = _dt.date.fromisoformat(str(r.get("date")))
        except ValueError:
            continue
        if t0 <= d <= t1 and (rid, d) not in seen:
            seen.add((rid, d))
            out.append((d.isoformat(), KEY_RELEASES[rid]))
    out.sort()
    return out


def fetch_release_calendar(api_key: str, days_ahead: int = 7) -> list:
    """本週重要總經數據發布日（FRED releases/dates，含未來排程）。失敗回 []。"""
    if not api_key:
        return []
    import datetime as _dt

    import requests
    try:
        r = requests.get("https://api.stlouisfed.org/fred/releases/dates",
                         params={"api_key": api_key, "file_type": "json",
                                 "include_release_dates_with_no_data": "true",
                                 "realtime_start": _dt.date.today().isoformat(),
                                 "realtime_end": "9999-12-31",
                                 "sort_order": "asc", "limit": 500},
                         timeout=20)
        rows = (r.json() or {}).get("release_dates", []) if r.ok else []
    except Exception:
        return []
    return filter_release_dates(rows, _dt.date.today().isoformat(), days_ahead)


def fetch_macro(api_key: str) -> dict:
    """
    抓所有關鍵總經序列。回 {key: {value, prev, chg, date, label, unit}}，
    外加 'cpi'（年增率）。無 key 或失敗回 {}。
    """
    if not api_key:
        return {}
    out = {}
    for key, (sid, label, unit) in FRED_SERIES.items():
        series = _fred_get(sid, api_key, limit=14)
        info = _latest_change(series)
        if info:
            info["label"] = label
            info["unit"] = unit
            out[key] = info
    # CPI 年增率
    cpi_series = _fred_get(CPI_SERIES[0], api_key, limit=15)
    cpi_info = _cpi_yoy(cpi_series)
    if cpi_info and cpi_info.get("value") is not None:
        cpi_info["label"] = CPI_SERIES[1]
        cpi_info["unit"] = CPI_SERIES[2]
        out["cpi"] = cpi_info
    return out


# ── 總經事件靜默窗（FOMC）─────────────────────────────────────────────────────
# FOMC 例會（聯準會官方行事曆；聲明於第二日 14:00 ET）。2027 為 Fed 2025-09 公布的暫定表。
# 只列「決議日」（第二日）；靜默窗 = 決議日往前 days_before 個日曆日 ～ 決議日當天。
FOMC_DECISION_DATES = {
    # 2024/2025 供 /engtest 歷史重放用（已開過的會，官方行事曆）
    2024: ["01-31", "03-20", "05-01", "06-12", "07-31", "09-18", "11-07", "12-18"],
    2025: ["01-29", "03-19", "05-07", "06-18", "07-30", "09-17", "10-29", "12-10"],
    2026: ["01-28", "03-18", "04-29", "06-17", "07-29", "09-16", "10-28", "12-09"],
    2027: ["01-27", "03-17", "04-28", "06-09", "07-28", "09-15", "10-27", "12-08"],
}


def fomc_dates(years=None) -> list[str]:
    """全部 FOMC 決議日 ISO 字串（排序）。"""
    ys = years or sorted(FOMC_DECISION_DATES)
    return sorted(f"{y}-{md}" for y in ys for md in FOMC_DECISION_DATES.get(y, []))


def next_fomc(today: str) -> str | None:
    """today（含）之後最近一次 FOMC 決議日；表外年份回 None（呼叫端顯示「未載入」）。"""
    t = str(today)[:10]
    return next((d for d in fomc_dates() if d >= t), None)


def event_blackout(today: str, days_before: int = 1, extra_dates: list[str] | None = None) -> tuple[bool, str]:
    """
    純函數：today 是否落在總經事件靜默窗內。回 (in_blackout, 說明)。
    靜默窗 = 事件日往前 days_before 個「日曆日」到事件日當天（FOMC 週二/週三，
    days_before=1 即會期兩天都靜默）。extra_dates：額外事件日（如 CPI）。
    語意由呼叫端定義（引擎：不開新倉/不加碼、出場照常）；壞日期 → (False, "")。
    """
    import datetime as _dt
    try:
        t = _dt.date.fromisoformat(str(today)[:10])
    except (TypeError, ValueError):
        return False, ""
    try:
        nb = max(0, min(int(days_before), 10))
    except (TypeError, ValueError):
        nb = 1
    all_fomc = fomc_dates()
    if all_fomc and t.isoformat() > all_fomc[-1]:
        return False, f"FOMC 日期表已過期（最後 {all_fomc[-1]}），請更新 FOMC_DECISION_DATES"   # 呼叫端可印警告
    events = [(d, "FOMC 決議") for d in all_fomc]
    for d in (extra_dates or []):
        events.append((str(d)[:10], "總經數據"))
    for d, label in sorted(events):
        try:
            ed = _dt.date.fromisoformat(d)
        except ValueError:
            continue
        if ed - _dt.timedelta(days=nb) <= t <= ed:
            when = "今日" if ed == t else f"{(ed - t).days} 天後（{d}）"
            return True, f"{label}{when}"
    return False, ""


# ── CLI 自我測試（純邏輯）──────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== _parse_observations（略過缺值 .）===")
    payload = {"observations": [
        {"date": "2026-06-01", "value": "4.25"},
        {"date": "2026-05-01", "value": "."},
        {"date": "2026-04-01", "value": "4.50"},
    ]}
    print(" ", _parse_observations(payload))

    print("\n=== _latest_change ===")
    print(" ", _latest_change([("2026-06-01", 4.25), ("2026-04-01", 4.50)]))

    print("\n=== _cpi_yoy（13 期）===")
    cpi = [("2026-06-01", 320.0)] + [(f"m{i}", 320.0 - i) for i in range(1, 14)]
    print(" ", _cpi_yoy(cpi))

    print("\n=== macro_regime（倒掛 + 高通膨）===")
    m = {"curve": {"value": -0.3}, "cpi": {"value": 3.6},
         "unemploy": {"value": 4.1, "chg": 0.2}}
    print(" ", macro_regime(m))
    print(" summary:", macro_summary_text({
        "fed_funds": {"value": 4.25}, "y10": {"value": 4.4},
        "curve": {"value": -0.3}, "cpi": {"value": 3.6}, "unemploy": {"value": 4.1}}))

    print("\n=== event_blackout / next_fomc ===")
    assert next_fomc("2026-09-14") == "2026-09-16" and next_fomc("2026-09-17") == "2026-10-28"
    assert next_fomc("2028-01-01") is None
    assert event_blackout("2026-09-14") == (False, "")            # 週一：決議前兩天，不靜默
    assert event_blackout("2026-09-15")[0] and "1 天後" in event_blackout("2026-09-15")[1]   # 會期第一天
    assert event_blackout("2026-09-16") == (True, "FOMC 決議今日")
    assert event_blackout("2026-09-17") == (False, "")            # 決議次日恢復
    assert event_blackout("2026-09-14", days_before=2)[0]         # 加大靜默窗
    assert event_blackout("2026-09-14", days_before=0) == (False, "") and event_blackout("2026-09-16", days_before=0)[0]
    assert event_blackout("2026-10-13", extra_dates=["2026-10-14"])[0] and "總經數據" in event_blackout("2026-10-13", extra_dates=["2026-10-14"])[1]
    assert event_blackout("bad-date") == (False, "") and event_blackout("2026-09-16", days_before="x")[0]
    assert len(fomc_dates([2026])) == 8 and len(fomc_dates()) == 32
    _late = event_blackout("2028-03-01")
    assert _late[0] is False and "過期" in _late[1]                 # 表外年份：不靜默但帶警告
    import datetime as _dtt
    assert max(fomc_dates()) >= f"{_dtt.date.today().year}-01-01", "FOMC 日期表需涵蓋今年（CI 提醒更新）"
    print("  ✅ 靜默窗：會期兩天靜默、前後正常、extra/壞輸入安全")
