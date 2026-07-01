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
    if cpi_info:
        cpi_info["label"] = CPI_SERIES[1]
        cpi_info["unit"] = CPI_SERIES[2]
        out["cpi"] = cpi_info
    return out


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
