"""
market_weather.py – 市場氣象台（regime v2：多因子市場體質綜合分，純邏輯/抓取分離）

用五個「領先體質指標」取代單一 MA50 濾網，合成 0-100 體質分 → 三態 regime：

  1. 廣度（權重 .30）：11 檔 SPDR 類股 ETF 站上各自 MA50 的比例
     ——大盤靠少數巨頭撐、廣度惡化，是頭部經典前兆
  2. 信用利差（.25）：HYG/LQD 比值 vs 其 MA50——債市鼻子比股市靈
  3. VIX 期限結構（.20）：VIX3M/VIX 比值——backwardation（近月貴）= 搶保險
  4. 殖利率曲線（.15）：10 年期 − 13 週國庫券（^TNX、^IRX 同為 ×10 標度，直接相減）
  5. 銅金比（.10）：HG=F/GC=F 的 20 日變化——景氣預期溫度計

穩健性設計：
  • 任一成分抓不到就缺席（補 50 中性值、不重新配權——審查團 F8）；有效成分 < 3 → 回 None，
    呼叫端退回舊 MA50 邏輯——代碼失效/來源改版都只是降級、不會癱瘓
  • regime 邊界帶遲滯（±5 分）：避免 15 分鐘 cron 在門檻附近翻來覆去
  • 快取 TTL 1 小時（state["weather"]），體質不需要分鐘級新鮮度

門檻：分數 ≥60 → risk_on；≤40 → risk_off；中間 → neutral。
"""

from __future__ import annotations

from datetime import datetime, timezone

SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI",
               "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"]

WEATHER_WEIGHTS = {
    "breadth":     0.30,
    "credit":      0.25,
    "vix_ts":      0.20,
    "yield_curve": 0.15,
    "copper_gold": 0.10,
}

RISK_ON_TH, RISK_OFF_TH = 60.0, 40.0
HYSTERESIS = 5.0          # 前一狀態的門檻讓分（防翻來覆去）
TTL_HOURS = 1.0


def _bad(x) -> bool:
    """None 或 NaN。NaN 會穿過所有 `is not None` 檢查並在 _lin 靜默變滿分——必須擋。"""
    return x is None or x != x


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _lin(x: float, lo: float, hi: float) -> float:
    """x 在 [lo,hi] 線性映射到 0-100（外側夾住）。"""
    if hi == lo:
        return 50.0
    return _clamp01((x - lo) / (hi - lo)) * 100.0


# ── 純邏輯：各成分 → 子分數（皆可離線測）─────────────────────────────────────

def breadth_score(above_ma50: int, total: int) -> float | None:
    """類股 ETF 站上 MA50 的比例 → 0-100。"""
    if total <= 0:
        return None
    return round(above_ma50 / total * 100.0, 1)


def credit_score(hyg_lqd_ratio: float, ratio_ma50: float) -> float | None:
    """HYG/LQD 相對其 MA50 的偏離：-2% → 0、+2% → 100。"""
    if _bad(hyg_lqd_ratio) or _bad(ratio_ma50) or not ratio_ma50:
        return None
    dev = hyg_lqd_ratio / ratio_ma50 - 1
    return round(_lin(dev, -0.02, 0.02), 1)


def vix_ts_score(vix: float, vix3m: float) -> float | None:
    """VIX3M/VIX：0.95（backwardation）→ 0、1.15（陡 contango）→ 100。"""
    if _bad(vix) or _bad(vix3m) or vix <= 0:
        return None
    return round(_lin(vix3m / vix, 0.95, 1.15), 1)


def yield_curve_score(spread_pct: float) -> float | None:
    """10 年 − 3 月利差（百分點）：-0.5 → 0、+1.5 → 100。"""
    if _bad(spread_pct):
        return None
    return round(_lin(spread_pct, -0.5, 1.5), 1)


def copper_gold_score(chg_20d: float) -> float | None:
    """銅金比 20 日變化：-8% → 0、+8% → 100。"""
    if _bad(chg_20d):
        return None
    return round(_lin(chg_20d, -0.08, 0.08), 1)


def composite(components: dict, weights: dict | None = None,
              min_components: int = 3) -> dict | None:
    """
    components: {name: score|None}。有效成分 < min_components → None。
    回 {"score", "components", "missing"}；缺席成分補 50 中性值（不是重新配權，見下方 F8 註解）。
    """
    w = weights or WEATHER_WEIGHTS
    valid = {k: v for k, v in components.items() if not _bad(v) and k in w}
    if len(valid) < min_components:
        return None
    # 審查團 F8：缺席成分「補 50 中性值」而非重新配權——重配權會讓悲觀成分
    # 缺席時同樣的資訊翻多（模擬實證 52.5→66.4），跨期分數失去可比性
    score = sum(w[k] * (valid[k] if k in valid else 50.0) for k in w) \
        / sum(w.values())
    return {"score": round(score, 1), "components": components,
            "missing": sorted(k for k in w if _bad(components.get(k)))}


BREADTH_VETO = 40.0   # 廣度分 < 40 → risk_on 降為 neutral（2026-09 實案：總分 66 偏多、廣度 36，
                      # 窄幅上漲環境動能進場全被掃；引擎在 neutral 縮量、不加碼）


def to_regime(score: float, prev: str | None = None, breadth: float | None = None) -> dict:
    """
    分數 → 三態（帶遲滯：前一狀態往回翻需多跨 HYSTERESIS 分）。
    breadth：廣度成分分數（可缺）；risk_on 但廣度 < BREADTH_VETO → neutral（廣度否決）。
    注意：否決後存下的 prev 為 neutral，之後回 risk_on 需總分 ≥ 60（否決會重置多方遲滯——刻意：
    廣度剛修復時不該立刻恢復滿額曝險）。
    回 {"regime","emoji","label"}。
    """
    on_th, off_th = RISK_ON_TH, RISK_OFF_TH
    if prev == "risk_on":
        on_th -= HYSTERESIS        # 已在多方 → 降到 55 以下才降級
    elif prev == "risk_off":
        off_th += HYSTERESIS       # 已在空方 → 升到 45 以上才升級
    if score >= on_th:
        if breadth is not None and not _bad(breadth) and float(breadth) < BREADTH_VETO:
            return {"regime": "neutral", "emoji": "🟡",
                    "label": f"偏多但廣度弱（氣象台 {score:.0f}/100、廣度 {float(breadth):.0f}）"}
        return {"regime": "risk_on", "emoji": "🟢",
                "label": f"偏多（氣象台 {score:.0f}/100）"}
    if score <= off_th:
        return {"regime": "risk_off", "emoji": "🔴",
                "label": f"偏空（氣象台 {score:.0f}/100）"}
    return {"regime": "neutral", "emoji": "🟡",
            "label": f"中性（氣象台 {score:.0f}/100）"}


COMP_LABELS = {
    "breadth":     "市場廣度（類股站上 MA50）",
    "credit":      "信用利差（HYG/LQD）",
    "vix_ts":      "VIX 期限結構",
    "yield_curve": "殖利率曲線（10Y−3M）",
    "copper_gold": "銅金比 20 日",
}


def weather_text(weather: dict, regime: dict | None = None) -> str:
    """/weather 指令文字（Telegram legacy Markdown：單 *）。"""
    lines = ["🌦 *市場氣象台*（體質綜合分，非投資建議）"]
    if regime:
        lines.append(f"{regime['emoji']} {regime['label']}")
    comps = weather.get("components") or {}
    for k, lbl in COMP_LABELS.items():
        v = comps.get(k)
        if v is None:
            lines.append(f"・{lbl}：資料缺")
            continue
        bar = "▰" * int(round(v / 12.5)) + "▱" * (8 - int(round(v / 12.5)))
        lines.append(f"・{lbl}：{v:.0f} {bar}")
    if weather.get("missing"):
        # 成分英文名含底線（vix_ts…）會讓 Telegram legacy Markdown 炸掉整則訊息
        # （PITFALLS D12）——一律轉中文短名
        miss = "、".join(COMP_LABELS.get(m, m).split("（")[0]
                         for m in weather["missing"])
        lines.append(f"（缺席成分以中性 50 分計：{miss}）")
    lines.append("_≥60 偏多／≤40 偏空；廣度與信用是領先權重最大的兩項_")
    return "\n".join(lines)


def _hours_since(ts: str | None, now: str) -> float:
    try:
        a = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        b = datetime.fromisoformat(str(now).replace("Z", "+00:00"))
        return (b - a).total_seconds() / 3600.0
    except Exception:
        return float("inf")


# ── 抓取層（需網路；全 yfinance 免 key）──────────────────────────────────────

def inputs_from_history(px_sector, px_cross) -> dict:
    """
    （純函式）收盤價寬表 → 原始輸入；線上 fetch_inputs 與歷史重放共用同一套組裝（研究計畫 §1：
    重放另寫一份就保證不了「原封不動」）。px_*：欄＝代碼、只含「到當時為止」的資料（重放時由呼叫端截斷）。
    回 {breadth: (above, total)|None, credit: (ratio, ma50)|None,
        vix: (vix, vix3m)|None, curve: spread|None, cu_au: chg|None}
    """
    out = {"breadth": None, "credit": None, "vix": None,
           "curve": None, "cu_au": None}

    # 1) 廣度：類股 ETF vs 各自 MA50
    try:
        if px_sector is not None:
            above = total = 0
            for t in SECTOR_ETFS:
                try:
                    s = px_sector[t].dropna()
                    if len(s) >= 50:
                        total += 1
                        if float(s.iloc[-1]) > float(s.rolling(50).mean().iloc[-1]):
                            above += 1
                except Exception:
                    continue
            if total >= 6:                       # 抓到過半才算數
                out["breadth"] = (above, total)
    except Exception as e:
        print(f"weather: 廣度計算失敗 {e}")

    # 2) 信用利差 + 3) VIX 期限 + 4) 曲線 + 5) 銅金
    try:
        px = px_cross
        if px is not None:
            def last(t):
                try:
                    s = px[t].dropna()
                    return float(s.iloc[-1]) if len(s) else None
                except Exception:
                    return None

            hyg, lqd = last("HYG"), last("LQD")
            if hyg and lqd:
                try:
                    ratio = (px["HYG"] / px["LQD"]).dropna()
                    if len(ratio) >= 50:
                        out["credit"] = (float(ratio.iloc[-1]),
                                         float(ratio.rolling(50).mean().iloc[-1]))
                except Exception:
                    pass
            vix, vix3m = last("^VIX"), last("^VIX3M")
            if vix and vix3m:
                out["vix"] = (vix, vix3m)
            tnx, irx = last("^TNX"), last("^IRX")
            if tnx is not None and irx is not None:
                # 審查團 F23：Yahoo 對 ^TNX/^IRX 以「百分比」報價（4.64=4.64%），
                # 不是 CBOE 原始 ×10 標度——先前錯除 10 讓此成分永遠卡 21~28 分
                # （valuation.py 用 ^TNX/100 取 rf 可互證）。直接相減即為百分點。
                out["curve"] = tnx - irx
            try:
                cu_au = (px["HG=F"] / px["GC=F"]).dropna()
                if len(cu_au) >= 21:
                    out["cu_au"] = float(cu_au.iloc[-1] / cu_au.iloc[-21] - 1)
            except Exception:
                pass
    except Exception as e:
        print(f"weather: 跨資產計算失敗 {e}")
    return out


CROSS_TICKERS = ["HYG", "LQD", "^VIX", "^VIX3M", "^TNX", "^IRX", "HG=F", "GC=F"]


def fetch_inputs() -> dict:
    """
    一次抓齊所有原始輸入（網路）→ inputs_from_history（純函式）。個別失敗 → 該成分 None。
    """
    import yfinance as yf

    def _closes(tickers, period="6mo"):
        raw = yf.download(tickers, period=period, auto_adjust=True,
                          progress=False, group_by="column")
        if raw is None or raw.empty:
            return None
        px = raw["Close"] if "Close" in raw else raw
        return px

    px_s = px_c = None
    try:
        px_s = _closes(SECTOR_ETFS)
    except Exception as e:
        print(f"weather: 廣度抓取失敗 {e}")
    try:
        px_c = _closes(CROSS_TICKERS)
    except Exception as e:
        print(f"weather: 跨資產抓取失敗 {e}")
    return inputs_from_history(px_s, px_c)


LIVE_LOOKBACK_MONTHS = 6       # 線上 fetch_inputs 用 yf.download(period="6mo")——重放每天也只看前 6 個月，才和線上同語意


def replay_regimes(px_sector, px_cross, dates, ma50_fallback: dict | None = None,
                   lookback_months: int = LIVE_LOOKBACK_MONTHS) -> dict:
    """
    （純函式）把現行氣象台逐日套回歷史：每個日期 d 只用「d 往前 6 個月到 d」的收盤（與線上抓 6 個月同語意）
    → inputs_from_history → composite →
    to_regime（遲滯的「前一狀態」＝上一個氣象台讀數，與線上快取同語意）。
    成分不足（composite 回 None）→ 該日用 ma50_fallback[d]（線上 market_regime 的 MA50 退路），
    而且**不更新**遲滯的前一狀態（線上此時不寫 weather 快取）。
    回 {d: {"regime", "score", "missing", "source"}}，source ∈ weather / ma50 / none。
    """
    import pandas as _pd
    out, prev = {}, None
    for d in dates:
        ts = _pd.Timestamp(d)
        t0 = ts - _pd.DateOffset(months=int(lookback_months))
        ps = px_sector.loc[t0:ts] if px_sector is not None else None
        pc = px_cross.loc[t0:ts] if px_cross is not None else None
        comp = composite(inputs_to_components(inputs_from_history(ps, pc)))
        if comp is None:
            fb = (ma50_fallback or {}).get(d)
            out[d] = {"regime": fb, "score": None, "missing": None, "source": "ma50" if fb else "none"}
            continue
        rg = to_regime(comp["score"], prev, (comp.get("components") or {}).get("breadth"))
        prev = rg["regime"]
        out[d] = {"regime": rg["regime"], "score": comp["score"], "missing": comp["missing"], "source": "weather"}
    return out


def log_daily(result: dict, path=None, today: str | None = None) -> bool:
    """線上讀數每日一筆存檔（公開市場資料；研究計畫 §1：以後核對「日收盤重放」與「線上每小時版」）。
    同一天重複呼叫 → 覆寫當天那筆（保留當天最後一次讀數）。失敗回 False、不影響線上。"""
    from pathlib import Path
    try:
        p = Path(path) if path else Path(__file__).resolve().parent / "data" / "weather" / "history.csv"
        today = today or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        comps = result.get("components") or {}
        keys = list(WEATHER_WEIGHTS)
        row = [today, f"{float(result['score']):.1f}", str((result.get("regime") or {}).get("regime") or "")] + \
              ["" if comps.get(k) is None or _bad(comps.get(k)) else f"{float(comps[k]):.1f}" for k in keys]
        header = "date,score,regime," + ",".join(keys)
        lines = p.read_text(encoding="utf-8").splitlines() if p.exists() else [header]
        if not lines or lines[0] != header:
            lines = [header] + [ln for ln in lines if ln and not ln.startswith("date,")]
        lines = [ln for ln in lines[1:] if not ln.startswith(today + ",")]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join([header] + lines + [",".join(row)]) + "\n", encoding="utf-8")
        return True
    except Exception as e:
        print(f"weather: 每日存檔失敗 {type(e).__name__}")
        return False


def inputs_to_components(inputs: dict) -> dict:
    """原始輸入 → 五個子分數（純函數）。"""
    comps = {}
    b = inputs.get("breadth")
    comps["breadth"] = breadth_score(*b) if b else None
    c = inputs.get("credit")
    comps["credit"] = credit_score(*c) if c else None
    v = inputs.get("vix")
    comps["vix_ts"] = vix_ts_score(*v) if v else None
    comps["yield_curve"] = yield_curve_score(inputs.get("curve")) \
        if inputs.get("curve") is not None else None
    comps["copper_gold"] = copper_gold_score(inputs.get("cu_au")) \
        if inputs.get("cu_au") is not None else None
    return comps


def get_weather(state: dict | None = None, now: str | None = None) -> dict | None:
    """
    進入點：帶 1 小時快取的氣象台讀數。回 {"score","components","missing","regime"}
    或 None（成分不足 → 呼叫端退回 MA50 邏輯）。就地寫 state["weather"]。
    """
    now = now or datetime.now(timezone.utc).isoformat()
    cache = (state or {}).get("weather") or {}
    if state is not None and _hours_since(cache.get("ts"), now) < TTL_HOURS \
            and cache.get("score") is not None:
        prev = (cache.get("regime") or {}).get("regime")
        rg = to_regime(float(cache["score"]), prev, (cache.get("components") or {}).get("breadth"))
        return {**{k: cache.get(k) for k in ("score", "components", "missing")},
                "regime": rg}

    comp = composite(inputs_to_components(fetch_inputs()))
    if comp is None:
        return None
    prev = ((cache.get("regime") or {}).get("regime")) if cache else None
    rg = to_regime(comp["score"], prev, (comp.get("components") or {}).get("breadth"))
    result = {**comp, "regime": rg}
    if state is not None:
        state["weather"] = {"ts": now, **comp, "regime": rg}
        log_daily(result, today=now[:10])            # 線上（有 state）才存檔；測試／重放不寫檔
    return result


# ── 自我測試（純邏輯）─────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1) 子分數映射與夾制
    assert breadth_score(11, 11) == 100.0 and breadth_score(0, 11) == 0.0
    assert breadth_score(5, 0) is None
    assert credit_score(1.02, 1.0) == 100.0 and credit_score(0.98, 1.0) == 0.0
    assert credit_score(1.0, 1.0) == 50.0
    assert vix_ts_score(20, 23) == 100.0          # 1.15 contango
    assert vix_ts_score(30, 27) == 0.0            # 0.9 backwardation → 夾 0
    assert yield_curve_score(-1.0) == 0.0 and yield_curve_score(1.5) == 100.0
    assert copper_gold_score(0.0) == 50.0
    # NaN 防護：NaN 穿過 is-not-None、在 min/max 裡靜默變滿分——一律視為缺席
    _nan = float("nan")
    assert credit_score(_nan, 1.0) is None and credit_score(1.0, _nan) is None
    assert vix_ts_score(_nan, 20) is None and vix_ts_score(20, _nan) is None
    assert yield_curve_score(_nan) is None and copper_gold_score(_nan) is None
    c_nan = composite({"breadth": 80, "credit": _nan, "vix_ts": 60,
                       "yield_curve": 50, "copper_gold": 50})
    assert c_nan and c_nan["score"] == c_nan["score"] and "credit" in c_nan["missing"]
    print("✅ 1 子分數映射 + NaN 防護")

    # 2) composite：缺席補 50 中性值（F8——重配權會讓悲觀成分缺席時分數翻多）、
    #    成分不足回 None
    c = composite({"breadth": 100, "credit": 100, "vix_ts": 100,
                   "yield_curve": None, "copper_gold": None})
    # 100×(.3+.25+.2) + 50×(.15+.10) = 87.5（不再是重配權的 100）
    assert c and abs(c["score"] - 87.5) < 0.05 and \
        c["missing"] == ["copper_gold", "yield_curve"], c
    c = composite({"breadth": 80, "credit": 40, "vix_ts": 60,
                   "yield_curve": 50, "copper_gold": 50})
    # 0.3*80+0.25*40+0.2*60+0.15*50+0.1*50 = 24+10+12+7.5+5 = 58.5
    assert c and abs(c["score"] - 58.5) < 0.05, c
    assert composite({"breadth": 90, "credit": 90, "vix_ts": None,
                      "yield_curve": None, "copper_gold": None}) is None
    # F8 核心迴歸：悲觀成分（credit 40、vix 30）缺席時分數不得高於在場時
    full = composite({"breadth": 80, "credit": 40, "vix_ts": 30,
                      "yield_curve": 50, "copper_gold": 50})
    part = composite({"breadth": 80, "credit": None, "vix_ts": None,
                      "yield_curve": 50, "copper_gold": 50})
    assert part["score"] - full["score"] < 12, (full, part)   # 中性填充：溫和收斂非翻多
    print("✅ 2 composite 缺席補中性值")

    # 3) regime 門檻與遲滯
    assert to_regime(65)["regime"] == "risk_on"
    assert to_regime(35)["regime"] == "risk_off"
    assert to_regime(50)["regime"] == "neutral"
    assert to_regime(57, prev="risk_on")["regime"] == "risk_on"     # 遲滯保持
    assert to_regime(57, prev=None)["regime"] == "neutral"
    assert to_regime(43, prev="risk_off")["regime"] == "risk_off"
    assert to_regime(43, prev=None)["regime"] == "neutral"
    assert to_regime(54, prev="risk_on")["regime"] == "neutral"     # 跌破 55 → 降級
    # 廣度否決：偏多總分但廣度 < 40 → neutral（不影響 risk_off / 中性；缺席不否決）
    assert to_regime(65.7, breadth=36.4)["regime"] == "neutral" and "廣度弱" in to_regime(65.7, breadth=36.4)["label"]
    assert to_regime(65.7, breadth=45)["regime"] == "risk_on" and to_regime(65.7, breadth=None)["regime"] == "risk_on"
    assert to_regime(65.7, breadth=float("nan"))["regime"] == "risk_on"
    assert to_regime(30, breadth=10)["regime"] == "risk_off" and to_regime(57, prev="risk_on", breadth=30)["regime"] == "neutral"
    print("✅ 3 三態遲滯")

    # 4) inputs→components 管線 + 快取路徑（不打網路：直接餵 cache）
    comps = inputs_to_components({"breadth": (8, 11), "credit": (1.01, 1.0),
                                  "vix": (18, 20), "curve": 0.5, "cu_au": 0.04})
    assert all(v is not None for v in comps.values()), comps
    st = {"weather": {"ts": "2026-07-20T13:30:00+00:00", "score": 62.0,
                      "components": comps, "missing": [],
                      "regime": {"regime": "risk_on", "emoji": "🟢", "label": "x"}}}
    w = get_weather(st, now="2026-07-20T14:00:00+00:00")     # 30 分鐘 < TTL → 走快取
    assert w and w["score"] == 62.0 and w["regime"]["regime"] == "risk_on", w
    print("✅ 4 管線 + 快取（未打網路）")

    # 5) weather_text 不炸（含缺項）；缺席名單不得含底線（TG legacy Markdown 會炸）
    txt = weather_text({"score": 58.5, "components": {**comps, "vix_ts": None},
                        "missing": ["vix_ts"]}, to_regime(58.5))
    assert "市場氣象台" in txt and "資料缺" in txt
    assert "vix_ts" not in txt, txt          # 缺席行必須用中文短名
    assert txt.count("_") % 2 == 0, txt      # 底線必須成對
    print("✅ 5 weather_text（Markdown 安全）")

    # 6) inputs_from_history：線上與重放共用的組裝（純函式）；replay_regimes：缺成分退 MA50、退路日不更新遲滯
    import numpy as _np
    import pandas as _pd
    _idx = _pd.bdate_range("2007-01-01", periods=160)
    _up = _pd.Series(100 * _np.exp(_np.linspace(0, 0.2, 160)), index=_idx)
    _ps = _pd.DataFrame({t: _up for t in SECTOR_ETFS}, index=_idx)          # 全部上升 → 廣度 11/11
    _pc = _pd.DataFrame({"HYG": _up, "LQD": _up * 0 + 100.0, "^VIX": 12.0, "^VIX3M": 15.0,
                         "^TNX": 4.5, "^IRX": 3.0, "HG=F": _up, "GC=F": 100.0}, index=_idx)
    _pc.loc[_idx[:100], ["HYG", "^VIX3M"]] = _np.nan                       # 前 100 天沒有 HYG、VIX3M（像 2007 前）
    i6 = inputs_from_history(_ps, _pc)
    assert i6["breadth"] == (11, 11) and i6["credit"] is not None and i6["vix"] == (12.0, 15.0) and abs(i6["curve"] - 1.5) < 1e-12
    i6b = inputs_from_history(_ps.loc[:_idx[80]], _pc.loc[:_idx[80]])        # 截斷到第 80 天：信用、VIX 期限缺席
    assert i6b["credit"] is None and i6b["vix"] is None and i6b["breadth"] == (11, 11)
    assert inputs_from_history(None, None) == {"breadth": None, "credit": None, "vix": None, "curve": None, "cu_au": None}
    _days = [str(d.date()) for d in _idx[40:]]
    fb = {d: "neutral" for d in _days}
    rp = replay_regimes(_ps, _pc, _days, fb)
    assert rp[_days[0]]["source"] == "ma50" and rp[_days[0]]["regime"] == "neutral"   # 第 40 天：廣度不到 50 根 → 成分不足
    srcs = [rp[d]["source"] for d in _days]
    assert "weather" in srcs and srcs.index("weather") > 0                   # 成分湊滿 3 個後才改用氣象台
    last = rp[_days[-1]]
    assert last["source"] == "weather" and last["regime"] == "risk_on" and last["missing"] == [], last
    # 遲滯：前一狀態只來自氣象台讀數（退路日不更新）——同一分數序列下，退路插在中間不改變之後的判定
    assert all(rp[d]["regime"] in ("risk_on", "neutral", "risk_off") for d in _days)
    # 遲滯的前一狀態只來自氣象台讀數：分數 62（偏多）→ 退路日 → 57（已偏多，遲滯下 ≥55 仍偏多）
    _orig_comp = composite
    _script = iter([{"score": 62.0, "components": {}, "missing": []}, None,
                    {"score": 57.0, "components": {}, "missing": []}])
    globals()["composite"] = lambda comps: next(_script)
    try:
        rh = replay_regimes(_ps, _pc, _days[:3], {_days[1]: "risk_off"})
    finally:
        globals()["composite"] = _orig_comp
    assert [rh[d]["regime"] for d in _days[:3]] == ["risk_on", "risk_off", "risk_on"], rh   # 若退路更新 prev，第三天會變中性
    print("✅ 6 inputs_from_history／replay_regimes（截斷即 PIT、缺成分退 MA50、退路日不更新遲滯）")

    # 7) log_daily：同一天覆寫、新的一天追加、格式固定
    import tempfile
    from pathlib import Path as _P
    _tp = _P(tempfile.mkdtemp()) / "history.csv"
    _res = {"score": 61.04, "regime": {"regime": "risk_on"}, "components": {"breadth": 80.0, "credit": None}}
    assert log_daily(_res, _tp, "2026-10-01") and log_daily(dict(_res, score=59.0), _tp, "2026-10-01")
    assert log_daily(_res, _tp, "2026-10-02")
    _ln = _tp.read_text().splitlines()
    assert _ln[0].startswith("date,score,regime,breadth") and len(_ln) == 3 and _ln[1].startswith("2026-10-01,59.0,risk_on,80.0,"), _ln
    print("✅ 7 log_daily（同日覆寫、跨日追加）")
    print()
    print(txt)
    print("\nmarket_weather selftest OK ✅")
