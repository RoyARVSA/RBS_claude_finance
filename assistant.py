"""
assistant.py – 對話式 AI 助理的純邏輯層（Phase 1，可離線測試）

負責「問題解析 → 意圖判斷 → 組裝精簡 context」，抓取與 LLM 呼叫由網頁層負責。
專業級要求寫進 SYSTEM_PROMPT：有據可查、風險優先、缺資料就說沒有、不得編造。
"""

from __future__ import annotations

import re

# 常見大寫字/縮寫，避免被誤認成股票代碼
STOPWORDS = {
    "AI", "US", "USA", "UK", "EU", "CEO", "CFO", "COO", "CTO", "ETF", "IT",
    "OR", "AND", "THE", "A", "AN", "I", "TO", "IN", "ON", "OF", "IS", "VS",
    "FED", "CPI", "GDP", "PMI", "USD", "EUR", "JPY", "TWD", "RSI", "MACD",
    "MA", "EMA", "ROE", "ROA", "PE", "PB", "PEG", "EPS", "IPO", "IRR", "NAV",
    "YOY", "QOQ", "TTM", "FY", "Q1", "Q2", "Q3", "Q4", "H1", "H2", "YTD",
    "OK", "API", "URL", "FAQ", "P", "E", "B", "VIX", "SEC", "FOMC",
    # 交易所/後綴代碼，避免從 2330.TW 這類被切出來誤認
    "TW", "TWO", "HK", "TSE", "TWSE", "TPEX", "HKEX", "NYSE",
}

# 中文/俗名 → 代碼別名（可持續擴充）
ALIASES = {
    "蘋果": "AAPL", "微軟": "MSFT", "輝達": "NVDA", "英偉達": "NVDA",
    "特斯拉": "TSLA", "亞馬遜": "AMZN", "谷歌": "GOOGL", "字母": "GOOGL",
    "臉書": "META", "元宇宙": "META", "超微": "AMD", "英特爾": "INTC",
    "台積電": "2330.TW", "聯發科": "2454.TW", "鴻海": "2317.TW",
    "台達電": "2308.TW", "聯電": "2303.TW",
    "阿里巴巴": "BABA", "阿里": "BABA", "騰訊": "0700.HK",
    "波克夏": "BRK-B", "巴菲特": "BRK-B", "可口可樂": "KO",
    "摩根大通": "JPM", "波音": "BA", "輝瑞": "PFE", "禮來": "LLY",
}

# 意圖關鍵字（中英）
_INTENT_KW = {
    "technical":   ["技術", "技術面", "rsi", "macd", "均線", "趨勢", "動能",
                    "支撐", "壓力", "波動", "回撤", "超買", "超賣",
                    "technical", "momentum", "trend", "volatility", "drawdown"],
    "fundamental": ["基本面", "財務", "體質", "估值", "本益比", "本益", "pe",
                    "roe", "roa", "獲利", "營收", "毛利", "淨利", "健康", "負債",
                    "現金流", "fundamental", "valuation", "margin", "earnings quality"],
    "macro":       ["總經", "利率", "通膨", "cpi", "失業", "殖利率", "聯準會",
                    "景氣", "衰退", "曲線", "fed", "inflation", "rate", "macro",
                    "recession", "yield"],
    "market":      ["大盤", "市場", "指數", "s&p", "sp500", "標普", "nasdaq",
                    "納斯達克", "道瓊", "vix", "恐慌", "index", "market"],
    "earnings":    ["財報日", "財報何時", "下次財報", "何時公布", "earnings date",
                    "報告日期"],
    "compare":     ["比較", "對比", "相比", "vs", "versus", "哪個好", "哪個較",
                    "誰比較", "compare"],
}


# ── 純邏輯 ─────────────────────────────────────────────────────────────────────

def extract_tickers(question: str, universe: set | None = None,
                    aliases: dict | None = None, max_n: int = 3) -> list[str]:
    """
    從問題抽出股票代碼（最多 max_n 檔，去重、保序）。
      1. 中文/俗名別名（台積電→2330.TW…）
      2. 明確後綴代碼（2330.TW / 0700.HK / 7203.T）
      3. 裸 4 碼數字 → 若 {n}.TW 在 universe 則採用
      4. 美股大寫 2-5 碼（排除停用字；in universe 或看起來像代碼皆收）
    """
    universe = universe or set()
    aliases = aliases or ALIASES
    found: list[str] = []

    def _add(t):
        if t and t not in found:
            found.append(t)

    # 1. 別名
    for name, tk in aliases.items():
        if name in question:
            _add(tk)

    # 2. 明確後綴
    for m in re.findall(r"\b\d{3,5}\.(?:TW|TWO|HK|T)\b", question, flags=re.I):
        _add(m.upper())

    # 3. 裸 4 碼 → TW
    for m in re.findall(r"\b(\d{4})\b", question):
        cand = f"{m}.TW"
        if cand in universe:
            _add(cand)

    # 4. Cashtag（$VRT / $vrt / $2330.TW）—— 明確指定，忽略大小寫、不受停用字限制
    for m in re.findall(r"\$([A-Za-z]{1,5}(?:\.[A-Za-z]{1,3})?|\d{3,5}\.[A-Za-z]{1,3})",
                        question):
        _add(m.upper())

    # 5. 美股大寫（含 BRK-B 帶連字號）；用 ASCII 前後界避免緊接中文時失效
    #    (?<![A-Za-z]) / (?![A-Za-z])：中文、空白、標點都算邊界，只排除英文字母相連
    for m in re.findall(r"(?<![A-Za-z])[A-Z]{2,5}(?:-[A-Z])?(?![A-Za-z])", question):
        base = m.split("-")[0]
        if base in STOPWORDS or m in STOPWORDS:
            continue
        _add(m)

    return found[:max_n]


def detect_intents(question: str, has_tickers: bool) -> set:
    """判斷問題涉及哪些分析維度。有標的但無明確意圖時，預設技術+基本面。"""
    ql = question.lower()
    intents = set()
    for intent, kws in _INTENT_KW.items():
        for kw in kws:
            if (kw in ql) if kw.isascii() else (kw in question):
                intents.add(intent)
                break
    if has_tickers and not (intents & {"technical", "fundamental"}):
        intents |= {"technical", "fundamental"}
    if not has_tickers and not intents:
        intents.add("market")   # 沒標的又沒意圖 → 當一般大盤問題
    return intents


def _fmt(v, pct=False, dp=2):
    if v is None:
        return "無資料"
    try:
        return f"{v*100:.1f}%" if pct else f"{v:.{dp}f}"
    except (TypeError, ValueError):
        return str(v)


def build_context(question: str, timestamp: str, intents: set,
                  ticker_data: dict | None = None,
                  macro_data: dict | None = None,
                  market_data: dict | None = None) -> str:
    """
    把已抓取的結構化資料組成精簡、標註時間與來源的 context 文字。
    ticker_data: {ticker: {"tech": {...}, "fund": {...}, "earnings": "YYYY-MM-DD"}}
    macro_data:  {"summary": "...", "signals": [...]}
    market_data: {"名稱": (價, 漲跌), ...}
    """
    lines = [f"=== 分析資料（截至 {timestamp}，來源 yfinance/FRED）==="]

    for tkr, d in (ticker_data or {}).items():
        lines.append(f"\n【{tkr}】")
        tech = d.get("tech") or {}
        if tech and ("technical" in intents or "compare" in intents):
            lines.append(
                f"  技術：現價 {_fmt(tech.get('price'))}　"
                f"近1月 {_fmt(tech.get('return_1m'), pct=True)}　"
                f"近3月 {_fmt(tech.get('return_3m'), pct=True)}　"
                f"年化波動 {_fmt(tech.get('ann_vol'), pct=True)}　"
                f"RSI {_fmt(tech.get('rsi'), dp=0)}　"
                f"最大回撤 {_fmt(tech.get('max_dd'), pct=True)}")
        fund = d.get("fund") or {}
        if fund and ("fundamental" in intents or "compare" in intents):
            hs = fund.get("health")
            lines.append(
                f"  基本面：財務健康 {hs if hs is not None else '無資料'}　"
                f"P/E {_fmt(fund.get('pe'), dp=1)}　"
                f"ROE {_fmt(fund.get('roe'), pct=True)}　"
                f"淨利率 {_fmt(fund.get('net_margin'), pct=True)}　"
                f"營收成長 {_fmt(fund.get('revenue_growth'), pct=True)}")
        if d.get("earnings") and "earnings" in intents:
            lines.append(f"  下次財報：{d['earnings']}")

    if "macro" in intents and macro_data:
        lines.append("\n【總經】")
        if macro_data.get("summary"):
            lines.append(f"  {macro_data['summary']}")
        for s in (macro_data.get("signals") or []):
            lines.append(f"  · {s}")

    if "market" in intents and market_data:
        lines.append("\n【大盤】")
        lines.append("  " + "　".join(
            f"{n} {c:+.2%}" for n, (p, c) in market_data.items()))

    return "\n".join(lines)


SYSTEM_PROMPT = (
    "你是財金專業團隊的研究副駕（research copilot）。嚴格遵守：\n"
    "1. 只根據『分析資料』區塊的數字回答，每個結論都引用具體數字。\n"
    "2. 資料顯示『無資料』的項目，就明說沒有該資料，絕不編造或臆測。\n"
    "3. 風險優先：談機會時務必同時點出風險、下檔、波動或回撤。\n"
    "4. 用繁體中文，條理清晰、精簡專業；必要時用『技術面/基本面/總經/風險』分段。\n"
    "5. 這是分析與教育用途，非投資建議；不做買賣指令式結論。\n"
    "若問題超出提供的資料範圍，請說明需要哪些額外資料。"
)


# ── CLI 自我測試 ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uni = {"2330.TW", "AAPL", "MSFT", "NVDA"}
    print("=== extract_tickers ===")
    print(" ", extract_tickers("比較 AAPL 和 MSFT 的財務體質", uni))
    print(" ", extract_tickers("台積電 2330.TW 技術面如何", uni))
    print(" ", extract_tickers("AI 產業的 US 前景，看好 NVDA", uni), "(AI/US 應被過濾)")
    print(" ", extract_tickers("蘋果跟輝達哪個好", uni))

    print("\n=== detect_intents ===")
    print(" ", detect_intents("AAPL 的 RSI 和技術面", True))
    print(" ", detect_intents("比較兩家的財務體質估值", True))
    print(" ", detect_intents("現在總經利率環境如何", False))
    print(" ", detect_intents("NVDA 怎麼樣", True), "(有標的無意圖→技術+基本面)")

    print("\n=== build_context ===")
    td = {"AAPL": {"tech": {"price": 190, "return_1m": 0.05, "rsi": 62, "ann_vol": 0.25,
                            "return_3m": 0.12, "max_dd": -0.08},
                   "fund": {"health": 82, "pe": 29, "roe": 1.5, "net_margin": 0.25,
                            "revenue_growth": 0.08},
                   "earnings": "2026-08-01"}}
    print(build_context("AAPL 分析", "2026-07-02 10:00",
                        {"technical", "fundamental", "earnings"}, ticker_data=td))
