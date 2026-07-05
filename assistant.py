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
    "outlook":     ["未來", "前景", "展望", "後市", "怎麼看", "看法",
                    "會漲", "會跌", "值得", "該買", "該進", "布局", "方向",
                    "outlook", "prospect", "future", "看多", "看空", "潛力"],
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
    # 前瞻問題需要全維度：有標的→技術+基本面+總經；無標的（問大盤後市）→大盤+總經
    if "outlook" in intents:
        intents |= ({"technical", "fundamental", "macro"} if has_tickers
                    else {"market", "macro"})
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
            # 趨勢結構（52週位置 + 均線）——有前瞻判斷的材料
            trend_bits = []
            if tech.get("pct_from_52w_high") is not None:
                trend_bits.append(f"距52週高 {_fmt(tech.get('pct_from_52w_high'), pct=True)}")
            if tech.get("pct_from_52w_low") is not None:
                trend_bits.append(f"距52週低 {_fmt(tech.get('pct_from_52w_low'), pct=True)}")
            if tech.get("vs_ma50") is not None:
                trend_bits.append(f"vs MA50 {_fmt(tech.get('vs_ma50'), pct=True)}")
            if tech.get("vs_ma200") is not None:
                trend_bits.append(f"vs MA200 {_fmt(tech.get('vs_ma200'), pct=True)}")
            if trend_bits:
                lines.append("  趨勢：" + "　".join(trend_bits))
        fund = d.get("fund") or {}
        if fund and ("fundamental" in intents or "compare" in intents):
            hs = fund.get("health")
            lines.append(
                f"  基本面：財務健康 {hs if hs is not None else '無資料'}　"
                f"P/E {_fmt(fund.get('pe'), dp=1)}　"
                f"ROE {_fmt(fund.get('roe'), pct=True)}　"
                f"淨利率 {_fmt(fund.get('net_margin'), pct=True)}　"
                f"營收成長 {_fmt(fund.get('revenue_growth'), pct=True)}")
        if d.get("earnings") and (intents & {"earnings", "outlook", "fundamental"}):
            lines.append(f"  催化劑—下次財報：{d['earnings']}")
        if d.get("peers_note"):
            lines.append(f"  同業對比：{d['peers_note']}")

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


# 單輪多空對辯（TradingAgents 式；只做一輪——文獻顯示多輪邊際效益遞減、成本超線性）
DEBATE_BULL_PROMPT = (
    "你是多方研究員。只根據下方『分析資料』，提出**最強的看多論證**（150字內）：\n"
    "1) 兩三個最有力的多方證據（引具體數字）2) 此論證成立的前提條件。\n"
    "不得編造資料中沒有的數字。用繁體中文。"
)
DEBATE_BEAR_PROMPT = (
    "你是空方研究員。只根據下方『分析資料』，提出**最強的看空論證**（150字內）：\n"
    "1) 兩三個最有力的空方證據/風險（引具體數字）2) 什麼情況下跌幅會擴大。\n"
    "不得編造資料中沒有的數字。用繁體中文。"
)

SYSTEM_PROMPT = (
    "你是財金專業團隊的資深研究分析師。目標不是複述數據，而是形成"
    "『有論點、有前瞻、可被檢驗』的分析。嚴守以下紀律：\n\n"
    "【事實 vs 判斷】\n"
    "· 標為『事實』的內容只能引用『分析資料』中的數字，並附上數值；資料為『無資料』"
    "就明說沒有，絕不編造。\n"
    "· 可以提出『判斷／推論』，但必須明確標示（例如以「推論：」開頭），並說明是根據"
    "哪些事實推得。事實與推論不可混為一談。\n\n"
    "【回答結構】（依問題深淺調整；前瞻／展望類問題盡量涵蓋）\n"
    "1. 核心論點：一句話講你的判斷，含方向與時間框架。\n"
    "2. 支持證據：引用具體數字（技術／基本面／總經／工具結果）。\n"
    "3. 反方與風險：下檔、波動、回撤，以及『什麼證據會推翻此論點』。\n"
    "4. 催化劑與時間點：財報日、總經事件、可能觸發價格變動的因素。\n"
    "5. 情境：牛／基準／熊三情境與各自的觸發條件。\n"
    "6. 觀察指標：接下來要盯哪些數據或價位來確認或否定論點。\n"
    "7. 信心度（低／中／高）與適用時間框架。\n\n"
    "【原則】\n"
    "· 相對思維：有多檔標的時要互相比較與排名，不要各講各的。\n"
    "· 風險優先：談機會必談下檔。\n"
    "· 主動：若資料不足以支撐前瞻判斷，明說還需要哪些資料（如回測、選擇權、內部人、總經）。\n"
    "· 繁體中文，精簡專業，善用分段。\n"
    "· 分析與教育用途，非投資建議；不下買賣指令式結論。"
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
    out = detect_intents("NVDA 未來前景怎麼看", True)
    print(" ", out, "(前瞻→技術+基本面+總經)")
    assert {"outlook", "technical", "fundamental", "macro"} <= out
    out0 = detect_intents("美股後市怎麼看", False)
    print(" ", out0, "(無標的前瞻→大盤+總經，不會空 context)")
    assert {"market", "macro"} <= out0

    print("\n=== build_context（含趨勢/催化劑/同業）===")
    td = {"AAPL": {"tech": {"price": 190, "return_1m": 0.05, "rsi": 62, "ann_vol": 0.25,
                            "return_3m": 0.12, "max_dd": -0.08,
                            "pct_from_52w_high": -0.06, "pct_from_52w_low": 0.35,
                            "vs_ma50": 0.03, "vs_ma200": 0.11},
                   "fund": {"health": 82, "pe": 29, "roe": 1.5, "net_margin": 0.25,
                            "revenue_growth": 0.08},
                   "earnings": "2026-08-01",
                   "peers_note": "科技硬體 8 檔中，近3月報酬排 2/8、P/E 排 5/8（偏貴）"}}
    ctx = build_context("AAPL 前景", "2026-07-02 10:00",
                        {"outlook", "technical", "fundamental", "macro"}, ticker_data=td)
    print(ctx)
    assert "趨勢：" in ctx and "催化劑" in ctx and "同業對比" in ctx
