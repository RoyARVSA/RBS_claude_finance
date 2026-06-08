"""
stock_db.py – Institutional Stock Selection Database
RBS Finance Dashboard

ADB structure:
  ADB[market][industry] = {
      "tickers":    [list of ticker symbols],
      "strats":     [compatible strategy keys],
      "macro":      [macro conditions that favour this industry],
      "asset_type": "股票" | "ETF" | "REIT",
      "desc":       short description (Traditional Chinese),
  }
"""

# ─────────────────────────── Markets ────────────────────────────────

MKTS = {
    "🇺🇸 美股 US":   "US",
    "🇹🇼 台股 TW":   "TW",
    "🇭🇰 港股 HK":   "HK",
    "🇯🇵 日股 JP":   "JP",
    "🌐 全球 ETF":    "ETF",
}

# ─────────────────────────── Strategies ─────────────────────────────

STRATS = {
    "growth":    {"label": "🚀 成長型", "desc": "追求高成長、高本益比標的，適合景氣擴張期"},
    "value":     {"label": "💎 價值型", "desc": "尋找被低估、具安全邊際的標的"},
    "dividend":  {"label": "💰 高股息", "desc": "穩定配息、高殖利率，適合防禦型配置"},
    "momentum":  {"label": "⚡ 動能型", "desc": "追蹤強勢趨勢與技術突破標的"},
    "defensive": {"label": "🛡️ 防禦型", "desc": "低波動、抗景氣循環，保護資本"},
    "macro":     {"label": "🌍 宏觀主題", "desc": "受總體經濟趨勢驅動的主題投資"},
}

# ─────────────────────────── Macro Factors ──────────────────────────

MACRO_FACTORS = [
    "升息環境",
    "降息環境",
    "高通膨",
    "通膨降溫",
    "景氣擴張",
    "景氣衰退",
    "強美元",
    "弱美元",
    "AI 浪潮",
    "科技週期上行",
    "地緣風險",
    "能源危機",
    "信用緊縮",
    "流動性充裕",
]

# Maps macro factor → industries that benefit
MACRO_BOOST: dict[str, list[str]] = {
    "升息環境":   ["金融銀行", "保險", "公用事業", "固定收益ETF"],
    "降息環境":   ["成長科技", "房地產REIT", "公用事業", "消費品牌", "全球ETF"],
    "高通膨":     ["能源油氣", "原物料商品", "農業ETF", "黃金/貴金屬ETF", "REITs"],
    "通膨降溫":   ["成長科技", "生技醫療", "消費品牌", "固定收益ETF"],
    "景氣擴張":   ["半導體", "大型科技", "工業製造", "消費品牌", "金融銀行"],
    "景氣衰退":   ["公用事業", "醫療設備", "固定收益ETF", "黃金/貴金屬ETF", "防禦型ETF"],
    "強美元":     ["美國大型股ETF", "金融銀行", "公用事業"],
    "弱美元":     ["新興市場ETF", "能源油氣", "黃金/貴金屬ETF", "台股科技", "港股科技"],
    "AI 浪潮":    ["半導體", "大型科技", "雲端/SaaS", "AI應用", "科技ETF"],
    "科技週期上行": ["半導體", "IC設計", "半導體製造", "科技ETF"],
    "地緣風險":   ["國防工業", "黃金/貴金屬ETF", "能源油氣", "公用事業"],
    "能源危機":   ["能源油氣", "再生能源", "農業ETF"],
    "信用緊縮":   ["固定收益ETF", "公用事業", "黃金/貴金屬ETF", "防禦型ETF"],
    "流動性充裕": ["成長科技", "新興市場ETF", "消費品牌", "房地產REIT"],
}

# ─────────────────────────── Asset DB ───────────────────────────────

ADB: dict[str, dict[str, dict]] = {

    # ════════════════ US ════════════════
    "US": {
        "半導體": {
            "tickers": ["NVDA", "AMD", "AVGO", "QCOM", "TXN", "MRVL",
                        "AMAT", "LRCX", "KLAC", "ASML", "MU", "INTC"],
            "strats":  ["growth", "momentum", "macro"],
            "macro":   ["AI 浪潮", "科技週期上行", "景氣擴張"],
            "asset_type": "股票",
            "desc": "全球半導體設計與設備龍頭，受惠AI及HPC需求爆發",
        },
        "大型科技": {
            "tickers": ["AAPL", "MSFT", "GOOGL", "META", "AMZN", "NFLX"],
            "strats":  ["growth", "momentum", "value"],
            "macro":   ["AI 浪潮", "降息環境", "景氣擴張", "流動性充裕"],
            "asset_type": "股票",
            "desc": "美國科技巨頭，具備護城河與多元營收來源",
        },
        "雲端/SaaS": {
            "tickers": ["CRM", "NOW", "SNOW", "PLTR", "DDOG", "NET",
                        "MDB", "ZS", "OKTA", "PANW", "WDAY"],
            "strats":  ["growth", "momentum", "macro"],
            "macro":   ["AI 浪潮", "景氣擴張", "降息環境", "流動性充裕"],
            "asset_type": "股票",
            "desc": "雲端軟體訂閱模式，受惠企業數位轉型",
        },
        "AI應用": {
            "tickers": ["PLTR", "AI", "SOUN", "BBAI", "RXRX", "ARQT"],
            "strats":  ["growth", "momentum", "macro"],
            "macro":   ["AI 浪潮", "科技週期上行"],
            "asset_type": "股票",
            "desc": "純AI應用及AI基礎設施廠商，高成長高風險",
        },
        "生技醫療": {
            "tickers": ["LLY", "NVO", "ABBV", "AMGN", "REGN", "GILD",
                        "MRNA", "VRTX", "BIIB", "BMY"],
            "strats":  ["growth", "value", "defensive"],
            "macro":   ["通膨降溫", "景氣衰退", "降息環境"],
            "asset_type": "股票",
            "desc": "製藥與生技，受惠人口老化及GLP-1等創新藥物",
        },
        "醫療設備": {
            "tickers": ["MDT", "ABT", "SYK", "ISRG", "EW", "BSX",
                        "BDX", "ZBH", "DXCM", "PODD"],
            "strats":  ["defensive", "value", "growth"],
            "macro":   ["通膨降溫", "景氣衰退", "降息環境"],
            "asset_type": "股票",
            "desc": "醫療設備製造商，穩定需求，防禦特性佳",
        },
        "金融銀行": {
            "tickers": ["JPM", "BAC", "GS", "MS", "WFC", "BLK",
                        "C", "SCHW", "USB", "PNC"],
            "strats":  ["value", "dividend", "macro"],
            "macro":   ["升息環境", "景氣擴張", "強美元"],
            "asset_type": "股票",
            "desc": "大型投資銀行及商業銀行，受惠淨利差擴大",
        },
        "保險": {
            "tickers": ["BRK-B", "CB", "PGR", "AIG", "TRV", "ALL", "MET", "PRU"],
            "strats":  ["value", "dividend", "defensive"],
            "macro":   ["升息環境", "景氣擴張"],
            "asset_type": "股票",
            "desc": "保險龍頭，具備浮存金優勢，升息環境獲益",
        },
        "能源油氣": {
            "tickers": ["XOM", "CVX", "COP", "SLB", "EOG", "DVN",
                        "OXY", "PSX", "VLO", "MPC"],
            "strats":  ["value", "dividend", "macro"],
            "macro":   ["高通膨", "能源危機", "地緣風險"],
            "asset_type": "股票",
            "desc": "整合石油公司及油服業者，油價上漲受惠標的",
        },
        "再生能源": {
            "tickers": ["NEE", "ENPH", "FSLR", "RUN", "BEP", "CWEN", "PLUG"],
            "strats":  ["growth", "macro", "defensive"],
            "macro":   ["降息環境", "能源危機", "流動性充裕"],
            "asset_type": "股票",
            "desc": "太陽能、風電及儲能廠商，長期政策驅動成長",
        },
        "消費品牌": {
            "tickers": ["AMZN", "TGT", "WMT", "COST", "HD", "LOW",
                        "NKE", "SBUX", "MCD", "YUM"],
            "strats":  ["value", "dividend", "defensive", "growth"],
            "macro":   ["降息環境", "通膨降溫", "景氣擴張", "流動性充裕"],
            "asset_type": "股票",
            "desc": "必需/非必需消費龍頭，多元通路與品牌護城河",
        },
        "工業製造": {
            "tickers": ["CAT", "HON", "DE", "GE", "MMM", "EMR", "ETN", "PH"],
            "strats":  ["value", "dividend", "macro"],
            "macro":   ["景氣擴張", "通膨降溫"],
            "asset_type": "股票",
            "desc": "工業龍頭，受惠製造業回流及基礎設施投資",
        },
        "國防工業": {
            "tickers": ["LMT", "RTX", "NOC", "GD", "BA", "L3H", "LDOS"],
            "strats":  ["defensive", "value", "macro"],
            "macro":   ["地緣風險", "景氣衰退"],
            "asset_type": "股票",
            "desc": "航太國防龍頭，地緣風險升溫受惠標的",
        },
        "房地產REIT": {
            "tickers": ["AMT", "PLD", "EQIX", "O", "SPG", "VICI",
                        "DLR", "CCI", "WELL", "AVB"],
            "strats":  ["dividend", "defensive", "value"],
            "macro":   ["降息環境", "高通膨", "流動性充裕"],
            "asset_type": "REIT",
            "desc": "多元REITs：數位基礎設施、物流、零售、醫療",
        },
        "公用事業": {
            "tickers": ["NEE", "DUK", "SO", "AEP", "EXC", "D", "ED", "FE"],
            "strats":  ["defensive", "dividend"],
            "macro":   ["升息環境", "景氣衰退", "地緣風險"],
            "asset_type": "股票",
            "desc": "電力/天然氣公用事業，穩定現金流，防禦屬性強",
        },
        "美股寬基ETF": {
            "tickers": ["SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "SCHB"],
            "strats":  ["growth", "defensive", "momentum"],
            "macro":   ["景氣擴張", "降息環境", "流動性充裕"],
            "asset_type": "ETF",
            "desc": "追蹤S&P500/NASDAQ/Russell的核心指數ETF",
        },
        "板塊ETF": {
            "tickers": ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLRE", "XLB", "XLC"],
            "strats":  ["macro", "momentum", "defensive"],
            "macro":   ["景氣擴張", "景氣衰退", "高通膨", "降息環境"],
            "asset_type": "ETF",
            "desc": "SPDR板塊ETF，精準暴露特定行業",
        },
        "主題/槓桿ETF": {
            "tickers": ["ARKK", "ARKW", "SMH", "SOXX", "BOTZ", "ROBO",
                        "SOXL", "TQQQ", "UPRO", "SPXL"],
            "strats":  ["growth", "momentum"],
            "macro":   ["AI 浪潮", "科技週期上行", "流動性充裕"],
            "asset_type": "ETF",
            "desc": "主題/槓桿ETF，高彈性高風險，適合短線動量策略",
        },
        "固收/對沖ETF": {
            "tickers": ["TLT", "IEF", "AGG", "BND", "HYG", "LQD", "GLD", "SLV", "USO", "UNG"],
            "strats":  ["defensive", "dividend", "macro"],
            "macro":   ["景氣衰退", "升息環境", "降息環境", "地緣風險", "高通膨"],
            "asset_type": "ETF",
            "desc": "債券/商品ETF，作為股票組合的對沖或穩定收益來源",
        },
    },

    # ════════════════ TW ════════════════
    "TW": {
        "半導體製造": {
            "tickers": ["2330.TW", "2303.TW", "2308.TW", "5347.TW", "3034.TW"],
            "strats":  ["growth", "momentum", "macro"],
            "macro":   ["AI 浪潮", "科技週期上行", "弱美元"],
            "asset_type": "股票",
            "desc": "台積電領軍的晶圓代工體系，全球最先進製程",
        },
        "IC設計": {
            "tickers": ["2454.TW", "3711.TW", "2337.TW", "6770.TW", "4966.TW",
                        "3034.TW", "2379.TW", "6488.TW"],
            "strats":  ["growth", "momentum"],
            "macro":   ["AI 浪潮", "科技週期上行", "弱美元"],
            "asset_type": "股票",
            "desc": "聯發科為首的IC設計業者，受惠AI邊緣運算",
        },
        "封測": {
            "tickers": ["2049.TW", "2478.TW", "6415.TW", "3711.TW"],
            "strats":  ["value", "dividend"],
            "macro":   ["科技週期上行", "景氣擴張"],
            "asset_type": "股票",
            "desc": "封裝測試龍頭，CoWoS先進封裝需求暴增",
        },
        "伺服器/AI硬體": {
            "tickers": ["2376.TW", "3231.TW", "2356.TW", "6669.TW", "3017.TW"],
            "strats":  ["growth", "momentum", "macro"],
            "macro":   ["AI 浪潮", "科技週期上行"],
            "asset_type": "股票",
            "desc": "伺服器機櫃、散熱、電源供應受惠AI資料中心建置",
        },
        "電子零組件": {
            "tickers": ["2317.TW", "2382.TW", "2354.TW", "2353.TW", "3045.TW"],
            "strats":  ["value", "dividend", "momentum"],
            "macro":   ["景氣擴張", "科技週期上行"],
            "asset_type": "股票",
            "desc": "鴻海、廣達等EMS/ODM大廠及連接器廠",
        },
        "台股金融": {
            "tickers": ["2891.TW", "2882.TW", "2881.TW", "2886.TW",
                        "2884.TW", "2892.TW", "5880.TW"],
            "strats":  ["dividend", "value"],
            "macro":   ["升息環境", "景氣擴張"],
            "asset_type": "股票",
            "desc": "台灣銀行、保險及金控，高股息殖利率",
        },
        "台股傳產": {
            "tickers": ["1301.TW", "1303.TW", "1326.TW", "1101.TW", "2002.TW"],
            "strats":  ["value", "dividend"],
            "macro":   ["高通膨", "景氣擴張"],
            "asset_type": "股票",
            "desc": "石化、鋼鐵、水泥等傳統產業，景氣循環股",
        },
    },

    # ════════════════ HK ════════════════
    "HK": {
        "港股科技": {
            "tickers": ["0700.HK", "9988.HK", "9999.HK", "3690.HK",
                        "1810.HK", "9618.HK"],
            "strats":  ["growth", "momentum", "value"],
            "macro":   ["弱美元", "流動性充裕", "降息環境"],
            "asset_type": "股票",
            "desc": "騰訊、阿里、美團等中國互聯網龍頭，估值修復機會",
        },
        "港股金融": {
            "tickers": ["0005.HK", "0011.HK", "0939.HK", "1398.HK",
                        "3968.HK", "2318.HK"],
            "strats":  ["dividend", "value"],
            "macro":   ["升息環境", "景氣擴張"],
            "asset_type": "股票",
            "desc": "滙豐、恒生、中行等中港大型銀行，高息優先",
        },
        "港股地產": {
            "tickers": ["0016.HK", "0001.HK", "0823.HK", "1997.HK", "0688.HK"],
            "strats":  ["value", "dividend"],
            "macro":   ["降息環境", "流動性充裕"],
            "asset_type": "股票",
            "desc": "香港地產商及REITs，降息週期估值修復",
        },
        "港股消費": {
            "tickers": ["9901.HK", "2020.HK", "6862.HK", "0291.HK", "0762.HK"],
            "strats":  ["growth", "value"],
            "macro":   ["流動性充裕", "弱美元"],
            "asset_type": "股票",
            "desc": "安踏、波司登、海底撈等消費品牌，中國復甦受惠",
        },
    },

    # ════════════════ JP ════════════════
    "JP": {
        "日股科技": {
            "tickers": ["6758.T", "6501.T", "8035.T", "6702.T",
                        "6954.T", "7974.T", "4661.T"],
            "strats":  ["growth", "momentum", "value"],
            "macro":   ["弱美元", "AI 浪潮", "科技週期上行"],
            "asset_type": "股票",
            "desc": "Sony、日立、東京威力、發那科等日本科技與機器人",
        },
        "日股汽車": {
            "tickers": ["7203.T", "7267.T", "7269.T", "7201.T", "7211.T"],
            "strats":  ["value", "dividend"],
            "macro":   ["弱美元", "景氣擴張"],
            "asset_type": "股票",
            "desc": "豐田、本田、Subaru等車廠，弱日圓出口受惠",
        },
        "日股金融": {
            "tickers": ["8306.T", "8316.T", "8411.T", "8591.T"],
            "strats":  ["value", "dividend"],
            "macro":   ["升息環境", "景氣擴張"],
            "asset_type": "股票",
            "desc": "三菱UFJ、三井住友等大型銀行，日本升息周期受惠",
        },
        "日股消費": {
            "tickers": ["9983.T", "9984.T", "3382.T", "2914.T"],
            "strats":  ["growth", "value"],
            "macro":   ["弱美元", "流動性充裕"],
            "asset_type": "股票",
            "desc": "迅銷、軟銀、Seven & i等日本消費零售龍頭",
        },
    },

    # ════════════════ ETF ════════════════
    "ETF": {
        "美股大盤ETF": {
            "tickers": ["SPY", "QQQ", "VOO", "IVV", "VTI", "DIA"],
            "strats":  ["growth", "value", "momentum", "defensive"],
            "macro":   ["景氣擴張", "降息環境", "流動性充裕"],
            "asset_type": "ETF",
            "desc": "追蹤S&P500、那斯達克、道瓊等主要指數",
        },
        "科技ETF": {
            "tickers": ["SMH", "SOXX", "XSD", "IGV", "HACK", "XLK", "VGT"],
            "strats":  ["growth", "momentum", "macro"],
            "macro":   ["AI 浪潮", "科技週期上行", "景氣擴張"],
            "asset_type": "ETF",
            "desc": "半導體、網路安全、軟體等科技主題ETF",
        },
        "新興市場ETF": {
            "tickers": ["EEM", "VWO", "KWEB", "EWZ", "EWJ", "FXI",
                        "MCHI", "EWT", "EWY"],
            "strats":  ["growth", "macro", "value"],
            "macro":   ["弱美元", "流動性充裕", "降息環境"],
            "asset_type": "ETF",
            "desc": "中國、台灣、韓國等新興亞洲市場ETF",
        },
        "固定收益ETF": {
            "tickers": ["AGG", "BND", "HYG", "LQD", "TLT", "SHY",
                        "IEF", "TIPS", "MBB"],
            "strats":  ["defensive", "dividend"],
            "macro":   ["升息環境", "景氣衰退", "信用緊縮"],
            "asset_type": "ETF",
            "desc": "美國國債、投資等級債、高收益債等固收ETF",
        },
        "黃金/貴金屬ETF": {
            "tickers": ["GLD", "IAU", "SLV", "GDX", "GDXJ", "PPLT"],
            "strats":  ["defensive", "macro"],
            "macro":   ["高通膨", "地緣風險", "弱美元", "景氣衰退"],
            "asset_type": "ETF",
            "desc": "黃金、白銀及礦業ETF，避險及通膨保值工具",
        },
        "農業ETF": {
            "tickers": ["DBA", "WEAT", "CORN", "SOYB", "MOO"],
            "strats":  ["macro", "defensive"],
            "macro":   ["高通膨", "能源危機", "地緣風險"],
            "asset_type": "ETF",
            "desc": "農業原物料及食品公司ETF，通膨避險",
        },
        "房地產ETF": {
            "tickers": ["VNQ", "SCHH", "IYR", "RWR", "REET"],
            "strats":  ["dividend", "defensive", "value"],
            "macro":   ["降息環境", "高通膨", "流動性充裕"],
            "asset_type": "ETF",
            "desc": "美國REITs ETF，涵蓋商業、住宅、工業地產",
        },
        "防禦型ETF": {
            "tickers": ["USMV", "SPLV", "XLP", "XLU", "XLV", "DVY", "VIG"],
            "strats":  ["defensive", "dividend"],
            "macro":   ["景氣衰退", "升息環境", "地緣風險", "信用緊縮"],
            "asset_type": "ETF",
            "desc": "低波動、必需消費、公用事業防禦型ETF",
        },
        "全球ETF": {
            "tickers": ["ACWI", "VT", "IXUS", "EFA", "IEFA", "VXUS"],
            "strats":  ["value", "growth", "defensive"],
            "macro":   ["降息環境", "流動性充裕", "弱美元"],
            "asset_type": "ETF",
            "desc": "全球股市分散配置ETF，一籃子覆蓋多國市場",
        },
        "多因子ETF": {
            "tickers": ["QUAL", "MTUM", "VLUE", "USMV", "DIVO", "SCHD"],
            "strats":  ["value", "momentum", "dividend", "defensive"],
            "macro":   ["景氣擴張", "通膨降溫"],
            "asset_type": "ETF",
            "desc": "品質、動能、價值、股息多因子Smart Beta ETF",
        },
    },
}

# ─────────────────────────── Insights ───────────────────────────────

INSIGHTS: dict[str, str] = {
    "升息環境":   "升息週期中，金融股及短天期債券具優勢，成長股估值承壓",
    "降息環境":   "降息有利科技成長股及REITs，帶動風險資產估值回升",
    "高通膨":     "通膨環境下，大宗商品、黃金及能源股具備保值功能",
    "通膨降溫":   "通膨下行有利於成長股及醫療，緩解供應鏈成本壓力",
    "景氣擴張":   "景氣上行期，周期性產業（工業、金融、消費）表現佳",
    "景氣衰退":   "衰退期宜轉向防禦型配置：公用事業、債券、黃金",
    "強美元":     "強美元壓制新興市場及大宗商品，利多美國大型股",
    "弱美元":     "弱美元有利出口導向市場（台股、日股）及新興市場",
    "AI 浪潮":    "AI算力需求帶動半導體、伺服器及雲端基礎設施超級週期",
    "科技週期上行": "半導體庫存去化完畢，進入新一輪補庫存及資本支出週期",
    "地緣風險":   "地緣衝突提升國防及能源需求，黃金避險需求升溫",
    "能源危機":   "能源短缺驅動傳統能源價格上漲，加速再生能源佈局",
    "信用緊縮":   "信用條件收緊，高評級債券及現金等值品種為佳選擇",
    "流動性充裕": "流動性寬鬆環境有利風險資產，成長股與新興市場受惠",
}

# Warnings by market
MWARN: dict[str, str] = {
    "US":  "美股估值偏高（Cyclically adjusted P/E > 30），需留意回調風險",
    "TW":  "台股集中度高，科技權重逾七成，景氣反轉時波動劇烈",
    "HK":  "港股受中國政策及地緣不確定性影響，流動性折價持續",
    "JP":  "日圓升值風險可能壓縮出口商獲利，需觀察日銀政策轉向",
    "ETF": "ETF費用率與追蹤誤差因產品而異，請確認底層資產流動性",
}

# Strategy avoidances
MAVOID: dict[str, str] = {
    "growth":    "避免高負債、負自由現金流、過度依賴融資的早期成長公司",
    "value":     "避免「價值陷阱」：業績持續惡化的低本益比股票",
    "dividend":  "避免殖利率過高但配息不可持續（payout ratio > 100%）的標的",
    "momentum":  "避免在技術面已過熱、RSI > 80且大量出貨的標的追高",
    "defensive": "避免具有高財務槓桿、現金流不穩定的所謂「防禦型」標的",
    "macro":     "避免過度集中單一宏觀主題，宏觀預測失準時損失集中",
}
