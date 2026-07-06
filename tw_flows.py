"""
tw_flows.py – 台股三大法人買賣超（TWSE 免 key 公開 API）

外資/投信/自營商是台股最重要的籌碼面。資料源：
  TWSE T86 https://www.twse.com.tw/rwd/zh/fund/T86?date=YYYYMMDD&selectType=ALLBUT0999&response=json
  （上市個股單日全表；免 key。欄位順序以回傳的 fields 為準，勿寫死索引。）
限制：v1 僅涵蓋上市（TWSE）；上櫃（TPEX）端點格式不同，之後再擴充。
單位：股（1 張 = 1000 股）。解析層純函數離線可測；端點格式需部署後實測（開發環境斷網）。
分析教育用途，非投資建議。
"""

from __future__ import annotations

T86_URL = "https://www.twse.com.tw/rwd/zh/fund/T86"

# 我們需要的欄位——以「欄名前綴」對應（網查證實的實際欄名；經消費端程式碼交叉驗證）：
# · 總計欄是「三大法人買賣超股數」（無「合計」——那是上櫃 TPEx 版的命名）
# · 必須用**前綴**比對：「自營商買賣超股數」是「外資自營商買賣超股數」的子字串，
#   用包含比對會錯抓到外資自營商欄
_FIELD_KEYS = {
    "foreign": "外陸資買賣超股數",     # 前綴匹配 → 外陸資買賣超股數(不含外資自營商)
    "trust":   "投信買賣超股數",
    "dealer":  "自營商買賣超股數",     # 前綴匹配 → 合計欄（在 自行買賣/避險 子欄之前）
    "total":   "三大法人買賣超股數",
}


def _num(s) -> float | None:
    """TWSE 數字帶千分位逗號、可能為空字串。"""
    try:
        return float(str(s).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def parse_t86(payload: dict, stock_no: str) -> dict | None:
    """
    從 T86 單日全表 JSON 取出某檔股票的法人買賣超（純函數）。
    回 {"foreign","trust","dealer","total"}（單位：股）；查無該股或格式異常回 None。
    """
    if not payload or payload.get("stat") not in ("OK", "ok"):
        return None
    fields = payload.get("fields") or []
    data = payload.get("data") or []
    if not fields or not data:
        return None
    # 欄名 → 索引（前綴比對；取第一個符合者）
    idx: dict = {}
    for key, kw in _FIELD_KEYS.items():
        for i, f in enumerate(fields):
            if str(f).replace(" ", "").startswith(kw):
                idx[key] = i
                break
    if "total" not in idx:
        return None
    code = stock_no.strip()
    for row in data:
        if str(row[0]).strip() == code:
            out = {k: _num(row[i]) for k, i in idx.items()}
            return out if out.get("total") is not None else None
    return None


def accumulate_flows(daily: list[dict]) -> dict:
    """
    多日法人資料彙總（純函數）。daily：[{date, foreign, trust, dealer, total}]（新→舊或舊→新皆可）。
    回：各法人累計張數、投信/外資連買天數（從最近一天往回數）。
    """
    if not daily:
        return {}
    rows = sorted([d for d in daily if d.get("total") is not None],
                  key=lambda d: d.get("date", ""))
    if not rows:
        return {}

    def _sum(key):
        vals = [d.get(key) for d in rows if d.get(key) is not None]
        return sum(vals) if vals else None

    def _streak(key):
        s = 0
        for d in reversed(rows):                 # 從最近一天往回
            v = d.get(key)
            if v is None or v <= 0:
                break
            s += 1
        return s

    def _lots(v):                                 # 股 → 張
        return v / 1000 if v is not None else None

    return {
        "days": len(rows),
        "foreign_lots": _lots(_sum("foreign")),
        "trust_lots": _lots(_sum("trust")),
        "dealer_lots": _lots(_sum("dealer")),
        "total_lots": _lots(_sum("total")),
        "foreign_streak": _streak("foreign"),
        "trust_streak": _streak("trust"),
        "last_date": rows[-1].get("date"),
    }


def flows_text(acc: dict) -> str | None:
    """組成中文摘要（給頁面/委員會 context）。"""
    if not acc or not acc.get("days"):
        return None

    def _fmt(v):
        return f"{v:+,.0f} 張" if v is not None else "無資料"

    parts = [f"近 {acc['days']} 個交易日三大法人合計 {_fmt(acc.get('total_lots'))}"
             f"（外資 {_fmt(acc.get('foreign_lots'))}、投信 {_fmt(acc.get('trust_lots'))}、"
             f"自營 {_fmt(acc.get('dealer_lots'))}）"]
    if acc.get("foreign_streak", 0) >= 3:
        parts.append(f"外資連 {acc['foreign_streak']} 日買超")
    if acc.get("trust_streak", 0) >= 3:
        parts.append(f"投信連 {acc['trust_streak']} 日買超（投信連買常為台股中小型股的強訊號）")
    return "；".join(parts) + f"（至 {acc.get('last_date', '?')}）"


# ── TPEX（上櫃）───────────────────────────────────────────────────────────────
# 端點回傳 aaData 純陣列（無欄名），採 hedge 版 24 欄的社群共識版面：
#   4=外資及陸資買賣超(不含外資自營商)、13=投信買賣超、22=自營商買賣超(合計)、
#   23=三大法人買賣超股數合計。有長度防護；版面若改版會整批回 None（待部署實測）。

TPEX_URL = "https://www.tpex.org.tw/web/stock/3insti/daily_trade/3itrade_hedge_result.php"
_TPEX_COLS = {"foreign": 4, "trust": 13, "dealer": 22, "total": 23}


def parse_tpex(payload: dict, stock_no: str) -> dict | None:
    """從 TPEX 上櫃法人日報 JSON 取某檔的買賣超（純函數）。查無/版面異常回 None。"""
    rows = (payload or {}).get("aaData") or []
    code = stock_no.strip()
    for row in rows:
        try:
            if len(row) >= 24 and str(row[0]).strip() == code:
                out = {k: _num(row[i]) for k, i in _TPEX_COLS.items()}
                return out if out.get("total") is not None else None
        except (TypeError, IndexError):
            continue
    return None


def _roc_date(d) -> str:
    return f"{d.year - 1911}/{d.month:02d}/{d.day:02d}"


# ── 抓取層（需網路；開發環境斷網，部署後實測）──────────────────────────────────

def _walk_back(fetch_one, days: int, lookback: int) -> list[dict]:
    """從今天往回逐日呼叫 fetch_one(date)→row|None，收滿 days 筆或試完 lookback 天。"""
    import datetime as _dt
    import time
    got: list[dict] = []
    d = _dt.date.today()
    tried = 0
    while len(got) < days and tried < lookback:
        try:
            row = fetch_one(d)
            if row is not None:
                row["date"] = d.isoformat()
                got.append(row)
            time.sleep(0.4)                       # 對交易所客氣點
        except Exception:
            pass
        d -= _dt.timedelta(days=1)
        tried += 1
    return got


def fetch_flows(ticker: str, days: int = 5, lookback: int = 12) -> dict | None:
    """
    抓某台股近 N 個交易日的法人買賣超並彙總。
    接受 2330 / 2330.TW / 6488.TWO。.TWO 直接查上櫃；其餘先查上市（TWSE），
    整段查無再退上櫃（TPEX）。抓不到回 None。
    """
    import requests
    t = ticker.upper().strip()
    otc_first = t.endswith(".TWO")
    code = t.replace(".TWO", "").replace(".TW", "")
    if not code.isdigit():
        return None
    sess = requests.Session()
    sess.headers.update({"User-Agent": "Mozilla/5.0"})

    def _twse_one(d):
        r = sess.get(T86_URL, params={"date": d.strftime("%Y%m%d"),
                                      "selectType": "ALLBUT0999",
                                      "response": "json"}, timeout=15)
        return parse_t86(r.json(), code) if r.ok else None

    def _tpex_one(d):
        r = sess.get(TPEX_URL, params={"l": "zh-tw", "o": "json",
                                       "se": "EW", "t": "D",
                                       "d": _roc_date(d)}, timeout=15)
        return parse_tpex(r.json(), code) if r.ok else None

    order = ([(_tpex_one, "TPEX")] if otc_first
             else [(_twse_one, "TWSE"), (_tpex_one, "TPEX")])
    for fetch_one, market in order:
        got = _walk_back(fetch_one, days, lookback)
        if got:
            acc = accumulate_flows(got)
            acc["ticker"] = f"{code}.{'TWO' if market == 'TPEX' else 'TW'}"
            acc["market"] = market
            return acc
    return None


# ── CLI 自我測試（純解析）─────────────────────────────────────────────────────

if __name__ == "__main__":
    PAYLOAD = {
        "stat": "OK",
        "fields": ["證券代號", "證券名稱",
                   "外陸資買進股數(不含外資自營商)", "外陸資賣出股數(不含外資自營商)",
                   "外陸資買賣超股數(不含外資自營商)",
                   "外資自營商買進股數", "外資自營商賣出股數", "外資自營商買賣超股數",
                   "投信買進股數", "投信賣出股數", "投信買賣超股數",
                   "自營商買賣超股數", "自營商買進股數(自行買賣)", "自營商賣出股數(自行買賣)",
                   "自營商買賣超股數(自行買賣)", "自營商買進股數(避險)", "自營商賣出股數(避險)",
                   "自營商買賣超股數(避險)", "三大法人買賣超股數"],
        "data": [
            ["2330", "台積電", "30,000,000", "20,000,000", "10,000,000",
             "0", "0", "0", "2,000,000", "500,000", "1,500,000",
             "300,000", "0", "0", "0", "0", "0", "0", "11,800,000"],
            ["2317", "鴻海", "5,000,000", "8,000,000", "-3,000,000",
             "0", "0", "0", "100,000", "600,000", "-500,000",
             "-100,000", "0", "0", "0", "0", "0", "0", "-3,600,000"],
        ],
    }
    row = parse_t86(PAYLOAD, "2330")
    print("2330:", row)
    assert row["foreign"] == 10_000_000 and row["trust"] == 1_500_000
    assert row["total"] == 11_800_000
    assert row["dealer"] == 300_000        # 必須抓到合計欄(idx11)，不是外資自營商欄(idx7=0)
    row2 = parse_t86(PAYLOAD, "2317")
    assert row2["total"] == -3_600_000
    assert parse_t86(PAYLOAD, "9999") is None
    assert parse_t86({"stat": "很抱歉，沒有符合條件的資料!"}, "2330") is None

    daily = [
        {"date": "2026-07-01", "foreign": 1_000_000, "trust": 200_000, "dealer": 0, "total": 1_200_000},
        {"date": "2026-07-02", "foreign": 2_000_000, "trust": 300_000, "dealer": -100_000, "total": 2_200_000},
        {"date": "2026-07-03", "foreign": -500_000, "trust": 400_000, "dealer": 0, "total": -100_000},
        {"date": "2026-07-04", "foreign": 800_000, "trust": 500_000, "dealer": 0, "total": 1_300_000},
    ]
    acc = accumulate_flows(daily)
    print("acc:", {k: v for k, v in acc.items() if k != "last_date"})
    assert acc["days"] == 4
    assert abs(acc["foreign_lots"] - 3300) < 1e-9        # (1M+2M-0.5M+0.8M)/1000
    assert acc["trust_streak"] == 4                      # 投信連 4 買
    assert acc["foreign_streak"] == 1                    # 外資最近 1 日買（前一日賣）
    txt = flows_text(acc)
    print(txt)
    assert "投信連 4 日買超" in txt and "+3,300 張" in txt
    assert flows_text({}) is None

    # TPEX 24 欄 hedge 版
    tpex_row = ["6488", "環球晶"] + ["0"] * 22
    tpex_row[4], tpex_row[13], tpex_row[22], tpex_row[23] = \
        "1,200,000", "300,000", "-50,000", "1,450,000"
    tp = parse_tpex({"aaData": [tpex_row]}, "6488")
    print("tpex 6488:", tp)
    assert tp["foreign"] == 1_200_000 and tp["trust"] == 300_000
    assert tp["dealer"] == -50_000 and tp["total"] == 1_450_000
    assert parse_tpex({"aaData": [["6488", "短列"]]}, "6488") is None   # 欄數不足防護
    assert parse_tpex({}, "6488") is None
    import datetime as _dt
    assert _roc_date(_dt.date(2026, 7, 5)) == "115/07/05"

    print("\n✅ tw_flows 純解析測試通過")
