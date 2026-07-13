"""
taifex.py — 台灣期交所（TAIFEX）三大法人台指期部位 + 選擇權 P/C 比
RBS Finance Dashboard

台股最強免費籌碼訊號之一：
- 外資/投信/自營 台指期「未平倉淨額」（多空口數淨額）與 5 日變化
  —— 與 TWSE 現貨三大法人買賣超互相印證（現貨買+期貨空=避險 vs 真看多）
- 全市場選擇權 Put/Call 比（成交量比 + 未平倉量比）

資料源：TAIFEX 官網 CSV 下載端點（免金鑰、政府公開資料）
  期貨三大法人：POST https://www.taifex.com.tw/cht/3/futContractsDateDown
  P/C 比：      POST https://www.taifex.com.tw/cht/3/pcRatioDown
編碼通常為 Big5/CP950。欄位用「關鍵字寬鬆匹配」以抗官方欄名微調。

純邏輯（parse_fut_csv / parse_pc_csv / inst_summary / taifex_text）離線可測；
fetch_* 需網路（本開發環境 proxy 會 403——不是 bug）。教育用途，非投資建議。
"""
from __future__ import annotations

import datetime as _dt

FUT_URL = "https://www.taifex.com.tw/cht/3/futContractsDateDown"
PC_URL = "https://www.taifex.com.tw/cht/3/pcRatioDown"
PRODUCT_TX = "臺股期貨"          # 大台；小台=小型臺指期貨
IDENTITIES = ("外資", "投信", "自營商")


# ── 純邏輯：CSV 解析（欄位寬鬆匹配）──────────────────────────────────────────

def _split_csv(text: str) -> list[list[str]]:
    """標準 csv.reader（引號內的千分位逗號不可用 split 硬切）。"""
    import csv
    import io
    rows = []
    for r in csv.reader(io.StringIO((text or "").replace("\r", ""))):
        if r and any(c.strip() for c in r):
            rows.append([c.strip() for c in r])
    return rows


def _col(headers: list[str], *keywords) -> int | None:
    """回第一個「包含全部關鍵字」的欄位索引。"""
    for i, h in enumerate(headers):
        if all(k in h for k in keywords):
            return i
    return None


def _num(s) -> float | None:
    try:
        return float(str(s).replace(",", "").replace("%", "").strip())
    except (TypeError, ValueError):
        return None


def parse_fut_csv(text: str, product: str = PRODUCT_TX) -> list[dict]:
    """
    期貨三大法人 CSV → [{date, identity, net_oi, long_oi, short_oi}]（僅指定商品）。
    預期欄位（寬鬆匹配）：日期 / 商品名稱 / 身份別 /
    多方未平倉口數 / 空方未平倉口數 / 多空未平倉口數淨額。
    """
    rows = _split_csv(text)
    if len(rows) < 2:
        return []
    hd = rows[0]
    i_date = _col(hd, "日期")
    i_prod = _col(hd, "商品")
    i_id = _col(hd, "身份") if _col(hd, "身份") is not None else _col(hd, "身分")
    i_long = _col(hd, "多方", "未平倉", "口數")
    i_short = _col(hd, "空方", "未平倉", "口數")
    i_net = _col(hd, "多空", "未平倉", "淨額")
    if i_net is None:
        i_net = _col(hd, "多空", "未平倉", "口數")
    if None in (i_date, i_prod, i_id) or (i_net is None and None in (i_long, i_short)):
        return []
    out = []
    for r in rows[1:]:
        if len(r) <= max(x for x in (i_date, i_prod, i_id, i_long, i_short, i_net)
                         if x is not None):
            continue
        prod = r[i_prod]
        if product and product not in prod:
            continue
        ident = r[i_id].replace("外資及陸資", "外資")
        # 統一身份標籤
        for std in IDENTITIES:
            if std in ident:
                ident = std
                break
        lo = _num(r[i_long]) if i_long is not None else None
        so = _num(r[i_short]) if i_short is not None else None
        net = _num(r[i_net]) if i_net is not None else None
        if net is None and lo is not None and so is not None:
            net = lo - so
        if net is None:
            continue
        out.append({"date": r[i_date].replace("/", "-"), "identity": ident,
                    "net_oi": int(net),
                    "long_oi": int(lo) if lo is not None else None,
                    "short_oi": int(so) if so is not None else None})
    return out


def parse_pc_csv(text: str) -> list[dict]:
    """
    P/C 比 CSV → [{date, put_vol, call_vol, vol_ratio, put_oi, call_oi, oi_ratio}]。
    預期欄位：日期 / 賣權成交量 / 買權成交量 / 買賣權成交量比率% /
    賣權未平倉量 / 買權未平倉量 / 買賣權未平倉量比率%。
    """
    rows = _split_csv(text)
    if len(rows) < 2:
        return []
    hd = rows[0]
    i_date = _col(hd, "日期")
    i_pv = _col(hd, "賣權", "成交量")
    i_cv = _col(hd, "買權", "成交量")
    i_vr = _col(hd, "成交量", "比率")
    i_po = _col(hd, "賣權", "未平倉")
    i_co = _col(hd, "買權", "未平倉")
    i_or = _col(hd, "未平倉", "比率")
    if None in (i_date, i_po, i_co):
        return []
    out = []
    for r in rows[1:]:
        if len(r) <= max(x for x in (i_date, i_pv, i_cv, i_vr, i_po, i_co, i_or)
                         if x is not None):
            continue
        rec = {"date": r[i_date].replace("/", "-"),
               "put_vol": _num(r[i_pv]) if i_pv is not None else None,
               "call_vol": _num(r[i_cv]) if i_cv is not None else None,
               "vol_ratio": _num(r[i_vr]) if i_vr is not None else None,
               "put_oi": _num(r[i_po]), "call_oi": _num(r[i_co]),
               "oi_ratio": _num(r[i_or]) if i_or is not None else None}
        if rec["oi_ratio"] is None and rec["put_oi"] and rec["call_oi"]:
            rec["oi_ratio"] = round(rec["put_oi"] / rec["call_oi"] * 100, 2)
        if rec["date"]:
            out.append(rec)
    return out


def inst_summary(fut_rows: list[dict]) -> dict:
    """
    多日期貨列 → 各身份別 {net_oi(最新), chg_5d, dates}。
    chg_5d：最新 vs 往前第 5 個「交易日」（資料不足 5 日則用最舊一筆）。
    """
    by_id: dict[str, dict[str, int]] = {}
    for r in fut_rows:
        by_id.setdefault(r["identity"], {})[r["date"]] = r["net_oi"]
    out = {}
    for ident, série in by_id.items():
        dates = sorted(série)
        if not dates:
            continue
        latest = série[dates[-1]]
        base = série[dates[-6]] if len(dates) >= 6 else série[dates[0]]
        out[ident] = {"net_oi": latest,
                      "chg_5d": latest - base if len(dates) >= 2 else None,
                      "as_of": dates[-1], "days": len(dates)}
    return out


def taifex_text(summ: dict, pc_rows: list[dict]) -> str:
    """Bot / 晨報文字。"""
    lines = ["🇹🇼 *台指期籌碼（三大法人）*"]
    if summ:
        as_of = next(iter(summ.values()))["as_of"]
        lines[0] += f"　{as_of}"
        for ident in ("外資", "投信", "自營商"):
            d = summ.get(ident)
            if not d:
                continue
            arrow = "🟢" if d["net_oi"] > 0 else ("🔴" if d["net_oi"] < 0 else "⚪")
            seg = f"{arrow} {ident} 淨未平倉 {d['net_oi']:+,} 口"
            if d.get("chg_5d") is not None:
                seg += f"（5日 {d['chg_5d']:+,}）"
            lines.append(seg)
    if pc_rows:
        p = pc_rows[-1]
        if p.get("oi_ratio") is not None:
            bias = "偏空避險重" if p["oi_ratio"] < 90 else ("偏多" if p["oi_ratio"] > 110 else "中性")
            lines.append(f"⚖️ 選擇權 P/C 未平倉比 {p['oi_ratio']:.0f}%（{bias}）")
        if p.get("vol_ratio") is not None:
            lines.append(f"　　成交量比 {p['vol_ratio']:.0f}%")
    if len(lines) == 1:
        return "台指期籌碼：暫無資料（假日或資料源異常）"
    lines.append("_外資期現貨對照：現貨買+期貨空=避險；同買才是真看多。非投資建議。_")
    return "\n".join(lines)


# ── 抓取層（需網路；本地 proxy 會擋——不是 bug）────────────────────────────────

def _post_csv(url: str, form: dict, timeout: int = 20) -> str | None:
    import requests
    try:
        r = requests.post(url, data=form, timeout=timeout,
                          headers={"User-Agent": "Mozilla/5.0"})
        if not r.ok or not r.content:
            return None
        for enc in ("cp950", "big5", "utf-8"):
            try:
                return r.content.decode(enc)
            except UnicodeDecodeError:
                continue
        return r.content.decode("utf-8", errors="replace")
    except Exception:
        return None


def fetch_futures(days: int = 10) -> list[dict]:
    """近 N 日三大法人台指期。TAIFEX 查詢區間上限約 30 天。"""
    end = _dt.date.today()
    start = end - _dt.timedelta(days=days + 5)       # 含假日緩衝
    form = {"queryStartDate": start.strftime("%Y/%m/%d"),
            "queryEndDate": end.strftime("%Y/%m/%d"),
            "commodityId": "TXF"}
    text = _post_csv(FUT_URL, form)
    rows = parse_fut_csv(text) if text else []
    if not rows and text is not None:                # 有回應但解析不到 → 不帶商品過濾再試
        rows = parse_fut_csv(text, product="")
    return rows


def fetch_pc_ratio(days: int = 10) -> list[dict]:
    end = _dt.date.today()
    start = end - _dt.timedelta(days=days + 5)
    form = {"queryStartDate": start.strftime("%Y/%m/%d"),
            "queryEndDate": end.strftime("%Y/%m/%d")}
    text = _post_csv(PC_URL, form)
    return parse_pc_csv(text) if text else []


def fetch_summary() -> tuple[dict, list[dict]]:
    """端到端：期貨三大法人摘要 + P/C 序列。"""
    fut = fetch_futures()
    return inst_summary(fut), fetch_pc_ratio()


# ── CLI 自我測試（離線純邏輯，官方欄位格式合成樣本）────────────────────────────

if __name__ == "__main__":
    fut_csv = (
        "日期,商品名稱,身份別,多方交易口數,多方交易契約金額(千元),空方交易口數,"
        "空方交易契約金額(千元),多空交易口數淨額,多空交易契約金額淨額(千元),"
        "多方未平倉口數,多方未平倉契約金額(千元),空方未平倉口數,"
        "空方未平倉契約金額(千元),多空未平倉口數淨額,多空未平倉契約金額淨額(千元)\n"
        '2026/07/06,臺股期貨,自營商,"1,000",100,"1,200",120,-200,-20,"5,000",500,"6,000",600,"-1,000",-100\n'
        '2026/07/06,臺股期貨,投信,"300",30,"100",10,200,20,"2,500",250,"1,000",100,"1,500",150\n'
        '2026/07/06,臺股期貨,外資及陸資,"20,000",2000,"18,000",1800,"2,000",200,"90,000",9000,"70,000",7000,"20,000",2000\n'
        '2026/07/07,臺股期貨,外資及陸資,"21,000",2100,"18,500",1850,"2,500",250,"92,000",9200,"71,000",7100,"21,000",2100\n'
        '2026/07/07,小型臺指期貨,外資及陸資,"9,000",900,"9,500",950,-500,-50,"30,000",3000,"33,000",3300,"-3,000",-300\n'
        '2026/07/08,臺股期貨,外資及陸資,"22,000",2200,"18,000",1800,"4,000",400,"95,000",9500,"70,500",7050,"24,500",2450\n'
    )
    rows = parse_fut_csv(fut_csv)
    assert len(rows) == 5, rows                     # 小台被商品過濾排除
    fx = [r for r in rows if r["identity"] == "外資"]
    assert len(fx) == 3 and fx[-1]["net_oi"] == 24500, fx
    assert fx[0]["net_oi"] == 20000                 # 千分位逗號正確解析
    tw = [r for r in rows if r["identity"] == "投信"][0]
    assert tw["net_oi"] == 1500 and tw["long_oi"] == 2500

    summ = inst_summary(rows)
    assert summ["外資"]["net_oi"] == 24500 and summ["外資"]["as_of"] == "2026-07-08"
    assert summ["外資"]["chg_5d"] == 24500 - 20000   # 不足 6 日 → 用最舊一筆
    assert summ["自營商"]["chg_5d"] is None           # 單日無變化基準

    pc_csv = (
        "日期,賣權成交量,買權成交量,買賣權成交量比率%,賣權未平倉量,買權未平倉量,買賣權未平倉量比率%\n"
        '2026/07/07,"400,000","500,000",80.00,"250,000","300,000",83.33\n'
        '2026/07/08,"450,000","480,000",93.75,"270,000","290,000",93.10\n'
    )
    pcs = parse_pc_csv(pc_csv)
    assert len(pcs) == 2 and pcs[-1]["oi_ratio"] == 93.10 and pcs[0]["put_oi"] == 250000

    # 欄名缺「比率」欄 → 自算 OI 比
    pc2 = parse_pc_csv("日期,賣權未平倉量,買權未平倉量\n2026/07/08,90,100\n")
    assert pc2[0]["oi_ratio"] == 90.0

    txt = taifex_text(summ, pcs)
    assert "外資 淨未平倉 +24,500" in txt and "P/C 未平倉比 93%" in txt and "非投資建議" in txt
    assert taifex_text({}, []).startswith("台指期籌碼：暫無資料")

    # 壞輸入
    assert parse_fut_csv("") == [] and parse_pc_csv("garbage,no,headers\n1,2,3") == []

    print(f"✅ taifex 離線自我測試通過（外資淨 OI {summ['外資']['net_oi']:+,}、"
          f"P/C {pcs[-1]['oi_ratio']:.1f}%）")
