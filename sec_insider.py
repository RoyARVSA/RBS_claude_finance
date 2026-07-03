"""
sec_insider.py – SEC 內部人交易（Form 4 / EDGAR）

「內部人」＝公司董事、經理人（officers）、持股 10% 以上大股東。依美國證管會規定，
他們買賣自家股票須在 2 個營業日內申報 Form 4。公開市場買進（尤其多位內部人同時買）
常被視為偏多訊號；賣出訊號較弱（常為節稅/分散，非看空）。

解析層 parse_form4 / summarize_insiders 為純函數（可離線測試）；
抓取層走 EDGAR（免費、免 key，但須帶 User-Agent；此開發環境代理擋外網，部署後實測）。
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

BASE = "https://www.sec.gov"
DATA = "https://data.sec.gov"

# 交易代碼（Form 4 transactionCode 常見值）
CODE_DESC = {
    "P": "公開市場買進", "S": "公開市場賣出", "A": "獲授予/獎勵",
    "M": "選擇權行使", "F": "繳稅扣股", "G": "贈與", "X": "行使買權",
    "C": "轉換", "D": "處分", "J": "其他",
}


# ── 純解析 ─────────────────────────────────────────────────────────────────────

def _txt(node, path):
    """取子節點文字（Form 4 多為 <tag><value>X</value></tag> 或直接 <tag>X</tag>）。"""
    el = node.find(path)
    if el is None:
        return None
    v = el.find("value")
    s = (v.text if v is not None else el.text)
    return s.strip() if s and s.strip() else None


def _fnum(s):
    try:
        return float(s) if s not in (None, "") else None
    except (TypeError, ValueError):
        return None


def parse_form4(xml_text: str) -> dict:
    """
    解析單份 Form 4 XML → {issuer, owner, title, relationship, transactions:[...]}。
    只取非衍生（nonDerivative）交易（實際股數買賣），衍生（選擇權）另計數。
    每筆 transaction：{date, code, code_desc, shares, price, value, ad}（ad: A=取得 D=處分）。
    解析失敗回 {"ok": False}。
    """
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return {"ok": False, "transactions": []}

    issuer = _txt(root, "issuer/issuerTradingSymbol")

    owner = title = None
    rel = []
    ro = root.find("reportingOwner")
    if ro is not None:
        owner = _txt(ro, "reportingOwnerId/rptOwnerName")
        r = ro.find("reportingOwnerRelationship")
        if r is not None:
            if (_txt(r, "isDirector") or "").lower() in ("1", "true"):
                rel.append("董事")
            if (_txt(r, "isOfficer") or "").lower() in ("1", "true"):
                rel.append("經理人")
            if (_txt(r, "isTenPercentOwner") or "").lower() in ("1", "true"):
                rel.append("10%大股東")
            title = _txt(r, "officerTitle")

    txns = []
    for t in root.findall("nonDerivativeTable/nonDerivativeTransaction"):
        code = _txt(t, "transactionCoding/transactionCode")
        shares = _fnum(_txt(t, "transactionAmounts/transactionShares"))
        price = _fnum(_txt(t, "transactionAmounts/transactionPricePerShare"))
        ad = _txt(t, "transactionAmounts/transactionAcquiredDisposedCode")
        date = _txt(t, "transactionDate")
        value = (shares * price) if (shares is not None and price is not None) else None
        txns.append({
            "date": date, "code": code, "code_desc": CODE_DESC.get(code, code),
            "shares": shares, "price": price, "value": value, "ad": ad,
        })

    n_deriv = len(root.findall("derivativeTable/derivativeTransaction"))
    return {"ok": True, "issuer": issuer, "owner": owner, "title": title,
            "relationship": rel, "transactions": txns, "n_derivative": n_deriv}


def _within(date_str, cutoff):
    """date_str(YYYY-MM-DD) >= cutoff(YYYY-MM-DD 字串) ；cutoff 為 None 時全收。"""
    if cutoff is None:
        return True
    if not date_str:
        return False
    return date_str >= cutoff


def summarize_insiders(parsed: list, cutoff_date: str | None = None) -> dict:
    """
    彙總多份 Form 4（純函數）。cutoff_date：只計該日(含)之後的交易；None=全部。
    以『公開市場』買(P)賣(S)為主評估情緒；獎勵/繳稅等不計入買賣情緒。
    """
    buys, sells = [], []            # (owner, value, shares, date)
    other = 0
    for doc in parsed:
        if not doc or not doc.get("ok"):
            continue
        owner = doc.get("owner") or "?"
        for t in doc.get("transactions", []):
            if not _within(t.get("date"), cutoff_date):
                continue
            code, ad = t.get("code"), t.get("ad")
            val = t.get("value") or 0.0
            if code == "P" and ad == "A":
                buys.append((owner, val, t.get("shares"), t.get("date")))
            elif code == "S" and ad == "D":
                sells.append((owner, val, t.get("shares"), t.get("date")))
            else:
                other += 1

    buy_val = sum(v for _, v, _, _ in buys)
    sell_val = sum(v for _, v, _, _ in sells)
    buyers = {o for o, *_ in buys}
    sellers = {o for o, *_ in sells}
    net = buy_val - sell_val
    denom = buy_val + sell_val

    # 買賣互見→依淨值比例；只買→+1；只賣→弱負分(-0.3，賣出訊號噪音大)；皆無→None
    if buys and sells:
        score = round(max(-1.0, min(1.0, net / denom)), 2) if denom > 0 else 0.0
    elif buys:
        score = 1.0
    elif sells:
        score = -0.3
    else:
        score = None

    cluster_buy = len(buyers) >= 2
    if score is None:
        label = "近期無公開市場交易"
    elif cluster_buy and net > 0:
        label = "多位內部人買超（偏多，訊號較強）"
    elif net > 0:
        label = "內部人買超（偏多）"
    elif sells and not buys:
        label = "內部人賣超（訊號較弱，常為調節/節稅）"
    else:
        label = "買賣互見（中性）"

    return {
        "n_buys": len(buys), "n_sells": len(sells),
        "buy_value": buy_val, "sell_value": sell_val, "net_value": net,
        "n_buyers": len(buyers), "n_sellers": len(sellers),
        "cluster_buy": cluster_buy, "n_other": other,
        "score": score, "label": label,
        "recent_buys": sorted(buys, key=lambda x: x[3] or "", reverse=True)[:5],
        "recent_sells": sorted(sells, key=lambda x: x[3] or "", reverse=True)[:5],
    }


def _money(v):
    if v is None:
        return "無資料"
    a = abs(v)
    if a >= 1e9:
        return f"${v/1e9:.2f}B"
    if a >= 1e6:
        return f"${v/1e6:.2f}M"
    if a >= 1e3:
        return f"${v/1e3:.1f}K"
    return f"${v:.0f}"


def format_insider_text(summary: dict, window_label: str = "近90天") -> str:
    """給 AI 助理/Bot 的精簡文字（純函數）。"""
    sc = summary.get("score")
    lines = [f"內部人交易（{window_label}，來源 SEC Form 4）："]
    lines.append(
        f"  買進 {summary['n_buys']} 筆 / {summary['n_buyers']} 人（{_money(summary['buy_value'])}）　"
        f"賣出 {summary['n_sells']} 筆 / {summary['n_sellers']} 人（{_money(summary['sell_value'])}）")
    lines.append(f"  淨買賣：{_money(summary['net_value'])}"
                 + ("　🔶 多位內部人買進（cluster buy）" if summary.get("cluster_buy") else ""))
    lines.append(f"  情緒：{summary.get('label')}"
                 + (f"（分數 {sc:+.2f}）" if sc is not None else ""))
    return "\n".join(lines)


# ── 抓取層（EDGAR；需網路 + User-Agent）────────────────────────────────────────

def _ua():
    import os
    # SEC 要求帶可識別的 User-Agent；可用 SEC_USER_AGENT 覆寫為自己的聯絡方式
    return os.environ.get("SEC_USER_AGENT",
                          "RBS-Finance-Dashboard/1.0 (contact via GitHub repo)")


def ticker_to_cik(ticker: str, session=None) -> str | None:
    """用 SEC 對照表把美股代碼轉 10 碼 CIK。"""
    import requests
    s = session or requests
    try:
        r = s.get(f"{BASE}/files/company_tickers.json",
                  headers={"User-Agent": _ua()}, timeout=15)
        data = r.json() if r.ok else {}
    except Exception:
        return None
    tk = ticker.upper()
    for row in data.values():
        if str(row.get("ticker", "")).upper() == tk:
            return f"{int(row['cik_str']):010d}"
    return None


def fetch_insider(ticker: str, max_filings: int = 20, window_days: int = 90) -> dict | None:
    """
    抓某美股近期 Form 4 並彙總。非美股/查無 CIK 回 None。
    window_days：情緒統計只看近 N 天的交易。
    """
    import requests
    import pandas as pd

    sess = requests.Session()
    sess.headers.update({"User-Agent": _ua()})

    cik = ticker_to_cik(ticker, sess)
    if not cik:
        return None
    try:
        r = sess.get(f"{DATA}/submissions/CIK{cik}.json", timeout=15)
        sub = r.json() if r.ok else {}
    except Exception:
        return None

    recent = (sub.get("filings") or {}).get("recent") or {}
    forms = recent.get("form") or []
    accns = recent.get("accessionNumber") or []
    docs = recent.get("primaryDocument") or []
    parsed = []
    n = 0
    for i, form in enumerate(forms):
        if form != "4":
            continue
        try:
            acc = accns[i].replace("-", "")
            doc = docs[i]
            url = f"{BASE}/Archives/edgar/data/{int(cik)}/{acc}/{doc}"
            rr = sess.get(url, timeout=15)
            if rr.ok and rr.text.lstrip().startswith("<"):
                parsed.append(parse_form4(rr.text))
        except Exception:
            pass
        n += 1
        if n >= max_filings:
            break

    if not parsed:
        return None
    cutoff = (pd.Timestamp.now().normalize() - pd.Timedelta(days=window_days)).strftime("%Y-%m-%d")
    summary = summarize_insiders(parsed, cutoff)
    summary["ticker"] = ticker
    summary["window_days"] = window_days
    summary["n_filings"] = len(parsed)
    return summary


# ── CLI 自我測試（純解析）─────────────────────────────────────────────────────

if __name__ == "__main__":
    XML_BUY = """<?xml version="1.0"?>
    <ownershipDocument>
      <issuer><issuerTradingSymbol>AAPL</issuerTradingSymbol></issuer>
      <reportingOwner>
        <reportingOwnerId><rptOwnerName>DOE JOHN</rptOwnerName></reportingOwnerId>
        <reportingOwnerRelationship><isDirector>1</isDirector><isOfficer>1</isOfficer>
          <officerTitle>CEO</officerTitle></reportingOwnerRelationship>
      </reportingOwner>
      <nonDerivativeTable>
        <nonDerivativeTransaction>
          <transactionDate><value>2026-06-30</value></transactionDate>
          <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
          <transactionAmounts>
            <transactionShares><value>10000</value></transactionShares>
            <transactionPricePerShare><value>190.5</value></transactionPricePerShare>
            <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
          </transactionAmounts>
        </nonDerivativeTransaction>
      </nonDerivativeTable>
    </ownershipDocument>"""

    XML_BUY2 = XML_BUY.replace("DOE JOHN", "SMITH JANE").replace("10000", "5000")
    XML_SELL = XML_BUY.replace("DOE JOHN", "ROE RICHARD").replace(
        "<transactionCode>P</transactionCode>", "<transactionCode>S</transactionCode>").replace(
        "<value>A</value>", "<value>D</value>").replace("10000", "8000")

    d1, d2, d3 = parse_form4(XML_BUY), parse_form4(XML_BUY2), parse_form4(XML_SELL)
    print("parsed owner/title:", d1["owner"], d1["title"], d1["relationship"])
    print("txn:", d1["transactions"][0])
    assert d1["transactions"][0]["value"] == 10000 * 190.5
    assert d1["transactions"][0]["code_desc"] == "公開市場買進"

    summ = summarize_insiders([d1, d2, d3], cutoff_date="2026-01-01")
    print("\n", format_insider_text(summ))
    assert summ["n_buys"] == 2 and summ["n_sells"] == 1
    assert summ["cluster_buy"] is True                 # 2 位不同買家
    assert summ["net_value"] == (10000*190.5 + 5000*190.5) - 8000*190.5
    assert summ["score"] is not None and summ["score"] > 0

    # 只有賣：弱負分
    only_sell = summarize_insiders([d3], "2026-01-01")
    assert only_sell["score"] == -0.3 and "賣超" in only_sell["label"]
    # 視窗過濾：cutoff 在交易之後 → 無交易
    none_win = summarize_insiders([d1, d2, d3], "2026-12-01")
    assert none_win["score"] is None
    # 壞 XML 安全
    assert parse_form4("<not xml")["ok"] is False

    print("\n✅ sec_insider 純解析測試通過")
