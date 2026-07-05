"""
whales_13f.py – 超級投資人 13F 持倉追蹤（SEC EDGAR，免 key）

13F-HR：管理 ≥1 億美元的機構每季申報全部美股多頭持倉（45 天延遲）。
本模組追蹤知名投資人最近兩季的增減倉（新進/加碼/減碼/清倉），Dataroma 式。
與 sec_insider.py 共用 EDGAR 基礎設施慣例（User-Agent、CIK 兩種格式）。
解析/比較層為純函數（離線可測）。註：45 天延遲＝看到時人家可能已經賣了；
13F 只含美股多頭，不含空單/債券/海外。分析教育用途，非投資建議。
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET

DATA = "https://data.sec.gov"
BASE = "https://www.sec.gov"

# 知名投資人 CIK（可擴充；名稱為慣用中文稱呼）
WHALES = {
    "0001067983": "巴菲特 Berkshire Hathaway",
    "0001649339": "Michael Burry（Scion）",
    "0001336528": "Pershing Square（Ackman）",
    "0001079114": "Howard Marks（Oaktree）",
    "0001061768": "Baupost（Klarman）",
    "0001167483": "Third Point（Loeb）",
    "0001656456": "Duquesne（Druckenmiller）",
    "0000921669": "Icahn Enterprises",
}


# ── 純解析 ─────────────────────────────────────────────────────────────────────

def parse_13f_xml(xml_text: str) -> list[dict]:
    """
    解析 13F information table XML → [{issuer, cusip, value, shares, put_call}]。
    注意：2023 起 value 為整數美元（舊檔為千元）——本模組只比較同檔內相對權重與
    兩季股數變化，不跨年比較絕對金額。解析失敗回 []。
    """
    if not xml_text:
        return []
    # 去 namespace（13F 的 ns 前綴不固定）
    txt = re.sub(r"xmlns(:\w+)?=\"[^\"]+\"", "", xml_text)
    txt = re.sub(r"<(/?)\w+:", r"<\1", txt)
    try:
        root = ET.fromstring(txt)
    except Exception:
        return []
    rows = []
    for it in root.iter("infoTable"):
        def _t(tag):
            el = it.find(".//" + tag)
            return el.text.strip() if (el is not None and el.text) else None
        try:
            shares = float(_t("sshPrnamt") or 0)
            value = float(_t("value") or 0)
        except ValueError:
            continue
        issuer = _t("nameOfIssuer")
        if not issuer:
            continue
        rows.append({"issuer": issuer.upper(), "cusip": _t("cusip"),
                     "value": value, "shares": shares,
                     "put_call": (_t("putCall") or "").upper() or None})
    return rows


def aggregate_holdings(rows: list[dict]) -> dict:
    """同發行人多列（不同股類/多列申報）→ 以 issuer 彙總股數與市值；排除選擇權列。"""
    out: dict = {}
    for r in rows:
        if r.get("put_call"):          # PUT/CALL 列不算現股持倉
            continue
        k = r["issuer"]
        cur = out.setdefault(k, {"issuer": k, "value": 0.0, "shares": 0.0,
                                 "cusip": r.get("cusip")})
        cur["value"] += r["value"]
        cur["shares"] += r["shares"]
    return out


def compare_quarters(cur: dict, prev: dict, top_n: int = 8) -> dict:
    """
    兩季彙總持倉比較（純函數）。
    回 {top（本季市值前N）, new, added, reduced, exited}，各為清單。
    """
    tot = sum(v["value"] for v in cur.values()) or 1.0
    top = sorted(cur.values(), key=lambda v: -v["value"])[:top_n]
    for t in top:
        t["weight"] = t["value"] / tot

    new, added, reduced = [], [], []
    for k, v in cur.items():
        p = prev.get(k)
        if p is None or p["shares"] <= 0:
            if not prev:               # 沒有上一季資料 → 無從判斷新進
                continue
            new.append({**v, "weight": v["value"] / tot})
        elif v["shares"] > p["shares"] * 1.02:
            added.append({**v, "chg": v["shares"] / p["shares"] - 1})
        elif v["shares"] < p["shares"] * 0.98:
            reduced.append({**v, "chg": v["shares"] / p["shares"] - 1})
    exited = [{"issuer": k, "value": p["value"]}
              for k, p in prev.items() if k not in cur and p["shares"] > 0]

    return {"top": top,
            "new": sorted(new, key=lambda v: -v["value"])[:top_n],
            "added": sorted(added, key=lambda v: -v.get("chg", 0))[:top_n],
            "reduced": sorted(reduced, key=lambda v: v.get("chg", 0))[:top_n],
            "exited": sorted(exited, key=lambda v: -v["value"])[:top_n],
            "n_holdings": len(cur)}


def format_whale_text(name: str, cmp: dict, period: str = "") -> str:
    """給 Bot/助理的精簡文字（純函數）。"""
    lines = [f"🐋 *{name}*" + (f"（{period}）" if period else "")
             + f"　持倉 {cmp.get('n_holdings', '?')} 檔"]
    if cmp.get("top"):
        lines.append("前五大：" + "、".join(
            f"{t['issuer'][:14]} {t['weight']:.0%}" for t in cmp["top"][:5]))
    for key, label in (("new", "🆕 新進"), ("added", "➕ 加碼"),
                       ("reduced", "➖ 減碼"), ("exited", "🚪 清倉")):
        items = cmp.get(key) or []
        if items:
            if key in ("added", "reduced"):
                seg = "、".join(f"{i['issuer'][:14]}({i['chg']:+.0%})" for i in items[:4])
            else:
                seg = "、".join(i["issuer"][:14] for i in items[:4])
            lines.append(f"{label}：{seg}")
    return "\n".join(lines)


# ── 抓取層（EDGAR；需網路）────────────────────────────────────────────────────

def _ua() -> dict:
    ua = os.environ.get("SEC_USER_AGENT",
                        "RBS-Finance-Dashboard/1.0 (contact via GitHub repo)")
    return {"User-Agent": ua}


def _fetch_13f_rows(sess, cik: str, accession: str) -> list[dict]:
    """單一 13F-HR：從 filing index 找 information table XML 並解析。"""
    acc = accession.replace("-", "")
    try:
        r = sess.get(f"{BASE}/Archives/edgar/data/{int(cik)}/{acc}/index.json", timeout=20)
        idx = r.json() if r.ok else {}
    except Exception:
        return []
    names = [it.get("name", "") for it in (idx.get("directory", {}) or {}).get("item", [])]
    # information table 檔名慣例：含 infotable / form13f 的 .xml；排除主檔 primary_doc
    cands = [n for n in names if n.lower().endswith(".xml")
             and "primary_doc" not in n.lower()]
    cands.sort(key=lambda n: ("infotable" not in n.lower(), "13f" not in n.lower()))
    for name in cands[:3]:
        try:
            rr = sess.get(f"{BASE}/Archives/edgar/data/{int(cik)}/{acc}/{name}", timeout=25)
            if rr.ok:
                rows = parse_13f_xml(rr.text)
                if rows:
                    return rows
        except Exception:
            continue
    return []


def fetch_whale(cik: str, name: str | None = None) -> dict | None:
    """抓某機構最近兩季 13F 並比較。查無回 None。"""
    import requests
    sess = requests.Session()
    sess.headers.update(_ua())
    cik10 = f"{int(cik):010d}"
    try:
        r = sess.get(f"{DATA}/submissions/CIK{cik10}.json", timeout=20)
        sub = r.json() if r.ok else {}
    except Exception:
        return None
    recent = (sub.get("filings") or {}).get("recent") or {}
    forms = recent.get("form") or []
    accs = recent.get("accessionNumber") or []
    dates = recent.get("reportDate") or recent.get("filingDate") or []
    idxs = [i for i, f in enumerate(forms) if f == "13F-HR"][:2]
    if not idxs:
        return None
    cur_rows = _fetch_13f_rows(sess, cik, accs[idxs[0]])
    if not cur_rows:
        return None
    prev_rows = _fetch_13f_rows(sess, cik, accs[idxs[1]]) if len(idxs) > 1 else []
    cmp = compare_quarters(aggregate_holdings(cur_rows), aggregate_holdings(prev_rows))
    cmp["name"] = name or (sub.get("name") or cik)
    cmp["period"] = dates[idxs[0]] if idxs[0] < len(dates) else ""
    return cmp


# ── CLI 自我測試（純解析/比較）────────────────────────────────────────────────

if __name__ == "__main__":
    XML = """<?xml version="1.0"?>
    <informationTable xmlns="http://www.sec.gov/edgar/document/thirteenf/informationtable">
      <infoTable><nameOfIssuer>Apple Inc</nameOfIssuer><cusip>037833100</cusip>
        <value>90000000000</value><shrsOrPrnAmt><sshPrnamt>900000000</sshPrnamt>
        <sshPrnamtType>SH</sshPrnamtType></shrsOrPrnAmt></infoTable>
      <infoTable><nameOfIssuer>Apple Inc</nameOfIssuer><cusip>037833100</cusip>
        <value>10000000000</value><shrsOrPrnAmt><sshPrnamt>100000000</sshPrnamt>
        <sshPrnamtType>SH</sshPrnamtType></shrsOrPrnAmt></infoTable>
      <infoTable><nameOfIssuer>Chevron Corp</nameOfIssuer><cusip>166764100</cusip>
        <value>20000000000</value><shrsOrPrnAmt><sshPrnamt>120000000</sshPrnamt>
        <sshPrnamtType>SH</sshPrnamtType></shrsOrPrnAmt></infoTable>
      <infoTable><nameOfIssuer>SPY PUT</nameOfIssuer><cusip>78462F103</cusip>
        <value>5000000000</value><shrsOrPrnAmt><sshPrnamt>10000000</sshPrnamt>
        <sshPrnamtType>SH</sshPrnamtType></shrsOrPrnAmt><putCall>Put</putCall></infoTable>
    </informationTable>"""
    rows = parse_13f_xml(XML)
    assert len(rows) == 4
    agg = aggregate_holdings(rows)
    assert "SPY PUT" not in agg                       # PUT 列排除
    assert agg["APPLE INC"]["shares"] == 1e9          # 同發行人兩列合併
    print("aggregate:", {k: v["shares"] for k, v in agg.items()})

    prev = {"APPLE INC": {"issuer": "APPLE INC", "value": 8e10, "shares": 1.1e9},
            "COCA COLA": {"issuer": "COCA COLA", "value": 2e10, "shares": 4e8}}
    cmp = compare_quarters(agg, prev)
    assert cmp["top"][0]["issuer"] == "APPLE INC"
    assert any(r["issuer"] == "APPLE INC" for r in cmp["reduced"])   # 1.1e9→1e9 減碼
    assert any(n["issuer"] == "CHEVRON CORP" for n in cmp["new"])
    assert any(e["issuer"] == "COCA COLA" for e in cmp["exited"])
    print(format_whale_text("測試鯨魚", cmp, "2026-03-31"))

    # 無上一季 → new 判定關閉、不誤報
    cmp0 = compare_quarters(agg, {})
    assert cmp0["new"] == [] and cmp0["exited"] == []
    assert parse_13f_xml("<not xml") == []
    print("\n✅ whales_13f 純解析/比較測試通過")
