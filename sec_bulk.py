"""
sec_bulk.py – SEC 大量資料的「當時可得」資料層（Colab Cell 5 一鍵跑完；研究計畫 RESEARCH_PLAN.md §4）

為什麼：財報超預期（§4a）與內部人買進（§4b）其實有免費、帶申報日的歷史，但 SEC 擋 GitHub Actions 的 IP
（PITFALLS B10）→ 在使用者的 Colab 跑一次，產出四個檔上傳到 repo：
  1. data/edgar/annual_pit.json.gz        年報（edgar_pit.build_store；#91 擴充負債科目後重建，免得再跑 Cell 4）
  2. data/earnings/quarterly_pit.json.gz  季報數字：稀釋 EPS／淨利／加權稀釋股數，**每一個申報版本都留**
                                          （3 個月、9 個月累計、全年三種期間；10-Q／10-K 系列）
  3. data/earnings/announce_pit.json.gz   財報公布日：8-K／8-K/A 含 Item 2.02（submissions.zip，**含分頁檔**）
  4. data/insider/form4_pit.json.gz       內部人 Form 4／4/A 的公開市場買賣（代碼 P／S；DERA 季資料集 2006q1 起）
每個檔 > 24 MB 自動切成 xxx.part2.json.gz …（edgar_pit.save_parts，GitHub 網頁上傳上限 25 MB）；load_* 透明合併。

當時可得：每筆都帶申報日（filed／filingDate／FILING_DATE）；研究端一律在「申報日的下一個交易日開盤」才可用。
不用 companyfacts 的 frame 欄（frame 只標最後申報版本、還會挪到相鄰日曆季）。acceptanceDateTime 照原字串保存
（時區標示不可靠，研究計畫規定不用它決定進場時點）。
公司範圍：與年報檔同一個 CIK 宇宙（edgar_pit.match_ciks：2009 年後曾為 S&P 500 成分、現行代碼查得到 CIK）；
內部人另含 2006–2008 年間曾為成分者（資料集 2006 起）。已下市、現行代碼查不到的公司不在檔內（存活偏誤，研究端揭露）。

公平存取（SEC fair access）：User-Agent 帶使用者當場輸入的 email（只放在請求標頭、不寫進任何檔案，
也不送給 GitHub 等非 SEC 網站）、每秒最多 8 次、逾時／5xx／429 退避重試；403＝被擋 → 停下並說明怎麼辦；
404＝該季資料集還沒發布 → 記錄後跳過，最後列出缺哪幾季。
純邏輯（解析、篩選、分檔）離線可測（python3 sec_bulk.py）；抓取層（SecClient、colab_build_all）只在 Colab 跑。
教育用途，非投資建議。
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
import time
import zipfile
from datetime import date
from pathlib import Path

import edgar_pit as ep

HERE = Path(__file__).resolve().parent
EARN_DIR = HERE / "data" / "earnings"
INSIDER_DIR = HERE / "data" / "insider"
QUARTERLY_PATH = EARN_DIR / "quarterly_pit.json.gz"
ANNOUNCE_PATH = EARN_DIR / "announce_pit.json.gz"
FORM4_PATH = INSIDER_DIR / "form4_pit.json.gz"

SUBMISSIONS_URL = "https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip"
SUBMISSIONS_API = "https://data.sec.gov/submissions/"          # 分頁檔不在 zip 裡時的退路（逐檔抓）
# DERA 內部人資料集：2026q1 以前在 structureddata、2026q2 起搬到 datastandardsinnovation（兩邊互相 404）→ 兩個都試
INSIDER_URLS = ("https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{q}_form345.zip",
                "https://www.sec.gov/files/datastandardsinnovation/data/insider-transactions-data-sets/{q}_form345.zip")
INSIDER_NEW_PATH_FROM = "2026q2"
INSIDER_FIRST_Q = "2006q1"
INSIDER_SINCE = "2006-01-01"                                    # 內部人宇宙：2006 年後曾為 S&P 500 成分
ITEM_202_FROM = "2004-08-23"                                    # 8-K Item 2.02 生效日（之前的分頁檔不必逐檔抓）
ALL_STEPS = ("annual", "quarterly", "announce", "insider")

# 季報科目（companyfacts us-gaap）：EPS 依序退路（同一份申報、同一期間只留最優先的一個；科目名本身就是旗標）
Q_TAGS = ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted", "EarningsPerShareBasic",
          "NetIncomeLoss", "WeightedAverageNumberOfDilutedSharesOutstanding"]
Q_UNITS = {"EarningsPerShareDiluted": "USD/shares", "EarningsPerShareBasicAndDiluted": "USD/shares",
           "EarningsPerShareBasic": "USD/shares", "NetIncomeLoss": "USD",
           "WeightedAverageNumberOfDilutedSharesOutstanding": "shares"}
EPS_CHAIN = ("EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted", "EarningsPerShareBasic")
Q_FORMS = ("10-Q", "10-Q/A", "10-K", "10-K/A", "10-KT")
DUR_BANDS = ((80, 100, "Q"), (260, 290, "9M"), (350, 380, "FY"))     # 天數＝end − start（52/53 週年度、14 週季度都在內）
Q_FIELDS = ["tag", "end", "start", "val", "accn", "filed", "form", "fp", "fy"]
ANN_FIELDS = ["accn", "filed", "accepted", "items", "form"]
F4_FILING_FIELDS = ["filed", "doc", "symbol", "owners"]          # owners: [[owner_cik, rel, title], …]
F4_TRANS_FIELDS = ["accn", "code", "date", "shares", "price", "ad", "di", "after"]
F4_CODES = ("P", "S")
F4_DOCS = ("4", "4/A")

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
ACCN_RE = re.compile(r"^\d{10}-\d{2}-\d{6}$")
_MON = {m: i for i, m in enumerate(("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"), 1)}


# ── 1. 共用小工具（純函數）────────────────────────────────────────────────────

def _num(v):
    """字串／數字 → float（整數值存成 int 省空間）；空白、NaN、無法解析 → None。"""
    if v is None or (isinstance(v, str) and not v.strip()):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return int(x) if x.is_integer() and abs(x) < 1e15 else x


def _int(v):
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def parse_sec_date(s) -> str | None:
    """DERA 資料集的日期（說明文件：DD-MON-YYYY，例 14-NOV-2025；也容忍 ISO、YYYYMMDD）→ ISO；無法解析 → None。"""
    t = str(s or "").strip()
    if not t:
        return None
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})", t) or re.match(r"^(\d{4})(\d{2})(\d{2})(?:\.0+)?$", t)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    else:
        m = re.match(r"^(\d{1,2})-([A-Za-z]{3})-(\d{2}|\d{4})$", t)
        if not m or m.group(2).upper() not in _MON:
            return None
        y, mo, d = int(m.group(3)), _MON[m.group(2).upper()], int(m.group(1))
        if y < 100:
            y += 2000 if y < 50 else 1900
    try:
        return date(y, mo, d).isoformat()
    except ValueError:
        return None


def dur_kind(start: str, end: str) -> str | None:
    """期間長度 → "Q"（80–100 天）／"9M"（260–290）／"FY"（350–380）；其他（半年累計、多年合計、時點值）→ None。"""
    if not start or not end:
        return None
    try:
        d = ep._days(start, end)
    except ValueError:
        return None
    for lo, hi, k in DUR_BANDS:
        if lo <= d <= hi:
            return k
    return None


def quarter_of(d: date) -> str:
    return f"{d.year}q{(d.month - 1) // 3 + 1}"


def quarter_range(first: str, last: str) -> list[str]:
    """"2006q1" … "2026q3"（含頭尾）。"""
    y, q = int(first[:4]), int(first[-1])
    out = []
    while f"{y}q{q}" <= last:
        out.append(f"{y}q{q}")
        y, q = (y, q + 1) if q < 4 else (y + 1, 1)
    return out


def last_complete_quarter(today: date) -> str:
    """今天所在季度的前一季（資料集在季末之後才發布；進行中的季度不必試）。"""
    y, q = today.year, (today.month - 1) // 3 + 1
    return f"{y}q{q - 1}" if q > 1 else f"{y - 1}q4"


def insider_urls(q: str) -> list[str]:
    """該季資料集的候選網址：預期路徑先試、另一個路徑當退路（2026q2 起搬家）。"""
    old, new = (u.format(q=q) for u in INSIDER_URLS)
    return [new, old] if q >= INSIDER_NEW_PATH_FROM else [old, new]


def rel_flags(s) -> str:
    """RPTOWNER_RELATIONSHIP（例 "Director,Officer"、"TenPercentOwner"）→ 精簡旗標：D 董事、O 高管、T 10% 股東、X 其他。"""
    u = str(s or "").upper().replace(" ", "").replace("_", "")
    return ("D" if "DIRECTOR" in u else "") + ("O" if "OFFICER" in u else "") + \
           ("T" if ("TENPERCENT" in u or "10PERCENT" in u or "10%" in u) else "") + ("X" if "OTHER" in u else "")


def _manifest(kind: str, built_at: str, counts: dict, sources: list, **extra) -> dict:
    notes = {
        "annual": "edgar_pit 年報（#91 擴充負債科目）；rows＝[科目索引, end, start, val, filed, form]，科目表見 tags",
        "quarterly": "companyfacts 季報；每個申報版本都留（同一期間的重編＝不同 accn/filed）；不用 frame 欄；"
                     "EPS 依序 Diluted → BasicAndDiluted → Basic（同一份申報同一期間只留最優先者，科目名即旗標）；"
                     "期間 Q＝80–100 天、9M＝260–290、FY＝350–380；研究端申報日次一交易日才可用",
        "announce": "submissions（recent＋分頁檔）的 8-K／8-K/A 含 Item 2.02；accepted＝acceptanceDateTime 原字串（時區不可靠，不用來定時點）",
        "insider": "DERA Insider Transactions Data Sets；只留 Form 4／4/A、代碼 P／S、發行人在 S&P 500 CIK 宇宙；"
                   "4/A 與原申報分開保存（不自動取代）；rel：D 董事、O 高管、T 10% 股東、X 其他；"
                   "ad：A 取得／D 處分；di：D 直接／I 間接；研究端申報日次一交易日才可用",
    }
    return {"kind": kind, "built_at": built_at, "generator": "sec_bulk.py", "counts": counts, "sources": sources,
            "notes": notes.get(kind, ""), **extra}


# ── 2. 季報數字（companyfacts.zip；純函數）─────────────────────────────────────

def extract_quarterly(cf: dict) -> list[list]:
    """companyfacts JSON → [科目索引, end, start, val, accn, filed, form, fp, fy]（Q_TAGS 的索引）。
    只留 10-Q／10-K 系列、三種期間長度；每個申報版本都留（完全相同的列才去重）；不看 frame。
    EPS：同一份申報（accn）同一期間，已有 Diluted 就不收 BasicAndDiluted／Basic（依序退路）。"""
    gaap = (cf.get("facts") or {}).get("us-gaap") or {}
    pos = {t: i for i, t in enumerate(Q_TAGS)}
    rows, seen, eps_ctx = [], set(), set()
    for tag in Q_TAGS:
        add_ctx = []
        for f in ((gaap.get(tag) or {}).get("units") or {}).get(Q_UNITS[tag]) or []:
            form = str(f.get("form") or "")
            end, start = str(f.get("end") or "")[:10], str(f.get("start") or "")[:10]
            filed, accn = str(f.get("filed") or "")[:10], str(f.get("accn") or "")
            if form not in Q_FORMS or not (end and start and filed and accn) or dur_kind(start, end) is None:
                continue
            v = _num(f.get("val"))
            if v is None:
                continue
            if tag in EPS_CHAIN:
                ctx = (accn, end, start)
                if ctx in eps_ctx:
                    continue
                add_ctx.append(ctx)
            fy = _int(f.get("fy"))
            fp = str(f.get("fp") or "") or None
            r = (pos[tag], end, start, v, accn, filed, form, fp, fy)
            if r not in seen:
                seen.add(r)
                rows.append(list(r))
        eps_ctx.update(add_ctx)
    rows.sort(key=lambda r: (r[0], r[1], r[2], r[5], r[4]))
    return rows


def build_quarterly(zip_path, cik_map: dict, built_at: str, log=print) -> dict:
    """companyfacts.zip ＋ CIK 宇宙 → quarterly store（純本地 I/O，可離線測）。"""
    companies, n_rows, absent, bad = {}, 0, 0, 0
    by_kind = {"Q": 0, "9M": 0, "FY": 0}
    by_tag = {t: 0 for t in Q_TAGS}
    with zipfile.ZipFile(zip_path) as z:
        for i, (cik, tks) in enumerate(sorted(cik_map.items())):
            fn = f"CIK{int(cik):010d}.json"
            try:
                raw = z.read(fn)
            except KeyError:
                absent += 1
                continue
            try:
                cf = json.loads(raw)
                rows = extract_quarterly(cf)
            except Exception as e:                            # 單一公司壞檔不毀整批（計數揭露）
                log(f"  CIK {cik} 解析失敗（{type(e).__name__}），略過")
                bad += 1
                continue
            if rows:
                companies[str(cik)] = {"cik": int(cik), "name": cf.get("entityName"), "tickers": sorted(set(tks)),
                                       "rows": rows}
                n_rows += len(rows)
                for r in rows:
                    by_tag[Q_TAGS[r[0]]] += 1
                    by_kind[dur_kind(r[2], r[1])] += 1
            if (i + 1) % 100 == 0:
                log(f"  已處理 {i + 1}/{len(cik_map)} 家（{n_rows:,} 列）")
    counts = {"ciks": len(cik_map), "companies": len(companies), "rows": n_rows, "no_file": absent, "parse_error": bad,
              "by_kind": by_kind, "by_tag": by_tag}
    return {"version": 1, "manifest": _manifest("quarterly", built_at, counts, [ep.COMPANYFACTS_URL]),
            "tags": Q_TAGS, "fields": Q_FIELDS, "companies": companies}


# ── 3. 財報公布日（submissions.zip；純函數）───────────────────────────────────

_ANN_KEYS = ("accessionNumber", "filingDate", "acceptanceDateTime", "items", "form")


def _filing_rows(block) -> list[dict]:
    """submissions 的申報清單 → 每筆一個 dict。容忍三種形狀：主檔（filings.recent）、分頁檔（頂層就是欄式陣列）、列式 list。"""
    if isinstance(block, dict):
        if isinstance(block.get("filings"), dict):
            block = block["filings"].get("recent") or {}
        acc = block.get("accessionNumber")
        if not isinstance(acc, list):
            return []
        cols = {k: block.get(k) if isinstance(block.get(k), list) else [] for k in _ANN_KEYS}
        return [{k: (cols[k][i] if i < len(cols[k]) else None) for k in _ANN_KEYS} for i in range(len(acc))]
    if isinstance(block, list):
        return [r for r in block if isinstance(r, dict)]
    return []


def has_item_202(items) -> bool:
    return any(t.strip() == "2.02" for t in str(items or "").split(","))


def extract_announcements(blocks: list) -> list[list]:
    """主檔＋分頁檔 → [accn, filingDate, acceptanceDateTime 原字串, items, form]（只留 8-K／8-K/A 含 2.02；依申報日排序、accn 去重）。"""
    seen, rows = set(), []
    for b in blocks:
        for r in _filing_rows(b):
            form = str(r.get("form") or "").strip().upper()
            if form not in ("8-K", "8-K/A") or not has_item_202(r.get("items")):
                continue
            acc = str(r.get("accessionNumber") or "").strip()
            if not acc or acc in seen:
                continue
            seen.add(acc)
            rows.append([acc, str(r.get("filingDate") or "")[:10], str(r.get("acceptanceDateTime") or ""),
                         str(r.get("items") or ""), form])
    rows.sort(key=lambda x: (x[1], x[0]))
    return rows


def build_announcements(zip_path, cik_map: dict, built_at: str, fetch_page=None, log=print) -> dict:
    """submissions.zip（每家公司 CIK##########.json ＋ 分頁檔 CIK##########-submissions-NNN.json）→ announce store。
    分頁檔不在 zip 裡 → fetch_page(name)（data.sec.gov 逐檔抓；回 None＝抓不到，記入 missing_pages）；
    分頁檔全部早於 Item 2.02 生效日（filingTo < 2004-08-23）就不必逐檔抓。"""
    companies, n_rows, absent, bad = {}, 0, 0, 0
    pages = {"listed": 0, "in_zip": 0, "fetched": 0, "skipped_pre_2004": 0}
    missing_pages: list[str] = []
    with zipfile.ZipFile(zip_path) as z:
        def _read(name):
            try:
                return json.loads(z.read(name))
            except KeyError:
                return None
        for i, (cik, tks) in enumerate(sorted(cik_map.items())):
            try:
                main = _read(f"CIK{int(cik):010d}.json")
            except Exception as e:
                log(f"  CIK {cik} 解析失敗（{type(e).__name__}），略過")
                bad += 1
                continue
            if main is None:
                absent += 1
                continue
            blocks = [main]
            for fi in ((main.get("filings") or {}).get("files") or []):
                nm = str((fi or {}).get("name") or "")
                if not nm:
                    continue
                pages["listed"] += 1
                try:
                    pg = _read(nm)
                except Exception as e:                        # 分頁檔壞掉：記成缺頁、不毀整批
                    log(f"  {nm} 解析失敗（{type(e).__name__}），記為缺頁")
                    missing_pages.append(nm)
                    continue
                if pg is not None:
                    pages["in_zip"] += 1
                    blocks.append(pg)
                    continue
                if str((fi or {}).get("filingTo") or "9999-12-31") < ITEM_202_FROM:
                    pages["skipped_pre_2004"] += 1
                    continue
                try:
                    pg = fetch_page(nm) if fetch_page is not None else None
                except SecBlocked:
                    raise                                     # 被擋 → 停下（訊息說明怎麼辦）
                except Exception as e:
                    log(f"  {nm} 抓取失敗（{type(e).__name__}），記為缺頁")
                    pg = None
                if pg is None:
                    missing_pages.append(nm)
                else:
                    pages["fetched"] += 1
                    blocks.append(pg)
            rows = extract_announcements(blocks)
            if rows:
                companies[str(cik)] = {"cik": int(cik), "name": main.get("name"), "tickers": sorted(set(tks)),
                                       "rows": rows}
                n_rows += len(rows)
            if (i + 1) % 100 == 0:
                log(f"  已處理 {i + 1}/{len(cik_map)} 家（{n_rows:,} 筆 8-K 2.02）")
    counts = {"ciks": len(cik_map), "companies": len(companies), "rows": n_rows, "no_file": absent,
              "parse_error": bad, "pages": pages, "missing_pages": len(missing_pages)}
    return {"version": 1, "manifest": _manifest("announce", built_at, counts, [SUBMISSIONS_URL, SUBMISSIONS_API],
                                                missing_pages=missing_pages[:500]),
            "fields": ANN_FIELDS, "companies": companies}


# ── 4. 內部人（DERA Form 3/4/5 季資料集；純函數）────────────────────────────────

# 欄名依 SEC insider_transactions_readme；REPORTINGOWNER 曾有表頭命名不對的回報 → 表頭對不上時退回說明文件的欄位順序
SUB_COLS = {"accn": ("ACCESSION_NUMBER",), "filed": ("FILING_DATE",), "doc": ("DOCUMENT_TYPE",),
            "issuer": ("ISSUERCIK", "ISSUER_CIK"), "symbol": ("ISSUERTRADINGSYMBOL", "ISSUER_TRADING_SYMBOL")}
OWN_COLS = {"accn": ("ACCESSION_NUMBER",), "owner": ("RPTOWNERCIK", "RPTOWNER_CIK"),
            "rel": ("RPTOWNER_RELATIONSHIP",), "title": ("RPTOWNER_TITLE",)}
OWN_POS = {"accn": 0, "owner": 1, "rel": 3, "title": 4}
TRN_COLS = {"accn": ("ACCESSION_NUMBER",), "code": ("TRANS_CODE",), "date": ("TRANS_DATE",),
            "shares": ("TRANS_SHARES",), "price": ("TRANS_PRICEPERSHARE",), "ad": ("TRANS_ACQUIRED_DISP_CD",),
            "di": ("DIRECT_INDIRECT_OWNERSHIP",), "after": ("SHRS_OWND_FOLWNG_TRANS",)}
_TSV_ALIASES = {"SUBMISSION": ("SUBMISSION",), "REPORTINGOWNER": ("REPORTINGOWNER", "REPORTING_OWNER"),
                "NONDERIV_TRANS": ("NONDERIV_TRANS", "NON_DERIV_TRANS")}


def _member(z: zipfile.ZipFile, table: str) -> str | None:
    want = {f"{a}.TSV" for a in _TSV_ALIASES[table]}
    for n in z.namelist():
        if n.rsplit("/", 1)[-1].upper() in want:                # 容忍放在子資料夾、大小寫不同
            return n
    return None


def _iter_tsv(z: zipfile.ZipFile, table: str, cols: dict, required: tuple, positional: dict | None = None,
              stats: dict | None = None):
    """逐列讀 TSV（不載入整檔）→ {欄位鍵: 字串}。不把引號當跳脫（QUOTE_NONE）：備註欄的引號不會吃掉後面好幾列；
    備註含換行造成的殘列（第一欄不是 accession 格式、或欄數不夠）略過並計數。"""
    name = _member(z, table)
    if name is None:
        raise ValueError(f"資料集裡沒有 {table}.tsv")
    with z.open(name) as fh:
        rd = csv.reader(io.TextIOWrapper(fh, encoding="utf-8", errors="replace", newline=""),
                        delimiter="\t", quoting=csv.QUOTE_NONE)
        hdr = [h.strip().lstrip("﻿").upper() for h in (next(rd, None) or [])]
        idx = {}
        for k, names in cols.items():
            for nm in names:
                if nm in hdr:
                    idx[k] = hdr.index(nm)
                    break
        if any(k not in idx for k in required):
            if positional and all(k in positional for k in required):
                idx = dict(positional)
                if stats is not None:
                    stats.setdefault("positional_tables", []).append(table)
            else:
                raise ValueError(f"{table}.tsv 表頭缺欄位 {[k for k in required if k not in idx]}（表頭：{hdr[:12]}）")
        need = max(idx.values())
        for row in rd:
            if len(row) <= need or not ACCN_RE.match(row[idx["accn"]].strip()):
                if stats is not None and any(c.strip() for c in row):
                    stats["bad_lines"] = stats.get("bad_lines", 0) + 1
                continue
            yield {k: row[i].strip() for k, i in idx.items()}


def parse_form345_zip(path, cik_set: set, codes=F4_CODES, docs=F4_DOCS) -> tuple[dict, dict]:
    """一季的 form345 zip → ({發行人 CIK: {"filings": {accn: [filed, doc, symbol, owners]}, "trans": [[…F4_TRANS_FIELDS]]}}, 統計)。
    只留：DOCUMENT_TYPE 為 4／4/A、發行人在 cik_set、代碼 P／S（其餘計數後丟掉）；一次只處理一季（記憶體安全）。"""
    st = {"sub_rows": 0, "kept_filings": 0, "other_doc": 0, "non_sp": 0, "trans_rows": 0, "other_code": 0, "kept": 0,
          "bad_dates": 0}
    subs: dict[str, tuple] = {}
    with zipfile.ZipFile(path) as z:
        for r in _iter_tsv(z, "SUBMISSION", SUB_COLS, ("accn", "filed", "doc", "issuer"), stats=st):
            st["sub_rows"] += 1
            doc = r["doc"].upper()
            if doc not in docs:
                st["other_doc"] += 1
                continue
            ik = _int(r["issuer"])
            if ik is None or ik not in cik_set:
                st["non_sp"] += 1
                continue
            fd = parse_sec_date(r["filed"])
            st["bad_dates"] += fd is None
            subs[r["accn"]] = (ik, fd, doc, (r.get("symbol") or "").upper())
        owners: dict[str, list] = {}
        for r in _iter_tsv(z, "REPORTINGOWNER", OWN_COLS, ("accn", "owner", "rel"), positional=OWN_POS, stats=st):
            if r["accn"] in subs:
                owners.setdefault(r["accn"], []).append([_int(r["owner"]), rel_flags(r["rel"]), r.get("title") or ""])
        out: dict[int, dict] = {}
        for r in _iter_tsv(z, "NONDERIV_TRANS", TRN_COLS, ("accn", "code", "date", "shares", "ad"), stats=st):
            s = subs.get(r["accn"])
            if s is None:
                continue
            st["trans_rows"] += 1
            code = r["code"].upper()
            if code not in codes:
                st["other_code"] += 1
                continue
            ik, filed, doc, sym = s
            c = out.setdefault(ik, {"filings": {}, "trans": []})
            if r["accn"] not in c["filings"]:
                c["filings"][r["accn"]] = [filed, doc, sym, owners.get(r["accn"], [])]
                st["kept_filings"] += 1
            c["trans"].append([r["accn"], code, parse_sec_date(r["date"]), _num(r["shares"]), _num(r.get("price")),
                               (r["ad"].upper()[:1] or None), ((r.get("di") or "").upper()[:1] or None),
                               _num(r.get("after"))])
            st["kept"] += 1
    return out, st


def build_insider(fetch_quarter, cik_map: dict, quarters: list[str], built_at: str, log=print) -> dict:
    """逐季：fetch_quarter(q) → (zip 路徑, 來源) 或 (None, None)＝SEC 沒有這一季（404）→ 解析、合併。
    403／429 由 fetch_quarter 丟 SecBlocked（不在這裡吞掉）。最後一段連續的缺季＝尚未發布。"""
    cik_set = {int(c) for c in cik_map}
    companies: dict[str, dict] = {}
    missing, bad_q, src = [], {}, {}
    tot = {"kept": 0, "kept_filings": 0, "other_doc": 0, "non_sp": 0, "other_code": 0, "bad_lines": 0, "bad_dates": 0}
    positional = set()
    for q in quarters:
        try:
            path, url = fetch_quarter(q)
        except SecBlocked:
            raise                                             # 被擋 → 停下（訊息說明怎麼辦）
        except Exception as e:                                # 這一季一直下載不完整 → 記下來、繼續下一季
            bad_q[q] = f"{type(e).__name__}: {str(e)[:120]}"
            missing.append(q)
            log(f"  {q}：下載失敗（{bad_q[q]}），略過")
            continue
        if path is None:
            missing.append(q)
            log(f"  {q}：SEC 沒有這一季（兩個路徑都 404），略過")
            continue
        try:
            per, st = parse_form345_zip(path, cik_set)
        except (zipfile.BadZipFile, ValueError, KeyError) as e:
            bad_q[q] = f"{type(e).__name__}: {str(e)[:120]}"
            missing.append(q)
            log(f"  {q}：檔案解析失敗（{bad_q[q]}），略過")
            continue
        src[q] = "new" if "datastandardsinnovation" in str(url) else ("old" if "structureddata" in str(url) else "cache")
        for k in tot:
            tot[k] += st.get(k, 0)
        positional.update(st.get("positional_tables") or [])
        for ik, d in per.items():
            c = companies.setdefault(str(ik), {"cik": ik, "tickers": sorted(set(cik_map.get(ik) or [])),
                                               "filings": {}, "trans": []})
            c["filings"].update(d["filings"])
            c["trans"].extend(d["trans"])
        log(f"  {q}：{st['kept']:,} 筆 P／S（{len(per)} 家）")
    for c in companies.values():                              # 依申報日、accn 排序
        fl = c["filings"]
        c["trans"].sort(key=lambda t: ((fl.get(t[0]) or [""])[0] or "", t[0], t[2] or ""))
    got = [q for q in quarters if q not in missing]
    tail = [q for q in missing if not got or q > got[-1]]
    counts = dict(tot, quarters_tried=len(quarters), quarters_ok=len(got), companies=len(companies),
                  ciks=len(cik_set), trans=sum(len(c["trans"]) for c in companies.values()))
    man = _manifest("insider", built_at, counts, list(INSIDER_URLS), missing_quarters=missing,
                    unpublished_tail=tail, gap_quarters=[q for q in missing if q not in tail], parse_errors=bad_q,
                    quarter_source=src, positional_header_fallback=sorted(positional), codes=list(F4_CODES),
                    docs=list(F4_DOCS))
    return {"version": 1, "manifest": man, "filing_fields": F4_FILING_FIELDS, "trans_fields": F4_TRANS_FIELDS,
            "companies": companies}


# ── 5. 讀檔（Actions／本地；分檔透明合併）──────────────────────────────────────

def load_quarterly(path: Path | str = QUARTERLY_PATH) -> dict | None:
    return ep.load_parts(path)


def load_announcements(path: Path | str = ANNOUNCE_PATH) -> dict | None:
    return ep.load_parts(path)


def load_form4(path: Path | str = FORM4_PATH) -> dict | None:
    return ep.load_parts(path)


# ── 6. 抓取層（Colab；SEC 公平存取）──────────────────────────────────────────────

class SecBlocked(RuntimeError):
    """SEC 回 403（拒絕）或一直回 429（太頻繁）：停下來，訊息說明怎麼辦。"""


def blocked_msg(url: str, code: int) -> str:
    why = "回覆請求太頻繁（HTTP 429）" if code == 429 else "拒絕連線（HTTP 403）"
    return (f"SEC {why}：{url}\n"
            "可能原因：① email 格式不對（SEC 規定 User-Agent 要有聯絡方式）；② 短時間請求太多，"
            "SEC 暫時封鎖這台機器的 IP（約 10 分鐘）；③ SEC 封鎖這個雲端 IP。\n"
            "怎麼辦：等 10–15 分鐘後重跑 Cell 5（已下載的檔案會沿用、不重抓；已完成的步驟可在 STEPS 拿掉）；"
            "仍被擋 → 選單「執行階段 → 中斷並刪除執行階段」換一台機器，再依序重跑 Cell 1、2、5；"
            "還是不行 → 用自己電腦的瀏覽器下載上面的網址，上傳到 Drive 的 RBS_app 資料夾後重跑；或把這段訊息貼給 Claude。")


def _close(r) -> None:
    try:
        r.close()
    except Exception:
        pass


class SecClient:
    """requests 包一層：SEC 網址才帶含 email 的 User-Agent；每秒 ≤ max_rps 次；逾時／5xx／429 退避重試；403 → SecBlocked；
    200／404 回傳給呼叫端判斷。session／sleep／clock 可注入（離線自測）。"""

    def __init__(self, contact: str, session=None, max_rps: float = 8.0, max_retries: int = 4, timeout: int = 120,
                 sleep=time.sleep, clock=time.monotonic, log=print):
        c = str(contact or "").strip()
        if not EMAIL_RE.match(c):
            raise ValueError("請輸入有效的 email（SEC 規定 User-Agent 要有聯絡方式；空白或格式不對會被當成機器人擋掉）")
        if max_rps > 10:
            raise ValueError("SEC 公平存取上限是每秒 10 次")
        self.ua = f"RBS-research {c}"
        if session is None:
            import requests
            session = requests.Session()
        self.s, self.min_dt, self.max_retries, self.timeout = session, 1.0 / max_rps, max_retries, timeout
        self.sleep, self.clock, self.log = sleep, clock, log
        self._last = None
        self.n_requests = 0

    def headers_for(self, url: str) -> dict:
        host = re.sub(r"^https?://", "", url).split("/", 1)[0].lower()
        if host == "sec.gov" or host.endswith(".sec.gov"):
            return {"User-Agent": self.ua, "Accept-Encoding": "gzip, deflate"}
        return {}                                             # 非 SEC 網站（GitHub raw）不送 email

    def _wait(self) -> None:
        now = self.clock()
        if self._last is not None and now - self._last < self.min_dt:
            self.sleep(self.min_dt - (now - self._last))
            now = self.clock()
        self._last = now

    def _backoff(self, attempt: int, retry_after) -> None:
        try:
            d = float(retry_after) if retry_after is not None else None
        except (TypeError, ValueError):
            d = None
        self.sleep(min(120.0, d) if d is not None else min(60.0, 2.0 * 2 ** attempt))

    def get(self, url: str, stream: bool = False):
        for attempt in range(self.max_retries + 1):
            self._wait()
            self.n_requests += 1
            try:
                r = self.s.get(url, headers=self.headers_for(url), stream=stream, timeout=self.timeout)
            except Exception as e:                            # 連線中斷／逾時 → 退避重試
                if attempt >= self.max_retries:
                    raise RuntimeError(f"連不上 SEC（{type(e).__name__}）：{url}——請稍後重跑這格") from e
                self._backoff(attempt, None)
                continue
            code = int(getattr(r, "status_code", 0) or 0)
            if code in (200, 404):
                return r
            if code == 403:
                _close(r)
                raise SecBlocked(blocked_msg(url, 403))
            if code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                ra = (getattr(r, "headers", None) or {}).get("Retry-After")
                _close(r)
                self._backoff(attempt, ra)
                continue
            _close(r)
            if code == 429:
                raise SecBlocked(blocked_msg(url, 429))
            raise RuntimeError(f"SEC 回覆 HTTP {code}：{url}——請稍後重跑這格")
        raise RuntimeError(f"重試用完：{url}")

    def download(self, url: str, dest: Path | str, progress: bool = True) -> Path | None:
        """串流下載 zip 到 dest（先寫 .part、完整且是合法 zip 才改名）；404 → None。中斷重試 3 次。"""
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        for attempt in range(3):
            r = self.get(url, stream=True)
            if r.status_code == 404:
                _close(r)
                return None
            hd = getattr(r, "headers", None) or {}
            total = 0 if hd.get("Content-Encoding") else int(hd.get("Content-Length") or 0)
            done, mark = 0, 256 << 20
            try:
                with open(part, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8 << 20):
                        if chunk:
                            f.write(chunk)
                            done += len(chunk)
                            if progress and done >= mark:
                                self.log(f"   {done / 2**20:,.0f}" + (f" / {total / 2**20:,.0f}" if total else "") + " MB")
                                mark += 256 << 20
                ok = (not total or done == total) and zipfile.is_zipfile(part)
            except Exception as e:                            # 傳輸中斷
                ok = False
                self.log(f"   下載中斷（{type(e).__name__}），重試 {attempt + 1}/3")
            finally:
                _close(r)
            if ok:
                part.replace(dest)
                return dest
            if part.exists():
                part.unlink()
            self.sleep(min(60.0, 5.0 * 2 ** attempt))
        raise IOError(f"下載不完整：{url}")


def ensure_bulk_zip(client: SecClient, url: str, name: str, base: Path, work: Path, log=print) -> Path:
    """先找已下載的完整 zip（Drive 的 RBS_app、Colab 暫存區 /content；Cell 4 下載的 companyfacts.zip 可沿用），沒有才下載。"""
    cands = []
    for c in (Path(base) / name, Path(work) / name, Path("/content") / name):
        if c not in cands:
            cands.append(c)
    for c in cands:
        if c.exists() and not zipfile.is_zipfile(c):
            log(f"   刪除殘缺的 {c}")
            c.unlink()
    for c in cands:
        if c.exists():
            log(f"   沿用已下載的 {c}（{date.fromtimestamp(c.stat().st_mtime).isoformat()} 下載；要最新版就刪掉它重跑）")
            return c
    log(f"   下載 {url}（約 1–2 GB，存在 Colab 暫存區、不佔 Drive）…")
    try:
        p = client.download(url, Path(work) / name)
    except SecBlocked:
        raise
    except Exception as e:
        raise RuntimeError(f"{name} 下載失敗（{type(e).__name__}：{str(e)[:80]}）。請用自己電腦的瀏覽器下載 {url}，"
                           f"上傳到 {Path(base) / name} 後重跑 Cell 5") from e
    if p is None:
        raise RuntimeError(f"SEC 找不到 {url}（404）——請把這段訊息貼給 Claude")
    return p


def fetch_submissions_page(client: SecClient, name: str) -> dict | None:
    r = client.get(SUBMISSIONS_API + name)
    try:
        return None if r.status_code == 404 else r.json()
    finally:
        _close(r)


def fetch_insider_quarter(client: SecClient, q: str, cache_dir: Path) -> tuple:
    """該季 zip（快取在 Colab 暫存區，重跑不重抓）→ (路徑, 來源網址)；兩個路徑都 404 → (None, None)。"""
    dest = Path(cache_dir) / f"{q}_form345.zip"
    if dest.exists() and zipfile.is_zipfile(dest):
        return dest, "cache"
    for url in insider_urls(q):
        p = client.download(url, dest, progress=False)
        if p is not None:
            return p, url
    return None, None


def colab_build_all(base_dir, contact: str, *, steps=ALL_STEPS, client: SecClient | None = None,
                    today: date | None = None, work_dir=None, log=print) -> dict:
    """Colab Cell 5 進入點：①年報重建（#91）②季報 ③財報公布日 ④內部人 → 檔案寫到 base_dir，回傳各檔路徑。
    contact：SEC 規定的聯絡 email（只放在 SEC 請求的 User-Agent，不寫進任何檔案）。
    前面步驟完成的檔會先存好；之後某一步被 SEC 擋下（SecBlocked）時，重跑可只跑剩下的步驟（steps）。"""
    asked = {steps} if isinstance(steps, str) else set(steps)
    if not asked or asked - set(ALL_STEPS):
        raise ValueError(f"steps 只能是 {ALL_STEPS} 的子集合（收到 {sorted(asked)}）")
    steps = tuple(s for s in ALL_STEPS if s in asked)
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    client = client or SecClient(contact, log=log)               # 檢查 email 格式
    work = Path(work_dir) if work_dir else (Path("/content") if Path("/content").exists() else base / "_sec_cache")
    work.mkdir(parents=True, exist_ok=True)
    today = today or date.today()
    built_at = today.isoformat()
    t0 = time.time()

    log("【準備】讀取 S&P 500 成分歷史（fja05680）與 SEC 代碼對照表…")
    r = client.get(ep.SP500_PERIODS_URL)
    try:
        if r.status_code != 200:
            raise RuntimeError(f"成分歷史下載失敗（HTTP {r.status_code}），GitHub 可能暫時失敗，請重跑")
        periods = ep.parse_sp500_periods(r.text)
    finally:
        _close(r)
    sp, sp_ins = ep.sp500_tickers_since(periods), ep.sp500_tickers_since(periods, INSIDER_SINCE)
    if len(sp) < 400:
        raise RuntimeError(f"成分歷史只讀到 {len(sp)} 檔（應 > 800），GitHub raw 可能暫時失敗，請重跑")
    tj_path = base / "company_tickers.json"
    try:
        r = client.get(ep.TICKERS_URL)
        try:
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}")
            tickers_json = r.json()
        finally:
            _close(r)
    except Exception as e:
        if not tj_path.exists():
            raise RuntimeError(f"SEC 代碼表下載失敗（{type(e).__name__}）。請用瀏覽器下載 {ep.TICKERS_URL} 放到 {tj_path} 後重跑"
                               ) from e
        tickers_json = json.loads(tj_path.read_text())
        log("   （使用手動上傳的 company_tickers.json）")
    cik_map, miss = ep.match_ciks(tickers_json, sp)
    cik_ins, _miss_ins = ep.match_ciks(tickers_json, sp_ins)
    log(f"   2009 年後曾為成分 {len(sp)} 檔 → 對到 {len(cik_map)} 個 CIK（對不到 {len(miss)} 檔，多為已下市）；"
        f"內部人宇宙（2006 起）{len(cik_ins)} 個 CIK")

    out: dict[str, list[str]] = {}
    stats: dict[str, dict] = {}
    n_step = len(steps)

    def _hdr(k: int, text: str) -> None:
        log(f"\n【{k}/{n_step}】{text}（已用 {(time.time() - t0) / 60:.1f} 分鐘）")

    k = 0
    cf_zip = None
    if "annual" in steps or "quarterly" in steps:
        log("取得 companyfacts.zip（SEC 夜間打包）…")
        cf_zip = ensure_bulk_zip(client, ep.COMPANYFACTS_URL, "companyfacts.zip", base, work, log)
    if "annual" in steps:
        k += 1
        _hdr(k, "重建年報檔（擴充負債科目，修 #91）")
        st = ep.build_store(cf_zip, tickers_json, sp, built_at=built_at, log=log)
        st["manifest"] = _manifest("annual", built_at, st["stats"], [ep.COMPANYFACTS_URL, ep.TICKERS_URL,
                                                                     ep.SP500_PERIODS_URL])
        out["annual"] = [str(p) for p in ep.save_parts(st, base / "annual_pit.json.gz")]
        stats["annual"] = st["stats"]
        if st["stats"]["companies"] < 400:
            log(f"⚠️ 只抽到 {st['stats']['companies']} 家公司（預期 > 600），結果可能不完整——請把這段輸出貼給 Claude")
        del st
    if "quarterly" in steps:
        k += 1
        _hdr(k, "季報數字：稀釋 EPS／淨利／加權股數（每個申報版本都留）")
        st = build_quarterly(cf_zip, cik_map, built_at, log=log)
        out["quarterly"] = [str(p) for p in ep.save_parts(st, base / "quarterly_pit.json.gz")]
        stats["quarterly"] = st["manifest"]["counts"]
        del st
    if "announce" in steps:
        k += 1
        _hdr(k, "財報公布日：8-K Item 2.02（submissions.zip，含分頁檔）")
        sub_zip = ensure_bulk_zip(client, SUBMISSIONS_URL, "submissions.zip", base, work, log)
        st = build_announcements(sub_zip, cik_map, built_at, fetch_page=lambda nm: fetch_submissions_page(client, nm),
                                 log=log)
        out["announce"] = [str(p) for p in ep.save_parts(st, base / "announce_pit.json.gz")]
        stats["announce"] = st["manifest"]["counts"]
        if st["manifest"]["missing_pages"]:
            log(f"⚠️ {len(st['manifest']['missing_pages'])} 個分頁檔抓不到（已記在檔內 manifest）")
        del st
    missing_q: list[str] = []
    if "insider" in steps:
        k += 1
        last_q = last_complete_quarter(today)
        _hdr(k, f"內部人 Form 4（DERA 季資料集 {INSIDER_FIRST_Q}–{last_q}，逐季處理）")
        cache = work / "sec_form345"
        cache.mkdir(parents=True, exist_ok=True)
        st = build_insider(lambda q: fetch_insider_quarter(client, q, cache), cik_ins,
                           quarter_range(INSIDER_FIRST_Q, last_q), built_at, log=log)
        out["form4"] = [str(p) for p in ep.save_parts(st, base / "form4_pit.json.gz")]
        stats["form4"] = st["manifest"]["counts"]
        missing_q = st["manifest"]["missing_quarters"]
        tail, gaps = st["manifest"]["unpublished_tail"], st["manifest"]["gap_quarters"]
        log(f"   缺季：{', '.join(missing_q) if missing_q else '無'}"
            + (f"（其中 {', '.join(tail)} 是最近一季、SEC 多半季末後一兩個月才發布）" if tail else "")
            + (f"；⚠️ 中間缺 {', '.join(gaps)}——請把這段輸出貼給 Claude" if gaps else ""))
        del st

    folders = {"annual": "data/edgar", "quarterly": "data/earnings", "announce": "data/earnings", "form4": "data/insider"}
    upload: dict[str, list[str]] = {}
    log(f"\n完成（共 {(time.time() - t0) / 60:.1f} 分鐘、{client.n_requests} 次請求）：")
    for key, paths in out.items():
        upload.setdefault(folders[key], []).extend(paths)
        for p in paths:
            log(f"   {Path(p).name:<32} {Path(p).stat().st_size / 2**20:6.1f} MB → 上傳到 {folders[key]}/")
    if "insider" in steps:
        log(f"   內部人缺季（SEC 回 404）：{', '.join(missing_q) if missing_q else '無'}")
    return {"paths": out, "upload": upload, "stats": stats, "missing_quarters": missing_q}


# ── 7. 自我測試（合成 zip、假網路；離線）────────────────────────────────────────

if __name__ == "__main__":
    import gzip
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    EMAIL = "tester@example.com"

    # 1) 日期、期間、季度工具
    assert parse_sec_date("14-NOV-2025") == "2025-11-14" and parse_sec_date("03-Jan-2006") == "2006-01-03"
    assert parse_sec_date("2019-02-15") == "2019-02-15" and parse_sec_date("20190215") == "2019-02-15"
    assert parse_sec_date("31-FEB-2019") is None and parse_sec_date("") is None and parse_sec_date("x") is None
    assert dur_kind("2019-01-01", "2019-03-31") == "Q" and dur_kind("2019-01-01", "2019-06-30") is None
    assert dur_kind("2019-01-01", "2019-09-30") == "9M" and dur_kind("2019-12-29", "2021-01-02") == "FY"   # 53 週
    assert dur_kind("2020-09-27", "2021-01-02") == "Q" and dur_kind("", "2019-03-31") is None              # 14 週季
    assert quarter_range("2025q3", "2026q2") == ["2025q3", "2025q4", "2026q1", "2026q2"]
    assert last_complete_quarter(date(2026, 10, 25)) == "2026q3" and last_complete_quarter(date(2026, 2, 1)) == "2025q4"
    assert insider_urls("2019q1")[0].startswith("https://www.sec.gov/files/structureddata/")
    assert insider_urls("2026q2")[0].startswith("https://www.sec.gov/files/datastandardsinnovation/")
    assert all(u.endswith("/2026q2_form345.zip") for u in insider_urls("2026q2"))
    assert rel_flags("Director,Officer") == "DO" and rel_flags("TenPercentOwner") == "T" and rel_flags("Other") == "X"
    print("✅ 1 日期（DD-MON-YYYY／ISO）、期間分類（含 52/53 週）、季度範圍、新舊兩個下載路徑、身分旗標")

    # 2) 季報：每個版本都留、三種期間、EPS 退路、只收 S&P CIK
    def F(val, start, end, filed, accn, form="10-Q", fp="Q1", fy=2019, frame=None):
        d = {"val": val, "start": start, "end": end, "filed": filed, "accn": accn, "form": form, "fp": fp, "fy": fy}
        if frame:
            d["frame"] = frame
        return d
    eps = [F(1.00, "2019-01-01", "2019-03-31", "2019-05-01", "0000000001-19-000001", frame="CY2019Q1"),
           F(0.95, "2019-01-01", "2019-03-31", "2019-06-15", "0000000001-19-000002", form="10-Q/A"),  # 重編
           F(0.95, "2019-01-01", "2019-03-31", "2020-05-01", "0000000001-20-000001", fy=2020),       # 次年比較欄
           F(2.10, "2019-01-01", "2019-06-30", "2019-08-01", "0000000001-19-000003", fp="Q2"),       # 半年累計 → 不收
           F(3.30, "2019-01-01", "2019-09-30", "2019-11-01", "0000000001-19-000004", fp="Q3"),       # 9 個月
           F(1.40, "2020-09-27", "2021-01-02", "2021-02-20", "0000000001-21-000001", form="10-K", fp="FY", fy=2020),
           F(5.00, "2019-12-29", "2021-01-02", "2021-02-20", "0000000001-21-000001", form="10-K", fp="FY", fy=2020),
           F(9.99, "2019-01-01", "2019-03-31", "2019-04-20", "0000000001-19-000009", form="8-K"),      # 8-K → 不收
           {"val": 7, "end": "2019-03-31", "filed": "2019-05-01", "accn": "0000000001-19-000001", "form": "10-Q"}]
    basic = [F(1.02, "2019-01-01", "2019-03-31", "2019-05-01", "0000000001-19-000001"),               # 有稀釋 → 不收
             F(0.50, "2018-10-01", "2018-12-31", "2019-02-20", "0000000001-19-000000", form="10-K", fp="FY", fy=2018)]
    cf1 = {"cik": 1, "entityName": "TestCo", "facts": {"us-gaap": {
        "EarningsPerShareDiluted": {"units": {"USD/shares": eps}},
        "EarningsPerShareBasic": {"units": {"USD/shares": basic}},
        "NetIncomeLoss": {"units": {"USD": [F(1e8, "2019-01-01", "2019-03-31", "2019-05-01", "0000000001-19-000001")]}},
        "WeightedAverageNumberOfDilutedSharesOutstanding": {"units": {"shares": [
            F(1e8, "2019-01-01", "2019-03-31", "2019-05-01", "0000000001-19-000001")]}},
        "LongTermDebtNoncurrent": {"units": {"USD": [{"val": 5e8, "end": "2019-12-31", "filed": "2020-02-20",
                                                      "form": "10-K", "accn": "0000000001-20-000000"}]}},
        "SeniorLongTermNotes": {"units": {"USD": [{"val": 4e8, "end": "2019-12-31", "filed": "2020-02-20",
                                                   "form": "10-K", "accn": "0000000001-20-000000"}]}},
        "Revenues": {"units": {"USD": [F(9e8, "2019-01-01", "2019-12-31", "2020-02-20", "0000000001-20-000000",
                                         form="10-K", fp="FY")]}}}}}
    rows = extract_quarterly(cf1)
    dil = [r for r in rows if Q_TAGS[r[0]] == "EarningsPerShareDiluted"]
    q1 = [r for r in dil if (r[2], r[1]) == ("2019-01-01", "2019-03-31")]
    assert [(r[3], r[4], r[6]) for r in q1] == [(1, "0000000001-19-000001", "10-Q"), (0.95, "0000000001-19-000002", "10-Q/A"),
                                                (0.95, "0000000001-20-000001", "10-Q")], q1        # 三個版本都在
    assert sorted(dur_kind(r[2], r[1]) for r in dil) == ["9M", "FY", "Q", "Q", "Q", "Q"]          # 半年累計不收、8-K 不收
    bas = [r for r in rows if Q_TAGS[r[0]] == "EarningsPerShareBasic"]
    assert [(r[1], r[3]) for r in bas] == [("2018-12-31", 0.5)]                                    # 只在沒有稀釋時當退路
    assert len(rows[0]) == len(Q_FIELDS) and all(r[7] for r in rows) and not any("CY2019Q1" in map(str, r) for r in rows)
    cf2 = {"cik": 2, "entityName": "NotSP", "facts": {"us-gaap": {"EarningsPerShareDiluted": {"units": {"USD/shares": eps[:1]}}}}}
    cfz = tmp / "companyfacts.zip"
    with zipfile.ZipFile(cfz, "w") as z:
        z.writestr("CIK0000000001.json", json.dumps(cf1))
        z.writestr("CIK0000000002.json", json.dumps(cf2))
    qs = build_quarterly(cfz, {1: ["TEST"], 3: ["NOFILE"]}, "2026-10-25", log=lambda *_: None)
    assert list(qs["companies"]) == ["1"] and qs["manifest"]["counts"]["no_file"] == 1
    assert qs["companies"]["1"]["rows"] == rows and qs["manifest"]["counts"]["by_kind"]["9M"] == 1
    print(f"✅ 2 季報：同一期 3 個版本（含 10-Q/A 重編）都留、Q／9M／FY（53 週）、半年累計與 8-K 不收、EPS 退路、只收 S&P CIK（{len(rows)} 列）")

    # 3) 財報公布日：recent ＋ zip 內分頁檔 ＋ zip 外分頁檔（data.sec.gov 退路）；只留 8-K／8-K/A 含 2.02
    main1 = {"cik": "1", "name": "TestCo", "filings": {
        "recent": {"accessionNumber": ["0000000001-24-000010", "0000000001-24-000011", "0000000001-24-000012",
                                       "0000000001-24-000013"],
                   "filingDate": ["2024-10-30", "2024-11-02", "2024-11-01", "2024-07-30"],
                   "acceptanceDateTime": ["2024-10-30T16:05:11.000Z", "2024-11-02T09:00:00.000Z",
                                          "2024-11-01T16:30:00.000Z", "2024-07-30T16:01:00.000Z"],
                   "form": ["8-K", "8-K", "10-Q", "8-K/A"], "items": ["2.02,9.01", "5.02", "", "2.02"],
                   "primaryDocument": ["a.htm", "b.htm", "c.htm", "d.htm"]},
        "files": [{"name": "CIK0000000001-submissions-001.json", "filingCount": 2, "filingFrom": "2005-01-01",
                   "filingTo": "2012-12-31"},
                  {"name": "CIK0000000001-submissions-002.json", "filingCount": 1, "filingFrom": "2004-09-01",
                   "filingTo": "2004-12-31"},
                  {"name": "CIK0000000001-submissions-003.json", "filingCount": 1, "filingFrom": "1996-01-01",
                   "filingTo": "2001-12-31"}]}}
    page1 = {"accessionNumber": ["0000000001-08-000001", "0000000001-24-000010"], "filingDate": ["2008-01-25", "2024-10-30"],
             "acceptanceDateTime": ["2008-01-25T08:00:00.000Z", "2024-10-30T16:05:11.000Z"], "form": ["8-K", "8-K"],
             "items": ["2.02", "2.02,9.01"]}                                       # 與 recent 重複的一筆 → 去重
    page2 = {"accessionNumber": ["0000000001-04-000001"], "filingDate": ["2004-10-21"],
             "acceptanceDateTime": ["2004-10-21T17:00:00.000Z"], "form": ["8-K"], "items": ["2.02,9.01"]}
    main2 = {"cik": "2", "name": "NotSP", "filings": {"recent": dict(main1["filings"]["recent"]), "files": []}}
    subz = tmp / "submissions.zip"
    with zipfile.ZipFile(subz, "w") as z:
        z.writestr("CIK0000000001.json", json.dumps(main1))
        z.writestr("CIK0000000001-submissions-001.json", json.dumps(page1))
        z.writestr("CIK0000000002.json", json.dumps(main2))
    asked = []
    ann = build_announcements(subz, {1: ["TEST"]}, "2026-10-25",
                              fetch_page=lambda nm: asked.append(nm) or (page2 if nm.endswith("002.json") else None),
                              log=lambda *_: None)
    arows = ann["companies"]["1"]["rows"]
    assert [r[0] for r in arows] == ["0000000001-04-000001", "0000000001-08-000001", "0000000001-24-000013",
                                     "0000000001-24-000010"], arows
    assert arows[-1] == ["0000000001-24-000010", "2024-10-30", "2024-10-30T16:05:11.000Z", "2.02,9.01", "8-K"]
    assert arows[2][4] == "8-K/A" and asked == ["CIK0000000001-submissions-002.json"]             # 2001 以前的分頁不抓
    pg = ann["manifest"]["counts"]["pages"]
    assert pg == {"listed": 3, "in_zip": 1, "fetched": 1, "skipped_pre_2004": 1} and "2" not in ann["companies"]
    assert has_item_202("7.01, 2.02") and not has_item_202("2.020") and not has_item_202(None)
    subz2 = tmp / "submissions_bad.zip"                                              # 壞分頁檔 → 記成缺頁、不毀整批
    with zipfile.ZipFile(subz2, "w") as z:
        z.writestr("CIK0000000001.json", json.dumps(main1))
        z.writestr("CIK0000000001-submissions-001.json", "{not json")
    ann2 = build_announcements(subz2, {1: ["TEST"]}, "2026-10-25", fetch_page=lambda nm: 1 / 0, log=lambda *_: None)
    assert ann2["manifest"]["missing_pages"] == ["CIK0000000001-submissions-001.json", "CIK0000000001-submissions-002.json"]
    assert [r[0] for r in ann2["companies"]["1"]["rows"]] == ["0000000001-24-000013", "0000000001-24-000010"]
    print("✅ 3 公布日：recent＋分頁檔合併（zip 內、zip 外退路）、去重、只留 8-K／8-K/A 含 2.02、acceptanceDateTime 原字串保存")

    # 4) 內部人：真實表頭、4/A、非 S&P 發行人、Form 3/5、非 P/S 代碼、備註換行殘列
    SUB_H = ["ACCESSION_NUMBER", "FILING_DATE", "PERIOD_OF_REPORT", "DATE_OF_ORIG_SUB", "NO_SECURITIES_OWNED",
             "NOT_SUBJECT_SEC16", "FORM3_HOLDINGS_REPORTED", "FORM4_TRANS_REPORTED", "DOCUMENT_TYPE", "ISSUERCIK",
             "ISSUERNAME", "ISSUERTRADINGSYMBOL", "REMARKS"]
    OWN_H = ["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNERNAME", "RPTOWNER_RELATIONSHIP", "RPTOWNER_TITLE",
             "RPTOWNER_TXT", "RPTOWNER_STREET1", "RPTOWNER_STREET2", "RPTOWNER_CITY", "RPTOWNER_STATE",
             "RPTOWNER_ZIPCODE", "RPTOWNER_STATE_DESC", "FILE_NUMBER"]
    TRN_H = ["ACCESSION_NUMBER", "NONDERIV_TRANS_SK", "SECURITY_TITLE", "SECURITY_TITLE_FN", "TRANS_DATE", "TRANS_DATE_FN",
             "DEEMED_EXECUTION_DATE", "DEEMED_EXECUTION_DATE_FN", "TRANS_FORM_TYPE", "TRANS_CODE", "EQUITY_SWAP_INVOLVED",
             "EQUITY_SWAP_TRANS_CD_FN", "TRANS_TIMELINESS", "TRANS_TIMELINESS_FN", "TRANS_SHARES", "TRANS_SHARES_FN",
             "TRANS_PRICEPERSHARE", "TRANS_PRICEPERSHARE_FN", "TRANS_ACQUIRED_DISP_CD", "TRANS_ACQUIRED_DISP_CD_FN",
             "SHRS_OWND_FOLWNG_TRANS", "SHRS_OWND_FOLWNG_TRANS_FN", "VALU_OWND_FOLWNG_TRANS", "VALU_OWND_FOLWNG_TRANS_FN",
             "DIRECT_INDIRECT_OWNERSHIP", "DIRECT_INDIRECT_OWNERSHIP_FN", "NATURE_OF_OWNERSHIP", "NATURE_OF_OWNERSHIP_FN"]

    def tsv(header, rows_):
        def line(d):
            return "\t".join(str(d.get(h, "")) for h in header)
        return "\n".join(["\t".join(header)] + [r if isinstance(r, str) else line(r) for r in rows_]) + "\n"

    def form345_zip(path, accn_prefix, filed, sub_extra=()):
        a = [f"{accn_prefix}-{i:06d}" for i in range(1, 7)]
        subs_ = [{"ACCESSION_NUMBER": a[0], "FILING_DATE": filed, "DOCUMENT_TYPE": "4", "ISSUERCIK": "0000000001",
                  "ISSUERNAME": "TestCo", "ISSUERTRADINGSYMBOL": "test", "REMARKS": 'He said "buy'},
                 {"ACCESSION_NUMBER": a[1], "FILING_DATE": filed, "DOCUMENT_TYPE": "4/A", "ISSUERCIK": "1",
                  "ISSUERTRADINGSYMBOL": "TEST", "REMARKS": "line one"},
                 "continued remark line\twith tab",                                                      # 備註換行殘列
                 {"ACCESSION_NUMBER": a[2], "FILING_DATE": filed, "DOCUMENT_TYPE": "4", "ISSUERCIK": "999",
                  "ISSUERTRADINGSYMBOL": "ZZZ"},                                                         # 非 S&P
                 {"ACCESSION_NUMBER": a[3], "FILING_DATE": filed, "DOCUMENT_TYPE": "3", "ISSUERCIK": "1"},
                 {"ACCESSION_NUMBER": a[4], "FILING_DATE": filed, "DOCUMENT_TYPE": "5", "ISSUERCIK": "1"},
                 *sub_extra]
        own_ = [{"ACCESSION_NUMBER": a[0], "RPTOWNERCIK": "0000111111", "RPTOWNERNAME": "Doe Jane",
                 "RPTOWNER_RELATIONSHIP": "Director"},
                {"ACCESSION_NUMBER": a[0], "RPTOWNERCIK": "0000222222", "RPTOWNERNAME": "Roe Rick",
                 "RPTOWNER_RELATIONSHIP": "Director,Officer", "RPTOWNER_TITLE": "CEO"},
                {"ACCESSION_NUMBER": a[1], "RPTOWNERCIK": "333333", "RPTOWNER_RELATIONSHIP": "TenPercentOwner"},
                {"ACCESSION_NUMBER": a[2], "RPTOWNERCIK": "444444", "RPTOWNER_RELATIONSHIP": "Officer"}]
        trn_ = [{"ACCESSION_NUMBER": a[0], "NONDERIV_TRANS_SK": 1, "SECURITY_TITLE": "Common Stock", "TRANS_DATE": "13-FEB-2019",
                 "TRANS_CODE": "P", "TRANS_SHARES": "1000", "TRANS_PRICEPERSHARE": "50.25", "TRANS_ACQUIRED_DISP_CD": "A",
                 "SHRS_OWND_FOLWNG_TRANS": "5000", "DIRECT_INDIRECT_OWNERSHIP": "D"},
                {"ACCESSION_NUMBER": a[0], "NONDERIV_TRANS_SK": 2, "TRANS_DATE": "13-FEB-2019", "TRANS_CODE": "M",
                 "TRANS_SHARES": "10", "TRANS_ACQUIRED_DISP_CD": "A"},                                 # 非 P/S
                {"ACCESSION_NUMBER": a[1], "NONDERIV_TRANS_SK": 3, "TRANS_DATE": "14-FEB-2019", "TRANS_CODE": "S",
                 "TRANS_SHARES": "200.5", "TRANS_PRICEPERSHARE": "60", "TRANS_ACQUIRED_DISP_CD": "D",
                 "SHRS_OWND_FOLWNG_TRANS": "", "DIRECT_INDIRECT_OWNERSHIP": "I"},
                {"ACCESSION_NUMBER": a[2], "NONDERIV_TRANS_SK": 4, "TRANS_DATE": "14-FEB-2019", "TRANS_CODE": "P",
                 "TRANS_SHARES": "5", "TRANS_ACQUIRED_DISP_CD": "A"},                                  # 非 S&P
                {"ACCESSION_NUMBER": a[4], "NONDERIV_TRANS_SK": 5, "TRANS_DATE": "01-JAN-2019", "TRANS_CODE": "P",
                 "TRANS_SHARES": "7", "TRANS_ACQUIRED_DISP_CD": "A"}]                                  # Form 5
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("SUBMISSION.tsv", tsv(SUB_H, subs_))
            z.writestr("REPORTINGOWNER.tsv", tsv(OWN_H, own_))
            z.writestr("NONDERIV_TRANS.tsv", tsv(TRN_H, trn_))
            z.writestr("DERIV_TRANS.tsv", "ACCESSION_NUMBER\n")
        return a

    zq1 = tmp / "q1.zip"
    acc = form345_zip(zq1, "0000000001-19", "15-FEB-2019")
    per, pst = parse_form345_zip(zq1, {1})
    c1 = per[1]
    assert set(c1["filings"]) == {acc[0], acc[1]}, c1["filings"]
    assert c1["filings"][acc[0]] == ["2019-02-15", "4", "TEST", [[111111, "D", ""], [222222, "DO", "CEO"]]]
    assert c1["filings"][acc[1]][:2] == ["2019-02-15", "4/A"] and c1["filings"][acc[1]][3] == [[333333, "T", ""]]
    assert c1["trans"] == [[acc[0], "P", "2019-02-13", 1000, 50.25, "A", "D", 5000],
                           [acc[1], "S", "2019-02-14", 200.5, 60, "D", "I", None]], c1["trans"]
    assert pst["non_sp"] == 1 and pst["other_doc"] == 2 and pst["other_code"] == 1 and pst["bad_lines"] == 1, pst
    # REPORTINGOWNER 表頭命名不對 → 依說明文件的欄位順序
    zbad = tmp / "qbad.zip"
    form345_zip(zbad, "0000000001-19", "15-FEB-2019")
    with zipfile.ZipFile(zbad) as z:
        parts_ = {n: z.read(n).decode() for n in z.namelist()}
    parts_["REPORTINGOWNER.tsv"] = parts_["REPORTINGOWNER.tsv"].replace(
        "\t".join(OWN_H), "\t".join(f"column{i:02d}" for i in range(len(OWN_H))), 1)
    with zipfile.ZipFile(zbad, "w") as z:
        for n, s in parts_.items():
            z.writestr(f"sub/{n}", s)                                                                 # 也測子資料夾
    per_b, pst_b = parse_form345_zip(zbad, {1})
    assert per_b[1]["filings"] == c1["filings"] and pst_b["positional_tables"] == ["REPORTINGOWNER"]
    print("✅ 4 內部人解析：4 與 4/A 分開保存、多位申報人與身分、P／S 以外丟掉、非 S&P／Form 3／5 丟掉、殘列略過、表頭錯位退路")

    # 5) 抓取層：假網路（SEC 才帶 email、速率 ≤ 10 次/秒、404＝缺季、429 重試、403 停下）
    class FakeResp:
        def __init__(self, code, body=b"", headers=None):
            self.status_code, self.body, self.headers = code, body, headers or {}
            self.text = body.decode("utf-8", "replace") if isinstance(body, bytes) else str(body)

        def json(self):
            return json.loads(self.body)

        def iter_content(self, chunk_size=1):
            for i in range(0, len(self.body), chunk_size):
                yield self.body[i:i + chunk_size]

        def close(self):
            pass

    class FakeClock:
        def __init__(self):
            self.t = 1000.0

        def __call__(self):
            return self.t

        def sleep(self, d):
            self.t += max(0.0, d)

    class FakeSession:
        def __init__(self, routes, clock):
            self.routes, self.calls, self.clock = routes, [], clock

        def get(self, url, headers=None, stream=False, timeout=None):
            self.calls.append((url, dict(headers or {}), self.clock()))
            r = self.routes.get(url)
            if callable(r):
                r = r()
            if r is None:
                return FakeResp(404)
            if isinstance(r, FakeResp):
                return r
            if isinstance(r, int):
                return FakeResp(r)
            return FakeResp(200, r, {"Content-Length": str(len(r))})

    zq_new = tmp / "qnew.zip"
    form345_zip(zq_new, "0000000001-26", "2026-05-14")
    seq_429 = iter([FakeResp(429, headers={"Retry-After": "2"}), None])
    sp_csv = "ticker,start_date,end_date\n" + "".join(f"T{i},2000-01-01,\n" for i in range(450)) + \
             "TEST,2005-01-01,\nOLD6,2006-01-01,2008-06-30\n"
    tickers = {"0": {"cik_str": 1, "ticker": "TEST", "title": "TestCo"}, "1": {"cik_str": 5, "ticker": "OLD6", "title": "x"}}
    old_u = INSIDER_URLS[0].format(q="2019q1")
    routes = {ep.SP500_PERIODS_URL: sp_csv.encode(), ep.TICKERS_URL: json.dumps(tickers).encode(),
              old_u: lambda: next(seq_429) or FakeResp(200, zq1.read_bytes(), {"Content-Length": str(zq1.stat().st_size)}),
              INSIDER_URLS[1].format(q="2026q2"): zq_new.read_bytes(),
              INSIDER_URLS[0].format(q="2010q1"): b"<html>not a zip</html>"}                    # 一直下載不完整
    clk = FakeClock()
    sess = FakeSession(routes, clk)
    cli = SecClient(EMAIL, session=sess, sleep=clk.sleep, clock=clk, log=lambda *_: None)
    assert EMAIL in cli.ua and cli.headers_for(ep.TICKERS_URL)["User-Agent"].endswith(EMAIL)
    assert cli.headers_for(SUBMISSIONS_API + "x.json")["User-Agent"] == cli.ua and cli.headers_for(ep.SP500_PERIODS_URL) == {}
    for bad_email in ("", "abc", "a@b"):
        try:
            SecClient(bad_email, session=sess)
            raise AssertionError("email 格式不對應該擋下")
        except ValueError as e:
            assert "email" in str(e)
    base = tmp / "drive"
    base.mkdir()
    cfz.rename(base / "companyfacts.zip")                       # Cell 4 已下載的 companyfacts.zip → 沿用
    subz.rename(base / "submissions.zip")
    msgs: list[str] = []
    res = colab_build_all(base, EMAIL, client=cli, today=date(2026, 10, 25), work_dir=tmp / "work", log=msgs.append)
    # 內部人：2006q1–2026q3 共 83 季；只有 2019q1（舊路徑，先 429 再成功）與 2026q2（新路徑）有資料
    assert res["missing_quarters"] == [q for q in quarter_range("2006q1", "2026q3") if q not in ("2019q1", "2026q2")]
    f4 = load_form4(base / "form4_pit.json.gz")
    man = f4["manifest"]
    assert man["unpublished_tail"] == ["2026q3"] and "2019q2" in man["gap_quarters"] and len(man["gap_quarters"]) == 80
    assert man["quarter_source"] == {"2019q1": "old", "2026q2": "new"} and man["counts"]["quarters_ok"] == 2
    assert list(man["parse_errors"]) == ["2010q1"] and "IOError" in man["parse_errors"]["2010q1"] or \
        "OSError" in man["parse_errors"]["2010q1"], man["parse_errors"]
    assert [u for u, _h, _t in sess.calls].count(INSIDER_URLS[0].format(q="2010q1")) == 3            # 重試 3 次才放棄
    assert set(f4["companies"]) == {"1"} and len(f4["companies"]["1"]["trans"]) == 4
    assert f4["companies"]["1"]["trans"][0][2] == "2019-02-13" and f4["companies"]["1"]["trans"][-1][0].startswith("0000000001-26")
    calls_2026q2 = [u for u, _h, _t in sess.calls if "2026q2" in u]
    assert calls_2026q2 == [INSIDER_URLS[1].format(q="2026q2")]                                      # 新路徑先試、一次就中
    assert [u for u, _h, _t in sess.calls].count(old_u) == 2                                         # 429 → 退避後重試成功
    assert any("2019q1" in u for u, _h, _t in sess.calls) and any(m.startswith("   缺季：") for m in msgs)
    for u, h, _t in sess.calls:                                                                      # email 只送 SEC
        assert (h.get("User-Agent") == cli.ua) == ("sec.gov" in u), (u, h)
    ts = sorted(t for _u, _h, t in sess.calls)
    assert all(ts[i + 10] - ts[i] >= 1.0 for i in range(len(ts) - 10)), "每秒超過 10 次"
    # 其餘三個檔
    an = ep.load(base / "annual_pit.json.gz")
    assert an["tags"] == ep.TAGS and an["manifest"]["kind"] == "annual" and an["stats"]["companies"] == 1
    p_ = ep.pit_periods(an["companies"]["1"], "2020-03-01")[0]
    assert p_["total_debt"] == 5e8 and p_["revenue"] == 9e8                                          # 有長債科目 → 不走退路
    qq = load_quarterly(base / "quarterly_pit.json.gz")
    assert qq["companies"]["1"]["rows"] == rows and qq["manifest"]["counts"]["ciks"] == 1
    aa = load_announcements(base / "announce_pit.json.gz")
    assert len(aa["companies"]["1"]["rows"]) == 3 and aa["manifest"]["counts"]["missing_pages"] == 1   # 002 這次抓不到
    assert res["upload"] == {"data/edgar": [str(base / "annual_pit.json.gz")],
                             "data/earnings": [str(base / "quarterly_pit.json.gz"), str(base / "announce_pit.json.gz")],
                             "data/insider": [str(base / "form4_pit.json.gz")]}
    for p in base.glob("*.json.gz"):                                                                  # 公開 repo：檔內不可有 email
        assert EMAIL not in gzip.decompress(p.read_bytes()).decode("utf-8"), p
    print(f"✅ 5 抓取層：沿用已下載 zip、email 只送 SEC、{len(sess.calls)} 次請求皆 ≤ 10 次/秒、新舊路徑、"
          "429 退避重試、404 缺季清單（最近一季標尚未發布）、四個檔可讀回、檔內無 email")

    # 6) 403／一直 429 → SecBlocked（中文說明）；前面步驟的檔已存好、可只重跑剩下的步驟
    clk2 = FakeClock()
    cli403 = SecClient(EMAIL, session=FakeSession({INSIDER_URLS[0].format(q="2006q1"): 403}, clk2), sleep=clk2.sleep,
                       clock=clk2, log=lambda *_: None)
    try:
        fetch_insider_quarter(cli403, "2006q1", tmp / "c403")
        raise AssertionError("403 應該停下")
    except SecBlocked as e:
        assert "403" in str(e) and "Cell 5" in str(e) and "10–15 分鐘" in str(e)
    clk3 = FakeClock()
    cli429 = SecClient(EMAIL, session=FakeSession({"https://www.sec.gov/x.zip": 429}, clk3), sleep=clk3.sleep,
                       clock=clk3, max_retries=2, log=lambda *_: None)
    try:
        cli429.download("https://www.sec.gov/x.zip", tmp / "x.zip")
        raise AssertionError("一直 429 應該停下")
    except SecBlocked as e:
        assert "429" in str(e) and len(cli429.s.calls) == 3
    clk4 = FakeClock()
    sess4 = FakeSession({ep.SP500_PERIODS_URL: sp_csv.encode(), ep.TICKERS_URL: json.dumps(tickers).encode(),
                         INSIDER_URLS[0].format(q="2006q1"): 403}, clk4)
    cli4 = SecClient(EMAIL, session=sess4, sleep=clk4.sleep, clock=clk4, log=lambda *_: None)
    before = (base / "quarterly_pit.json.gz").read_bytes()
    try:
        colab_build_all(base, EMAIL, steps=("insider",), client=cli4, today=date(2026, 10, 25), work_dir=tmp / "w4",
                        log=lambda *_: None)
        raise AssertionError("403 應該停下")
    except SecBlocked:
        pass
    assert (base / "quarterly_pit.json.gz").read_bytes() == before                                   # 沒跑的步驟不動
    for bad_steps in (("bogus",), ("insider", "bogus"), ()):
        try:
            colab_build_all(base, EMAIL, steps=bad_steps, client=cli4)
            raise AssertionError("未知步驟應報錯")
        except ValueError:
            pass
    print("✅ 6 403／持續 429 停下並附中文處理步驟；只重跑指定步驟、其他檔不動")

    # 7) 自動分檔：內部人檔變大時切成多個 ≤ 上限的分檔、load_form4 透明合併
    import random
    rnd = random.Random(3)
    big = {"version": 1, "manifest": dict(man), "filing_fields": F4_FILING_FIELDS, "trans_fields": F4_TRANS_FIELDS,
           "companies": {}}
    for ik in range(1, 81):
        fl, tr = {}, []
        for j in range(30):
            a = f"{ik:010d}-19-{j:06d}"
            fl[a] = ["2019-02-15", "4", f"T{ik}", [[rnd.randrange(10**6), "D", ""]]]
            tr.append([a, "P", "2019-02-13", rnd.randrange(1, 10**5), round(rnd.random() * 100, 2), "A", "D", rnd.randrange(10**6)])
        big["companies"][str(ik)] = {"cik": ik, "tickers": [f"T{ik}"], "filings": fl, "trans": tr}
    pp = ep.save_parts(big, tmp / "form4_pit.json.gz", max_bytes=20_000)
    assert len(pp) >= 3 and [p.name for p in pp[:3]] == ["form4_pit.json.gz", "form4_pit.part2.json.gz",
                                                         "form4_pit.part3.json.gz"]
    assert all(p.stat().st_size <= 20_000 for p in pp)
    back = load_form4(tmp / "form4_pit.json.gz")
    assert back["companies"] == big["companies"] and back["manifest"]["parts_loaded"] == len(pp)
    assert back["manifest"]["missing_quarters"] == man["missing_quarters"]
    print(f"✅ 7 自動分檔：{len(pp)} 個分檔皆 ≤ 上限、load_form4 合併還原、manifest 每檔都有")
    print("\nsec_bulk selftest OK ✅")
