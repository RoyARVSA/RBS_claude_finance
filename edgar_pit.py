"""
edgar_pit.py – SEC EDGAR XBRL 年度財報的「當時可得」（point-in-time）資料層（VRT 方法 15 年回測的地基）

為什麼：company_model（VRT 方法泛化版）的輸入是「今天的財報」，無法回答「2015 年用這套方法會怎麼判斷」。
EDGAR 的 companyfacts 每筆數字都帶申報日（filed），同一期數字會在後續 10-K 的比較欄重複出現（含重編）——
依申報日篩選就能還原「當天看得到的版本」。

流程（SEC 封鎖 GitHub Actions 的 IP——PITFALLS B10，所以抽取在使用者的 Colab 跑一次）：
  1. Colab：`colab_build()` 下載 companyfacts.zip（SEC 夜間打包）＋ company_tickers.json，只留
     「2009 年後曾是 S&P 500 成分、且現在代碼查得到 CIK」的公司 × DCF 需要的科目 × 年報（10-K 系列），
     版本去重後寫成精簡檔（約數 MB）→ 使用者上傳到 repo `data/edgar/annual_pit.json.gz`
  2. Actions／本地：`load()` 讀檔、`pit_periods(company, as_of)` 轉成 fin_data 的年期別格式（新→舊），
     直接餵 company_model.run_model——模型本身不改

當時可得規則：版本的 filed ≤ as_of 才可見（申報當天可見 → 呼叫端必須在次一交易日才交易），
available_at = filed + 1 天（與 fin_data.normalize_finnhub 同義）；
同一期多版本取「as_of 前最後申報的版本」（含重編，但不含 as_of 之後的重編）。
同一欄位多個候選科目（例：營收 2018 年 ASC 606 換科目、資本支出換科目）**逐期合併**：每一期取優先序最高、
且當時已申報的科目——不是「全期取第一個有資料的科目」。
已知限制：已下市公司（現在代碼查不到 CIK）不在檔內——與價格資料的存活偏誤同向，報告會揭露覆蓋率；
公司自訂科目與分部資料不在 companyfacts（SEC 只彙整標準科目）。教育用途，非投資建議。
"""

from __future__ import annotations

import gzip
import json
import math
from datetime import date, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_PATH = HERE / "data" / "edgar" / "annual_pit.json.gz"
SP500_PERIODS_URL = "https://raw.githubusercontent.com/fja05680/sp500/master/sp500_ticker_start_end.csv"
COMPANYFACTS_URL = "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
ANNUAL_FORMS = ("10-K", "10-K/A", "10-KT", "10-KT/A")
SINCE = "2009-01-01"                       # S&P 500 大型公司 2009-06 起強制 XBRL

# fin_data 欄位 → us-gaap 科目候選（優先序由前到後；逐期合併）。與 fin_data.FH_CONCEPTS 同源，補上 EDGAR 常見替代科目
FIELD_TAGS = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet",
                "SalesRevenueGoodsNet", "SalesRevenueServicesNet"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "pretax_income": ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
                      "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"],
    "tax_provision": ["IncomeTaxExpenseBenefit"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "interest_expense": ["InterestExpense", "InterestExpenseNonoperating", "InterestExpenseDebt", "InterestAndDebtExpense"],
    "diluted_shares": ["WeightedAverageNumberOfDilutedSharesOutstanding"],
    "sga": ["SellingGeneralAndAdministrativeExpense"],
    "rnd": ["ResearchAndDevelopmentExpense"],
    "da": ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization",
           "DepreciationAmortizationAndAccretionNet", "Depreciation"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets",
              "PaymentsForCapitalImprovements"],
    "sbc": ["ShareBasedCompensation", "AllocatedShareBasedCompensationExpense"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "buyback": ["PaymentsForRepurchaseOfCommonStock"],
    "dividends_paid": ["PaymentsOfDividendsCommonStock", "PaymentsOfDividends"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue", "Cash"],
    "cash_and_sti": ["CashCashEquivalentsAndShortTermInvestments"],
    "short_term_investments": ["ShortTermInvestments", "MarketableSecuritiesCurrent", "AvailableForSaleSecuritiesCurrent"],
    "receivables": ["AccountsReceivableNetCurrent"],
    "inventory": ["InventoryNet"],
    "payables": ["AccountsPayableCurrent"],
    "deferred_revenue": ["ContractWithCustomerLiabilityCurrent", "DeferredRevenueCurrent"],
    "net_ppe": ["PropertyPlantAndEquipmentNet"],
    "goodwill_intang": ["Goodwill"],
    "total_assets": ["Assets"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "total_equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "retained_earnings": ["RetainedEarningsAccumulatedDeficit"],
    "minority_interest": ["MinorityInterest"],
    "shares_out": ["CommonStockSharesOutstanding"],
    # 負債組成（合成 total_debt；科目語意：LongTermDebtAndCapitalLeaseObligations＝非流動部分、
    # DebtCurrent＝已含一年內到期長債、LongTermDebt＝含一年內到期的長債總額——驗證 High）
    "debt_nc": ["LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations"],
    "debt_cur_total": ["DebtCurrent"],
    "debt_lt_cur": ["LongTermDebtCurrent"],
    "debt_st": ["ShortTermBorrowings", "CommercialPaper"],
    "debt_lt_total": ["LongTermDebt"],
}
# 代碼改名（舊→新，同一個 CIK）：舊代碼期間的成分要接到新代碼的公司（現行代碼表查不到舊代碼）
RENAMES = {"FB": "META", "UTX": "RTX", "ANTM": "ELV", "BBT": "TFC", "CTL": "LUMN", "HRS": "LHX", "ABC": "COR",
           "DISCA": "WBD", "DISCK": "WBD", "BK": "BNY", "FI": "FISV", "COG": "CTRA", "WLTW": "WTW", "PKI": "RVTY",
           "FLT": "CPAY", "GPS": "GAP", "TMK": "GL", "ADS": "BFH", "HCP": "DOC", "PEAK": "DOC", "SYMC": "GEN",
           "NLOK": "GEN", "LB": "BBWI", "WYND": "TNL", "JEC": "J"}
# （2026-10 以 fja05680 期間表核對：舊代碼結束日＝新代碼開始日；併購〔不同 CIK〕不可列入，例如 ANDV→MPC 已排除）
# 代碼重複使用（不同公司）：此日期之前的成分期間屬於別家公司 → 回測遮掉（現行代碼表會對到新公司的財報）
TICKER_REUSE = {"JCI": "2016-09-02", "CB": "2016-01-14", "IR": "2020-03-02", "DOW": "2019-04-02",
                "DD": "2019-06-03", "GM": "2010-11-18", "FOXA": "2019-03-20", "FOX": "2019-03-20",
                "NWSA": "2013-06-28", "NWS": "2013-06-28", "ADT": "2018-01-19", "SNDK": "2025-02-24"}
# （保守遮罩：寧可少算一段，也不要把別家公司的財報接到這個代碼）
DEI_TAGS = {"cover_shares": ["EntityCommonStockSharesOutstanding"]}     # 封面股數（申報日附近的流通股）
FLOW_FIELDS = {"revenue", "gross_profit", "operating_income", "pretax_income", "tax_provision", "net_income",
               "interest_expense", "diluted_shares", "sga", "rnd", "da", "capex", "sbc", "cfo", "buyback",
               "dividends_paid"}
SHARE_FIELDS = {"diluted_shares", "shares_out", "cover_shares"}


def _tag_index() -> list[str]:
    tags = []
    for d in (FIELD_TAGS, DEI_TAGS):
        for lst in d.values():
            for t in lst:
                if t not in tags:
                    tags.append(t)
    return tags


TAGS = _tag_index()


# ── 1. 抽取（Colab 端；純函數，離線可測）───────────────────────────────────────

def _days(a: str, b: str) -> int:
    return (date.fromisoformat(b[:10]) - date.fromisoformat(a[:10])).days


def extract_company(cf: dict, since: str = SINCE) -> list[list]:
    """companyfacts JSON → 精簡列 [tag_idx, end, start, val, filed, form]（只留年報、年期間或時點值、版本去重）。
    flow 科目只留 ~1 年期（300–400 天）；存量科目（無 start）只留期末時點值。"""
    rows = []
    facts = cf.get("facts") or {}
    tag_pos = {t: i for i, t in enumerate(TAGS)}
    for tax in ("us-gaap", "dei"):
        for tag, body in (facts.get(tax) or {}).items():
            if tag not in tag_pos:
                continue
            units = body.get("units") or {}
            want = "shares" if tag in ("WeightedAverageNumberOfDilutedSharesOutstanding", "CommonStockSharesOutstanding",
                                       "EntityCommonStockSharesOutstanding") else "USD"
            seen: dict[tuple, list] = {}
            for f in units.get(want) or []:
                form = str(f.get("form") or "")
                filed, end = str(f.get("filed") or "")[:10], str(f.get("end") or "")[:10]
                start = str(f.get("start") or "")[:10] or None
                v = f.get("val")
                if form not in ANNUAL_FORMS or not filed or not end or filed < since or v is None:
                    continue
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(v):
                    continue
                if start and not (300 <= _days(start, end) <= 400):
                    continue                                  # 只留年期間（排除季度與多年合計）
                seen.setdefault((end, start), []).append([filed, v, form])
            for (end, start), vers in seen.items():
                vers.sort()
                last = None
                for filed, v, form in vers:                   # 版本去重：值沒變就不再記
                    if last is None or abs(v - last) > 1e-9 * max(1.0, abs(last)):
                        rows.append([tag_pos[tag], end, start, v, filed, form])
                        last = v
    rows.sort(key=lambda r: (r[0], r[1], r[4]))
    return rows


# ── 2. 當時可得的年期別（Actions／本地；純函數）────────────────────────────────

def _visible(rows: list[list], as_of: str) -> dict:
    """{(tag_idx, end, start): (val, filed)}：as_of（含）以前最後申報的版本。"""
    out = {}
    for ti, end, start, v, filed, _form in rows:
        if filed <= as_of:
            k = (ti, end, start)
            if k not in out or filed >= out[k][1]:
                out[k] = (v, filed)
    return out


def _near(a: str, b: str, tol: int = 10) -> bool:
    return abs(_days(a, b)) <= tol


def pit_periods(company: dict, as_of: str, max_periods: int = 5) -> list[dict]:
    """某公司在 as_of 當天看得到的年度期別（新→舊），fin_data 格式（欄位名同 FIELD_TAGS 鍵；capex 為負值）。
    期末日由「已可見的年期間 flow 科目」決定；存量科目取期末 ±10 天的時點值。"""
    rows = company.get("rows") or []
    vis = _visible(rows, as_of)
    tag_pos = {t: i for i, t in enumerate(TAGS)}
    ends = sorted({k[1] for k in vis if k[2] is not None}, reverse=True)
    # 同一財年可能有 52/53 週差異：期末相距 < 20 天視為同一期，保留較新者
    uniq: list[str] = []
    for e in ends:
        # 與上一個保留的期末相距 < 300 天：52/53 週差異或 10-KT 重編的重疊年度 → 不另成一期（驗證 Low）
        if not uniq or _days(uniq[-1], e) <= -300:
            uniq.append(e)
    periods = []
    for pe in uniq:
        if len(periods) >= max_periods:                      # 有效期別才佔名額
            break
        row = {"period_end": pe, "freq": "A", "source": "edgar"}
        filed_max = ""
        for field, tags in list(FIELD_TAGS.items()) + list(DEI_TAGS.items()):
            # 逐期合併：先取「涵蓋這一期的最新一份申報」，再在那份申報裡依科目優先序挑——
            # 避免舊的高優先科目蓋過新申報以低優先科目重編的數字（驗證 Low：ASC 606／重編混基準）
            cands = []
            for rank, t in enumerate(tags):
                ti = tag_pos[t]
                for k, (v, filed) in vis.items():
                    if k[0] == ti and ((field in FLOW_FIELDS and k[2] is not None and _near(k[1], pe)) or
                                       (field not in FLOW_FIELDS and k[2] is None and (_near(k[1], pe) or
                                        (field == "cover_shares" and 0 <= _days(pe, k[1]) <= 120)))):
                        cands.append((filed, -rank, v))
            val = None
            if cands:
                fl = max(c[0] for c in cands)
                val = max((c for c in cands if c[0] == fl), key=lambda c: c[1])[2]
                filed_max = max(filed_max, fl)
            row[field] = val
        cur_ = row.get("debt_cur_total")
        if cur_ is None and (row.get("debt_lt_cur") is not None or row.get("debt_st") is not None):
            cur_ = (row.get("debt_lt_cur") or 0.0) + (row.get("debt_st") or 0.0)
        if row.get("debt_nc") is not None:
            row["total_debt"] = row["debt_nc"] + (cur_ or 0.0)
        elif row.get("debt_lt_total") is not None:                   # LongTermDebt 已含一年內到期 → 只加短借
            row["total_debt"] = row["debt_lt_total"] + (row.get("debt_st") or 0.0)
        else:
            row["total_debt"] = cur_
        if row.get("cash_and_sti") is None and row.get("cash") is not None:
            row["cash_and_sti"] = row["cash"] + (row.get("short_term_investments") or 0.0)
        if row.get("capex") is not None and row["capex"] > 0:
            row["capex"] = -row["capex"]                      # 統一負值（現金流出），同 fin_data
        if row.get("shares_out") is None and row.get("cover_shares") is not None:
            row["shares_out"] = row["cover_shares"]
        row["ebit"] = row.get("operating_income")
        row["fcf"] = (row["cfo"] + row["capex"]) if row.get("cfo") is not None and row.get("capex") is not None else None
        row["available_at"] = (date.fromisoformat(filed_max) + timedelta(days=1)).isoformat() if filed_max else None
        if any(row.get(k) is not None for k in ("revenue", "net_income", "total_assets")):
            periods.append(row)
    return periods


def latest_filed(company: dict, as_of: str) -> str | None:
    """as_of 以前最新一份年報的申報日（回測用：有新 10-K 才重算模型）。"""
    fs = [r[4] for r in company.get("rows") or [] if r[4] <= as_of]
    return max(fs) if fs else None


# ── 3. 檔案 ──────────────────────────────────────────────────────────────────

def save(store: dict, path: Path | str = OUT_PATH) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(p, "wt", encoding="utf-8") as f:
        json.dump(store, f, separators=(",", ":"))
    return p


def load(path: Path | str = OUT_PATH) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    with gzip.open(p, "rt", encoding="utf-8") as f:
        st = json.load(f)
    if st.get("tags") != TAGS:                                # 科目表變了：用檔內順序重建索引
        old = st.get("tags") or []
        pos = {t: i for i, t in enumerate(TAGS)}
        for c in (st.get("companies") or {}).values():
            c["rows"] = [[pos[old[r[0]]]] + r[1:] for r in c.get("rows") or [] if r[0] < len(old) and old[r[0]] in pos]
        st["tags"] = TAGS
    return st


def ticker_to_company(store: dict) -> dict:
    """{代碼: 公司}：抽取時對上的代碼（含 RENAMES 舊代碼）。代碼重複使用的早期期間另由 reuse_ok() 遮掉。"""
    out = {}
    for c in (store.get("companies") or {}).values():
        for t in c.get("tickers") or []:
            out[t] = c
    return out


def reuse_ok(ticker: str, d: str) -> bool:
    """代碼在日期 d 是否屬於現在這家公司（TICKER_REUSE：此日期前是別家公司 → False）。"""
    s = TICKER_REUSE.get(_norm(ticker))
    return s is None or str(d)[:10] >= s


# ── 4. Colab 進入點（需網路；SEC 擋雲端 IP 時退回手動下載）──────────────────────

def _norm(t: str) -> str:
    return str(t).strip().upper().replace(".", "-")


def sp500_tickers_since(periods: dict, since: str = SINCE) -> list[str]:
    return sorted(t for t, ps in periods.items() if any(b >= since for _a, b in ps))


def build_store(zip_path: str | Path, tickers_json: dict, sp_tickers: list[str], built_at: str,
                log=print) -> dict:
    """companyfacts.zip ＋ company_tickers.json ＋ 歷年成分代碼 → store（純本地 I/O，可離線測）。"""
    import zipfile
    by_tk: dict[str, int] = {}
    for v in (tickers_json or {}).values():
        try:
            by_tk[_norm(v["ticker"])] = int(v["cik_str"])
        except (KeyError, TypeError, ValueError):
            continue
    want: dict[int, list[str]] = {}
    miss = []
    for t in sp_tickers:
        cik = by_tk.get(_norm(t)) or by_tk.get(_norm(RENAMES.get(_norm(t), "")))   # 舊代碼 → 改名後的同一家公司
        if cik is None:
            miss.append(t)
        else:
            want.setdefault(cik, []).append(_norm(t))
    companies, n_rows, absent = {}, 0, 0
    with zipfile.ZipFile(zip_path) as z:
        names = set(z.namelist())
        for i, (cik, tks) in enumerate(sorted(want.items())):
            fn = f"CIK{cik:010d}.json"
            if fn not in names:
                absent += 1
                continue
            try:
                cf = json.loads(z.read(fn))
                rows = extract_company(cf)
            except Exception as e:                            # 單一公司壞檔不毀整批（計數揭露）
                log(f"  CIK {cik} 解析失敗（{type(e).__name__}），略過")
                absent += 1
                continue
            if rows:
                companies[str(cik)] = {"cik": cik, "name": cf.get("entityName"), "tickers": sorted(set(tks)), "rows": rows}
                n_rows += len(rows)
            if (i + 1) % 100 == 0:
                log(f"  已處理 {i + 1}/{len(want)} 家（{n_rows:,} 列）")
    return {"version": 1, "built_at": built_at, "since": SINCE, "tags": TAGS, "companies": companies,
            "stats": {"sp_tickers": len(sp_tickers), "matched_cik": len(want), "unmatched_tickers": len(miss),
                      "no_xbrl_file": absent, "companies": len(companies), "rows": n_rows}}


def colab_build(base_dir: str | Path, contact: str, out_name: str = "annual_pit.json.gz") -> dict:
    """Colab 用：下載（或讀取手動上傳的）companyfacts.zip 與 company_tickers.json → 寫精簡檔到 base_dir。
    contact：SEC 要求 User-Agent 帶聯絡資訊（只用於這次請求，不寫進任何檔案）。"""
    import csv
    import io
    import requests
    base = Path(base_dir)
    ua = {"User-Agent": f"RBS-research {str(contact).strip()}", "Accept-Encoding": "gzip, deflate"}
    print("1/4 讀取 S&P 500 成分歷史（fja05680）…")
    if "@" not in str(contact):
        raise ValueError("請輸入有效的 email（SEC 規定 User-Agent 要有聯絡方式；空白會被當成機器人擋掉）")
    rq = requests.get(SP500_PERIODS_URL, timeout=60)
    rq.raise_for_status()
    txt = rq.text
    periods: dict[str, list] = {}
    for r in csv.DictReader(io.StringIO(txt)):
        t = _norm(r.get("ticker") or "")
        if t:
            periods.setdefault(t, []).append(((r.get("start_date") or "")[:10], (r.get("end_date") or "")[:10] or "9999-12-31"))
    sp = sp500_tickers_since(periods)
    print(f"   2009 年後曾為成分：{len(sp)} 檔")
    if len(sp) < 400:
        raise RuntimeError(f"成分歷史只讀到 {len(sp)} 檔（應 > 800），GitHub raw 可能暫時失敗，請重跑")

    print("2/4 讀取 SEC 代碼對照表…")
    tj_path = base / "company_tickers.json"
    try:
        r = requests.get(TICKERS_URL, headers=ua, timeout=60)
        r.raise_for_status()
        tickers_json = r.json()
    except Exception as e:
        if not tj_path.exists():
            raise RuntimeError(f"SEC 代碼表下載失敗（{type(e).__name__}）。請用瀏覽器下載 {TICKERS_URL}"
                               f" 放到 {tj_path} 後重跑") from e
        tickers_json = json.loads(tj_path.read_text())
        print("   （使用手動上傳的 company_tickers.json）")

    print("3/4 取得 companyfacts.zip（SEC 夜間打包，約 1–2 GB）…")
    import zipfile
    zp = Path("/content/companyfacts.zip") if Path("/content").exists() else base / "companyfacts.zip"
    manual = base / "companyfacts.zip"
    for cand in (manual, zp):                                 # 殘缺的舊檔（下載中斷）先刪掉，否則會一直 BadZipFile
        if cand.exists() and not zipfile.is_zipfile(cand):
            print(f"   刪除殘缺的 {cand}")
            cand.unlink()
    if manual.exists():
        zp = manual
        print(f"   （使用手動上傳的 {manual}）")
    elif not zp.exists():
        part = zp.with_suffix(".part")
        try:
            with requests.get(COMPANYFACTS_URL, headers=ua, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("Content-Length") or 0)
                done, next_mark = 0, 256 << 20
                with open(part, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8 << 20):
                        f.write(chunk)
                        done += len(chunk)
                        if done >= next_mark:
                            print(f"   {done / 2**20:,.0f}" + (f" / {total / 2**20:,.0f}" if total else "") + " MB")
                            next_mark += 256 << 20
            if (total and done != total) or not zipfile.is_zipfile(part):
                raise IOError(f"下載不完整（{done}/{total} bytes）")
            part.rename(zp)                                   # 完整才改名成正式檔
        except Exception as e:
            if part.exists():
                part.unlink()
            raise RuntimeError(f"companyfacts.zip 下載失敗（{type(e).__name__}：{str(e)[:80]}；SEC 可能封鎖雲端 IP）。"
                               f"請用自己電腦的瀏覽器下載 {COMPANYFACTS_URL}，上傳到 {manual} 後重跑") from e

    print("4/4 抽取 DCF 所需科目（年報、帶申報日、版本去重）…")
    store = build_store(zp, tickers_json, sp, built_at=date.today().isoformat())
    out = save(store, base / out_name)
    st = store["stats"]
    if st["companies"] < 400:
        print(f"⚠️ 只抽到 {st['companies']} 家公司（預期 > 600），結果可能不完整——請把這段輸出貼給 Claude")
    print(f"完成：{st['companies']} 家公司、{st['rows']:,} 列；代碼對不到 CIK（多為已下市）{st['unmatched_tickers']} 檔；"
          f"檔案 {out.stat().st_size / 2**20:.1f} MB → {out}")
    return {"path": str(out), **st}


# ── 5. 自我測試（合成 companyfacts；離線）──────────────────────────────────────

def _fact(val, end, filed, start=None, form="10-K", fy=2020, fp="FY"):
    f = {"val": val, "end": end, "filed": filed, "form": form, "fy": fy, "fp": fp, "accn": "x"}
    if start:
        f["start"] = start
    return f


if __name__ == "__main__":
    import tempfile
    import zipfile

    cf = {"cik": 1, "entityName": "TestCo", "facts": {
        "us-gaap": {
            # 營收：2017 用 SalesRevenueNet，2018 起 ASC 606 換科目；2019 年報重編 2018 年營收
            "SalesRevenueNet": {"units": {"USD": [_fact(100, "2017-12-31", "2018-02-20", "2017-01-01")]}},
            "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
                _fact(110, "2018-12-31", "2019-02-20", "2018-01-01"),
                _fact(112, "2018-12-31", "2020-02-20", "2018-01-01", fy=2019),            # 重編
                _fact(110, "2018-12-31", "2019-02-20", "2018-01-01", fy=2018),            # 重複版本（值同）→ 去重
                _fact(125, "2019-12-31", "2020-02-20", "2019-01-01", fy=2019),
                _fact(30, "2019-09-30", "2019-11-01", "2019-07-01", form="10-Q"),        # 季報 → 不收
                _fact(31, "2019-12-31", "2020-02-20", "2019-10-01"),                     # 季期間 → 不收
            ]}},
            "OperatingIncomeLoss": {"units": {"USD": [_fact(20, "2018-12-31", "2019-02-20", "2018-01-01"),
                                                      _fact(25, "2019-12-31", "2020-02-20", "2019-01-01")]}},
            "PaymentsToAcquirePropertyPlantAndEquipment": {"units": {"USD": [_fact(8, "2018-12-31", "2019-02-20", "2018-01-01")]}},
            "PaymentsToAcquireProductiveAssets": {"units": {"USD": [_fact(9, "2019-12-31", "2020-02-20", "2019-01-01")]}},
            "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": [_fact(30, "2019-12-31", "2020-02-20", "2019-01-01")]}},
            "Assets": {"units": {"USD": [_fact(500, "2018-12-31", "2019-02-20"), _fact(550, "2019-12-31", "2020-02-20")]}},
            "LongTermDebtNoncurrent": {"units": {"USD": [_fact(100, "2019-12-31", "2020-02-20")]}},
            "LongTermDebtCurrent": {"units": {"USD": [_fact(10, "2019-12-31", "2020-02-20")]}},
            "CashAndCashEquivalentsAtCarryingValue": {"units": {"USD": [_fact(40, "2019-12-31", "2020-02-20")]}},
            "WeightedAverageNumberOfDilutedSharesOutstanding": {"units": {"shares": [_fact(1e6, "2019-12-31", "2020-02-20", "2019-01-01")]}},
            "SomeOtherTag": {"units": {"USD": [_fact(1, "2019-12-31", "2020-02-20")]}},
        },
        "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": [_fact(9.9e5, "2020-02-10", "2020-02-20")]}}},
    }}
    rows = extract_company(cf)
    rev_rows = [r for r in rows if TAGS[r[0]] == "RevenueFromContractWithCustomerExcludingAssessedTax"]
    assert len(rev_rows) == 3, rev_rows                            # 110、112（重編）、125；重複與季報不收
    assert not any(TAGS[r[0]] == "SomeOtherTag" for r in rows)
    print(f"✅ 1 抽取：只收年報年期間／時點值、版本去重（{len(rows)} 列）")

    comp = {"rows": rows}
    p19 = pit_periods(comp, "2019-06-30")                          # 2019 年中：只看得到 2018 年報
    assert [p["period_end"] for p in p19] == ["2018-12-31", "2017-12-31"], p19
    assert p19[0]["revenue"] == 110 and p19[1]["revenue"] == 100   # 2018 原始值（重編尚未發生）、2017 舊科目
    assert p19[0]["capex"] == -8 and p19[0]["available_at"] == "2019-02-21"
    p20 = pit_periods(comp, "2020-03-01")
    assert p20[0]["period_end"] == "2019-12-31" and p20[1]["revenue"] == 112   # 重編後的 2018
    assert p20[0]["capex"] == -9                                   # 換科目（ProductiveAssets）逐期合併
    assert p20[0]["total_debt"] == 110 and p20[0]["cash_and_sti"] == 40
    assert p20[0]["shares_out"] == 9.9e5 and p20[0]["diluted_shares"] == 1e6 and p20[0]["fcf"] == 30 - 9
    assert pit_periods(comp, "2018-01-01") == [] and latest_filed(comp, "2019-06-30") == "2019-02-20"
    def _debt(**tags):
        rr = [[TAGS.index(t), "2019-12-31", None, v, "2020-02-20", "10-K"] for t, v in tags.items()]
        rr.append([TAGS.index("Revenues"), "2019-12-31", "2019-01-01", 1.0, "2020-02-20", "10-K"])
        return pit_periods({"rows": rr}, "2020-03-01")[0]["total_debt"]
    assert _debt(LongTermDebtAndCapitalLeaseObligations=9000, LongTermDebtCurrent=500) == 9500        # A
    assert _debt(LongTermDebt=9500, ShortTermBorrowings=300) == 9800                                 # B
    assert _debt(LongTermDebtNoncurrent=9000, DebtCurrent=800, LongTermDebtCurrent=500) == 9800      # C：DebtCurrent 已含
    assert _debt(DebtCurrent=50) == 50 and _debt() is None
    # 重編以低優先科目申報 → 取最新申報（不混舊基準）
    rr2 = [[TAGS.index("Revenues"), "2017-12-31", "2017-01-01", 100.0, "2018-02-20", "10-K"],
           [TAGS.index("RevenueFromContractWithCustomerExcludingAssessedTax"), "2017-12-31", "2017-01-01", 95.0, "2019-02-20", "10-K"],
           [TAGS.index("RevenueFromContractWithCustomerExcludingAssessedTax"), "2018-12-31", "2018-01-01", 105.0, "2019-02-20", "10-K"]]
    pp = pit_periods({"rows": rr2}, "2019-03-01")
    assert [p["revenue"] for p in pp] == [105.0, 95.0], pp
    assert reuse_ok("JCI", "2017-01-03") and not reuse_ok("JCI", "2015-06-30") and reuse_ok("AAPL", "2010-01-04")
    print("✅ 2 當時可得：未來申報不可見、重編只在申報後生效、換科目逐期合併、負債組成（三種科目組合）、改名/重用代碼")

    # 3) build_store：zip ＋ 代碼表 ＋ 成分代碼（含對不到 CIK 的已下市代碼）；save/load 往返
    tmp = Path(tempfile.mkdtemp())
    zp = tmp / "companyfacts.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("CIK0000000001.json", json.dumps(cf))
    tj = {"0": {"cik_str": 1, "ticker": "TEST", "title": "TestCo"}, "1": {"cik_str": 2, "ticker": "NOFILE", "title": "X"}}
    st = build_store(zp, tj, ["TEST", "NOFILE", "GONE"], "2026-10-10", log=lambda *_: None)
    tj2 = {"0": {"cik_str": 1, "ticker": "META", "title": "TestCo"}}
    st2 = build_store(zp, tj2, ["FB"], "2026-10-10", log=lambda *_: None)          # 舊代碼 FB 接到 META 的 CIK
    assert st2["stats"]["matched_cik"] == 1 and ticker_to_company(st2)["FB"]["cik"] == 1
    assert st["stats"] == {"sp_tickers": 3, "matched_cik": 2, "unmatched_tickers": 1, "no_xbrl_file": 1,
                           "companies": 1, "rows": len(rows)}, st["stats"]
    out = save(st, tmp / "a.json.gz")
    back = load(out)
    assert back["companies"]["1"]["rows"] == [list(r) for r in rows] and ticker_to_company(back)["TEST"]["name"] == "TestCo"
    assert sp500_tickers_since({"A": [("2000-01-01", "2008-12-31")], "B": [("2000-01-01", "9999-12-31")]}) == ["B"]
    # company_model 吃得下 pit_periods 的輸出
    import company_model as cm
    res = cm.run_model(p20, {"ticker": "TEST", "price": 50.0, "mkt_cap": 5e7, "beta": 1.0, "shares": 1e6}, mc=False)
    assert res.get("method") == "dcf_fcff" and res.get("signal"), res.get("signal")
    print("✅ 3 build_store（未對上 CIK／缺檔計數）、save/load 往返、company_model 可直接吃 pit_periods 輸出")
    print("\nedgar_pit selftest OK ✅")
