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

#91 負債科目擴充（2026-10）：rows 用「科目在 TAGS 的整數索引」存科目，舊檔用的順序凍結在 LEGACY_TAGS，
新科目一律接在後面（舊檔解碼不變）。沒有長債總額科目時才走 DEBT_FALLBACK_TAGS 的「不重複計算下限」退路。
重建由 Colab Cell 5（sec_bulk.colab_build_all）負責；大於 24 MB 的檔自動切成 *.partN.json.gz（save_parts／load_parts）。
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import time
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
    # #91 擴充（新科目接在各清單尾端＝優先序最低）：…AndCapitalLeaseObligationsCurrent＝長債＋租賃的一年內到期部分
    # （同 LongTermDebtCurrent 語意、多含融資租賃）；…IncludingCurrentMaturities＝長債＋租賃總額、含一年內到期（同 LongTermDebt）
    "debt_nc": ["LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations"],
    "debt_cur_total": ["DebtCurrent"],
    "debt_lt_cur": ["LongTermDebtCurrent", "LongTermDebtAndCapitalLeaseObligationsCurrent"],
    "debt_st": ["ShortTermBorrowings", "CommercialPaper"],
    "debt_lt_total": ["LongTermDebt", "LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities"],
}
# #91 退路：只在「沒有非流動長債（debt_nc）也沒有長債總額（debt_lt_total）」時使用——這些科目多半只是負債的一部分
# （只有優先債、只有可轉債…），所以每一族只取一個值、各族取最大（max 不是加總）＝不重複計算的下限；
# 語意不明的 LongTermNotesPayable 一律當「含一年內到期」處理（只加短借，不加流動長債 → 寧可少算、不重複算）。
# 這些欄位只用來合成 total_debt，不出現在 pit_periods 的輸出（沒用到這些科目的公司，輸出與擴充前逐位相同）。
DEBT_FALLBACK_TAGS = {
    "debt_all_total": ["DebtLongtermAndShorttermCombinedAmount",              # 長債（含一年內到期）＋短債
                       "DebtAndCapitalLeaseObligations"],                     # 全部借款＋租賃
    "debt_lt_conv": ["ConvertibleNotesPayable", "ConvertibleDebt"],          # 可轉債（流動＋非流動）
    "debt_lt_senior": ["SeniorNotes"],                                        # 優先債（流動＋非流動）
    "debt_lt_notes": ["LongTermNotesPayable"],                                # 長期應付票據（語意不明 → 保守當總額）
    "debt_nc_conv": ["ConvertibleDebtNoncurrent", "ConvertibleLongTermNotesPayable"],   # 可轉債非流動
    "debt_nc_senior": ["SeniorLongTermNotes"],                                # 優先債非流動
    "debt_nc_other": ["OtherLongTermDebtNoncurrent"],                         # 其他長債非流動
    "debt_cur_conv": ["ConvertibleNotesPayableCurrent", "ConvertibleDebtCurrent"],      # 可轉債流動
    "debt_cur_senior": ["SeniorNotesCurrent"],                                # 優先債流動
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
# 分拆除權日（S&P 500 母公司把子公司股票配給股東；鍵＝母公司現行代碼）：除權後價格下跳，但前一份年報
# 仍含被分拆事業 → 估值看起來很便宜。回測把除權日當斷點（break_dates）。資料：2026-10 網路研究＋逐筆對抗核對，
# 排除交換要約與 < 5% 的分拆；出處與排除理由見 data/edgar/spinoffs.json（self-test 核對兩邊一致）
SPINOFFS: dict[str, list[str]] = {
    "A": ["2014-11-03"], "AA": ["2016-11-01"], "ABT": ["2013-01-02"], "ADP": ["2014-10-01"],
    "AIV": ["2020-12-15"], "APD": ["2016-10-03"], "APTV": ["2017-12-05", "2026-04-01"], "ARNC": ["2020-04-01"],
    "BAX": ["2015-07-01"], "BBWI": ["2021-08-03"], "BDX": ["2026-02-10"], "BEAM": ["2011-10-04"],
    "BIIB": ["2017-02-02"], "BWA": ["2023-07-05"], "CAG": ["2016-11-10"], "CHK": ["2014-07-01"],
    "CMCSA": ["2026-01-05"], "COP": ["2012-05-01"], "COV": ["2013-07-01"], "CSC": ["2015-11-30"],
    "CTVA": ["2026-10-01"], "CTXS": ["2017-02-01"], "CVC": ["2011-07-01"],
    "DD": ["2015-07-01", "2019-04-02", "2019-06-03", "2025-11-03"], "DF": ["2013-05-24"],
    "DHR": ["2016-07-05", "2023-10-02"], "DLPH": ["2017-12-05"], "DOC": ["2016-11-01"],
    "DOV": ["2014-03-03", "2018-05-09"], "DRI": ["2015-11-10"], "DTE": ["2021-07-01"],
    "DWDP": ["2019-04-02", "2019-06-03"], "DXC": ["2015-11-30", "2018-06-01"], "EBAY": ["2015-07-20"],
    "EQT": ["2018-11-13"], "EXC": ["2022-02-02"], "EXE": ["2014-07-01"], "EXPE": ["2011-12-21"],
    "FBHS": ["2022-12-15"], "FBIN": ["2022-12-15"], "FDX": ["2026-06-01"], "FMC": ["2019-03-04"],
    "FO": ["2011-10-04"], "FOXA": ["2019-03-19"], "FTV": ["2020-10-09", "2025-06-30"], "GCI": ["2015-06-29"],
    "GE": ["2023-01-04", "2024-04-02"], "HON": ["2025-10-30", "2026-06-29"], "HOT": ["2016-05-13"],
    "HPE": ["2017-04-03", "2017-09-01"], "HPQ": ["2015-11-02"], "HSH": ["2012-06-29"], "HSIC": ["2019-02-08"],
    "HWM": ["2016-11-01", "2020-04-01"], "IR": ["2013-12-02", "2020-03-02"], "ITT": ["2011-11-01"],
    "J": ["2024-09-30"], "JCI": ["2012-10-01", "2016-10-31"], "K": ["2023-10-02"], "KFT": ["2012-10-02"],
    "LDOS": ["2013-09-30"], "LEN": ["2025-01-21"], "LH": ["2023-07-03"], "MAR": ["2011-11-22"],
    "MAS": ["2015-07-01"], "MDLZ": ["2012-10-02"], "MET": ["2017-08-07"], "MMM": ["2024-04-01"],
    "MOT": ["2011-01-04"], "MRO": ["2011-07-01"], "MSI": ["2011-01-04"], "MUR": ["2013-09-03"],
    "MWV": ["2012-05-01"], "NE": ["2014-08-04"], "NI": ["2015-07-02"], "NOC": ["2011-03-31"],
    "NOV": ["2014-06-02"], "NWSA": ["2013-07-01"], "OKE": ["2014-02-03"], "PFE": ["2020-11-17"],
    "PNR": ["2018-05-01"], "PPL": ["2015-06-02"], "RTX": ["2020-04-03"], "SAI": ["2013-09-30"],
    "SLE": ["2012-06-29"], "SLM": ["2014-05-01"], "SPG": ["2014-05-29"], "SPGI": ["2026-07-01"],
    "STR": ["2010-07-01"], "SUN": ["2012-01-18"], "SWY": ["2014-04-15"], "T": ["2022-04-11"],
    "TFCFA": ["2019-03-19"], "TGNA": ["2015-06-29", "2017-06-01"], "TNL": ["2018-06-01"],
    "TT": ["2013-12-02", "2020-03-02"], "TYC": ["2012-10-01"], "VAR": ["2017-01-30"], "VFC": ["2019-05-23"],
    "VLO": ["2013-05-02"], "VNO": ["2015-01-16", "2017-07-18"], "VTR": ["2015-08-18"], "VZ": ["2010-07-02"],
    "WDC": ["2025-02-24"], "WMB": ["2012-01-03"], "WRK": ["2016-05-16"], "WYN": ["2018-06-01"],
    "XRX": ["2017-01-03"], "YUM": ["2016-11-01"],
}
DEI_TAGS = {"cover_shares": ["EntityCommonStockSharesOutstanding"]}     # 封面股數（申報日附近的流通股）
FLOW_FIELDS = {"revenue", "gross_profit", "operating_income", "pretax_income", "tax_provision", "net_income",
               "interest_expense", "diluted_shares", "sga", "rnd", "da", "capex", "sbc", "cfo", "buyback",
               "dividends_paid"}
SHARE_FIELDS = {"diluted_shares", "shares_out", "cover_shares"}


# 舊檔（2026-10-17 建置的 data/edgar/annual_pit.json.gz）的科目索引順序——**凍結：不可改順序、不可刪除**。
# rows 的第一欄是這份清單的整數索引；新科目只能接在 TAGS 尾端（自測鎖住）
LEGACY_TAGS = (
    "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet", "SalesRevenueGoodsNet",
    "SalesRevenueServicesNet", "GrossProfit", "OperatingIncomeLoss",
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
    "IncomeTaxExpenseBenefit", "NetIncomeLoss", "ProfitLoss", "InterestExpense", "InterestExpenseNonoperating",
    "InterestExpenseDebt", "InterestAndDebtExpense", "WeightedAverageNumberOfDilutedSharesOutstanding",
    "SellingGeneralAndAdministrativeExpense", "ResearchAndDevelopmentExpense", "DepreciationDepletionAndAmortization",
    "DepreciationAndAmortization", "DepreciationAmortizationAndAccretionNet", "Depreciation",
    "PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets", "PaymentsForCapitalImprovements",
    "ShareBasedCompensation", "AllocatedShareBasedCompensationExpense", "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations", "PaymentsForRepurchaseOfCommonStock",
    "PaymentsOfDividendsCommonStock", "PaymentsOfDividends", "CashAndCashEquivalentsAtCarryingValue", "Cash",
    "CashCashEquivalentsAndShortTermInvestments", "ShortTermInvestments", "MarketableSecuritiesCurrent",
    "AvailableForSaleSecuritiesCurrent", "AccountsReceivableNetCurrent", "InventoryNet", "AccountsPayableCurrent",
    "ContractWithCustomerLiabilityCurrent", "DeferredRevenueCurrent", "PropertyPlantAndEquipmentNet", "Goodwill",
    "Assets", "AssetsCurrent", "LiabilitiesCurrent", "StockholdersEquity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest", "RetainedEarningsAccumulatedDeficit",
    "MinorityInterest", "CommonStockSharesOutstanding", "LongTermDebtNoncurrent",
    "LongTermDebtAndCapitalLeaseObligations", "DebtCurrent", "LongTermDebtCurrent", "ShortTermBorrowings",
    "CommercialPaper", "LongTermDebt", "EntityCommonStockSharesOutstanding",
)


def _tag_index() -> list[str]:
    """LEGACY_TAGS（凍結順序）＋之後新增的科目依 FIELD_TAGS → DEBT_FALLBACK_TAGS → DEI_TAGS 的出現順序接在後面。"""
    tags = list(LEGACY_TAGS)
    for d in (FIELD_TAGS, DEBT_FALLBACK_TAGS, DEI_TAGS):
        for lst in d.values():
            for t in lst:
                if t not in tags:
                    tags.append(t)
    return tags


TAGS = _tag_index()
SHARE_TAGS = ("WeightedAverageNumberOfDilutedSharesOutstanding", "CommonStockSharesOutstanding",
              "EntityCommonStockSharesOutstanding")


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
            want = "shares" if tag in SHARE_TAGS else "USD"
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


def _pick(by_tag: dict, tag_pos: dict, field: str, tags: list[str], pe: str) -> tuple:
    """某欄位在期末 pe 的值：先取「涵蓋這一期的最新一份申報」，再在那份申報裡依科目優先序挑 → (值, 申報日)；沒有 → (None, None)。
    逐期合併：避免舊的高優先科目蓋過新申報以低優先科目重編的數字（驗證 Low：ASC 606／重編混基準）。"""
    cands = []
    for rank, t in enumerate(tags):
        for k, (v, filed) in by_tag.get(tag_pos[t], ()):
            if ((field in FLOW_FIELDS and k[2] is not None and _near(k[1], pe)) or
                    (field not in FLOW_FIELDS and k[2] is None and (_near(k[1], pe) or
                     (field == "cover_shares" and 0 <= _days(pe, k[1]) <= 120)))):
                cands.append((filed, -rank, v))
    if not cands:
        return None, None
    fl = max(c[0] for c in cands)
    return max((c for c in cands if c[0] == fl), key=lambda c: c[1])[2], fl


def _debt_fallback(row: dict, aux: dict) -> tuple | None:
    """#91 退路（呼叫端保證 debt_nc 與 debt_lt_total 都沒有）：各候選都是「不重複計算的下限」，取最大者 → (值, 申報日)。
    候選：①全部借款合計 ②長債某一族（含一年內到期）＋短借 ③非流動某一族＋流動負債（DebtCurrent／流動長債＋短借；
    都沒有時用流動的某一族）④只有流動某一族。不比原本合成值大 → None（維持原值）。"""
    def best(*fields):
        got = [aux[f] for f in fields if aux.get(f) and aux[f][0] is not None]
        return max(got) if got else (None, None)
    cur_ = row.get("debt_cur_total")
    if cur_ is None and (row.get("debt_lt_cur") is not None or row.get("debt_st") is not None):
        cur_ = (row.get("debt_lt_cur") or 0.0) + (row.get("debt_st") or 0.0)
    cur_lb = (cur_, "") if cur_ is not None else best("debt_cur_conv", "debt_cur_senior")
    lt_lb, nc_lb = best("debt_lt_conv", "debt_lt_senior", "debt_lt_notes"), best("debt_nc_conv", "debt_nc_senior", "debt_nc_other")
    cands = []
    if aux.get("debt_all_total") and aux["debt_all_total"][0] is not None:
        cands.append(aux["debt_all_total"])
    if lt_lb[0] is not None:
        cands.append((lt_lb[0] + (row.get("debt_st") or 0.0), lt_lb[1]))
    if nc_lb[0] is not None:
        cands.append((nc_lb[0] + (cur_lb[0] or 0.0), max(nc_lb[1], cur_lb[1] or "")))
    if cur_ is None and cur_lb[0] is not None:
        cands.append(cur_lb)
    if not cands:
        return None
    top = max(cands, key=lambda c: c[0])
    old = row.get("total_debt")
    return None if old is not None and top[0] <= old else top


_FIELD_ITEMS = list(FIELD_TAGS.items()) + list(DEI_TAGS.items())


def pit_periods(company: dict, as_of: str, max_periods: int = 5) -> list[dict]:
    """某公司在 as_of 當天看得到的年度期別（新→舊），fin_data 格式（欄位名同 FIELD_TAGS 鍵；capex 為負值）。
    期末日由「已可見的年期間 flow 科目」決定；存量科目取期末 ±10 天的時點值。"""
    rows = company.get("rows") or []
    vis = _visible(rows, as_of)
    tag_pos = {t: i for i, t in enumerate(TAGS)}
    by_tag: dict[int, list] = {}                             # 先依科目分桶（逐科目掃全部可見值太慢）
    for k, vf in vis.items():
        by_tag.setdefault(k[0], []).append((k, vf))
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
        filed_of: dict[str, str] = {}                        # 每個科目取自哪一天的申報（股數換拆股基準用）
        filed_max = ""
        for field, tags in _FIELD_ITEMS:                     # 逐期合併（見 _pick）
            val, fl = _pick(by_tag, tag_pos, field, tags, pe)
            if fl is not None:
                filed_max = max(filed_max, fl)
                filed_of[field] = fl
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
            fb = _debt_fallback(row, {f: _pick(by_tag, tag_pos, f, tags, pe) for f, tags in DEBT_FALLBACK_TAGS.items()})
            if fb is not None:                                        # #91：沒有長債總額科目 → 不重複計算的下限
                row["total_debt"], row["_debt_src"] = fb[0], "fallback"
                filed_max = max(filed_max, fb[1] or "")
        if row.get("cash_and_sti") is None and row.get("cash") is not None:
            row["cash_and_sti"] = row["cash"] + (row.get("short_term_investments") or 0.0)
        if row.get("capex") is not None and row["capex"] > 0:
            row["capex"] = -row["capex"]                      # 統一負值（現金流出），同 fin_data
        if row.get("shares_out") is None and row.get("cover_shares") is not None:
            row["shares_out"] = row["cover_shares"]
            filed_of["shares_out"] = filed_of.get("cover_shares")
        row["ebit"] = row.get("operating_income")
        row["fcf"] = (row["cfo"] + row["capex"]) if row.get("cfo") is not None and row.get("capex") is not None else None
        row["available_at"] = (date.fromisoformat(filed_max) + timedelta(days=1)).isoformat() if filed_max else None
        row["_filed"] = filed_of
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
    """讀年報檔（含自動合併 *.partN.json.gz 分檔）。檔內科目表是現行 TAGS 的前綴（舊檔＝LEGACY_TAGS）→ 索引不變、不重建；
    其他順序 → 用檔內順序重建索引（丟掉現行版本不認得的科目）。"""
    st = load_parts(path)
    if st is None:
        return None
    old = st.get("tags") or []
    if old != TAGS[:len(old)]:                                # 科目表順序不同：用檔內順序重建索引
        pos = {t: i for i, t in enumerate(TAGS)}
        for c in (st.get("companies") or {}).values():
            c["rows"] = [[pos[old[r[0]]]] + r[1:] for r in c.get("rows") or [] if r[0] < len(old) and old[r[0]] in pos]
    st["tags"] = TAGS
    return st


# ── 3b. 自動分檔（GitHub 網頁上傳上限 25 MB → 每個檔 ≤ 24 MB；sec_bulk 的三個輸出檔共用）──────────
MAX_PART_BYTES = 24 * 2**20


def part_path(path: Path | str, i: int) -> Path:
    """第 i 個分檔的檔名：第 1 個就是原檔名；之後 xxx.json.gz → xxx.part2.json.gz、xxx.part3.json.gz …"""
    p = Path(path)
    if i <= 1:
        return p
    stem = p.name[:-len(".json.gz")] if p.name.endswith(".json.gz") else p.stem
    return p.with_name(f"{stem}.part{i}.json.gz")


def _gz(doc: dict) -> bytes:
    return gzip.compress(json.dumps(doc, separators=(",", ":"), ensure_ascii=False).encode("utf-8"), 9, mtime=0)


def save_parts(store: dict, path: Path | str, max_bytes: int = MAX_PART_BYTES, key: str = "companies") -> list[Path]:
    """把 store 寫成一個或多個 gzip JSON（每個 ≤ max_bytes）。只切 store[key]（依公司切，同一家公司不跨檔）；
    其他頂層鍵（含 manifest）每個分檔都有一份，manifest 另記 part／parts／build_id／part_key 供 load_parts 核對。
    多出來的舊分檔（上次切得比較多）會刪掉。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    head = {k: v for k, v in store.items() if k != key}
    items = list((store.get(key) or {}).items())             # 保留原順序（build_store 依 CIK 排序）
    man0 = dict(head.get("manifest") or {})
    build_id = man0.get("build_id") or hashlib.sha1(
        (str(man0.get("built_at") or head.get("built_at")) + str(len(items)) + str(time.time())).encode()).hexdigest()[:12]
    sizes = [len(json.dumps(v, separators=(",", ":"), ensure_ascii=False)) + len(str(k)) + 8 for k, v in items]
    n = 1
    while True:
        target = sum(sizes) / n if n > 1 else float("inf")
        groups, cur, acc = [], [], 0.0
        for it, sz in zip(items, sizes):                      # 依序切成 n 段、每段未壓縮大小接近
            if cur and acc + sz > target and len(groups) < n - 1:
                groups.append(cur)
                cur, acc = [], 0.0
            cur.append(it)
            acc += sz
        groups.append(cur)
        n = len(groups)
        blobs = []
        for i, g in enumerate(groups, 1):
            doc = dict(head)
            doc["manifest"] = dict(man0, build_id=build_id, part=i, parts=n, part_key=key, part_items=len(g))
            doc[key] = dict(g)
            blobs.append(_gz(doc))
        big = max(len(b) for b in blobs)
        if big <= max_bytes:
            break
        if n >= len(items):
            raise ValueError(f"單一項目壓縮後就超過 {max_bytes / 2**20:.1f} MB，無法再切")
        n = min(len(items), max(n + 1, math.ceil(n * big / max_bytes * 1.15)))
    out = []
    for i, b in enumerate(blobs, 1):
        q = part_path(p, i)
        q.write_bytes(b)
        out.append(q)
    i = n + 1
    while part_path(p, i).exists():                           # 清掉上一次多出來的分檔
        part_path(p, i).unlink()
        i += 1
    return out


def load_parts(path: Path | str) -> dict | None:
    """讀 save_parts 寫的檔：第 1 個檔的 manifest 說有幾個分檔 → 逐一讀入、核對同一次建置（build_id）、合併 store[part_key]。
    沒有 manifest 的舊單檔照舊讀。缺分檔／混到別次建置 → 明確報錯（不默默少一段資料）。"""
    p = Path(path)
    if not p.exists():
        return None
    with gzip.open(p, "rt", encoding="utf-8") as f:
        st = json.load(f)
    man = st.get("manifest") or {}
    n, key = int(man.get("parts") or 1), man.get("part_key") or "companies"
    for i in range(2, n + 1):
        q = part_path(p, i)
        if not q.exists():
            raise FileNotFoundError(f"{p.name} 共 {n} 個分檔，缺 {q.name}——請把同一次產生的分檔全部上傳到同一個資料夾")
        with gzip.open(q, "rt", encoding="utf-8") as f:
            d = json.load(f)
        m2 = d.get("manifest") or {}
        if m2.get("build_id") != man.get("build_id") or int(m2.get("part") or 0) != i:
            raise ValueError(f"{q.name} 跟 {p.name} 不是同一次產生的（build_id 不符）——請重新上傳同一批分檔")
        st.setdefault(key, {}).update(d.get(key) or {})
    if man:
        st["manifest"] = dict(man, part=None, part_items=None, parts_loaded=n)
    return st


def ticker_to_company(store: dict) -> dict:
    """{代碼: 公司}：抽取時對上的代碼（含 RENAMES 舊代碼）＋各自的現行代碼（canonical）。
    離開指數後才改名的公司（ADS→BFH、GPS→GAP、WYND→TNL）成分表只有舊代碼，回測卻用現行代碼抓價格
    → 兩個都要對得到（對抗驗證 Low）。既有對應優先。代碼重複使用的早期期間另由 reuse_ok() 遮掉。"""
    out = {}
    for c in (store.get("companies") or {}).values():
        for t in c.get("tickers") or []:
            out[t] = c
    for t, c in list(out.items()):
        out.setdefault(canonical(t), c)
    return out


def reuse_ok(ticker: str, d: str) -> bool:
    """代碼在日期 d 是否屬於現在這家公司（TICKER_REUSE：此日期前是別家公司 → False）。"""
    s = TICKER_REUSE.get(_norm(ticker))
    return s is None or str(d)[:10] >= s


def canonical(ticker: str) -> str:
    """改名鏈走到底（FB→META、HCP→DOC）：回測一律用現行代碼抓價格、併成分期間。"""
    t, seen = _norm(ticker), set()
    while t in RENAMES and t not in seen:
        seen.add(t)
        t = RENAMES[t]
    return t


def break_dates(ticker: str) -> list[str]:
    """「斷點」：代碼重複使用日＋分拆除權日（SPINOFFS）。**期末早於斷點**的年報，於斷點（含）之後失效，
    等下一份期末在斷點之後的年報——看期末不看申報日：分拆後才申報、但期末在分拆前的年報仍含被分拆事業。
    表的鍵一律正規化成現行代碼再比對（填表時用分拆當時的舊代碼也對得到）。"""
    t = canonical(ticker)
    out = []
    for table in (SPINOFFS, {k: [v] for k, v in TICKER_REUSE.items()}):
        for k, ds in table.items():
            if canonical(k) == t:
                out.extend(ds)
    return sorted(set(out))


# ── 4. Colab 進入點（需網路；SEC 擋雲端 IP 時退回手動下載）──────────────────────

def _norm(t: str) -> str:
    return str(t).strip().upper().replace(".", "-")


def sp500_tickers_since(periods: dict, since: str = SINCE) -> list[str]:
    return sorted(t for t, ps in periods.items() if any(b >= since for _a, b in ps))


def parse_sp500_periods(csv_text: str) -> dict[str, list[tuple[str, str]]]:
    """fja05680 sp500_ticker_start_end.csv → {代碼: [(起, 迄)]}（迄空白＝仍在指數 → 9999-12-31）。"""
    import csv
    import io
    periods: dict[str, list] = {}
    for r in csv.DictReader(io.StringIO(csv_text)):
        t = _norm(r.get("ticker") or "")
        if t:
            periods.setdefault(t, []).append(((r.get("start_date") or "")[:10], (r.get("end_date") or "")[:10] or "9999-12-31"))
    return periods


def match_ciks(tickers_json: dict, sp_tickers: list[str]) -> tuple[dict[int, list[str]], list[str]]:
    """SEC company_tickers.json ＋ 歷年成分代碼 → ({CIK: [代碼…]}, 對不到 CIK 的代碼)。舊代碼經 RENAMES 接到改名後的同一家公司。
    年報（build_store）與 sec_bulk 的季報／8-K／內部人都用這個對應 → 同一個 CIK 宇宙。"""
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
    return want, miss


def build_store(zip_path: str | Path, tickers_json: dict, sp_tickers: list[str], built_at: str,
                log=print) -> dict:
    """companyfacts.zip ＋ company_tickers.json ＋ 歷年成分代碼 → store（純本地 I/O，可離線測）。"""
    import zipfile
    want, miss = match_ciks(tickers_json, sp_tickers)
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
    import requests
    base = Path(base_dir)
    ua = {"User-Agent": f"RBS-research {str(contact).strip()}", "Accept-Encoding": "gzip, deflate"}
    print("1/4 讀取 S&P 500 成分歷史（fja05680）…")
    if "@" not in str(contact):
        raise ValueError("請輸入有效的 email（SEC 規定 User-Agent 要有聯絡方式；空白會被當成機器人擋掉）")
    rq = requests.get(SP500_PERIODS_URL, timeout=60)
    rq.raise_for_status()
    periods = parse_sp500_periods(rq.text)
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

    # 4) 分拆表：鍵已正規化、日期是合法的平日、已排序不重複、與 data/edgar/spinoffs.json 的 used 事件一致
    for k, ds in SPINOFFS.items():
        assert k == _norm(k) and ds == sorted(set(ds)), (k, ds)
        for d_ in ds:
            assert date.fromisoformat(d_).weekday() < 5, (k, d_)
    sp_json = Path(__file__).resolve().parent / "data" / "edgar" / "spinoffs.json"
    if sp_json.exists():
        used = {e["ex_date"] for e in json.loads(sp_json.read_text(encoding="utf-8"))["events"] if e.get("used")}
        assert used == {d_ for ds in SPINOFFS.values() for d_ in ds}, used ^ {d_ for ds in SPINOFFS.values() for d_ in ds}
    print(f"✅ 4 分拆表 {len(SPINOFFS)} 檔（鍵正規化、平日、與 spinoffs.json 一致）")

    # 5) #91 舊檔相容：科目索引只能往後加；舊檔（2026-10-17 建置）解碼與 pit_periods 輸出逐位不變
    assert tuple(TAGS[:len(LEGACY_TAGS)]) == LEGACY_TAGS and len(set(TAGS)) == len(TAGS) and len(LEGACY_TAGS) == 63
    _all_lists = [t for d in (FIELD_TAGS, DEBT_FALLBACK_TAGS, DEI_TAGS) for lst in d.values() for t in lst]
    assert set(_all_lists) == set(TAGS), set(TAGS) ^ set(_all_lists)          # 舊科目都還在某個欄位裡（照樣抽取）
    assert not set(DEBT_FALLBACK_TAGS) & (set(FIELD_TAGS) | set(DEI_TAGS))
    real = HERE / "data" / "edgar" / "annual_pit.json.gz"
    if real.exists():
        with gzip.open(real, "rt", encoding="utf-8") as f:
            raw = json.load(f)
        rt = raw.get("tags") or []
        assert rt == TAGS[:len(rt)], "檔內科目表不是現行 TAGS 的前綴"
        assert [TAGS[i] for i in range(len(rt))] == rt                         # (a) 舊索引解碼成同樣的科目名
        st_real = load(real)
        some = sorted(st_real["companies"], key=int)[:20]
        assert all(st_real["companies"][k]["rows"] == raw["companies"][k]["rows"] for k in some)   # 前綴 → 不重建索引

        def _sample(store, n=50):
            ciks = sorted(store["companies"], key=int)
            step = max(1, len(ciks) // n)
            for cik in ciks[::step][:n]:
                c = store["companies"][cik]
                yield cik, c, sorted({r[4] for r in c.get("rows") or []}) + [f"{y}-06-30" for y in range(2011, 2027, 2)]

        if rt == list(LEGACY_TAGS) and raw.get("built_at") == "2026-10-17":
            # (b) 黃金摘要：擴充前的程式（git 941305d）在同一批 50 家 × 1,174 個日期算出的 pit_periods 雜湊
            GOLDEN = "5b18fc1dafd142f8023b1b94810ed4e15122ea2e589cad2d7ea7255e4656c200"
            h, k = hashlib.sha256(), 0
            for cik, c, ds in _sample(st_real):
                for d_ in ds:
                    h.update(json.dumps([cik, d_, pit_periods(c, d_)], sort_keys=True, separators=(",", ":")).encode())
                    k += 1
            assert k == 1174 and h.hexdigest() == GOLDEN, (k, h.hexdigest())
            # (b') 有 git 歷史時再直接跟舊版程式逐筆比（CI 淺層 clone 沒有舊 commit → 略過，黃金摘要已涵蓋）
            import importlib.util
            import subprocess
            LEGACY_REF = "941305d"                                            # 最後一個用舊科目表的 edgar_pit 版本
            try:
                src = subprocess.run(["git", "show", f"{LEGACY_REF}:edgar_pit.py"], cwd=HERE, capture_output=True,
                                     text=True, timeout=20)
                legacy_src = src.stdout if src.returncode == 0 else ""
            except (OSError, subprocess.SubprocessError):
                legacy_src = ""
            if legacy_src:
                lp = Path(tempfile.mkdtemp()) / "edgar_pit_legacy.py"
                lp.write_text(legacy_src, encoding="utf-8")
                spec = importlib.util.spec_from_file_location("edgar_pit_legacy", lp)
                old_mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(old_mod)
                st_old = old_mod.load(real)
                n_cmp = 0
                for cik, c, ds in _sample(st_real):
                    for d_ in ds:
                        assert old_mod.pit_periods(st_old["companies"][cik], d_) == pit_periods(c, d_), (cik, d_)
                        n_cmp += 1
                print(f"✅ 5 舊檔相容：索引前綴不變、50 家 × {k} 個日期與舊版程式逐筆相同（git {LEGACY_REF}）、黃金摘要相符")
            else:
                print(f"✅ 5 舊檔相容：索引前綴不變、50 家 × {k} 個日期黃金摘要相符（無 git 歷史 → 略過逐筆比對）")
        else:
            print(f"✅ 5 新版年報檔（{len(rt)} 個科目、前 {len(LEGACY_TAGS)} 個與舊表相同）解碼正常（舊檔黃金摘要不適用）")
    else:
        print("✅ 5 科目索引只往後加（repo 內沒有年報檔 → 略過真資料比對）")

    # 6) #91 退路：沒有長債總額科目時取「不重複計算的下限」；有長債科目時完全照舊
    def _td(**tags):
        rr = [[TAGS.index(t), "2019-12-31", None, v, "2020-02-20", "10-K"] for t, v in tags.items()]
        rr.append([TAGS.index("Revenues"), "2019-12-31", "2019-01-01", 1.0, "2020-02-20", "10-K"])
        p_ = pit_periods({"rows": rr}, "2020-03-01")[0]
        assert not set(DEBT_FALLBACK_TAGS) & set(p_), "退路欄位不可出現在輸出"
        return p_["total_debt"], p_.get("_debt_src")
    assert _td(DebtCurrent=7199, LongTermNotesPayable=124718) == (124718, "fallback")    # 語意不明 → 不加流動長債
    assert _td(SeniorNotes=1788, SeniorLongTermNotes=1788, SeniorNotesCurrent=0) == (1788, "fallback")   # 不重複算
    assert _td(DebtLongtermAndShorttermCombinedAmount=5000, ShortTermBorrowings=300) == (5000, "fallback")
    assert _td(ConvertibleNotesPayableCurrent=400, ConvertibleDebtNoncurrent=2000) == (2400, "fallback")
    assert _td(ConvertibleNotesPayableCurrent=400, ConvertibleDebtNoncurrent=2000, ConvertibleNotesPayable=2400) == (2400, "fallback")
    assert _td(ShortTermBorrowings=35, SeniorLongTermNotes=9000, OtherLongTermDebtNoncurrent=200) == (9035, "fallback")
    assert _td(LongTermDebtNoncurrent=9000, SeniorNotes=99999) == (9000, None)          # 有非流動長債 → 退路不啟用
    assert _td(LongTermDebt=9500, ConvertibleNotesPayable=99999) == (9500, None)
    assert _td(LongTermDebtAndCapitalLeaseObligations=9000, LongTermDebtAndCapitalLeaseObligationsCurrent=600) == (9600, None)
    assert _td(LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities=9600, ShortTermBorrowings=100) == (9700, None)
    assert _td(DebtCurrent=800, SeniorNotesCurrent=50) == (800, None)                     # 下限沒比較大 → 維持原值
    assert _td(SeniorNotesCurrent=50) == (50, "fallback") and _td() == (None, None)
    cf_sn = {"facts": {"us-gaap": {"SeniorLongTermNotes": {"units": {"USD": [_fact(9000, "2019-12-31", "2020-02-20")]}}}}}
    assert [TAGS[r[0]] for r in extract_company(cf_sn)] == ["SeniorLongTermNotes"] and TAGS.index("SeniorLongTermNotes") >= 63
    print("✅ 6 #91 退路：合計科目、各族取最大（不加總）、語意不明保守處理；有長債科目時完全照舊")

    # 7) 自動分檔：切成多個 ≤ 上限的檔、透明合併、缺檔／混檔報錯、舊分檔清掉
    import random
    rnd = random.Random(7)
    big = {"version": 1, "manifest": {"kind": "test", "built_at": "2026-10-25"}, "tags": TAGS,
           "companies": {str(i): {"cik": i, "rows": [[rnd.randrange(60), f"20{rnd.randrange(10, 26)}-12-31", None,
                                                       rnd.random() * 1e9, "2020-02-20", "10-K"] for _ in range(40)]}
                         for i in range(1, 61)}}
    paths = save_parts(big, tmp / "split.json.gz", max_bytes=9000)
    assert len(paths) >= 3 and all(q.stat().st_size <= 9000 for q in paths), [q.stat().st_size for q in paths]
    assert [q.name for q in paths[:2]] == ["split.json.gz", "split.part2.json.gz"]
    back = load_parts(tmp / "split.json.gz")
    assert back["companies"] == big["companies"] and back["manifest"]["parts_loaded"] == len(paths)
    assert load(tmp / "split.json.gz")["companies"]["5"]["rows"] == big["companies"]["5"]["rows"]
    paths[-1].rename(tmp / "gone.json.gz")
    try:
        load_parts(tmp / "split.json.gz")
        raise AssertionError("缺分檔應報錯")
    except FileNotFoundError:
        pass
    (tmp / "gone.json.gz").rename(paths[-1])
    other = save_parts(dict(big, manifest={"kind": "test", "build_id": "zzz"}), tmp / "other.json.gz", max_bytes=9000)
    paths[1].write_bytes(other[1].read_bytes())                                   # 混進別次建置的分檔
    try:
        load_parts(tmp / "split.json.gz")
        raise AssertionError("混檔應報錯")
    except ValueError:
        pass
    one = save_parts(big, tmp / "split.json.gz")                                   # 重寫成單檔 → 舊分檔刪掉
    assert len(one) == 1 and not part_path(tmp / "split.json.gz", 2).exists()
    assert load_parts(tmp / "split.json.gz")["companies"] == big["companies"]
    print(f"✅ 7 自動分檔：{len(paths)} 個檔皆 ≤ 上限、合併還原、缺檔／混檔報錯、舊分檔清除")
    print("\nedgar_pit selftest OK ✅")
