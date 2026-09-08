"""
fin_data.py – point-in-time 財報取數層（估值層 P0-c）

給 company_model / quality 用的統一三表資料：每期一筆 dict，**每筆帶 `period_end`（指涉期末）
與 `available_at`（可得知日）**——這是 Lean/Zipline/Qlib 共同的 PIT 紀律（VALUATION_LANDSCAPE §2）：
回測只能讀 `available_at ≤ as_of` 的期別；重編不覆寫（first-seen 先贏）。

來源與可得知日：
  • yfinance 三表（年 4 期 / 季 5 期；無申報日）→ available_at = period_end + 75 天（年）/ 45 天（季）
    ——比大型加速申報人的 60/40 天期限保守（寧可晚看到，不可早看到）
  • Finnhub `/stock/financials-reported`（免費、as-reported XBRL、含 filedDate）→ available_at = filedDate + 1 日
    作為 yfinance 缺席/不足時的備援；SEC 直連在 GitHub 機房被封（PITFALLS B10），不進主路徑
  • 每檔存 data/fin/{TICKER}.json（公開資料衍生，明文；append-only：同期別只補缺欄不改值）

純邏輯（normalize_* / merge_first_seen / pit_view / ttm）離線可測；fetch_* 需網路。教育用途，非投資建議。
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta
from pathlib import Path

FIN_DIR = Path(__file__).parent / "data" / "fin"
LAG_DAYS = {"A": 75, "Q": 45}          # yfinance 無申報日時的保守可得知延遲
STALE_DAYS = 7                          # 多久重抓一次
MAX_PERIODS = {"A": 12, "Q": 12}        # 每檔保留期數（Finnhub 可回填 10 年+）

# 標準欄 → (yfinance 報表 is/bs/cf, 列名候選由優先到備援；列名為 yfinance pretty=True 的 Title Case)
YF_FIELDS = {
    "revenue":            ("is", ["Total Revenue", "Operating Revenue"]),
    "gross_profit":       ("is", ["Gross Profit"]),
    "operating_income":   ("is", ["Operating Income", "EBIT"]),
    "ebit":               ("is", ["EBIT", "Operating Income"]),
    "pretax_income":      ("is", ["Pretax Income"]),
    "tax_provision":      ("is", ["Tax Provision"]),
    "net_income":         ("is", ["Net Income", "Net Income Common Stockholders"]),
    "interest_expense":   ("is", ["Interest Expense", "Interest Expense Non Operating"]),
    "diluted_shares":     ("is", ["Diluted Average Shares"]),
    "sga":                ("is", ["Selling General And Administration"]),
    "rnd":                ("is", ["Research And Development"]),
    "da":                 ("cf", ["Depreciation And Amortization", "Depreciation Amortization Depletion", "Depreciation"]),
    "capex":              ("cf", ["Capital Expenditure"]),
    "sbc":                ("cf", ["Stock Based Compensation"]),
    "cfo":                ("cf", ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"]),
    "fcf":                ("cf", ["Free Cash Flow"]),
    "buyback":            ("cf", ["Repurchase Of Capital Stock"]),
    "dividends_paid":     ("cf", ["Cash Dividends Paid", "Common Stock Dividend Paid"]),
    "cash":               ("bs", ["Cash And Cash Equivalents"]),
    "cash_and_sti":       ("bs", ["Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents"]),
    "total_debt":         ("bs", ["Total Debt"]),
    "shares_out":         ("bs", ["Ordinary Shares Number", "Share Issued"]),
    "receivables":        ("bs", ["Accounts Receivable", "Receivables"]),
    "inventory":          ("bs", ["Inventory"]),
    "payables":           ("bs", ["Accounts Payable", "Payables"]),
    "deferred_revenue":   ("bs", ["Current Deferred Revenue"]),
    "net_ppe":            ("bs", ["Net PPE"]),
    "goodwill_intang":    ("bs", ["Goodwill And Other Intangible Assets"]),
    "total_assets":       ("bs", ["Total Assets"]),
    "current_assets":     ("bs", ["Current Assets"]),
    "current_liabilities": ("bs", ["Current Liabilities"]),
    "total_equity":       ("bs", ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"]),
    "retained_earnings":  ("bs", ["Retained Earnings"]),
    "minority_interest":  ("bs", ["Minority Interest"]),
}
FLOW_FIELDS = {"revenue", "gross_profit", "operating_income", "ebit", "pretax_income", "tax_provision",
               "net_income", "interest_expense", "sga", "rnd", "da", "capex", "sbc", "cfo", "fcf",
               "buyback", "dividends_paid"}          # TTM 用加總；其餘為存量取最新

# Finnhub as-reported（us-gaap concept）fallback 鏈
FH_CONCEPTS = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "pretax_income": ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
                      "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"],
    "tax_provision": ["IncomeTaxExpenseBenefit"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "interest_expense": ["InterestExpense", "InterestExpenseNonoperating"],
    "diluted_shares": ["WeightedAverageNumberOfDilutedSharesOutstanding"],
    "sga": ["SellingGeneralAndAdministrativeExpense"],
    "rnd": ["ResearchAndDevelopmentExpense"],
    "da": ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization", "DepreciationAmortizationAndAccretionNet"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsForCapitalImprovements"],
    "sbc": ["ShareBasedCompensation", "AllocatedShareBasedCompensationExpense"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue"],
    "cash_and_sti": ["CashCashEquivalentsAndShortTermInvestments", "CashAndCashEquivalentsAtCarryingValue"],
    "receivables": ["AccountsReceivableNetCurrent"],
    "inventory": ["InventoryNet"],
    "payables": ["AccountsPayableCurrent"],
    "deferred_revenue": ["ContractWithCustomerLiabilityCurrent", "DeferredRevenueCurrent"],
    "net_ppe": ["PropertyPlantAndEquipmentNet"],
    "total_assets": ["Assets"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "total_equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "retained_earnings": ["RetainedEarningsAccumulatedDeficit"],
    "shares_out": ["CommonStockSharesOutstanding"],
}
FH_DEBT_PARTS = ["LongTermDebtNoncurrent", "LongTermDebtCurrent", "ShortTermBorrowings", "LongTermDebt"]


# ── 小工具 ────────────────────────────────────────────────────────────────

def _f(x):
    try:
        if x is None:
            return None
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _plus_days(d: str, n: int) -> str:
    return (datetime.strptime(d[:10], "%Y-%m-%d") + timedelta(days=n)).strftime("%Y-%m-%d")


def _period_key(p: dict) -> tuple:
    return (p.get("freq"), p.get("period_end"))


# ── 1. yfinance 三表 → 標準期別（純邏輯）────────────────────────────────────

def normalize_yf(income_df=None, balance_df=None, cashflow_df=None, freq: str = "A",
                 source: str = "yf") -> list[dict]:
    """yfinance DataFrame（列=項目、欄=期末 Timestamp）→ [{period_end, freq, available_at, fields…}]（新→舊）。"""
    frames = {"is": income_df, "bs": balance_df, "cf": cashflow_df}
    cols: set = set()
    for df in frames.values():
        if df is not None and not getattr(df, "empty", True):
            for c in df.columns:
                try:
                    cols.add(str(getattr(c, "date", lambda: c)())[:10] if hasattr(c, "date") else str(c)[:10])
                except Exception:
                    continue
    out = []
    for pe in sorted(cols, reverse=True):
        row = {"period_end": pe, "freq": freq, "available_at": _plus_days(pe, LAG_DAYS.get(freq, 75)),
               "source": source}
        for field, (sheet, names) in YF_FIELDS.items():
            df = frames.get(sheet)
            row[field] = None
            if df is None or getattr(df, "empty", True):
                continue
            col = next((c for c in df.columns
                        if (str(getattr(c, "date", lambda: c)())[:10] if hasattr(c, "date") else str(c)[:10]) == pe), None)
            if col is None:
                continue
            for nm in names:
                if nm in df.index:
                    v = _f(df.at[nm, col])
                    if v is not None:
                        row[field] = v
                        break
        if row.get("fcf") is None and row.get("cfo") is not None and row.get("capex") is not None:
            row["fcf"] = row["cfo"] + row["capex"]          # yfinance capex 為負值
        if any(row.get(k) is not None for k in ("revenue", "net_income", "total_assets")):
            out.append(row)
    return out


# ── 2. Finnhub as-reported → 標準期別（純邏輯）─────────────────────────────

def _concept_value(items: list, concepts: list[str]):
    """items: [{concept, value, unit, label}]；依 fallback 鏈取第一個有值者。"""
    if not isinstance(items, list):
        return None
    idx = {}
    for it in items:
        if isinstance(it, dict) and it.get("concept") is not None:
            c = str(it["concept"]).split(":")[-1]
            if c not in idx:
                idx[c] = _f(it.get("value"))
    for c in concepts:
        if c in idx and idx[c] is not None:
            return idx[c]
    return None


def normalize_finnhub(payload: dict | None, freq: str = "A") -> list[dict]:
    """Finnhub financials-reported → 標準期別；available_at = filedDate + 1（真 PIT）。"""
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        return []
    out = []
    for rep in payload["data"]:
        if not isinstance(rep, dict):
            continue
        r = rep.get("report") or {}
        pe = str(rep.get("endDate") or "")[:10]
        if not pe:
            continue
        filed = str(rep.get("filedDate") or rep.get("acceptedDate") or "")[:10]
        row = {"period_end": pe, "freq": freq,
               "available_at": _plus_days(filed, 1) if filed else _plus_days(pe, LAG_DAYS.get(freq, 75)),
               "source": "finnhub", "accession": rep.get("accessNumber"), "form": rep.get("form")}
        allitems = (r.get("ic") or []) + (r.get("bs") or []) + (r.get("cf") or [])
        for field, concepts in FH_CONCEPTS.items():
            row[field] = _concept_value(allitems, concepts)
        parts = [v for v in (_concept_value(allitems, [c]) for c in FH_DEBT_PARTS[:3]) if v is not None]
        row["total_debt"] = sum(parts) if parts else _concept_value(allitems, ["LongTermDebt"])
        if row.get("capex") is not None and row["capex"] > 0:
            row["capex"] = -row["capex"]                  # 統一為負值（現金流出）
        row["fcf"] = (row["cfo"] + row["capex"]) if (row.get("cfo") is not None and row.get("capex") is not None) else None
        if any(row.get(k) is not None for k in ("revenue", "net_income", "total_assets")):
            out.append(row)
    out.sort(key=lambda x: x["period_end"], reverse=True)
    return out


# ── 3. 儲存與合併（純邏輯）─────────────────────────────────────────────────

def new_store(ticker: str) -> dict:
    return {"ticker": ticker.upper(), "updated": None, "periods": []}


def merge_first_seen(store: dict, periods: list[dict], seen_date: str) -> int:
    """
    同 (freq, period_end)：既有值不改（first-seen = 當時已知，重編不覆寫），只補缺欄；
    新期別加入並記 first_seen。回新增/補欄的期數。
    """
    by = {_period_key(p): p for p in store["periods"]}
    changed = 0
    for p in periods:
        k = _period_key(p)
        if k in by:
            cur = by[k]
            filled = False
            for f, v in p.items():
                if f in ("period_end", "freq", "source", "available_at"):
                    continue
                if cur.get(f) is None and v is not None:
                    cur[f] = v
                    filled = True
            if filled:
                changed += 1
        else:
            q = dict(p)
            q["first_seen"] = seen_date[:10]
            # 第一次看到就已過期的期別（回填）：可得知日不得晚於 first_seen（避免回測看不到早已公開的資料）
            if q.get("available_at") and q["available_at"] > seen_date[:10]:
                q["available_at"] = min(q["available_at"], seen_date[:10])
            store["periods"].append(q)
            changed += 1
    # 期數上限（新→舊）
    for fr in ("A", "Q"):
        ps = sorted([p for p in store["periods"] if p.get("freq") == fr], key=lambda x: x["period_end"], reverse=True)
        keep = {id(p) for p in ps[: MAX_PERIODS.get(fr, 12)]}
        store["periods"] = [p for p in store["periods"] if p.get("freq") != fr or id(p) in keep]
    store["periods"].sort(key=lambda x: (x.get("freq"), x["period_end"]), reverse=True)
    store["updated"] = seen_date[:10]
    return changed


def load_store(ticker: str, base_dir: Path | None = None) -> dict:
    p = Path(base_dir or FIN_DIR) / f"{ticker.upper()}.json"
    if not p.exists():
        return new_store(ticker)
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(d, dict) and isinstance(d.get("periods"), list):
            return d
    except Exception:
        pass
    return new_store(ticker)


def save_store(store: dict, base_dir: Path | None = None) -> Path:
    base = Path(base_dir or FIN_DIR)
    base.mkdir(parents=True, exist_ok=True)
    p = base / f"{store['ticker']}.json"
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(store, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    tmp.replace(p)
    return p


def is_stale(store: dict, today: str, days: int = STALE_DAYS) -> bool:
    u = store.get("updated")
    if not u:
        return True
    try:
        return (datetime.strptime(today[:10], "%Y-%m-%d") - datetime.strptime(u, "%Y-%m-%d")).days >= days
    except Exception:
        return True


# ── 4. PIT 取用（純邏輯）───────────────────────────────────────────────────

def pit_view(store: dict, as_of: str | None = None, freq: str = "A") -> list[dict]:
    """as_of 當天已可得知的期別（新→舊）；as_of=None 取全部。"""
    ps = [p for p in store.get("periods", []) if p.get("freq") == freq]
    if as_of:
        ps = [p for p in ps if str(p.get("available_at") or "9999") <= as_of[:10]]
    return sorted(ps, key=lambda x: x["period_end"], reverse=True)


def ttm(store: dict, as_of: str | None = None) -> dict | None:
    """最近 4 季 TTM（流量加總、存量取最新季）；不足 4 季回 None。"""
    qs = pit_view(store, as_of, "Q")
    if len(qs) < 4:
        return None
    last4 = qs[:4]
    out = {"period_end": last4[0]["period_end"], "freq": "TTM", "available_at": last4[0]["available_at"]}
    for f in YF_FIELDS:
        vals = [q.get(f) for q in last4]
        if f in FLOW_FIELDS:
            known = [v for v in vals if v is not None]
            out[f] = sum(known) if len(known) == 4 else None    # 缺任一季不硬算（對抗驗證 B3）
        else:
            out[f] = next((v for v in vals if v is not None), None)
    return out


def series(store: dict, field: str, as_of: str | None = None, freq: str = "A", n: int = 5) -> list[tuple[str, float]]:
    """[(period_end, value)] 舊→新，最多 n 期。"""
    ps = pit_view(store, as_of, freq)[:n]
    return [(p["period_end"], p.get(field)) for p in reversed(ps)]


# ── 5. 抓取層（需網路）────────────────────────────────────────────────────────

def fetch_yf(ticker: str) -> tuple[list[dict], list[dict]]:
    """回 (年期別, 季期別)；任一失敗回空 list。"""
    import yfinance as yf
    tk = yf.Ticker(ticker)

    def _get(name):
        try:
            return getattr(tk, name)
        except Exception:
            return None
    annual = normalize_yf(_get("income_stmt"), _get("balance_sheet"), _get("cashflow"), "A")
    quarterly = normalize_yf(_get("quarterly_income_stmt"), _get("quarterly_balance_sheet"),
                             _get("quarterly_cashflow"), "Q")
    return annual, quarterly


def fetch_finnhub(ticker: str, key: str, freq: str = "annual") -> list[dict]:
    if not key:
        return []
    import requests
    try:
        r = requests.get("https://finnhub.io/api/v1/stock/financials-reported",
                         params={"symbol": ticker, "freq": freq, "token": key}, timeout=25)
        if r.ok:
            return normalize_finnhub(r.json(), "A" if freq == "annual" else "Q")
    except Exception:
        pass
    return []


def fetch_profile(ticker: str) -> dict:
    """估值用的市場資料：價、市值、β、股數、產業、幣別（yfinance info；失敗給空）。"""
    out = {}
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        try:
            fi = tk.fast_info
            out["price"] = _f(getattr(fi, "last_price", None))
            out["mkt_cap"] = _f(getattr(fi, "market_cap", None))
            out["shares"] = _f(getattr(fi, "shares", None))
        except Exception:
            pass
        try:
            info = tk.info or {}
            out.setdefault("price", _f(info.get("currentPrice") or info.get("regularMarketPrice")))
            out.setdefault("mkt_cap", _f(info.get("marketCap")))
            out.setdefault("shares", _f(info.get("sharesOutstanding")))
            out["beta"] = _f(info.get("beta"))
            out["sector"] = info.get("sector")
            out["industry"] = info.get("industry")
            out["currency"] = info.get("currency") or info.get("financialCurrency")
            out["name"] = info.get("shortName") or info.get("longName")
        except Exception:
            pass
    except Exception:
        pass
    return out


def get_financials(ticker: str, today: str, base_dir: Path | None = None, key: str | None = None,
                   force: bool = False, fetch_yf_fn=None, fetch_fh_fn=None) -> dict:
    """
    讀本地 store；過期或強制時抓 yfinance（主）→ 年期 <2 或失敗時補 Finnhub（備援）→
    first-seen 合併 → 落檔。任何抓取失敗都回既有 store（不炸）。
    """
    import os
    store = load_store(ticker, base_dir)
    if not force and not is_stale(store, today):
        return store
    fy = fetch_yf_fn or fetch_yf
    try:
        annual, quarterly = fy(ticker)
    except Exception:
        annual, quarterly = [], []
    merged = 0
    if annual or quarterly:
        merged += merge_first_seen(store, annual + quarterly, today)
    ann = pit_view(store, None, "A")
    thin = (not ann) or ann[0].get("revenue") is None or ann[0].get("cfo") is None
    if len(ann) < 2 or thin:                          # 年期不足或關鍵欄缺 → 備援（對抗驗證 B4）
        k = key if key is not None else os.environ.get("FINNHUB_API_KEY", "").strip()
        ff = fetch_fh_fn or (lambda t: fetch_finnhub(t, k, "annual"))
        try:
            fh = ff(ticker)
        except Exception:
            fh = []
        if fh:
            merged += merge_first_seen(store, fh, today)
    if merged or store.get("updated") != today[:10]:
        store["updated"] = today[:10]
        save_store(store, base_dir)
    return store


# ── 6. 自我測試（離線）─────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile
    import pandas as pd
    cols = [pd.Timestamp("2025-12-31"), pd.Timestamp("2024-12-31"), pd.Timestamp("2023-12-31")]
    is_df = pd.DataFrame({c: [10e9 * (1.1 ** -i), 2e9 * (1.1 ** -i), 1.5e9 * (1.1 ** -i), 0.4e9, 380e6]
                          for i, c in enumerate(cols)},
                         index=["Total Revenue", "Operating Income", "Net Income", "Tax Provision", "Diluted Average Shares"])
    bs_df = pd.DataFrame({c: [1.7e9, 2.9e9, 3.1e9, 1.4e9, 12e9, 3.9e9] for c in cols},
                         index=["Cash And Cash Equivalents", "Total Debt", "Accounts Receivable", "Inventory",
                                "Total Assets", "Stockholders Equity"])
    cf_df = pd.DataFrame({c: [3.0e8, -2.2e8, 2.1e9, 4.6e7] for c in cols},
                         index=["Depreciation And Amortization", "Capital Expenditure", "Operating Cash Flow",
                                "Stock Based Compensation"])

    # 1) yfinance 正規化：欄位對映、fcf 推導、available_at 延遲、缺表不炸
    A = normalize_yf(is_df, bs_df, cf_df, "A")
    assert len(A) == 3 and A[0]["period_end"] == "2025-12-31" and A[0]["available_at"] == "2026-03-16"
    assert A[0]["revenue"] == 10e9 and A[0]["capex"] == -2.2e8 and abs(A[0]["fcf"] - (2.1e9 - 2.2e8)) < 1
    assert A[0]["gross_profit"] is None and A[0]["total_debt"] == 2.9e9
    assert normalize_yf(None, None, None) == [] and normalize_yf(is_df.iloc[0:0], None, None) == []
    Q = normalize_yf(is_df, None, None, "Q")
    assert Q[0]["available_at"] == "2026-02-14"
    print("✅ 1 yfinance 正規化（對映/推導/PIT 延遲/缺表）")

    # 2) Finnhub 正規化：concept fallback、filedDate→available_at、capex 正負、債務加總
    fh = {"data": [{"endDate": "2025-12-31 00:00:00", "filedDate": "2026-02-20 00:00:00", "form": "10-K",
                    "accessNumber": "0001-26", "report": {
                        "ic": [{"concept": "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax", "value": 10e9},
                               {"concept": "OperatingIncomeLoss", "value": 2e9}, {"concept": "NetIncomeLoss", "value": 1.5e9}],
                        "bs": [{"concept": "LongTermDebtNoncurrent", "value": 2.8e9}, {"concept": "LongTermDebtCurrent", "value": 0.1e9},
                               {"concept": "Assets", "value": 12e9}, {"concept": "CashAndCashEquivalentsAtCarryingValue", "value": 1.7e9}],
                        "cf": [{"concept": "PaymentsToAcquirePropertyPlantAndEquipment", "value": 2.2e8},
                               {"concept": "NetCashProvidedByUsedInOperatingActivities", "value": 2.1e9}]}},
                   {"endDate": "2024-12-31", "filedDate": "2025-02-21", "report": {"ic": [{"concept": "Revenues", "value": 8e9}], "bs": [], "cf": []}},
                   {"endDate": "", "report": {}}, "junk"]}
    F = normalize_finnhub(fh, "A")
    assert len(F) == 2 and F[0]["available_at"] == "2026-02-21" and F[0]["revenue"] == 10e9
    assert F[0]["total_debt"] == 2.9e9 and F[0]["capex"] == -2.2e8 and abs(F[0]["fcf"] - 1.88e9) < 1
    assert F[1]["revenue"] == 8e9 and F[1]["total_assets"] is None
    assert normalize_finnhub({"data": "x"}) == [] and normalize_finnhub(None) == []
    print("✅ 2 Finnhub as-reported 正規化（fallback 鏈/filedDate/債務加總）")

    # 3) first-seen 合併：不覆寫既有值、只補缺欄、期數上限、first_seen 與回填 available_at
    st = new_store("TEST")
    n1 = merge_first_seen(st, A, "2026-04-01")
    assert n1 == 3 and st["periods"][0]["first_seen"] == "2026-04-01"
    A2 = json.loads(json.dumps(A)); A2[0]["revenue"] = 99e9; A2[0]["gross_profit"] = 3.6e9   # 重編 + 補欄
    n2 = merge_first_seen(st, A2, "2026-05-01")
    p0 = [p for p in st["periods"] if p["period_end"] == "2025-12-31"][0]
    assert p0["revenue"] == 10e9 and p0["gross_profit"] == 3.6e9 and n2 == 1        # 值不改、欄補上
    old = [{"period_end": f"20{y:02d}-12-31", "freq": "A", "available_at": f"20{y + 1:02d}-03-16", "revenue": 1.0}
           for y in range(5, 22)]
    merge_first_seen(st, old, "2026-05-01")
    assert len(pit_view(st, None, "A")) == MAX_PERIODS["A"]
    assert all(p["available_at"] <= "2026-05-01" for p in pit_view(st, None, "A"))    # 回填的可得知日不晚於 first_seen
    print("✅ 3 first-seen 合併（重編不覆寫/補缺欄/期數上限/回填可得知日）")

    # 4) PIT 取用：as_of 早於可得知日看不到；TTM 流量加總/存量取最新；series 舊→新
    st2 = new_store("PIT"); merge_first_seen(st2, A + Q, "2026-09-01")
    assert [p["period_end"] for p in pit_view(st2, "2026-03-01", "A")] == ["2024-12-31", "2023-12-31"]
    assert pit_view(st2, "2026-03-16", "A")[0]["period_end"] == "2025-12-31"
    qs = [{"period_end": d, "freq": "Q", "available_at": _plus_days(d, 45), "revenue": 1.0, "cash": c, "capex": -0.1}
          for d, c in (("2025-03-31", 5), ("2025-06-30", 6), ("2025-09-30", 7), ("2025-12-31", 8), ("2026-03-31", 9))]
    st3 = new_store("TTM"); merge_first_seen(st3, qs, "2026-09-01")
    t = ttm(st3, "2026-06-01")
    assert t and t["revenue"] == 4.0 and t["cash"] == 9 and abs(t["capex"] + 0.4) < 1e-9 and t["period_end"] == "2026-03-31"
    assert ttm(st3, "2025-08-01") is None                                                  # 只看得到 2 季
    assert series(st2, "revenue", None, "A", 2) == [("2024-12-31", A[1]["revenue"]), ("2025-12-31", 10e9)]
    print("✅ 4 PIT 取用（as_of 過濾/TTM/series）")

    # 5) 儲存 + get_financials 流程（假抓取器）：過期判斷、Finnhub 備援只在年期 <2 時、失敗不炸
    tmpd = Path(tempfile.mkdtemp())
    calls = {"yf": 0, "fh": 0}

    def fyf(t):
        calls["yf"] += 1
        return (A[:1], Q) if t == "ONEYEAR" else ((A, Q) if t != "DEAD" else (_ for _ in ()).throw(RuntimeError("x")))

    def ffh(t):
        calls["fh"] += 1
        return F
    s1 = get_financials("FULL", "2026-09-08", tmpd, key="k", fetch_yf_fn=fyf, fetch_fh_fn=ffh)
    assert len(pit_view(s1, None, "A")) == 3 and calls == {"yf": 1, "fh": 0} and (tmpd / "FULL.json").exists()
    s1b = get_financials("FULL", "2026-09-09", tmpd, key="k", fetch_yf_fn=fyf, fetch_fh_fn=ffh)
    assert calls["yf"] == 1 and s1b["updated"] == "2026-09-08"                             # 7 天內不重抓
    s2 = get_financials("ONEYEAR", "2026-09-08", tmpd, key="k", fetch_yf_fn=fyf, fetch_fh_fn=ffh)
    assert calls["fh"] == 1 and len(pit_view(s2, None, "A")) == 2                          # 備援補到 2 年
    s3 = get_financials("DEAD", "2026-09-08", tmpd, key="k", fetch_yf_fn=fyf, fetch_fh_fn=lambda t: [])
    assert s3["periods"] == [] and load_store("DEAD", tmpd)["ticker"] == "DEAD"
    assert is_stale(load_store("FULL", tmpd), "2026-09-20") and not is_stale(load_store("FULL", tmpd), "2026-09-10")
    print("✅ 5 取數流程（快取/備援條件/失敗不炸）")
    print("\nfin_data selftest OK ✅")
