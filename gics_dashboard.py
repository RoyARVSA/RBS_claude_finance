"""
gics_dashboard.py – GICS 產業輪動 Dashboard 的資料組裝（移植自使用者 gics_nn 專案的 dashboard.py + build 流程）

網頁畫面全部在 gics_template.html（使用者原始風格：深色終端機、RRG 四象限、RS vs SPX、
Strongest/Weakest、RS Ranking，L1→L4 切換）；本模組只負責：
  build_payload()  成分股收盤 + 股數 + GICS 代碼 → 模板吃的 JSON（可套「當時成分」遮罩）
  render_html()    把 payload 塞進模板（Streamlit 用 components.html 直接嵌入）
  build_live()     需網路：Wikipedia 成分 → SPY 官方持股權重 → Yahoo 收盤 → 分拆修正 → 成分期間遮罩
                   → 對不上分類的代碼套用每週神經網路分類結果（data/gics/classified.json）
  demo_payload()   合成資料（離線自測、CI）

研究工具、非投資建議；GICS 對照表為工作用，非 MSCI/S&P 官方 GICS Direct。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import gics_taxonomy as tx

KEEP_DAYS = 345          # 12M (252d) + RS-Ratio/Momentum warm-up (~14 weeks)
TEMPLATE = Path(__file__).parent / "gics_template.html"
GICS_DIR = Path(__file__).parent / "data" / "gics"


def build_payload(prices: pd.DataFrame, spx: pd.Series, universe: pd.DataFrame, shares: dict,
                  src: dict | None = None, conf: dict | None = None, source: str = "Yahoo Finance",
                  demo: bool = False, member_mask: pd.DataFrame | None = None, corp_actions: list | None = None) -> dict:
    """member_mask：與 prices 同 index 的布林表（True=當天在指數內）；False 的日子收盤設為 null，
    模板的群組指數自動排除（`C[i][t-1]>0 && C[i][t]>0` 才計入）。"""
    prices = prices.sort_index()
    spx = spx.reindex(prices.index).ffill()
    ok = spx.notna()
    prices, spx = prices[ok].iloc[-KEEP_DAYS:], spx[ok].iloc[-KEEP_DAYS:]
    uni = universe[universe["code8"].notna() & universe["ticker"].isin(prices.columns)].reset_index(drop=True)
    px = prices[uni["ticker"]].ffill(limit=3)   # tolerate short data gaps / holidays
    last = px.iloc[-1]
    if member_mask is not None:
        mm = member_mask.reindex(index=px.index, columns=px.columns).fillna(True).astype(bool)
        px_out = px.where(mm)
    else:
        px_out = px

    sh = []
    for t in uni["ticker"]:
        s = shares.get(t)
        sh.append(float(s) if s and s > 0 else np.nan)
    sh = pd.Series(sh, index=uni["ticker"])
    # missing share counts → median market cap / last price (so the name still gets a sensible weight)
    med_cap = float((sh * last).median()) if sh.notna().any() else 1e10
    sh = sh.fillna(med_cap / last.replace(0, np.nan)).fillna(1.0)

    used = set()
    for c in uni["code8"]:
        used.update(c[:d] for d in (2, 4, 6, 8))
    closes = [[None if pd.isna(v) else round(float(v), 4) for v in px_out[t].values] for t in uni["ticker"]]
    src = src or {}
    conf = conf or {}
    return {
        "demo": demo,
        "source": source,
        "pit": member_mask is not None,
        "dates": [d.strftime("%Y-%m-%d") for d in px.index],
        "spx": [round(float(v), 4) for v in spx.values],
        "tickers": uni["ticker"].tolist(),
        "names": uni["name"].astype(str).tolist(),
        "code8": uni["code8"].tolist(),
        "shares": [float(v) for v in sh.values],
        "src": [src.get(t, "table") for t in uni["ticker"]],
        "conf": [float(conf.get(t, 1.0)) for t in uni["ticker"]],
        "n_nn": int(sum(1 for t in uni["ticker"] if src.get(t) == "nn")),
        "names_gics": {c: tx.NAMES[c] for c in sorted(used)},
        "closes": closes,
        "corp_actions": list(corp_actions or []),
    }


def render_html(payload: dict) -> str:
    tpl = TEMPLATE.read_text(encoding="utf-8")
    payload = dict(payload)
    payload.setdefault("built_at", datetime.now().strftime("%Y-%m-%d %H:%M"))
    # 所有 < > & 轉成 \u003c 等：防 </script> 斷開與 <!--<script 讓瀏覽器進入 double-escaped 狀態（#64）
    data = json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return tpl.replace("__DATA__", data)


def write_html(payload: dict, out: str | Path) -> Path:
    out = Path(out)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(render_html(payload), encoding="utf-8")
    tmp.replace(out)
    return out


# ------------------------------------------------------------------ live build（需網路）
def classify_unmatched(uni: pd.DataFrame, gics_dir: Path | None = None) -> tuple[dict, dict]:
    """對 code8 缺值的代碼套用每週工作流的神經網路分類結果（data/gics/classified.json）；回 (src, conf)。
    模型權重不入庫（過大），所以這裡只讀結果檔；沒有檔或沒有該代碼 → 留空（模板只畫有分類的）。"""
    import gics_weekly as gw
    table = gw.lookup_table(gics_dir)
    src, conf = {}, {}
    for t in uni[uni["code8"].isna()]["ticker"].tolist():
        r = table.get(t)
        if r and r["code8"] in tx.NAMES:
            uni.loc[uni["ticker"] == t, "code8"] = r["code8"]
            src[t], conf[t] = r["src"], r["conf"]
    return src, conf


def build_live(universe: str = "sp500", cache_dir: Path | None = None, pit: bool = True) -> dict:
    """Streamlit / 夜間共用：抓資料 → payload。任何非關鍵來源失敗都降級繼續（SPY 權重 → Yahoo 股數 → 中位數）。"""
    import gics_data as gd
    from gics_model import FeatureBuilder
    cache = Path(cache_dir or GICS_DIR)
    cache.mkdir(parents=True, exist_ok=True)
    uni = gd.load_universe(universe)
    tickers = uni["ticker"].tolist()
    spinoffs = gd.load_spinoffs(cache)
    spincos = [s for p, _, s, _ in spinoffs if p in tickers and s and s not in tickers]
    px = gd.download_prices(tickers + spincos + ["SPY"] + FeatureBuilder.SECTOR_ETFS, period="2y")
    px, ca_log = gd.apply_corporate_actions(px, tickers, spinoffs)
    meta = {}
    mfile = cache / "meta_cache.json"
    if mfile.exists():                        # 夜間工作流維護的快取（股數、公司描述）；Streamlit 端不即時抓 .info
        try:
            meta = json.loads(mfile.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    src, conf = classify_unmatched(uni, cache)
    uni["name"] = [meta.get(t, {}).get("name") or n for t, n in zip(uni["ticker"], uni["name"])]
    shares = {t: (meta.get(t) or {}).get("shares") for t in tickers}
    spy = gd.spy_holdings_shares(cache / "spy_holdings.xlsx")
    if spy:
        shares = {t: spy.get(t, shares.get(t)) for t in tickers}
    mask = None
    if pit and universe == "sp500":
        try:
            import universe as un
            mask = gd.membership_mask(px.index, tickers, un.fetch_sp500_periods())
        except Exception:
            mask = None
    stock_px = px[[c for c in tickers if c in px.columns]]
    label = f"Yahoo Finance · {universe}" + (" · SPY official weights" if spy else " · Yahoo shares")
    out = build_payload(stock_px, px["SPY"], uni, shares, src, conf, source=label,
                        member_mask=mask, corp_actions=ca_log)
    out["built_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")    # 資料抓取時間（不是畫面渲染時間）
    return out


# ------------------------------------------------------------------ demo（離線）
def demo_payload(seed: int = 42, n_days: int = 380) -> dict:
    """合成宇宙：每個 L4 1–5 檔、產業有慢週期輪動（RRG 會繞象限）。不需網路。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end="2026-09-25", periods=n_days)
    n = len(dates)
    t = np.arange(n)
    mkt = np.cumsum(rng.normal(0.0004, 0.009, n))
    fac = {}
    for s in tx.codes_at(1):
        fac[s] = 0.10 * np.sin(2 * np.pi * t / rng.uniform(120, 260) + rng.uniform(0, 6.28)) + np.cumsum(rng.normal(0, 0.004, n))
    for c in tx.codes_at(2) + tx.codes_at(3):
        fac[c] = np.cumsum(rng.normal(0, 0.003, n)) + 0.04 * np.sin(2 * np.pi * t / rng.uniform(80, 200) + rng.uniform(0, 6))
    rows, cols, shares, used = [], {}, {}, set()
    alpha = list("ABCDEFGHIJKLMNOPRSTUVWXZ")
    for c8 in tx.SUB_INDUSTRIES:
        for _ in range(int(rng.integers(1, 6))):
            while True:
                tk = "".join(rng.choice(alpha, int(rng.integers(3, 5))))
                if tk not in used:
                    used.add(tk)
                    break
            logp = rng.uniform(0.7, 1.3) * mkt + fac[c8[:2]] + fac[c8[:4]] + fac[c8[:6]] + np.cumsum(rng.normal(0, 0.014, n))
            cols[tk] = 50 * np.exp(logp) * rng.uniform(0.3, 6)
            rows.append({"ticker": tk, "name": f"{tk} Corp", "sub_industry": tx.NAMES[c8], "code8": c8})
            shares[tk] = float(rng.lognormal(19.5, 1.0))
    uni = pd.DataFrame(rows)
    px = pd.DataFrame(cols, index=dates)
    spx = pd.Series(4000 * np.exp(mkt), index=dates)
    return build_payload(px, spx, uni, shares, source="synthetic demo data", demo=True)


# ------------------------------------------------------------------ 產業曝險（顯示用，不強制）
def sector_exposure(holdings: dict[str, float], code_of: dict[str, str], level: int = 1, warn_pct: float = 0.40) -> list[dict]:
    """holdings：{ticker: 市值}；code_of：{ticker: code8}。回依 level 彙總的曝險（佔比由大到小），
    超過 warn_pct 的標 warn。沒有分類的代碼歸「未分類」。純函數。"""
    tot = sum(v for v in holdings.values() if v and v > 0)
    if tot <= 0:
        return []
    d = tx.LEVEL_DIGITS[level]
    agg: dict[str, dict] = {}
    for t, v in holdings.items():
        if not v or v <= 0:
            continue
        c8 = code_of.get(t)
        key = c8[:d] if c8 else "?"
        a = agg.setdefault(key, {"code": key, "name": tx.NAMES.get(key, "未分類"), "value": 0.0, "tickers": []})
        a["value"] += float(v)
        a["tickers"].append(t)
    out = sorted(agg.values(), key=lambda a: -a["value"])
    for a in out:
        a["pct"] = a["value"] / tot
        a["warn"] = a["pct"] > warn_pct
    return out


if __name__ == "__main__":
    p = demo_payload()
    T, NT = len(p["dates"]), len(p["tickers"])
    assert T == KEEP_DAYS and NT > 300 and len(p["closes"]) == NT and all(len(c) == T for c in p["closes"])
    assert len(p["spx"]) == T and set(c[:2] for c in p["code8"]) == set(tx.codes_at(1))
    assert all(k in p["names_gics"] for c in p["code8"] for k in (c[:2], c[:4], c[:6], c))
    html = render_html(p)
    assert "__DATA__" not in html and '"demo":true' in html and "<\\/" not in html.split("<script id=\"data\"")[0]
    blob = html.split('<script id="data" type="application/json">', 1)[1].split("</script>", 1)[0]
    assert json.loads(blob)["tickers"] == p["tickers"]                       # 內嵌 JSON 可解析、沒被 </ 斷開
    # 成分遮罩：遮掉的日子為 null
    idx = pd.bdate_range("2026-01-05", periods=400)
    pxs = pd.DataFrame({"A": np.linspace(10, 20, 400), "B": np.linspace(20, 10, 400)}, index=idx)
    spx = pd.Series(np.linspace(100, 110, 400), index=idx)
    uni = pd.DataFrame({"ticker": ["A", "B"], "name": ["A", "B"], "code8": ["45301020", "10102010"]})
    mm = pd.DataFrame({"A": [i >= 200 for i in range(400)], "B": [True] * 400}, index=idx)
    pp = build_payload(pxs, spx, uni, {"A": 1e6, "B": 2e6}, member_mask=mm)
    a = pp["closes"][0]
    assert pp["pit"] and a[0] is None and a[-1] is not None and all(v is not None for v in pp["closes"][1])
    # 名稱含 </script> 不會斷開
    uni2 = uni.copy(); uni2.loc[0, "name"] = "Evil</script><script>alert(1)</script>"
    h2 = render_html(build_payload(pxs, spx, uni2, {"A": 1e6, "B": 2e6}))
    assert "Evil</script>" not in h2 and "Evil\\u003c/script\\u003e" in h2
    uni3 = uni.copy(); uni3.loc[0, "name"] = "X<!--<script>"
    h3 = render_html(build_payload(pxs, spx, uni3, {"A": 1e6, "B": 2e6}))
    blob3 = h3.split('<script id="data" type="application/json">', 1)[1].split("</script>", 1)[0]
    assert "<" not in blob3 and json.loads(blob3)["names"][0] == "X<!--<script>"         # 跳脫後仍能還原
    # 產業曝險
    ex = sector_exposure({"NVDA": 60, "AMD": 20, "XOM": 20, "ZZZ": 0}, {"NVDA": "45301020", "AMD": "45301020", "XOM": "10102010"})
    assert ex[0]["code"] == "45" and abs(ex[0]["pct"] - 0.8) < 1e-9 and ex[0]["warn"] and not ex[1]["warn"]
    ex2 = sector_exposure({"NVDA": 50, "NEW": 50}, {"NVDA": "45301020"}, level=4)
    assert {e["code"] for e in ex2} == {"45301020", "?"} and sector_exposure({}, {}) == []
    print(f"gics_dashboard selftest OK ✅（demo {NT} names × {T} days, html {len(html) / 1e6:.1f} MB）")
