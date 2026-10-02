"""
gics_weekly.py – GICS 每週工作流（.github/workflows/gics_weekly.yml，週六一次）

  1. 成分對照：Wikipedia S&P 500/400/600 → data/gics/members.json {ticker: code8}（約 1,500 檔，小檔）
  2. 公司資料：Yahoo .info（股數、描述、Yahoo 產業）有快取 → data/gics/meta_cache.json（描述截斷 600 字）
     對象 = S&P 500（訓練）∪ Alpha 選股池 broad ∪ watchlist（要分類的對象）
  3. 神經網路：S&P 1500 有標籤樣本 5 折交叉驗證 → 路徑解碼 L1 準確率 ≥ 門檻才採用 → 全樣本重訓 →
     對「不在 S&P 1500 清單」的選股池／watchlist 代碼分類 → data/gics/classified.json
     模型本身不入庫（權重檔過大；每週重訓即可），只存分類結果與交叉驗證報告
  4. 對帳：gics_verify → data/gics/verify.json（網頁角落與 /gics verify）

只讀 state 的明文鍵（watchlist）；輸出皆為公開資料衍生；不 print 持倉。
離線：`python3 gics_weekly.py --offline` 用合成資料走完整流程（CI 自測）。
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import gics_taxonomy as tx

ROOT = Path(__file__).parent
GICS_DIR = ROOT / "data" / "gics"
CV_MIN_L1 = 0.80          # 路徑解碼 L1 準確率門檻（未達 → 不更新分類結果）
SUMMARY_CHARS = 600


def lookup_table(gics_dir: Path | None = None) -> dict[str, dict]:
    """{ticker: {code8, src, conf}}：S&P 1500 對照表優先、其次神經網路分類結果。純讀檔、不連網。"""
    d = Path(gics_dir or GICS_DIR)
    out: dict[str, dict] = {}
    for name, src in (("classified.json", "nn"), ("members.json", "table")):
        p = d / name
        if not p.exists():
            continue
        try:
            blob = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows = blob.get("tickers", blob) if isinstance(blob, dict) else {}
        for t, v in rows.items():
            if isinstance(v, str):
                out[t] = {"code8": v, "src": src, "conf": 1.0}
            elif isinstance(v, dict) and v.get("code8"):
                out[t] = {"code8": v["code8"], "src": src, "conf": float(v.get("conf", 1.0))}
    return out


def lookup_text(ticker: str, table: dict) -> str:
    t = (ticker or "").strip().upper().replace(".", "-")[:12]
    shown = "".join(ch for ch in t if ch.isalnum() or ch == "-")       # 使用者輸入不進 Markdown 特殊字元（底線等）
    r = table.get(t)
    if not r or r["code8"] not in tx.NAMES:
        return f"🔎 {shown}：不在 S&P 1500 對照表，也還沒被神經網路分類（每週六更新選股池與觀察清單）"
    t = shown
    p = tx.path(r["code8"])
    src = "S&P 成分表" if r["src"] == "table" else f"神經網路 {r['conf']:.0%}"
    return (f"🔎 *{t}* GICS（{src}）\n"
            + "\n".join(f"L{l} {p[f'L{l}']['code']} {p[f'L{l}']['name']}" for l in (1, 2, 3, 4))
            + "\n_工作用對照表，非 MSCI/S&P 官方資料_")


def verify_text(gics_dir: Path | None = None) -> str:
    p = Path(gics_dir or GICS_DIR) / "verify.json"
    if not p.exists():
        return "🧾 GICS 對帳報告尚未產生（每週六工作流執行後才有）"
    v = json.loads(p.read_text(encoding="utf-8"))
    mark = {"PASS": "✅", "WARN": "⚠️", "NOTE": "ℹ️", "SKIP": "➖"}
    lines = [f"🧾 *GICS 對帳*（{v.get('as_of')}）：" + ("全部通過" if v.get("passed") else f"{v.get('n_warn')} 項警示")]
    for c in v.get("checks", [])[:8]:
        lines.append(f"{mark.get(c['status'], '·')} {c['name']}：{c['value']}")
    bad = [r for r in v.get("returns", []) if r.get("status") == "WARN"]
    ok = sum(1 for r in v.get("returns", []) if r.get("status") == "PASS")
    lines.append(f"報酬對帳：{ok} 組通過" + (f"、{len(bad)} 組超出容忍：" + "、".join(r["group"] for r in bad[:5]) if bad else ""))
    return "\n".join(lines)


def exposure_text(positions: dict, table: dict, level: int = 1) -> str:
    """Alpaca 持倉 → 產業曝險文字（只顯示、不強制）。私聊回覆用，不寫日誌。"""
    import gics_dashboard as gdb
    hold = {s.upper().replace(".", "-"): abs(float((p or {}).get("market_value") or 0)) for s, p in (positions or {}).items()}
    rows = gdb.sector_exposure(hold, {t: r["code8"] for t, r in table.items()}, level=level)
    if not rows:
        return "🧮 目前沒有持倉"
    lines = [f"🧮 *持倉產業曝險*（GICS L{level}，只顯示不強制）"]
    for r in rows:
        lines.append(f"{'⚠️' if r['warn'] else '•'} {r['name']} {r['pct']:.0%}（{len(r['tickers'])} 檔）")
    if any(r["warn"] for r in rows):
        lines.append("_單一群組超過 40%：集中度偏高，自動交易不受影響_")
    return "\n".join(lines)


# ------------------------------------------------------------------ 流程（資料可注入）
def run(members_df: pd.DataFrame, meta: dict, prices: pd.DataFrame, targets: list[str], out_dir: Path,
        epochs: int = 250, folds: int = 5) -> dict:
    """members_df：[ticker, code8]（S&P 1500 有標籤）；meta：{ticker: .info 摘要}；prices：收盤寬表（含 11 檔 SPDR ETF）；
    targets：要分類的代碼（選股池 ∪ watchlist）。回摘要 dict 並寫 members.json / classified.json / cv_report.json。"""
    import gics_data as gd
    from gics_model import FeatureBuilder, HierMLP, cross_validate
    out_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lab = members_df[members_df["code8"].notna()].drop_duplicates("ticker")
    members = dict(zip(lab["ticker"], lab["code8"]))
    (out_dir / "members.json").write_text(json.dumps({"as_of": today, "tickers": members}, separators=(",", ":")), encoding="utf-8")
    train_t = [t for t in lab["ticker"] if t in meta]
    y = [members[t] for t in train_t]
    summary = {"as_of": today, "n_members": len(members), "n_train": len(train_t), "cv": None, "adopted": False,
               "n_classified": 0}
    if len(train_t) < 50:
        summary["reason"] = "訓練樣本不足"
        return summary
    corr = gd.sector_etf_corr(prices, prices, train_t)
    metas = [meta[t] for t in train_t]
    rep = cross_validate(metas, corr, y, folds=folds, epochs=epochs)
    summary["cv"] = rep
    (out_dir / "cv_report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    l1 = rep["hierarchical_path_acc"]["L1"]
    if l1 < CV_MIN_L1:
        summary["reason"] = f"交叉驗證 L1 {l1:.2f} < {CV_MIN_L1}，不更新分類"
        return summary
    fb = FeatureBuilder()
    X = fb.fit_transform(metas, corr)
    mdl = HierMLP(epochs=epochs).fit(X, y)
    todo = [t for t in dict.fromkeys(targets) if t not in members and t in meta]
    classified = {}
    if todo:
        cc = gd.sector_etf_corr(prices, prices, todo)
        for t, p in zip(todo, mdl.predict(fb.transform([meta[t] for t in todo], cc))):
            classified[t] = {"code8": p["best"]["code8"], "conf": round(float(p["best"]["confidence"]), 3)}
    (out_dir / "classified.json").write_text(json.dumps({"as_of": today, "cv_l1": round(l1, 3),
                                                         "cv_l4": round(rep["hierarchical_path_acc"]["L4"], 3),
                                                         "tickers": classified}, separators=(",", ":")), encoding="utf-8")
    summary.update(adopted=True, n_classified=len(classified))
    return summary


def slim_meta(meta: dict) -> dict:
    """公開入庫前精簡：只留分類與權重需要的欄位、描述截斷。"""
    out = {}
    for t, m in (meta or {}).items():
        if not isinstance(m, dict):
            continue
        out[t] = {"name": m.get("name"), "shares": m.get("shares"), "yf_sector": m.get("yf_sector", ""),
                  "yf_industry": m.get("yf_industry", ""), "summary": (m.get("summary") or "")[:SUMMARY_CHARS]}
    return out


def main(argv: list[str]) -> int:
    if "--offline" in argv:
        return _selftest()
    import gics_data as gd
    import gics_verify as gv
    import universe as un
    from gics_model import FeatureBuilder
    GICS_DIR.mkdir(parents=True, exist_ok=True)
    touched: list[Path] = []
    # 1) 成分對照（S&P 1500）
    uni = gd.load_universe("sp1500")
    # 2) 要分類的對象：Alpha 選股池 broad ∪ watchlist（明文）
    try:
        wl = [x for x in (json.loads((ROOT / "watchlist_state.json").read_text(encoding="utf-8")).get("watchlist") or [])
              if isinstance(x, str)]
    except Exception:
        wl = []
    snap = un.load_latest_snapshot() or {}
    broad = [r.get("t") for r in (snap.get("broad") or []) if isinstance(r, dict) and r.get("t")]
    unmatched = uni[uni["code8"].isna()]["ticker"].tolist()          # 成分表標籤對不上的也要分類（#64）
    targets = [gd.yahoo_symbol(t) for t in dict.fromkeys(wl + broad + unmatched) if t]
    sp500 = uni[uni["index"] == "sp500"]["ticker"].tolist()
    need = list(dict.fromkeys(sp500 + [t for t in uni["ticker"] if t in targets] + targets))
    mfile = GICS_DIR / "meta_cache.json"
    # 單輪抓取 + 時間預算：Yahoo 限流時不讓工作流超時（被殺會丟掉整週進度，#64）；沒抓到的下週再補
    meta = gd.fetch_meta(need, mfile, passes=1, budget_s=float(os.environ.get("GICS_META_BUDGET_S", 3600)))
    mfile.write_text(json.dumps(slim_meta(meta), separators=(",", ":")), encoding="utf-8")
    touched.append(mfile)
    _write_touched(touched)                                            # 先記一次：後面步驟失敗也能入庫快取
    # 3) 訓練樣本 = S&P 500 ∪（S&P 400/600 ∩ 選股池）；價格只抓需要的
    train_pool = uni[uni["ticker"].isin(set(sp500) | set(targets))]
    px = gd.download_prices(list(dict.fromkeys(train_pool["ticker"].tolist() + targets)) + FeatureBuilder.SECTOR_ETFS, period="1y")
    summ = run(uni[["ticker", "code8"]], meta, px, targets, GICS_DIR)
    print(f"[gics] members {summ['n_members']}, train {summ['n_train']}, adopted {summ['adopted']}, "
          f"classified {summ['n_classified']}" + (f" — {summ.get('reason')}" if summ.get("reason") else ""))
    touched += [GICS_DIR / n for n in ("members.json", "classified.json", "cv_report.json") if (GICS_DIR / n).exists()]
    # 4) 對帳
    try:
        gv.run("sp500", None, GICS_DIR, meta=meta)                     # 沿用已抓的公司資料，不再重抓一次
        touched.append(GICS_DIR / "verify.json")
    except Exception as e:
        print(f"[gics] verify failed {type(e).__name__}")
    # 不入庫：價格快取、SPY 原始 xlsx、verify_report.csv（.gitignore 也排除）
    _write_touched(touched)
    return 0


def _write_touched(touched: list[Path]) -> None:
    tf = os.environ.get("GICS_TOUCHED_FILE")
    if tf:
        Path(tf).write_text("\n".join(str(p.relative_to(ROOT)) for p in dict.fromkeys(touched) if p.exists()), encoding="utf-8")


def _selftest() -> int:
    import tempfile
    rng = np.random.default_rng(3)
    subs = tx.SUB_INDUSTRIES
    rows, meta, cols = [], {}, {}
    idx = pd.bdate_range("2025-10-01", periods=260)
    sec_fac = {s: np.cumsum(rng.normal(0, 0.01, 260)) for s in tx.codes_at(1)}
    etf_codes = ["10", "15", "20", "25", "30", "35", "40", "45", "50", "55", "60"]
    from gics_model import FeatureBuilder
    for e, s in zip(FeatureBuilder.SECTOR_ETFS, etf_codes):
        cols[e] = 100 * np.exp(sec_fac[s])
    k = 0
    for c8 in subs:
        for _ in range(3):
            t = f"T{k:04d}"; k += 1
            rows.append({"ticker": t, "code8": c8})
            meta[t] = {"name": t, "shares": 1e8, "summary": " ".join(tx.NAMES[x] for x in (c8[:2], c8[:4], c8[:6], c8)).lower(),
                       "yf_sector": tx.NAMES[c8[:2]], "yf_industry": tx.NAMES[c8[:6]]}
            cols[t] = 50 * np.exp(sec_fac[c8[:2]] + np.cumsum(rng.normal(0, 0.008, 260)))
    members = pd.DataFrame(rows)
    hold = members.sample(20, random_state=1)["ticker"].tolist()                  # 假裝這 20 檔不在成分表
    mem = members[~members["ticker"].isin(hold)]
    px = pd.DataFrame(cols, index=idx)
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        s = run(mem, meta, px, hold + ["NOPE"], out, epochs=60, folds=3)
        assert s["adopted"] and s["n_classified"] == 20, s
        tab = lookup_table(out)
        acc = np.mean([tab[t]["code8"][:2] == members.set_index("ticker").loc[t, "code8"][:2] for t in hold])
        assert acc >= 0.8, acc
        assert tab[mem["ticker"].iloc[0]]["src"] == "table" and tab[hold[0]]["src"] == "nn"
        assert "L4" in lookup_text(mem["ticker"].iloc[0], tab) and "不在" in lookup_text("ZZZZ", tab)
        assert "_" not in lookup_text("A_B*", tab) and "*" not in lookup_text("A_B*", tab).split("：")[0]
        # 門檻未達 → 不寫 classified
        s2 = run(mem.assign(code8=rng.permutation(mem["code8"].values)), meta, px, hold, out / "bad", epochs=20, folds=3)
        assert not s2["adopted"] and not (out / "bad" / "classified.json").exists()
        # verify 文字（無檔、有檔）
        assert "尚未" in verify_text(out)
        (out / "verify.json").write_text(json.dumps({"as_of": "2026-10-02", "passed": False, "n_warn": 1,
                                                     "checks": [{"name": "prices present", "status": "PASS", "value": "503/503"}],
                                                     "returns": [{"group": "Energy", "status": "WARN"}, {"group": "IT", "status": "PASS"}]}))
        vt = verify_text(out); assert "1 項警示" in vt and "Energy" in vt and "**" not in vt
        # 曝險文字
        et = exposure_text({"T0000": {"market_value": 600}, "T0003": {"market_value": 400}}, tab)
        assert "持倉產業曝險" in et and "**" not in et
        assert exposure_text({}, tab).startswith("🧮 目前沒有持倉")
    sm = slim_meta({"A": {"name": "A", "summary": "x" * 2000, "shares": 1, "market_cap": 5}})
    assert len(sm["A"]["summary"]) == SUMMARY_CHARS and "market_cap" not in sm["A"]
    print(f"gics_weekly selftest OK ✅（held-out L1 {acc:.2f}）")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
