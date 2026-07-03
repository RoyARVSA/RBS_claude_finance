"""
options_sentiment.py – 選擇權情緒分析（Put/Call 比、隱含波動、偏斜）

純邏輯層（summarize_chains / sentiment / format_*）可離線測試；
抓取層 fetch_options 走 yfinance 選擇權鏈（此開發環境代理擋外網，需部署後實測）。

情緒分數 -1~+1：+1 偏多樂觀、-1 偏空避險。以「當前定價的情緒/避險需求」為主，
非買賣建議；PCR 與偏斜屬市場定位訊號，解讀需搭配趨勢與基本面。
"""

from __future__ import annotations

import numpy as np


def _clamp(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))


def _col_sum(df, col):
    """安全加總 DataFrame 某欄（缺欄/NaN 皆當 0）。"""
    try:
        if df is None or col not in df.columns:
            return 0.0
        s = df[col].fillna(0)
        return float(s.sum())
    except Exception:
        return 0.0


def _atm_iv(df, spot, band=0.10):
    """
    ATM 附近的 OI 加權隱含波動（純函數）。
    取行權價在 spot±band 內、IV 落在 (0,3) 的合約，用未平倉量加權平均。
    無有效資料回 None。
    """
    try:
        if df is None or spot is None or spot <= 0 or "strike" not in df.columns \
                or "impliedVolatility" not in df.columns:
            return None
        d = df[["strike", "impliedVolatility"]].copy()
        d["oi"] = df["openInterest"].fillna(0) if "openInterest" in df.columns else 1.0
        d = d.dropna(subset=["strike", "impliedVolatility"])
        d = d[(d["impliedVolatility"] > 0) & (d["impliedVolatility"] < 3)]
        d = d[(d["strike"] >= spot * (1 - band)) & (d["strike"] <= spot * (1 + band))]
        if d.empty:
            return None
        w = d["oi"].to_numpy(dtype=float)
        iv = d["impliedVolatility"].to_numpy(dtype=float)
        if w.sum() <= 0:
            return float(np.mean(iv))          # 無 OI 時退回等權
        return float(np.average(iv, weights=w))
    except Exception:
        return None


def summarize_chains(chains: list, spot: float,
                     expiries: list | None = None) -> dict:
    """
    彙總多個到期日的選擇權鏈（純函數）。
      chains:   [(calls_df, puts_df), ...]（近月在前）
      spot:     標的現價
    PCR（量/未平倉）跨所有傳入到期日彙總；IV/偏斜取『最近月』(chains[0])。
    """
    call_vol = put_vol = call_oi = put_oi = 0.0
    for calls, puts in chains:
        call_vol += _col_sum(calls, "volume")
        put_vol  += _col_sum(puts, "volume")
        call_oi  += _col_sum(calls, "openInterest")
        put_oi   += _col_sum(puts, "openInterest")

    pcr_vol = (put_vol / call_vol) if call_vol > 0 else None
    pcr_oi  = (put_oi / call_oi) if call_oi > 0 else None

    atm_call_iv = atm_put_iv = atm_iv = iv_skew = None
    if chains:
        near_calls, near_puts = chains[0]
        atm_call_iv = _atm_iv(near_calls, spot)
        atm_put_iv  = _atm_iv(near_puts, spot)
        ivs = [v for v in (atm_call_iv, atm_put_iv) if v is not None]
        atm_iv = float(np.mean(ivs)) if ivs else None
        if atm_call_iv is not None and atm_put_iv is not None:
            iv_skew = atm_put_iv - atm_call_iv     # 正=賣權較貴（下檔保護需求高）

    return {
        "spot": spot,
        "expiries": list(expiries or []),
        "n_expiries": len(chains),
        "call_vol": call_vol, "put_vol": put_vol,
        "call_oi": call_oi, "put_oi": put_oi,
        "pcr_vol": pcr_vol, "pcr_oi": pcr_oi,
        "atm_iv": atm_iv, "atm_call_iv": atm_call_iv, "atm_put_iv": atm_put_iv,
        "iv_skew": iv_skew,
    }


def sentiment(summary: dict) -> dict:
    """
    由 PCR(OI) 與隱含波動偏斜綜合情緒分數（純函數）。
      score -1~+1：+1 偏多樂觀、-1 偏空避險。
    慣例：PCR(OI) 越高＝賣權/避險部位越重（偏空）；正偏斜＝下檔保護需求高（偏空）。
    資料不足時 score 可能為 None。
    """
    pcr = summary.get("pcr_oi")
    skew = summary.get("iv_skew")
    parts, notes = [], []

    if pcr is not None:
        # PCR(OI) 0.5→+1（樂觀）, 1.0→0（中性）, 1.5→-1（避險）
        s_pcr = _clamp((1.0 - pcr) / 0.5)
        parts.append(("pcr", 0.6, s_pcr))
        tag = "偏空避險" if pcr > 1.15 else ("偏多樂觀" if pcr < 0.75 else "中性")
        notes.append(f"未平倉 Put/Call 比 {pcr:.2f}（{tag}）")

    if skew is not None:
        # 偏斜 +5%→-1（下檔避險濃）, -5%→+1（追買買權）
        s_skew = _clamp(-skew / 0.05)
        parts.append(("skew", 0.4, s_skew))
        if skew > 0.01:
            notes.append(f"賣權隱含波動高於買權 {skew*100:.1f} 個百分點（下檔保護需求高）")
        elif skew < -0.01:
            notes.append(f"買權隱含波動高於賣權 {-skew*100:.1f} 個百分點（偏多投機）")
        else:
            notes.append("買賣權隱含波動相近（偏斜中性）")

    if not parts:
        return {"score": None, "label": "資料不足", "notes": ["選擇權資料不足，無法評估情緒"]}

    wsum = sum(w for _, w, _ in parts)
    score = round(sum(w * s for _, w, s in parts) / wsum, 2)
    label = "偏多樂觀" if score >= 0.3 else ("偏空避險" if score <= -0.3 else "中性")
    return {"score": score, "label": label, "notes": notes}


def format_options_text(summary: dict, sent: dict | None = None) -> str:
    """組成給 AI 助理/Bot 的精簡文字（純函數）。"""
    sent = sent or sentiment(summary)
    exps = "、".join(summary.get("expiries", [])[:3]) or "近月"

    def _pcr(v):
        return f"{v:.2f}" if v is not None else "無資料"

    def _iv(v):
        return f"{v*100:.1f}%" if v is not None else "無資料"

    lines = [f"選擇權情緒（到期日 {exps}）："]
    lines.append(
        f"  Put/Call 比：量 {_pcr(summary.get('pcr_vol'))}　"
        f"未平倉 {_pcr(summary.get('pcr_oi'))}")
    lines.append(
        f"  ATM 隱含波動：{_iv(summary.get('atm_iv'))}"
        f"（買權 {_iv(summary.get('atm_call_iv'))} / 賣權 {_iv(summary.get('atm_put_iv'))}）")
    sk = summary.get("iv_skew")
    lines.append(f"  隱含波動偏斜（賣-買）：{sk*100:+.1f}pp" if sk is not None
                 else "  隱含波動偏斜：無資料")
    sc = sent.get("score")
    lines.append(f"  情緒分數：{sc:+.2f}（{sent.get('label')}）" if sc is not None
                 else f"  情緒：{sent.get('label')}")
    for n in sent.get("notes", []):
        lines.append(f"  · {n}")
    return "\n".join(lines)


# ── 抓取層（需網路；此環境代理擋 yfinance，部署後實測）────────────────────────

def fetch_options(ticker: str, max_expiries: int = 3, within_days: int = 45) -> dict | None:
    """
    抓 yfinance 選擇權鏈：取 within_days 內最多 max_expiries 個到期日。
    無選擇權（非選擇權標的/ETF/外股）回 None。
    """
    import pandas as pd
    import yfinance as yf

    tk = yf.Ticker(ticker)
    try:
        exps = list(tk.options or [])
    except Exception:
        return None
    if not exps:
        return None

    # 現價：fast_info 優先，退回歷史收盤
    spot = None
    try:
        spot = float(tk.fast_info["last_price"])
    except Exception:
        try:
            h = tk.history(period="5d")
            if not h.empty:
                spot = float(h["Close"].dropna().iloc[-1])
        except Exception:
            spot = None

    today = pd.Timestamp.now().normalize()
    picked = []
    for e in exps:
        try:
            dte = (pd.Timestamp(e) - today).days
        except Exception:
            continue
        if 0 <= dte <= within_days:
            picked.append(e)
        if len(picked) >= max_expiries:
            break
    if not picked:                       # 都超過天數 → 至少取最近一個
        picked = exps[:1]

    chains = []
    for e in picked:
        try:
            oc = tk.option_chain(e)
            chains.append((oc.calls, oc.puts))
        except Exception:
            continue
    if not chains:
        return None

    summary = summarize_chains(chains, spot, picked)
    summary["ticker"] = ticker
    return summary


# ── CLI 自我測試（純邏輯）─────────────────────────────────────────────────────

if __name__ == "__main__":
    import pandas as pd

    spot = 100.0
    # 偏空情境：賣權量/OI 高、賣權 IV 高（正偏斜）
    calls = pd.DataFrame({
        "strike": [90, 95, 100, 105, 110],
        "volume": [50, 80, 120, 40, 20],
        "openInterest": [200, 300, 500, 150, 80],
        "impliedVolatility": [0.28, 0.26, 0.25, 0.27, 0.30],
    })
    puts = pd.DataFrame({
        "strike": [90, 95, 100, 105, 110],
        "volume": [120, 200, 260, 90, 40],
        "openInterest": [600, 800, 1000, 300, 120],
        "impliedVolatility": [0.40, 0.36, 0.33, 0.31, 0.30],
    })
    summ = summarize_chains([(calls, puts)], spot, ["2026-07-17"])
    print("PCR vol =", round(summ["pcr_vol"], 3), " PCR oi =", round(summ["pcr_oi"], 3))
    print("ATM call IV =", round(summ["atm_call_iv"], 4),
          " put IV =", round(summ["atm_put_iv"], 4),
          " skew =", round(summ["iv_skew"], 4))
    sent = sentiment(summ)
    print("sentiment:", sent["score"], sent["label"])
    print()
    print(format_options_text(summ, sent))

    # 斷言
    assert summ["pcr_oi"] > 1.3                     # 賣權 OI 遠多於買權
    assert summ["iv_skew"] > 0                        # 正偏斜（賣權較貴）
    assert sent["score"] < 0 and sent["label"] == "偏空避險"

    # 偏多情境：買權主導、負偏斜
    c2 = calls.assign(volume=[200, 300, 400, 250, 150],
                      openInterest=[900, 1200, 1500, 800, 400],
                      impliedVolatility=[0.34, 0.33, 0.32, 0.31, 0.30])
    p2 = puts.assign(volume=[40, 60, 80, 30, 20],
                     openInterest=[150, 200, 300, 100, 50],
                     impliedVolatility=[0.29, 0.28, 0.27, 0.28, 0.30])
    s2 = summarize_chains([(c2, p2)], spot, ["2026-07-17"])
    sent2 = sentiment(s2)
    print("\nbull case:", round(s2["pcr_oi"], 2), round(s2["iv_skew"], 3),
          "→", sent2["score"], sent2["label"])
    assert s2["pcr_oi"] < 0.6 and s2["iv_skew"] < 0 and sent2["score"] > 0

    # 空資料安全
    empty = summarize_chains([], spot, [])
    assert empty["pcr_oi"] is None and sentiment(empty)["score"] is None

    print("\n✅ options_sentiment 純邏輯測試通過")
