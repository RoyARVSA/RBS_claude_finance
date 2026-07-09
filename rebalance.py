"""
rebalance.py — 持倉再平衡顧問
RBS Finance Dashboard

現有持倉 vs 目標權重（HRP／最大 Sharpe／最小波動／等風險貢獻／等權重）
→ 具體加減碼清單：每檔買賣股數、金額、漂移度，含最小交易門檻（防瑣碎單）
與賣出上限（不可賣超過持有）。教育模擬用途，非投資建議。

慣例：純邏輯（current_weights / rebalance_orders / rebalance_text）離線可測；
target_weights 依 scheme 分派到 portfolio_opt / quant_tools（也是純邏輯，吃報酬矩陣）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SCHEMES = {
    "hrp":        "HRP 階層風險平價",
    "max_sharpe": "最大 Sharpe（MPT）",
    "min_vol":    "最小波動（MPT）",
    "erc":        "等風險貢獻（風險平價）",
    "equal":      "等權重",
}

DEFAULT_MIN_TRADE_PCT = 0.01   # 單筆交易 < 總值 1% 就略過（不值得付出摩擦成本）


# ── 純邏輯 ────────────────────────────────────────────────────────────────────

def current_weights(qty: dict[str, float],
                    prices: dict[str, float]) -> tuple[dict[str, float], float, list[str]]:
    """持股股數 + 現價 → (權重, 總市值, 略過的代碼)。
    只收多頭（q>0）：空頭部位會讓權重/總值全面失真，一律列入 skipped。"""
    mv, skipped = {}, []
    for tk, q in qty.items():
        px = prices.get(tk)
        if q and q > 0 and px and px > 0:
            mv[tk] = float(q) * float(px)
        elif q:
            skipped.append(tk)
    total = sum(mv.values())
    w = {tk: v / total for tk, v in mv.items()} if total > 0 else {}
    return w, total, skipped


def target_weights(returns_df: pd.DataFrame, scheme: str) -> pd.Series | None:
    """報酬矩陣 → 目標權重（Series，總和 1）。失敗回 None。"""
    if returns_df is None or returns_df.shape[1] < 2 or len(returns_df) < 40:
        return None
    cols = list(returns_df.columns)
    try:
        if scheme == "hrp":
            from portfolio_opt import hrp_weights
            w = hrp_weights(returns_df)
        elif scheme == "max_sharpe":
            from portfolio_opt import max_sharpe_weights
            w = max_sharpe_weights(returns_df)
        elif scheme == "min_vol":
            from portfolio_opt import min_vol_weights
            w = min_vol_weights(returns_df)
        elif scheme == "erc":
            from quant_tools import risk_parity_weights
            arr = risk_parity_weights(returns_df.cov().to_numpy())
            w = pd.Series(arr, index=cols)
        elif scheme == "equal":
            w = pd.Series(1.0 / len(cols), index=cols)
        else:
            return None
    except Exception:
        return None
    if w is None or w.isna().any() or w.sum() <= 0:
        return None
    w = w.clip(lower=0.0)
    return w / w.sum()


def rebalance_orders(qty: dict[str, float], prices: dict[str, float],
                     tgt_w: dict[str, float],
                     min_trade_pct: float = DEFAULT_MIN_TRADE_PCT,
                     no_data: tuple | list = ()) -> dict:
    """
    產生再平衡訂單清單。
    no_data：歷史數據不足、沒進優化的代碼——**不下單**（避免被誤判成「目標歸零」
    而清倉），列入 skipped_no_history 讓使用者知道。
    回 {"orders": [...], "turnover": 單邊換手率, "total_value": 總市值,
        "skipped_small": [...], "skipped_no_price": [...], "skipped_no_history": [...]}
    orders 元素：{ticker, action 買進/賣出, shares, value, price,
                  cur_w, tgt_w, drift}；賣出以持有股數為上限、買進整股。
    """
    cur_w, total, no_px = current_weights(qty, prices)
    out = {"orders": [], "turnover": 0.0, "total_value": total,
           "skipped_small": [], "skipped_no_price": no_px,
           "skipped_no_history": sorted(set(no_data))}
    if total <= 0:
        return out

    tickers = sorted((set(cur_w) | set(tgt_w)) - set(no_data))
    gross = 0.0
    for tk in tickers:
        px = prices.get(tk)
        if not px or not (px > 0):        # not (px>0) 同時擋 NaN（NaN 比較恆 False）
            if tgt_w.get(tk, 0) > 0:
                out["skipped_no_price"].append(tk)
            continue
        cw, tw = cur_w.get(tk, 0.0), float(tgt_w.get(tk, 0.0))
        drift = tw - cw
        trade_val = drift * total
        if abs(trade_val) < min_trade_pct * total:
            if abs(drift) > 1e-9:
                out["skipped_small"].append(tk)
            continue
        if trade_val > 0:
            shares = int(trade_val / px)              # 買進：無條件捨去整股
            action = "買進"
        else:
            held = float(qty.get(tk, 0.0))
            shares = min(int(abs(trade_val) / px + 0.5), int(held))   # 不可賣超
            action = "賣出"
        if shares <= 0:
            out["skipped_small"].append(tk)
            continue
        val = shares * px
        gross += val
        out["orders"].append({"ticker": tk, "action": action, "shares": shares,
                              "value": round(val, 2), "price": float(px),
                              "cur_w": round(cw, 4), "tgt_w": round(tw, 4),
                              "drift": round(drift, 4)})
    out["turnover"] = round(gross / (2 * total), 4) if total > 0 else 0.0
    out["skipped_no_price"] = sorted(set(out["skipped_no_price"]))
    # 賣單在前（先騰出資金）、再依金額大小
    out["orders"].sort(key=lambda o: (o["action"] != "賣出", -o["value"]))
    return out


def rebalance_text(res: dict, scheme_label: str) -> str:
    """訂單清單 → 文字版（下載 / Bot 共用）。"""
    lines = [f"⚖️ 再平衡建議（目標：{scheme_label}）",
             f"組合總市值 ${res['total_value']:,.0f}｜單邊換手率 {res['turnover']:.1%}"]
    if not res["orders"]:
        lines.append("\n漂移皆在門檻內——目前無需再平衡。")
    for o in res["orders"]:
        lines.append(f"{'🔴' if o['action'] == '賣出' else '🟢'} {o['action']} "
                     f"{o['ticker']} {o['shares']} 股 ≈ ${o['value']:,.0f}"
                     f"（{o['cur_w']:.1%} → {o['tgt_w']:.1%}，漂移 {o['drift']:+.1%}）")
    if res["skipped_small"]:
        lines.append(f"（{ '、'.join(res['skipped_small']) }：漂移小於門檻，略過）")
    if res["skipped_no_price"]:
        lines.append(f"（{ '、'.join(res['skipped_no_price']) }：抓不到現價或非多頭部位，略過）")
    if res.get("skipped_no_history"):
        lines.append(f"（{ '、'.join(res['skipped_no_history']) }：歷史數據不足未進優化，"
                     "持倉維持不動）")
    lines.append("\n⚠️ 模擬教育用途，非投資建議；未含手續費/稅與滑價。")
    return "\n".join(lines)


# ── CLI 自我測試（離線純邏輯）─────────────────────────────────────────────────

if __name__ == "__main__":
    # 三檔持倉：A 超配、B 低配、C 未持有（目標要買）、D 持有但目標歸零
    qty = {"AAA": 100.0, "BBB": 10.0, "DDD": 50.0}
    px = {"AAA": 100.0, "BBB": 100.0, "CCC": 50.0, "DDD": 20.0}
    # 市值：A 10000、B 1000、D 1000 → 總 12000
    tgt = {"AAA": 0.40, "BBB": 0.30, "CCC": 0.30}

    w, total, sk = current_weights(qty, px)
    assert abs(total - 12000) < 1e-9 and not sk
    assert abs(w["AAA"] - 10000 / 12000) < 1e-9

    res = rebalance_orders(qty, px, tgt, min_trade_pct=0.01)
    od = {o["ticker"]: o for o in res["orders"]}
    # A：0.833→0.40 → 賣 ~5200/100 = 52 股
    assert od["AAA"]["action"] == "賣出" and abs(od["AAA"]["shares"] - 52) <= 1, od["AAA"]
    # B：0.083→0.30 → 買 2600/100 = 26 股
    assert od["BBB"]["action"] == "買進" and od["BBB"]["shares"] == 26, od["BBB"]
    # C：新倉 3600/50 = 72 股
    assert od["CCC"]["action"] == "買進" and od["CCC"]["shares"] == 72, od["CCC"]
    # D：目標 0 → 全賣（上限 = 持有 50 股）
    assert od["DDD"]["action"] == "賣出" and od["DDD"]["shares"] == 50, od["DDD"]
    # 賣單排在買單前
    assert res["orders"][0]["action"] == "賣出"
    assert 0 < res["turnover"] < 1

    # 最小門檻：漂移 0.5% < 1% → 略過
    res2 = rebalance_orders({"AAA": 100.0}, {"AAA": 100.0, "BBB": 100.0},
                            {"AAA": 0.995, "BBB": 0.005}, min_trade_pct=0.01)
    assert not res2["orders"] and "BBB" in res2["skipped_small"], res2

    # 缺價：目標含無價代碼 → skipped_no_price（且不重複列示）
    res3 = rebalance_orders({"AAA": 10.0, "ZZZ": 5.0}, {"AAA": 100.0},
                            {"AAA": 0.5, "ZZZ": 0.5})
    assert res3["skipped_no_price"] == ["ZZZ"], res3

    # 空頭部位：負股數不得進權重/訂單（否則總值淨額化、金額全面放大）
    w4, tot4, sk4 = current_weights({"LONG": 100.0, "SHORT": -50.0},
                                    {"LONG": 100.0, "SHORT": 100.0})
    assert "SHORT" in sk4 and abs(tot4 - 10000) < 1e-9 and w4 == {"LONG": 1.0}, (w4, tot4)
    res4 = rebalance_orders({"LONG": 100.0, "SHORT": -50.0},
                            {"LONG": 100.0, "SHORT": 100.0}, {"LONG": 1.0})
    assert all(o["ticker"] != "SHORT" for o in res4["orders"]), res4

    # 歷史不足（no_data）：不下清倉單、列入 skipped_no_history
    res5 = rebalance_orders({"AAA": 50.0, "IPO": 100.0},
                            {"AAA": 100.0, "IPO": 10.0},
                            {"AAA": 1.0}, no_data=["IPO"])
    assert all(o["ticker"] != "IPO" for o in res5["orders"]), res5
    assert res5["skipped_no_history"] == ["IPO"], res5
    assert "持倉維持不動" in rebalance_text(res5, "等權重")

    # target_weights：合成報酬 → 每個 scheme 權重和為 1
    rng = np.random.default_rng(5)
    rets = pd.DataFrame(rng.normal(0.0005, 0.01, (260, 4)),
                        columns=["W", "X", "Y", "Z"])
    for sch in SCHEMES:
        tw = target_weights(rets, sch)
        assert tw is not None and abs(tw.sum() - 1) < 1e-6 and (tw >= -1e-9).all(), \
            (sch, tw)
    assert target_weights(rets.iloc[:, :1], "hrp") is None      # <2 檔 → None
    assert target_weights(rets, "nonsense") is None

    txt = rebalance_text(res, SCHEMES["hrp"])
    assert "再平衡建議" in txt and "非投資建議" in txt and "賣出 AAA" in txt

    print("✅ rebalance 離線自我測試通過"
          f"（{len(res['orders'])} 單、換手 {res['turnover']:.1%}）")
