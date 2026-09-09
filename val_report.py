"""
val_report.py – 估值層治理月報（P6；純邏輯、離線可測）

回答「估值層到底準不準、有沒有在偷懶」：
  • 覆蓋：watchlist 已建模比例、過期（>30 天）、審核未過、品質否決、產業路由（RIM/DDM）數
  • 事後命中：val_hist 每列的 (verdict, px) 對照之後的價格——≥ 20/60 個交易日的列，
    accumulate 之後應漲、exit/trim 之後應弱；給各 verdict 的平均後續報酬與命中率（樣本數如實）
  • 公允價值穩定度：同一檔 base 的變異係數（公允價每週大跳＝模型不穩）
  • MoS 因子 IC（factor_eval：val_hist 週頻列當快照、21/63 日前瞻）——有效期數不足時明講「累積中」
  • 指引覆蓋：幾檔有 /guidance、修訂方向分佈
所有數字為研究參考；非投資建議。
"""

from __future__ import annotations

import math
from datetime import datetime


def _f(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _days(a, b):
    try:
        return (datetime.strptime(str(b)[:10], "%Y-%m-%d") - datetime.strptime(str(a)[:10], "%Y-%m-%d")).days
    except Exception:
        return None


def coverage(state: dict, today: str, stale_days: int = 30) -> dict:
    wl = list(state.get("watchlist") or [])
    vh = state.get("val_hist") or {}
    modeled = [t for t in wl if vh.get(t)]
    stale = [t for t in modeled if (_days(vh[t][-1].get("d"), today) or 999) > stale_days]
    review = [t for t in modeled if vh[t][-1].get("verdict") == "review"]
    routed = [t for t in modeled if vh[t][-1].get("method") in ("rim", "ddm")]
    skip = state.get("model_skip") or {}
    verdicts = {}
    for t in modeled:
        v = vh[t][-1].get("verdict") or "—"
        verdicts[v] = verdicts.get(v, 0) + 1
    return {"n_watch": len(wl), "n_modeled": len(modeled), "coverage": (len(modeled) / len(wl)) if wl else 0.0,
            "stale": stale, "review": review, "routed": routed, "skipped": sorted(k for k in skip if k in wl),
            "verdicts": verdicts}


def hindsight(state: dict, price_now: dict, today: str, min_days: int = 20) -> dict:
    """val_hist 列（≥ min_days 天前、有 px）→ 之後到今天的報酬，依 verdict 分組。"""
    vh = state.get("val_hist") or {}
    groups: dict[str, list[float]] = {}
    for t, rows in vh.items():
        pn = _f(price_now.get(t))
        if not pn:
            continue
        for r in rows or []:
            px, v = _f(r.get("px")), r.get("verdict")
            age = _days(r.get("d"), today)
            if not px or not v or age is None or age < min_days:
                continue
            groups.setdefault(v, []).append(pn / px - 1)
    out = {}
    for v, rets in groups.items():
        n = len(rets)
        mean = sum(rets) / n
        if v in ("accumulate", "hold"):
            hit = sum(1 for x in rets if x > 0) / n
        elif v in ("trim", "exit"):
            hit = sum(1 for x in rets if x <= 0) / n
        else:
            hit = None
        out[v] = {"n": n, "mean_ret": mean, "hit": hit, "min": min(rets), "max": max(rets)}
    return out


def stability(state: dict, n_last: int = 8) -> dict:
    """同一檔最近 n 列 base 的變異係數（>15% 視為不穩）。"""
    vh = state.get("val_hist") or {}
    out = {}
    for t, rows in vh.items():
        bases = [_f(r.get("base")) for r in (rows or [])[-n_last:]]
        bases = [b for b in bases if b]
        if len(bases) >= 3:
            m = sum(bases) / len(bases)
            sd = (sum((b - m) ** 2 for b in bases) / (len(bases) - 1)) ** 0.5
            out[t] = {"n": len(bases), "cv": (sd / m) if m else None}
    unstable = sorted(t for t, d in out.items() if d["cv"] is not None and d["cv"] > 0.15)
    return {"per_ticker": out, "unstable": unstable}


def mos_factor(state: dict, closes: dict | None) -> dict | None:
    """val_hist → {date: {ticker: mos}} 快照 → factor_eval.evaluate（需 closes）。"""
    vh = state.get("val_hist") or {}
    fac: dict[str, dict] = {}
    for t, rows in vh.items():
        for r in rows or []:
            d, mos = r.get("d"), _f(r.get("mos"))
            if d and mos is not None:
                fac.setdefault(str(d)[:10], {})[t] = mos
    if not closes or len(fac) < 4:
        return {"n_dates": len(fac), "note": "累積中（需 ≥4 期且有行情）"}
    try:
        import factor_eval as fe
        ev = fe.evaluate(fac, closes)
        ok, why = fe.passes_gate(ev)
        return {"n_dates": len(fac), "eval": ev, "gate": ok, "why": why}
    except Exception as e:
        return {"n_dates": len(fac), "note": f"評估失敗 {type(e).__name__}"}


def guidance_summary(state: dict) -> dict:
    g = state.get("guidance") or {}
    rev = {}
    for t, d in g.items():
        for it in (d or {}).get("items") or []:
            if it.get("kind") == "guidance":
                rev[it.get("revision") or "—"] = rev.get(it.get("revision") or "—", 0) + 1
    return {"n_tickers": len(g), "revisions": rev}


def build_report(state: dict, today: str, price_now: dict | None = None, closes: dict | None = None) -> dict:
    return {"as_of": today, "coverage": coverage(state, today), "hindsight": hindsight(state, price_now or {}, today),
            "stability": stability(state), "mos_factor": mos_factor(state, closes),
            "guidance": guidance_summary(state)}


def report_text(rep: dict) -> str:
    c, h, s, m, g = rep["coverage"], rep["hindsight"], rep["stability"], rep["mos_factor"], rep["guidance"]
    lines = [f"📋 *估值層治理月報*（{rep['as_of']}）",
             f"覆蓋 {c['n_modeled']}/{c['n_watch']}（{c['coverage']:.0%}）｜過期 {len(c['stale'])}｜待審 {len(c['review'])}｜"
             f"RIM/DDM {len(c['routed'])}｜建不了模 {len(c['skipped'])}"]
    if c.get("verdicts"):
        lines.append("判定分佈：" + "、".join(f"{k} {v}" for k, v in sorted(c["verdicts"].items())))
    if c.get("stale"):
        lines.append("過期：" + " ".join(c["stale"][:10]))
    if c.get("review"):
        lines.append("待審（審核未過/品質否決）：" + " ".join(c["review"][:10]))
    if h:
        lines.append("*事後命中*（列日 → 今日報酬；≥20 天）：")
        for v in ("accumulate", "hold", "trim", "exit", "review"):
            if v in h:
                d = h[v]
                hit = f"命中 {d['hit']:.0%}" if d.get("hit") is not None else "—"
                lines.append(f"・{v}：n={d['n']}，均 {d['mean_ret']:+.1%}（{d['min']:+.0%}～{d['max']:+.0%}），{hit}")
    else:
        lines.append("事後命中：樣本累積中（列滿 20 天後開始統計）")
    if s.get("unstable"):
        lines.append("⚠️ 公允價值不穩（近 8 列變異係數 >15%）：" + " ".join(s["unstable"][:8]))
    if m:
        if m.get("eval"):
            ic = (m["eval"].get("horizons") or {}).get(21, {}).get("ic") or {}
            lines.append(f"MoS 因子 21 日 IC {ic.get('mean', 0):+.3f}（有效期數 {ic.get('n_eff', 0):.0f}，NW t {ic.get('t_nw') if ic.get('t_nw') is None else round(ic['t_nw'], 1)}）"
                         + ("｜✅ 過配置門檻" if m.get("gate") else f"｜➖ 未過（{m.get('why')}）"))
        else:
            lines.append(f"MoS 因子 IC：{m.get('note')}（快照 {m.get('n_dates', 0)} 期）")
    lines.append(f"指引萃取：{g['n_tickers']} 檔" + ("；修訂 " + "、".join(f"{k} {v}" for k, v in g["revisions"].items()) if g["revisions"] else ""))
    lines.append("估值層仍為顯示層（val·enabled 關）之前，本報只評估「準不準」；非投資建議")
    return "\n".join(lines)


if __name__ == "__main__":
    today = "2026-09-09"
    state = {"watchlist": ["AAA", "BBB", "CCC", "SPY"], "model_skip": {"SPY": "2026-09-01"},
             "val_hist": {"AAA": [{"d": "2026-07-01", "px": 100, "base": 130, "mos": 0.3, "verdict": "accumulate"},
                                  {"d": "2026-07-08", "px": 105, "base": 131, "mos": 0.25, "verdict": "accumulate"},
                                  {"d": "2026-07-15", "px": 110, "base": 200, "mos": 0.8, "verdict": "accumulate"},
                                  {"d": "2026-09-08", "px": 120, "base": 135, "mos": 0.12, "verdict": "hold"}],
                          "BBB": [{"d": "2026-06-01", "px": 50, "base": 30, "mos": -0.4, "verdict": "exit"},
                                  {"d": "2026-09-01", "px": 40, "base": 30, "mos": -0.25, "verdict": "trim"}],
                          "CCC": [{"d": "2026-07-01", "px": 20, "base": 22, "mos": 0.1, "verdict": "review", "method": "rim"}]},
             "guidance": {"AAA": {"items": [{"kind": "guidance", "revision": "raise"}, {"kind": "kpi"}]}}}
    px_now = {"AAA": 125, "BBB": 38, "CCC": 21}
    rep = build_report(state, today, px_now, None)
    c = rep["coverage"]
    assert c["n_modeled"] == 3 and c["stale"] == ["CCC"] and c["review"] == ["CCC"] and c["routed"] == ["CCC"] and c["skipped"] == ["SPY"]
    h = rep["hindsight"]
    assert h["accumulate"]["n"] == 3 and h["accumulate"]["hit"] == 1.0 and h["exit"]["hit"] == 1.0 and "hold" not in h   # hold 列 <20 天
    assert rep["stability"]["unstable"] == ["AAA"]                          # 130/131/200/135 → CV >15%
    assert rep["mos_factor"]["note"].startswith("累積中") and rep["guidance"]["revisions"] == {"raise": 1}
    t = report_text(rep)
    assert t.count("*") % 2 == 0 and "_" not in t
    print(t)
    print("\nval_report selftest OK ✅")
