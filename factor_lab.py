"""
factor_lab.py – 因子研發迴圈（C 段；規劃見 ALPHA_SPINE.md §3）

任何人（或日後的委員會）提出一條候選因子公式 → 用同一把尺（factor_eval 的 Rank IC / ICIR / NW t）
評估 → DSR 扣「試過很多次總會有一個好看」的幸運上限 → 記進帳本；人工核准的才進 A 段合成（仍過 IC 閘門）。

DSL（白名單 ast，只允許數字、四則、負號與下列原語；全部只用當日以前含當日資料）：
  ret(n)        n 日報酬                     mom(a,b)   a 日前→b 日前報酬（12-1 動能 = mom(252,21)）
  vol(n)        n 日日報酬標準差             ma_dist(n) 價／n 日均線 − 1
  ext(n)        (價 − MA n) / ATR14           rsi(n)     RSI(n)（0–100）
  volratio(n)   量／n 日均量                  hi_dist(n) 價／n 日最高 − 1      lo_dist(n) 價／n 日最低 − 1
方向：公式的值越高 = 越看好；想反向就在公式前加負號（如 -vol(60)）。

帳本 state["factor_lab"] = {"trials": [...], "approved": {name: {expr, added, ic}}}（公式非敏感、明文）。
"""

from __future__ import annotations

import ast

import numpy as np
import pandas as pd

PRIMS = ("ret", "mom", "vol", "ma_dist", "ext", "rsi", "volratio", "hi_dist", "lo_dist")
ARITY = {"ret": (1, 1), "mom": (1, 2), "vol": (1, 1), "ma_dist": (1, 1), "ext": (1, 1), "rsi": (1, 1),
         "volratio": (1, 1), "hi_dist": (1, 1), "lo_dist": (1, 1)}          # (最少, 最多) 參數數
TRIAL_CAP = 100
MAX_EXPR_LEN = 120
DSR_HORIZON = 5              # DSR 用短視窗 IC 序列（有效期數才夠 ≥30）；閘門仍看 21 日


class DSLError(ValueError):
    pass


# ── 1. 解析（白名單）───────────────────────────────────────────────────────────

def parse_expr(expr: str) -> ast.Expression:
    if not isinstance(expr, str) or not expr.strip():
        raise DSLError("空公式")
    if len(expr) > MAX_EXPR_LEN:
        raise DSLError(f"公式過長（>{MAX_EXPR_LEN}）")
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as e:
        raise DSLError(f"語法錯誤：{e.msg}")
    for node in ast.walk(tree):
        if isinstance(node, ast.Expression):
            continue
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            continue
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            continue
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            continue
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in PRIMS) or node.keywords:
                raise DSLError(f"不允許的函數：{getattr(node.func, 'id', '?')}")
            lo_n, hi_n = ARITY[node.func.id]
            if not (lo_n <= len(node.args) <= hi_n):
                raise DSLError(f"{node.func.id}() 需要 {lo_n}–{hi_n} 個參數")
            for a in node.args:
                if not (isinstance(a, ast.Constant) and isinstance(a.value, int) and 1 <= a.value <= 500):
                    raise DSLError(f"{node.func.id}() 參數須為 1–500 的整數")
            continue
        if isinstance(node, ast.Name) and node.id in PRIMS:
            continue
        if isinstance(node, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd, ast.Load)):
            continue
        raise DSLError(f"不允許的語法：{type(node).__name__}")
    return tree


# ── 2. 原語（寬表向量化）───────────────────────────────────────────────────────

def _atr(frames):
    import alpha_spine as asp
    return asp.atr14(frames["close"], frames.get("high"), frames.get("low"))


def _rsi_wide(close: pd.DataFrame, n: int) -> pd.DataFrame:
    d = close.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / n, min_periods=n).mean()
    loss = (-d).clip(lower=0).ewm(alpha=1 / n, min_periods=n).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(loss != 0, 100.0)


def _prim(name: str, args: list[int], frames: dict, cache: dict) -> pd.DataFrame:
    key = (name, tuple(args))
    if key in cache:
        return cache[key]
    c = frames["close"]
    v = frames.get("volume")
    if name == "ret":
        out = c / c.shift(args[0]) - 1
    elif name == "mom":
        a, b = args[0], args[1] if len(args) > 1 else 0
        out = c.shift(b) / c.shift(a) - 1
    elif name == "vol":
        out = c.pct_change().rolling(args[0]).std()
    elif name == "ma_dist":
        out = c / c.rolling(args[0]).mean() - 1
    elif name == "ext":
        out = (c - c.rolling(args[0]).mean()) / _atr(frames).replace(0, np.nan)
    elif name == "rsi":
        out = _rsi_wide(c, args[0])
    elif name == "volratio":
        if v is None or v.empty:
            raise DSLError("無成交量資料")
        out = v / v.rolling(args[0]).mean().replace(0, np.nan)
    elif name == "hi_dist":
        out = c / c.rolling(args[0]).max() - 1
    elif name == "lo_dist":
        out = c / c.rolling(args[0]).min() - 1
    else:
        raise DSLError(f"未知原語 {name}")
    out = out.replace([np.inf, -np.inf], np.nan)
    cache[key] = out
    return out


def _eval(node, frames, cache):
    if isinstance(node, ast.Expression):
        return _eval(node.body, frames, cache)
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.UnaryOp):
        v = _eval(node.operand, frames, cache)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp):
        a, b = _eval(node.left, frames, cache), _eval(node.right, frames, cache)
        if isinstance(node.op, ast.Add):
            return a + b
        if isinstance(node.op, ast.Sub):
            return a - b
        if isinstance(node.op, ast.Mult):
            return a * b
        if isinstance(node.op, ast.Div):
            if isinstance(b, (int, float)):
                if b == 0:
                    raise DSLError("除以零")
                return a / b
            return a / b.replace(0, np.nan)
    if isinstance(node, ast.Call):
        return _prim(node.func.id, [int(a.value) for a in node.args], frames, cache)
    raise DSLError(f"無法求值：{type(node).__name__}")


def evaluate_expr(expr: str, frames: dict) -> pd.DataFrame:
    """公式 → 寬表（index=日期、columns=代碼）；純常數結果視為錯誤。"""
    tree = parse_expr(expr)
    out = _eval(tree, frames, {})
    if not isinstance(out, pd.DataFrame):
        raise DSLError("公式必須含至少一個原語")
    return out.replace([np.inf, -np.inf], np.nan)


def min_lookback(expr: str) -> int:
    """公式需要的最少歷史根數（原語參數最大值 + 緩衝）。"""
    tree = parse_expr(expr)
    mx = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            mx = max([mx] + [int(a.value) for a in node.args])
    return mx + 15


# ── 3. 評估 + DSR 帳本 ─────────────────────────────────────────────────────────

def test_factor(expr: str, frames: dict, dates: list[str], horizon: int = 21, ledger: dict | None = None) -> dict:
    """
    公式 → factor_by_date → factor_eval（21 日閘門）+ 5 日 IC 序列的 DSR（帳本筆數為 N、歷史 ICIR 為 trial 分佈）。
    回 {expr, ic, icir, t_nw, n_eff, spread, pass, reason, dsr, sr_star, n_trials}。不寫帳本（呼叫端決定）。
    """
    import alpha_spine as asp
    import factor_eval as fe
    wide = evaluate_expr(expr, frames)
    fbd = asp.factor_by_date(wide, dates)
    closes = {str(c): frames["close"][c].dropna() for c in frames["close"].columns}
    usable = {d: v for d, v in fbd.items() if len(v) >= min(asp.MIN_COVERAGE, max(6, len(closes) // 2))}
    if len(usable) < 3:
        return {"expr": expr, "ic": None, "pass": False, "reason": "覆蓋不足或公式在此資料上多為缺值", "n_trials": 1 + len((ledger or {}).get("trials", []))}
    ev = fe.evaluate(usable, closes, horizons=(horizon, DSR_HORIZON))
    ok, why = fe.passes_gate(ev, horizon)
    h = ev["horizons"][horizon]["ic"]
    short = ev["horizons"][DSR_HORIZON]["ic"]
    trials = list((ledger or {}).get("trials", []))
    n_trials = len(trials) + 1
    dsr = None
    try:
        import falsifier as fz
        if short.get("icir") is not None and (short.get("n_eff") or 0) >= 30:
            prev = [t["icir5"] for t in trials if t.get("icir5") is not None]
            d = fz.deflated_sharpe(float(short["icir"]), int(short["n_eff"]), n_trials, prev if len(prev) >= 2 else None)
            dsr = d
        else:
            dsr = {"dsr": None, "sr_star": None, "note": "5 日 IC 有效期數 < 30"}
    except ImportError:
        dsr = {"dsr": None, "sr_star": None, "note": "scipy 不可用（Actions 環境未安裝；本地可算）"}
    except Exception as e:
        dsr = {"dsr": None, "sr_star": None, "note": f"DSR 失敗（{type(e).__name__}）"}
    return {"expr": expr, "ic": h.get("mean"), "icir": h.get("icir"), "t_nw": h.get("t_nw"), "n_eff": h.get("n_eff"),
            "hit": h.get("hit"), "spread": ev["horizons"][horizon]["quantiles"].get("spread"),
            "autocorr": ev.get("autocorr"), "pass": bool(ok), "reason": why,
            "icir5": short.get("icir"), "n_eff5": short.get("n_eff"),
            "dsr": (dsr or {}).get("dsr"), "sr_star": (dsr or {}).get("sr_star"), "dsr_note": (dsr or {}).get("note"),
            "n_trials": n_trials, "n_dates": len(usable)}


def ledger_add(state: dict, res: dict, today: str) -> dict:
    lab = state.setdefault("factor_lab", {"trials": [], "approved": {}})
    rec = {"date": today[:10], "expr": res.get("expr"), "ic": res.get("ic"), "icir": res.get("icir"), "t_nw": res.get("t_nw"),
           "icir5": res.get("icir5"), "dsr": res.get("dsr"), "pass": bool(res.get("pass"))}
    lab["trials"] = (lab.get("trials", []) + [rec])[-TRIAL_CAP:]
    return rec


def approve(state: dict, name: str, expr: str, today: str, res: dict | None = None) -> tuple[bool, str]:
    """人工核准：名稱只允許 a-z0-9_（≤20），公式須可解析；覆寫同名。"""
    import re
    name = str(name).strip().lower()
    if not re.fullmatch(r"[a-z][a-z0-9_]{1,19}", name):
        return False, "名稱須為小寫英數底線、2–20 字、字母開頭"
    if name in ("mom_12_1", "rev_1m", "ext_atr", "vol_60", "quality", "rev", "composite"):
        return False, "與內建因子同名"
    try:
        parse_expr(expr)
    except DSLError as e:
        return False, f"公式無效：{e}"
    lab = state.setdefault("factor_lab", {"trials": [], "approved": {}})
    lab.setdefault("approved", {})[name] = {"expr": expr.strip(), "added": today[:10],
                                            "ic": (res or {}).get("ic"), "icir": (res or {}).get("icir")}
    return True, f"已核准 {name}：{expr.strip()}（夜間 A 段納入，仍過 IC 閘門）"


def drop(state: dict, name: str) -> bool:
    lab = state.get("factor_lab") or {}
    return (lab.get("approved") or {}).pop(str(name).strip().lower(), None) is not None


def approved_factors(state: dict) -> dict[str, str]:
    return {k: v["expr"] for k, v in ((state.get("factor_lab") or {}).get("approved") or {}).items() if v.get("expr")}


# ── 4. 文字 ───────────────────────────────────────────────────────────────────

def _f(x, nd=3, pct=False):
    if x is None:
        return "—"
    return f"{x:+.1%}" if pct else f"{x:+.{nd}f}"


def test_text(res: dict) -> str:
    lines = [f"🔬 *因子測試* `{res.get('expr')}`"]
    if res.get("ic") is None:
        lines.append(f"➖ {res.get('reason')}")
        return "\n".join(lines)
    hit_s = "—" if res.get("hit") is None else f"{res['hit']:.0%}"
    lines.append(f"21 日：IC {_f(res['ic'])}｜ICIR {_f(res.get('icir'), 2)}｜NW t {_f(res.get('t_nw'), 1)}｜有效期數 {res.get('n_eff', 0):.0f}｜"
                 f"命中 {hit_s}｜Q高−Q低 {_f(res.get('spread'), pct=True)}")
    lines.append(f"閘門：{'✅ 通過' if res.get('pass') else '➖ ' + str(res.get('reason'))}")
    if res.get("dsr") is not None:
        verdict = "通過" if res["dsr"] > 0.95 else "未達 0.95"
        lines.append(f"DSR {res['dsr']:.2f}（{verdict}；5 日 IC 序列、已扣帳本第 {res.get('n_trials')} 次嘗試的幸運上限 SR*={res.get('sr_star')}）")
    else:
        lines.append(f"DSR：{res.get('dsr_note') or '樣本不足'}（帳本第 {res.get('n_trials')} 次嘗試）")
    if res.get("autocorr") is not None:
        lines.append(f"換手：相鄰快照 rank 相關 {res['autocorr']:+.2f}")
    lines.append("_通過不等於上線：`/factor add 名稱 公式` 核准後由夜間 A 段納入合成、仍逐日過 IC 閘門。非投資建議_")
    return "\n".join(lines)


def list_text(state: dict) -> str:
    lab = state.get("factor_lab") or {}
    ap, tr = lab.get("approved") or {}, lab.get("trials") or []
    lines = ["🧫 *因子實驗室*"]
    lines.append("核准（進 A 段合成）：" + ("、".join(f"`{k}` = `{v['expr']}`" for k, v in ap.items()) if ap else "無"))   # 名稱含底線也放 code span（M4）
    if tr:
        lines.append(f"帳本 {len(tr)} 次嘗試，最近：")
        for t in tr[-5:]:
            dsr_s = "" if t.get("dsr") is None else f" DSR {t['dsr']:.2f}"
            lines.append(f"• {t['date']} `{t['expr']}` IC {_f(t.get('ic'))} ICIR {_f(t.get('icir'), 2)}"
                         f"{' ✅' if t.get('pass') else ''}{dsr_s}")
    lines.append("用法：`/factor test <公式>`｜`/factor add <名稱> <公式>`｜`/factor drop <名稱>`\n"
                 "原語：ret(n) mom(a,b) vol(n) ma_dist(n) ext(n) rsi(n) volratio(n) hi_dist(n) lo_dist(n)")
    return "\n".join(lines)


# ── 自我測試 ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import alpha_spine as asp
    rng = np.random.default_rng(3)
    # 1) DSL 白名單
    for bad in ["__import__('os')", "ret(5) ** 2", "x + 1", "ret(0)", "ret(5, 3, 1)", "open('/etc/passwd')", "ret('a')", "vol(600)", ""]:
        try:
            parse_expr(bad); raise AssertionError(bad)
        except DSLError:
            pass
    for good in ["mom(252,21)", "-vol(60)", "0.5*ret(20) - vol(20)/2", "rsi(14) - 50", "hi_dist(252)", "volratio(20)"]:
        parse_expr(good)
    assert min_lookback("mom(252,21) - vol(60)") == 267
    print("✅ 1 DSL 白名單與注入拒絕")

    # 2) 求值：與手算一致、無前視
    n_sym, n_day = 40, 400
    idx = pd.bdate_range("2025-01-06", periods=n_day)
    data = {}
    for k in range(n_sym):
        c = 100 * np.cumprod(1 + rng.normal(0.0003, 0.02, n_day))
        data[f"F{k:02d}"] = pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99, "Close": c,
                                          "Volume": rng.integers(1e5, 1e6, n_day).astype(float)}, index=idx)
    frames = asp.wide_frames(data)
    w = evaluate_expr("ret(20)", frames)
    c0 = frames["close"]["F00"]
    assert abs(w["F00"].iloc[-1] - (c0.iloc[-1] / c0.iloc[-21] - 1)) < 1e-12
    w2 = evaluate_expr("mom(252,21)", frames)
    assert abs(w2["F00"].iloc[-1] - (c0.iloc[-22] / c0.iloc[-253] - 1)) < 1e-12
    w3 = evaluate_expr("rsi(14)", frames)
    import indicators as ind
    assert abs(round(float(w3["F00"].iloc[-1]), 1) - ind._rsi(c0)) < 0.11
    data2 = {k: v.copy() for k, v in data.items()}
    for k in data2:
        data2[k].iloc[-30:, :4] *= 1.4
    w4 = evaluate_expr("0.5*ret(20) - vol(20)/2", asp.wide_frames(data2))
    w4o = evaluate_expr("0.5*ret(20) - vol(20)/2", frames)
    assert np.allclose(w4.iloc[:-30].fillna(-9).values, w4o.iloc[:-30].fillna(-9).values)
    try:
        evaluate_expr("1 + 2", frames); raise AssertionError
    except DSLError:
        pass
    print("✅ 2 求值一致、無前視、純常數拒絕")

    # 3) 測試 + 帳本 + 核准（植入：低 vol 有正漂移 → -vol(60) 應正 IC）
    sig = rng.uniform(0.008, 0.03, n_sym)
    for k in range(n_sym):
        drift = 0.0015 * (0.03 - sig[k]) / 0.022 - 0.0003
        c = 100 * np.cumprod(1 + rng.normal(drift, sig[k], n_day))
        data[f"F{k:02d}"]["Close"] = c; data[f"F{k:02d}"]["High"] = c * 1.01; data[f"F{k:02d}"]["Low"] = c * 0.99
    frames = asp.wide_frames(data)
    dates = asp.sample_dates(frames["close"].index)
    st: dict = {}
    res = test_factor("-vol(60)", frames, dates, ledger=st.get("factor_lab"))
    assert res["ic"] is not None and res["ic"] > 0.03 and res["n_trials"] == 1, res
    ledger_add(st, res, "2026-09-19")
    res2 = test_factor("ret(5)", frames, dates, ledger=st.get("factor_lab"))
    assert res2["n_trials"] == 2
    ledger_add(st, res2, "2026-09-19")
    ok, msg = approve(st, "lowvol", "-vol(60)", "2026-09-19", res)
    assert ok and approved_factors(st) == {"lowvol": "-vol(60)"}
    assert not approve(st, "vol_60", "-vol(60)", "2026-09-19")[0] and not approve(st, "Bad Name!", "-vol(60)", "2026-09-19")[0]
    assert not approve(st, "x", "-vol(60)", "2026-09-19")[0] and not approve(st, "inj", "__import__('os')", "2026-09-19")[0]
    assert drop(st, "lowvol") and not drop(st, "lowvol") and approved_factors(st) == {}
    approve(st, "low_vol", "-vol(60)", "2026-09-19")
    assert "`low_vol`" in list_text(st) and "low_vol=" not in list_text(st)                 # 底線名稱進 code span
    drop(st, "low_vol")
    for txt in (test_text(res), test_text({"expr": "x", "ic": None, "reason": "r", "n_trials": 1}), list_text(st)):
        assert "**" not in txt and txt
    print(test_text(res))
    print("✅ 3 測試/帳本/核准/撤銷")
    print("\nfactor_lab selftest OK ✅")
