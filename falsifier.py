"""
falsifier.py — Hypothesis Falsifier：投資故事反駁引擎（Batch 1：核心測試）
RBS Finance Dashboard

⚠️ 本工具只能告訴你故事何時是假的，永遠不能告訴你它是真的。
「存活」只代表此樣本內未被拒絕；歷史上通過同樣測試後死掉的故事，
沒有籃子、沒有資料、不在任何資料庫裡——基準率缺席，通過率就沒有意義。

輸入一個假設規格（spec）：
  {"statement": "AI 資本支出增加 → 記憶體供應商未來六個月跑贏大盤",
   "basket": ["MU","WDC","STX","SNDK"], "benchmark": "SPY",
   "horizon_days": 126, "direction": "outperform"}
對它跑反駁測試（Batch 1 實作 5 項）：
  T1 漂移顯著性（circular block bootstrap——重疊視窗的正確處理：
     六個月持有逐日進場≠上千獨立樣本，有效樣本 ≈ n/視窗長）
  T2 日期穩健性（rolling 進場勝率 + 逐年分解——抓「只有特定起訖日成立」）
  T3 晚進場測試（故事已見報——trailing 超額前 1/3 的日子——才進場還成立嗎）
  T4 交易成本後存活
  T5 事件日進場（使用者提供公告日時的 CAR-lite：+lag 進場超額）
統計方法依正統文獻：block bootstrap（Politis-Romano）、事後漂移檢定（CAR/PEAD
精神）、AHM 研究日誌紀律。Batch 2/3 加 regime 切分、動能對照、跨市場、DSR 帳本。

已知且無法修復的限制（報告中強制揭露）：
- 籃子成分是「今天的名單」——回填/存活者偏誤無法用歷史數據消除
- 下市公司在免費資料源缺席——墳墓對照組天然缺席
純邏輯離線可測；fetch 層走 yfinance。教育用途，非投資建議。
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

DEFAULT_HORIZON = 126        # 六個月交易日
DEFAULT_COST_RT = 0.001      # 來回成本（與 backtest.py 一致）
BOOT_N = 1000
BOOT_BLOCK = 21              # 月級區塊（處理報酬自相關）
MIN_OBS = 252                # 至少一年日資料才給判定


# ── 純邏輯：籃子與超額報酬 ────────────────────────────────────────────────────

def basket_series(closes: dict[str, pd.Series]) -> pd.Series | None:
    """等權籃子指數（每日再平衡＝正規化後平均）。<2 檔回 None。"""
    ser = [s.dropna() for s in closes.values() if s is not None and len(s.dropna()) > 1]
    if len(ser) < 2:
        return None
    df = pd.concat(ser, axis=1).dropna()
    if len(df) < 2:
        return None
    norm = df / df.iloc[0]
    return norm.mean(axis=1)


def _align(basket: pd.Series, bench: pd.Series) -> pd.DataFrame:
    df = pd.concat({"b": basket, "m": bench}, axis=1).dropna()
    return df


def daily_rel_log(basket: pd.Series, bench: pd.Series) -> pd.Series:
    """每日相對對數報酬 d_t = Δln(basket) − Δln(bench)。"""
    df = _align(basket, bench)
    return (np.log(df["b"]).diff() - np.log(df["m"]).diff()).dropna()


def rolling_excess(basket: pd.Series, bench: pd.Series,
                   horizon: int = DEFAULT_HORIZON, lag: int = 1) -> pd.Series:
    """
    逐日訊號、+lag 日收盤進場（無前視）、持有 horizon 交易日的
    籃子−基準 簡單超額報酬序列（索引=進場日）。**視窗高度重疊**——
    顯著性判定必須走 block bootstrap（T1），不可對本序列直接 t 檢定。
    """
    df = _align(basket, bench)
    b, m = df["b"].to_numpy(float), df["m"].to_numpy(float)
    n = len(df)
    if n < lag + horizon + 1:      # 至少要能開出一個完整視窗
        return pd.Series(dtype=float)
    e0 = lag                                  # 第一個進場位置
    last = n - horizon                        # 進場位置上限（exclusive）
    idx = np.arange(e0, last)
    exc = (b[idx + horizon] / b[idx]) - (m[idx + horizon] / m[idx])
    return pd.Series(exc, index=df.index[idx])


# ── T1 漂移顯著性（circular block bootstrap）─────────────────────────────────

def block_bootstrap_test(basket: pd.Series, bench: pd.Series,
                         horizon: int = DEFAULT_HORIZON,
                         direction: str = "outperform",
                         n_boot: int = BOOT_N, block: int = BOOT_BLOCK,
                         seed: int = 7) -> dict:
    """
    H0：籃子相對基準無漂移。統計量 = 每日相對對數報酬均值（重疊視窗的
    超額報酬本質上就是它的視窗和）。circular block bootstrap 對「去均值後」
    的序列重抽（保留自相關），單尾 p 值。有效樣本數誠實回報 ≈ n/block。
    """
    d = daily_rel_log(basket, bench).to_numpy(float)
    n = len(d)
    if n < MIN_OBS:
        return {"test": "T1 漂移顯著性", "survived": None, "p_value": None,
                "detail": f"資料不足（{n} 日 < {MIN_OBS}）"}
    obs = float(d.mean())
    dd = d - obs                              # 去均值 → 建 H0 分佈
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(n / block)
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    # circular blocks：每列串起 n_blocks 段長 block 的環狀切片，截到 n
    offs = np.arange(block)
    pos = (starts[:, :, None] + offs[None, None, :]) % n     # (B, nb, block)
    samples = dd[pos].reshape(n_boot, -1)[:, :n]
    boot_means = samples.mean(axis=1)
    if direction == "underperform":
        p = float((boot_means <= obs).mean())
    else:
        p = float((boot_means >= obs).mean())
    p = max(p, 1.0 / n_boot)
    eff_n = n / block
    ann = obs * 252
    survived = bool(p < 0.05)
    return {"test": "T1 漂移顯著性（block bootstrap）", "survived": survived,
            "p_value": round(p, 4),
            "detail": (f"日均相對報酬 {obs * 1e4:+.1f}bp（年化 {ann:+.1%}），"
                       f"單尾 p={p:.3f}，有效樣本 ≈{eff_n:.0f} 區塊"
                       f"（{n} 日 ÷ {block} 日區塊——別被重疊視窗的假樣本數騙了）")}


# ── T2 日期穩健性 ─────────────────────────────────────────────────────────────

def date_robustness(excess: pd.Series, direction: str = "outperform") -> dict:
    """rolling 進場勝率 + 逐年均值——抓「只有特定起訖日期成立」。"""
    if len(excess) < 60:
        return {"test": "T2 日期穩健性", "survived": None,
                "detail": f"視窗數不足（{len(excess)}）"}
    sign = -1.0 if direction == "underperform" else 1.0
    ex = excess * sign
    win = float((ex > 0).mean())
    yearly = ex.groupby(ex.index.year).mean()
    neg_years = [str(y) for y, v in yearly.items() if v < 0]
    frac_neg = len(neg_years) / len(yearly)
    survived = bool(win >= 0.5 and frac_neg <= 0.4)
    yr_s = "、".join(f"{y}:{v:+.1%}" for y, v in yearly.items())
    detail = f"進場日勝率 {win:.0%}；逐年均超額 {yr_s}"
    if neg_years:
        detail += f"——{'、'.join(neg_years)} 年為負：故事有明顯的時段依賴"
    return {"test": "T2 日期穩健性（rolling 進場）", "survived": survived,
            "detail": detail}


# ── T3 晚進場測試（故事已見報才進場）─────────────────────────────────────────

def late_entry_test(basket: pd.Series, bench: pd.Series,
                    horizon: int = DEFAULT_HORIZON, trailing: int = 63,
                    direction: str = "outperform") -> dict:
    """
    「等到故事已經跑出來（trailing 3 個月超額位於前 1/3 的日子）才進場」
    的前瞻超額 vs 全期平均——顯著縮水甚至轉負 ⇒ 你看到報導時已太晚。
    （無事件日資料時對「公告後才買是否太晚」的價格空間近似。）
    """
    df = _align(basket, bench)
    if len(df) < trailing + horizon + 60:
        return {"test": "T3 晚進場", "survived": None, "detail": "資料不足"}
    b, m = df["b"], df["m"]
    trail = (b / b.shift(trailing) - 1) - (m / m.shift(trailing) - 1)
    fwd = (b.shift(-horizon) / b - 1) - (m.shift(-horizon) / m - 1)
    ok = pd.concat({"t": trail, "f": fwd}, axis=1).dropna()
    if len(ok) < 60:
        return {"test": "T3 晚進場", "survived": None, "detail": "重疊樣本不足"}
    sign = -1.0 if direction == "underperform" else 1.0
    hot = ok[ok["t"] * sign >= ok["t"].mul(sign).quantile(2 / 3)]
    all_mean = float(ok["f"].mean() * sign)
    hot_mean = float(hot["f"].mean() * sign)
    decay = (hot_mean - all_mean)
    survived = bool(hot_mean > 0)
    detail = (f"任何時點進場平均前瞻超額 {all_mean:+.1%}；"
              f"故事已跑出來後才進場 {hot_mean:+.1%}"
              f"（衰減 {decay:+.1%}，樣本 {len(hot)} 個重疊日）")
    if hot_mean <= 0 < all_mean:
        detail += "——動能兌現後進場的人接的是別人的獲利了結"
    return {"test": "T3 晚進場（故事見報後）", "survived": survived, "detail": detail}


# ── T4 交易成本後存活 ─────────────────────────────────────────────────────────

def cost_survival(excess: pd.Series, cost_rt: float = DEFAULT_COST_RT,
                  direction: str = "outperform") -> dict:
    """單次進出來回成本後，平均視窗超額是否仍為正。"""
    if not len(excess):
        return {"test": "T4 成本後存活", "survived": None, "detail": "無視窗"}
    sign = -1.0 if direction == "underperform" else 1.0
    gross = float(excess.mean() * sign)
    net = gross - cost_rt
    return {"test": "T4 交易成本後存活", "survived": bool(net > 0),
            "detail": (f"視窗平均超額 毛 {gross:+.2%} → 扣 {cost_rt:.2%} 來回成本後 "
                       f"淨 {net:+.2%}")}


# ── T5 事件日進場（CAR-lite，使用者提供公告日時）─────────────────────────────

def event_entry_excess(basket: pd.Series, bench: pd.Series,
                       events: list[str], horizon: int = DEFAULT_HORIZON,
                       lag: int = 1, direction: str = "outperform") -> dict:
    """
    事件日 +lag 收盤進場、持有 horizon 的平均超額（market-adjusted CAR 精神）。
    事件 <4 個只報數字不判定（橫斷面檢定力不足——誠實優先）。
    """
    df = _align(basket, bench)
    sign = -1.0 if direction == "underperform" else 1.0
    vals = []
    for ev in events or []:
        try:
            ts = pd.Timestamp(ev)
        except ValueError:
            continue
        pos = df.index.searchsorted(ts)        # 事件日或其後第一個交易日
        e = pos + lag
        x = e + horizon
        if x >= len(df):
            continue
        b, m = df["b"], df["m"]
        vals.append(((b.iloc[x] / b.iloc[e] - 1) - (m.iloc[x] / m.iloc[e] - 1)) * sign)
    if not vals:
        return {"test": "T5 事件日進場", "survived": None,
                "detail": "未提供事件日（spec 加 events:[YYYY-MM-DD,…] 可啟用）"}
    mean_v = float(np.mean(vals))
    if len(vals) < 4:
        return {"test": "T5 事件日進場", "survived": None,
                "detail": (f"僅 {len(vals)} 個事件（<4，不足以判定）："
                           f"平均事後超額 {mean_v:+.1%}——當線索看，別當結論")}
    t_stat = mean_v / (float(np.std(vals, ddof=1)) / math.sqrt(len(vals)) + 1e-12)
    survived = bool(mean_v > 0 and t_stat > 1.0)
    return {"test": "T5 事件日進場（CAR-lite）", "survived": survived,
            "detail": (f"{len(vals)} 個事件、+{lag} 日進場：平均事後超額 {mean_v:+.1%}"
                       f"（t≈{t_stat:.1f}）——t<2 就別太興奮")}


# ── 組裝與報告 ────────────────────────────────────────────────────────────────

def run_tests(closes: dict[str, pd.Series], bench_close: pd.Series,
              spec: dict) -> dict:
    """對已抓好的價格跑 Batch 1 全部測試。回 {basket_n, results:[...]}。"""
    horizon = int(spec.get("horizon_days", DEFAULT_HORIZON))
    direction = spec.get("direction", "outperform")
    bk = basket_series(closes)
    if bk is None or bench_close is None or bench_close.dropna().empty:
        return {"basket_n": 0, "results": [], "error": "籃子或基準價格不足（需 ≥2 檔）"}
    excess = rolling_excess(bk, bench_close, horizon)
    results = [
        block_bootstrap_test(bk, bench_close, horizon, direction),
        date_robustness(excess, direction),
        late_entry_test(bk, bench_close, horizon, direction=direction),
        cost_survival(excess, spec.get("cost_rt", DEFAULT_COST_RT), direction),
        event_entry_excess(bk, bench_close, spec.get("events"), horizon,
                           direction=direction),
    ]
    return {"basket_n": len([s for s in closes.values()
                             if s is not None and len(s.dropna()) > 1]),
            "n_windows": len(excess), "results": results}


HEADER_WARNING = "⚠️ 本工具只能告訴你故事何時是假的，永遠不能告訴你它是真的。"


def falsify_text(spec: dict, out: dict) -> str:
    lines = ["🔨 *假設反駁報告*", HEADER_WARNING, "",
             f"📌 假設：「{spec.get('statement', '?')}」",
             f"籃子：{'、'.join(spec.get('basket', []))}（{out.get('basket_n', 0)} 檔）"
             f" vs {spec.get('benchmark', '?')}｜持有 "
             f"{spec.get('horizon_days', DEFAULT_HORIZON)} 交易日"]
    if out.get("error"):
        lines.append(f"❌ {out['error']}")
        return "\n".join(lines)
    lines.append("")
    n_surv = n_ref = 0
    for r in out["results"]:
        if r["survived"] is True:
            icon = "🟢 存活"
            n_surv += 1
        elif r["survived"] is False:
            icon = "🔴 被推翻"
            n_ref += 1
        else:
            icon = "⚪ 無法判定"
        lines.append(f"{icon}｜{r['test']}")
        lines.append(f"　{r['detail']}")
    lines.append("")
    if n_ref:
        lines.append(f"⚖️ {n_ref} 項被推翻——故事至少在這些角度站不住；先解釋得了它們再談加碼。")
    elif n_surv:
        lines.append(f"⚖️ {n_surv} 項存活、0 項推翻——注意：這些測試餵的是**同一段歷史**，"
                     "是一份證據拍了幾個角度，不是幾份獨立證據。")
    lines.append("_已知限制：籃子成分是今天的名單（回填偏誤）；下市公司缺席（存活偏誤）；"
                 "免費日線數據。教育用途，非投資建議。_")
    return "\n".join(lines)


# ── 抓取層（需網路）───────────────────────────────────────────────────────────

def fetch_and_run(spec: dict, period: str = "5y") -> tuple[dict, str] | None:
    """抓價 → 跑測試 → (out, report_text)。"""
    try:
        from sector_scan import _batch_closes
        tks = list(dict.fromkeys(spec.get("basket", [])))
        bench = spec.get("benchmark", "SPY")
        closes = _batch_closes(tks + [bench], period, min_len=MIN_OBS)
        bench_s = closes.pop(bench, None)
        out = run_tests(closes, bench_s, spec)
        return out, falsify_text(spec, out)
    except Exception:
        return None


# ── CLI 自我測試（離線純邏輯，工程化合成資料）─────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 1500                                              # ~6 年日資料
    idx = pd.bdate_range("2020-06-01", periods=n)
    m_ret = rng.normal(0.0003, 0.01, n)
    bench = pd.Series(100 * np.cumprod(1 + m_ret), index=idx)

    # 情境 A：真有優勢（每日 +6bp 相對漂移）→ T1 應存活、p 小
    edge = 0.0006
    bA = pd.Series(100 * np.cumprod(1 + m_ret + edge + rng.normal(0, 0.004, n)), index=idx)
    closesA = {"X1": bA * 1.0, "X2": bA * rng.uniform(0.9, 1.1)}
    bkA = basket_series(closesA)
    assert bkA is not None and abs(bkA.iloc[0] - 1.0) < 1e-9

    t1 = block_bootstrap_test(bkA, bench, 126)
    assert t1["survived"] is True and t1["p_value"] < 0.05, t1

    # 情境 B：純噪音（無漂移）→ T1 不應宣稱顯著
    bB = pd.Series(100 * np.cumprod(1 + m_ret + rng.normal(0, 0.004, n)), index=idx)
    t1b = block_bootstrap_test(bB, bench, 126)
    assert t1b["p_value"] > 0.01, t1b            # 固定 seed 下不得誤判為極顯著
    # 方向反轉：underperform 主張對上正漂移 → 必不顯著
    t1c = block_bootstrap_test(bkA, bench, 126, direction="underperform")
    assert t1c["survived"] is False, t1c

    # rolling_excess 手算對照：造 5 點玩具序列
    tb = pd.Series([100, 110, 121, 133.1, 146.41],
                   index=pd.bdate_range("2026-01-05", periods=5))   # +10%/日
    tm = pd.Series([100.0] * 5, index=tb.index)                      # 基準不動
    ex = rolling_excess(tb, tm, horizon=2, lag=1)
    # 進場位置 1、2：b[3]/b[1]-1 = 21%、b[4]/b[2]-1 = 21%
    assert len(ex) == 2 and abs(ex.iloc[0] - 0.21) < 1e-9, ex
    # 無前視：改動最後一根之後的資料不影響先前視窗（玩具上等價於截斷不變）
    ex_trunc = rolling_excess(tb.iloc[:4], tm.iloc[:4], horizon=2, lag=1)
    assert abs(ex_trunc.iloc[0] - ex.iloc[0]) < 1e-12

    # T2 日期穩健性：優勢籃子勝率應高；再造「只有前半段成立」的故事 → 抓出時段依賴
    exA = rolling_excess(bkA, bench, 126)
    t2 = date_robustness(exA)
    assert t2["survived"] is True and "勝率" in t2["detail"]
    half = n // 2
    m2 = rng.normal(0.0003, 0.01, n)
    bench2 = pd.Series(100 * np.cumprod(1 + m2), index=idx)
    r_half = np.where(np.arange(n) < half, m2 + 0.0012, m2 - 0.0012)   # 泡沫：漲完全跌回
    bH = pd.Series(100 * np.cumprod(1 + r_half + rng.normal(0, 0.003, n)), index=idx)
    t2h = date_robustness(rolling_excess(bH, bench2, 126))
    assert t2h["survived"] is False and "為負" in t2h["detail"], t2h

    # T3 晚進場——兩種故事該給出相反判定：
    # (a) 持續型趨勢（bkA）：故事見報後進場仍賺 → 存活（動能持續）
    t3a = late_entry_test(bkA, bench, 126)
    assert t3a["survived"] is True, t3a
    # (b) 曇花一現：30 天暴走（短於 63 天偵測窗）後緩慢回吐——
    #     等 trailing 動能訊號亮起時行情已結束 → 晚進場必接刀
    r_spike = m2.copy()
    r_spike[100:130] += 0.02
    r_spike[130:] -= 0.0006
    bS = pd.Series(100 * np.cumprod(1 + r_spike + rng.normal(0, 0.003, n)), index=idx)
    t3s = late_entry_test(bS, bench2, 126)
    assert "衰減" in t3s["detail"]
    assert t3s["survived"] is False, t3s

    # T4 成本：毛 0.30% 視窗超額、成本 0.1% → 淨 0.2% 存活；毛 0.05% → 被推翻
    t4a = cost_survival(pd.Series([0.003] * 100))
    assert t4a["survived"] is True
    t4b = cost_survival(pd.Series([0.0005] * 100))
    assert t4b["survived"] is False

    # T5 事件日：4 個事件手工對照
    evs = [str(idx[i].date()) for i in (100, 400, 700, 1000)]
    t5 = event_entry_excess(bkA, bench, evs, horizon=126, lag=1)
    assert t5["survived"] is not None and "4 個事件" in t5["detail"], t5
    t5few = event_entry_excess(bkA, bench, evs[:2], horizon=126)
    assert t5few["survived"] is None and "不足以判定" in t5few["detail"]
    t5none = event_entry_excess(bkA, bench, None)
    assert t5none["survived"] is None

    # 端到端組裝 + 報告
    outA = run_tests(closesA, bench, {"statement": "測試故事", "basket": ["X1", "X2"],
                                      "benchmark": "SPY", "horizon_days": 126,
                                      "events": evs})
    txt = falsify_text({"statement": "測試故事", "basket": ["X1", "X2"],
                        "benchmark": "SPY", "horizon_days": 126}, outA)
    assert HEADER_WARNING in txt and "回填偏誤" in txt and "非投資建議" in txt
    assert ("同一段歷史" in txt) or ("被推翻" in txt)
    out_bad = run_tests({"X1": bA}, bench, {"basket": ["X1"]})
    assert out_bad.get("error")

    print(f"✅ falsifier 離線自我測試通過（T1 優勢 p={t1['p_value']}、"
          f"噪音 p={t1b['p_value']}、時段依賴/晚進場均被抓出）")
