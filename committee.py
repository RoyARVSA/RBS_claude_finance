"""
committee.py – 機構決策委員會（TradingAgents 式金融公司模擬）的純邏輯層

模擬一家金融機構的完整決策鏈：
  四位分析師（技術/基本面/籌碼/總經）→ 多空研究員對辯 → 交易員提案
  → 風控主管（**硬限制由本模組的確定性規則判定，LLM 只做定性補充**）
  → 投資經理最終裁決 → 與本平台量化評分交叉比較（一致/分歧）。

本模組只放：角色提示、立場/結論解析、硬風控規則、量化交叉比較——全部純函數離線可測。
LLM 呼叫與資料抓取由 app.py 負責。約 9 次小型 LLM 呼叫/次，屬高成本模式。
文獻註記：LLM 委員會無經獨立驗證的長期實績；本模式價值在「結構化多視角＋強制反方
＋風控閘門」的決策紀律，非預測能力。分析教育用途，非投資建議。
"""

from __future__ import annotations

import re

# ── 角色提示 ───────────────────────────────────────────────────────────────────

ANALYST_ROLES = {
    "technical":   ("技術分析師", "趨勢、動能、波動、支撐壓力與量化評分"),
    "fundamental": ("基本面分析師", "財務健康、估值、成長、分析師共識與 EPS 紀錄"),
    "chips":       ("籌碼分析師", "選擇權定位、內部人買賣、做空數據"),
    "macro":       ("總經策略師", "利率、通膨、景氣循環對此標的與其產業的影響"),
}


def analyst_prompt(domain: str, tickers: list[str] | None = None) -> str:
    title, scope = ANALYST_ROLES[domain]
    base = (f"你是機構裡的{title}，負責領域：{scope}。\n"
            "只根據下方資料發表專業觀點，引用具體數字，並盡可能給出**關鍵價位或關鍵門檻**"
            "（例如：跌破哪個價位/數據惡化到什麼程度會改變你的看法）；"
            "資料缺漏就明說觀點受限，不得編造。\n")
    if tickers and len(tickers) > 1:
        return base + (f"本次會議討論多檔標的：{'、'.join(tickers)}。"
                       "請逐檔簡評（各 ≤60 字），並在最後一行指出你領域內"
                       "「最看好」與「最不看好」的各一檔。")
    return base + ("（≤120字）最後獨立一行必須是「立場: <數字>」，"
                   "數字介於 -1（極空）到 +1（極多），資料不足時給 0。")


RESEARCHER_BULL = (
    "你是多方研究員。下方是四位分析師的報告。整合出**最強的看多論證**（≤150字），"
    "必須引用至少兩位分析師的具體數據，並指出論證成立的前提。不得編造。")

RESEARCHER_BEAR = (
    "你是空方研究員。下方是四位分析師的報告。整合出**最強的看空論證**（≤150字），"
    "必須引用至少兩位分析師的具體數據，並指出下檔情境的觸發條件。不得編造。")

TRADER_PROMPT = (
    "你是交易員。根據分析師報告、多空對辯與下方的**系統參考價位（ATR 基準）**，"
    "提出可執行的交易計畫（≤150字）：\n"
    "方向、進場區間（具體價位）、停損價（可採用系統 ATR 停損或說明為何調整）、"
    "目標/持有週期、認錯條件（什麼情況證明看錯）。\n"
    "最後獨立一行必須是「方向: 做多」或「方向: 觀望」或「方向: 迴避」。")


def trader_prompt_multi(tickers: list[str]) -> str:
    return (f"你是交易員。本次會議討論 {len(tickers)} 檔標的：{'、'.join(tickers)}。\n"
            "根據分析師報告、多空對辯與系統參考價位，對**每一檔**提出方向與關鍵價位（各 ≤50 字），"
            "並給出資金配置優先序（最想買的排最前，可以全部觀望）。\n"
            "最後獨立一行必須是「優先序: 代碼1 > 代碼2 > …」（觀望/迴避者可不列）。")


def risk_prompt(hard_constraints: list[str]) -> str:
    hc = "\n".join(f"· {c}" for c in hard_constraints) if hard_constraints else "（無硬性限制觸發）"
    return ("你是風控主管。系統已依確定性規則判定以下**硬性限制（不可推翻、必須照列）**：\n"
            f"{hc}\n"
            "請針對交易員提案補充定性風險（≤100字：流動性、事件、集中度等），"
            "並在最後獨立一行給出「風控意見: 放行」或「風控意見: 有條件放行」或「風控意見: 否決」。"
            "若有任何硬性限制觸發，意見不得為「放行」。")


PM_PROMPT = (
    "你是投資經理，做最終裁決。你收到：四位分析師報告、多空對辯、交易員提案、風控意見。\n"
    "用 ≤180 字裁決，必須包含：(1) 採納或修正交易員提案及理由 "
    "(2) 部位建議（占組合 %，依風控硬限制調整）(3) 你最重視的一個分歧點 "
    "(4) 再評估時點（什麼日期或什麼事件後重審此決策）。\n"
    "最後兩行獨立為：\n「結論: 買進」或「結論: 觀望」或「結論: 迴避」\n「信心: 低」或「信心: 中」或「信心: 高」")


def pm_prompt_multi(tickers: list[str]) -> str:
    lines_req = "\n".join(f"「{t}: 買進」或「{t}: 觀望」或「{t}: 迴避」" for t in tickers)
    return ("你是投資經理，做最終裁決。你收到：四位分析師逐檔報告、多空對辯、交易員排序、風控意見。\n"
            f"用 ≤200 字裁決這 {len(tickers)} 檔的取捨：哪檔值得配置、各占組合多少 %（依風控硬限制）、"
            "最重要的一個跨標的分歧點、再評估時點。\n"
            "最後幾行獨立為（每檔一行，然後首選與信心）：\n"
            f"{lines_req}\n「首選: <代碼或 無>」\n「信心: 低」或「信心: 中」或「信心: 高」")


def parse_multi_verdict(text: str, tickers: list[str]) -> dict:
    """逐檔結論 + 首選。回 {"verdicts": {tk: 買進/觀望/迴避|None}, "top_pick", "confidence"}。"""
    out = {"verdicts": {}, "top_pick": None, "confidence": None}
    for tk in tickers:
        m = re.findall(rf"{re.escape(tk)}\s*[:：]\s*(買進|觀望|迴避)", text or "")
        out["verdicts"][tk] = m[-1] if m else None
    p = re.findall(r"首選\s*[:：]\s*([A-Z0-9.\-]+|無)", text or "")
    if p:
        out["top_pick"] = None if p[-1] == "無" else p[-1]
    c = re.findall(r"信心\s*[:：]\s*(低|中|高)", text or "")
    out["confidence"] = c[-1] if c else None
    return out


# ── 純解析 ─────────────────────────────────────────────────────────────────────

def parse_stance(text: str) -> float | None:
    """從分析師回覆抽「立場: +0.x」。抽不到回 None。夾在 [-1,1]。"""
    m = re.search(r"立場\s*[:：]\s*([+-]?\d+(?:\.\d+)?)", text or "")
    if not m:
        return None
    try:
        return max(-1.0, min(1.0, float(m.group(1))))
    except ValueError:
        return None


def parse_direction(text: str) -> str | None:
    # 取「最後一個」匹配：提示要求結論在末行；前文可能討論「若…則方向: 觀望」
    m = re.findall(r"方向\s*[:：]\s*(做多|觀望|迴避)", text or "")
    return m[-1] if m else None


def parse_risk_opinion(text: str) -> str | None:
    m = re.findall(r"風控意見\s*[:：]\s*(放行|有條件放行|否決)", text or "")
    return m[-1] if m else None


def parse_verdict(text: str) -> dict:
    v = re.findall(r"結論\s*[:：]\s*(買進|觀望|迴避)", text or "")
    c = re.findall(r"信心\s*[:：]\s*(低|中|高)", text or "")
    return {"verdict": v[-1] if v else None,
            "confidence": c[-1] if c else None}


# ── 硬風控閘門（確定性規則——LLM 不可推翻）────────────────────────────────────

def hard_risk_check(facts: dict) -> list[str]:
    """
    facts 鍵（皆可缺）：regime_label, ann_vol, quant_score,
    reflection_hit_rate, reflection_n, atr_pos_pct。
    回觸發的硬性限制清單（空 = 無）。規則保守、可解釋。
    """
    out = []
    regime = str(facts.get("regime_label") or "")
    if any(k in regime for k in ("風險", "偏空", "避險")):
        out.append(f"大盤風險濾網觸發（{regime}）：新倉上限減半")
    av = facts.get("ann_vol")
    if av is not None and av > 0.8:
        out.append(f"年化波動 {av:.0%} > 80%：單一部位上限 5%")
    hr, n = facts.get("reflection_hit_rate"), facts.get("reflection_n") or 0
    if hr is not None and n >= 5 and hr < 0.4:
        out.append(f"近 {n} 次 AI 判斷命中率僅 {hr:.0%}：本次信心強制下修一級")
    q = facts.get("quant_score")
    if q is not None and abs(q) < 0.15:
        out.append(f"量化綜合評分 {q:+.2f} 接近中性：不支持重倉方向性部位")
    pp = facts.get("atr_pos_pct")
    if pp is not None and pp > 0.25:
        out.append(f"ATR 建議部位 {pp:.0%} 超過單檔上限 25%：以 25% 為限")
    return out


# ── 量化交叉比較 ───────────────────────────────────────────────────────────────

_VERDICT_NUM = {"買進": 1, "觀望": 0, "迴避": -1}


def compare_with_quant(verdict: str | None, quant_score: float | None,
                       deadband: float = 0.2) -> dict:
    """委員會結論 vs 量化評分方向的一致性（純函數）。"""
    if verdict not in _VERDICT_NUM or quant_score is None:
        return {"agreement": "無法比較", "note": "缺結論或量化評分"}
    v = _VERDICT_NUM[verdict]
    q = 1 if quant_score > deadband else (-1 if quant_score < -deadband else 0)
    if v == q:
        agree, note = "✅ 一致", "委員會與量化系統同向——訊號互相印證，但仍可能同時犯錯"
    elif v * q < 0:
        agree, note = "❌ 分歧", "方向相反——通常代表質性敘事與價格行為打架，寧可觀望或小倉試錯"
    else:
        agree, note = "◐ 部分一致", "一方中性一方有方向——資訊優勢可能在有方向的一方，降低倉位跟隨"
    return {"agreement": agree, "quant_dir": q, "committee_dir": v, "note": note}


# ── CLI 自我測試 ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    assert parse_stance("…看多。\n立場: +0.6") == 0.6
    assert parse_stance("立場：-0.35") == -0.35
    assert parse_stance("立場: 5") == 1.0            # 夾限
    assert parse_stance("沒有立場行") is None
    assert parse_direction("……\n方向: 做多") == "做多"
    assert parse_risk_opinion("風控意見：有條件放行") == "有條件放行"
    v = parse_verdict("……\n結論: 觀望\n信心: 中")
    assert v == {"verdict": "觀望", "confidence": "中"}
    # 前文提到假設性結論時，以最後一個為準（末行才是正式裁決）
    v2 = parse_verdict("若風控否決則結論: 觀望。但綜合評估後——\n結論: 買進\n信心: 中")
    assert v2["verdict"] == "買進"
    assert parse_direction("討論過方向: 迴避 的情境…\n方向: 做多") == "做多"

    hc = hard_risk_check({"regime_label": "⚠️ 風險偏空", "ann_vol": 0.95,
                          "reflection_hit_rate": 0.3, "reflection_n": 8,
                          "quant_score": 0.05, "atr_pos_pct": 0.4})
    print("hard constraints:")
    for c in hc:
        print("  ·", c)
    assert len(hc) == 5
    assert hard_risk_check({}) == []                 # 全缺 → 不觸發
    assert hard_risk_check({"reflection_hit_rate": 0.2, "reflection_n": 3}) == []  # 樣本不足

    assert compare_with_quant("買進", 0.5)["agreement"].endswith("一致")
    assert compare_with_quant("迴避", 0.5)["agreement"].endswith("分歧")
    assert compare_with_quant("觀望", 0.5)["agreement"].startswith("◐")
    assert compare_with_quant("買進", None)["agreement"] == "無法比較"
    assert compare_with_quant("買進", 0.1)["agreement"].startswith("◐")  # 量化中性

    for d in ANALYST_ROLES:
        assert "立場" in analyst_prompt(d)
        assert "逐檔簡評" in analyst_prompt(d, ["NVDA", "VRT"])
    assert "不可推翻" in risk_prompt(["測試限制"])

    # 多檔裁決解析
    mv = parse_multi_verdict(
        "考量後配置如下…\nNVDA: 買進\nVRT: 觀望\nGLD: 迴避\n首選: NVDA\n信心: 中",
        ["NVDA", "VRT", "GLD", "DRAM"])
    assert mv["verdicts"] == {"NVDA": "買進", "VRT": "觀望", "GLD": "迴避", "DRAM": None}
    assert mv["top_pick"] == "NVDA" and mv["confidence"] == "中"
    mv2 = parse_multi_verdict("全部觀望\nNVDA: 觀望\n首選: 無\n信心: 低", ["NVDA"])
    assert mv2["top_pick"] is None and mv2["verdicts"]["NVDA"] == "觀望"
    assert "優先序" in trader_prompt_multi(["A", "B"])
    assert "NVDA: 買進" in pm_prompt_multi(["NVDA"])
    print("\n✅ committee 純邏輯測試通過（含多檔模式）")
