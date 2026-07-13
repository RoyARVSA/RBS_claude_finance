"""
thesis.py — 投資論點追蹤器（Thesis Tracker）
RBS Finance Dashboard

方法論依據 Anthropic 官方 financial-services 的 thesis-tracker skill：
「論點必須可否證（falsifiable）——若沒有任何事能推翻它，那不是論點。」
每檔標的記錄：方向、核心論點、支柱（pillars）、風險、催化劑、目標價、
**失效價（stop）**與信念等級；掃描循環自動監測失效觸發並推播警示；
超過 90 天未更新的論點在晨報提醒複查（官方：至少每季複查一次）。

狀態存於 watchlist_state.json 的 "theses" 鍵（bot 管理、網頁唯讀）。
純邏輯（set/close/check_triggers/stale/text）離線可測。非投資建議。
"""
from __future__ import annotations

import datetime as _dt

STALE_DAYS = 90          # 官方：至少每季複查
MAX_THESES = 30
MAX_LIST_ITEMS = 8       # pillars/risks/catalysts 各自上限


def _th(state: dict) -> dict:
    return state.setdefault("theses", {})


def _today() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")


# ── 純邏輯：建立 / 更新 / 結案 ────────────────────────────────────────────────

def set_thesis(state: dict, ticker: str, direction: str, statement: str) -> tuple[bool, str]:
    """建立或改寫論點。direction：多/空。回 (成功, 訊息)。"""
    ticker = (ticker or "").upper().strip()
    if not ticker or direction not in ("多", "空"):
        return False, "用法：/thesis TICKER 多|空 論點一句話"
    if not statement.strip():
        return False, "論點不可為空——寫下「為什麼」與可被推翻的主張"
    ths = _th(state)
    if ticker not in ths and len(ths) >= MAX_THESES:
        return False, f"論點數已達上限 {MAX_THESES}——先 /thesis TICKER close 結掉舊的"
    prev = ths.get(ticker)
    ths[ticker] = {
        "direction": direction, "statement": statement.strip()[:300],
        "pillars": (prev or {}).get("pillars", []),
        "risks": (prev or {}).get("risks", []),
        "catalysts": (prev or {}).get("catalysts", []),
        "target": (prev or {}).get("target"),
        "stop": (prev or {}).get("stop"),
        "conviction": (prev or {}).get("conviction", "中"),
        "status": "active",
        "created": (prev or {}).get("created", _today()),
        "updated": _today(),
        "log": ((prev or {}).get("log", []) +
                [{"date": _today(), "note": "論點" + ("改寫" if prev else "建立")}])[-20:],
        "stop_alerted": False, "target_alerted": False,
    }
    return True, f"✅ {ticker} 論點已{'更新' if prev else '建立'}（{direction}）"


def add_item(state: dict, ticker: str, kind: str, text: str) -> tuple[bool, str]:
    """kind：pillar/risk/cat。加一條支柱/風險/催化劑。"""
    ticker = (ticker or "").upper()
    th = _th(state).get(ticker)
    if not th:
        return False, f"{ticker} 尚無論點——先 /thesis {ticker} 多|空 論點"
    key = {"pillar": "pillars", "risk": "risks", "cat": "catalysts"}.get(kind)
    if not key or not text.strip():
        return False, "用法：/thesis TICKER pillar|risk|cat 內容"
    if len(th[key]) >= MAX_LIST_ITEMS:
        return False, f"{key} 已達 {MAX_LIST_ITEMS} 條上限"
    th[key].append(text.strip()[:150])
    th["updated"] = _today()
    lbl = {"pillar": "支柱", "risk": "風險", "cat": "催化劑"}[kind]
    return True, f"✅ {ticker} 新增{lbl}：{text.strip()[:60]}"


def set_level(state: dict, ticker: str, kind: str, value: float) -> tuple[bool, str]:
    """kind：target/stop。設定目標價 / 失效價（stop 由掃描自動監測）。"""
    ticker = (ticker or "").upper()
    th = _th(state).get(ticker)
    if not th:
        return False, f"{ticker} 尚無論點"
    if kind not in ("target", "stop") or value is None or value <= 0:
        return False, "用法：/thesis TICKER target|stop 價位"
    th[kind] = float(value)
    th[f"{kind}_alerted"] = False               # 改價位後重新武裝
    th["updated"] = _today()
    lbl = "目標價" if kind == "target" else "失效價"
    return True, f"✅ {ticker} {lbl} → {value:g}（掃描自動監測）"


def set_conviction(state: dict, ticker: str, level: str) -> tuple[bool, str]:
    ticker = (ticker or "").upper()
    th = _th(state).get(ticker)
    if not th:
        return False, f"{ticker} 尚無論點"
    if level not in ("高", "中", "低"):
        return False, "用法：/thesis TICKER conv 高|中|低"
    th["conviction"] = level
    th["updated"] = _today()
    th["log"] = (th.get("log", []) + [{"date": _today(), "note": f"信念 → {level}"}])[-20:]
    return True, f"✅ {ticker} 信念等級 → {level}"


def log_note(state: dict, ticker: str, note: str) -> tuple[bool, str]:
    """記一筆更新日誌（新數據點對論點的影響——官方 Step 2）。"""
    ticker = (ticker or "").upper()
    th = _th(state).get(ticker)
    if not th:
        return False, f"{ticker} 尚無論點"
    if not note.strip():
        return False, "用法：/thesis TICKER note 新數據點與影響"
    th["log"] = (th.get("log", []) + [{"date": _today(), "note": note.strip()[:200]}])[-20:]
    th["updated"] = _today()
    return True, f"✅ {ticker} 已記錄更新"


def close_thesis(state: dict, ticker: str, note: str = "") -> tuple[bool, str]:
    ticker = (ticker or "").upper()
    th = _th(state).get(ticker)
    if not th:
        return False, f"{ticker} 尚無論點"
    th["status"] = "closed"
    th["updated"] = _today()
    th["log"] = (th.get("log", []) +
                 [{"date": _today(), "note": f"結案{('：' + note.strip()[:150]) if note.strip() else ''}"}])[-20:]
    return True, f"🏁 {ticker} 論點已結案（紀錄保留可查）"


# ── 純邏輯：自動監測 ──────────────────────────────────────────────────────────

def check_triggers(state: dict, prices: dict) -> list[str]:
    """
    掃描時呼叫：檢查 active 論點的失效價/目標價觸發。
    多單：價 ≤ stop 失效、價 ≥ target 達標；空單方向相反。
    每個價位只警示一次（改價位後重新武裝）。就地標記，回警示訊息清單。
    """
    msgs = []
    for tk, th in _th(state).items():
        if th.get("status") not in ("active", "damaged"):   # damaged 仍監測（可重武裝）
            continue
        px = prices.get(tk)
        if not px or px <= 0:
            continue
        is_long = th.get("direction") == "多"
        stop, target = th.get("stop"), th.get("target")
        if stop and not th.get("stop_alerted"):
            breached = px <= stop if is_long else px >= stop
            if breached:
                th["stop_alerted"] = True
                th["status"] = "damaged"
                th["log"] = (th.get("log", []) +
                             [{"date": _today(), "note": f"⚠️ 觸發失效價 {stop:g}（現價 {px:.2f}）"}])[-20:]
                msgs.append(f"⚠️ *{tk} 論點失效警示*：現價 {px:.2f} "
                            f"{'跌破' if is_long else '突破'}失效價 {stop:g}。\n"
                            f"論點：「{th.get('statement','')[:80]}」\n"
                            f"該認錯還是該加碼？重新檢視支柱與風險：`/thesis {tk}`")
        if target and not th.get("target_alerted"):
            hit = px >= target if is_long else px <= target
            if hit:
                th["target_alerted"] = True
                th["log"] = (th.get("log", []) +
                             [{"date": _today(), "note": f"🎯 到達目標價 {target:g}（現價 {px:.2f}）"}])[-20:]
                msgs.append(f"🎯 *{tk} 論點達標*：現價 {px:.2f} 已達目標 {target:g}。\n"
                            f"檢視是否兌現或上調：`/thesis {tk}`")
    return msgs


def stale_theses(state: dict, today: str | None = None,
                 days: int = STALE_DAYS) -> list[str]:
    """超過 N 天未更新的 active 論點（晨報季度複查提醒——官方 Important Notes）。"""
    today_d = _dt.date.fromisoformat(today or _today())
    out = []
    for tk, th in _th(state).items():
        if th.get("status") != "active":
            continue
        try:
            upd = _dt.date.fromisoformat(th.get("updated", th.get("created", "")))
        except ValueError:
            continue
        if (today_d - upd).days >= days:
            out.append(tk)
    return sorted(out)


# ── 純邏輯：輸出 ──────────────────────────────────────────────────────────────

def thesis_text(ticker: str, th: dict) -> str:
    st_icon = {"active": "🟢", "damaged": "🟠", "closed": "🏁"}.get(th.get("status"), "❔")
    lines = [f"{st_icon} *{ticker}*（{th.get('direction','?')}｜信念 {th.get('conviction','中')}"
             f"｜{th.get('status','?')}）",
             f"📌 {th.get('statement','')}"]
    if th.get("pillars"):
        lines.append("*支柱*：" + "；".join(th["pillars"]))
    if th.get("risks"):
        lines.append("*風險*：" + "；".join(th["risks"]))
    if th.get("catalysts"):
        lines.append("*催化劑*：" + "；".join(th["catalysts"]))
    lv = []
    if th.get("target"):
        lv.append(f"目標 {th['target']:g}")
    if th.get("stop"):
        lv.append(f"失效 {th['stop']:g}")
    if lv:
        lines.append("🎯 " + "｜".join(lv))
    for e in th.get("log", [])[-3:]:
        lines.append(f"　· {e['date']} {e['note']}")
    lines.append(f"_建立 {th.get('created','?')}　更新 {th.get('updated','?')}_")
    return "\n".join(lines)


def theses_list_text(state: dict) -> str:
    ths = _th(state)
    if not ths:
        return ("📖 尚無投資論點。\n`/thesis NVDA 多 AI 資本支出週期未完，數據中心營收"
                "年增 >50% 可持續` 建立第一個——論點要可被推翻才算論點。")
    lines = ["📖 *投資論點清單*"]
    order = {"active": 0, "damaged": 1, "closed": 2}
    for tk in sorted(ths, key=lambda t: (order.get(ths[t].get("status"), 9), t)):
        th = ths[tk]
        icon = {"active": "🟢", "damaged": "🟠", "closed": "🏁"}.get(th.get("status"), "❔")
        lv = f"　失效 {th['stop']:g}" if th.get("stop") else "　⚠️ 未設失效價"
        if th.get("status") != "active":
            lv = ""
        lines.append(f"{icon} {tk}（{th.get('direction')}）{th.get('statement','')[:50]}…{lv}")
    lines.append("_`/thesis TICKER` 看細節；未設失效價的論點不可否證——快補上。_")
    return "\n".join(lines)


HELP_TEXT = (
    "📖 *論點追蹤器用法*\n"
    "`/thesis` — 清單\n"
    "`/thesis NVDA` — 看單檔\n"
    "`/thesis NVDA 多 論點一句話` — 建立/改寫（多|空）\n"
    "`/thesis NVDA pillar|risk|cat 內容` — 加支柱/風險/催化劑\n"
    "`/thesis NVDA target 1200`｜`stop 850` — 目標/失效價（自動監測）\n"
    "`/thesis NVDA conv 高|中|低`｜`note 新數據點` — 信念/日誌\n"
    "`/thesis NVDA close [原因]` — 結案\n"
    "_論點要可否證：沒有失效條件的信仰不是論點。_")


# ── CLI 自我測試 ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    st: dict = {}
    ok, msg = set_thesis(st, "nvda", "多", "AI 資本支出週期未完")
    assert ok and st["theses"]["NVDA"]["direction"] == "多"
    assert not set_thesis(st, "X", "漲", "y")[0]              # 方向非法
    assert not set_thesis(st, "X", "多", "  ")[0]             # 空論點

    ok, _ = add_item(st, "NVDA", "pillar", "數據中心營收年增 >50%")
    ok2, _ = add_item(st, "NVDA", "risk", "出口管制擴大")
    ok3, _ = add_item(st, "NVDA", "cat", "Q3 財報 11 月")
    assert ok and ok2 and ok3
    assert not add_item(st, "AMD", "pillar", "x")[0]          # 未建論點

    assert set_level(st, "NVDA", "stop", 850)[0]
    assert set_level(st, "NVDA", "target", 1400)[0]
    assert not set_level(st, "NVDA", "stop", -1)[0]

    # 觸發：多單跌破失效價 → 一次性警示 + 狀態 damaged
    msgs = check_triggers(st, {"NVDA": 840.0})
    assert len(msgs) == 1 and "失效警示" in msgs[0], msgs
    assert st["theses"]["NVDA"]["status"] == "damaged"
    assert check_triggers(st, {"NVDA": 830.0}) == []          # 不重複警示
    # 改失效價 → 重新武裝
    set_level(st, "NVDA", "stop", 800)
    assert st["theses"]["NVDA"]["stop_alerted"] is False
    assert len(check_triggers(st, {"NVDA": 790.0})) == 1

    # 空單：突破失效價才觸發
    set_thesis(st, "TSLA", "空", "估值透支交付成長")
    set_level(st, "TSLA", "stop", 300)
    assert check_triggers(st, {"TSLA": 290.0}) == []
    assert len(check_triggers(st, {"TSLA": 305.0})) == 1

    # 目標達標（多單）
    set_thesis(st, "AAPL", "多", "服務營收再加速")
    set_level(st, "AAPL", "target", 250)
    m2 = check_triggers(st, {"AAPL": 251.0})
    assert len(m2) == 1 and "達標" in m2[0]
    assert st["theses"]["AAPL"]["status"] == "active"          # 達標不改狀態

    # 季度複查
    st["theses"]["AAPL"]["updated"] = "2026-01-01"
    assert stale_theses(st, "2026-07-13") == ["AAPL"]

    # 結案 + 文字
    assert close_thesis(st, "TSLA", "已回補")[0]
    assert st["theses"]["TSLA"]["status"] == "closed"
    t = thesis_text("NVDA", st["theses"]["NVDA"])
    assert "支柱" in t and "失效 800" in t
    lst = theses_list_text(st)
    assert "NVDA" in lst and "🏁 TSLA" in lst
    assert "尚無投資論點" in theses_list_text({})

    print(f"✅ thesis 離線自我測試通過（{len(st['theses'])} 檔論點、觸發/重武裝/空單方向全過）")
