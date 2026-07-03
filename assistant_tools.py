"""
assistant_tools.py – AI 助理 Phase 2：工具編排的純邏輯層（可離線測試）

流程：規劃(plan) → 執行(execute，由網頁層負責抓取) → 回答(answer)。
本模組只放「純邏輯」：工具清單、規劃提示、計畫解析、結果格式化。
真正的抓取/計算（回測/風險/選股）由 app.py 的執行器呼叫現有模組完成。

工具：
  backtest  對單一標的跑 Triple-Barrier 回測，回各訊號規則勝率/獲利因子
  risk      對一組標的算等權組合風險（年化波動、歷史 VaR/CVaR、最大回撤）
  screen    掃描某產業的標的，依動能/風險排名
"""

from __future__ import annotations

import json
import re

KNOWN_TOOLS = {"backtest", "risk", "screen", "options"}
MAX_TOOLS = 4

# 觸發字（純啟發式）：問題含這些字才啟動規劃器，省一次 LLM 呼叫
_TOOL_HINTS = [
    # 回測 / 勝率
    "回測", "backtest", "勝率", "獲利因子", "profit factor", "期望值", "訊號有效",
    "策略表現", "歷史表現", "進出場",
    # 風險（去掉過寬的 "risk"/"volatility"，中文 風險/波動 已涵蓋；var/cvar 夠專指）
    "風險", "var", "cvar", "波動", "回撤", "drawdown", "下檔", "虧損機率",
    # 選股 / 篩選（"有哪些股/標的" 而非過寬的 "有哪些"）
    "選股", "篩選", "screen", "找標的", "推薦標的", "哪些股", "哪幾檔", "強勢股",
    "產業裡", "族群裡", "有哪些股", "有哪些標的", "候選", "掃描",
    # 選擇權情緒（不用裸 "iv"：會誤中 positive/derivative 等英文字）
    "選擇權", "put/call", "put call", "pcr", "隱含波動", "偏斜", "options",
    "避險情緒",
]


def might_need_tools(question: str) -> bool:
    """純啟發式：問題是否可能需要跑分析工具（回測/風險/選股）。"""
    ql = question.lower()
    return any((h in ql) if h.isascii() else (h in question) for h in _TOOL_HINTS)


def build_planner_prompt(question: str, tickers: list[str],
                         industries: list[str] | None = None) -> str:
    """組出給規劃器 LLM 的提示，要求它只回 JSON 計畫（不需要就回空 tools）。"""
    inds = "、".join((industries or [])[:40]) or "（無清單，可用常見產業名）"
    tk = "、".join(tickers) if tickers else "（問題未指定明確標的）"
    return (
        "你是分析工具的『調度器』。根據使用者問題，決定要不要呼叫下列工具取得客觀數據。\n"
        "只輸出 JSON，格式：{\"tools\":[{\"tool\":\"名稱\",\"args\":{...}}]}。不需任何工具就回 {\"tools\":[]}。\n\n"
        "可用工具：\n"
        "1. backtest  args={\"ticker\":\"AAPL\"}　對單一標的跑歷史回測，取得各訊號規則的勝率/獲利因子。\n"
        "   —— 使用者問『這支能不能買/訊號準不準/回測/勝率/歷史表現』時用。\n"
        "2. risk      args={\"tickers\":[\"AAPL\",\"MSFT\"]}　算等權組合的年化波動、歷史 VaR/CVaR、最大回撤。\n"
        "   —— 問『風險多大/波動/下檔/會賠多少/VaR』時用。\n"
        "3. screen    args={\"industry\":\"半導體\"}　掃描該產業標的，依動能與風險排名，找強勢/候選股。\n"
        "   —— 問『某產業有哪些強勢股/幫我找標的/篩選』時用。\n"
        "4. options   args={\"ticker\":\"AAPL\"}　取得選擇權情緒（Put/Call 比、隱含波動偏斜、情緒分數）。\n"
        "   —— 問『選擇權情緒/Put Call 比/隱含波動/避險情緒』時用（多為美股大型股才有）。\n\n"
        f"已辨識標的：{tk}\n"
        f"可用產業名（screen 用，請盡量對到最接近的一個）：{inds}\n\n"
        f"使用者問題：{question}\n\n"
        "規則：最多 4 個工具；只在真的需要客觀數據時呼叫；純觀念/定義問題回空 tools。只輸出 JSON。"
    )


def _extract_json(text: str) -> dict:
    """從 LLM 回覆中盡量抽出 JSON 物件（容忍 ```json 圍欄與前後贅字）。"""
    if not text:
        return {}
    t = text.strip()
    # 去除 ``` 圍欄
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I | re.M).strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    # 退而求其次：抓第一個 { 到最後一個 }
    i, j = t.find("{"), t.rfind("}")
    if 0 <= i < j:
        try:
            return json.loads(t[i:j + 1])
        except Exception:
            return {}
    return {}


def parse_plan(text: str, valid_tickers: set | None = None,
               valid_industries: set | None = None) -> list[dict]:
    """
    解析規劃器輸出成乾淨的工具呼叫清單（純函數，防呆）。
    回 [{"tool":..., "args":{...}}]，未知工具/壞參數會被濾掉，最多 MAX_TOOLS 個。
    """
    obj = _extract_json(text)
    raw = obj.get("tools") if isinstance(obj, dict) else None
    if not isinstance(raw, list):
        return []

    out: list[dict] = []
    seen: set = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        tool = str(item.get("tool", "")).strip().lower()
        if tool not in KNOWN_TOOLS:
            continue
        args = item.get("args") if isinstance(item.get("args"), dict) else {}
        clean = _clean_args(tool, args, valid_tickers, valid_industries)
        if clean is None:
            continue
        key = (tool, json.dumps(clean, sort_keys=True, ensure_ascii=False))
        if key in seen:
            continue
        seen.add(key)
        out.append({"tool": tool, "args": clean})
        if len(out) >= MAX_TOOLS:
            break
    return out


def _norm_ticker(x) -> str | None:
    if not isinstance(x, str):
        return None
    t = x.strip().upper()
    return t if re.fullmatch(r"[A-Z0-9]{1,6}(?:[.\-][A-Z0-9]{1,4})?", t) else None


def _clean_args(tool: str, args: dict, valid_tickers, valid_industries):
    """依工具正規化/驗證參數；不合法回 None（整個工具略過）。"""
    if tool in ("backtest", "options"):
        t = _norm_ticker(args.get("ticker"))
        return {"ticker": t} if t else None

    if tool == "risk":
        raw = args.get("tickers") or ([args.get("ticker")] if args.get("ticker") else [])
        if isinstance(raw, str):          # 容忍 LLM 回字串（單檔或逗號分隔）
            raw = [p for p in re.split(r"[,\s]+", raw) if p]
        ts = []
        for x in raw if isinstance(raw, list) else []:
            t = _norm_ticker(x)
            if t and t not in ts:
                ts.append(t)
        return {"tickers": ts[:8]} if ts else None

    if tool == "screen":
        ind = args.get("industry")
        if not isinstance(ind, str) or not ind.strip():
            return None
        ind = ind.strip()
        # 若有產業白名單，盡量對到最接近的（完全相符或包含）
        if valid_industries:
            if ind not in valid_industries:
                cand = [v for v in valid_industries if ind in v or v in ind]
                if cand:
                    ind = cand[0]
                else:
                    return None
        return {"industry": ind}

    return None


# ── 結果格式化（純函數）──────────────────────────────────────────────────────

def _pct(v, dp=1):
    try:
        return f"{v * 100:.{dp}f}%"
    except (TypeError, ValueError):
        return "無資料"


def _num(v, dp=2):
    try:
        return f"{v:.{dp}f}"
    except (TypeError, ValueError):
        return "無資料"


def format_tool_results(results: list[dict]) -> str:
    """把執行器回傳的結構化結果組成標註來源的 context 文字（純函數）。"""
    if not results:
        return ""
    lines = ["=== 工具分析結果（客觀數據，來源：本平台回測/風險/選股引擎）==="]
    for r in results:
        if not r or not r.get("ok"):
            lines.append(f"\n【{(r or {}).get('tool', '工具')}】執行失敗："
                         f"{(r or {}).get('error', '無資料')}")
            continue
        tool = r.get("tool")
        if tool == "backtest":
            lines.append(f"\n【回測 {r.get('ticker')}】"
                         f"（{r.get('bars', '?')} 根日線，已扣成本、無前視）")
            top = r.get("top") or []
            if not top:
                lines.append("  交易樣本不足，無有效規則。")
            for x in top:
                lines.append(
                    f"  · {x.get('rule')}：勝率 {_pct(x.get('win_rate'))}　"
                    f"獲利因子 {_num(x.get('profit_factor'))}　"
                    f"期望值 {_pct(x.get('expectancy'), dp=2)}　"
                    f"交易數 {x.get('trades')}")
        elif tool == "risk":
            lines.append(f"\n【風險 {'+'.join(r.get('tickers', []))}】（等權組合，日頻）")
            lines.append(
                f"  年化波動 {_pct(r.get('vol_ann'))}　"
                f"歷史 VaR95(1日) {_pct(r.get('var95'))}　"
                f"CVaR95 {_pct(r.get('cvar95'))}　"
                f"最大回撤 {_pct(r.get('max_dd'))}")
        elif tool == "screen":
            lines.append(f"\n【選股 {r.get('industry')}】（依 3 月動能排名，共 {r.get('scanned', '?')} 檔）")
            for x in (r.get("top") or []):
                lines.append(
                    f"  · {x.get('ticker')}：近3月 {_pct(x.get('return_3m'))}　"
                    f"年化波動 {_pct(x.get('ann_vol'))}　"
                    f"Sharpe {_num(x.get('sharpe'))}　RSI {_num(x.get('rsi'), dp=0)}")
        elif tool == "options":
            lines.append(f"\n【選擇權情緒 {r.get('ticker')}】")
            sc = r.get("score")
            lines.append(
                f"  情緒分數 {(('%+.2f' % sc) if sc is not None else '無資料')}"
                f"（{r.get('label', '')}）　"
                f"Put/Call(未平倉) {_num(r.get('pcr_oi'))}　"
                f"ATM 隱含波動 {_pct(r.get('atm_iv'))}　"
                f"偏斜(賣-買) {(('%+.1fpp' % (r['iv_skew']*100)) if r.get('iv_skew') is not None else '無資料')}")
            for n in (r.get("notes") or []):
                lines.append(f"  · {n}")
    return "\n".join(lines)


# ── CLI 自我測試（純邏輯）─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== might_need_tools ===")
    for q in ["AAPL 回測勝率如何", "半導體有哪些強勢股", "什麼是本益比",
              "這組合風險多大"]:
        print(f"  {might_need_tools(q)!s:5}  {q}")

    print("\n=== parse_plan ===")
    uni = {"AAPL", "MSFT", "NVDA"}
    inds = {"半導體", "軟體服務", "大型金融"}
    txt = ('```json\n{"tools":[{"tool":"backtest","args":{"ticker":"aapl"}},'
           '{"tool":"risk","args":{"tickers":["AAPL","msft","BAD$$"]}},'
           '{"tool":"screen","args":{"industry":"半導"}},'
           '{"tool":"unknown","args":{}}]}\n```')
    plan = parse_plan(txt, uni, inds)
    for p in plan:
        print("  ", p)
    assert plan[0] == {"tool": "backtest", "args": {"ticker": "AAPL"}}
    assert plan[1]["args"]["tickers"] == ["AAPL", "MSFT"]      # 壞代碼被濾
    assert plan[2]["args"]["industry"] == "半導體"              # 模糊對到白名單
    assert all(p["tool"] in KNOWN_TOOLS for p in plan)         # 未知工具被濾
    assert parse_plan("這題不需要工具，普通聊天") == []
    assert parse_plan('{"tools":[]}') == []

    print("\n=== format_tool_results ===")
    res = [
        {"tool": "backtest", "ok": True, "ticker": "AAPL", "bars": 252,
         "top": [{"rule": "RSI<30 反彈", "win_rate": 0.62, "profit_factor": 1.8,
                  "expectancy": 0.012, "trades": 14}]},
        {"tool": "risk", "ok": True, "tickers": ["AAPL", "MSFT"],
         "vol_ann": 0.28, "var95": -0.021, "cvar95": -0.031, "max_dd": -0.15},
        {"tool": "screen", "ok": True, "industry": "半導體", "scanned": 12,
         "top": [{"ticker": "NVDA", "return_3m": 0.22, "ann_vol": 0.42,
                  "sharpe": 1.6, "rsi": 61}]},
        {"tool": "backtest", "ok": False, "error": "抓不到資料"},
    ]
    print(format_tool_results(res))
    print("\n✅ assistant_tools 純邏輯測試通過")
