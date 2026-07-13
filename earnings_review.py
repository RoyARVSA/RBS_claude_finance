"""
earnings_review.py — 財報前瞻（Preview）與覆盤（Review）
RBS Finance Dashboard

方法論依據 Anthropic 官方 financial-services 的 earnings-preview 與
earnings-analysis skills，用免費資料實作可自動化的核心：
前瞻：共識 EPS/營收、近八季 beat 率、選擇權隱含預期波動（ATM IV 粗估）、
      歷史財報反應、多/基/空三情境框架、觀察清單
覆盤：實際 vs 共識（beat/miss + surprise%）、隔日股價反應、
      分析師評等動向（升降評）

模式自動判定：財報在 N 天內 → 前瞻；剛公布 14 天內 → 覆盤；否則兩者摘要。
純邏輯（beat_stats/expected_move/scenarios/reaction/texts）離線可測；
fetch 層走 yfinance + 現有 analyst_data / options_sentiment。非投資建議。
"""
from __future__ import annotations

import math

PREVIEW_WINDOW = 21      # 財報前 3 週內出前瞻
REVIEW_WINDOW = 14       # 財報後 2 週內出覆盤


# ── 純邏輯 ────────────────────────────────────────────────────────────────────

def expected_move_pct(atm_iv: float | None, days_to_event: int | None) -> float | None:
    """選擇權 ATM IV → 事件前粗估預期波動（±%）。IV×√(T/365)，僅供定錨。"""
    if not atm_iv or atm_iv <= 0 or days_to_event is None or days_to_event < 0:
        return None
    t = max(days_to_event, 1) / 365.0
    return round(atm_iv * math.sqrt(t) * 100, 1)


def build_scenarios(implied_move: float | None, beat_rate: float | None) -> list[dict]:
    """官方 Step 3 的三情境框架（免費數據版：以隱含波動定錨反應幅度）。"""
    mv = implied_move if implied_move is not None else 5.0
    hist = ""
    if beat_rate is not None:
        hist = f"（歷史 beat 率 {beat_rate:.0%}）"
    return [
        {"case": "多方", "what": f"營收+EPS 雙 beat 且上修指引{hist}",
         "watch": "指引上修幅度、利潤率方向、管理層語氣",
         "reaction": f"約 +{mv:.0f}% 或更多"},
        {"case": "基準", "what": "符合預期、指引持平",
         "watch": "細項品質（一次性 vs 經常性）、隱含波動釋放",
         "reaction": f"±{max(mv * 0.4, 1):.0f}% 內震盪"},
        {"case": "空方", "what": "miss 或下修指引（beat 也可能因指引差而跌）",
         "watch": "下修的歸因：需求 vs 供給 vs 一次性",
         "reaction": f"約 -{mv:.0f}% 或更多"},
    ]


def reaction_after(closes: dict[str, float], event_date: str) -> float | None:
    """closes：{ISO日期: 收盤}。回財報後第一個交易日 vs 前一日的 %。"""
    if not closes or not event_date:
        return None
    days = sorted(closes)
    prior = [d for d in days if d <= event_date]
    after = [d for d in days if d > event_date]
    if not prior or not after:
        return None
    p0, p1 = closes[prior[-1]], closes[after[0]]
    if not p0 or p0 <= 0:
        return None
    return round((p1 / p0 - 1) * 100, 2)


def preview_text(d: dict) -> str:
    """前瞻卡（官方 Step 5 的單頁結構，Telegram 版）。"""
    tk = d.get("ticker", "?")
    lines = [f"🔭 *{tk} 財報前瞻*　{d.get('earnings_date', '日期未定')}"
             + (f"（{d['days_to']} 天後）" if d.get("days_to") is not None else "")]
    est = []
    if d.get("eps_est") is not None:
        est.append(f"EPS 共識 {d['eps_est']:.2f}")
    if d.get("rev_est"):
        est.append(f"營收共識 {d['rev_est'] / 1e9:.2f}B")
    if est:
        lines.append("📊 " + "｜".join(est))
    if d.get("beat_rate") is not None:
        lines.append(f"🎯 近 {d.get('beat_n', 0)} 季 beat 率 {d['beat_rate']:.0%}"
                     + (f"（平均 surprise {d['avg_surprise']:+.1%}）"
                        if d.get("avg_surprise") is not None else ""))
    if d.get("implied_move") is not None:
        lines.append(f"🌪 選擇權隱含預期波動 ±{d['implied_move']:.1f}%（ATM IV 粗估）")
    if d.get("past_reactions"):
        rs = "、".join(f"{r:+.1f}%" for r in d["past_reactions"][:4])
        lines.append(f"📈 近幾次財報隔日反應：{rs}")
    lines.append("*三情境框架*")
    for s in d.get("scenarios", []):
        lines.append(f"　{s['case']}：{s['what']} → {s['reaction']}")
    if d.get("watch"):
        lines.append("*觀察重點*：" + "；".join(d["watch"]))
    lines.append("_共識估計會變動；隱含波動是市場定價非預測。非投資建議。_")
    return "\n".join(lines)


def review_text(d: dict) -> str:
    """覆盤卡（beat/miss + 反應 + 評等動向）。"""
    tk = d.get("ticker", "?")
    lines = [f"📋 *{tk} 財報覆盤*　{d.get('report_date', '?')}"]
    if d.get("eps_actual") is not None and d.get("eps_est") is not None:
        beat = d["eps_actual"] > d["eps_est"]
        icon = ("✅ Beat" if beat
                else ("➖ 符合預期" if d["eps_actual"] == d["eps_est"] else "❌ Miss"))
        sur = (f"（surprise {d['surprise_pct']:+.1f}%）"
               if d.get("surprise_pct") is not None else "")
        lines.append(f"{icon}：EPS {d['eps_actual']:.2f} vs 共識 {d['eps_est']:.2f}{sur}")
    if d.get("reaction_pct") is not None:
        emo = "🟢" if d["reaction_pct"] > 0 else "🔴"
        lines.append(f"{emo} 隔日股價反應 {d['reaction_pct']:+.2f}%")
        if d.get("eps_actual") is not None and d.get("eps_est") is not None:
            beat = d["eps_actual"] > d["eps_est"]
            if beat and d["reaction_pct"] < -2:
                lines.append("⚠️ Beat 卻大跌——市場在意的是指引/品質，去讀法說重點")
            elif not beat and d["reaction_pct"] > 2:
                lines.append("💡 Miss 卻上漲——利空出盡或指引優於恐懼")
    if d.get("upgrades"):
        ups = "；".join(f"{u.get('firm','?')} {u.get('action','')}→{u.get('to','')}"
                        for u in d["upgrades"][:3])
        lines.append(f"🏦 財報後評等動向：{ups}")
    lines.append("_覆盤重點不是漲跌，是「當初的預期錯在哪」。非投資建議。_")
    return "\n".join(lines)


def decide_mode(days_to_next: int | None, days_since_last: int | None) -> str:
    """preview / review / both。"""
    if days_to_next is not None and 0 <= days_to_next <= PREVIEW_WINDOW:
        return "preview"
    if days_since_last is not None and 0 <= days_since_last <= REVIEW_WINDOW:
        return "review"
    return "both"


# ── 抓取層（需網路）───────────────────────────────────────────────────────────

def fetch_data(ticker: str) -> dict | None:
    """一次抓齊前瞻+覆盤所需資料。"""
    try:
        import datetime as _dt

        import yfinance as yf
        tk = yf.Ticker(ticker)
        today = _dt.date.today()
        out: dict = {"ticker": ticker.upper()}

        # 下次財報日
        try:
            from fundamentals import next_earnings_date
            ed = next_earnings_date(ticker)
            if ed:
                out["earnings_date"] = ed.isoformat()
                out["days_to"] = (ed - today).days
        except Exception:
            pass

        # 共識估計（yfinance earnings_estimate/revenue_estimate；0q=本季）
        try:
            ee = tk.earnings_estimate
            if ee is not None and "avg" in ee.columns and "0q" in ee.index:
                out["eps_est"] = float(ee.loc["0q", "avg"])
        except Exception:
            pass
        try:
            re_ = tk.revenue_estimate
            if re_ is not None and "avg" in re_.columns and "0q" in re_.index:
                out["rev_est"] = float(re_.loc["0q", "avg"])
        except Exception:
            pass

        # 歷史 surprise / beat 率 + 上次財報
        past_dates: list = []
        try:
            df = tk.get_earnings_dates(limit=12)
            if df is not None and len(df):
                rows = []
                for idx, r in df.iterrows():
                    dt_ = idx.date() if hasattr(idx, "date") else idx
                    est = r.get("EPS Estimate")
                    act = r.get("Reported EPS")
                    if act is not None and not (isinstance(act, float) and math.isnan(act)):
                        rows.append({"date": dt_, "estimate": est, "actual": act})
                rows.sort(key=lambda x: x["date"], reverse=True)
                if rows:
                    from analyst_data import summarize_surprises
                    s = summarize_surprises([{"estimate": r["estimate"],
                                              "actual": r["actual"]} for r in rows[:8]])
                    if s:
                        out["beat_rate"] = s["beat_rate"]
                        out["beat_n"] = s["n"]
                        out["avg_surprise"] = s["avg_surprise"]
                    last = rows[0]
                    out["report_date"] = last["date"].isoformat()
                    out["days_since"] = (today - last["date"]).days
                    out["eps_actual"] = float(last["actual"])
                    if last["estimate"] is not None and not (
                            isinstance(last["estimate"], float) and math.isnan(last["estimate"])):
                        est_v = float(last["estimate"])
                        if est_v != 0:
                            out["surprise_pct"] = (out["eps_actual"] - est_v) / abs(est_v) * 100
                        out["eps_est_review"] = est_v
                    past_dates = [r["date"] for r in rows[:5]]
        except Exception:
            pass

        # 歷史反應 + 上次財報隔日反應（日線）
        # 注意：reaction_pct 只能來自「本次」財報日——財報昨晚剛公布、隔日還沒
        # 收盤時 reaction_after 回 None，此時絕不可拿上一季的反應冒充（會連動
        # 觸發錯誤的「Beat 卻大跌」評語）。
        try:
            hist = tk.history(period="1y")
            if hist is not None and len(hist):
                closes = {i.date().isoformat(): float(v)
                          for i, v in hist["Close"].items()}
                reacts = [r_ for r_ in (reaction_after(closes, dt_.isoformat())
                                        for dt_ in past_dates) if r_ is not None]
                if reacts:
                    out["past_reactions"] = reacts
                if out.get("report_date"):
                    r0 = reaction_after(closes, out["report_date"])
                    if r0 is not None:
                        out["reaction_pct"] = r0
        except Exception:
            pass

        # 選擇權隱含波動 → 預期波動
        try:
            from options_sentiment import fetch_options
            opt = fetch_options(ticker, max_expiries=1)
            if opt and opt.get("atm_iv"):
                out["implied_move"] = expected_move_pct(opt["atm_iv"],
                                                        out.get("days_to", 7))
        except Exception:
            pass

        # 財報後評等動向
        try:
            from analyst_data import fetch_analyst
            an = fetch_analyst(ticker)
            if an and an.get("upgrades"):
                out["upgrades"] = an["upgrades"]
        except Exception:
            pass

        out["scenarios"] = build_scenarios(out.get("implied_move"),
                                           out.get("beat_rate"))
        out["watch"] = ["指引 vs 共識（最會動股價）", "利潤率方向",
                        "細項品質：一次性 vs 經常性"]
        if not out.get("earnings_date") and not out.get("report_date"):
            return None
        return out
    except Exception:
        return None


def run(ticker: str) -> tuple[str, dict] | None:
    """端到端：抓資料 → 判模式。回 (mode, data) 或 None。"""
    d = fetch_data(ticker)
    if not d:
        return None
    mode = decide_mode(d.get("days_to"), d.get("days_since"))
    if mode == "review" and d.get("eps_est_review") is not None:
        d["eps_est"] = d["eps_est_review"]      # 覆盤用「當季」共識而非下季
    return mode, d


def full_text(mode: str, d: dict) -> str:
    if mode == "preview":
        return preview_text(d)
    if mode == "review":
        return review_text(d)
    parts = []
    if d.get("earnings_date"):
        parts.append(f"🔭 下次財報：{d['earnings_date']}"
                     + (f"（{d['days_to']} 天後）" if d.get("days_to") is not None else "")
                     + "——3 週內會有完整前瞻")
    if d.get("report_date"):
        d2 = dict(d)
        d2["eps_est"] = d.get("eps_est_review", d.get("eps_est"))
        parts.append(review_text(d2))
    return "\n\n".join(parts) if parts else "查無財報資料"


# ── CLI 自我測試（離線純邏輯）─────────────────────────────────────────────────

if __name__ == "__main__":
    # 預期波動：IV 60%、7 天 → 60×√(7/365) ≈ 8.3%
    em = expected_move_pct(0.60, 7)
    assert em is not None and abs(em - 60 * math.sqrt(7 / 365)) < 0.11, em
    assert expected_move_pct(None, 7) is None and expected_move_pct(0.5, None) is None
    assert expected_move_pct(0.5, 0) == expected_move_pct(0.5, 1)   # 當日下限 1 天

    sc = build_scenarios(8.3, 0.75)
    assert len(sc) == 3 and "+8%" in sc[0]["reaction"] and "75%" in sc[0]["what"]
    sc2 = build_scenarios(None, None)
    assert "+5%" in sc2[0]["reaction"]                    # 無 IV 用預設定錨

    # 隔日反應：財報日 7/8（盤後），7/8 收 100 → 7/9 收 106
    closes = {"2026-07-07": 98.0, "2026-07-08": 100.0, "2026-07-09": 106.0}
    assert reaction_after(closes, "2026-07-08") == 6.0
    assert reaction_after(closes, "2026-07-09") is None   # 無後續交易日
    assert reaction_after({}, "2026-07-08") is None

    assert decide_mode(5, None) == "preview"
    assert decide_mode(40, 3) == "review"
    assert decide_mode(40, 40) == "both"
    assert decide_mode(None, None) == "both"
    assert decide_mode(0, 0) == "preview"                 # 今天出財報 → 前瞻優先

    pv = preview_text({"ticker": "NVDA", "earnings_date": "2026-08-27", "days_to": 45,
                       "eps_est": 1.25, "rev_est": 46.5e9, "beat_rate": 0.875,
                       "beat_n": 8, "avg_surprise": 0.06, "implied_move": 8.3,
                       "past_reactions": [9.3, -6.1, 12.8, 0.5],
                       "scenarios": build_scenarios(8.3, 0.875),
                       "watch": ["指引 vs 共識"]})
    assert "財報前瞻" in pv and "beat 率 88%" in pv and "±8.3%" in pv and "非投資建議" in pv

    rv = review_text({"ticker": "AAPL", "report_date": "2026-07-01", "eps_actual": 2.10,
                      "eps_est": 2.00, "surprise_pct": 5.0, "reaction_pct": -3.4,
                      "upgrades": [{"firm": "MS", "action": "up", "to": "Overweight"}]})
    assert "Beat" in rv and "-3.40%" in rv and "Beat 卻大跌" in rv, rv
    rv2 = review_text({"ticker": "X", "eps_actual": 1.0, "eps_est": 1.2,
                       "surprise_pct": -16.7, "reaction_pct": 4.0})
    assert "Miss 卻上漲" in rv2

    ft = full_text("both", {"ticker": "T", "earnings_date": "2026-09-01", "days_to": 50,
                            "report_date": "2026-06-01", "eps_actual": 1.0,
                            "eps_est_review": 0.9, "reaction_pct": 2.0})
    assert "下次財報" in ft and "覆盤" in ft

    print(f"✅ earnings_review 離線自我測試通過（預期波動 ±{em}%、三情境、覆盤標語全過）")
