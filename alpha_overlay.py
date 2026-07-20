"""
alpha_overlay.py – Alpha 層主動資訊布建（純邏輯與抓取分離、離線可測）

把 repo 既有的資訊模組整合成進場前的「資訊疊加層」，餵給 trade_engine：
  • SEC 內部人（sec_insider）：cluster buy 加分、賣超減分
  • 選擇權情緒（options_sentiment）：PCR(OI) + IV 偏斜 → 傾斜分數
  • 空單面（short_data / yfinance info）：短倉占流通 ≥15% → 降評
  • 財報日（scan_signals earnings_cache）：財報前 N 天禁新倉（事件風險迴避）
  • 雙恐懼貪婪（sentiment_fg）：同時極度貪婪 → 新倉風險減半（帳戶級 size_mult）

輸出三件事：
  1. per-symbol score delta（夾在 ±max_abs_delta，微調不翻案——alpha 主體仍是原評分）
  2. per-symbol no_entry veto（只擋新買進與加碼；出場機制照常）
  3. 帳戶級 size_mult（乘在 risk_pct 上）

設計：
  • 抓取有 12 小時 TTL 快取（state["alpha_cache"]），每輪最多刷新 refresh_per_run 檔
    （最舊優先輪替）——15 分鐘 cron 絕不因資訊抓取而超時
  • fetchers 可注入（測試用假抓取器離線驗證快取/輪替/容錯行為）
  • 參數可用 /set ao_<參數> 覆蓋（如 /set ao_earnings_veto_days 5）
"""

from __future__ import annotations

from datetime import datetime, timezone

OVERLAY_DEFAULTS = {
    "earnings_veto_days":  3,     # 財報前 N 天（含當日）禁新倉/加碼
    "insider_w":           0.15,  # 內部人分數(-1..1) × 權重
    "cluster_bonus":       0.05,  # cluster buy（≥2 位內部人買）額外加分
    "options_w":           0.10,  # 選擇權情緒分數(-1..1) × 權重
    "short_hi":            0.15,  # 短倉占流通 ≥ 此值 → 降評
    "short_damp":          -0.10,
    "max_abs_delta":       0.25,  # delta 總量上限（資訊只微調，不翻案）
    "greed_size_mult":     0.5,   # 雙極度貪婪 → 新倉風險減半
    "fear_size_mult":      1.0,   # 雙極度恐懼 → 不縮倉（歷史上常近底部），僅提示
    "ttl_hours":           12,    # 個股資訊快取壽命
    "fg_ttl_hours":        2,     # 恐貪指數快取壽命
    "refresh_per_run":     4,     # 每輪最多刷新幾檔（限額輪替，防 cron 超時）
    "insider_max_filings": 8,     # 每檔最多抓幾份 Form 4（控制 SEC 請求數）
}


def _cfg(config: dict | None) -> dict:
    return {**OVERLAY_DEFAULTS, **(config or {})}


def _hours_since(ts: str | None, now: str) -> float:
    """兩個 ISO 時間戳的小時差。壞值回 inf（視為過期，觸發重抓）。"""
    try:
        a = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        b = datetime.fromisoformat(str(now).replace("Z", "+00:00"))
        return (b - a).total_seconds() / 3600.0
    except Exception:
        return float("inf")


# ── 純邏輯：單檔 overlay ──────────────────────────────────────────────────

def compute_symbol_overlay(inputs: dict, config: dict | None = None) -> dict:
    """
    inputs: {insider_score, cluster_buy, opt_score, short_pct_float, days_to_earnings}
            （任一欄可缺/None → 該來源跳過）
    回 {"delta", "no_entry", "reasons": [...]}
    """
    cfg = _cfg(config)
    delta, reasons, no_entry = 0.0, [], False

    dte = inputs.get("days_to_earnings")
    if dte is not None and 0 <= int(dte) <= int(cfg["earnings_veto_days"]):
        no_entry = True
        reasons.append(f"財報 {int(dte)} 天內 → 禁新倉/加碼（事件風險迴避）")

    ins = inputs.get("insider_score")
    if ins is not None:
        d = float(ins) * float(cfg["insider_w"])
        if inputs.get("cluster_buy") and float(ins) > 0:
            d += float(cfg["cluster_bonus"])
            reasons.append(f"內部人 cluster buy（{d:+.2f}）")
        elif abs(d) >= 0.005:
            reasons.append(f"內部人情緒 {float(ins):+.2f}（{d:+.2f}）")
        delta += d

    opt = inputs.get("opt_score")
    if opt is not None:
        d = float(opt) * float(cfg["options_w"])
        if abs(d) >= 0.005:
            reasons.append(f"選擇權情緒 {float(opt):+.2f}（{d:+.2f}）")
        delta += d

    spf = inputs.get("short_pct_float")
    if spf is not None and float(spf) >= float(cfg["short_hi"]):
        delta += float(cfg["short_damp"])
        reasons.append(f"短倉占流通 {float(spf):.0%} ≥ {float(cfg['short_hi']):.0%}"
                       f"（{float(cfg['short_damp']):+.2f}）")

    cap = abs(float(cfg["max_abs_delta"]))
    delta = max(-cap, min(cap, delta))
    return {"delta": round(delta, 3), "no_entry": no_entry, "reasons": reasons}


# ── 純邏輯：帳戶級 overlay（雙恐懼貪婪）─────────────────────────────────────

def account_overlay(cnn_score: float | None, crypto_score: float | None,
                    config: dict | None = None) -> dict:
    """回 {"size_mult", "reasons"}。size_mult 乘在新倉 risk_pct 上。"""
    cfg = _cfg(config)
    out = {"size_mult": 1.0, "reasons": []}
    try:
        from sentiment_fg import dual_signal
        sig = dual_signal(cnn_score, crypto_score)
    except Exception:
        return out
    if sig.get("both_extreme_greed"):
        out["size_mult"] = float(cfg["greed_size_mult"])
        out["reasons"].append(
            f"雙極度貪婪（美股 {cnn_score:.0f}／加密 {crypto_score:.0f}）"
            f"→ 新倉風險降為 {out['size_mult']:.0%}")
    elif sig.get("both_extreme_fear"):
        out["size_mult"] = float(cfg["fear_size_mult"])
        out["reasons"].append(
            f"雙極度恐懼（美股 {cnn_score:.0f}／加密 {crypto_score:.0f}）"
            "→ 統計上常近短期底部，維持正常倉位（反向參考）")
    return out


# ── 純邏輯：套用到 scored ─────────────────────────────────────────────────

def apply_overlay(scored: list[dict], overlays: dict) -> tuple[list[dict], list[str]]:
    """
    回 (新 scored 清單, notes)。原 scored 不改；新清單的 score 加上 delta、
    帶 no_entry 旗標（trade_engine 只用它擋新買/加碼）。
    """
    out, notes = [], []
    for s in scored:
        s2 = dict(s)
        ov = overlays.get(s.get("ticker"))
        if ov:
            base = float(s.get("score") or 0)      # score=None 防炸
            if ov.get("delta"):
                s2["score"] = round(base + float(ov["delta"]), 3)
            if ov.get("no_entry"):
                s2["no_entry"] = True
            if ov.get("delta") or ov.get("no_entry"):
                why = "；".join(ov.get("reasons") or [])
                notes.append(f"🧠 {s['ticker']} 評分 {base:+.2f}"
                             f"→{float(s2.get('score') or 0):+.2f}"
                             f"{'（禁新倉）' if ov.get('no_entry') else ''} {why}".rstrip())
        out.append(s2)
    return out, notes


# ── 快取刷新（限額輪替；fetchers 可注入供離線測試）──────────────────────────

def _default_fetchers(cfg: dict) -> dict:
    def _insider(sym):
        import sec_insider
        r = sec_insider.fetch_insider(sym, max_filings=int(cfg["insider_max_filings"]))
        return {"insider_score": r.get("score"),
                "cluster_buy": bool(r.get("cluster_buy"))} if r else {}

    def _options(sym):
        import options_sentiment as osent
        summ = osent.fetch_options(sym)
        if not summ:
            return {}
        return {"opt_score": osent.sentiment(summ).get("score")}

    def _short(sym):
        import yfinance as yf
        from short_data import short_summary
        try:
            info = dict(yf.Ticker(sym).info or {})
        except Exception:
            return {}
        return {"short_pct_float": short_summary(None, info, None).get("short_pct_float")}

    return {"insider": _insider, "options": _options, "short": _short}


def refresh_cache(state: dict, symbols: list[str], now: str,
                  config: dict | None = None, fetchers: dict | None = None) -> None:
    """
    就地更新 state["alpha_cache"]：{sym: {"ts", "inputs": {...}}}。
    只挑「缺快取或過期」的標的，最舊優先，每輪最多 refresh_per_run 檔。
    個別抓取器失敗只缺該欄位（存部分結果），不整檔報廢。
    """
    cfg = _cfg(config)
    cache = state.setdefault("alpha_cache", {})
    symbols = list(dict.fromkeys(symbols))               # 去重（重複會浪費刷新名額）
    if symbols:
        # 移除已不在清單的標的（避免快取只增不減）；空清單不剪——
        # 一輪抓價失敗不該把整個快取炸掉再花 3 輪重建
        for k in [k for k in cache if k not in symbols]:
            del cache[k]

    stale = []
    for s in symbols:
        ent = cache.get(s) or {}
        age = _hours_since(ent.get("ts"), now)
        ttl = float(cfg["ttl_hours"])
        if ent and not ent.get("inputs"):
            # 上次全部抓取失敗（空 inputs）→ 用短 TTL 提早重試，不盲 12 小時
            ttl = min(ttl, float(cfg["fg_ttl_hours"]))
        if age >= ttl:
            stale.append((s, age))
    stale.sort(key=lambda x: -x[1])                      # 最舊優先
    picked = [s for s, _ in stale[: int(cfg["refresh_per_run"])]]
    if not picked:
        return
    fs = fetchers or _default_fetchers(cfg)
    for sym in picked:
        inputs = {}
        for name, fn in fs.items():
            try:
                inputs.update(fn(sym) or {})
            except Exception as e:
                print(f"alpha_overlay: {sym} {name} 抓取失敗 {e}")
        cache[sym] = {"ts": now, "inputs": inputs}


def _refresh_fg(state: dict, now: str, cfg: dict, fg_fetch=None) -> dict:
    """恐貪指數快取（state["alpha_fg"]）。回 {"cnn": score|None, "crypto": score|None}。"""
    fg = state.get("alpha_fg") or {}
    if _hours_since(fg.get("ts"), now) >= float(cfg["fg_ttl_hours"]):
        try:
            if fg_fetch is None:
                from sentiment_fg import fetch_all
                fg_fetch = fetch_all
            r = fg_fetch() or {}
            fg = {"ts": now,
                  "cnn": (r.get("cnn") or {}).get("score"),
                  "crypto": (r.get("crypto") or {}).get("score")}
            state["alpha_fg"] = fg
        except Exception as e:
            print(f"alpha_overlay: 恐貪抓取失敗 {e}")
    return {"cnn": fg.get("cnn"), "crypto": fg.get("crypto")}


def _earnings_days_map(state: dict, today: str) -> dict:
    """從 scan_signals 維護的 earnings_cache 讀出 {sym: 距財報天數}（不抓網路）。"""
    from datetime import date
    out = {}
    try:
        t = date.fromisoformat(str(today)[:10])
    except Exception:
        return out
    for sym, c in (state.get("earnings_cache") or {}).items():
        ed = (c or {}).get("earnings")
        if not ed:
            continue
        try:
            out[sym] = (date.fromisoformat(ed) - t).days
        except Exception:
            continue
    return out


# ── 進入點（bot 端呼叫）───────────────────────────────────────────────────

def enrich(state: dict, scored: list[dict], thresholds: dict | None = None,
           now: str | None = None, fetchers: dict | None = None,
           fg_fetch=None) -> tuple[list[dict], list[str], float]:
    """
    回 (調整後 scored, notes, size_mult)。就地更新 state 的快取；
    呼叫端負責 save_state。now 不給則取當下 UTC（測試時注入固定值）。
    """
    th = thresholds or {}
    cfg = _cfg({k: th[f"ao_{k}"] for k in OVERLAY_DEFAULTS if f"ao_{k}" in th})
    now = now or datetime.now(timezone.utc).isoformat()

    symbols = list(dict.fromkeys(s["ticker"] for s in scored if s.get("ticker")))
    refresh_cache(state, symbols, now, cfg, fetchers)
    edays = _earnings_days_map(state, now[:10])

    overlays = {}
    for sym in symbols:
        inputs = dict(((state.get("alpha_cache") or {}).get(sym) or {}).get("inputs") or {})
        if sym in edays:
            inputs["days_to_earnings"] = edays[sym]
        ov = compute_symbol_overlay(inputs, cfg)
        if ov["delta"] or ov["no_entry"]:
            overlays[sym] = ov

    scored2, notes = apply_overlay(scored, overlays)
    fg = _refresh_fg(state, now, cfg, fg_fetch)
    acct = account_overlay(fg.get("cnn"), fg.get("crypto"), config=cfg)
    notes.extend(f"🎭 {r}" for r in acct["reasons"])
    return scored2, notes, float(acct["size_mult"])


def overlay_text(state: dict, today: str | None = None) -> str:
    """/alpha 指令：顯示快取中每檔的資訊疊加現況（Telegram legacy Markdown 單 *）。
    參數與 enrich 同源（讀 thresholds 的 ao_*），否則使用者 /set 後看到的現況是假的。"""
    today = (today or datetime.now(timezone.utc).isoformat())[:10]
    th = state.get("thresholds") or {}
    cfg = _cfg({k: th[f"ao_{k}"] for k in OVERLAY_DEFAULTS if f"ao_{k}" in th})
    cache = state.get("alpha_cache") or {}
    edays = _earnings_days_map(state, today)
    lines = ["🧠 *Alpha 資訊疊加層*（進場評分微調；出場不受影響）"]
    if not cache and not edays:
        return lines[0] + "\n尚無快取——開 /autotrade on 後每輪自動布建（每輪最多 4 檔、12 小時更新）"
    for sym in sorted(set(cache) | set(edays)):
        inputs = dict((cache.get(sym) or {}).get("inputs") or {})
        if sym in edays:
            inputs["days_to_earnings"] = edays[sym]
        ov = compute_symbol_overlay(inputs, cfg)
        flag = "⛔" if ov["no_entry"] else ("➕" if ov["delta"] > 0 else ("➖" if ov["delta"] < 0 else "・"))
        why = "；".join(ov["reasons"]) if ov["reasons"] else "無顯著訊息"
        lines.append(f"{flag} *{sym}* Δ{ov['delta']:+.2f} — {why}")
    fg = state.get("alpha_fg") or {}
    if fg.get("cnn") is not None or fg.get("crypto") is not None:
        acct = account_overlay(fg.get("cnn"), fg.get("crypto"), config=cfg)

        def _s(v):
            return f"{v:.0f}" if isinstance(v, (int, float)) else "－"
        seg = f"恐貪：美股 {_s(fg.get('cnn'))}／加密 {_s(fg.get('crypto'))}"
        if acct["reasons"]:
            seg += "\n" + "\n".join(acct["reasons"])
        lines.append(seg)
    lines.append("_參數 /set ao_earnings_veto_days 5 等；非投資建議_")
    return "\n".join(lines)


# ── 自我測試 ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    NOW = "2026-07-20T14:00:00+00:00"

    # 1) 單檔 overlay：cluster buy + 選擇權偏多 → 正 delta；財報近 → veto
    ov = compute_symbol_overlay({"insider_score": 1.0, "cluster_buy": True,
                                 "opt_score": 0.5, "short_pct_float": 0.03})
    assert abs(ov["delta"] - 0.25) < 1e-9 and not ov["no_entry"], ov   # 0.15+0.05+0.05=0.25
    ov = compute_symbol_overlay({"days_to_earnings": 2})
    assert ov["no_entry"] and ov["delta"] == 0, ov
    ov = compute_symbol_overlay({"days_to_earnings": 4})
    assert not ov["no_entry"], ov
    print("✅ 1 單檔 overlay（cluster/選擇權/財報 veto）")

    # 2) 空單降評 + delta 夾制
    ov = compute_symbol_overlay({"short_pct_float": 0.22})
    assert ov["delta"] == -0.10, ov
    ov = compute_symbol_overlay({"insider_score": -1.0, "opt_score": -1.0,
                                 "short_pct_float": 0.30})
    assert ov["delta"] == -0.25, ov          # -0.15-0.10-0.10 → 夾在 -0.25
    print("✅ 2 空單降評 + 夾制 ±0.25")

    # 3) 帳戶級：雙極貪 → 減半；雙極恐 → 維持 1.0；缺資料 → 1.0
    a = account_overlay(80, 82)
    assert a["size_mult"] == 0.5 and a["reasons"], a
    a = account_overlay(20, 15)
    assert a["size_mult"] == 1.0 and a["reasons"], a
    a = account_overlay(None, 50)
    assert a["size_mult"] == 1.0 and not a["reasons"], a
    print("✅ 3 恐貪帳戶級 size_mult")

    # 4) 快取輪替：TTL、每輪限額、最舊優先、部分失敗容錯、清單修剪
    calls = []

    def fk_ins(sym):
        calls.append(sym)
        if sym == "BAD":
            raise RuntimeError("boom")
        return {"insider_score": 0.5, "cluster_buy": False}

    def fk_opt(sym):
        return {"opt_score": -0.2}

    st = {"alpha_cache": {
        "OLD": {"ts": "2026-07-19T00:00:00+00:00", "inputs": {}},   # 38h → 最舊
        "MID": {"ts": "2026-07-20T00:00:00+00:00", "inputs": {}},   # 14h
        "FRESH": {"ts": "2026-07-20T10:00:00+00:00", "inputs": {"opt_score": 0.9}},  # 4h
        "GONE": {"ts": "2026-07-01T00:00:00+00:00", "inputs": {}},  # 不在清單 → 修剪
    }}
    syms = ["OLD", "MID", "FRESH", "BAD", "NEW1", "NEW2"]
    refresh_cache(st, syms, NOW, {"refresh_per_run": 3},
                  fetchers={"insider": fk_ins, "options": fk_opt})
    assert "GONE" not in st["alpha_cache"]
    assert st["alpha_cache"]["FRESH"]["inputs"] == {"opt_score": 0.9}   # 未動
    refreshed = {s for s in syms if st["alpha_cache"].get(s, {}).get("ts") == NOW}
    # 限額 3；「從未抓過」(age=inf) 優先於「過期最久」→ 本輪 BAD/NEW1/NEW2，OLD 下輪
    assert refreshed == {"BAD", "NEW1", "NEW2"}, refreshed
    assert st["alpha_cache"]["OLD"]["ts"] != NOW and st["alpha_cache"]["MID"]["ts"] != NOW
    if "BAD" in refreshed:                                              # 失敗只缺該欄
        assert st["alpha_cache"]["BAD"]["inputs"] == {"opt_score": -0.2}
    print("✅ 4 快取輪替（TTL/限額/最舊優先/容錯/修剪）")

    # 5) enrich 端到端（假抓取器 + 假 F&G + 財報快取）
    st = {"earnings_cache": {"NVDA": {"checked": "2026-07-20", "earnings": "2026-07-22"}},
          "thresholds": {}}
    scored = [{"ticker": "NVDA", "score": 0.8, "price": 100.0},
              {"ticker": "AAPL", "score": 0.55, "price": 200.0}]
    s2, notes, mult = enrich(
        st, scored, {}, now=NOW,
        fetchers={"insider": lambda s: {"insider_score": 1.0, "cluster_buy": True}},
        fg_fetch=lambda: {"cnn": {"score": 80}, "crypto": {"score": 90}})
    nv = next(s for s in s2 if s["ticker"] == "NVDA")
    ap = next(s for s in s2 if s["ticker"] == "AAPL")
    assert nv.get("no_entry") is True, nv                       # 財報 2 天 → veto
    assert abs(ap["score"] - 0.75) < 1e-9, ap                   # 0.55+0.20 cluster
    assert mult == 0.5 and any("🎭" in n for n in notes), (mult, notes)
    assert scored[0].get("no_entry") is None and scored[1]["score"] == 0.55  # 原件不改
    print("✅ 5 enrich 端到端（veto/加分/縮倉/不改原件）")

    # 6) /alpha 文字（不炸、含 veto 標記）
    txt = overlay_text(st, today="2026-07-20")
    assert "⛔" in txt and "NVDA" in txt, txt
    print("✅ 6 overlay_text")
    print("\n" + txt)

    # 7) 全空 inputs 快取（上次抓取全敗）→ 短 TTL 提早重試；空清單不剪快取
    st = {"alpha_cache": {
        "BLIND": {"ts": "2026-07-20T10:00:00+00:00", "inputs": {}},          # 4h、空
        "OK4H": {"ts": "2026-07-20T10:00:00+00:00", "inputs": {"opt_score": 0.1}},
    }}
    refresh_cache(st, ["BLIND", "OK4H"], NOW, {"refresh_per_run": 4},
                  fetchers={"f": lambda s: {"opt_score": 0.5}})
    assert st["alpha_cache"]["BLIND"]["ts"] == NOW      # 空 inputs 4h → 重試
    assert st["alpha_cache"]["OK4H"]["ts"] != NOW       # 正常快取 4h < 12h → 不動
    refresh_cache(st, [], NOW, None, fetchers={"f": lambda s: {}})
    assert len(st["alpha_cache"]) == 2                  # 空清單不剪
    print("✅ 7 空 inputs 短 TTL 重試 + 空清單不剪快取")

    # 8) /alpha 顯示與 enrich 同源：/set ao_earnings_veto_days 10 → 6 天也顯示 ⛔
    st = {"thresholds": {"ao_earnings_veto_days": 10.0},
          "earnings_cache": {"LATE": {"checked": "2026-07-20",
                                      "earnings": "2026-07-26"}},
          "alpha_cache": {}}
    txt = overlay_text(st, today="2026-07-20")
    assert "⛔" in txt and "LATE" in txt, txt
    print("✅ 8 overlay_text 吃 ao_ 覆蓋參數")

    print("\nalpha_overlay selftest OK ✅")
