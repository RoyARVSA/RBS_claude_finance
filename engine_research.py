"""
engine_research.py – 引擎回測研究執行器（GitHub Actions `engine_research.yml`，workflow_dispatch）

為什麼要有它：開發環境連不到 Yahoo（proxy），/engtest 只能請使用者在 Telegram 傳指令再貼回結果。
這個執行器在 Actions（有行情網路）上跑同一套 engine_backtest，結果寫進 job summary 與
`research-results` 分支的 `results/`，開發端用 `gh api` 觸發與讀回——不需要使用者轉傳。

    python engine_research.py --cmd "pit 20 5 2y; try trail_tighten_r=off scale_out_r=off pit 20 5" --out research.md
    python engine_research.py --selftest          # 離線（合成行情）自測

子指令與 Bot 的 /engtest 相同（解析共用 engine_backtest.parse_engtest_args）：
  run [期間]、opt [期間] [entry|loose]、pit [檔數] [組數] [期間]、try 參數=值 … [期間] [pit …]；多個用「;」分隔。

隱私（公開 repo、公開日誌）：只用公開資料——觀察清單（watchlist_state.json 明文欄位）、當時 S&P 500 成分、
**程式預設參數**。不讀加密的個人設定（thresholds／calibration／engine／eng_opt），所以「現行」＝程式預設，
不是使用者用 /set 或 apply 改過的值；評分也不含個人校準權重。opt 不會 apply、不寫 state。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_CMDS = 6                     # 一次 dispatch 最多跑幾個子指令（防誤觸長時間佔用 runner）


def baseline_config() -> dict:
    """「現行」＝程式預設：與 scan_signals._engine_base_config({}) + _engtest_baseline 的預設值一致。"""
    return {"buy_threshold": 0.5, "exit_threshold": -0.2, "max_positions": 10,
            "max_position_pct": 0.15, "risk_pct": 0.01,
            "regime_filter": True, "event_blackout": True, "event_blackout_days": 1, "val_enabled": False}


def legacy_config() -> dict:
    b = baseline_config()
    return {k: b[k] for k in ("buy_threshold", "exit_threshold", "max_positions", "max_position_pct", "risk_pct")}


def load_watchlist(path: str | None = None) -> list[str]:
    """觀察清單（state 明文欄位）。檔案不存在或格式不對回 []。"""
    try:
        with open(path or os.path.join(HERE, "watchlist_state.json"), encoding="utf-8") as f:
            wl = json.load(f).get("watchlist")
        return [str(t).upper() for t in wl if isinstance(t, str)] if isinstance(wl, list) else []
    except Exception:
        return []


def split_cmds(cmd: str) -> list[str]:
    return [c.strip() for c in (cmd or "").split(";") if c.strip()][:MAX_CMDS]


def run_one(cmd: str, watchlist: list[str], fetch_fn=None, periods: dict | None = None,
            today: str | None = None) -> str:
    """單一子指令 → 文字結果（與 Bot 回覆同格式）。"""
    import engine_backtest as eb
    toks = cmd.split()
    if toks and toks[0].lower() in ("/engtest", "engtest"):
        toks = toks[1:]                                  # 容忍貼上時帶了指令前綴
    if toks and toks[0].lower() not in ("run", "opt", "pit", "try", "clear") \
            and toks[0].lower() not in eb.PERIOD_DAYS and "=" not in toks[0]:
        return f"⚠️ 未知子指令 `{toks[0]}`（可用：run／opt／pit／try）"
    if toks and toks[0].lower() == "run":
        toks = toks[1:]
    a = eb.parse_engtest_args(toks)
    if a["errs"]:
        return "⚠️ " + "；".join(a["errs"])
    if a["kind"] == "clear":
        return "⚠️ clear 只對 Bot 的個人設定有意義，研究執行器不寫任何設定"
    base, lg, period = baseline_config(), legacy_config(), a["period"]
    if a["kind"] in ("pit", "try") and a["use_pit"]:
        return eb.run_pit(period, a["k"], a["n_seeds"], a["seed0"], baseline=base, legacy_cfg=lg,
                          candidate=a["cand"] or None, thresholds={}, periods=periods,
                          fetch_fn=fetch_fn, today=today)["text"]
    syms = watchlist[:12]
    if not syms:
        return "❌ 觀察清單是空的（watchlist_state.json）"
    if a["kind"] == "try":
        return eb.run_try(syms, period, base, a["cand"], {}, None, None, lg, fetch_fn=fetch_fn)["text"]
    if a["kind"] == "opt":
        opt = eb.run_optimize(syms, period, baseline=base, thresholds={}, calibration=None,
                              grid=dict(eb.GRIDS[a["grid"]]), legacy_cfg=lg, fetch_fn=fetch_fn)
        return opt["text"] + ("\n（研究執行器不 apply；要套用請在 Bot 跑 `/engtest opt … apply`）" if a["apply"] else "")
    return eb.run(syms, period, params=base, thresholds={}, fetch_fn=fetch_fn)["text"]


def report(cmd: str, watchlist: list[str], fetch_fn=None, periods: dict | None = None,
           today: str | None = None, sha: str | None = None, out_path: str | None = None) -> str:
    """多個子指令 → Markdown 報告（每段附耗時；單段失敗不影響其他段）。
    out_path 給了就每完成一段重寫一次檔案——中途逾時被殺，已完成的段落仍會被發佈。"""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    run_id, ref = os.environ.get("GITHUB_RUN_ID"), os.environ.get("GITHUB_REF_NAME")
    out = [f"# 引擎研究結果（{now}）", "",
           f"- 指令：`{cmd}`", f"- 程式版本：`{sha or '?'}`"
           + (f"｜Run `{run_id}`（分支 `{ref or '?'}`）" if run_id else ""),
           f"- 觀察清單：{len(watchlist)} 檔（取前 12 檔）",
           "- 「現行」＝程式預設參數（不讀加密的個人設定與校準權重）；非投資建議", ""]
    for c in split_cmds(cmd):
        t0 = time.time()
        try:
            txt = run_one(c, watchlist, fetch_fn, periods, today)
        except Exception as e:
            txt = f"❌ 失敗：{type(e).__name__} {str(e)[:200]}"
        out += [f"## `{c}`（{time.time() - t0:.0f} 秒）", "", "```text", txt, "```", ""]
        if out_path:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(out) + "\n")
    if len(split_cmds(cmd)) < len([c for c in (cmd or "").split(";") if c.strip()]):
        out.append(f"⚠️ 一次最多 {MAX_CMDS} 個子指令，其餘略過")
    return "\n".join(out)


def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, capture_output=True,
                              text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


def _selftest() -> int:
    import engine_backtest as eb
    syn = {f"T{i:02d}": eb._synthetic(n=760, seed=i + 1, drift=0.0004 * ((i % 5) - 1), vol=0.02,
                                      start="2023-07-03") for i in range(24)}
    syn["SPY"] = eb._synthetic(n=760, seed=99, drift=0.0004, vol=0.01, start="2023-07-03")
    fetch = lambda syms, period: {k: syn[k] for k in list(syms) + ["SPY"] if k in syn}   # noqa: E731
    per = {k: [("2000-01-01", "9999-12-31")] for k in syn if k != "SPY"}
    wl = [f"T{i:02d}" for i in range(6)]
    # 1) 預設基準與 Bot 一致（scan_signals 對空 thresholds 的結果）
    import scan_signals as ss
    assert baseline_config() == ss._engtest_baseline({}), (baseline_config(), ss._engtest_baseline({}))
    assert legacy_config() == ss._engine_base_config({})
    # 2) 指令切分與上限
    assert split_cmds(" pit 20 5 ; ;try x=1;") == ["pit 20 5", "try x=1"]
    assert len(split_cmds(";".join(["run"] * 9))) == MAX_CMDS
    # 3) 各子指令都能跑完、Markdown 報告含各段、錯誤段不影響其他段
    md = report("run 1y; opt loose 1y apply; pit 6 3 1y; try scale_out_r=off 1y; try bogus=1; clear",
                wl, fetch_fn=fetch, periods=per, today="2026-05-29", sha="abc1234")
    for frag in ("引擎歷史重放", "引擎參數學習", "PBO", "無事後偏誤回測", "單組參數試算", "未知參數",
                 "不 apply", "clear 只對", "程式預設參數", "abc1234", "沒進場的原因"):
        assert frag in md, frag
    assert md.count("```text") == 6
    # 5) 前綴容忍、未知子指令報錯（不默默改跑別的）、逐段寫檔
    assert "未知子指令" in run_one("bogus 1y", wl, fetch_fn=fetch)
    assert "引擎歷史重放" in run_one("/engtest run 1y", wl, fetch_fn=fetch)
    import tempfile
    tp = os.path.join(tempfile.mkdtemp(), "r.md")
    report("run 1y; try bogus=1", wl, fetch_fn=fetch, out_path=tp)
    assert open(tp, encoding="utf-8").read().count("```text") == 2
    # 4) 觀察清單讀取：缺檔 → []；不讀加密欄位
    assert load_watchlist("/nonexistent.json") == []
    print("engine_research selftest OK ✅")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="引擎回測研究執行器")
    ap.add_argument("--cmd", default="", help="子指令，多個用 ; 分隔")
    ap.add_argument("--out", default="", help="Markdown 報告輸出路徑（預設印到 stdout）")
    ap.add_argument("--selftest", action="store_true")
    ns = ap.parse_args(argv)
    if ns.selftest:
        return _selftest()
    if not split_cmds(ns.cmd):
        print("❌ 沒有指令（--cmd \"pit 20 5 2y; …\"）")
        return 2
    md = report(ns.cmd, load_watchlist(), sha=_git_sha(), out_path=ns.out or None)
    if ns.out:
        with open(ns.out, "w", encoding="utf-8") as f:
            f.write(md + "\n")
        print(f"研究報告已寫入 {ns.out}（{len(md)} 字）")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
