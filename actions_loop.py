"""
actions_loop.py – GitHub Actions 長駐迴圈（#60：排程被 GitHub 大量丟棄的解法）

問題：signal_scan.yml 設定每 15 分鐘，GitHub 實際一天只觸發 3–8 次（2026-09/10 實測），
Telegram 指令要等幾小時才有回應、停損評估延遲數小時。

解法：每次被觸發的 Actions 工作不再「跑一輪就結束」，而是在時間預算內（預設 330 分鐘）持續：
  • 每 ROUND_SEC（預設 900 秒）跑一次完整的 scan_signals.main()——指令、晨報、掃描、自動交易、存檔
  • 兩輪之間每 POLL_SEC（預設 60 秒）只處理 Telegram 指令（秒級回應）
  • 每輪結束（或指令改了 state）就呼叫 scripts/persist_state.sh 把 state commit + push 回 repo
排程改成每小時觸發；concurrency 讓同時只有一個工作，下一個排隊、前一個結束立刻接手。
一天只要有幾次觸發成功，整個交易時段都有人看著。

日誌安全（PITFALLS D14）：只印輪次、秒數、是否有變更；不印指令參數、持倉、淨值。
純邏輯可注入（clock / sleep / round_fn / poll_fn / persist_fn），離線自測：`python3 actions_loop.py --selftest`。
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

LOOP_MINUTES = min(float(os.environ.get("LOOP_MINUTES", 330)), 335.0)   # 上限：job timeout 350 分（對抗驗證 L1）
ROUND_SEC = int(os.environ.get("ROUND_SEC", 900))
POLL_SEC = int(os.environ.get("POLL_SEC", 60))
SAFETY_SEC = 240            # 預算尾端保留：最後一輪 main() 最長約 4 分鐘，不在這段時間開新輪（第一輪除外）
END_QUIET_SEC = 600         # 預算最後 10 分鐘不再處理指令（/engtest opt 等重型指令可能跑好幾分鐘，避免被 timeout 硬殺）
IDLE_PUSH_SEC = 3600        # state 只有 health/時間戳變動時，最多每小時推一次（每次 commit 都會觸發 Streamlit 重啟）
VOLATILE_KEYS = ("health", "last_run_ts", "last_scan_time")
STATE_PATHS = ("watchlist_state.json", "trade_journal.json", "estimates_ledger.json", "data/universe", "data/fin")


def run_loop(round_fn, poll_fn, persist_fn, budget_s: float, round_s: int = ROUND_SEC, poll_s: int = POLL_SEC,
             clock=time.monotonic, sleep=time.sleep, log=print, sync_fn=None, restart_fn=None) -> dict:
    """
    round_fn()：完整一輪；poll_fn() → bool（state 有變更、需要推送）；persist_fn(force) → bool（True = 已推送或無需推送）；
    sync_fn() → "restart" | None：每輪開始前（上一次 persist 成功才呼叫）把工作目錄同步到遠端最新，程式碼有變回 "restart"；
    restart_fn(remaining_s)：自我重啟（不回傳）。回統計 {rounds, polls, persists, errors}。
    """
    t0 = clock()
    stats = {"rounds": 0, "polls": 0, "persists": 0, "errors": 0}
    next_round = t0
    last_ok = True

    def _persist(force):
        nonlocal last_ok
        try:
            last_ok = bool(persist_fn(force))
            stats["persists"] += 1
        except Exception as e:
            last_ok = False
            stats["errors"] += 1
            log(f"[loop] persist error {type(e).__name__}")

    while True:
        now = clock()
        left = budget_s - (now - t0)
        if left <= 0:
            break
        first = stats["rounds"] == 0
        if now >= next_round and (left > SAFETY_SEC or first):
            if sync_fn is not None and last_ok:
                try:
                    if sync_fn() == "restart" and restart_fn is not None:
                        log("[loop] code changed on remote → restarting with remaining budget")
                        restart_fn(left)
                        return stats                     # 測試用：restart_fn 回傳時結束
                except Exception as e:
                    stats["errors"] += 1
                    log(f"[loop] sync error {type(e).__name__}")
            r0 = clock()
            try:
                round_fn()
            except Exception as e:                     # main() 自己已吞大部分例外；這裡是最後防線
                stats["errors"] += 1
                log(f"[loop] round error {type(e).__name__}")
            stats["rounds"] += 1
            _persist(False)
            log(f"[loop] round {stats['rounds']} done in {clock() - r0:.0f}s")
            next_round = r0 + round_s
            continue
        if left <= SAFETY_SEC and now >= next_round:
            next_round = float("inf")                  # 預算尾端：不再開完整輪
        if left > END_QUIET_SEC:
            try:
                changed = bool(poll_fn())
            except Exception as e:
                changed = True                         # 毒訊息防護：poll_fn 自己已落盤，這裡確保推送
                stats["errors"] += 1
                log(f"[loop] poll error {type(e).__name__}")
            stats["polls"] += 1
            if changed:
                _persist(True)
        wait = min(poll_s, max(0.0, budget_s - (clock() - t0)))
        if next_round != float("inf"):
            wait = min(wait, max(0.0, next_round - clock()))
        if wait > 0:
            sleep(wait)
    _persist(True)
    log(f"[loop] budget used — rounds {stats['rounds']}, polls {stats['polls']}, errors {stats['errors']}")
    return stats


def _sh(cmd: list[str], timeout: int = 180) -> subprocess.CompletedProcess:
    """子程序開新 session；逾時時整個程序群組一起殺（git 孫程序不會殘留 index.lock，對抗驗證 L3）。"""
    import signal
    p = subprocess.Popen(cmd, start_new_session=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        out, _ = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except Exception:
            pass
        out, _ = p.communicate()
        return subprocess.CompletedProcess(cmd, 124, out, "")
    return subprocess.CompletedProcess(cmd, p.returncode, out, "")


HERE = os.path.dirname(os.path.abspath(__file__))


def state_digest(state: dict) -> str:
    """解密後 state 去掉易變鍵（health／時間戳）的雜湊——密文每次存檔都換 nonce，不能比檔案。"""
    import hashlib
    import json
    s = {k: v for k, v in (state or {}).items() if k not in VOLATILE_KEYS}
    return hashlib.sha256(json.dumps(s, sort_keys=True, default=str).encode()).hexdigest()


def _make_persist(ss, clock=time.monotonic):
    memo = {"digest": None, "pushed_at": None}

    def persist(force: bool) -> bool:
        if not force:
            try:
                d = state_digest(ss.load_state())
            except Exception:
                d = None
            now = clock()
            if d is not None and d == memo["digest"] and memo["pushed_at"] is not None \
                    and now - memo["pushed_at"] < IDLE_PUSH_SEC:
                return True                            # 只有 health/時間戳變動：暫不推（工作樹保留，下次一起推）
        else:
            try:
                d = state_digest(ss.load_state())
            except Exception:
                d = None
        r = _sh(["bash", os.path.join(HERE, "scripts", "persist_state.sh")])
        print((r.stdout or "").strip()[-200:])
        if r.returncode == 0:
            memo["pushed_at"] = clock()
            memo["digest"] = d                          # 只在推送成功後記錄（失敗時下一輪必須重試，複驗 M4）
            return True
        return False
    return persist


def code_hash() -> str:
    r = _sh(["git", "ls-tree", "-r", "HEAD", "--", "*.py", "gics_template.html", "scripts"], timeout=30)
    return r.stdout or ""


def _make_sync():
    """每輪開始前同步到遠端最新（排隊接手的工作 checkout 的是觸發當下的舊版本，對抗驗證 H2）；程式碼有變 → restart。"""
    base = {"code": code_hash()}
    br = os.environ.get("GITHUB_REF_NAME", "")

    def sync():
        if not br:
            return None
        f = _sh(["git", "fetch", "-q", "origin", br], timeout=120)
        if f.returncode != 0:
            return None
        if _sh(["git", "diff", "--quiet", "HEAD", "--", *STATE_PATHS], timeout=30).returncode != 0:
            return None                                 # state 有未推送的變更（被節流暫存的 health）→ 先不 reset
        _sh(["git", "reset", "-q", "--hard", "FETCH_HEAD"], timeout=60)
        return "restart" if code_hash() != base["code"] else None
    return sync


def _restart(remaining_s: float):
    sys.stdout.flush(); sys.stderr.flush()              # execv 會丟掉未 flush 的輸出
    os.environ["LOOP_MINUTES"] = f"{max(1.0, remaining_s / 60):.1f}"
    os.execv(sys.executable, [sys.executable, os.path.join(HERE, "actions_loop.py")])


def _make_poll(ss):
    def poll() -> bool:
        """只處理 Telegram 指令。讀取位置（last_update_id）前進就一定存檔——唯讀指令（/help、/closeall…）
        不設 changed，不存檔的話同一則訊息下一分鐘會被重複處理（對抗驗證 H1）。"""
        if not ss.TELEGRAM_TOKEN:
            return False
        state = ss.load_state()
        before = state.get("last_update_id")
        changed = False
        try:
            state, changed = ss.process_commands(ss.TELEGRAM_TOKEN, ss.TELEGRAM_CHAT_ID, state)
        except Exception as e:
            print(f"[loop] command processing error {type(e).__name__}")
            changed = True                              # 毒訊息：offset 已就地前進，必須落盤消耗掉
        if changed or state.get("last_update_id") != before:
            ss.save_state(state)
            return True
        return False
    return poll


def _make_round(ss):
    def rnd():
        try:                                            # 熔斷器每輪重置（同 bot_daemon：收盤時段一次熔斷不能卡死到隔天）
            import net_guard
            net_guard.reset()
            import sec_insider
            sec_insider.reset_breaker()
        except Exception:
            pass
        ss.main()
    return rnd


def main(argv: list[str]) -> int:
    if "--selftest" in argv:
        return _selftest()
    import scan_signals as ss
    budget = max(60.0, LOOP_MINUTES * 60)
    print(f"[loop] start — budget {budget / 60:.0f} min, round {ROUND_SEC}s, poll {POLL_SEC}s")
    run_loop(_make_round(ss), _make_poll(ss), _make_persist(ss), budget, sync_fn=_make_sync(), restart_fn=_restart)
    return 0


def _selftest() -> int:
    class Clock:
        def __init__(self):
            self.t = 0.0
        def __call__(self):
            return self.t
        def sleep(self, s):
            self.t += s
    # 1) 節奏：預算 3600s、每輪 900s（耗 60s）、poll 60s → 4 輪；尾端不開新輪；最後 10 分鐘不 poll；結束必 persist
    c = Clock()
    calls = {"round": 0, "poll": 0, "persist": 0, "force": 0, "sync": 0}
    def rnd():
        calls["round"] += 1; c.t += 60
    def poll():
        calls["poll"] += 1
        assert 3600 - c.t > END_QUIET_SEC - 1e-9, c.t
        return calls["poll"] % 7 == 0
    def persist(force):
        calls["persist"] += 1; calls["force"] += int(force); return True
    def sync():
        calls["sync"] += 1; return None
    st = run_loop(rnd, poll, persist, 3600, 900, 60, clock=c, sleep=c.sleep, log=lambda *_: None, sync_fn=sync)
    assert st["rounds"] == 4 and calls["sync"] == 4, (st, calls)
    assert calls["persist"] == 4 + calls["poll"] // 7 + 1 and calls["force"] == calls["poll"] // 7 + 1
    assert c.t <= 3600 + 1e-9
    # 2) 例外隔離 + persist 失敗時不 sync（免得 reset 丟掉本輪 state）
    c2 = Clock()
    n = {"p": 0, "s": 0}
    def bad_round():
        c2.t += 30; raise RuntimeError("x")
    def bad_poll():
        raise ValueError("y")
    def failing_persist(force):
        n["p"] += 1; return False
    def sync2():
        n["s"] += 1
    st2 = run_loop(bad_round, bad_poll, failing_persist, 1500, 300, 60, clock=c2, sleep=c2.sleep, log=lambda *_: None, sync_fn=sync2)
    assert st2["rounds"] >= 2 and st2["errors"] > 0 and n["s"] == 1, (st2, n)      # 只有第一輪前 sync 一次
    # 3) 預算小於安全邊際：第一輪仍執行（loop_minutes=1 = 只跑一輪，對抗驗證 M2）
    c3 = Clock()
    k = {"r": 0}
    def r3():
        k["r"] += 1; c3.t += 20
    st3 = run_loop(r3, lambda: False, lambda f: True, 60, 900, 60, clock=c3, sleep=c3.sleep, log=lambda *_: None)
    assert k["r"] == 1 and st3["persists"] == 2
    # 4) 程式碼變更 → restart_fn 帶剩餘預算
    c4 = Clock()
    got = {}
    st4 = run_loop(lambda: None, lambda: False, lambda f: True, 3600, 900, 60, clock=c4, sleep=c4.sleep,
                   log=lambda *_: None, sync_fn=lambda: "restart", restart_fn=lambda left: got.setdefault("left", left))
    assert got["left"] == 3600 and st4["rounds"] == 0
    # 5) poll：唯讀指令（不設 changed）也要因 offset 前進而存檔；例外路徑也存檔
    class FakeSS:
        TELEGRAM_TOKEN, TELEGRAM_CHAT_ID = "t", "c"
        def __init__(self):
            self.disk = {"last_update_id": 100}; self.saves = 0; self.mode = "read"
        def load_state(self):
            return dict(self.disk)
        def save_state(self, st):
            self.disk = dict(st); self.saves += 1
        def process_commands(self, tok, chat, st):
            st["last_update_id"] = st["last_update_id"] + 1
            if self.mode == "boom":
                raise RuntimeError("poison")
            return st, False
    fs = FakeSS()
    pl = _make_poll(fs)
    assert pl() is True and fs.disk["last_update_id"] == 101 and fs.saves == 1
    fs.mode = "boom"
    assert pl() is True and fs.disk["last_update_id"] == 102
    # 5b) persist 失敗後下一輪必須重試（不能因 digest 已記錄而跳過，複驗 M4）
    global _sh
    real_sh = _sh
    rc = {"v": 0, "calls": 0}
    def fake_sh(cmd, timeout=180):
        rc["calls"] += 1
        return subprocess.CompletedProcess(cmd, rc["v"], "", "")
    _sh = fake_sh
    try:
        clk = {"t": 0.0}
        class SS2:
            st = {"watchlist": ["A"]}
            def load_state(self):
                return dict(self.st)
        s2 = SS2()
        ps = _make_persist(s2, clock=lambda: clk["t"])
        assert ps(False) and rc["calls"] == 1                                 # t=0 推送成功
        clk["t"] = 900; s2.st = {"watchlist": ["B"]}; rc["v"] = 2
        assert ps(False) is False and rc["calls"] == 2                        # 實質變動、推送失敗
        clk["t"] = 1800; rc["v"] = 0
        assert ps(False) and rc["calls"] == 3                                 # 下一輪必須重試
        clk["t"] = 2700
        assert ps(False) and rc["calls"] == 3                                 # 已推、只剩 health → 節流
    finally:
        _sh = real_sh
    # 6) state 雜湊忽略 health／時間戳
    a = {"watchlist": ["A"], "health": {"runs": 1}, "last_run_ts": "x"}
    b = {"watchlist": ["A"], "health": {"runs": 9}, "last_run_ts": "y"}
    assert state_digest(a) == state_digest(b) and state_digest(a) != state_digest({"watchlist": ["B"]})
    print("✅ 長駐迴圈：節奏、尾端靜默、第一輪必跑、存檔失敗不同步、程式碼變更重啟、唯讀指令不重複、雜湊忽略易變鍵")
    print("\nactions_loop selftest OK ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
