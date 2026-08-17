"""
net_guard.py – 通用「慢源熔斷」decorator（純標準庫、離線可測）

背景：SEC 事件（PITFALLS B10）證明「外部源不快速拒絕、而是掛連線等逾時」
會把 15 分鐘 cron 拖成每小時。sec_insider 已有專用熔斷；維運審查評估
options_sentiment（Yahoo 選擇權鏈 ×3 到期日 ×alpha 4 檔）與 short_data
（FINRA 日檔 ×lookback + SEC FTD zip）各有最壞 ~4 分鐘的拖速面——
本模組把熔斷抽成可重用 decorator，一行掛上任何抓取函數。

語意（per-process＝per-cron-輪，下輪自動重試）：
  • 被包函數單次呼叫「耗時 ≥ budget_s 且回傳為空」→ 判定該源掛了，
    本製程內後續呼叫直接回 fallback（不再等逾時）
  • 例外穿出 → 同樣熔斷（多數抓取函數內部吞例外回空，走上一條）
  • 快而空（正常查無）、慢而有料（網路慢但活著）都**不**熔斷
  • fallback 型別必須與原函數的失敗回傳一致（None/{}/[]），呼叫端才不炸
"""

from __future__ import annotations

import functools
import time

_TRIPPED: dict = {}     # key -> 原因字串（製程內；每輪 cron 全新製程自動重置）


def reset(key: str | None = None) -> None:
    """測試/常駐版換輪用：清除熔斷狀態。"""
    if key is None:
        _TRIPPED.clear()
    else:
        _TRIPPED.pop(key, None)


def is_tripped(key: str) -> bool:
    return key in _TRIPPED


def guarded(key: str, budget_s: float = 20.0, fallback=None):
    """
    decorator 工廠。fallback 需與被包函數的「失敗回傳」同型別。
    daemon 長製程請在每輪掃描前呼叫 reset()（bot_daemon 已接）。
    """
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if key in _TRIPPED:
                return fallback
            t0 = time.monotonic()
            try:
                out = fn(*args, **kwargs)
            except Exception as e:
                _TRIPPED[key] = f"例外 {type(e).__name__}"
                print(f"net_guard: {key} 熔斷（{_TRIPPED[key]}）——本輪跳過此源")
                return fallback
            dur = time.monotonic() - t0
            empty = out is None or out == {} or out == []
            if empty and dur >= budget_s:
                _TRIPPED[key] = f"慢 {dur:.0f}s 且空"
                print(f"net_guard: {key} 熔斷（{_TRIPPED[key]}）——本輪跳過此源")
            return out
        return wrapper
    return deco


# ── 自我測試 ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    calls = {"n": 0}

    # 1) 慢而空 → 熔斷；後續呼叫不再進函數、回 fallback
    @guarded("slow_empty", budget_s=0.05, fallback={})
    def f_slow_empty():
        calls["n"] += 1
        time.sleep(0.08)
        return {}

    assert f_slow_empty() == {} and calls["n"] == 1
    assert is_tripped("slow_empty")
    assert f_slow_empty() == {} and calls["n"] == 1      # 沒再進函數
    print("✅ 1 慢而空 → 熔斷 + 跳過")

    # 2) 快而空（正常查無）→ 不熔斷
    reset()
    @guarded("fast_empty", budget_s=0.05, fallback=None)
    def f_fast_empty():
        return None
    f_fast_empty(); f_fast_empty()
    assert not is_tripped("fast_empty")
    print("✅ 2 快而空不熔斷")

    # 3) 慢而有料（網路慢但活著）→ 不熔斷
    @guarded("slow_full", budget_s=0.05, fallback=None)
    def f_slow_full():
        time.sleep(0.08)
        return {"x": 1}
    assert f_slow_full() == {"x": 1} and not is_tripped("slow_full")
    print("✅ 3 慢而有料不熔斷")

    # 4) 例外 → 熔斷 + fallback（型別依註冊）
    @guarded("boom", budget_s=1.0, fallback=[])
    def f_boom():
        raise RuntimeError("網路炸了")
    assert f_boom() == [] and is_tripped("boom")
    assert f_boom() == []
    print("✅ 4 例外熔斷 + fallback 型別")

    # 5) reset 恢復；key 隔離
    f_slow_empty()                       # 重新熔斷（test 2 的全域 reset 清過）
    assert is_tripped("slow_empty") and is_tripped("boom")
    reset("boom")
    assert not is_tripped("boom") and is_tripped("slow_empty")
    reset()
    assert not is_tripped("slow_empty")
    print("✅ 5 reset / key 隔離")

    print("\nnet_guard selftest OK ✅")
