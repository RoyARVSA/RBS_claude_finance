"""
bot_daemon.py – 常駐版 Telegram Bot（即時回應，不用等排程）

與原本 scan_signals.py 的關係：
  • 完全沿用 scan_signals.py 的所有邏輯（指令處理、訊號、評分）
  • 差別只在「執行方式」：
      scan_signals.py  → GitHub Actions 每 15 分鐘跑一次（適合零成本、免主機）
      bot_daemon.py    → 常駐迴圈，秒級回應指令 + 定時自動掃描（適合 VPS / 本機）
  • 兩者共用同一個 watchlist_state.json，可隨時「切換」——
    ⚠️ 但**絕不可同時跑**（審查團紅隊實測）：Telegram getUpdates 只容一個
    消費者，並行會指令雙派工（/closeall 執行兩次）、雙下單、state 互相覆蓋
    （停損事件遺失→保險絲失準）。開常駐版前先停用 GitHub Actions 排程。

用法：
  # 環境變數
  export TELEGRAM_TOKEN="..."
  export TELEGRAM_CHAT_ID="..."

  # 直接跑（前景）
  python bot_daemon.py

  # 背景常駐（簡易）
  nohup python bot_daemon.py > bot.log 2>&1 &

  # 正式常駐見 PERSISTENT_BOT.md（systemd 設定）

參數（環境變數，皆可選）：
  POLL_INTERVAL   指令輪詢秒數（預設 3）
  SCAN_INTERVAL   自動掃描間隔秒數（預設 3600 = 1 小時）
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from datetime import datetime, timezone

# 重用 scan_signals 的所有邏輯（單一真實來源）
import scan_signals as ss

POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", 3))
SCAN_INTERVAL = int(os.environ.get("SCAN_INTERVAL", 3600))

TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


def _auto_scan(state: dict) -> None:
    """執行一次自動掃描（沿用 scan_signals 的市場/靜音判斷與訊號邏輯）。"""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    if ss._is_muted(state):
        print(f"[{now}] 靜音中，跳過自動掃描")
        return

    ms = ss.market_status()
    if state["thresholds"].get("scan_market_only", True) and not ms["open"]:
        print(f"[{now}] 市場關閉（{ms['reason']}），跳過自動掃描")
        return

    print(f"[{now}] 執行自動掃描 {len(state['watchlist'])} 支…")
    # 與 main() 共用的掃描+防護流程（校準/大盤濾網/冷卻去重）
    msg, results = ss.scan_and_report(state, now)
    if msg and TOKEN and CHAT_ID:
        ss._tg_send(TOKEN, CHAT_ID, msg)
        print(f"[{now}] 已推送訊號通知")
    else:
        print(f"[{now}] 無訊號觸發（或全在冷卻中）")

    # 自動交易（Alpaca 模擬；預設關閉）
    try:
        at_msg = ss.run_autotrade(state, results)
    except Exception as e:
        print(f"Autotrade error: {e}")   # 不讓例外擋掉後面的 save_state
        at_msg = None
    if at_msg and TOKEN and CHAT_ID:
        ss._tg_send(TOKEN, CHAT_ID, at_msg)
    ss.save_state(state)


def main() -> int:
    if not TOKEN or not CHAT_ID:
        print("ERROR: 需要設定 TELEGRAM_TOKEN 與 TELEGRAM_CHAT_ID 環境變數")
        return 1

    print("=" * 60)
    print("RBS 常駐 Bot 啟動")
    print(f"  指令輪詢：每 {POLL_INTERVAL} 秒")
    print(f"  自動掃描：每 {SCAN_INTERVAL} 秒（{SCAN_INTERVAL//60} 分鐘）")
    print("=" * 60)

    state = ss.load_state()
    ss.save_state(state)

    # 啟動通知
    ss._tg_send(TOKEN, CHAT_ID,
                "🟢 *RBS 常駐 Bot 已上線*\n"
                f"觀察清單 {len(state['watchlist'])} 支 · 輸入 /help 查看指令")

    last_scan = 0.0
    tick = 0
    _reload_at = 0.0
    _guard_reset_at = 0.0
    while True:
        try:
            # 0. 週期重讀磁碟 state（審查團 F5）：若外部（cron/人工）改了檔案，
            #    別讓記憶體舊資料覆蓋掉——每 60 秒同步一次
            nowt = time.monotonic()
            if nowt - _reload_at >= 60:
                state = ss.load_state()
                _reload_at = nowt

            # 0.5 熔斷重置：每 15 分鐘（=cron 節奏）。放主迴圈而非 _auto_scan——
            #     收盤時段 _auto_scan 早退，晚間一次熔斷會讓 /opt /short 等指令
            #     死到隔天開盤（對抗驗證 M1）
            if nowt - _guard_reset_at >= 900:
                try:
                    import net_guard
                    net_guard.reset()
                    import sec_insider
                    sec_insider.reset_breaker()
                except Exception:
                    pass
                _guard_reset_at = nowt

            # 1. 即時處理 Telegram 指令
            try:
                state, changed = ss.process_commands(TOKEN, CHAT_ID, state)
            except Exception as e:
                # 毒訊息防護（審查團 F13）：process_commands 內 offset 已前進，
                # 必須落盤消耗掉，否則 systemd 重啟會重放毒訊息
                print(f"Command processing error: {e}")
                changed = True
            if changed:
                ss.save_state(state)

            # 1.5 每日晨報（盤前獨立推送）
            if ss._should_send_briefing(state):
                from datetime import datetime as _dt
                print("發送每日晨報…")
                bmsg = ss.daily_briefing(state)
                if bmsg:
                    ss._tg_send(TOKEN, CHAT_ID, bmsg)
                state["last_briefing_date"] = _dt.now(ss.ET).strftime("%Y-%m-%d")
                ss.save_state(state)

            # 1.6 每週深度週報（審查團 F11：原本只在排程版，daemon 部署收不到）
            if ss._should_send_weekly(state):
                print("發送每週週報…")
                try:
                    wmsg = ss.weekly_report(state)
                    if wmsg:
                        ss._tg_send(TOKEN, CHAT_ID, wmsg)
                except Exception as e:
                    print(f"Weekly report error: {e}")
                from datetime import datetime as _dt2
                _nw = _dt2.now(ss.ET)
                # 格式必須與 scan_signals.main 一致（週 ID 非日期，寫錯會每輪狂發）
                state["last_weekly"] = f"{_nw.isocalendar().year}-W{_nw.isocalendar().week}"
                ss.save_state(state)

            # 2. 定時自動掃描
            if nowt - last_scan >= SCAN_INTERVAL:
                _auto_scan(state)
                ss.save_state(state)
                last_scan = nowt

            # 3. 心跳（每 ~5 分鐘印一次，確認還活著）
            tick += 1
            if tick % max(1, (300 // POLL_INTERVAL)) == 0:
                print(f"[{datetime.now(timezone.utc):%H:%M:%S}] alive · "
                      f"watchlist={len(state['watchlist'])}")

            time.sleep(POLL_INTERVAL)

        except KeyboardInterrupt:
            print("\n收到中斷訊號，關閉中…")
            ss._tg_send(TOKEN, CHAT_ID, "🔴 *RBS 常駐 Bot 已下線*")
            ss.save_state(state)
            return 0
        except Exception as e:
            print(f"迴圈錯誤（已忽略繼續）：{e}")
            traceback.print_exc()
            time.sleep(POLL_INTERVAL * 2)


if __name__ == "__main__":
    sys.exit(main())
