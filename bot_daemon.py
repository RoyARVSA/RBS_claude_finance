"""
bot_daemon.py – 常駐版 Telegram Bot（即時回應，不用等排程）

與原本 scan_signals.py 的關係：
  • 完全沿用 scan_signals.py 的所有邏輯（指令處理、訊號、評分）
  • 差別只在「執行方式」：
      scan_signals.py  → GitHub Actions 每 15 分鐘跑一次（適合零成本、免主機）
      bot_daemon.py    → 常駐迴圈，秒級回應指令 + 定時自動掃描（適合 VPS / 本機）
  • 兩者共用同一個 watchlist_state.json，可隨時切換，互不衝突

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

    # 每週自動回測校準（自我優化迴圈）
    if ss.maybe_calibrate(state):
        ss.save_state(state)
        print(f"[{now}] 校準已更新")

    print(f"[{now}] 執行自動掃描 {len(state['watchlist'])} 支…")
    results = ss.scan(state["watchlist"], state["thresholds"],
                      calibration=ss._calibration_weights(state))
    state["last_scan_time"] = now
    msg = ss._build_message(results, now)
    if msg and TOKEN and CHAT_ID:
        ss._tg_send(TOKEN, CHAT_ID, msg)
        print(f"[{now}] 已推送訊號通知")
    else:
        print(f"[{now}] 無訊號觸發")


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
    while True:
        try:
            # 1. 即時處理 Telegram 指令
            state, changed = ss.process_commands(TOKEN, CHAT_ID, state)
            if changed:
                ss.save_state(state)

            # 2. 定時自動掃描
            nowt = time.monotonic()
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
