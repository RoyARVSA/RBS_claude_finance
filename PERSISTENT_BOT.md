# 常駐 Telegram Bot 教學

兩種執行方式，**可隨時切換、互不衝突**（共用同一個 `watchlist_state.json`）：

| 方式 | 檔案 | 回應速度 | 成本 | 需要主機 |
|------|------|----------|------|----------|
| **排程版**（現有） | `scan_signals.py` + GitHub Actions | 每 15 分鐘 | 免費 | ❌ |
| **常駐版**（新增） | `bot_daemon.py` | 秒級即時 | 需主機 | ✅ |

> 常駐版完全沿用排程版的所有邏輯（指令、訊號、評分），只是改成不停迴圈。
> 原本的 GitHub Actions 排程**不用動，繼續保留**。

---

## 差別在哪？

**排程版**：GitHub Actions 每 15 分鐘喚醒一次 → 處理這段時間累積的指令 + 掃描一次 → 關閉。
所以你傳 `/add AAPL` 後最多等 15 分鐘才會回。

**常駐版**：程式一直開著，每 3 秒檢查一次有沒有新指令 → **傳完馬上回**，同時每小時自動掃描一次。

---

## 方法 A：本機跑（最簡單，測試用）

```bash
# 1. 安裝套件
pip install yfinance pandas numpy requests

# 2. 設定環境變數
export TELEGRAM_TOKEN="你的Token"
export TELEGRAM_CHAT_ID="你的ChatID"

# 3. 啟動
python bot_daemon.py
```

跑起來後傳 `/help` 給 Bot，會**立刻**回應。
缺點：關掉終端機或電腦睡眠就停了。

背景跑（關掉終端機也持續，需電腦不關機）：
```bash
nohup python bot_daemon.py > bot.log 2>&1 &
# 查看 log
tail -f bot.log
# 停止
pkill -f bot_daemon.py
```

---

## 方法 B：DigitalOcean VPS 24/7 常駐（推薦，用學生 $200 額度）

### 1. 開一台最便宜的 Droplet

- 登入 https://cloud.digitalocean.com/
- Create → Droplet → Ubuntu 24.04
- 方案選 **Basic / Regular $4-6/月**（512MB-1GB 就夠，學生額度可用一年多）
- 認證方式建議用 SSH key，或設密碼
- Create

### 2. 連線進去

```bash
ssh root@你的Droplet_IP
```

### 3. 安裝環境 + 抓程式碼

```bash
apt update && apt install -y python3-pip git
git clone https://github.com/RoyARVSA/RBS_claude_finance.git
cd RBS_claude_finance
pip install -r requirements.txt --break-system-packages
```

### 4. 用 systemd 設定常駐（開機自動啟動、當掉自動重啟）

```bash
nano /etc/systemd/system/rbsbot.service
```

貼上（記得換成你的 Token / Chat ID）：

```ini
[Unit]
Description=RBS Telegram Bot Daemon
After=network.target

[Service]
Type=simple
WorkingDirectory=/root/RBS_claude_finance
Environment=TELEGRAM_TOKEN=你的Token
Environment=TELEGRAM_CHAT_ID=你的ChatID
Environment=SCAN_INTERVAL=3600
ExecStart=/usr/bin/python3 /root/RBS_claude_finance/bot_daemon.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

啟動：

```bash
systemctl daemon-reload
systemctl enable rbsbot      # 開機自動啟動
systemctl start rbsbot       # 立即啟動
systemctl status rbsbot      # 查看狀態（按 q 離開）
journalctl -u rbsbot -f      # 即時看 log（Ctrl+C 離開）
```

完成後，Telegram 會收到「🟢 RBS 常駐 Bot 已上線」，從此 24 小時即時回應。

### 5. 常用維護指令

```bash
systemctl restart rbsbot     # 重啟（改了設定後）
systemctl stop rbsbot        # 停止
# 更新程式碼
cd /root/RBS_claude_finance && git pull && systemctl restart rbsbot
```

---

## 參數調整（環境變數）

| 變數 | 預設 | 說明 |
|------|------|------|
| `POLL_INTERVAL` | 3 | 幾秒檢查一次新指令（越小越即時，但耗流量） |
| `SCAN_INTERVAL` | 3600 | 自動掃描間隔秒數（3600=1小時，1800=30分） |
| `TELEGRAM_TOKEN` | — | 必填 |
| `TELEGRAM_CHAT_ID` | — | 必填 |

在 systemd 檔案的 `Environment=` 加一行即可，例如盤中想更頻繁掃描：
```ini
Environment=SCAN_INTERVAL=1800
```

---

## 兩種版本可以並存嗎？

可以，但**建議擇一自動掃描**避免重複通知：

- 用常駐版 → 把 GitHub Actions 的 cron 註解掉（保留 `workflow_dispatch` 手動觸發備援）：
  ```yaml
  on:
    # schedule:
    #   - cron: '*/15 * * * *'
    workflow_dispatch:
  ```
- 指令處理兩邊都會搶讀 Telegram 訊息（`getUpdates` 的 offset 機制會讓先讀到的處理掉），
  所以**只開一個自動掃描來源**最乾淨。

平常用 GitHub Actions 免費版即可；要即時互動或盤中高頻掃描時，再開 VPS 常駐版。

---

## 疑難排解

**Q: Bot 沒回應？**
`journalctl -u rbsbot -f` 看 log，常見是 Token/Chat ID 填錯。

**Q: 一直重複收到同樣通知？**
可能同時開了常駐版 + GitHub Actions 都在掃描。關掉其中一個的自動掃描。

**Q: VPS 重開機後 Bot 沒起來？**
確認有跑 `systemctl enable rbsbot`（開機自啟）。

**Q: 想省錢？**
平日用免費 GitHub Actions 排程版就好，VPS 只在需要盤中即時時才開。
DigitalOcean 可隨時 Destroy droplet 停止計費，$200 額度用很久。
