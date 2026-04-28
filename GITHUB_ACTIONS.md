# GitHub Actions 自動訊號掃描設定指引

每小時自動掃描 watchlist → 有訊號時推 Telegram 通知。

---

## 快速設定（5 分鐘）

### 1. 取得 Telegram Bot Token 和 Chat ID

如果還沒設定：

1. Telegram 搜尋 `@BotFather` → 傳 `/newbot` → 依指示建立 Bot → 複製 **Token**
2. 把 Bot 加為好友，傳任意一句話給它
3. 瀏覽器開：
   ```
   https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
   ```
   回傳 JSON 裡 `result[0].message.chat.id` 就是你的 **Chat ID**

---

### 2. 在 GitHub 設定 Secrets

前往 `https://github.com/RoyARVSA/RBS_claude_finance/settings/secrets/actions`

點 **New repository secret**，逐一新增：

| Secret 名稱 | 必填 | 說明 |
|-------------|------|------|
| `TELEGRAM_TOKEN` | ✅ | Bot Token，例如 `7123456789:AAF...` |
| `TELEGRAM_CHAT_ID` | ✅ | 你的 Chat ID，例如 `123456789` |
| `WATCHLIST` | 選填 | 逗號分隔股票代碼，例如 `AAPL,TSLA,NVDA,SPY`<br>不填則掃描預設清單 |
| `MIN_RSI_OVERSOLD` | 選填 | RSI 超賣門檻（預設 `35`） |
| `MAX_RSI_OVERBOUGHT` | 選填 | RSI 超買門檻（預設 `70`） |
| `PRICE_CHANGE_PCT` | 選填 | 單日漲跌 % 警示門檻（預設 `3.0`） |

---

### 3. 啟用 Actions

1. 前往 repo → **Actions** 頁籤
2. 若出現「Workflows aren't being run」警告，點 **I understand my workflows, go ahead and enable them**
3. 在左欄找到 **RBS Signal Scanner** → 點 **Run workflow** 手動測試一次

---

### 4. 確認 cron 排程

`.github/workflows/signal_scan.yml` 中已設定：

```yaml
schedule:
  - cron: '0 * * * *'   # 每小時整點（UTC）執行
```

UTC 整點對應台灣時間（UTC+8）：
- UTC 00:00 = 台灣 08:00
- UTC 08:00 = 台灣 16:00
- UTC 22:00 = 台灣 06:00（隔天）

---

## 通知範例

當有訊號觸發時，Bot 會傳送：

```
🚨 RBS 自動訊號掃描 — 2026-04-28 10:00 UTC

🔴 TSLA $165.20  (-4.3%)
   ↳ 單日暴跌 -4.3% | RSI 超賣 (31.5)
🟢 NVDA $880.50  (+3.8%)
   ↳ 單日暴漲 +3.8% | 黃金交叉 (MA20↑MA50)

共掃描 10 支，觸發 2 支
由 GitHub Actions 自動執行
```

沒有任何訊號時，不會發送通知（避免騷擾）。

---

## 手動觸發掃描

1. GitHub → Actions → **RBS Signal Scanner**
2. 點 **Run workflow**
3. 可選填「自訂 watchlist」欄位，例如 `AAPL,TSLA`（只掃這兩支）
4. 點 **Run workflow** 執行

---

## 本地測試

```bash
# 安裝相依套件
pip install yfinance pandas numpy requests

# 設定環境變數後執行
export TELEGRAM_TOKEN="你的Token"
export TELEGRAM_CHAT_ID="你的ChatID"
export WATCHLIST="AAPL,TSLA,NVDA"

python scan_signals.py
```

---

## 常見問題

**Q: GitHub Actions cron 不準時怎麼辦？**
A: GitHub 免費版 cron 可能延遲最多 15 分鐘，屬正常現象。

**Q: Actions 跑完但沒收到 Telegram？**
A: 可能是沒有訊號觸發（屬正常），或 Secrets 設定有誤。查看 Actions 執行 log 確認。

**Q: 想要更高頻率怎麼辦？**
A: 改成 `'*/30 * * * *'`（每 30 分鐘）。注意 yfinance 15 分鐘延遲，太高頻無實質意義。

**Q: 免費 GitHub Actions 額度夠嗎？**
A: 公開 repo 無限制；私人 repo 每月 2000 分鐘免費額度，每次掃描約 1-2 分鐘，每小時一次 = 月用 ~60-90 分鐘，綽綽有餘。

---

## 擴充方向

- **更多指標**: 在 `scan_signals.py` 的 `scan()` 函數中加入 MACD、布林通道等
- **Email 備援**: 在 `main()` 裡加入 `smtplib.SMTP_SSL` 作為 Telegram 失敗時的備援
- **Discord**: 改用 Discord Webhook 替代 Telegram
- **標的分組**: 區分個股 / ETF / 加密貨幣分組掃描，各自有不同門檻
