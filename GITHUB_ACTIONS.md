# GitHub Actions 自動訊號掃描設定指引

每 15 分鐘自動掃描 watchlist → 有訊號時推 Telegram 通知，並回應你傳的指令。

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
| `LLM_API_KEY` | 選填 | 每日 AI 晨報用的 LLM key（Claude 或 OpenAI）。**不設則晨報走純數據版** |
| `LLM_BASE_URL` | 選填 | 自訂 API 端點（留空自動判斷 Anthropic/OpenAI） |
| `LLM_MODEL` | 選填 | 模型名（預設 Claude 用 `claude-3-5-haiku`，OpenAI 用 `gpt-4o-mini`） |
| `FRED_API_KEY` | 選填 | 晨報總經數據（免費申請 fred.stlouisfed.org） |
| `FINNHUB_API_KEY` | 選填 | 基本面備援；yfinance `.info` 被限流時的後援（免費申請 finnhub.io） |
| `SEC_USER_AGENT` | 選填 | SEC 內部人交易（Form 4）自訂 User-Agent；**有預設值即可用，免申請** |
| `ALPACA_KEY_ID` | 選填 | Alpaca **paper** trading key（模擬自動交易；**不設則不下單**） |
| `ALPACA_SECRET_KEY` | 選填 | Alpaca paper secret |

> 🤖 **Alpaca 模擬交易**：預設**關閉**，須 Telegram 傳 `/autotrade on` 才會下單，
> 且僅在美股開盤時、由**分層交易引擎**（`trade_engine.py`，重製自 freqtrade / Lean /
> nautilus 機制）自動下*模擬*單：評分只決定進場；出場走硬停損、+1R 保本追蹤停損、
> +1.5R 分批鎖利、死錢釋放——評分轉弱只在獲利時了結（虧損中續抱、不在低點殺出）。
> 帳戶層保險絲：7 天 3 次硬停損→冷卻 3 天；回撤 ≥10% 或大盤 risk_off →暫停新倉；
> 贏家每 +1R 加碼最多 2 次。引擎參數可用 `/set eng_<參數> 值` 調（如 `/set eng_trail_pct 0.1`），
> `/protections` 查保險絲狀態。用 `/positions`、`/pnl` 查績效，`/closeall` 一鍵平倉。
> 純模擬不涉真錢。

> ☀️ **每日 AI 晨報**：每交易日 ET 08:30 自動推送大盤+觀察清單評分+訊號+最強標的內部人亮點。
> 設了 `LLM_API_KEY` 會多一段 AI 白話解讀；沒設則只推數據排名（仍可用）。
> 用 `/briefing` 可隨時手動測試、`/set briefing_enabled off` 關閉。

---

### Telegram 指令（傳給 Bot，下次掃描時處理；輸入 `/help` 看全部）

| 分類 | 指令 |
|------|------|
| 清單 | `/add AAPL`、`/remove AAPL`、`/list` |
| 分析 | `/rank`、`/fundamentals AAPL`（`/f`）、`/options AAPL`（`/opt`，選擇權情緒）、`/insider AAPL`（`/ins`，SEC 內部人）、`/whales [編號]`（13F 大戶動向）、`/earnings [天數]`、`/briefing`、`/weekly`（每週深度週報）、`/today [帳戶 風險%]`（`/plan`）— 當日交易計畫：VWAP/ORB/RVOL 訂單票（進場/停損/停利/股數；財報日自動迴避；進場票記入決策計分板隔日結算；排程版最多延遲 ~15 分處理）、`/plantest [apply\|clear]`（當日計畫 60 日回測；apply 套用 walk-forward 校準到 /today；每週自動重跑）、`/plantest opt [apply]`（參數尋優：ORB×停損×R:R 掃 27 組，驗證段勝過預設才推薦） |
| 警報 | `/alert AAPL 200`（到價通知，觸發自動移除）、`/alert`（清單）、`/alert del AAPL` |
| AI | `/committee NVDA`（`/cmt`）— 機構決策會議：分析師×4→多空對辯→交易員→風控→投資經理，裁決自動記入計分板（需 `LLM_API_KEY`，約 1-3 分鐘） |
| 風控 | `/risk [帳戶 風險%]`、`/protections`、`/calibrate` |
| 模擬交易 | `/autotrade on\|off`、`/positions`、`/pnl`、`/journal [N]`、`/rebalance [hrp\|max_sharpe\|min_vol\|erc\|equal]`（持倉再平衡顧問）、`/closeall` |
| 估值 | `/dcf AAPL [成長%]` — DCF 內在價值（FCF→WACC→期中折現→終值→隱含股價；可覆蓋成長率假設） |
| 情緒/籌碼 | `/fg`（雙恐懼貪婪：美股 CNN+加密，晨報自動附一行）、`/taifex`（台指期三大法人淨未平倉 + 選擇權 P/C 比） |
| 論點/財報 | `/thesis [TICKER 多\|空 論點 / pillar / risk / cat / target / stop / conv / note / close]`（論點追蹤，失效價自動監測）、`/preview TICKER`（財報前瞻/覆盤自動判定） |
| 反駁器 | `/falsify TICK1,TICK2 [vs 基準] [持有日] [多\|空] 故事`（8 類反駁測試；口語模式需 LLM key）、`/falsify trials +K`（自報場外試錯餵 DSR）、`/falsify ledger`（假設帳本） |
| 基金 | `/fund QQQ [vs SMH]`（費用率/追蹤誤差/α β/捕獲率）、`/fund overlap QQQ,VGT`（兩檔持股重疊度） |

> `/options`、`/insider` 僅美股；選擇權走 yfinance、內部人走 SEC EDGAR，皆免 key。

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
  - cron: '*/15 * * * *'   # 每 15 分鐘執行
```

每 15 分鐘掃描一次，因此你傳的指令最多等 ~15 分鐘會被處理（要秒級即時回應請改用
[常駐版](PERSISTENT_BOT.md)）。晨報僅在每交易日 ET 08:30 之後的那次掃描推送一次。

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

**Q: 想調整掃描頻率？**
A: 預設 `'*/15 * * * *'`（每 15 分鐘）。yfinance 本身有 ~15 分鐘延遲，再高頻無實質意義；
想省額度可改 `'*/30 * * * *'`（每 30 分鐘）或 `'0 * * * *'`（每小時）。

**Q: 免費 GitHub Actions 額度夠嗎？**
A: **公開 repo 無限制**（本專案即是）。私人 repo 每月 2000 分鐘免費，每次掃描約 1-2 分鐘、
每 15 分鐘一次 ≈ 月用 ~2900-5800 分鐘會超額，私人 repo 建議改每小時或用[常駐版](PERSISTENT_BOT.md)。

---

## 擴充方向

- **更多指標**: 在 `scan_signals.py` 的 `scan()` 函數中加入 MACD、布林通道等
- **Email 備援**: 在 `main()` 裡加入 `smtplib.SMTP_SSL` 作為 Telegram 失敗時的備援
- **Discord**: 改用 Discord Webhook 替代 Telegram
- **標的分組**: 區分個股 / ETF / 加密貨幣分組掃描，各自有不同門檻
