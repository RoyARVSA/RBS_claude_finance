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
| `FINNHUB_API_KEY` | **建議** | 基本面備援 + **內部人交易備援**：SEC EDGAR 封鎖 GitHub 機房 IP 時（2026-08 實測發生），/ins、晨報亮點與 Alpha 疊加層的內部人資料改走 Finnhub（免費申請 finnhub.io，60 次/分） |
| `SEC_USER_AGENT` | **建議** | SEC 內部人交易（Form 4）的 User-Agent。SEC 公平使用政策要求「`名字 email`」格式，**GitHub Actions 的雲端 IP 配預設 UA 容易被 SEC WAF 拒（403）→ 內部人資料靜默缺席**。設成如 `你的名字 your-email@example.com`（用你真實信箱）可解；免申請、只是自我識別 |
| `ALPACA_KEY_ID` | 選填 | Alpaca **paper** trading key（模擬自動交易；**不設則不下單**） |
| `ALPACA_SECRET_KEY` | 選填 | Alpaca paper secret |
| `ALPHA_VANTAGE_KEY` | 選填 | Alpha Vantage **免費 key**（alphavantage.co/support 填 email 即得，不需 premium）。只用 `EARNINGS` 端點回填每季「公告日共識 vs 實際」的財報驚奇歷史（長壽股回溯至 1996）；免費層 25 次/日由程式計數、用罄自動停；不設則 `/est` 少「財報驚奇」一段 |
| `STATE_ENC_KEY` | **建議** | 敏感區塊加密金鑰（自訂任意長隨機字串）。設了之後，投資論點/帳戶淨值/引擎簿記/策略參數在 commit 前加密，公開 repo 只見密文。**須同步設進 Streamlit Secrets**（網頁端才能解讀）。⚠️ **key 遺失＝加密資料永久無法還原**；**換 key＝舊密文鎖死，輪替前務必先在舊 key 環境解密取回**（key 設定後解不開會發 TG 告警）。防瀏覽級保護（HMAC-SHA256 keystream），過去已 commit 的明文歷史仍在 git 內 |

> 🤖 **Alpaca 模擬交易**：預設**關閉**，須 Telegram 傳 `/autotrade on` 才會下單，
> 且僅在美股開盤時、由**分層交易引擎**（`trade_engine.py`，重製自 freqtrade / Lean /
> nautilus 機制）自動下*模擬*單：評分只決定進場；出場走硬停損、+1R 保本追蹤停損、
> +1.5R 分批鎖利、死錢釋放——評分轉弱只在獲利時了結（虧損中續抱、不在低點殺出）。
> 帳戶層保險絲：7 天 3 次硬停損→冷卻 3 天；回撤 ≥10% 或大盤 risk_off →暫停新倉；
> 贏家每 +1R 加碼最多 2 次（加碼量＝min(半倍、空間、現金)，可用空間至單檔
> 15%×1.3＝19.5%——僅贏家加碼適用，初始進場仍限 15%）。引擎參數可用 `/set eng_<參數> 值` 調（如 `/set eng_trail_pct 0.1`），
> `/protections` 查保險絲狀態。用 `/positions`、`/pnl` 查績效，`/closeall` 一鍵平倉。
> 進場評分另疊加 **Alpha 資訊層**（`/alpha` 查看）：SEC 內部人 cluster buy 加分、
> 選擇權情緒傾斜、空單占流通 ≥15% 降評、財報前 3 天禁新倉、雙恐貪極貪→新倉減半；
> 12 小時快取、每輪最多抓 4 檔輪替（`/set ao_<參數> 值` 可調）。
> **Portfolio 層**再加相關性控制：新倉與持倉平均相關 ≥0.75 縮半、≥0.85 跳過
> （`/set corr_hi 0.9`、`/set corr_mid 0.8` 可調）；大盤濾網升級為**市場氣象台**
> 五因子體質分（`/weather` 查看）。純模擬不涉真錢。

> ☀️ **每日 AI 晨報**：每交易日 ET 08:30 自動推送大盤+觀察清單評分+訊號+最強標的內部人亮點。
> 設了 `LLM_API_KEY` 會多一段 AI 白話解讀；沒設則只推數據排名（仍可用）。
> 用 `/briefing` 可隨時手動測試、`/set briefing_enabled off` 關閉。

---

### Telegram 指令（傳給 Bot，下次掃描時處理；輸入 `/help` 看全部）

| 分類 | 指令 |
|------|------|
| 清單 | `/add AAPL`、`/remove AAPL`、`/list` |
| 分析 | `/rank`、`/fundamentals AAPL`（`/f`）、`/options AAPL`（`/opt`，選擇權情緒）、`/insider AAPL`（`/ins`，SEC 內部人）、`/whales [編號]`（13F 大戶動向）、`/earnings [天數]`、`/briefing`、`/weekly`（每週深度週報）、`/today [帳戶 風險%]`（`/plan`）— 當日交易計畫：VWAP/ORB/RVOL 訂單票（進場/停損/停利/股數；財報日自動迴避；進場票記入決策計分板隔日結算；排程版最多延遲 ~15 分處理）、`/plantest [apply\|clear]`（當日計畫 60 日回測；apply 套用 walk-forward 校準到 /today；每週自動重跑）、`/plantest opt [apply]`（參數尋優：ORB×停損×R:R 掃 27 組，三段 walk-forward、holdout 把關通過才推薦） |
| 警報 | `/alert AAPL 200`（到價通知，觸發自動移除）、`/alert`（清單）、`/alert del AAPL` |
| AI | `/committee NVDA`（`/cmt`）— 機構決策會議：分析師×4→多空對辯→交易員→風控→投資經理，裁決自動記入計分板（需 `LLM_API_KEY`，約 1-3 分鐘） |
| 風控 | `/risk [帳戶 風險%]`、`/protections`、`/calibrate` |
| 模擬交易 | `/autotrade on\|off`、`/alpha`（資訊疊加層現況）、`/positions`、`/pnl`、`/journal [N]`、`/checkup`（行為體檢：追高/頻率/太早出場/持有期）、`/attrib`（機制歸因：各出場/進場機制損益/勝率/賣後追蹤）、`/shadow`（舊邏輯 vs 新引擎對照）、`/mirror [init 現金 代碼:股數:成本…\|reset]`（鏡像帳：引擎接管你的實倉起點自主模擬；`/attrib mirror`、`/checkup mirror` 看鏡像帳版歸因/體檢）、`/universe [rebuild]`（選股池快照：yf.screen 寬宇宙→品質/流動性/12-1 動能→候選前 N；每月自動重建、快照落 `data/universe/`；P0 只顯示）、`/est [TICKER]`（分析師預估快照：共識/修正動能/目標價/評等/財報驚奇史；每輪自動輪替刷新到 `estimates_ledger.json`，無參數看上修下修排行）、`/engtest [3m\|6m\|1y\|2y]`（整台引擎歷史重放：現行參數過去 N 個月報酬/回撤/機制分佈，次日開盤成交含成本、對照 SPY）、`/engtest opt [期間] [apply]`（引擎參數學習：108 組 × 三段 walk-forward + DSR，holdout 通過才推薦；apply 寫入 eng_* 參數、`/engtest clear` 還原）、`/rebalance [hrp\|max_sharpe\|min_vol\|erc\|equal]`（持倉再平衡顧問）、`/closeall` |
| 候選/治理 | `/screen` — **候選篩選（P5）**：選股池動能前 N ∪ stock_db AI 供應鏈主題 − watchlist；Stage 3 每次 ≤8 檔補品質分（PIT 三表）與修正動能（yfinance 預估），加上選股池 12-1 動能 → 綜合分（品質 0.4／修正 0.3／動能 0.3，缺成分只降信心）；品質否決剔除；每個閉市日一批（≤8 檔）自動刷新 `state["screen"]`，單檔 7 天更新；**只建議，`/add` 後才進建模輪替與佈局計畫**、`/valreport` — **估值治理月報（P6）**：覆蓋/過期/待審/建不了模、各判定（accumulate/hold/trim/exit）列日到今日的事後報酬與命中率、公允價值穩定度（近 8 列變異係數）、MoS 因子 21/63 日 IC（有效期數與 Newey-West t）、指引萃取覆蓋；每月第一個閉市輪自動推播（`/set valreport_enabled off` 關） |
| 指引 | `/guidance TICKER [季別]` — **指引/KPI 萃取（P4）**：Alpha Vantage `EARNINGS_CALL_TRANSCRIPT`（免費 key、占 25 次/日配額 1 次）→ 便宜 LLM 只做「定位 + 逐字轉錄」到封閉列舉 schema（revenue/eps/gross_margin/op_margin/capex/backlog/rpo/book_to_bill/fcf/segment_revenue…）→ **程式驗證**：quote 必須逐字存在原文、low/high 必須能由 quote 解析、修訂（raise/lower/maintain）由程式比對上一期中點 ±1%、實際值與財報對帳（差 >2% 標疑非 GAAP）、原文視為不受信任輸入（注入無效）、JSON 壞掉 → 棄權不猜。通過項目加密存 state 供論點監測 |
| 估值層接引擎 | `/set val_enabled on\|off`（預設關）— val_hist 的 MoS → 部位乘數 0.5–1.25×（只乘風險預算）、市價高於牛市情境不加碼、MoS>30% 提早加碼（0.75R）、候選排序傾斜 ±0.1；**不觸發進場、不否決出場**。`/engtest opt` 會自動加入「估值層 開/關」A/B 維度（PIT 由 val_hist 列日期保證），holdout 沒贏就別開；`/rebalance bl` — Black-Litterman 用公允價值當觀點（信心＝情境寬度）配置權重 |
| 佈局 | `/playbook [build N]` — **佈局計畫整合層**：選股池 → 分析師修正動能 → 公司模型（MoS／區間位置／判定）→ 品質旗標 → 技術評分 → 大盤 regime → 持倉 → 論點，輸出每檔四象限（估值/論點 × 價格/動能）、層級（迴避／減碼／累積候選／持有／觀察）、conviction 與權重帶（單檔 ≤10%、主題 ≤25%、依 regime 打折）、現金目標；`build N` 逐批建模未建模的 watchlist；閒置輪每 7 天自動輪替更新模型（`/set model_auto_refresh off` 關）；週報自動附摘要。**全部為參考、不下單、不改引擎** |
| 估值 | `/model TICKER [set k=v…\|clear]` — 公司模型引擎：PIT 三表（yfinance 主、Finnhub as-reported 備援）→ 驅動推導（共識/指引覆蓋前兩年、3 年均值、線性淡出）→ 專業 WACC（Blume β、合成信評利差、市值權重）→ 5+5 年 FCFF、價值中性終值（再投資率 = g/ROIC）→ 熊/基/牛 + 蒙地卡羅 + WACC×g 敏感度 + 反向 DCF（市價隱含成長）→ 九條審核（未過=待審）→ 品質評分（Piotroski/Altman/Beneish/Sloan/ROIC 價差）→ MoS 分層門檻判定；金融股自動走 RIM（剩餘收益）、地產走股利 H-model；`set` 覆蓋驅動（`opm_target=0.24 beta=1.4 rev_g=0.3,0.25,0.2 guidance_rev=14000`；RIM 用 `payout=0.6 roe_target=0.15`、DDM 用 `div_g=0.04`）、`clear` 還原；估值歷史加密存 state；`/model` 看清單。**只調部位不觸發進場（P3 才接引擎）**、`/dcf AAPL [成長%]` — 簡版 DCF 內在價值（FCF→WACC→期中折現→終值→隱含股價；可覆蓋成長率假設） |
| 情緒/籌碼 | `/fg`（雙恐懼貪婪：美股 CNN+加密，晨報自動附一行）、`/taifex`（台指期三大法人淨未平倉 + 選擇權 P/C 比）、`/weather`（市場氣象台：廣度/信用利差/VIX 期限/曲線/銅金五因子體質分，大盤濾網 v2） |
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
