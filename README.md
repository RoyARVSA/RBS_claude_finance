# RBS Finance Dashboard

整合式金融分析平台：即時市場儀表板 + 訊號回測 + 自我優化的 Telegram 警報 Bot + Alpaca 模擬交易。

Streamlit 網頁應用 + 獨立的訊號掃描 Bot（GitHub Actions 排程版 / VPS 常駐版），
涵蓋市場總覽、風險管理、技術面/基本面/總經分析、回測、部位配置、即時警報與策略模擬驗證。

---

## 功能總覽

### 📊 網頁儀表板（`app.py`，12 個頁面）

| 頁面 | 功能 |
|------|------|
| 🏠 市場總覽 | 即時指數/板塊/宏觀指標、**總經數據(FRED)**、市場快訊、**AI 自主市場分析** |
| 📈 持倉分析 | 多資產組合：權益曲線、回撤、Sharpe、IR、Beta |
| ⚠️ 風險管理 | VaR/CVaR、蒙地卡羅、Kupiec 回測、壓力測試、**風險平價配置** |
| 🔍 股票研究 | K 線+RSI、AI 深度報告、市場篩選器、**訊號回測 + 參數最佳化** |
| 🏢 公司分析 | **基本面體質：財務健康評分、估值旗標、三表趨勢、AI 解讀** |
| 🗂️ 產業總覽 | **一次掃描整個市場：產業強弱 vs 風險散佈、鑽取個股（可加基本面）** |
| 🚨 即時警報 | 監控清單、盤中走勢、訊號掃描、Telegram/Email 推播 |
| 🛠️ 交易工具 | 部位大小、**波動率目標部位**、Kelly、風險報酬比、複利 |
| 📉 模擬交易 | **Alpaca 紙上交易：帳戶績效、持倉、權益曲線 vs SPY** |
| 🏦 機構選股 | 6 步驟系統化選股（市場→策略→宏觀→資產類型→產業→標的）|
| 📰 新聞情報 | 多來源 RSS 聚合、LLM 情緒分析、金融報告生成 |
| 📦 匯出報告 | 圖表 + KPI 表打包下載 |

### 🤖 訊號掃描 Bot（`scan_signals.py` / `bot_daemon.py`）

- **技術訊號**：RSI、MACD、布林通道、ATR 進出場、MA 交叉、爆量
- **綜合評分**：趨勢/MACD/RSI/布林/動量合成 -1~+1 分數，5 級評級
- **多時間框架確認**：日線分數與週線同向加強、背離減弱（軟性調整）
- **每日 AI 晨報**：每交易日 ET 08:30 自動推送大盤+排名+訊號
- **自我優化迴圈**：回測各訊號歷史勝率 → 動態調整評分權重（每週校準）
- **防護機制**（freqtrade 式）：訊號冷卻去重、大盤風險濾網
- **部位建議**：每個訊號附 ATR 風險基準的建議股數
- **財報行事曆提醒**：觀察清單標的財報前 N 天自動提醒（晨報 + `/earnings`）
- **Alpaca 模擬交易**：訊號自動下模擬單驗證策略實效（預設關閉，`/autotrade on` 啟用）
- **Telegram 指令**：清單 `/add /remove /list`、分析 `/rank /fundamentals /earnings /briefing`、
  風控 `/risk /protections /calibrate`、模擬交易 `/autotrade /positions /pnl /closeall`（`/help` 看全部）

### 🧪 回測引擎（`backtest.py`）

- **Triple-Barrier 三重關卡法**（López de Prado）：停利/停損/時間三道關卡
- **無前視偏誤**：下一根 K 棒進場；**已扣交易成本**
- **Walk-forward 樣本外驗證**：偵測過擬合，折減不穩定訊號
- **參數最佳化**：網格搜尋最佳停利/停損/持有，目標函數含一致性懲罰

### ⚙️ 量化工具（`quant_tools.py`）

ATR 部位、波動率目標、Kelly、反波動加權、等風險貢獻（ERC）風險平價。

---

## 快速開始

### 網頁（本機）
```bash
pip install -r requirements.txt
streamlit run app.py
```

### 網頁（雲端部署）
見 [`DEPLOY.md`](DEPLOY.md) — Streamlit Cloud（推薦）/ HF Spaces / Colab / VPS。

### Telegram Bot
- **排程版**（免費，無需主機）：見 [`GITHUB_ACTIONS.md`](GITHUB_ACTIONS.md)
- **常駐版**（即時回應，需主機）：見 [`PERSISTENT_BOT.md`](PERSISTENT_BOT.md)

---

## API Keys（皆為選填、皆免費）

核心功能（技術訊號、回測、風險）**不需任何 key**。以下為進階功能所需，缺了會優雅退回：

| Key | 用途 | 缺了會怎樣 | 申請 |
|-----|------|-----------|------|
| `TELEGRAM_TOKEN` + `TELEGRAM_CHAT_ID` | Bot 推播 | Bot 無法推送 | @BotFather |
| `LLM_API_KEY` | AI 分析 / 晨報解讀 | 晨報走純數據版；AI 頁不可用 | Claude / OpenAI |
| `FRED_API_KEY` | 總經數據 | 總經指標區塊不顯示 | [fred.stlouisfed.org](https://fred.stlouisfed.org/) |
| `ALPACA_KEY_ID` + `ALPACA_SECRET_KEY` | 模擬交易（**paper**）| 模擬交易不執行 | [alpaca.markets](https://alpaca.markets/) Trading API |

**放哪裡：**
- **Bot（GitHub Actions）** → repo Settings → Secrets and variables → Actions
- **網頁（Streamlit Cloud）** → Manage app → Settings → Secrets（TOML），或直接在網頁欄位當場輸入

> 🔒 **公開 repo 安全守則**：key 只放 Secrets 或當場輸入，**絕不寫進任何 commit 的檔案**。
> `.gitignore` 已排除 `.env`、`secrets.toml`、`alerts_config.json` 等敏感檔。

---

## 檔案結構

```
app.py                  Streamlit 主程式（12 頁）
scan_signals.py         Bot 核心：訊號/評分/校準/防護/指令（排程版進入點）
bot_daemon.py           常駐版 Bot（重用 scan_signals 全部邏輯）
backtest.py             Triple-Barrier 回測 + walk-forward + 參數最佳化
quant_tools.py          部位配置與風險管理（部位/Kelly/風險平價）
fundamentals.py         公司基本面：抓取 + 財務健康評分 + 估值旗標 + 財報日
macro.py                總經數據（FRED）：Fed利率/殖利率曲線/CPI/失業率 + 判讀
sector_scan.py          產業總覽：批次掃描 stock_db 全市場 + 產業風險彙總
alpaca_trader.py        Alpaca 紙上交易：下單決策(decide_orders) + REST client
stock_db.py             選股資料庫（5 市場、30+ 產業、200+ 標的）
rbs_lib.py              風險計算函式庫（VaR/CVaR/共變異數/情境）
streamlit_app.py        雲端部署進入點
.github/workflows/      GitHub Actions 排程掃描
watchlist_state.json    Bot 狀態（清單/門檻/校準/訊號歷史，自動維護）
```

文件：[`DEPLOY.md`](DEPLOY.md)（部署）、[`GITHUB_ACTIONS.md`](GITHUB_ACTIONS.md)（排程 Bot）、
[`PERSISTENT_BOT.md`](PERSISTENT_BOT.md)（常駐 Bot）。

---

## 設計理念

對照成熟開源量化專案（freqtrade、intelligent-trading-bot、Riskfolio-Lib）的優點：

- **誠實的回測**：無前視、扣成本、樣本外驗證 — 避免過擬合自欺
- **自我優化**：訊號權重隨歷史勝率動態調整，但用一致性抑制雜訊追逐
- **風險優先**：部位依波動配置、組合等風險貢獻、防護機制避免洗版
- **多維分析**：技術面（訊號/回測）+ 基本面（財務健康）+ 總經（FRED）三面向
- **策略驗證閉環**：訊號 → Alpaca 模擬下單 → 追蹤真實模擬績效，驗證訊號實效
- **單一真實來源**：Bot 與儀表板共用同一套訊號/回測/部位邏輯

> ⚠️ 本平台為分析與教育用途，所有回測為歷史模擬，不構成投資建議。
>
> 註：多時間框架（週線）確認同時用於**即時評分/通知**與**回測**
> （回測頁可勾選「加入週線確認」對照有/無 MTF 的勝率差異，無前視偏誤）。
