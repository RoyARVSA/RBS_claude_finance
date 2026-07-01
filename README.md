# RBS Finance Dashboard

整合式金融分析平台：即時市場儀表板 + 訊號回測 + 自我優化的 Telegram 警報 Bot。

Streamlit 網頁應用 + 獨立的訊號掃描 Bot（GitHub Actions 排程版 / VPS 常駐版），
涵蓋市場總覽、風險管理、股票研究、回測、部位配置與即時警報。

---

## 功能總覽

### 📊 網頁儀表板（`app.py`，10 個頁面）

| 頁面 | 功能 |
|------|------|
| 🏠 市場總覽 | 即時指數/板塊/宏觀指標、**總經數據(FRED)**、市場快訊、**AI 自主市場分析** |
| 📈 持倉分析 | 多資產組合：權益曲線、回撤、Sharpe、IR、Beta |
| ⚠️ 風險管理 | VaR/CVaR、蒙地卡羅、Kupiec 回測、壓力測試、**風險平價配置** |
| 🔍 股票研究 | K 線+RSI、AI 深度報告、市場篩選器、**訊號回測 + 參數最佳化** |
| 🏢 公司分析 | **基本面體質：財務健康評分、估值旗標、三表趨勢、AI 解讀** |
| 🚨 即時警報 | 監控清單、盤中走勢、訊號掃描、Telegram/Email 推播 |
| 🛠️ 交易工具 | 部位大小、**波動率目標部位**、Kelly、風險報酬比、複利 |
| 🏦 機構選股 | 6 步驟系統化選股（市場→策略→宏觀→資產類型→產業→標的）|
| 📰 新聞情報 | 多來源 RSS 聚合、LLM 情緒分析、金融報告生成 |
| 💳 信用模型 | WoE/IV、邏輯回歸、KS/AUC、評分卡 |
| 📦 匯出報告 | 圖表 + KPI 表打包下載 |

### 🤖 訊號掃描 Bot（`scan_signals.py` / `bot_daemon.py`）

- **技術訊號**：RSI、MACD、布林通道、ATR 進出場、MA 交叉、爆量
- **綜合評分**：趨勢/MACD/RSI/布林/動量合成 -1~+1 分數，5 級評級
- **多時間框架確認**：日線分數與週線同向加強、背離減弱（軟性調整）
- **每日 AI 晨報**：每交易日 ET 08:30 自動推送大盤+排名+訊號
- **自我優化迴圈**：回測各訊號歷史勝率 → 動態調整評分權重（每週校準）
- **防護機制**（freqtrade 式）：訊號冷卻去重、大盤風險濾網
- **部位建議**：每個訊號附 ATR 風險基準的建議股數
- **Telegram 指令**：`/add`、`/rank`、`/calibrate`、`/risk`、`/protections`、`/fundamentals` 等

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

## 檔案結構

```
app.py                  Streamlit 主程式（10 頁）
scan_signals.py         Bot 核心：訊號/評分/校準/防護/指令（排程版進入點）
bot_daemon.py           常駐版 Bot（重用 scan_signals 全部邏輯）
backtest.py             Triple-Barrier 回測 + walk-forward + 參數最佳化
quant_tools.py          部位配置與風險管理（部位/Kelly/風險平價）
fundamentals.py         公司基本面：抓取 + 財務健康評分 + 估值旗標 + 財報日
macro.py                總經數據（FRED）：Fed利率/殖利率曲線/CPI/失業率 + 判讀
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
- **單一真實來源**：Bot 與儀表板共用同一套訊號/回測/部位邏輯

> ⚠️ 本平台為分析與教育用途，所有回測為歷史模擬，不構成投資建議。
>
> 註：多時間框架（週線）確認同時用於**即時評分/通知**與**回測**
> （回測頁可勾選「加入週線確認」對照有/無 MTF 的勝率差異，無前視偏誤）。
