# RBS Finance Dashboard

整合式金融分析平台：即時市場儀表板 + 訊號回測 + 自我優化的 Telegram 警報 Bot
+ Alpaca 模擬交易 + AI 研究副駕（資深分析師模式）。

Streamlit 網頁應用 + 獨立的訊號掃描 Bot（GitHub Actions 排程版 / VPS 常駐版），涵蓋
市場總覽、風險管理、技術/基本面/總經分析、回測、部位配置、**選擇權情緒與 SEC 內部人籌碼**、
即時警報與策略模擬驗證，並內建 **AI 供應鏈「瓶頸」主題**選股宇宙。

---

## 功能總覽

### 📊 網頁儀表板（`app.py`，13 個頁面）

| 頁面 | 功能 |
|------|------|
| 💬 AI 助理 | **資深分析師模式＋🏛 機構決策委員會（分析師×4→多空對辯→交易員→硬風控→投資經理，與量化評分交叉比較；支援多檔同場會議、深度模式(研究主管+風控對辯)、新聞逐篇多空標注、會議紀錄下載、Telegram 推播）＋📊 決策計分板（量化 vs 委員會誰更準）＋單輪多空對辯＋反思記憶** |
| 🏠 市場總覽 | 即時指數/板塊/宏觀指標、**總經數據(FRED)**、市場快訊、**AI 自主市場分析**、**🩺 資料源健康檢查（9 大數據源一鍵驗收）** |
| 📈 持倉分析 | 多資產組合：權益曲線、回撤、Sharpe、IR、Beta、**績效報告（CAGR/Sortino/Calmar/月報酬熱力圖/回撤期間）**、**交易帳本（成本基礎/TWR/XIRR/股息收入）**、**⚖️ 再平衡顧問（HRP/最大 Sharpe/風險平價 目標權重 → 具體加減碼股數清單）** |
| ⚠️ 風險管理 | VaR/CVaR、蒙地卡羅、Kupiec 回測、壓力測試、**風險平價 + 效率前緣（MPT）+ HRP 階層風險平價** |
| 🔍 股票研究 | K 線+RSI、AI 深度報告、市場篩選器、**訊號回測 + 參數最佳化（含參數熱力圖與 walk-forward 逐段檢視）**、TradingView、**選擇權情緒(Put/Call、IV 偏斜)** |
| 🏢 公司分析 | **基本面體質：財務健康評分、估值旗標、三表趨勢、AI 解讀、分析師共識+EPS Beat 率、做空籌碼(FINRA/FTD)、台股三大法人買賣超(上市 TWSE + 上櫃 TPEX)、SEC 內部人交易(Form 4)、📄 一鍵完整研究報告（彙整全部區塊下載）** |
| 🗂️ 產業總覽 | **一次掃描整個市場：產業強弱 vs 風險散佈、鑽取個股（可加基本面）、RRG 板塊輪動象限圖** |
| 🚨 即時警報 | 監控清單、盤中走勢、訊號掃描、Telegram/Email 推播、**🎯 當日交易計畫（VWAP/ORB/RVOL 盤中訂單票：進場區間/停損/停利/股數，財報日自動迴避，可選 Alpaca IEX 即時價，一鍵送模擬 bracket 單）** |
| 🛠️ 交易工具 | 部位大小、**波動率目標部位**、Kelly、風險報酬比、複利 |
| 📉 模擬交易 | **Alpaca 紙上交易：帳戶績效、持倉、權益曲線 vs SPY、交易日誌（原因）、訊號實測勝率** |
| 🏦 機構選股 | 6 步驟系統化選股（市場→策略→宏觀→資產類型→產業→標的）、**超級投資人 13F 持倉動向** |
| 📰 新聞情報 | 多來源 RSS 聚合、LLM 情緒分析、金融報告生成 |
| 📦 匯出報告 | 圖表 + KPI 表打包下載 |

### 🤖 訊號掃描 Bot（`scan_signals.py` / `bot_daemon.py`）

- **技術訊號**：RSI、MACD、布林通道、ATR 進出場、MA 交叉、爆量
- **綜合評分**：趨勢/MACD/RSI/布林/動量合成 -1~+1 分數，5 級評級
- **多時間框架確認**：日線分數與週線同向加強、背離減弱（軟性調整）
- **每日 AI 晨報**：每交易日 ET 08:30 推送大盤+排名+訊號+內部人亮點+AI 判斷回顧+**本週總經發布日（CPI/非農/GDP）**
- **每週深度週報**：週日 ET 晚間自動推送——指數週表現、清單強弱、**決策計分板**、RRG 板塊輪動、下週財報/總經行事曆（`/weekly` 隨時手動）
- **自我優化迴圈**：回測各訊號歷史勝率 → 動態調整評分權重（每週校準）
- **防護機制**（freqtrade 式）：訊號冷卻去重、大盤風險濾網
- **部位建議**：每個訊號附 ATR 風險基準的建議股數
- **財報行事曆提醒**：觀察清單標的財報前 N 天自動提醒（晨報 + `/earnings`）
- **到價警報**：`/alert AAPL 200` 突破/跌破即推播，觸發後自動移除（上限 20 個）
- **Alpaca 模擬交易**：訊號自動下模擬單驗證策略實效（預設關閉，`/autotrade on` 啟用）
- **當日交易計畫**：`/today [帳戶 風險%]` 盤中訂單票（VWAP/ORB/RVOL 進場、停損/停利/股數、財報日迴避）；進場票自動記入決策計分板（隔日結算，與量化/委員會同板比較）
- **當日計畫歷史回測**：`/plantest` 用過去 ~60 交易日 5 分 K 逐日重放訂單票（無前視、扣成本、停損優先），統計各型態實證勝率/R 期望值；`/plantest apply` 把 walk-forward 校準（負期望型態停用、不穩定降信心）套進 /today——**讓判定吃歷史實證自我修正**
- **Telegram 指令**：清單 `/add /remove /list`、分析 `/rank /fundamentals /options /insider /whales /earnings /briefing /weekly /today`、
  **AI `/committee`（手機開機構決策會議）**、警報 `/alert`、風控 `/risk /protections /calibrate`、
  模擬交易 `/autotrade /positions /pnl /journal /closeall`（`/help` 看全部）

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
| `FINNHUB_API_KEY` | 基本面備援 | yfinance `.info` 被限流時，市值/P/E/ROE 顯示「—」 | [finnhub.io](https://finnhub.io/) |
| `ALPACA_KEY_ID` + `ALPACA_SECRET_KEY` | 模擬交易（**paper**）| 模擬交易不執行 | [alpaca.markets](https://alpaca.markets/) Trading API |
| `GITHUB_TOKEN` | 決策計分板持久化（委員會紀錄 commit 進 repo） | 紀錄只存本地，app 重啟即消失 | GitHub → Fine-grained PAT，**只授權本 repo 的 Contents 讀寫**（勿用全域 classic token） |

> SEC 內部人交易（Form 4）走 EDGAR、選擇權情緒走 yfinance，兩者**皆免 key**。
> SEC 可選設 `SEC_USER_AGENT` 自訂聯絡用 User-Agent（選填，有預設值即可用）。

**放哪裡：**
- **Bot（GitHub Actions）** → repo Settings → Secrets and variables → Actions
- **網頁（Streamlit Cloud）** → Manage app → Settings → Secrets（TOML），或直接在網頁欄位當場輸入

> 🔒 **公開 repo 安全守則**：key 只放 Secrets 或當場輸入，**絕不寫進任何 commit 的檔案**。
> `.gitignore` 已排除 `.env`、`secrets.toml`、`alerts_config.json` 等敏感檔。

---

## 檔案結構

```
app.py                  Streamlit 主程式（13 頁）
scan_signals.py         Bot 核心：訊號/評分/校準/防護/指令（排程版進入點）
bot_daemon.py           常駐版 Bot（重用 scan_signals 全部邏輯）
backtest.py             Triple-Barrier 回測 + walk-forward + 參數最佳化
quant_tools.py          部位配置與風險管理（部位/Kelly/風險平價）
fundamentals.py         公司基本面：抓取 + 財務健康評分 + 估值旗標 + 財報日
macro.py                總經數據（FRED）：Fed利率/殖利率曲線/CPI/失業率 + 判讀
finnhub_data.py         基本面備援（Finnhub）：yfinance 被限流時後援市值/P/E/ROE
sector_scan.py          產業總覽：批次掃描 stock_db 全市場 + 產業風險彙總
assistant.py            對話式 AI 助理：代碼/意圖解析 + grounded context builder
assistant_tools.py      AI 助理工具編排：規劃(plan)/解析/格式化，讓助理自跑回測/風險/選股/選擇權
options_sentiment.py    選擇權情緒：Put/Call 比、ATM 隱含波動、偏斜 + 情緒評分（CBOE 免 key 備援）
sec_insider.py          SEC 內部人交易：Form 4 XML 解析 + 買賣彙總（cluster buy）+ 情緒
perf_report.py          績效報告（quantstats 風格）：CAGR/Sortino/Calmar/月報酬表/回撤期間
portfolio_opt.py        效率前緣（MPT）：最小波動/最大 Sharpe 權重 + 前緣曲線（scipy）
analyst_data.py         分析師共識：評等分佈/目標價上檔/EPS surprise 歷史（yfinance+Finnhub）
short_data.py           做空籌碼：FINRA 日做空量 + 短倉/回補天數 + SEC 失券 FTD（免 key）
whales_13f.py           超級投資人 13F：EDGAR 13F-HR 解析 + 兩季增減倉比較（免 key）
ledger.py               交易帳本：平均成本/已未實現損益/TWR/XIRR/股息收入（Ghostfolio 式）
reflection.py           AI 反思記憶：判斷 vs N 日後結果 → 命中率 + 決策者計分板（FinMem 式）
trade_plan.py           當日交易計畫：VWAP/ORB/RVOL 盤中訂單票 + Alpaca IEX 即時價備援（免費）
rebalance.py            持倉再平衡顧問：現有持倉 vs 目標權重 → 加減碼清單（免費）
plan_backtest.py        當日計畫歷史回測：60 日 5 分 K 逐日重放 + walk-forward 校準（免費）
tw_flows.py             台股三大法人買賣超（TWSE T86 上市 + TPEX 上櫃，免 key）：外資/投信/自營 + 連買天數
committee.py            機構決策委員會：角色提示/立場解析/硬風控閘門/量化交叉比較，支援多檔與深度模式（TradingAgents 式）
alpaca_trader.py        Alpaca 紙上交易：下單決策(decide_orders) + REST client
stock_db.py             選股資料庫（5 市場、30+ 產業、200+ 標的，含 AI 供應鏈瓶頸主題）
rbs_lib.py              風險計算函式庫（VaR/CVaR/共變異數/情境）
streamlit_app.py        雲端部署進入點
.github/workflows/      GitHub Actions：排程掃描（signal_scan）+ CI 離線自測（ci）
watchlist_state.json    Bot 狀態（清單/門檻/校準/訊號歷史，自動維護）
```

文件：[`DEPLOY.md`](DEPLOY.md)（部署）、[`GITHUB_ACTIONS.md`](GITHUB_ACTIONS.md)（排程 Bot）、
[`PERSISTENT_BOT.md`](PERSISTENT_BOT.md)（常駐 Bot）。
AI 開發協作：[`CLAUDE.md`](CLAUDE.md)（守則路由）、[`PITFALLS.md`](PITFALLS.md)（已知的坑）、
[`AGENT_PLAYBOOK.md`](AGENT_PLAYBOOK.md)（派工/驗證模板）。

---

## 設計理念

對照成熟開源量化專案（freqtrade、intelligent-trading-bot、Riskfolio-Lib）的優點：

- **誠實的回測**：無前視、扣成本、樣本外驗證 — 避免過擬合自欺
- **自我優化**：訊號權重隨歷史勝率動態調整，但用一致性抑制雜訊追逐
- **風險優先**：部位依波動配置、組合等風險貢獻、防護機制避免洗版
- **多維分析**：技術面（訊號/回測）+ 基本面（財務健康）+ 總經（FRED）
  + 籌碼面（選擇權定位、SEC 內部人）四面向
- **AI 研究副駕**：論點導向、事實與推論分開、風險優先；可自主呼叫回測/風險/選股/選擇權/
  內部人工具取客觀數據，前瞻問題主動取證再作結論
- **策略驗證閉環**：訊號 → Alpaca 模擬下單 → 交易日誌記錄原因 → 回算訊號實測勝率，
  對照回測（真實向前測試 vs 歷史模擬的落差）
- **單一真實來源**：Bot 與儀表板共用同一套訊號/回測/部位邏輯

> ⚠️ 本平台為分析與教育用途，所有回測為歷史模擬，不構成投資建議。
>
> 註：多時間框架（週線）確認同時用於**即時評分/通知**與**回測**
> （回測頁可勾選「加入週線確認」對照有/無 MTF 的勝率差異，無前視偏誤）。
