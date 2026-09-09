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
| 🏠 市場總覽 | 即時指數/板塊/宏觀指標、**總經數據(FRED)**、市場快訊、**AI 自主市場分析**、**🎭 雙恐懼貪婪指數（美股 CNN + 加密，雙極端警示）**、**🇹🇼 台指期籌碼（三大法人淨未平倉 + 選擇權 P/C 比，TAIFEX）**、**🩺 資料源健康檢查（9 大數據源一鍵驗收）** |
| 📈 持倉分析 | 多資產組合：權益曲線、回撤、Sharpe、IR、Beta、**績效報告（CAGR/Sortino/Calmar/月報酬熱力圖/回撤期間）**、**交易帳本（成本基礎/TWR/XIRR/股息收入）**、**⚖️ 再平衡顧問（HRP/最大 Sharpe/風險平價 目標權重 → 具體加減碼股數清單）**、**🎯 基金/ETF 評估（費用率、追蹤誤差、α/β、上下行捕獲、兩檔持股重疊度）** |
| ⚠️ 風險管理 | VaR/CVaR、蒙地卡羅、Kupiec 回測、壓力測試、**風險平價 + 效率前緣（MPT）+ HRP 階層風險平價** |
| 🔍 股票研究 | K 線+RSI、AI 深度報告、市場篩選器、**訊號回測 + 參數最佳化（含參數熱力圖與 walk-forward 逐段檢視）**、TradingView、**選擇權情緒(Put/Call、IV 偏斜)**、**🔨 假設反駁器（8 類測試推翻投資故事，只證偽不證實）** |
| 🏢 公司分析 | **基本面體質：財務健康評分、估值旗標、三表趨勢、AI 解讀、分析師共識+EPS Beat 率、做空籌碼(FINRA/FTD)、台股三大法人買賣超(上市 TWSE + 上櫃 TPEX)、SEC 內部人交易(Form 4)、💰 DCF+Comps 內在價值估值（投行標準流程：FCF/WACC/期中折現/敏感度表＋同業倍數回推，方法論採 Anthropic 官方 financial-services skills）、📄 一鍵完整研究報告（彙整全部區塊下載）** |
| 🗂️ 產業總覽 | **一次掃描整個市場：產業強弱 vs 風險散佈、鑽取個股（可加基本面）、RRG 板塊輪動象限圖** |
| 🚨 即時警報 | 監控清單、盤中走勢、訊號掃描、Telegram/Email 推播、**🎯 當日交易計畫（VWAP/ORB/RVOL 盤中訂單票：進場區間/停損/停利/股數，財報日自動迴避，可選 Alpaca IEX 即時價，一鍵送模擬 bracket 單）** |
| 🛠️ 交易工具 | 部位大小、**波動率目標部位**、Kelly、風險報酬比、複利 |
| 📉 模擬交易 | **Alpaca 紙上交易：帳戶績效、持倉、權益曲線 vs SPY、交易日誌（原因）、訊號實測勝率** |
| 🧭 佈局計畫 | **整合層：把各分頁／各指令的產出串成一份計畫——每檔四象限（估值/論點 × 價格/動能）、層級（迴避／減碼／累積候選／持有／觀察）、conviction、權重帶、現金目標、主題集中、待建模清單；散點圖一眼看全 watchlist** |
| 🏛️ 公司模型 | **估值層網頁版：建模（DCF／金融 RIM／地產 DDM 路由）、驅動滑桿即時重算、football field（熊–牛、蒙地卡羅區間、現價）、WACC×g 敏感度熱圖、反向 DCF、FCFF 投影、品質面板、公允價值歷史、JSON/CSV 匯出、手工 Excel 匯入橋（產生 `/model set` 指令）** |
| 🪞 鏡像帳 | **引擎接管你實倉起點的虛擬帳（獨立檢視）：起始/目前淨值、持倉分布圓餅、淨值曲線、交易歷史（為什麼交易）、引擎保險絲狀態** |
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
- **Alpaca 模擬交易**：分層自動交易引擎（重製自 freqtrade / QuantConnect Lean / nautilus 的成熟機制）——
  訊號只決定進場；出場由硬停損、+1R 保本追蹤停損、+1.5R 分批鎖利、死錢釋放驅動，
  訊號轉弱只在獲利時了結（虧損中續抱等待、不在低點殺出）；
  停損保險絲（7 天 3 次硬停損→全帳戶冷卻）、回撤 10% / 大盤 risk_off →停新倉、
  贏家每 +1R 加碼最多 2 次（預設關閉，`/autotrade on` 啟用）
- **Alpha 資訊疊加層**（`/alpha`）：進場評分自動疊加 SEC 內部人（cluster buy 加分）、
  選擇權情緒（PCR/IV 偏斜）、空單占流通降評、財報前 3 天禁新倉、
  雙恐貪極度貪婪→新倉風險減半；12 小時快取＋每輪限額輪替抓取（不拖慢 cron）
- **市場氣象台**（`/weather`）：大盤濾網 v2——市場廣度（11 類股 ETF vs MA50）、
  信用利差（HYG/LQD）、VIX 期限結構、殖利率曲線、銅金比五因子合成 0-100 體質分，
  帶遲滯的三態 regime 餵引擎曝險與晨報；成分缺席自動重新配權、不足退回 MA50
- **相關性/集中度控制**：新倉候選與現有持倉的 60 日報酬平均相關 ≥0.75 → 部位縮半、
  ≥0.85 → 跳過進場——直接對付「10 檔高相關 megacap ≈ 貼著大盤」的組合病
- **交易行為體檢**（`/checkup`）：從交易日誌實測行為偏誤——追高（進場價 20 日區間
  百分位）、過度交易（週頻率+短進短出）、太早出場（各出場機制賣後 10 日追蹤=
  機制歸因雛形）、持有期分佈（概念參考 HKUDS/Vibe-Trading，MIT）
- **Shadow 對照**（`/shadow`）：舊版決策邏輯以虛擬帳本平行記帳（同一訊號流、
  疊加前原始評分）——直接量化引擎重製的增量價值，比任何回測都有說服力
- **鏡像帳**（`/mirror`）：以你的實際持倉與資金量為起點的虛擬帳戶，
  由完整引擎堆疊自主模擬操作（模式 A：真實買賣不同步）——回答
  「該不該把實倉交給系統」；實倉資料存加密區、持倉標的自動納入掃描
- **機制歸因報告**（`/attrib`）：FIFO 重建每筆已實現損益，按出場機制
  （硬停損/追蹤/分批/訊號/死錢）與進場機制（首進/加碼）分組：損益/勝率/
  持有天 + 賣後 10 日追蹤——用實測數據回答「引擎哪一層在賺錢、哪個機制太急」，
  自我學習閉環的依據；broker 對帳淘汰 journal 外平倉的 lot
- **當日交易計畫**：`/today [帳戶 風險%]` 盤中訂單票（VWAP/ORB/RVOL 進場、停損/停利/股數、財報日迴避）；進場票自動記入決策計分板（隔日結算，與量化/委員會同板比較）
- **當日計畫歷史回測**：`/plantest` 用過去 ~60 交易日 5 分 K 逐日重放訂單票（無前視、扣成本、停損優先），統計各型態實證勝率/R 期望值；`/plantest apply` 把 walk-forward 校準（負期望型態停用、不穩定降信心）套進 /today——**讓判定吃歷史實證自我修正**；**每週自動重跑校準**（動作有變時推播通知，`/set plan_autocal_enabled off` 關閉）
- **參數尋優**：`/plantest opt` 掃 ORB 分鐘 × 停損 ATR 倍數 × 目標 R:R 共 27 組參數，訓練段排序、**驗證段沒明確勝過現行預設就不推薦**（防過擬合）；`opt apply` 一鍵套用推薦參數＋對應校準
- **分析師預估快照帳本**：`/est [TICKER]`——免費資料拿不到「逐日共識歷史」，所以 Bot 每輪用閒置名額輪替把 yfinance 的 EPS 共識 / 7–90 天修正 / 上下修家數 / 目標價與 Finnhub 評等家數存成週頻列（append-only，`estimates_ledger.json`），從第一天起累積自有 point-in-time 歷史；輸出修正動能分（冷啟動用 yfinance 90 天回看）與 Alpha Vantage 回填的財報驚奇史（beat 率、SUE）。**只顯示與論點監測、不進部位**——估值層（見 VALUATION_PLAN.md）的第一塊地基
- **選股池（Universe）層**：`/universe`——不再只看固定 watchlist：每月用 `yf.screen` 三次呼叫建 500–750 檔寬宇宙（美股、市值 ≥ 20 億、3 月均量 ≥ 100 萬、價 ≥ $5），品質門檻在篩選器端做（ROE/流動比/負債比/淨利率；金融與地產以 sector 白名單排除、另路由）+ 12-1 動能排序取候選前 N；快照帶 `as_of`/`available_at`（T+1 可見）每次重建一檔落在 `data/universe/YYYY-MM-DD.json`，加上 S&P 500 成分歷史期間表（1996 起），讓回測能用「當天成分」而非今日名單（消除倖存者偏差）。**P0 只快照與顯示，不接引擎**——宇宙只決定引擎看得到誰，進場仍看技術評分、出場仍走價格機制
- **公司模型引擎（估值層 P1）**：`/model TICKER`——把使用者手工三表模型的驅動層泛化到任意代碼：point-in-time 三表（每期帶可得知日，重編不覆寫）、共識/指引覆蓋前兩年成長、專業級 WACC（Blume 收縮 β、利息保障→合成信評利差、市值權重）、5+5 年 FCFF 含 SBC 成本化、**價值中性終值**（再投資率 = g/ROIC，終值 ROIC 預設收斂至 WACC）、熊/基/牛三情境（熊＝論點錯了的世界）+ 蒙地卡羅 + 敏感度、**反向 DCF**（市價隱含 5 年營收 CAGR 與營益率）、九條模型審核（任一不過＝待審、不推買訊）、品質與會計風險（Piotroski F、Altman Z/Z''、Beneish M、Sloan 應計、FCF 轉換、SBC、稀釋、ROIC−WACC 價差）→ MoS 分層門檻（品質好 25%／一般 35%／差 50%）與區間位置判定。`set` 覆蓋任何驅動、估值歷史加密保存。折現引擎以使用者的 VRT Excel 為回歸基準（並發現其終值折現期錯誤）。**P1 只顯示與監測，不接引擎**
- **估值層接資金佈建（P3，預設關閉）**：`/set val_enabled on` 後引擎讀 val_hist 的 MoS 給新倉部位乘數 0.5–1.25×（只乘風險預算、單檔上限不放大）、市價高於牛市情境不加碼、MoS>30% 提早加碼、候選排序傾斜 ±0.1——不觸發進場、不否決出場；`/engtest opt` 自動做「估值層 開/關」A/B（PIT 由估值歷史列日期保證），holdout 沒贏就維持關閉；`/rebalance bl` 用 Black-Litterman 把公允價值轉成期望報酬觀點（Idzorek 信心＝情境寬度）配置權重
- **指引/KPI 萃取（P4）**：`/guidance TICKER`——Alpha Vantage 逐字稿 → 便宜 LLM 只做定位與逐字轉錄 → 程式做所有驗證（原文回對、數字 regex 回對、修訂方向由程式判定、與財報對帳、注入無效、壞 JSON 棄權）；這是「指引上調 → 公允價值先動 → 部位跟著動」的輸入端
- **候選篩選與治理（P5/P6）**：`/screen` 從選股池與 AI 主題層挑 watchlist 之外的候選（Stage 3 每閉市日一批補品質與修正動能、品質否決剔除、綜合分排名；只建議不自動加入）；`/valreport` 每月治理月報——估值層準不準：各判定的事後命中率、公允價值穩定度、MoS 因子 IC（重疊修正）、覆蓋與過期、指引覆蓋
- **佈局計畫整合層**：`/playbook` 與網頁「🧭 佈局計畫」——同一套邏輯（`playbook.py`）把選股池、分析師修正動能、公司模型、品質旗標、技術評分、大盤 regime、持倉與論點收成一份分層計畫：迴避（品質否決／高於牛市情境／跌破失效價）、減碼（持有且區間位置 > 0.7）、累積候選（有安全邊際且品質不差；標示技術訊號是否已達門檻）、持有、觀察；每檔 conviction 只由可得成分構成並附成分數，權重帶有界且依 regime 打折。閒置輪每 7 天自動輪替更新模型、週報附摘要。**全部為參考：進場仍由技術訊號、出場仍由價格機制、估值層未過 holdout 不接引擎**
- **產業路由與因子把關（估值層 P1.5/P2）**：金融股走剩餘收益模型（RIM：ROE 十年淡出至均值與股權成本的中點、配息＋回購率、持續係數 0.6 終值）、地產／高股息走股利 H-model（三年股利 CAGR → 長期 3%，5 年半衰）；`factor_eval.py` 提供 alphalens 式 Rank IC／ICIR／分位報酬／因子自相關與「21 日 IC > 0.03 且 ICIR > 0.3」配置門檻——MoS、修正動能、品質分累積夠歷史後先過這關，才談進部位
- **引擎歷史重放與參數學習**：`/engtest [3m|6m|1y|2y]` 把**整台波段引擎**（進場門檻、停損/追蹤/分批/死錢、保險絲、regime 三態）逐日重放過去 N 個月——每日評分只用當日以前 K 棒、t 日決策 t+1 開盤成交、單邊 0.05% 成本、對照 SPY 買進持有，回答「如果用現行參數過去會賺多少」；`/engtest opt [apply]` 掃 進場門檻×停損倍數×追蹤回落×分批R×死錢天數 108 組，三段 walk-forward（訓練排序/驗證挑選/holdout 只看一次把關）+ DSR 扣多重測試幸運上限——這是「從歷史學規則」的誠實版（參數搜索，非深度 RL：日 K 樣本太少會學到雜訊）；`clear` 還原
- **假設反駁器**：`/falsify` 對投資故事跑 8 類反駁測試——block bootstrap 漂移顯著性（誠實處理重疊視窗）、日期穩健性、晚進場、成本存活、事件日 CAR、regime/利率週期切分、動能混淆兩因子回歸、跨市場泛化——外加 **DSR 多重假設帳本**（試了幾個才挑到這個→折減）。**只能證偽、不能證實**，報告頁首永遠印這句話
- **投資論點追蹤**：`/thesis` 記錄每檔的論點/支柱/風險/催化劑與**失效價**，掃描自動監測失效與達標即推播；逾 90 天未複查晨報提醒（「不可否證的不是論點」）
- **財報前瞻/覆盤**：`/preview TICKER` 財報前 3 週出前瞻（共識、beat 率、選擇權隱含波動、三情境框架），公布後 2 週出覆盤（beat/miss、隔日反應、評等動向），模式自動判定
- **Telegram 指令**：清單 `/add /remove /list`、分析 `/rank /fundamentals /options /insider /whales /earnings /briefing /weekly /today`、
  **AI `/committee`（手機開機構決策會議）**、警報 `/alert`、風控 `/risk /protections /calibrate`、
  模擬交易 `/autotrade /positions /pnl /journal /closeall /mirror /engtest`（`/help` 看全部）

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
| `FINNHUB_API_KEY` | 基本面 + 內部人備援 | yfinance 限流時市值/P/E/ROE 顯示「—」；SEC 封鎖雲端 IP 時內部人資料缺席 | [finnhub.io](https://finnhub.io/) |
| `ALPACA_KEY_ID` + `ALPACA_SECRET_KEY` | 模擬交易（**paper**）| 模擬交易不執行 | [alpaca.markets](https://alpaca.markets/) Trading API |
| `ALPHA_VANTAGE_KEY` | 財報驚奇（SUE）歷史回填（每季公告日共識 vs 實際，回溯至 1996） | `/est` 少一段 | [alphavantage.co](https://www.alphavantage.co/support/#api-key) 免費 key（25 次/日，程式計數） |
| `STATE_ENC_KEY` | 敏感區塊加密（論點/淨值/簿記/參數以密文 commit）| 明文照舊 | 自訂長隨機字串；**GitHub 與 Streamlit Secrets 都要設；遺失或換 key＝舊密文無法解**（輪替前先用舊 key 取回） |
| `GITHUB_TOKEN` | 決策計分板持久化（委員會紀錄 commit 進 repo） | 紀錄只存本地，app 重啟即消失 | GitHub → Fine-grained PAT，**只授權本 repo 的 Contents 讀寫**（勿用全域 classic token） |

> SEC 內部人交易（Form 4）走 EDGAR、選擇權情緒走 yfinance，兩者**皆免 key**。
> SEC **建議**設 `SEC_USER_AGENT`（格式「`名字 email`」）：SEC 對雲端 IP + 匿名 UA
> 常回 403，GitHub Actions 上不設會導致內部人資料靜默缺席（Alpha 疊加層少一源）。

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
valuation.py            DCF+Comps 估值：FCF/WACC/期中折現/敏感度 + 同業倍數（方法論：Anthropic financial-services）
sentiment_fg.py         雙恐懼貪婪指數：美股 CNN（含鏡像備援）+ 加密 alternative.me（免費免金鑰）
thesis.py               投資論點追蹤器：論點/支柱/失效價自動監測（方法論：Anthropic thesis-tracker）
earnings_review.py      財報前瞻/覆盤：共識、beat 率、隱含波動、三情境（方法論：Anthropic earnings skills）
falsifier.py            假設反駁器：block bootstrap/regime/動能混淆/跨市場/DSR 帳本（只證偽不證實）
fund_eval.py            基金/ETF 評估：費用率/追蹤誤差/α β/捕獲率/持股重疊（yfinance funds_data）
taifex.py               台指期籌碼：三大法人淨未平倉 + 選擇權 P/C 比（TAIFEX 免費公開資料）
tw_flows.py             台股三大法人買賣超（TWSE T86 上市 + TPEX 上櫃，免 key）：外資/投信/自營 + 連買天數
committee.py            機構決策委員會：角色提示/立場解析/硬風控閘門/量化交叉比較，支援多檔與深度模式（TradingAgents 式）
trade_engine.py         分層自動交易引擎：追蹤停損/分批/保險絲/三態曝險（純邏輯）
alpha_overlay.py        Alpha 資訊疊加層：內部人/選擇權/空單/財報 veto/恐貪縮倉
market_weather.py       市場氣象台：廣度/信用/VIX期限/曲線/銅金五因子體質分
behavior_check.py       交易行為體檢：追高/頻率/出場品質/持有期
shadow_book.py          Shadow 對照帳本：舊決策邏輯平行記帳 vs 新引擎
mirror_book.py          鏡像帳：引擎接管使用者實倉起點的虛擬帳戶(模式 A)
engine_backtest.py      引擎歷史重放 + 參數學習（walk-forward 三段 + DSR，/engtest）
fin_data.py             point-in-time 三表取數（yfinance 主、Finnhub as-reported 備援、first-seen 合併、data/fin/）
quality.py              品質/會計風險：Piotroski、Altman、Beneish、Sloan、ROIC 價差、旗標與否決
screener.py             候選篩選：選股池 ∪ 主題 − watchlist，Stage 3 限額，綜合分（/screen）
val_report.py           估值治理月報：覆蓋/事後命中/穩定度/因子 IC/指引覆蓋（/valreport）
guidance.py             指引/KPI 萃取：LLM 定位轉錄 + 程式驗證（原文/數字回對、修訂、對帳；/guidance）
playbook.py             佈局計畫整合層：分層/四象限/conviction/權重帶/組合層（/playbook、網頁 🧭）
factor_eval.py          因子評估：Rank IC/ICIR/分位報酬/自相關 + 配置門檻（alphalens 式，純 pandas）
company_model.py        公司模型引擎：驅動推導、專業 WACC、價值中性 DCF、三情境/MC/敏感度、反向 DCF、審核、訊號（/model）
universe.py             選股池層：yf.screen 寬宇宙 → 品質/流動性/動能篩 → PIT 快照 + 成分歷史（/universe）
estimates_ledger.py     分析師預估快照帳本：週頻 append-only、修正動能、Alpha Vantage SUE 回填（/est）
attribution.py          機制歸因報告：各機制實測損益/勝率/賣後追蹤(FIFO)
state_crypto.py         敏感區塊加密：論點/淨值/簿記/參數以密文 commit(STATE_ENC_KEY)
net_guard.py            慢源熔斷 decorator：慢且空→本輪跳過該源(掛 options/short)
indicators.py           技術指標+綜合評分(自 scan_signals 抽取,含自測)
alpaca_trader.py        Alpaca 紙上交易 REST client + bracket 單（decide_orders=legacy）
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
