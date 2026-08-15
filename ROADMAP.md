# ROADMAP — 七人審查團總彙整（2026-08-15）

> 七個獨立視角（紅隊/金融演算法/自動化維運/財金/網頁執行/資安/首席架構師）
> 完整檢視後的三層彙整。發現共 29 項編號 + 各批「已驗證穩健」清單。
> 方法論：紅隊與演算法全部結論來自**模擬實跑**；財金與演算法的外部事實經
> WebSearch 多源查證；兩組獨立視角匯合的發現（如 concurrency 鎖）視為坐實。

## ✅ 第一層：立即修（本輪已全部修復，2026-08-15）

| 修復 | 來源 | 摘要 |
|---|---|---|
| workflow concurrency 鎖 | 紅隊+維運（獨立匯合） | cron 自我重疊 → 指令雙派工/雙下單/state 互踩 |
| 衝突安全推送 | 維運 | 舊 rebase-retry 撞 state 衝突必敗＋可能寫入衝突標記 |
| load_state 壞檔防護 | 紅隊（實跑復現） | 壞檔曾使 last_update_id 歸零 → 24h 指令全量重播（含 /closeall）；現改備份壞檔＋正則搶救 offset＋欄位級驗證（watchlist:null 不再每輪硬崩） |
| pip 釘版（signal_scan.yml） | 資安 | 載全 secrets 的環境跑未釘版套件＝供應鏈投毒開門 |
| Alpaca paper 鎖 hostname 精確比對 | 資安 | 子字串檢查可被 userinfo 繞過至真錢端點 |
| ci.yml permissions: read | 資安 | PR 以完整 token 跑不受信碼 |
| TELEGRAM_TOKEN/CHAT_ID 消毒 | 資安 | D5 家族：尾隨換行炸 URL |
| 氣象台殖利率利差 ×10 修正 | 財金（多源查證） | ^TNX/^IRX 是百分比報價非 ×10 標度；15% 權重成分先前永遠卡 21-28 分 |
| 氣象台缺席成分改補 50 中性值 | 演算法（模擬實證） | 重配權會讓悲觀成分缺席時同資訊翻多（52.5→66.4），跨期失去可比性 |
| corr_guard 改 max/avg 取嚴 | 演算法（2 萬次模擬） | 攣生股（ρ=0.98）被 4 檔平均稀釋成 0.28 放行——正好放過該擋的 |
| /set 數值夾制 SET_CLAMPS | 紅隊 | /set risk_pct 50 曾可設出 5000% 風險繞過全部下游夾制 |
| DEFAULT_THRESHOLDS 補齊散落 key | 架構師 | corr_hi/corr_mid/weekly/plan_autocal 預設值不再散落讀取點 |
| daemon 三修 | 紅隊+維運 | 每 60s 重讀磁碟 state（防覆蓋 cron 寫入）；毒訊息落盤；補週報；「不可與 cron 並存」警語 |
| 晨報補「非投資建議」+ top_call 用 regime label | 財金 | 最高頻輸出漏揭露；risk_off 文案寫死 MA50 與氣象台來源不符 |
| app.py 四修 | 網頁 | _cached_macro（三處共用，止住每 rerun 重打 FRED）；風險頁部分失敗防紅屏；_GH_BRANCH env 化預設 main；st.columns(0) 保護 |
| MC 漂移標度 | 財金 | drift×days、擴散×√days 分離；算術報酬不再套 exp |
| 死 artifact 步驟移除 | 維運 | 永遠 no-files-found 的殭屍步驟 |

## 📋 第二層：短期 backlog（1-4 週內，依價值排序）

1. **plan_backtest.optimize 加第三段 hold-out**（演算法：驗證段被選擇污染；val n≥10 才准推薦）
2. **健康度回報**：state 加 health 區塊、晨報尾附「🩺 昨日 N 輪·最慢 Xs·最大間隔 Ym」（維運，~40 行）
3. **熔斷器 decorator 化**，掛 options_sentiment、short_data（維運：各有最壞 ~4 分鐘拖速面）
4. **indicators.py 抽取 + scan_signals --offline 自測**（架構師：評分心臟 3120 行零斷言、app.py 偷用私有函數——同一個 PR 解決）
5. **attribution 加 β/資訊比率歸因**（財金：回答「α 還是 β」——Shadow 月底覆盤前做好最有價值）
6. **信用利差改 FRED OAS**（BAMLH0A0HYM2）：HYG/LQD 有 duration 混淆，2022 式升息會誤報（財金）
7. WACC 權重改總債務；終端年 CapEx 收斂至 D&A；comps 改 football field 區間（財金三項慣例修正）
8. 低頻資料 persist="disk"（網頁：bot push 每 15 分重啟 Streamlit，記憶體快取形同虛設）
9. requirements.txt 釘版（需先在 Streamlit Cloud 確認現行版本，貿然釘會炸網頁端）
10. journal 達 300 筆時啟動淘汰歸檔（維運；attribution 完整性）
11. reflection 加二項式 CI、同 ticker 去重疊（演算法）
12. 期望移動標註「1σ≈68%」（財金）；attribution 報告加註 exit_profit_only 的選擇偏差警語（演算法）

## 🧭 第三層：方向性 roadmap（實驗驅動，讓數據決定）

- **追蹤停損改 3×ATR Chandelier 制**：文獻回測 PF 1.61 vs 固定 8% 的 1.28——用 shadow_book 開平行帳跑 60 日再決定（演算法）
- **alpha 各源消融實驗**：scoreboard 加 overlay on/off 雙軌，90 日比命中率；PCR 符號（options_w 方向）一併檢驗（演算法）
- **氣象台門檻/遲滯網格化**：歷史重放分數 vs SPY 未來 20 日報酬分桶（餵 falsifier T6 思路）（演算法）
- **盈餘修正動能因子**：yfinance eps_trend 免費可做，文獻支持度最高的前瞻因子（財金）
- **macro 加實質利率 DFII10/美元 DTWEXBGS/NFCI**（財金）
- **app.py 重構三刀**：app_theme → app_cache（24 個快取集中，搬家全冷要挑部署時機）→ app_assistant（1600 行，依賴第二刀）（網頁+架構師）
- CI mock smoke test：觀望，等第二次「CI 綠但 cron 炸」再做（架構師）
- 指令表指定 GITHUB_ACTIONS.md 為唯一正本（架構師；已載入 DoD 待執行）

## 🗳 待使用者拍板（品味類，審查團不代決）

1. **alpha「微調不翻案」語意**：delta ±0.25 實際可把基礎分 0.25-0.49 推過 0.5 門檻。
   選項 A：夾更緊（±0.15，真正不翻案）／B：維持現狀但修文件宣稱／C：改 veto-only（資訊只擋不加）
2. **state 檔公開範圍**：paper 淨值+策略參數+/thesis /falsify 個人論點文字都被 commit 到公開 repo。
   選項 A：接受現狀（純模擬）／B：敏感區塊拆私有檔（gitignore + Actions artifact）／C：只把 theses/falsify 論點拆出
3. **指令面收斂**：dispatch 已 39 個指令。要不要做一輪保留/合併/淘汰（產品體驗官角色本輪未跑，可補）

## 已驗證穩健（審查團明確蓋章，別浪費時間重查)

零循環依賴；130 commits 歷史零活金鑰；bot chat_id 驗證有效；無 shell/eval 注入面；
/set 白名單嚴格；毒訊息防護（cron 路徑）；state 各鍵皆有清理機制（31.7KB）；
falsifier DSR 偏保守、bootstrap type-I 1.7%；plan_backtest 無前視；
Kelly/ERC/ES/期中折現數學正確；app.py 失敗路徑模式良好；24 個快取全有 TTL。
