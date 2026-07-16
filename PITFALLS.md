# PITFALLS.md — 已知的坑（症狀 → 原因 → 修法）

> 每一條都是本專案**真實發生過**的 bug 或事故。動相關區域前先掃對應分類。
> 新增格式見檔尾「維護規則」。

## A. 環境

### A1. 開發環境全面斷網
- **症狀**：yfinance / EDGAR / Finnhub / Telegram / FRED 在本地全部 timeout 或代理錯誤。
- **原因**：開發容器 outbound 走 proxy，金融資料源全被擋。**部署後（Streamlit Cloud / GitHub Actions）是通的。**
- **修法**：不要把「本地連不上」當 bug 修。純邏輯用 `python3 <module>.py` 自我測試；
  股票代碼/公司名/API 行為等事實派 web-search 子代理查證（本專案曾以此逐一驗證 29 個代碼，全對）。

### A2. app.py 無法 import
- **症狀**：`import app` 直接 Traceback。
- **原因**：開發環境沒裝 streamlit。
- **修法**：語法檢查用 `python3 -c "import ast; ast.parse(open('app.py').read())"`；
  要測 app.py 裡的純函數，用 `re.search` 取函數原始碼 + `exec` 在乾淨 namespace 跑
  （被取函數若引用 `st.*` 或模組全域，要先在 namespace 塞 stub——那種 NameError 是缺 stub，不是 bug）。

### A3. Streamlit Cloud 模組快取
- **症狀**：给既有模組加了新函數並 push，網頁報 `AttributeError: module 'X' has no attribute 'Y'`。
- **原因**：執行中的 app 保留舊模組物件，redeploy 不一定清。
- **修法**：叫使用者 **Reboot app**（Manage app → Reboot）。防禦性寫法：`if not hasattr(mod, "fn")` 給友善訊息。

### A5. 遠端容器隨時會被回收——工作只存在於 push 之後
- **症狀**：續跑 session 時剛寫的檔案消失、pandas 未安裝、scratchpad 清空（2026-07 一天內發生兩次）。
- **原因**：remote 容器 ephemeral；斷點重建＝fresh clone + 空 pip 環境。
- **修法**：每批功能完成**立即 commit+push**（驗證債可註記在 commit 訊息後補）；續跑第一步
  `git log -3 && git status` 並確認關鍵新檔存在；重裝環境用子集
  `pip install pandas numpy scipy yfinance requests pyflakes`（完整 requirements 裡
  feedparser 的 sgmllib3k 在新容器編不過，離線自測不需要它）。

### A6. 往長檔案插入大段程式的縮排殘行
- **症狀**：插入後 IndentationError 報在**你沒動過的行**。
- **原因**：插入點後面還有原屬前一區塊的尾行（如收尾 caption），被擠進錯誤縮排層。
- **修法**：插入前先讀錨點後 5-10 行確認歸屬；插入後立刻 `ast.parse` 驗證（app.py 5400+ 行時踩過）。

### A4. Streamlit Secrets 不會自動進 os.environ
- **症狀**：使用者在 Secrets 設了 key，模組用 `os.environ.get()` 拿不到。
- **原因**：`st.secrets` 與環境變數是兩回事。
- **修法**：app.py 開機處有 secrets→env 複製迴圈（搜 `_os_boot`）。**新增任何 key 必須加進那個 tuple**，否則設了等於沒設（SEC_USER_AGENT 就漏過一次）。

## B. 資料源

### B1. yfinance `.info` 雲端被限流
- **症狀**：市值/P/E/ROE 全顯示「—」，本地卻正常。
- **原因**：`.info`（quoteSummary 端點）被 Yahoo 對雲端 IP 限流；`.history` 與 `.fast_info` 通常仍可用。
- **修法**：備援鏈三層：`.info` → `.fast_info`（有 market_cap/52週/價格，**沒有** P/E/EPS/ROE）→ Finnhub（`finnhub_data.py`）。已內建於 `fundamentals.py` 與 app.py `_cached_ticker_data`。

### B2. `dict.setdefault` 補不了「key 存在但值是 None」
- **症狀**：部份限流時 `.info` 回 `{"trailingPE": None}`，備援明明啟動卻沒補值。
- **原因**：`setdefault` 只在 key **缺席**時寫入。
- **修法**：一律 `if d.get(k) is None: d[k] = v`。（審查抓到的真 bug。）

### B3. yfinance 批次下載的兩種欄位版面
- **症狀**：批次 `yf.download` 後取 `Close` 偶爾 KeyError，或**所有代碼拿到同一條序列**。
- **原因**：多檔回 MultiIndex，欄位可能是 `("Close", tkr)` 或 `(tkr, "Close")` 兩種版面；
  單檔回平面欄位——若在多檔迴圈裡直接用平面 `Close`，會把同一條塞給每個代碼。
- **修法**：照 `sector_scan._batch_closes` 的寫法：兩種 MultiIndex 版面都試；
  平面欄位分支必須 `if len(tickers) != 1: continue`。

### B4. Finnhub 單位換算
- **症狀**：市值小一百萬倍、ROE 大一百倍。
- **原因**：Finnhub 市值單位是**百萬**，roe/margin 是**百分比數字**（156.08 = 156%）。
- **修法**：`market_cap × 1e6`、`roe/margin ÷ 100`；本專案慣例比率存小數（0.15=15%）、殖利率存百分比。
  另：`{"metric": null}` 要用 `(x.get("metric") or {})`，`.get(k, {})` 擋不住值為 None。

### B5. SEC EDGAR 的 CIK 兩種格式
- **症狀**：submissions 或 Archives 其中一邊 404。
- **原因**：`data.sec.gov/submissions/CIK{cik}.json` 要 **10 碼零填充**；
  `www.sec.gov/Archives/edgar/data/{cik}/...` 要**去零的整數**。
- **修法**：照 `sec_insider.py` 現行寫法。另：EDGAR 必須帶 User-Agent header（可用 `SEC_USER_AGENT` 覆寫）。

### B6. 個股資料的歷史斷層
- **症狀**：某些代碼長期回測/圖形詭異。
- **實例**：WOLF（Wolfspeed）2025-09 破產重整、~1:120 換股，2025-10 前價格不連續；
  SNDK、GEV 為 2024-25 新分拆，歷史短。
- **修法**：對新分拆/重整股，長期回測結果標註不可靠或直接略過。

### B8. yfinance 財報列名是 camel2title 轉換出來的
- **症狀**：`income_stmt.loc["Total Revenue"]` 在某版本 KeyError 或拿到空。
- **原因**：列名由 const.py 的 camelCase key 經 `camel2title(acronyms=["EBIT",…])` 生成
  （"Total Revenue"、"EBIT"、"Operating Income"、"Capital Expenditure"…），升版可能漂移。
- **修法**：一律多候選名 fallback（見 valuation.py `_row()`）；查證法＝直接讀
  site-packages/yfinance 原始碼跑 converter，不要猜（2026-07 對六個列名逐一驗證過）。

### B7. 死檔案裡的活金鑰
- **症狀**：2026-07 全面審計在**早已沒人用的** `colab_setup.py` 裡發現寫死的 ngrok token——
  在公開 repo 的 git 歷史裡放了幾個月，日常開發從未觸碰該檔所以從未被審查。
- **原因**：金鑰掃描只掃「有在動的」檔案；死碼是盲區。
- **修法**：檔案已刪、token 已請使用者撤銷。教訓：(1) 廢棄檔案要**立刻刪**，不要留著；
  (2) 定期全 repo 掃金鑰 pattern（sk-/ghp_/token=/長 base64），包含沒人動的檔案；
  (3) 刪檔救不了 git 歷史——**撤銷（revoke）才是真正的補救**。

## C. Python 生態版本

### C1. NumPy 2.0：`float(array)` 對 ndim>0 陣列拋 TypeError
- **症狀**：`float(w.T @ cov @ w)` 之類在雲端炸掉（本地舊版 numpy 沒事）。
- **修法**：改用 `.item()`：`float((w.T @ cov @ w).item())`。rbs_lib.py 曾多處中獎。

### C2. pandas 3.0 移除的 API
- **症狀**：`Styler.applymap` AttributeError；`resample("M")` ValueError。
- **修法**：`Styler.map`；resample 用 `"ME"`（app.py 有 `_RESAMPLE_ALIAS` 對照）。

### C3. CJK 旁的正規表達式邊界
- **症狀**：「VRT技術面」抽不出 VRT。
- **原因**：`\b` 在中文字與英文字母之間不成立。
- **修法**：用 ASCII lookaround：`(?<![A-Za-z])[A-Z]{2,5}(?![A-Za-z])`（見 `assistant.extract_tickers`）。

## D. Bot / git / 流程

### D1. `git add A B` 是原子操作
- **症狀**：workflow 該 commit 的狀態檔沒進 repo，晨報重複發送。
- **原因**：`git add stateA stateB` 其中一個檔不存在 → 整個 staging 失敗。
- **修法**：workflow 裡每個檔分開 `git add`，各自容錯。

### D2. Bot 指令參數要 clamp
- **症狀**：`/journal 0` 把全部紀錄倒出來洗版。
- **修法**：`n = max(1, int(arg))`；任何使用者輸入的數字都設上下限。

### D3. 併發 push 撞牆
- **症狀**：cron 跑到一半 push 被 reject（non-fast-forward）。
- **修法**：workflow 用 rebase-retry 迴圈；本地 push 失敗時 `git pull --rebase` 再推。
  **絕不 `push --force`**（Bot 會往同一分支 commit 狀態檔，force 會毀掉它的歷史）；
  rebase 重試三次仍失敗就停下向使用者回報。

### D4. 刪 app.py 死碼的近事故
- **症狀**：差點把仍在用的 `page_portfolio_performance` 當死碼刪掉（live 函數與死函數交錯排列）。
- **修法**：刪除前逐一 Grep 確認函數未被引用（查 `PAGES` dict + 所有呼叫點）；
  大規模刪除前先 commit（此檢查點 commit 不需先過子代理驗證，照常帶 footer），
  刪後立即語法檢查 + 抽查存活函數還在。

### D5. LLM API key 隱形字元
- **症狀**：`'ascii' codec can't encode characters in position 7-10`（Bearer 前綴剛好 7 字元）。
- **原因**：使用者貼 key 時混入零寬空格/BOM/全形空白，HTTP header 只吃 ASCII。
- **修法**：`_llm_client` 已接 `_clean_secret` 自動清理；新的 key 輸入點也要過同一函數。

### D6. Anthropic 相容端點的訊息順序
- **症狀**：多輪對話第二次呼叫 400。
- **原因**：對話必須以 user 訊息開頭。
- **修法**：組 messages 前把開頭的 assistant 訊息 pop 掉（見 `page_ai_assistant`）。

### D8. Streamlit session_state 的 key 是全域命名空間
- **症狀**：2026-07 優化時把選擇權結果存成 `st.session_state["opt_result"]`——同頁「參數最佳化」
  早就用同名 key（存 DataFrame vs 存 dict），兩個功能都用過後整頁崩潰。
- **原因**：session_state 全 app 共享，`st.tabs` 每個 tab 每次 rerun 都會執行。
- **修法**：key 一律帶功能前綴（`optsent_`、`ins_`、`pp_`…）；新增 key 前先
  `grep 'session_state\["'` 查衝突。另一教訓：改一段程式時，被刪變數（如 `r_s`）
  可能在下方 30 行外還有人用——刪前 grep 該變數名。

### D7. 內嵌 widget 高度為零
- **症狀**：TradingView iframe「太扁」看不到。
- **原因**：`autosize: true` 在零高度容器裡就是 0。
- **修法**：容器與 div 給明確 px 高度（現有高度滑桿實作可參考）。

### D9. PR merge 競態會把最後一批 commit 留在分支上
- **症狀**：以為功能進了 main，其實 PR merge 抓的是**較早的分支 tip**（day-plan 閉環整批曾這樣滯留）。
- **原因**：使用者按 merge 的時點早於你最後一次 push。
- **修法**：merge 後必查 `git log origin/main..origin/<branch>`；不為空 → 分支
  `checkout -B` 到 origin/main 後 **cherry-pick** 孤兒 commit（force push 會被權限系統擋，別試）。

### D10. 別 rm 已被追蹤的 watchlist_state.json
- **症狀**：`git status` 出現 `D watchlist_state.json`，差點 commit 掉 bot 的活狀態。
- **原因**：該檔早期是自測產物（untracked），後被 workflow commit 成 tracked；慣性 `rm -f` 誤刪。
- **修法**：自測後用 `git checkout -- watchlist_state.json` 還原，永遠不要 rm 它。

### D11. 子代理有 session 用量上限
- **症狀**：驗證代理死於 `hit your session limit · resets HH:MM (UTC)`。
- **修法**：額度將盡時先 commit 保工作（訊息註明驗證待補）→ `ScheduleWakeup` 排 reset 後補驗；
  多批功能可合併一次對抗驗證省額度；已跑一半的代理，其 transcript 裡的部分證據仍可先採用。

### D12. Telegram 舊版 Markdown 不認 `**`
- **症狀**：粗體變空白實體、重點警語不顯示。
- **原因**：`_tg_send` 用 `parse_mode="Markdown"`（legacy），只認單 `*`；`_` 也會炸。
- **修法**：bot 文字一律**單星號**；網頁端要粗體再 `.replace("*","**")`（falsifier 的做法）。
  含底線的字串（罕見 ticker、URL）進 bot 訊息前先想一下。

---

## E. 統計 / 回測

### E1. 重疊視窗的假樣本數
- **症狀**：「六個月持有、逐日進場」的回測有上千筆樣本，顯著性漂亮到不像話。
- **原因**：視窗高度重疊，有效獨立樣本 ≈ n ÷ 視窗長（5 年資料 ≈ 10 個）。直接 t 檢定＝自欺。
- **修法**：用 circular block bootstrap（`falsifier.block_bootstrap_test` 直接複用，
  誤報率經 200 次零模擬驗證 3.5-5.5%），並在輸出報告 effective n。

### E2. 事後策展的主題籃子＝答案寫在輸入裡
- **症狀**：拿 stock_db 主題籃子回測該主題的故事，結果必然漂亮。
- **原因**：名單是**行情走完後**、由知道劇情的人加進資料庫的（回填偏誤）；
  且免費資料源的下市股缺席（存活偏誤——墳墓對照組不存在）。
- **修法**：這類回測必須強制揭露兩個偏誤（falsifier 報告已內建）；
  主題籃子可以拿來「反駁」故事，**不可以**拿來「驗證」故事。

---

## 維護規則

- 踩到**新坑**（花超過 10 分鐘 debug 或審查抓到的真 bug）→ 在對應分類**追加一條**，
  格式照上：症狀/原因/修法，最多 6 行，附檔名。
- 超過 ~35 條時：合併同類、刪掉已被程式碼結構性根除的條目（例如該 API 已不再使用）。
- 本檔可由任何 session 自行更新，**不需**先問使用者。
