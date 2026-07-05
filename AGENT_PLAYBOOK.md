# AGENT_PLAYBOOK.md — 派工、驗證、判斷準則與維護協議

> 讀者是未來在本專案工作的 AI session（任何等級的模型）。
> 規則刻意寫得具體、可照抄。模板直接複製後填空即可。

## §1 本 harness 的三大漏洞（制度就是為了堵這三個）

1. **「完工」散落在 4+ 個檔案**：新 key 要同步 app.py `_os_boot` 清單 / workflow env /
   GITHUB_ACTIONS.md / README；新模組要進 Colab Cell 2。靠記憶必漏（漏過 SEC_USER_AGENT、
   險漏 trade_journal 提交）。→ **對策**：一律走 CLAUDE.md 的 DoD 清單逐項打勾。
2. **app.py ~4300 行（會漂移，以 `wc -l` 為準）**：整檔閱讀極耗 token 且易看漏。→ **對策**：
   只用 Grep 定位（app.py 內：`def page_` / `PAGES = {` / `def _cached_` / `def _run_.*_tool`；
   Bot 指令在 scan_signals.py：`elif cmd ==`），一次只讀目標函數 ±30 行；
   新的可離線測試邏輯寫成獨立模組，不要再往 app.py 塞純邏輯。
3. **驗證品質靠自覺**：自己寫的碼自己看不出盲點。實績：本專案的子代理對抗驗證在
   risk-parity 不收斂、`setdefault`-None、無標的 outlook 空 context、workflow 原子 add、
   單位換算等處抓到**至少 8 個真 bug**，幾乎每個功能一個。→ **對策**：§3 模板不可省略。

## §2 派工守則

**什麼派子代理**（fresh context，回報只要結論）：
- 對抗式驗證 / code review（每個非小修功能，必派）
- 廣域搜尋、跨多檔掃描（「X 在哪裡被用到」層級以上）
- **一切外部事實查證**：股票代碼、API 行為、套件版本行為——開發環境斷網，主對話無法驗證，
  一律派帶 WebSearch 的子代理（模板 T2）
**什麼不派**：實作與修改（主對話直接做，效率高且脈絡完整）、單點查詢（自己 Grep 更快）。

**回報合約**：子代理只回「結論 + file:line + 嚴重度」，不回檔案傾印；長產物落檔傳路徑。

**驗證不自驗**：驗收預設派 fresh-context 子代理（不被主對話的假設污染）。
**唯一例外**：同一子任務重派兩輪仍回報空泛時，主對話自驗，並在給使用者的回報中註明是自驗。

**模型選擇**：預設不指定（繼承 session 模型）。機械式批次工作可用 `model: "haiku"` 省成本。
（顯式調 model/effort 的增益未經本專案實測，屬建議而非鐵律。）

**升降級路徑**：
- 子代理回報空泛（只有「看起來沒問題」、沒有 file:line）→ 換更尖銳的問題重派一次，
  同一子任務**最多重派兩輪**，仍不行才適用上面的自驗例外。
- 子代理報 High/Med → 必修；報 Low → 修一行能解的就修，要大動的記入 PITFALLS 後跳過。

**醒來協議**：scheduled wakeup / task-notification 觸發時，第一件事
`git log --oneline -3 && git status --short`。已完成 → 簡短回報即止，不重做。

## §3 模板（複製填空）

### T1 對抗式驗證（每個功能 commit 前必用；驗已 commit 的舊功能時把 UNCOMMITTED 換成 commit hash）
```
Review the new <功能名> in /home/user/RBS_claude_finance (branch <branch>, <UNCOMMITTED 或 commit hash>).
目的：<一句話：這功能該做到什麼>。

Files to review:
1. <file> — <哪些函數，各自該有什麼行為>
2. ...
先跑 <自我測試指令，如 python3 module.py> 確認基準。

Check for REAL correctness bugs (not style), focus on:
- <具體疑點1：例如單位換算（百分比vs小數）、None/空輸入、除零>
- <具體疑點2：例如 dict key 對不對得上下游、作用域/NameError>
- <邊界：空清單、非美股代碼、網路失敗降級路徑>

Report findings as file:line + one-line description + severity (High/Med/Low).
If no real bugs, say so explicitly. Do NOT modify files.
```
**要點**：疑點必須具體（「檢查殖利率是百分比還是小數」），不要寫「檢查品質」。

### T2 外部事實查證（斷網環境的必需品）
```
Verify the following facts via web search; the dev environment cannot reach financial APIs.
For each item give verdict OK / FIX→<correct value> / DROP, with a one-line source note:
1. <ticker 或事實聲明>
2. ...
Prioritize accuracy over speed. Do NOT modify files.
```

### T3 廣域搜尋/理解
```
In /home/user/RBS_claude_finance, find <目標>。
我需要的結論格式：<例如「每個呼叫點的 file:line + 該處傳入的參數型別」>。
不要貼大段程式碼，只要結論清單。
```

## §4 判斷準則（rubric，各附正反例）

**何時算「真的完成」** = CLAUDE.md DoD 全勾。
- ✅ 正例：Finnhub 備援——模組自測過、app.py 接上、4 處 key 同步、Colab 清單、README、
  子代理驗證抓到 setdefault bug 並修掉、commit、提醒 Reboot。
- ❌ 反例：「程式碼寫完且語法過了」就宣告完成——Colab 清單沒加，使用者隔天 Cell 2 同步不到新模組。

**何時停下來問使用者**：只在「品味/方向/取捨」。
- ✅ 正例：「分析師模式要做到 A、A+B 還是全做？」（範圍取捨，使用者的錢和時間）
- ❌ 反例：「P/E 該不該排除負值？」——這是技術正確性問題，自己判斷（排除）並在 commit 訊息說明即可。

**方向錯了的訊號（該換路，不是重試）**：
同一修法失敗兩次；或修法開始跟資料源/框架「對抗」（例如試圖讓本地連上 yfinance）。
- ✅ 正例：risk parity 乘法更新調參兩輪仍只有 16% 收斂率 → 換演算法（座標下降），一次到位。
- ❌ 反例：第三次微調同一個乘法更新的步長。

**品質底線怎麼驗**：純邏輯 = 自我測試斷言（含空輸入、極端值）；整合 = 子代理對抗審查；
外部事實 = T2 網查。三者對應三種錯誤來源，不可互相替代。

## §5 維護協議

| 檔案 | 誰可以改 | 規則 |
|---|---|---|
| PITFALLS.md | 任何 session 自行改 | 新坑追加（格式照檔內規則）；>35 條時精簡 |
| AGENT_PLAYBOOK.md | 任何 session 自行改 | 模板可迭代；§4 判準改動要附新的正反例 |
| CLAUDE.md | **鐵律區塊需使用者同意**；其餘區塊（DoD/導航/專案摘要等）可自行更新 | 保持 ≤100 行，細節推到引用檔 |

每次踩坑：教訓寫 PITFALLS（事實類）或本檔 §4（判斷類）。commit 訊息照常規。

## §6 給未來 session 的信

**三件最重要、但使用者不會主動說的事：**
1. **儀式是硬需求**：建任務→自我批判→實作＋md 同步→子代理驗證→commit+push。使用者會檢查
   你有沒有跳過驗證（曾明確要求「記得規畫自我監督執行」）。驗證的投報率有實據（§1.3）。
2. **斷網不是壞掉**：本地連不上金融 API 是環境設計。純邏輯離線測、事實網查、
   整合請使用者部署後實測並回報（他願意配合，貼截圖/錯誤訊息給你）。
3. **使用者是投入的合作者、非工程師**：繁體中文、口語、常給高層目標（「更主動更專業更有遠見」）。
   把模糊目標翻成 2-4 個具體選項讓他選，他決策很快。金鑰安全他很在意——公開 repo，別讓他失望。

**這套制度最可能的退化方式與預防：**
- *文件與程式碼漂移*（最可能）：改了行為沒改文件。→ DoD 最後一項就是 md 同步；
  引用文件內容前先 Grep 驗證它還是真的。
- *PITFALLS 膨脹成沒人讀的垃圾場*：→ 35 條上限 + 定期合併，寧缺勿濫。
- *驗證儀式形式化*（派了子代理但問題空泛，等於沒驗）：→ T1 要求「具體疑點」，
  空泛回報要重派（§2 升降級）。

**誠實的極限**（harness 補不了的）：
- 品味與模糊題（UI 美感、文案語氣、「這樣夠不夠專業」）：制度外化不了，直接問使用者。
- 市場判斷的對錯：本平台輸出的是「有據可查的分析」，不是預測能力；別在文件或回覆裡暗示能預測。
- 本檔的判準來自單一長 session 的經驗，樣本有限；與現實衝突時，相信現實並更新本檔。
