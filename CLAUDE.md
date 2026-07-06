# CLAUDE.md — RBS Finance Dashboard 開發守則（路由檔）

> 本檔每個 session 自動載入，只放「鐵律 + 完工檢查表 + 導航」，細節放引用檔：
> **[PITFALLS.md](PITFALLS.md)**（已知的坑：症狀→原因→修法）
> **[AGENT_PLAYBOOK.md](AGENT_PLAYBOOK.md)**（派工/驗證模板、判斷準則、維護協議）
> 動手做任何非小修的功能前，先掃一眼這兩個檔的目錄。

## 專案一句話

Streamlit 金融儀表板（`app.py`，13 頁）+ Telegram 訊號 Bot（`scan_signals.py` 排程版 /
`bot_daemon.py` 常駐版），部署於 Streamlit Cloud + GitHub Actions（cron `*/15`）。
使用者以繁體中文溝通；本專案為分析教育用途，所有輸出標「非投資建議」。

## 鐵律（違反 = 真實出過事故的等級）

1. **公開 repo：金鑰絕不寫進任何被 commit 的檔案。** 只放 GitHub/Streamlit Secrets 或 UI 當場輸入。
   `.gitignore` 已排除 `.env`、`secrets.toml`、`alerts_config.json`。
2. **開發環境對外網路被 proxy 擋住**：yfinance / SEC EDGAR / Finnhub / Telegram / FRED 本地全連不上。
   **「本地連不上」這件事本身不是 bug、不要修；程式碼內的真錯誤（如逾時處理寫錯）照修。**
   純邏輯離線測（`python3 <module>.py`）；
   代碼、公司名、API 行為等「事實」一律派 web-search 子代理查證（模板見 AGENT_PLAYBOOK）。
3. **工作流程儀式**（使用者的硬性期待，會明確檢查）：
   建任務（TaskCreate；無此工具就寫計畫清單）→ 自我批判 → 實作＋**md 同步** →
   **子代理對抗驗證** → commit+push。（md 在 commit 前改完，避免文件與程式碼漂移。）
   驗證不可省：本專案幾乎每個功能的驗證都抓到過真 bug（統計見 AGENT_PLAYBOOK §1）。
4. 開發分支照 session 指示（通常 `claude/...`），`git push -u origin <branch>`；
   commit 訊息帶 session 指定 footer；**不開 PR** 除非使用者明說。
5. 回測紀律：無前視（下一根 K 棒進場）、扣交易成本、walk-forward 樣本外驗證。

## 架構速覽（要改什麼 → 去哪個檔）

| 要改什麼 | 檔案 |
|---|---|
| 網頁頁面/UI | `app.py`（~4300 行、會持續漂移，以 `wc -l` 為準；**不要整檔讀**。導航：Grep `def page_` 找頁面、`PAGES = {` 看路由、`def _cached_` 找快取層、`def _run_.*_tool` 找 AI 助理工具執行器）|
| Bot 訊號/指令/晨報 | `scan_signals.py`（排程進入點；指令 dispatch 搜 `elif cmd ==`）；`bot_daemon.py` 重用其全部邏輯 |
| 回測引擎 | `backtest.py`（triple-barrier / walk-forward / 參數最佳化）|
| 部位與風險數學 | `quant_tools.py`（ATR/Kelly/風險平價）、`rbs_lib.py`（VaR/CVaR）|
| 公司基本面 | `fundamentals.py`（主）+ `finnhub_data.py`（限流備援）|
| 總經 / 產業掃描 / 選股庫 | `macro.py` / `sector_scan.py` / `stock_db.py`（含 AI 供應鏈瓶頸主題）|
| AI 助理 | `assistant.py`（意圖/context 純邏輯）+ `assistant_tools.py`（工具規劃/解析）+ app.py 的 `page_ai_assistant` 與 `_assistant_*` |
| 選擇權情緒 / SEC 內部人 | `options_sentiment.py` / `sec_insider.py` |
| Alpaca 模擬交易 | `alpaca_trader.py`（決策純邏輯 `decide_orders` + REST client）|

**模組慣例**：純邏輯（離線可測）與抓取層（需網路）分離；檔尾附 `if __name__ == "__main__"`
自我測試；使用者可見文字繁體中文。新模組照這個模式寫。

## 完工定義 DoD（加功能時逐項核對——這清單裡每一項都真實漏過、出過包）

- [ ] 模組自我測試通過：`python3 <module>.py`（backtest 用 `--offline`）；新模組記得加進 `.github/workflows/ci.yml` 的自測清單
- [ ] `python3 -c "import ast; ast.parse(open('app.py').read())"`（app.py 不能直接 import——開發環境沒裝 streamlit）
- [ ] **新增環境變數 key？四處同步**：app.py 開機 secrets→env 複製清單（搜 `_os_boot`）、
      `.github/workflows/signal_scan.yml` env、`GITHUB_ACTIONS.md` Secrets 表、`README.md` API Keys 表
- [ ] **新增 .py 模組？** `RBS_Finance_Colab.ipynb` Cell 2 同步清單加檔名（用 NotebookEdit；先 Read 該筆記本確認 cell id，目標是 source 以 `# ── CELL 2` 開頭的那格）
- [ ] **新增 Bot 指令？** `scan_signals.py`：dispatch elif + `/help` 文字 + 檔頭 docstring；`GITHUB_ACTIONS.md` 指令表
- [ ] README 同步：頁面表 / 功能列表 / 檔案結構
- [ ] 子代理對抗驗證（模板：AGENT_PLAYBOOK §3），**High/Med 發現必修**，Low 視成本
- [ ] commit（footer）+ push
- [ ] 回覆裡提醒使用者：**Reboot Streamlit app**（模組快取不清會 AttributeError）；Colab 用戶重跑 Cell 2

## 遇到不確定時

- **技術問題**：自己查、派子代理驗證，不要拿去問使用者。
- **品味/產品方向**（UI 樣式、文案語氣、功能優先序、要不要做某功能）：問使用者——這類判斷外化不了。
- 醒來（scheduled wakeup / 通知）先跑 `git log --oneline -3 && git status --short`：
  工作可能已完成，別重做。
