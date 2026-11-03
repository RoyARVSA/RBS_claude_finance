# data/insider — 內部人 Form 4 公開市場買賣（當時可得版本）

`form4_pit.json.gz` 由使用者在 Colab 執行 `RBS_Finance_Colab.ipynb` 的 **Cell 5（一次性）**（`sec_bulk.colab_build_all`）產生後上傳到這個資料夾，
給研究計畫 §4b「內部人買進」用。SEC 封鎖 GitHub Actions 的 IP（PITFALLS B10），所以在 Colab 跑。

- 來源：SEC DERA **Insider Transactions Data Sets**（Form 3/4/5 的 XML 欄位，每季一個 `YYYYqN_form345.zip`，2006q1 起）。
  2026q1 以前在 `/files/structureddata/data/insider-transactions-data-sets/`，2026q2 起搬到
  `/files/datastandardsinnovation/data/insider-transactions-data-sets/`——兩個路徑都試；兩個都 404 的季度記在 `manifest.missing_quarters`
  （最近一季多半只是還沒發布：`manifest.unpublished_tail`）。
- 保留：`DOCUMENT_TYPE` 為 4 或 4/A（修正申報**另存一筆、不自動取代原申報**）、交易代碼 **P（公開市場買進）與 S（公開市場賣出）**、
  發行人在 S&P 500 CIK 宇宙（2006 年後曾為成分、現行代碼查得到 CIK）。其他代碼（M、A、F、G…）、Form 3／5、非成分公司都丟掉（筆數記在 manifest）。
- 每家公司：`filings`＝{accession: [申報日, 表單, 發行人代碼, [[申報人 CIK, 身分, 職稱], …]]}；
  `trans`＝[[accession, 代碼, 交易日, 股數, 價格, 取得/處分（A/D）, 直接/間接（D/I）, 交易後持股], …]。
  身分旗標：D 董事、O 高管、T 10% 股東、X 其他（可組合，例如 DO）。
- 當時可得：時點＝申報日的**下一個交易日開盤**（Form 4 最晚晚上 10 點才申報）。
- 讀檔：`sec_bulk.load_form4()`；大於 24 MB 自動切成 `form4_pit.part2.json.gz` …——**同一次產生的分檔要全部上傳**，讀檔時自動合併。
- 申報人 CIK、職稱屬 SEC 公開資料（不存姓名）；檔內不含使用者的任何個人資料或金鑰。教育用途，非投資建議。
