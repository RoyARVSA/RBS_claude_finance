# data/earnings — 季報數字與財報公布日（當時可得版本）

這兩個檔由使用者在 Colab 執行 `RBS_Finance_Colab.ipynb` 的 **Cell 5（一次性）**（`sec_bulk.colab_build_all`）產生後上傳到這個資料夾，
給研究計畫 §4a「財報超預期（時間序列 SUE）」用。SEC 封鎖 GitHub Actions 的 IP（PITFALLS B10），所以在 Colab 跑。

| 檔案 | 來源 | 內容 |
|---|---|---|
| `quarterly_pit.json.gz` | SEC `companyfacts.zip`（XBRL，2009 起） | 稀釋 EPS（沒有時退回 BasicAndDiluted／Basic，科目名即旗標）、淨利、加權稀釋股數；期間 3 個月（80–100 天）、9 個月累計（260–290）、全年（350–380）；表單 10-Q／10-Q/A／10-K／10-K/A／10-KT |
| `announce_pit.json.gz` | SEC `submissions.zip`（含分頁檔 `CIK##########-submissions-NNN.json`） | 8-K／8-K/A 含 Item 2.02（財報新聞稿）的 accession、申報日、acceptanceDateTime 原字串、items、表單 |

- **每一個申報版本都留**：同一期數字在 10-Q/A 重編、或次年比較欄重複出現時各是一列（帶 accn 與申報日）；**不用 companyfacts 的 frame 欄**。
- 當時可得：研究端一律在「申報日的下一個交易日開盤」才可用（acceptanceDateTime 的時區標示不可靠，不用它決定時點）。
- 公司範圍：與 `data/edgar/annual_pit.json.gz` 同一個 CIK 宇宙（2009 年後曾為 S&P 500 成分、現行代碼查得到 CIK）。
- 讀檔：`sec_bulk.load_quarterly()`／`sec_bulk.load_announcements()`；檔內 `manifest` 有建置日、筆數、來源網址。
- 檔案大於 24 MB 會自動切成 `xxx.part2.json.gz`、`xxx.part3.json.gz` …（GitHub 網頁上傳上限 25 MB）——**同一次產生的分檔要全部上傳**，讀檔時自動合併。
- 更新：需要時重跑 Cell 5 再上傳覆蓋（每次都是完整重建）。不含任何個人資料或金鑰。教育用途，非投資建議。
