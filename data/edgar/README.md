# data/edgar — SEC EDGAR 年度財報（當時可得版本）

`annual_pit.json.gz` 由使用者在 Colab 執行 `RBS_Finance_Colab.ipynb` 的 **Cell 4（一次性）** 產生後上傳到這個資料夾。
內容是公開的 SEC XBRL 年報數字（2009 年後曾為 S&P 500 成分、現在代碼查得到 CIK 的公司 × DCF 所需科目），
每個版本都帶申報日，供 `edgar_pit.pit_periods()` 還原「當天看得到的財報」，給 VRT 方法（company_model）做長歷史回測。

- 為什麼不在 Actions 抓：SEC 封鎖 GitHub Actions 的 IP（PITFALLS B10）。
- 更新：需要時重跑 Cell 4 再上傳覆蓋即可（每次都是完整重建）。
- 不含任何個人資料或金鑰。教育用途，非投資建議。
