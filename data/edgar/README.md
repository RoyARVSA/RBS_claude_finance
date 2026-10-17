# data/edgar — SEC EDGAR 年度財報（當時可得版本）

`annual_pit.json.gz` 由使用者在 Colab 執行 `RBS_Finance_Colab.ipynb` 的 **Cell 4（一次性）** 產生後上傳到這個資料夾。
內容是公開的 SEC XBRL 年報數字（2009 年後曾為 S&P 500 成分、現在代碼查得到 CIK 的公司 × DCF 所需科目），
每個版本都帶申報日，供 `edgar_pit.pit_periods()` 還原「當天看得到的財報」，給 VRT 方法（company_model）做長歷史回測。

- 為什麼不在 Actions 抓：SEC 封鎖 GitHub Actions 的 IP（PITFALLS B10）。
- 更新：需要時重跑 Cell 4 再上傳覆蓋即可（每次都是完整重建）。
- 不含任何個人資料或金鑰。教育用途，非投資建議。

## spinoffs.json — 分拆除權日（`edgar_pit.SPINOFFS` 的出處）

S&P 500 母公司把子公司股票配給股東（分拆、reverse Morris trust）時，除權後股價下跳，但前一份年報仍含被分拆事業，
機械化估值會把它看得「很便宜」。`value_research` 把除權日當**斷點**：期末早於斷點的年報，在斷點（含）之後失效。

- 來源：2026-10 網路研究（三個時段各一個研究代理＋補漏代理）＋逐筆對抗核對（SEC 8-K、公司新聞稿、OCC/CME/交易所公告、媒體）
- `used: true` 才進 `SPINOFFS`。排除：查核否決（例如當時不是 S&P 500 成分）、交換要約（split-off：母公司每股價值不會機械性下跳）、
  分拆價值 < 母公司市值 5%
- 鍵用分拆當時的代碼與現行代碼（`break_dates` 會正規化）；看錯日期請直接改 `edgar_pit.SPINOFFS` 並在這裡補來源
