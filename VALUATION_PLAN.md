# 估值層規劃 — 從 VRT 三表模型到「前瞻性資金佈建」

> 2026-09-07。起點：使用者手工建置的 Vertiv（VRT）三表 + DCF 模型（`VRT_model.xlsx`，12 張工作表、2,494 個公式、資產負債表 Balance Check 全年 0）。
> 問題意識：目前引擎（技術評分 + SPY/氣象台濾網 + 價格驅動出場）本質是**回溯性的趨勢跟隨**，看不見「公司下一季會變成什麼樣子」。本文回答：這份模型的架構能給專案什麼、怎麼把「每家公司的財務前瞻分析」變成**資金佈建的參考輸入**，以及要按什麼順序、用什麼驗證標準做。
> 一如既往：教育用途、非投資建議；所有新輸出都掛此聲明。

---

## 0. 一頁結論

1. **VRT 模型是一份標準賣方等級的驅動式三表模型**：as-reported 原始表（含完整性檢核）→ 營運驅動層（分區成長、產品/服務毛利、SG&A 槓桿、營運資金天數、資本密集度、在手訂單合理性）→ 三表整合（含債務/營運資金/固定資產/權益四張附表）→ CAPM WACC → 10 年兩段式 FCFF DCF + 兩張敏感度表。可直接泛化成專案的「公司模型引擎」。
2. **它算出的公允價值 $117.7 vs 股價 $268.8（−56%）幾乎全由一個參數決定：β 2.08 → WACC 14.8%。** 這不是模型錯，是 DCF 的本性——所以專案端**不該把單點公允價值當訊號**，要用三情境區間 + 反向 DCF（市價隱含了什麼成長）+ 安全邊際。
3. **估值層的正確角色是「配多少、加不加碼、要不要退場觀察」，不是「什麼時候進場」。** 價值訊號的預測半衰期是季到年，動能是週到月；文獻與實務都指向兩者互補（價值選標的與部位、動能選時機）。專案架構已有現成插座：Alpha 疊加層（評分微調 ±0.15 上限）、Portfolio 層（相關性縮量）、trade_engine 的 `risk_pct`/`pyramid` 參數、thesis 失效價監測。
4. **真正「先一步佈署」的可行來源只有兩個免費管道**：(a) 分析師預估修正（yfinance 的 revenue/earnings estimate 與 EPS revisions）——這是有文獻支持的中期領先因子；(b) 公司指引 + 在手訂單/book-to-bill（VRT 模型的核心驅動）——需人工輸入（Telegram 指令）或從財報新聞抓。兩者都能讓模型公允價值**在股價反應前先動**，那就是佈署提前量的來源。
5. **分五階段、約 6–8 個工作 session**，每階段都可獨立上線；接進引擎的每一步都先過 `engine_backtest` 的 holdout 把關，沒證據就只顯示不動作（與專案一貫「不為調而調」一致）。

---

## 1. VRT 模型解剖

### 1.1 架構（資料流）

```
rIS / rBS / rCFS / rSegment / rOrders / rShares / rAdj      ← as-reported（藍字），每張有 Check 列 = 0
        │  (GAAP↔Adjusted 橋接：無形資產攤銷為主要差異)
        ▼
Operating Statistics（驅動層，黃底=關鍵假設，Notes 欄寫理由）
   ├ 營收：三大區域各自 YoY（美洲 49%→15% 遞減、APAC 18%→12%、EMEA 15%→18% 遞增）
   ├ 產品/服務占比（服務 17%→16.2%，附「服務隨安裝基礎 2-3 年遞延」的因果解釋）
   ├ 毛利率 Build：產品毛利 38.2%→38.8%、服務 42.5% 固定 → 混合 GPM 38.8%→39.3%
   ├ SG&A % 營收 15.1%→14.2%（營運槓桿遞減）→ EBIT margin 20.9%→24.0%
   ├ 營運資金天數：DSO 110、DIO 81→79、DPO 98→96、遞延收入/在手訂單 12.8%
   ├ 資本密集：PP&E/營收 9%→10.5%，折舊 = 期初 PP&E × 15.1%，攤銷排程遞減
   ├ 在手訂單合理性：Book-to-bill 1.15→1.0、backlog coverage、implied orders
   └ 其他：SBC 0.4%、資本化軟體 0.1%、債務利率 5%、現金利率 4%→3.5%、股利、股數 +1%/年
        ▼
Model（三表整合：IS → BS（Balance Check 0）→ CFS；附表：營運資金 / PP&E+無形 / 債務+租賃+利息 / 權益+股數）
        ▼
WACC（CAPM：rf 4.74%、β 2.08、MRP 5% → CoE 15.1%；CoD 6.04%、稅 25%、權益權重 97% → WACC 14.83%）
        ▼
DCF（2026-30 取自 Model、2031-35 以 8% 成長 / 24% OPM 淡出；TV Gordon g=3%；EV 46.4bn − 淨負債 1.33bn = 45.0bn ÷ 382.6m 股 = $117.7）
   └ 敏感度：WACC × 終端成長、WACC × 長期營收成長（Excel Data Table）
```

### 1.2 它做對的事（可直接搬進專案的設計原則）

| 原則 | 模型中的體現 | 專案對應 |
|---|---|---|
| **原始資料與判斷分離** | rXXX 只放 as-reported，判斷全在 Operating Statistics | 純邏輯/抓取層分離（本專案既有慣例） |
| **每個假設都有理由** | Notes 欄：「EMEA 法規開放後追上」「我們沒有 edge 預測匯兌，Base Case 設 0」 | thesis.py 的「論點要可被推翻」 |
| **完整性檢核** | rIS 四條 rebuild = 0、rSegment 對帳 rIS、BS Balance Check、Adjusted OP 對帳 | 選測 selftest 的「守恆律」思維 |
| **用天數而非 % 營收做營運資金** | DSO/DIO/DPO/遞延收入天數 → CCC 29→61 天 | 現行 valuation.py 用 ΔNWC = 1% 增量營收（過粗） |
| **資本支出從資產端推** | PP&E/營收 → 期末 PP&E → Capex = ΔPP&E + 折舊 | 現行用 Capex % 營收單一比率 |
| **營收有外部錨** | FY2026E 13,998 ≈ 公司指引中點 14,000；backlog coverage 與 implied orders 做 sanity | 專案沒有任何「指引/預估」輸入 |
| **兩段式預測期 + 淡出** | 2026-30 明確建模、2031-35 淡出到穩態 | 現行 5 年單段 |
| **敏感度表而非單點** | 兩張 5×5 | 現行 sensitivity_grid 有，但未進決策 |

### 1.3 審視發現（對模型本身，供你精修 Excel）

| # | 發現 | 影響方向 | 量級（就地估算） |
|---|---|---|---|
| 1 | **β 2.08 是原始迴歸 β**，未做 Blume 收縮（0.67β+0.33=1.72）或同業/基本面 β | 低估價值 | β 1.72 → WACC ≈13.1% → 由敏感度表內插約 $143；β 1.4 → WACC ≈11.5% → 約 $170 |
| 2 | 期末折現（`1/(1+WACC)^n`，n=1..10），未用期中慣例 | 低估 | 約 +3–5%（業界慣例估計；WACC 越高越接近上緣） |
| 3 | 股數用**發行在外 382.6m** 而非**稀釋加權 390.7m** | 高估 | 約 −2% |
| 4 | 淡出期 NWC = 18% × 增量營收；模型自身歷史 NWC/營收 6-13%，且 2025 CCC 只有 29 天（遞延收入預收撐住） | 低估 | 淡出期 FCF 少約 4-6 億/年 |
| 5 | 稅率 25% vs FY2025 有效 23.5%（含估值備抵釋放，正常化 25% 合理）——OK，只是註明 | 中性 | — |
| 6 | 現金滾到 2030 年 170 億、零回購（資本配置未建模）；對 FCFF 無影響，但 EPS 路徑與利息收入偏樂觀 | 中性（DCF） | — |
| 7 | 敏感度 WACC 軸只掃 12.8–16.8%，**沒有覆蓋市價隱含的區間**——反向 DCF 會告訴你：在 14.8% 下沒有任何合理成長路徑能到 $269；市價在賭的是「折現率」而不只是「成長」 | 認知 | 建議加一張「市價隱含成長」表 |
| 8 | 2026E 已錨定指引，但**指引更新（7/29 上調）後沒有版本紀錄**——模型無法回答「上次估值多少、這次為什麼變」 | 流程 | 這正是專案能補的：估值歷史序列 |
| 9 | **終值折現期錯誤（2026-09-09 建引擎對齊時發現）**：`DCF!G44 = G45*R10`，R 欄是 2031E（n=6）的折現因子 0.4361，終值卻是 2035E 末的現金流，應用 V10（n=10）的 0.2508 | **高估** | 終值 PV 26,860 → 15,450；EV 46.4bn → 35.0bn；**每股 $117.7 → $87.9**（同樣假設下）。與 #1、#2 疊加後的公允區間會整體下移；請先修這格再看其他 |

**結論**：模型的「引擎」很扎實，弱點集中在**折現率與呈現方式**（單點 vs 區間、缺反向 DCF、缺版本歷史），外加一個公式錯位（#9）。這正是為什麼專案端要把折現引擎程式化並用 Excel 當回歸測試——`company_model.py` 的自測明確重現了 Excel 的 117.7 與正確的 87.9 兩個數字。

### 1.4 模型引用的參考

- **一手資料**（rXXX 表的 SOURCE 欄）：Q4 2021–2025 earnings release、10-K 分部揭露、2026-07-29 上調後的 FY2026 指引（營收 13.8–14.2bn、Adj. OP 3,285–3,365m、Adj. FCF ≥2.4bn）、backlog 口述（FY2025 約 15bn）。
- **方法論**：McKinsey《Valuation》式驅動三表 + ROIC/IC 拆解；Damodaran 的 CAPM/ERP 框架；CFA 課綱的 DSO/DIO/DPO 營運資金法；工業股常用的 backlog coverage / book-to-bill 領先指標。
- 專案既有 `valuation.py` 依循 Anthropic financial-services skill 的 dcf-model / comps-analysis（FCF 五步驟、期中折現、g<WACC 硬約束、終值占比 50-70% 健康區間）——兩者可互補：**用專案的折現/敏感度骨架，換上 VRT 模型的驅動層。**

---

## 2. 專案現況缺口

| 面向 | 現況（valuation.py / fundamentals.py） | VRT 模型水準 | 缺口 |
|---|---|---|---|
| 營收 | 4 年 CAGR 夾 0–25% 一刀切 | 分區/分產品驅動 + 指引錨 + backlog 檢核 | **無任何前瞻輸入** |
| 利潤率 | 最近一年 EBIT margin 持平 | GPM build + SG&A 槓桿路徑 | 無路徑 |
| 營運資金/Capex | % 營收 | 天數法 / 資產端推 | 精度 |
| WACC | CAPM，β 原始值 | 同（同樣沒收縮） | 兩邊都要修 |
| 輸出 | 單點 + 敏感度文字 | 單點 + 兩張表 | **缺三情境、反向 DCF、歷史序列** |
| 用途 | `/dcf`、網頁公司分析頁**顯示** | 人腦判斷 | **完全沒接進引擎的資金佈建** |

---

## 3. 目標架構：估值層（Valuation Layer）

```
                 ┌──────────────── 資料層 fin_data.py ────────────────┐
 yfinance 三表(年/季)   yfinance 分析師預估(營收/EPS/修正)   Finnhub 備援   人工輸入(/model set：指引、backlog)
                 └───────────────────┬───────────────────────────────┘
                                     ▼
                     company_model.py（驅動式 5+5 年模型，純邏輯）
                     • 驅動預設 = VRT 模型的啟發式（3 年均值、線性淡出到穩態）
                     • 前瞻錨：下兩個 FY 營收/EPS 共識 → 覆蓋前兩年驅動
                     • 三情境 bear/base/bull（成長 ±、利潤率 ±、WACC ±）
                     • 反向 DCF：市價隱含的 5 年營收 CAGR / 穩態 OPM
                     • WACC：Blume 收縮 β、產業 β 上下限、rf 即時
                                     ▼
                     valuation_signal.py（把估值變成有界訊號）
                     • MoS = 公允(base)/市價 − 1；區間位置 = (市價 − bear)/(bull − bear)
                     • 預估修正動能 = 近 30/90 天共識 EPS/營收變動（有方向、有幅度）
                     • 品質 = ROIC、FCF 轉換、淨負債/EBITDA
                     • V ∈ [−1, +1]（夾制、可解釋、附一行理由）
                                     ▼
        ┌──────────────┬────────────────┬──────────────────┬────────────────────┐
   部位大小乘數     贏家加碼閘門      候選排序傾斜         論點 KPI 監測
   risk_pct×f(MoS)  價>bull 不加碼    score+λ·V 排序       每季實際 vs 模型路徑
   (0.5×–1.25×)     MoS>30% 提早加碼  (λ≤0.1，同 ±0.15 精神) → 偏離告警/信心降級
        └──────────────┴────────────────┴──────────────────┴────────────────────┘
                                     ▼
                  engine_backtest：PIT 滯後基本面 → holdout 通過才啟用（預設 off）
```

**設計鐵律**
- 估值層**只調整既有決策的量與傾斜，不新增進場觸發**：訊號仍由技術評分過門檻，避免「便宜所以買」的價值陷阱。
- 所有影響量**有界**（乘數 0.5–1.25、排序傾斜 λ≤0.1），與 Alpha 層 `max_abs_delta=±0.15` 同一哲學：權重未經校準前先限縮影響力。
- **預設關閉**（`thresholds["val_enabled"]=False`）。`/engtest opt` 加 `val` 維度，holdout 勝過才建議開。
- 使用者的手工模型（Excel）是**一等公民**：可匯入公允價值/情境/KPI 覆蓋自動模型；自動模型只是「沒人工模型時的預設」。

---

## 4. 分階段計畫

### P0 — 資料地基（1 session）
- `fin_data.py`：統一取數介面 `get_financials(ticker) → {annual, quarterly, estimates, price, shares, debt, cash, as_of}`；yfinance 主、Finnhub 備援（同 insider 的雙源模式）；**每個數字帶 `filed`/`period_end` 日期**（回測 PIT 用）。
- 快取：`state["fin_cache"]`（週 TTL、每輪最多刷新 3 檔輪替，同 alpha_cache 限額模式，防 cron 超時）。
- 自測：離線用內建範例 JSON（VRT FY2021-25 as-reported 數字就是最好的 fixture——從你的 Excel 抽出來當測試資料）。
- 事實查證：yfinance 各預估欄位、Finnhub 免費層端點（查證代理報告見附錄 B）。

### P1 — 公司模型引擎（1–2 sessions）
- `company_model.py`（純邏輯）：
  - `derive_drivers(hist, estimates)`：由歷史推預設驅動（GPM 3 年均值、SG&A% 線性延續遞減幅度收斂、DSO/DIO/DPO 3 年均值、PP&E/營收、D&A/期初 PP&E、SBC%、稅率正常化）；有共識就用共識覆蓋前兩年營收/EPS。
  - `project(drivers, years=5, fade=5)`：兩段式 FCFF（含 SBC 視為現金成本的開關）。
  - `wacc(beta_raw, ...)`：Blume 收縮 + 產業上下限 + 情境 ±1%。
  - `scenarios()`：bear/base/bull 三組驅動 → 三個公允價值。
  - `reverse_dcf(price)`：解「市價隱含 5 年營收 CAGR」（固定其他驅動）。
  - `model_text()`：Telegram 版（單 *、無底線）。
- 重用 `valuation.dcf_value`（期中折現、g<WACC 約束、TV 占比檢查）與 `sensitivity_grid`。
- Bot：`/model VRT`（自動模型三情境 + 反向 DCF + 驅動表）、`/model VRT set rev_g 0.37,0.29,0.22 opm 0.24 beta 1.4 guidance_rev 14000 backlog 15000`（人工覆蓋，存 `state["models"]`，**加入 SENSITIVE_KEYS**）、`/model VRT clear`。
- 自測：用 VRT Excel 的驅動值餵入，**公允價值須落在 Excel 的 $117.7 ±3%**（期末折現開關對齊）——這是最強的回歸測試：模型與你的手工模型互證。

### P2 — 估值訊號與歷史序列（1 session）
- `valuation_signal.py`：V 分數、MoS、區間位置、預估修正動能、品質；`state["val_hist"][ticker]` 每週一點（公允價值 base/bear/bull、市價、MoS）——回答「上次估多少、為什麼變」。
- 晨報加一行：「估值層：X 檔 MoS>25%、Y 檔價格高於 bull 情境」。
- 網頁「🏛️ 公司模型」頁：驅動表（可調滑桿即時重算）、三情境、反向 DCF、敏感度熱圖、公允價值歷史 vs 股價、**匯出 VRT 風格 Excel**（openpyxl，沿用你的 Format 慣例：藍字輸入/黑字公式/黃底假設，含 Balance Check）。
- Excel 匯入橋：網頁上傳含 `RBS_Summary` 工作表的 xlsx（欄位規格見附錄 A）→ 頁面生成一條 `/model VRT set ...` 指令讓你貼進 Telegram（網頁不直接寫 state：state 由 bot workflow commit，網頁唯讀——既有慣例）。

### P3 — 接進資金佈建（1–2 sessions，含回測）
- `trade_engine.decide` 新增可選輸入 `val_ctx = {sym: {"mos", "pos_in_range", "rev_mom", "quality"}}`：
  - **sizing**：`risk_pct_sym = risk_pct × clamp(1 + 0.5×MoS, 0.5, 1.25)`（MoS +50% → 1.25×；−50% → 0.75×；價超 bull → 0.5×）。仍受 `max_position_pct` 與 headroom 約束。
  - **pyramid 閘**：`price > bull_fair` 不加碼；`MoS > 0.3 且 rev_mom > 0` 加碼門檻由 +1.0R 降到 +0.75R。
  - **候選排序**：`sort_key = score + λ×V`，λ 預設 0.1（等於「估值最多影響一個評分級距」）。
  - **不做**：不因便宜進場、不因貴出場（出場仍全由價格機制）。
- `alpha_overlay` 的 notes 增加估值一行（`/alpha` 可見）；`/positions` 每檔附 MoS。
- `engine_backtest`：precompute 加 `val_ctx` 歷史（**PIT：財報數字只在 `filed` 日後 1 個交易日可見；共識用當時快照——沒有歷史共識就只回測 MoS 部分並明講**）；GRID 加 `val_enabled ∈ {0,1}` 與 `val_lambda ∈ {0, 0.1}`；同一套三段 walk-forward + DSR 把關。**holdout 沒贏就保持 off。**

### P4 — 論點 KPI 監測（前瞻性的核心，1 session）
- 模型的驅動就是**可證偽的 KPI**：把 base case 的營收/GPM/OPM/backlog 路徑存成 `thesis` 的 pillar；每季財報後（earnings_review 已能判定「剛出財報」）自動比對實際 vs 模型：偏離 >1 個標準差 → TG 告警 + `conviction` 降級 + 建議 `/model VRT set` 更新。
- 指引變動偵測：人工 `set guidance_rev` 後，公允價值變動 >10% 自動推播「估值重估：$X → $Y，MoS 由 a% → b%」——**這就是「先一步佈署」的觸發點**：估值先動、部位乘數跟著動、價格還沒完全反應。
- 預估修正動能：每週比較共識 EPS/營收快照，連續兩週上修且 MoS>0 → 加碼閘門放寬；連續下修 → sizing 乘數上限 1.0。

### P5 — 擴大宇宙（選做，1 session）
- `stock_db` 的 AI 供應鏈瓶頸主題（光通訊/HBM/電力液冷）→ 每檔跑自動模型 → 「MoS × 預估修正 × 技術評分」三維篩選 → 候選加入 watchlist 的建議清單（`/screen value`）。
- 這一步才是「用估值選標的」；前四步都是「對現有標的配多少」。

### 每階段 DoD（沿用 CLAUDE.md）
模組自測 → app.py AST → 環境變數四處同步（無新增）→ ci.yml/Colab Cell 2 加模組 → README/GITHUB_ACTIONS/CLAUDE.md → 子代理對抗驗證（金融演算法 + 網頁）→ commit+push → 提醒 Reboot。

---

## 5. 引擎接口細節（P3 的規格）

| 接口 | 公式 | 夾制 | 失效安全 |
|---|---|---|---|
| 部位乘數 | `m = 1 + 0.5 × MoS_base` | `[0.5, 1.25]`；`price > bull` 強制 `0.5` | 無估值資料 → `m = 1`（完全等於現狀） |
| 加碼閘 | `allow_add = price ≤ bull_fair` | — | 無資料 → 允許（現狀） |
| 提早加碼 | `pyramid_r_eff = 0.75 if MoS > 0.3 and rev_mom > 0 else pyramid_r` | 不低於 0.75 | 無資料 → 原值 |
| 候選傾斜 | `key = score + λ·V`，`V = clamp(0.6·MoS_norm + 0.4·rev_mom_norm, −1, 1)` | `λ ≤ 0.1` | 無資料 → `V = 0` |
| 縮倉（品質） | `淨負債/EBITDA > 4 或 FCF 轉換 < 0.5 連兩年` → `m = min(m, 0.75)` | — | 無資料 → 不動 |

所有輸入在 `decide` 入口做 `isfinite` 消毒（B9 原則：防護放純函數入口）。

---

## 6. 回測與驗證計畫

1. **PIT（point-in-time）紀律**：年報/季報數字自 `filed`+1 交易日起可見；yfinance 三表無 `filed` 欄時以期末 +60 天保守近似（會低估而非高估估值層效果）。
2. **兩層回測**：(a) `engine_backtest` 開關 `val_enabled` 的 A/B（同一套 walk-forward）；(b) 單因子檢驗：MoS 五分位 vs 未來 60/120 日超額報酬（Spearman IC），樣本 = watchlist ∪ 主題股 ≈ 40 檔 × 2 年——**樣本小，只求方向一致，不求顯著**。
3. **上線門檻**：holdout 報酬 ≥ baseline +1pp **且** 回撤不劣化 **且** DSR 不低於 baseline；三者缺一維持 off。
4. **對抗驗證焦點**：前視（共識快照日期、財報可見日）、倖存者偏差（watchlist 是事後挑的——只能承認，不能修）、估值層是否偷偷變成進場條件（grep 所有 `val_ctx` 使用點）。

---

## 7. 風險與已知限制（先講清楚）

- **免費資料的前瞻深度有限**：共識預估只到 +2 FY、無分區/分產品；VRT 模型那種「EMEA 法規開放」的判斷只能靠你手工輸入。這是為什麼設計上人工模型優先。
- **DCF 對 WACC 極度敏感**（§1.3 #1、#7）：軟體只能把敏感度攤開，不能消除。訊號用 MoS **區間位置**而非單點，就是為了鈍化這一點。
- **價值陷阱**：便宜的股票常有便宜的理由。估值層不觸發進場、不否決價格停損，就是防這個。
- **cron 預算**：每輪最多刷新 3 檔模型（各約 2-4 次 API 呼叫）；完整重算走 `/model` 手動或每週日。
- **SEC 封鎖 GitHub 機房 IP**（PITFALLS B10）：XBRL companyfacts 在 cron 環境不可靠，只作本地/網頁端補充，不進主路徑。
- **心理風險**：有了「公允價值」很容易想凹單（「它值 170 為什麼要停損」）。出場機制不讀估值層，這條線不能退。

---

## 8. 需要你拍板的決策

| # | 決策 | 選項 | 我的建議 |
|---|---|---|---|
| 1 | 估值層的權限 | A 只調部位/加碼/排序（不觸發進場） ／ B 也可作進場條件（MoS>x 且評分>y） | **A**。B 等單因子 IC 有數據再談 |
| 2 | 前瞻資料優先序 | A 共識預估自動抓為主、人工覆蓋為輔 ／ B 人工指引輸入為主 | **A**（可持續），但 P2 就做匯入橋讓你的 Excel 隨時覆蓋 |
| 3 | 是否做 Excel 匯出/匯入橋 | 做 ／ 不做 | **做**：這是把你手工研究接進系統的唯一路徑，也是回歸測試的來源 |
| 4 | 上線順序 | A 先 P0→P1→P2（只顯示，不動引擎）跑一個月看估值歷史合不合理，再 P3 ／ B 直接做到 P3 帶回測 | **A**。估值層的錯誤比技術指標隱蔽得多，先觀察 |
| 5 | 宇宙 | A 現有 18 檔 watchlist ／ B 加 stock_db AI 瓶頸主題約 20 檔 | 先 **A**，P5 再擴 |

---

## 9. 2026-09-08 修訂（開源生態對照研究後）

五組研究代理的完整對照與差異化規劃見 **[VALUATION_LANDSCAPE.md](VALUATION_LANDSCAPE.md)**。對本文的修訂：

- **§3 架構**新增「Universe 層」（四層：Benchmark / Core / Theme / Broad，三段管線月頻重建）與「PIT 快照帳本」——估值層要進回測，先修現行 `engine_backtest` 用今日名單的倖存者偏差。
- **§4 階段計畫由 LANDSCAPE §8 取代**（P0–P6）：P0 加入預估修正每日快照帳本與成分歷史；P1 加入 `quality.py`（Piotroski/Altman/Beneish/Sloan）、反向 DCF 三向求解、產業路由（金融股 RIM）、九條審核清單；P2 加入 `factor_eval.py`（alphalens 式 IC 驗證）；P3 部位乘數改以 Black-Litterman 觀點整合為主、乘數為退路；P4 新增 `guidance.py` 指引/KPI 萃取。
- **§5 引擎接口**維持有界與失效安全原則；新增「審核未過 → verdict=review，Bot 不推播買訊」。
- **§8 拍板決策**更新為 LANDSCAPE §9（新增第 6、7 項：不引入 FinanceToolkit 依賴、LLM 萃取用 Haiku 抽 + Sonnet 校）。
- 資料源定案：yfinance 共識/修正快照 + Finnhub 三支免費端點（recommendation / earnings calendar / financials-reported）+ 自建歷史；SEC 只作回測資料集；Alpha Vantage 可選。

## 10. 進度看板（2026-09-09）

| 階段 | 狀態 | 落地 |
|---|---|---|
| P0-a 預估快照帳本 | ✅ 上線（生產環境已寫 `estimates_ledger.json`） | `estimates_ledger.py`、`/est`、Alpha Vantage SUE 回填 |
| P0-b 選股池 | ✅ | `universe.py`、`/universe`、月頻快照 `data/universe/`、成分期間表解析 |
| P0-c PIT 三表 | ✅ | `fin_data.py`（first-seen、Finnhub as-reported 備援、`data/fin/`） |
| P1 公司模型 | ✅ | `quality.py`、`company_model.py`、`/model`、VRT Excel 回歸（含其終值折現期錯誤） |
| P1.5 產業路由 | ✅ | 金融 RIM、地產 DDM，金融業品質模式 |
| P2 訊號/驗證/網頁 | ✅ | `factor_eval.py`（IC/ICIR/NW t/門檻）、網頁「🏛️ 公司模型」（滑桿、football field、匯出/匯入橋） |
| 整合層 | ✅ | `playbook.py`、`/playbook`、網頁「🧭 佈局計畫」、閒置輪每 7 天輪替建模、週報摘要 |
| P3 接資金佈建 | ✅ 程式就緒、**預設關閉** | `trade_engine` val 欄位（乘數/加碼閘/傾斜）、`engine_backtest` PIT val_ctx + 估值層 A/B、`/set val_enabled`、`/rebalance bl`（Black-Litterman） |
| P4 指引萃取 | ✅ 首版 | `guidance.py`、`/guidance`（AV 逐字稿 + LLM 定位轉錄 + 程式驗證） |
| P5 宇宙擴大（主題層 Stage 3、/screen） | ✅ | `screener.py`、`/screen`、每週閉市輪刷新；只建議不自動加入 |
| P6 治理月報 | ✅ | `val_report.py`、`/valreport`、每月自動推播（事後命中／穩定度／MoS IC／覆蓋） |

**啟用順序（維持原拍板）**：先讓 val_hist 與預估帳本累積 → `/engtest opt` 看估值層 A/B 是否過 holdout →
才 `/set val_enabled on`。在此之前估值層只在 `/playbook`、`/model`、網頁顯示。

## 附錄 A — `RBS_Summary` 工作表規格（Excel 匯入橋）

在你的模型加一張名為 `RBS_Summary` 的表，A 欄鍵、B 欄值（全部用公式連到模型，不要手打）：

| 鍵 | 值 | 例（VRT） |
|---|---|---|
| ticker | 代碼 | VRT |
| model_date | 模型日期 | 2026-09-05 |
| fair_base / fair_bear / fair_bull | 三情境每股價值 | =DCF!G48 / 情境表 |
| wacc / tgr | 折現率 / 終端成長 | =WACC!F22 / =DCF!G33 |
| rev_path | 5 年營收（逗號分隔） | =TEXTJOIN(",",1,Model!L14:P14) |
| opm_path | 5 年 OPM | =TEXTJOIN(",",1,'Operating Statistics'!M37:Q37) |
| kpi_backlog_cov_min | 論點失效門檻：backlog coverage 低於此值 | 0.7 |
| kpi_gpm_min | GPM 低於此值視為論點受損 | 0.37 |
| note | 一句話論點 | AI 資料中心電力/冷卻瓶頸受益者 |

網頁讀取後生成：`/model VRT set fair 117.7 bear 95 bull 170 wacc 0.148 tgr 0.03 kpi backlog_cov>=0.7 gpm>=0.37`。

## 附錄 B — 事實查證（資料源與文獻，2026-09-07 查證代理）

> 證據等級：**A 節最強**（直接拆 yfinance 1.5.2 wheel 原始碼）；B/C/E 為官方頁面的搜尋摘要（開發環境 proxy 擋住 finnhub/sec.gov 直連），實作 P0 時在 GitHub Actions 環境再實測一次。

### B.1 yfinance 1.5.2 前瞻資料 — 全部存在、免費無 key（走 quoteSummary `earningsTrend` / `financialData`）

| 屬性 | 型別 | index / 欄位 |
|---|---|---|
| `revenue_estimate` | DataFrame | index `0q +1q 0y +1y`；`numberOfAnalysts avg low high yearAgoRevenue growth` |
| `earnings_estimate` | DataFrame | 同 index；`numberOfAnalysts avg low high yearAgoEps growth` |
| `eps_trend` | DataFrame | 同 index；`current 7daysAgo 30daysAgo 60daysAgo 90daysAgo` ← **預估修正動能的直接來源** |
| `eps_revisions` | DataFrame | 同 index；`upLast7days upLast30days downLast7days downLast30days` |
| `growth_estimates` | DataFrame | index 多 `+5y -5y`；欄實際為 `stockTrend industryTrend sectorTrend indexTrend`（docstring 寫法不同，以程式碼為準） |
| `analyst_price_targets` | **dict** | `current low high mean median` |
| `recommendations_summary` | DataFrame | = `recommendations` 別名；`period strongBuy buy hold sell strongSell` |
| `earnings_history` | DataFrame | index 季度；`epsEstimate epsActual epsDifference surprisePercent` |

注意：欄位是 Yahoo JSON key 攤平，缺欄就不存在 → 一律 `.get`；只取最近 4 期。
財報列名（`pretty=True`，Title Case、縮寫全大寫）：損益表 `Total Revenue / Operating Income / EBIT / Tax Provision / Pretax Income / Diluted Average Shares / Reconciled Depreciation`（**損益表沒有 `Depreciation And Amortization`**，D&A 要從現金流量表拿）；資產負債表 `Total Debt / Cash And Cash Equivalents / Ordinary Shares Number / Accounts Receivable / Inventory / Accounts Payable / Current Deferred Revenue / Net PPE`；現金流量表 `Depreciation And Amortization / Capital Expenditure / Stock Based Compensation / Free Cash Flow / Operating Cash Flow`。Yahoo 最多 4 年 / 5 季。

**對規劃的影響**：P0 的營運資金天數（AR/Inventory/AP/Deferred Revenue 都有）、PP&E 法（Net PPE + Capex + D&A）、SBC 成本化、共識營收/EPS 錨、預估修正動能——**全部可由 yfinance 免費取得**，不需要付費源。

### B.2 Finnhub 免費層

| 端點 | 結論 |
|---|---|
| `/stock/price-target`、`/stock/revenue-estimate`、`/stock/eps-estimate` | **Premium**（官方 docs 標示）→ 分析師預估**不走 Finnhub**，用 yfinance |
| `/stock/recommendation`、`/stock/metric?metric=all`、`/stock/financials-reported`、`/calendar/earnings`（含 epsEstimate/revenueEstimate） | 免費（部分確認；as-reported 在免費層可能有範圍限制）→ 只當 metric / 財報備援 |
| 限流 | 60 次/分 + 30 次/秒，超限 429（現行 finnhub_data.py 節流設計仍適用） |

### B.3 SEC XBRL companyfacts

- 結構 `facts.us-gaap.<Concept>.units.USD[]`，每筆 `start end val accn fy fp form filed frame`。
- Concept 需 fallback 鏈：`Revenues` → `RevenueFromContractWithCustomerExcludingAssessedTax` → `...IncludingAssessedTax`；Capex `PaymentsToAcquirePropertyPlantAndEquipment` → `PaymentsForCapitalImprovements`。
- **`filed` 可做 point-in-time**：同一 (concept, period) 會在 10-K/10-Q/修正案重複出現，回測取 `filed ≤ as_of` 的首次揭露；用 `fp` + `start/end` 區分季/年避免重複計算。
- 限流 10 req/s、必須帶識別 User-Agent；**GitHub 機房 IP 被 SEC 封鎖（PITFALLS B10）** → 只作本地/網頁端與回測資料集建置，不進 cron 主路徑。

### B.4 文獻與方法（實作時引用）

1. 價值×動能負相關、組合互補：Asness, Moskowitz & Pedersen, "Value and Momentum Everywhere", *Journal of Finance* 68(3), 2013.
2. 盈餘預估修正/盈餘驚奇的中期漂移：Chan, Jegadeesh & Lakonishok, "Momentum Strategies", *JF* 51(5), 1996（極端 SUE 組六個月價差約 7.5%；分析師修正反應遲緩）；實務框架 Zacks Rank（Agreement/Magnitude/Upside/Surprise）。
3. β 收縮：Blume, "Betas and Their Regression Tendencies", *JF* 1975；Bloomberg adjusted β = 0.67×raw + 0.33。
4. 反向 DCF：Mauboussin & Rappaport, *Expectations Investing*（2021 修訂版）；Damodaran 市場隱含法。
5. 期中折現：折現期 0.5, 1.5, 2.5…；對 EV 影響約 +3–5%（valuation.py 已採用）。
6. Backlog / book-to-bill 為營收領先指標：發行人 10-K 慣用表述（如 Parsons）；SEMI 半導體指標通論。

### B.5 VRT 公開數字核對（模型 rOrders 表）

| 項目 | 結論 |
|---|---|
| FY2025 營收 ≈ $10.2bn、年底 backlog $15.0bn（+109%）、Q4 book-to-bill ≈ 2.9x | **確認**（2026-02-11 新聞稿） |
| FY2026 指引路徑：2/11 初始 $13.25–13.75bn → 4/22 上調 $13.5–14.0bn → 7/29 再上調 +$250M、中點 $14.0bn | **確認**中點與上調幅度；模型寫的 $13.8–14.2bn 區間僅二手來源可見，與中點一致 |
| Q2 2026 backlog | 公開摘要未揭露 → 模型不應假設 2026 年中 backlog 數字（現行模型也沒有，OK） |

### B.6 查證後對計畫的三個修正

1. **前瞻資料源定案為 yfinance**（共識營收/EPS、`eps_trend` 90 天修正軌跡、`eps_revisions` 上下修計數、目標價區間）；Finnhub 只做 metric/財報備援；SEC XBRL 只做回測 PIT 資料集。
2. P1 的 D&A 取自現金流量表列 `Depreciation And Amortization`，不要在損益表找。
3. §1.3 #2 期中折現的量級由「約 +7%」修正為「約 +3–5%」。
