# 估值層 × 選股池：開源生態對照與差異化規劃（2026-09-08）

> 延續 [VALUATION_PLAN.md](VALUATION_PLAN.md)（VRT 模型解剖與 P0–P5 初版）。本文回答三個問題：
> **其他開源專案怎麼做？要做「未來部署」該參考什麼、算什麼？我們怎麼做得更完整、更專業、選股池更廣？**
> 方法：五組研究代理平行查證（開源估值函式庫／交易框架宇宙與 PIT／LLM 代理專案／免費前瞻資料與成分股來源／專業方法論與觀點整合），每項結論附證據等級與來源（§10）。開發環境 proxy 擋住多數官網，能直接讀原始碼的以原始碼為準。
> 教育用途、非投資建議。

---

## 0. 五個結論

1. **沒有任何開源專案把「多段 DCF + 反向求解 + 三情境/蒙地卡羅 + 品質/舞弊評分 + 預估修正序列 + 無前視回測」做在同一顆引擎裡。** FinanceToolkit 只有單段等速 DCF；反向 DCF 全在 0–1 星玩具倉庫；分析師預估修正因子全網沒有免費實作；估值接回測的專案是零。這是我們的空位。
2. **專業框架（Lean / Zipline / Qlib）的核心不是因子，是「宇宙管線 + point-in-time 紀律」**：基本面 T+1 才可見、`asof_date` 與 `timestamp` 兩欄、成分股期間表（回測任一天用「當天成分」）。我們現在的 `engine_backtest` 重放的是今天的名單，**內建倖存者偏差**，估值層要進回測前必須先補這一層。
3. **LLM 代理專案（ai-hedge-fund 63k★、TradingAgents 103k★）的估值是純 prompt，回測是收盤成交零成本固定持有 5 日，作者自己承認 agent 很少是 edge、正確 as-of 資料才是護城河。** 值得借的是工程紀律：LLM 只產可驗證假設、數值由確定性程式算、JSON 解析失敗即棄權、風控夾限留稽核軌跡、以 filing_date 過濾快照。
4. **免費資料的天花板很清楚**：yfinance 給當下的共識/修正快照（無歷史）、財報只 4 年；Finnhub 免費層有 recommendation 月頻歷史、earnings calendar、as-reported 財報約 15 年（補掉 SEC 對 GitHub 機房的封鎖）；歷史共識沒有免費源，**只能從今天起自建每日快照帳本**。宇宙成分的歷史反而免費齊全（S&P 500 自 1996、NDX 自 2015、iShares 持股月底可回溯）。
5. **「比他們更完整、更專業」的實質內容 = 十項組合**（§6）：專業級 WACC、football field 多法區間、反向 DCF、蒙地卡羅、品質/舞弊/破產三套評分、ROIC 成長融資恆等式健檢、自建修正動能序列、指引萃取 JSON、Black-Litterman 觀點整合、九條審核清單 + 版本歷史——而且**每一項都先過 walk-forward holdout 才准動引擎**。

---

## 1. 開源估值／基本面函式庫

| 專案 | 授權 / 版本 | DCF | 反向 / 情境 | WACC | 品質評分 | 預估 | 免費資料 | 裝進 cron |
|---|---|---|---|---|---|---|---|---|
| FinanceToolkit | MIT / 2.2.0 2026-08 | 單段等速 + Gordon | ✗ / ✗ | CAPM 但 Rm 用基準實際報酬、帳面債權重 | Piotroski/Altman/Beneish/Ohlson/Zmijewski/Springate/Fulmer/Grover、DuPont、EVA | ✗ | FMP 250/日 或 Yahoo 4 年 | △ py≥3.11、pandas≥3、拖 sklearn |
| FinanceDatabase | MIT / 2.4.0 2026-06 | ✗ | ✗ | ✗ | ✗ | ✗ | 自帶 30 萬檔 sector/industry/market_cap CSV | 直接讀 CSV ✓（pip 會拖進 Toolkit） |
| OpenBB Platform | AGPL / 4.7.2 | ✗ | ✗ | ✗ | ✗ | 僅 yfinance 目標價共識 | yfinance / sec | ✗ fastapi 全家桶 |
| edgartools | MIT / 5.56 2026-09 | ✗ | ✗ | ✗ | ✗ | ✗ | SEC XBRL，**`facts.query().as_of(date)` 原生 PIT** | ✓（pyarrow；SEC 對 GH 機房封鎖） |
| SimFin | MIT / 1.0.2 | ✗ | ✗ | ✗ | 內建 fin/growth/val signals | ✗ | 免費層延遲 12 個月、含 Publish Date | ✓ 極輕（只適合估值因子離線回測） |
| implied-expectations | MIT / 2026-07，1★ | 兩段 FCFF | **反向三向求解**（隱含成長 / 需維持年數 / 需要利潤率） | 外給 | ✗ | ✗ | SEC companyfacts | ✓ |
| EmanueleSturzo/DCF | MIT，3★ | 5 年 | 三情境 + 蒙地卡羅 10k | CAPM | ✗ | ✗ | yfinance | 腳本 |
| valinvest / PyValuation / autodcf / compdata | 2020–23 停更 | 簡單 | ✗ | ✗ | 改良 F | ✗ | FMP/爬蟲 | 勿裝 |

**可直接移植的設計**（MIT）：FinanceToolkit 的「一項準則一函數、控制器彙總」模組結構與 `sbc_adjusted_free_cash_flow`、`reinvestment_rate`、`income_quality_ratio`；implied-expectations 的 **再投資率 = g / ROIC、終值期 ROIC 收斂至 WACC（終值成長價值中性）** 約束；EmanueleSturzo 的 MC 分布參數（g N(μ,3%)、margin N(μ,2%)、WACC N(μ,1.5%)、終值 g U(1.5%,4%)）。

**環境警告**：stockdex 釘 `curl_cffi==0.12.0` 與 yfinance 1.5.2 衝突；openbb 不裝；edgartools 日更需釘死。

---

## 2. 交易框架：宇宙、PIT、分層

| 框架 | 宇宙 | 基本面進入點 | Point-in-time 規則 | 對我們的啟示 |
|---|---|---|---|---|
| **Lean** | Coarse（dollar volume 前 1000）→ Fine（總部/交易所/IPO ≥ 半年/市值 ≥ 5 億 → 產業配額 500），**只在換月重選** | `Fundamental` 物件：ValuationRatios ~130 欄、OperationRatios、EarningRatios、AssetClassification（Morningstar 148 產業、StyleBox、A–F 財務健康等級） | **T 日資料 T+1 可見**；`FileDate` 與 `PeriodEndingDate` 分欄；已下市補進但 2015–20 仍有缺 | Universe → Alpha(Insight: direction/magnitude/confidence/period) → PortfolioConstruction(PortfolioTarget) → Risk → Execution 五層；我們的 `decide()` 把後四層揉在一起 |
| **Zipline Pipeline** | `AverageDollarVolume(20).top(500)` | `CustomFactor.compute(today, assets, out, *inputs)` 橫截面；`rank/zscore/demean(groupby=產業)` | `asof_date`（資料指涉日）與 `timestamp`（可得知日）兩欄；重編放 deltas 表不覆寫 | 因子產業中性化；資料表雙日期欄是 PIT 的最小正確做法 |
| **Qlib** | `instruments/csi300.txt` = `symbol start end` 期間表 | Alpha158/360 **零基本面**；基本面走 PIT 資料庫 `P($$roewa_q)`，禁止引用未來期 | 「同一筆資料會被多次修正，只用最新版回測就是洩漏」 | RollingGen = 我們三段 walk-forward 的產品化；IC/ICIR/Rank IC 為標準報表 |
| **alphalens** | — | — | — | 慢因子用 21/63 日 Rank IC 與 ICIR、分位報酬 spread、`factor_rank_autocorrelation`（MoS 應 >0.9）、`group_adjust` 產業去均值 |
| **PyPortfolioOpt / skfolio** | — | — | — | `BlackLittermanModel(absolute_views, omega="idzorek", view_confidences)`；**沒有現成「DCF 公允價值當 view」範例**——要自建 |
| FinRL | — | Compustat 比率直接 bfill、無申報延遲 | 無 | 作者群自己的論文承認 DRL 回測過擬合；維持「不用深度 RL」決策 |

**我們 vs Lean/Zipline 的差距**（嚴重度）：
1. 無 Universe 層，watchlist 靠人工（高）
2. 回測用今日名單 → 倖存者偏差（高）
3. 基本面用 `.info` 即時快照，無歷史、無延遲欄 → 估值層目前不可回測（高）
4. 訊號無 magnitude/confidence/有效期（中）
5. 部位建構藏在 decide 內，`rebalance.py` 與引擎脫鉤（中）
6. 無流動性門檻（中，擴宇宙後必需）
7. 產業分類未用於中性化（低）
8. 無橫截面 IC 評估（做估值因子時變高）

---

## 3. LLM 代理型專案：借什麼、避什麼

| 專案 | 估值怎麼做 | 部位怎麼來 | 回測 | 借 | 避 |
|---|---|---|---|---|---|
| ai-hedge-fund v2 | 5 個 persona 純 prompt 輸出 signal/confidence；legacy 版有量化常數（owner earnings、三段 DCF cap 8%/4%/2.5%、折現 9–10%、四法加權 0.35/0.35/0.20/0.10、MoS 25%） | conviction 加權 → gross target；風控雙段夾限 + ClampEvent 稽核；夾掉的曝險留現金 | 收盤成交、零成本、固定 5 日、無基準 | filing_date 過濾快照；PromptCache；JSON 失敗即棄權；ClampEvent | 用 prompt 當估值；付費資料 |
| TradingAgents | 基本面分析師寫 Markdown 報告，**無估值模型** | 評級 5 級 + 自由文字 sizing；風控是三個 persona 吵架 | 三檔三個月，Sharpe 8.2（樣本小、疑 look-ahead） | regex `extract_rating()` 抓不到回 REVIEW 而非預設 Hold | 文字風控 |
| FinRobot / FinGPT | 年報 agent 結構化欄位（損益/資產負債/現金流/分部/風險/同業）；FinGPT 主力是情緒 | — | — | 年報欄位清單 | — |
| Anthropic financial-services | dcf-model（TV 50–70%、三張敏感度、每個 hardcode 附來源）、**earnings-analysis（指引新舊對照）、thesis-tracker（支柱配 KPI 與證偽條件）、model-update、catalyst-calendar、audit-xls** | — | — | 九個 equity-research skill 方法論可直接對應我們的 /preview、/thesis、模型審核 | — |

**LLM 估值的已知失敗模式**（2025–26 文獻）：參數化前視（記憶期回測報酬砍最多 −67%）、幻覺數字（通用偵測器漏掉 43% 需重算的算術錯誤）、StockBench 多數 LLM agent 輸給買進持有。緩解共識與我們既有 `committee.py` 一致：LLM 只產可驗證假設 → 數值由程式算 → 每個數字帶 evidence span → 只在 cutoff 後區間評估。

**我們有、它們沒有**：含成本次日開盤成交的 108 組三段 walk-forward + DSR；只證偽不證實的反駁器；機制歸因 + 行為體檢 + 鏡像帳 + Shadow；五因子遲滯氣象台 + Alpha 疊加層接硬風控；委員會 vs 量化交叉比較 + 反思命中率回饋。

---

## 4. 免費資料：矩陣與預算

| 來源 | 欄位 | 免費層 | PIT | GitHub cron |
|---|---|---|---|---|
| yfinance `eps_trend / eps_revisions / earnings_estimate / revenue_estimate / analyst_price_targets` | 共識 90 天軌跡、上下修家數、目標價 | 1 檔 1 請求、429 風險 | ✗ → **自建每日快照** | ✓（限流、輪替） |
| yfinance `screen(EquityQuery)` | marketCap / avgVol3m / sector / PE / EV-EBITDA / ROE / 負債比 / Altman Z… | 250 筆/頁 | ✗ | ✓（S&P 500 三次呼叫） |
| yfinance 三表 | 4 年年報 / 5 季 | Yahoo 硬限 | ✗ | ✓ |
| Finnhub `/stock/recommendation` | 月頻評等家數 | 60/分 | **✓ 回傳多月** | ✓ |
| Finnhub `/calendar/earnings` | 財報日 + EPS/營收預估 | 60/分 | 部分 | ✓ |
| Finnhub `/stock/financials-reported` | XBRL as-reported ~15 年、帶 accession | 60/分 | **✓** | ✓（補 SEC 封鎖） |
| Finnhub estimates / price-target / transcripts | — | **Premium** | — | 不用 |
| Alpha Vantage `EARNINGS_ESTIMATES` / `EARNINGS_CALL_TRANSCRIPT` | 共識 + 7/30 天修正數；逐字稿 + 情緒 | **25 次/日** | 逐字稿有歷史 | ✓（配給即將財報者） |
| SEC efts 全文搜尋 / companyfacts / FSDS 季 zip | 8-K 全文、XBRL、as-filed 季檔 | 10/s | **✓✓** | **✗ B10 封鎖**；Cloudflare Worker 代理可試但視為可缺席源 |
| fja05680/sp500、n100tickers、yfiua/index-constituents | 成分股歷史 | GitHub raw / pip | ✓（1996 / 2015 / 2023-07） | ✓ |
| iShares ajax CSV（IWB/IJH/SOXX/IVV，`asOfDate`）、SSGA xlsx | ETF 持股/權重 | 無公布限 | ✓ 月底可回溯 | ✓ |
| Wikipedia 表 | S&P 500/400、NDX、SOX 現值 + 變動 | 需 UA | 部分 | ✓ |
| USAspending v2 / senate-stock-watcher raw / Wikimedia Pageviews | 政府合約 / 參院交易 / 注意力 | 免 key | ✓ | ✓（更新不穩） |
| Macrotrends / Motley Fool / Capitol Trades 爬蟲 | — | **ToS 禁止** | — | 不做 |

**分層取數預算**（每輪 ≤ 20 分）：
- 每 15 分：只價量（現行不變）。
- 每日一次：watchlist ≤ 60 檔 yfinance 預估快照（1–2 秒間隔）→ 追加寫入 PIT 帳本；Finnhub recommendation / earnings calendar / insider-sentiment（< 200 次）；Alpha Vantage 25 次配給 3 天內財報者。
- 每週一次：宇宙刷新（Wikipedia + iShares/SSGA CSV）→ `yf.screen` 3–8 次 → 只對候選抓 `.info`（≤ 100 檔，限額輪替）；Finnhub financials-reported 更新有新 10-Q/10-K 者。
- 每月/季：成分股歷史 CSV 同步；SEC FSDS（若代理可用）。

**三個最值得先接**：Finnhub 三支免費端點（零新 secret）；yfinance 修正動能每日快照帳本；`yf.screen` + 成分歷史 CSV + iShares CSV 組成的可回測宇宙層。

---

## 5. 專業方法論清單（摘要；完整定義、公式、來源見 §10 代理報告）

**估值方法組合（football field）**：DCF-FCFF 三段 + 期中折現 + **終值 ROIC 一致性（再投資率 = g/ROIC_TV）**；反向 DCF（隱含成長 / 隱含 CAP 年數）；同業倍數 + **自身歷史 percentile**；PEG；growth-adjusted EV/S 與 Rule of 40（軟體）；SOTP（需 LLM 讀分部）；**RIM（金融股，現行對金融股套 DCF 是錯的）**；DDM/H-model（公用事業/REIT）。最終公允價**不是平均**，依產業指定主方法權重。

**品質與會計風險**：ROIC−WACC 價差與 4 年斜率；Piotroski F；Altman Z / Z''；Beneish M；Sloan 應計；FCF/NI 三年中位；SBC 占營收與稀釋率（「FCF − SBC」另列）；回購溢價與商譽。

**前瞻與催化劑**：共識修正動能（Zacks 三因子可算：Agreement / Magnitude / Surprise）；指引 vs 共識差距（需萃取）；PEAD/SUE；財報事件研究；RPO/backlog/book-to-bill（SEC tag `RevenueRemainingPerformanceObligation`）。

**情境與機率**：bear 必須是「論點錯了的世界」而非 base −10%；25/50/25 加權；蒙地卡羅給「市價位於分布第幾百分位」；MoS 門檻分層（高品質 25% / 一般 35% / 低品質 50%）；區間位置 `pos=(P−V_bear)/(V_bull−V_bear)` 與 upside/downside ≥ 3:1。

**觀點到部位**：Black-Litterman（絕對 view = (V/P)^(1/T)−1，Idzorek 信心 → Ω）；Kelly 由情境推 p、b 並強制 ¼；風險預算 × conviction 傾斜 ∈ [0.5,1.5]；單檔硬上限 10%、主題 25%；門檻式再平衡 + 成本閘。

**監測與證偽**：thesis KPI 數值化；pre-mortem；失效價 ≠ 停損價；**論點 vs 價格四象限**（論點對價格錯 = 加碼候選但先跑會計檢查；論點錯價格對 = 賣出，別被獲利留住）；時間止損。

**治理**：估值版本 + 變動歸因（成長/利潤率/WACC/g/股數）；假設變更日誌（無理由拒存）；九條審核清單（balance、TV/EV 50–75%、隱含終端倍數 ≤ 同業 P75、g ≤ min(rf, 4%)、終值 ROIC 收斂、淨負債含租賃/少數股權、稅率兩段、反向檢查、情境寬度 ≥ 1.8×）——**審核未過 verdict 強制 review、Bot 不推播買訊**。

**三個業餘錯誤與防呆**：終值吞掉模型；單點公允價 + 對稱情境 + 事後上修；估值/品質/催化劑混成一個分數且金融股硬套 DCF（現行 `health_score` 把 P/E 與 ROE 加總就是這型）。

---

## 6. 差異化定位：「更完整、更專業」的十項組合

| # | 項目 | 開源現況 | 我們的做法 |
|---|---|---|---|
| 1 | 專業級 WACC | Toolkit 用基準實際報酬當 Rm、帳面債權重 | rf 即時、ERP 常數可設、Blume β + 同業 bottom-up β、Rd 由利息保障倍數→合成信評→違約利差、市值權重 |
| 2 | 多法區間 football field | 各做一法 | DCF/反向/倍數 percentile/RIM/DDM 依產業路由與權重，輸出 low/base/high |
| 3 | 反向 DCF 三向求解 | 1★ 玩具 | 內建於同一引擎，附成長融資恆等式約束 |
| 4 | 蒙地卡羅 + 三情境 + 敏感度 | 各做其一 | 三者一起出，給「市價百分位」與「低估機率」 |
| 5 | 品質/舞弊/破產三套評分 | Toolkit 有公式無資料整合 | 在 yfinance 4 年 + Finnhub as-reported 上跑通，含缺值策略 |
| 6 | ROIC 成長融資恆等式健檢 | 零件散落 | 「共識 5 年成長 > ROIC 能撐的成長」標紅，直接餵 DCF |
| 7 | 自建修正動能時間序列 | 全網無免費實作 | 每日快照帳本，六個月後有自有歷史可回測 |
| 8 | 指引/KPI 萃取 JSON + 對帳 | 空白 | LLM 只定位轉錄、evidence 回對、數字 regex 回對、XBRL 對帳、revision 由程式判定 |
| 9 | Black-Litterman 觀點整合 | 無「公允價當 view」範例 | view = (V/P)^(1/T)−1、信心由情境寬度與品質推、接 `rebalance.py` |
| 10 | 治理：審核清單 + 版本歷史 + walk-forward 門檻 | 無 | 未過審不推播；估值層預設關閉、holdout 通過才接引擎 |

---

## 7. 選股池（Universe）設計 v2

### 7.1 四層結構

```
Layer 0  Benchmark      SPY / QQQ / 等權 watchlist B&H（永遠計算，不交易）
Layer 1  Core watchlist 現有 18 檔 ∪ 持倉（持倉永遠保留：出場只走價格機制）
Layer 2  Theme          stock_db 主題（AI 光學/連接/化合物半導體/HBM/電力散熱 … 約 60 檔美股）
                        + ETF 持股鏡像（SMH/SOXX/XLK/IWB 前 N 大）作為主題的客觀版
Layer 3  Broad          yf.screen：美股、市值 ≥ 20 億、3 月均量 ≥ 100 萬股、價 ≥ $5 → 500–750 檔
```

### 7.2 三段管線（月頻重建，與 15 分鐘 cron 解耦）

| 段 | 動作 | 呼叫 | 產出 |
|---|---|---|---|
| Stage 1 Broad | `yf.screen(EquityQuery)` 三次翻頁 | 3 | 500–750 檔 + marketCap/avgVol/sector |
| Stage 2 Screen | 在 screener 加品質條件（ROE > 8、流動比 > 1、D/E < 200、淨利率 > 0、排除金融/REIT 另路由）→ `yf.download` 一批算 12-1 動能、20 日 ADV、距 52 週高 | 1 | 150–250 檔 → 動能前 60 進「進場候選池」 |
| Stage 3 Rank | 對 60 檔抓 `.info` + 預估快照 + 自動模型 → 產業內 rank 的價值分數、MoS、修正動能、品質 | 60–120（限額 6/輪分攤，約一個交易日） | 候選池排序 + 每檔 `val_ctx` |

**規則**：Universe 只決定「誰能被引擎看到」與「配多少」；進場仍由技術評分過門檻、出場仍由價格機制。**持倉不因掉出宇宙而被賣**。

### 7.3 Point-in-time 快照帳本

- 每次重建把 Stage 1–3 結果追加寫入 `state["universe_hist"]`（或 repo 內 `data/universe/YYYY-MM.parquet`，明文可接受——皆為公開資料衍生）。
- 欄位規則照 Lean/Zipline：季報數字 `period_end + 45 天` 可見、`.info` 類欄位 T+1 可見、預估快照以抓取日為 `timestamp`。
- 成分歷史：S&P 500（fja05680，1996 起）、NDX（n100tickers，2015 起）、其他用 iShares 月底持股近似 → `engine_backtest` 改吃「當月成分」，消除倖存者偏差（這一步**與估值層無關也該做**）。
- 累積 ≥ 12 個月後，估值因子才進 walk-forward。

### 7.4 因子驗證門檻（alphalens 標準）

MoS、修正動能、品質各自：21/63 日 Rank IC 與 ICIR、三分位報酬 spread、因子自相關（MoS > 0.9 才算穩定低換手）、產業去均值。**進配置門檻：21 日 Rank IC > 0.03 且 ICIR > 0.3**；宇宙小時只求方向一致、不求顯著。

### 7.5 API 預算

現有 15 分鐘 cron 不變（每輪 6–8 次）；月度重建新增約 130–250 次，攤到約 2,000 輪 cron 每輪 < 0.2 次；Stage 3 走 `alpha_overlay` 既有的限額輪替與 Finnhub 備援，不在單輪同步抓完。

---

## 8. 修訂後的階段計畫（取代 VALUATION_PLAN §4）

| 階段 | 交付 | 新增/修訂 | 估時 |
|---|---|---|---|
| **P0 資料地基 + 快照帳本** | `fin_data.py`（yfinance 主、Finnhub as-reported 備援、每筆帶 `period_end/filed/timestamp`）；`estimates_ledger`（每日快照 eps_trend/eps_revisions/共識/目標價/recommendation）；`universe.py` 純邏輯 + Stage 1–2 抓取層；成分歷史 CSV 同步 | 快照帳本、宇宙層、PIT 欄位規則 | 2 sessions |
| **P1 公司模型引擎** | `company_model.py`：驅動推導（VRT 啟發式）、三段 FCFF、專業級 WACC、三情境 + MC + 敏感度、反向 DCF 三向求解、ROIC 恆等式健檢、產業路由（RIM/DDM/EV-S）；`quality.py`：Piotroski/Altman/Beneish/Sloan/SBC/稀釋；`/model` 指令；回歸測試對齊 VRT Excel ±3% | 品質模組、反向三向、產業路由、審核清單 | 2 sessions |
| **P2 訊號、驗證、網頁** | `valuation_signal.py`（V、MoS、區間位置、修正動能、品質旗標）；`factor_eval.py`（alphalens 式 IC/分位/自相關，純 pandas）；估值歷史 + 變動歸因 + 假設日誌；網頁「公司模型」頁（驅動滑桿、football field、MC 分布、公允價歷史、**Excel 匯出/匯入橋**）；晨報一行 | factor_eval、治理欄位 | 2 sessions |
| **P3 觀點到部位** | BL：`portfolio_opt.black_litterman(views, confidences)` 接 `rebalance.py`；引擎接口 `val_ctx`（部位乘數 0.5–1.25、pyramid 閘、候選傾斜 λ ≤ 0.1）；`engine_backtest` 改吃當月成分 + PIT 快照 + `val_enabled` 維度；**holdout 通過才開** | BL、成分歷史回測 | 2 sessions |
| **P4 指引與 KPI 萃取** | `guidance.py`：8-K EX-99.1 / 逐字稿 → 封閉列舉 metric 的 JSON，evidence 回對、數字回對、XBRL 對帳、revision 程式判定；thesis KPI 自動比對；論點 vs 價格四象限；指引修訂先進 `falsifier` 證偽再進 `alpha_overlay` | 全新 | 2 sessions |
| **P5 宇宙擴大** | Stage 3 對主題層 + 寬宇宙跑自動模型 → `/screen value|growth|quality` 候選清單；ETF 持股鏡像主題 | — | 1 session |
| **P6 治理與復盤** | 估值層月報：命中率、公允價變動歸因、審核未過清單；六個月後首次估值因子 walk-forward 報告 | — | 持續 |

**不變的鐵律**：估值層不觸發進場、不否決價格停損；所有影響量有界；預設關閉；每階段 DoD 照 CLAUDE.md；所有輸出「非投資建議」。

---

## 9. 拍板決策（更新版）

| # | 決策 | 建議 |
|---|---|---|
| 1 | 估值層權限：只調部位/加碼/排序 vs 也能觸發進場 | **只調部位**（不變） |
| 2 | 資料源：yfinance 共識自動 + Finnhub 三支免費端點 + 自建快照；SEC 只作回測資料集 | **是**；Alpha Vantage 25 次/日作可選加值（需新 secret，四處同步） |
| 3 | Excel 匯出/匯入橋 | **做** |
| 4 | 順序：P0→P1→P2 只顯示觀察，再 P3 | **是**；但 **P0 的成分歷史 + PIT 快照現在就開始累積**，晚一天少一天資料 |
| 5 | 宇宙：四層結構，Broad 層 500–750 檔月頻重建 | **是**；Core/Theme 先進 Stage 3，Broad 先只做 Stage 1–2 與快照 |
| 6 | 新增：是否引入 FinanceToolkit 作依賴 | **否**（使用者 2026-09-08 同意）——公式自己用 pandas 寫進 `quality.py` |
| 7 | 新增：LLM 指引萃取用哪個模型 | **便宜模型抽 + 程式驗證**（使用者 2026-09-08 拍板）；升級複核只在低信心/對帳不符時 |

---

## 11. 安全性與資料洩漏規範（2026-09-08 使用者要求納入；每階段 DoD 必查）

「資料洩漏」在本專案有三種意思，三種都要防：

### 11.1 機密與隱私外洩（公開 repo）

| 風險 | 規則 | 落實點 |
|---|---|---|
| **Actions 日誌公開**（本次實查：`Command: /mirror init 932 DRAM:23:…` 曾整行印進日誌） | 不 print 指令參數、持倉、股數、淨值、論點、chat id、API 回應原文；預設只印摘要，`RBS_VERBOSE_LOGS=1` 僅限本地 | 已修：`_log_cmd`/`_log_lines`（PITFALLS D14、CLAUDE.md 鐵律 1） |
| 使用者的模型假設、公允價值、MoS、部位觀點 = 策略本體 | `state["models"]`、`val_hist`、`eng_opt`、BL views 全進 `SENSITIVE_KEYS`；宇宙快照（公開資料衍生）可明文 | P0/P2 |
| LLM prompt 帶出私資料 | 送 LLM 的只有公開文件文字 + 代碼；**絕不**帶帳戶淨值、持倉、鏡像帳、API key；委員會/萃取的 prompt 進入前過一層 `redact()` 白名單 | P4 |
| 新 API key（Alpha Vantage 可選） | 四處同步規則（app.py `_os_boot`、workflow env、GITHUB_ACTIONS.md、README）；不進任何 commit | P0 |
| 第三方資料 ToS | Macrotrends / Motley Fool / Capitol Trades 等明文禁爬者不做；Wayback 只作離線研究、不進 cron | 全程 |
| SEC 代理（若試 Cloudflare Worker） | Worker 必須帶共享 token 驗證，否則就是開放代理；視為可缺席源、首次逾時即熔斷；**另案拍板才做** | 不在 P0–P6 內 |
| 供應鏈 | 不引入 openbb / FinanceToolkit / stockdex；新依賴（若有）釘版 + 進 `signal_scan.yml` 的 pip 釘版行；CI `permissions: read` 不變 | 全程 |
| 公開 repo 的自架 runner | 永遠不用（fork PR 可在你機器執行任意碼） | — |

### 11.2 LLM 特有：提示注入與幻覺

- 財報新聞稿/逐字稿是**不受信任輸入**：LLM 只做「定位 + 轉錄」到封閉列舉 schema；任何指令性文字（「忽略前面規則」）因輸出被 schema 與 evidence 回對雙重約束而無效；HTML 去標籤、截長度、不執行任何連結。
- 每個數字必須能從 `quote` 的 `char_start:char_end` 精確回對原文，數字用 regex 重新解析，midpoint/revision 由程式計算；對帳不符標 `xbrl_mismatch`；解析失敗 → 重試一次 → 棄權（寫入 `abstained`），**不預設中性值**。
- 便宜模型抽取（使用者拍板）：Haiku 級模型抽 → 程式驗證為主；只有 `confidence < 0.6` 或對帳不符的欄位才升級到較強模型複核，成本可控。
- 委員會與萃取的輸出進決策前，必須通過既有 `hard_risk_check` 與 `falsifier` 證偽；PM 裁決不得引用 JSON 外的數字。

### 11.3 回測資料洩漏（look-ahead / 前視）

- **PIT 雙日期欄**：所有基本面與預估資料表帶 `period_end`（指涉期）與 `available_at`（可得知日 = filed/抓取日 + 1 交易日）；回測只讀 `available_at ≤ as_of`。yfinance 三表無 `filed` → 一律 `period_end + 45 天`；季報以 Finnhub as-reported 的 accession/filed 為準。
- **重編不覆寫**：同一 (ticker, concept, period) 的修正值另存 delta，回測用首次揭露值（Zipline deltas / Qlib PIT 原則）。
- **快照帳本只往前寫**：預估修正、宇宙成分、估值歷史都是 append-only；回測重放快取的萃取結果，不重抓（避免今天的頁面回填昨天）。
- **成分股用當時名單**：`engine_backtest` 改吃成分期間表；watchlist 是事後挑的這件事寫進報告限制欄，不假裝解決。
- **LLM 參數化前視**：任何含 LLM 產物（指引 JSON、委員會裁決）的回測，只在模型知識截止日**之後**的區間評估，或只用確定性欄位；報告標明模型與截止日。
- **校準權重前視**（VALUATION_PLAN 已揭露）：`calibration` 用現值屬輕微前視，回測報告固定附註。
- **多重測試**：所有估值因子/參數搜索沿用 DSR 帳本；試過的組合數如實計入。

### 11.4 對使用者三個追問的回覆紀錄

1. **歷史共識來源**：見 §4 補充（查證代理結果）——結論先講：逐日共識歷史沒有免費合規源；「財報日當下的共識」有長歷史免費源可回填，足以做 SUE/PEAD 與冷啟動。
2. **FinanceToolkit「綁 FMP」的意思**：它預設向 Financial Modeling Prep 取數（需 key、免費層 250 次/日、美股、5 年），Yahoo 模式雖可用但仍拖進 pandas 3 / scikit-learn 依賴。**自己用 pandas 算完全可行且更好**：Piotroski/Altman/Beneish/Sloan/ROIC/DCF 都是三表列的四則運算，不需要 scikit-learn；真正的工作量在列名對映與缺值策略，不在數學。決策：移植公式、不裝套件。
3. **便宜模型抽指引**：採納。Haiku 抽 + 確定性驗證為主，升級複核只在低信心/對帳不符時觸發。

## 10. 來源（代理報告摘錄；完整清單見各代理輸出）

- 開源函式庫：FinanceToolkit / FinanceDatabase / OpenBB / edgartools / SimFin / implied-expectations / reverse-dcf topic / yfinance `scrapers/analysis.py` / HKUDS Vibe-Trading earnings-revision skill。
- 交易框架：Lean `FundamentalUniverseSelectionModel.py`、`QC500UniverseSelectionModel.py`、`Fundamental.cs`（T+1 註解）、`ValuationRatios.cs`；zipline-reloaded `factor.py`、`blaze/core.py`（asof_date/timestamp）；Qlib `docs/advanced/PIT.rst`、`data_collector/index.py`、`RollingGen`；alphalens-reloaded `performance.py`；PyPortfolioOpt `BlackLitterman.rst`；skfolio `_black_litterman.py`；FinRL arXiv 2209.05559。
- LLM 代理：ai-hedge-fund VISION.md / v2 `hedge_fund/` / legacy `src/agents/{valuation,warren_buffett,aswath_damodaran,risk_manager}.py`；TradingAgents `schemas.py`、`utils/rating.py`、arXiv 2412.20138；FinRobot `functional/analyzer.py`；anthropics/financial-services README；arXiv 2605.24564、2512.23847、2602.14233、2510.02209、2604.23588、2510.03195。
- 資料源：Finnhub docs/pricing、Alpha Vantage support、FMP FAQ/changelog、EODHD pricing、SEC EDGAR APIs/FSDS/efts、yfinance discussions #1655 #2129 #2159 issues #2480、fja05680/sp500、n100tickers、yfiua/index-constituents、etf-scraper、talsan/ishares、USAspending API、pytrends #602、Wikimedia Analytics API、Macrotrends terms。
- 方法論：Damodaran termvalue.pdf 與 Myth 5.3；Footnotes Analyst；Expectations Investing；Idzorek 2005；Morgan Stanley「Measuring the Moat」「ROIC and the Investment Process」；Piotroski 2000；Sloan 1996 / NBER w13525；Bernard & Thomas 1989；Zacks Rank Guide；GMT Research / StableBread（Beneish）；CreditGuru / WSP（Altman Z''）；Greenblatt sizing；Kelly 不確定性模擬；McKinsey DCF 模型指南。
