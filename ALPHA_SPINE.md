# ALPHA_SPINE.md — 橫斷面 Alpha 脊椎 / Meta-labeling / 因子研發迴圈（A/B/C）規劃與落地

> 2026-09-19 拍板。目標：把「18 檔擇時」換成「400 檔排名 + 每筆部位由歷史勝率決定」，
> 讓估值/品質/預估修正/氣象台/反思帳本全部變成**同一把尺**（IC、樣本外 Sharpe、DSR）下的輸入。
> 對照專案：Microsoft Qlib（特徵面板 → 排序模型 → TopK 持倉 → 回測）、Lean Algorithm Framework
> （Universe Selection → Alpha → Portfolio → Risk → Execution；CompositeAlphaModel）、
> López de Prado meta-labeling / purged CV、微軟 RD-Agent（自動因子研發）。

## 0. 一張圖

```
夜間工作流 alpha_nightly.yml（每個交易日收盤後一次，≤120 分）
  universe 快照 broad(≈415) ∪ watchlist
     │ yf.download 3y OHLCV（一次批次）
     ▼
  ┌─ A. alpha_spine ───────────────────────────────────────────────┐
  │ 價格因子（向量化）：mom_12_1 / rev_1m / ext_atr / vol_60       │
  │ 基本面因子（PIT）：quality（data/fin）/ rev（預估帳本）       │
  │ C 段核准因子（state["factor_lab"]["approved"] 的 DSL）        │
  │ factor_eval：週頻 Rank IC、ICIR、Newey-West t → 閘門           │
  │ ICIR 加權合成 → 今日排名 → data/alpha/rank.json                │
  └────────────────────────────────────────────────────────────────┘
     │ 同一份面板 + indicators.composite_series（向量化＝引擎評分）
     ▼
  ┌─ B. meta_label ────────────────────────────────────────────────┐
  │ 樣本：每檔每日「引擎進場條件成立」的時點（去重）               │
  │ 標籤：用引擎規則模擬單筆出場（停損/追蹤/收緊/時間柵欄）→ R    │
  │ 特徵：ext_atr、ret_5d、score、vol、mom、rev_1m、regime、廣度、  │
  │       quality(PIT)、rev(PIT)（訓練與線上同一函數）             │
  │ 模型：numpy 邏輯迴歸（L2）；purged walk-forward 4 折 + embargo │
  │ 閘門：OOS AUC>0.55、勝率尺寸化 Sharpe > 等額、n_oos≥300        │
  │ → data/alpha/meta.json（係數/標準化/校準/OOS 指標/閘門）        │
  └────────────────────────────────────────────────────────────────┘
     │ 另：基本面覆蓋輪替（每晚 12 檔 fin_data + 12 檔預估快照）
     ▼  commit data/alpha/*、data/fin/*、data/alpha/estimates_universe.json

15 分鐘 cron scan_signals.py（不變的部分略）
  run_autotrade：
    候選池 = watchlist ∪ 鏡像持倉 ∪ [alpha_pool_enabled 且 rank 閘門過 → rank.json 前 k]
    scored 每列附 ext_atr/ret_5d（已有）
    [meta_enabled 且 meta.json 閘門過] → 同一特徵函數 → p(勝) → meta_mult ∈ {0, 0.5…1.25}
    trade_engine：risk 預算 × val_mult × neutral_mult × meta_mult；meta_mult=0 → 跳過並說明
  /alpha [rank|factors|meta]、/factor test|add|list（C 段）
```

## 1. A 段：橫斷面 Alpha 脊椎（`alpha_spine.py`）

### 1.1 因子定義（全部只用「當日以前含當日」資料）

| 因子 | 定義 | 方向 | 來源 |
|---|---|---|---|
| `mom_12_1` | close[t−21] / close[t−252] − 1 | 高好 | 價格 |
| `rev_1m` | −(close[t] / close[t−21] − 1) | 高好（近月跌多者反彈；反思帳本實證短期反轉） | 價格 |
| `ext_atr` | −((close − MA20) / ATR14) | 高好（延伸越少越好；與引擎追高濾網同義） | 價格 |
| `vol_60` | −(60 日日報酬標準差) | 高好（低波動溢價） | 價格 |
| `quality` | `quality.quality_summary(pit_view(store, as_of=t))["score"]` | 高好 | data/fin（PIT `available_at ≤ t`） |
| `rev` | `estimates_ledger.revision_momentum` 以 ≤ t 的帳本列重建 | 高好 | estimates_ledger.json + data/alpha/estimates_universe.json |
| C 段核准 | DSL 表達式（見 §3） | 依核准時 IC 符號 | 價格面板 |

方向固定寫死；**不允許**依 IC 符號自動翻轉（翻轉＝資料探勘）。

### 1.2 評估與合成

- 快照日：每 5 個交易日一個（減少重疊），回看 `ALPHA_LOOKBACK_DAYS=300`。
- 每因子 `factor_eval.evaluate(factor_by_date, closes, horizons=(21,))`；閘門 `passes_gate`
  （IC>0.03、ICIR>0.3、n_eff≥12、NW t≥2）。
- 權重 = 通過者的 ICIR 正部歸一化；未通過權重 0。**沒有任何因子通過 → `gate_passed=false`，
  引擎不用候選池**（誠實退回 watchlist）。
- 合成分 = Σ w_f × 橫截面 rank-pct（0..1 中心化）；缺值因子不計入該檔分母（部分成分降信心 `conf`）。
- 也回報合成分自身的 IC 與三分位價差（描述性；權重由過去 IC 決定，仍有輕微樣本內偏差，文件標明）。

### 1.3 輸出 `data/alpha/rank.json`（公開資料衍生、明文）

```
{as_of, available_at, n_universe, lookback_days, horizon,
 factors: {name: {ic, icir, t_nw, n_eff, hit, spread, pass, reason, weight, coverage}},
 composite: {ic, icir, spread},
 gate_passed, top: [{t, score, conf, rank, f: {name: value}}]（前 50）,
 market: {breadth_pct, spy_regime, as_of}}
```

### 1.4 計算預算（GitHub Actions ubuntu-latest）

| 步驟 | 估計 |
|---|---|
| yf.download 415 檔 × 3y | 1–3 分 |
| 價格因子（pandas 向量化） | < 30 秒 |
| 基本面 PIT 重建（60 快照日 × 覆蓋檔數） | < 1 分 |
| IC 評估 6–10 因子 | < 1 分 |
| 引擎評分序列（B 段共用）`composite_series` 向量化 | 415 檔 × 504 日 ≈ 1–2 分（逐日切片版要 24 分） |
| 基本面覆蓋輪替 24 檔 | 2–4 分（含 sleep） |

## 2. B 段：交易層 meta-labeling（`meta_label.py`）

### 2.1 樣本與標籤

- 主模型 = 現行引擎進場條件：`score ≥ buy_threshold`、非延伸端（ext>2 且 5 日>6% 才算）、非 FOMC 靜默日。
- 去重：同一檔連續成立只取**首日**，之後每 10 個交易日再取一個（避免同一段行情重複計數）。
- 出場模擬 `simulate_exit(closes, i, rps, cfg)` 用引擎規則：硬停損 entry−1.0×rps（rps=1.5×ATR14）、
  +1R 後追蹤 8%（地板保本）、+2R 後收緊 5%、時間柵欄 45 日；成交＝次日開盤（無同棒前視）、
  單邊 0.05% 成本。標籤 `r_mult`、`win = r_mult > 0`。分批鎖利不模擬（只影響尺寸不影響方向）。
- 出場模擬也含「訊號轉弱且獲利中了結」與「死錢釋放」（用評分序列）。**未模擬**：分批鎖利（只影響尺寸）、
  曝險非 ACTIVE 時的收緊追蹤、盤中 15 分鐘價觸發（模擬只看收盤）、使用者 `/set eng_*` 偏離預設的參數
  （夜間讀不到加密 thresholds）。p(勝) 因此是「引擎預設出場政策」下的勝率。
- 與 `/engtest` 重放的差異：這裡**不受**最多 10 檔/現金限制（要的是每個訊號的條件勝率，不是投組路徑）。
- 行情視窗 3 年（2y 下 12-1 動能的可用快照不足、有效期數 <12 永遠過不了閘門）。

### 2.2 特徵（訓練＝線上同一函數 `meta_label.features(row, ctx)`）

`ext_atr, ret_5d, score, vol_60, mom_12_1, rev_1m, regime_on, regime_off, breadth_pct, quality(+缺值旗標), rev(+缺值旗標)`。
regime 用 SPY/MA50 三態（線上讀 rank.json 的 `market`，1 日延遲，與訓練一致；**不用**氣象台，因為氣象台沒有歷史）。
反思命中率不進特徵（無歷史可訓練）。

### 2.3 模型與驗證

- 兩個候選模型走**同一套** purged walk-forward：numpy 邏輯迴歸（標準化、L2、牛頓法）與 LightGBM
  （夜間工作流有裝；num_leaves 7、學習率 0.05、每折用訓練段最後 15% 做 early stopping、最終棵數取各折中位數）。
  擇優：先看閘門（含尺寸化是否改善 Sharpe），再比 OOS log-loss，同分取邏輯迴歸；兩者 OOS 都存進 `candidates` 供 `/alpha meta` 對照。
  GBM 以精簡節點表存 JSON（150 棵約 25 KB、上限 400 棵約 63 KB），線上用 `meta_label.predict_gbm` 純 Python 樹遍歷推論（自測與 lightgbm
  原生預測逐位一致），15 分鐘 cron 不加依賴。
- 特徵 18 個：延伸度、5 日漲幅、評分、波動、12-1 動能、1 月反轉、SPY 三態 one-hot、廣度、品質(+缺值旗標)、
  修正(+缺值旗標)、動能缺值旗標、RSI14、距 52 週高、ATR 占價比、20 日量比。
- **AUC 的期望值**：交易勝負標籤的可預測度很低，業界 0.55–0.60 是常態、0.65 已屬優異（Qlib 基準最好的
  橫斷面模型 IC 也只有 0.045）。閘門看的是 AUC≥0.55 **加上**尺寸化是否真的改善 Sharpe；AUC 0.7+ 通常代表洩漏。
- purged walk-forward：依時間 4 折，訓練段結尾與測試段開頭之間 **embargo 46 個交易日**（＝最長標籤視窗 45 + 1）。
- OOS 指標：AUC（rank 法）、log-loss vs 基準率、**尺寸化測試**：門檻相對基準勝率 b——size(p)=0（p<b−8pp）、
  線性 0.5→1.25（b−8pp..b+12pp）、1.25（p≥b+12pp）；比較 OOS 交易 R 序列的 mean/std（等額 vs 尺寸化）。
- 閘門：`auc ≥ 0.55 且 sized_sharpe > equal_sharpe 且 n_oos ≥ 300 且 跳過率 ≤ 80%`（靠什麼都不做取勝不算）。
  未過 → `gate_passed=false`，線上不用。
- 最終模型用全樣本重訓；輸出 `data/alpha/meta.json`：
  `{as_of, features, mean, std, coef, intercept, base_rate, n, oos: {...}, gate_passed, calib: [(p_bin, hit)]}`。

### 2.4 線上接法

`run_autotrade`：`meta_enabled`（預設關）且 `meta.json.gate_passed` → 每個候選算特徵 → p → `meta_mult=size(p)`
附在 scored 列；`trade_engine` 把 `meta_mult` 夾 [0, 1.25] 乘進風險預算（0 → 跳過並註明勝率）。
出場不受影響。鏡像帳共用同一 scored。

## 3. C 段：因子研發迴圈（`factor_lab.py`）

- DSL（白名單 `ast`）：原語 `ret(n) mom(a,b) vol(n) ma_dist(n) ext(n) rsi(n) volratio(n) hi_dist(n) lo_dist(n)`，
  四則運算與常數；例 `mom(126,21) - 0.5*vol(60)`。
- `/factor test <expr>`：抓 universe 2y 價格 → 週頻快照 → `factor_eval.evaluate` → DSR（`falsifier.deflated_sharpe`，
  以 ICIR 為 SR、n_eff 為 T、帳本筆數為 N、歷史 ICIR 為 trial 分佈）→ 記入 `state["factor_lab"]["trials"]`（明文，公式非敏感）。
- `/factor add <name> <expr>`：人工核准 → `approved`；夜間 A 段自動納入（仍過 IC 閘門）。`/factor list`、`/factor drop <name>`。
- 委員會自動提案（RD-Agent 式）留待 A/B 穩定後：提案 → test → 人工 add。

## 4. 旗標與閘門（全部預設關，資料先累積）

| 旗標 | 預設 | 作用 |
|---|---|---|
| `alpha_pool_enabled` | off | 候選池加入 rank.json 前 `alpha_pool_top`（20）名 |
| `meta_enabled` | off | 部位乘 meta_mult |
| （自動）`rank.json.gate_passed` | — | 沒有因子過 IC 閘門 → 候選池不用 |
| （自動）`meta.json.gate_passed` | — | OOS 未勝過等額 → 不用 |

啟用順序：夜間跑 2 週 → `/alpha factors` 看哪些因子過閘 → `/alpha meta` 看 OOS → 先開 `alpha_pool_enabled`
觀察 2 週 → 再開 `meta_enabled`。

### 4.1 多線平行帳（`lanes.py`，`/lanes`；`lanes_enabled` 預設開）

旗標關著的時候就能看「如果開了會怎樣」：每輪 cron 用同一台引擎、同一輪 scored/config/regime，各跑一本
10 萬起始的虛擬帳——`base`（現行 watchlist）、`pool`（＋候選池）、`pool_meta`（＋候選池＋meta 部位）——
掃描價成交含 0.05% 成本、每日淨值一點、殭屍倉防護同鏡像帳；`/lanes` 並排列出報酬／回撤／持倉／成交，
附 SPY 與真帳同期。閘門沒過時車道會標註「＝現行」「＝候選池」（此時該車道與上一條相同）。
車道持倉即使跌出前 k 名也會補掃報價（quiet）；候選池列同樣走 alpha overlay 與 FOMC 靜默窗，旗標開關前後語意一致。
state["lanes"] 加密（虛擬持倉與真帳高度相關）；不推播每筆單。`/lanes reset` 重新起算。

## 5. PIT 與洩漏保證

- 價格因子：只用 `iloc[:t+1]`；前瞻報酬從 t 之後第 1 個交易日起算（`factor_eval.forward_returns`）。
- 基本面：`fin_data.pit_view(store, as_of=t)`（`available_at` first-seen 不覆寫）；預估帳本列有日期。
- 標籤：進場＝訊號日次日開盤；出場只看之後的 K 棒。
- 驗證：purged + embargo；閘門看 OOS。
- 安全：rank/meta/因子公式皆為公開資料衍生（明文）；候選池代碼公開（≠ 持倉）；夜間工作流不 print 持倉；線上候選池與「持有但不在 watchlist」的補掃走 `scan(quiet=True)`，代碼不進 Actions 日誌。

## 6. 已知限制（誠實邊界）

- 基本面因子覆蓋從 18 檔起，每晚 +24 檔，約 4–6 週覆蓋 415 檔；覆蓋不足時 IC 算不出、權重 0（自動）。
- 訓練用 MA50 三態 regime，線上氣象台仍控制曝險三態（兩者職責不同：一個是特徵、一個是保險絲）。
- 邏輯迴歸不是 Qlib 的 LightGBM：特徵 12 個、樣本數千筆，線性模型更穩；等樣本 >2 萬再評估樹模型。
- 兩年美股多頭樣本：OOS 閘門能擋過擬合，擋不住「regime 沒見過」。
- 標籤的出場規則也用 `LABEL_CFG` 預設（追蹤 8%／+2R 收緊 5%／分批不模擬）；`/engtest opt apply` 改了 `eng_*` 之後，meta 標籤與實際引擎出場會脫鉤，需同步調整 `LABEL_CFG`。
- 訓練的「訊號成立」用 `LABEL_CFG` 預設門檻（0.5／延伸 2 ATR／5 日 6%），夜間讀不到加密的 thresholds；使用者 `/set` 偏離預設時線上訊號分佈與訓練略有不同。
- `score` 特徵：訓練用 `composite_series` 原始值（無校準權重）、線上用 overlay 前的 `raw_score`（watchlist 帶校準權重、候選池不帶）——已對齊到 overlay 前，校準差異屬已知小偏差。

## 7. 進度看板

| 項目 | 狀態 |
|---|---|
| `indicators.composite_series` 向量化（與逐日評分逐位相等） | ✅ 2026-09-19（自測 300 日 × 3 組設定最大差 0） |
| `alpha_spine.py` + `data/alpha/rank.json` | ✅ 2026-09-19 |
| `meta_label.py` + `data/alpha/meta.json` | ✅ 2026-09-19（合成植入效應 OOS AUC 0.63、閘門通過；打亂標籤閘門不過） |
| `factor_lab.py` + `/factor` | ✅ 2026-09-19 |
| `alpha_nightly.py` + `.github/workflows/alpha_nightly.yml` | ✅ 2026-09-19（首晚產出後 `/alpha factors` 可看） |
| `/alpha`、候選池、meta_mult、旗標 | ✅ 2026-09-19（兩旗標預設關） |
| LightGBM 候選 + 純 Python 推論 + 4 個新特徵 | ✅ 2026-09-19 |
| `lanes.py` 多線平行帳 + `/lanes` | ✅ 2026-09-19（`lanes_enabled` 預設開） |
| 網頁頁「🧬 Alpha 脊椎」 | 待辦（先 Telegram） |
| 委員會自動提案因子 | 待辦（A/B 穩定後） |
