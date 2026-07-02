"""
app.py – RBS Finance Dashboard (Streamlit)

Local:  streamlit run app.py
Colab:  see colab_setup.py (handles pip, pyngrok, Drive path, tunnel)
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ── Colab / local path resolution ────────────────────────────────
# In Colab the library lives in Google Drive; locally it's next to app.py
_COLAB_BASE = Path("/content/drive/MyDrive/RBS_app")
_LOCAL_BASE = Path(__file__).parent

BASE_DIR = _COLAB_BASE if _COLAB_BASE.exists() else _LOCAL_BASE
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from rbs_lib import (
    apply_shocks,
    calculate_volatility,
    calculate_woe_iv,
    conditional_var,
    delta_normal_var,
    ewma_cov,
    historical_replay,
    historical_var,
    kupiec_pof_test,
    load_price_data,
    lw_cov,
    mc_portfolio_pnl,
    portfolio_var,
    rolling_portfolio_var,
    scenario_pnl_value,
    scorecard_transform,
    to_returns,
)

# ─────────────────────────── Page Config ────────────────────────────

st.set_page_config(
    page_title="RBS Finance Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────── Custom CSS ─────────────────────────────

st.markdown(
    """
    <style>
    /* ══ Custom HTML components only — native Streamlit handled by config.toml ══ */

    /* Metric cards */
    .metric-card {
        background: #1A1D27;
        border: 1px solid #2D3142;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 4px 0;
    }
    .metric-label {
        font-size: 0.78rem;
        color: #A8B0C0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #FAFAFA;
    }
    .metric-positive { color: #4CAF50; }
    .metric-negative { color: #F44336; }

    /* Section header */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #5BA4EF;
        border-left: 3px solid #1E88E5;
        padding-left: 10px;
        margin: 16px 0 10px 0;
    }

    /* Sidebar */
    div[data-testid="stSidebar"] {
        background-color: #0D1117;
    }

    /* Dataframe — only target the canvas/cell layer, not the whole page */
    [data-testid="stDataFrame"] canvas { filter: none; }
    [data-testid="stDataFrame"] thead tr th {
        background-color: #1E2235 !important;
    }

    /* Active tab indicator color */
    button[data-baseweb="tab"][aria-selected="true"] {
        border-bottom: 2px solid #1E88E5 !important;
    }

    /* Plotly SVG: only override text that sits on dark chart backgrounds */
    .js-plotly-plot .plotly .gtitle { fill: #FAFAFA !important; }
    .js-plotly-plot .plotly .xtick text,
    .js-plotly-plot .plotly .ytick text { fill: #E8EAF0 !important; }
    .js-plotly-plot .plotly .g-xtitle text,
    .js-plotly-plot .plotly .g-ytitle text { fill: #C8CAD4 !important; }
    .js-plotly-plot .plotly .legendtext { fill: #E8EAF0 !important; }
    .js-plotly-plot .plotly .colorbar text { fill: #E8EAF0 !important; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0D1117; }
    ::-webkit-scrollbar-thumb { background: #2D3142; border-radius: 3px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────── Helpers ────────────────────────────────

_AXIS = dict(
    color="#E8EAF0",
    tickfont=dict(color="#E8EAF0", size=11),
    title_font=dict(color="#C8CAD4", size=12),
    gridcolor="#2D3142",
    linecolor="#2D3142",
    zerolinecolor="#3D4255",
)

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0F1117",
    plot_bgcolor="#14172A",
    font=dict(family="Inter, sans-serif", size=12, color="#E8EAF0"),
    margin=dict(l=40, r=20, t=45, b=40),
    xaxis=_AXIS,
    yaxis=_AXIS,
    legend=dict(
        font=dict(color="#E8EAF0", size=11),
        bgcolor="#1A1D27",
        bordercolor="#2D3142",
        borderwidth=1,
    ),
    coloraxis=dict(
        colorbar=dict(
            tickfont=dict(color="#E8EAF0"),
            title_font=dict(color="#E8EAF0"),
        )
    ),
    title_font=dict(color="#FAFAFA", size=14),
    hoverlabel=dict(
        bgcolor="#1E2235",
        bordercolor="#3D4255",
        font=dict(color="#FAFAFA", size=12),
    ),
)


def metric_card(label: str, value: str, delta: str = "", positive: bool | None = None):
    color_cls = (
        "metric-positive"
        if positive is True
        else "metric-negative" if positive is False else ""
    )
    delta_html = f'<div class="metric-label {color_cls}">{delta}</div>' if delta else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title: str):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


# pandas 3.0 移除了 "M"/"H"/"Q"/"Y" 舊別名；resample 時轉成新別名
_RESAMPLE_ALIAS = {"M": "ME", "W": "W", "D": "D", "Q": "QE", "Y": "YE", "H": "h"}


def _annualization(index: pd.DatetimeIndex, freq: str | None) -> int:
    if freq == "M":
        return 12
    if freq == "W":
        return 52
    if len(index) < 3:
        return 252
    diffs = np.diff(index.values).astype("timedelta64[D]").astype(int)
    m = np.median(diffs)
    return 12 if m >= 25 else (52 if 5 <= m <= 9 else 252)


# ─────────────────────────── Sidebar Nav ────────────────────────────

with st.sidebar:
    st.image(
        "https://img.shields.io/badge/RBS-Finance%20Dashboard-1E88E5?style=for-the-badge",
        use_container_width=True,
    )
    st.markdown("---")
    # 依功能邏輯分組排序（相鄰即分組）：市場 → 選股研究 → 組合風險 → 交易監控 → 工具
    page = st.radio(
        "Navigation",
        [
            # 市場面
            "🏠 市場總覽",
            "📰 新聞情報",
            # 選股與研究
            "🔍 股票研究",
            "🏢 公司分析",
            "🏦 機構選股",
            # 組合與風險
            "📈 持倉分析",
            "⚠️ 風險管理",
            "🛠️ 交易工具",
            # 交易與監控
            "🚨 即時警報",
            "📉 模擬交易",
            # 工具
            "📦 匯出報告",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("RBS Finance Dashboard v4.0")


# ════════════════════════════════════════════════════════════════════
# PAGE: Portfolio Performance
# ════════════════════════════════════════════════════════════════════

def page_portfolio_performance():
    st.title("📈 持倉分析")
    st.caption("多資產組合分析：權益曲線、回撤、Sharpe、資訊比率、Beta")

    import yfinance as yf

    DEFAULT_HOLDINGS = pd.DataFrame(
        {
            "Ticker": ["AGG", "BBH", "BND", "IVV", "KRE", "MBB", "SHY", "SMH", "SOXX", "VFH", "VOO", "XLF", "XLU", "XLV", "XSD"],
            "Shares": [6, 4, 7, 1, 5, 6, 7, 11, 10, 5, 1, 13, 8, 4, 3],
        }
    )

    with st.sidebar:
        st.markdown("### Settings")
        start = st.date_input("Start Date", value=date(2024, 1, 1))
        freq = st.selectbox("Frequency", ["M", "W", "D"], index=0, format_func=lambda x: {"M": "Monthly", "W": "Weekly", "D": "Daily"}[x])
        benchmark = st.text_input("Benchmark Ticker", "SPY")
        rf_choice = st.selectbox("Risk-Free Rate", ["^IRX (13W T-Bill)", "Constant"])
        rf_const = st.number_input("Constant RF (annual)", value=0.03, step=0.005, format="%.3f")
        st.markdown("### Holdings")
        holdings_df = st.data_editor(DEFAULT_HOLDINGS, num_rows="dynamic", use_container_width=True, key="holdings")
        usd_cash = st.number_input("USD Cash", value=4220.12, step=100.0)
        twd_cash = st.number_input("TWD Cash", value=27426.0, step=1000.0)
        run = st.button("Calculate", use_container_width=True, type="primary")

    if not run:
        st.info("Configure holdings in the sidebar and click **Calculate**.")
        return

    tickers = holdings_df.dropna().query("Shares != 0")["Ticker"].tolist()
    shares = holdings_df.dropna().set_index("Ticker")["Shares"].to_dict()
    if not tickers:
        st.error("Add at least one holding.")
        return

    with st.spinner("Downloading data…"):
        try:
            fx_pair = "TWD=X"
            dl = tickers + [benchmark, fx_pair]
            raw = yf.download(dl, start=pd.to_datetime(start), auto_adjust=True, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                data = raw["Close"]
            elif "Close" in raw.columns:
                data = raw[["Close"]]
            else:
                data = raw
            data = data.dropna(how="all").ffill()
            if isinstance(data, pd.Series):
                data = data.to_frame()
            if freq in ["W", "M"]:
                data = data.resample(_RESAMPLE_ALIAS[freq]).last()
            fx = data[fx_pair].dropna() if fx_pair in data.columns else pd.Series(index=data.index, data=30.0)
            bench_px = data[benchmark].dropna()
            prices = data[tickers].dropna(how="all")
        except Exception as e:
            st.error(f"Data error: {e}")
            return

    idx = prices.index.union(bench_px.index).union(fx.index)
    prices = prices.reindex(idx).ffill()
    bench_px = bench_px.reindex(idx).ffill()
    fx = fx.reindex(idx).ffill()

    port_val = (prices * pd.Series(shares)).sum(axis=1) + usd_cash + (twd_cash / fx)
    port_ret = port_val.pct_change()
    bench_ret = bench_px.pct_change()

    ppy = {"M": 12, "W": 52, "D": 252}[freq]
    if rf_choice.startswith("^IRX"):
        try:
            rf_raw = yf.download("^IRX", start=pd.to_datetime(start), progress=False)["Close"].ffill()
            if freq in ["W", "M"]:
                rf_raw = rf_raw.resample(_RESAMPLE_ALIAS[freq]).last()
            rf = (1 + rf_raw / 100) ** (1 / ppy) - 1
        except Exception:
            rf = pd.Series(index=port_ret.index, data=(1 + 0.05) ** (1 / ppy) - 1)
    else:
        rf = pd.Series(index=port_ret.index, data=(1 + rf_const) ** (1 / ppy) - 1)

    rf = rf.reindex(port_ret.index).ffill()

    # ── Metrics ────────────────────────────────────────────────────
    df = pd.concat([port_ret, bench_ret, rf], axis=1).dropna()
    df.columns = ["r_p", "r_b", "r_f"]
    df["rp_ex"] = df["r_p"] - df["r_f"]
    df["rb_ex"] = df["r_b"] - df["r_f"]
    df["active"] = df["r_p"] - df["r_b"]

    sharpe = (df["rp_ex"].mean() / df["rp_ex"].std(ddof=1)) * np.sqrt(ppy) if df["rp_ex"].std() else np.nan
    ir = (df["active"].mean() / df["active"].std(ddof=1)) * np.sqrt(ppy) if df["active"].std() else np.nan
    cov_pb = np.cov(df["rp_ex"], df["rb_ex"], ddof=1)[0, 1]
    var_b = np.var(df["rb_ex"], ddof=1)
    beta = cov_pb / var_b if var_b else np.nan
    treynor = (df["rp_ex"].mean() * ppy) / beta if beta else np.nan

    port_total = port_val.dropna()
    bench_norm = bench_px.dropna()
    port_dd = port_total / port_total.cummax() - 1
    bench_dd = bench_norm / bench_norm.cummax() - 1
    port_ann_ret = df["r_p"].mean() * ppy
    bench_ann_ret = df["r_b"].mean() * ppy

    section("Key Metrics")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        metric_card("Sharpe Ratio", f"{sharpe:.3f}" if pd.notna(sharpe) else "N/A", positive=sharpe > 1 if pd.notna(sharpe) else None)
    with c2:
        metric_card("Info Ratio", f"{ir:.3f}" if pd.notna(ir) else "N/A", positive=ir > 0 if pd.notna(ir) else None)
    with c3:
        metric_card("Beta", f"{beta:.3f}" if pd.notna(beta) else "N/A")
    with c4:
        metric_card("Ann. Return (Port)", f"{port_ann_ret:.2%}", positive=port_ann_ret > 0)
    with c5:
        metric_card("Ann. Return (Bench)", f"{bench_ann_ret:.2%}", positive=bench_ann_ret > 0)
    with c6:
        metric_card("Treynor", f"{treynor:.4f}" if pd.notna(treynor) else "N/A")

    # ── Charts ──────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["Equity Curve", "Drawdown", "Risk vs Return", "Rolling Sharpe"])

    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=port_val.index, y=(port_val / port_val.iloc[0]).values, name="Portfolio", line=dict(color="#1E88E5", width=2)))
        fig.add_trace(go.Scatter(x=bench_px.index, y=(bench_px / bench_px.iloc[0]).values, name=benchmark, line=dict(color="#FF9800", width=2, dash="dash")))
        fig.update_layout(**PLOTLY_LAYOUT, title="Normalised Equity Curve", height=420, yaxis_title="Normalised Value")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=["Portfolio Drawdown", f"{benchmark} Drawdown"])
        fig.add_trace(go.Scatter(x=port_dd.index, y=port_dd.values, fill="tozeroy", line=dict(color="#F44336"), name="Portfolio DD"), row=1, col=1)
        fig.add_trace(go.Scatter(x=bench_dd.index, y=bench_dd.values, fill="tozeroy", line=dict(color="#FF9800"), name=f"{benchmark} DD"), row=2, col=1)
        fig.update_layout(**PLOTLY_LAYOUT, height=480, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        prices_clean = prices.dropna(how="all")
        comp = {t: prices_clean[t].pct_change().dropna() for t in tickers if t in prices_clean.columns}
        rv = pd.DataFrame(
            [{"Ticker": k, "Ann. Return": v.mean() * ppy, "Ann. Vol": v.std(ddof=1) * np.sqrt(ppy)} for k, v in comp.items() if len(v) > 5]
        )
        if not rv.empty:
            fig = px.scatter(rv, x="Ann. Vol", y="Ann. Return", text="Ticker", color="Ann. Return",
                             color_continuous_scale="RdYlGn", template="plotly_dark")
            fig.update_traces(textposition="top center", marker_size=10)
            fig.update_layout(**PLOTLY_LAYOUT, title="Risk vs Return (Annualised)", height=420)
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        ex = (port_ret - rf).dropna()
        n = len(ex)
        default_w = min(max(3, n // 4), 52 if freq == "W" else 12 if freq == "M" else 126)
        roll_s = ex.rolling(default_w).agg(["mean", "std"]).dropna()
        if not roll_s.empty:
            sharpe_roll = (roll_s["mean"] / roll_s["std"]) * np.sqrt(ppy)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sharpe_roll.index, y=sharpe_roll.values, name="Rolling Sharpe", fill="tozeroy", line=dict(color="#1E88E5")))
            fig.add_hline(y=1, line_dash="dot", line_color="green", annotation_text="Sharpe=1")
            fig.add_hline(y=0, line_dash="solid", line_color="gray")
            fig.update_layout(**PLOTLY_LAYOUT, title=f"Rolling Sharpe (window={default_w})", height=380)
            st.plotly_chart(fig, use_container_width=True)

    # ── Download ────────────────────────────────────────────────────
    out_df = pd.concat([port_val.rename("Portfolio_Value"), bench_px.rename("Benchmark"), port_ret.rename("Port_Return"), bench_ret.rename("Bench_Return")], axis=1)
    st.download_button("⬇ Download time-series CSV", data=out_df.to_csv().encode("utf-8"), file_name="portfolio_series.csv", mime="text/csv")


# ════════════════════════════════════════════════════════════════════
# PAGE: News & Sentiment
# ════════════════════════════════════════════════════════════════════

def _parse_llm_json(text: str) -> dict:
    """Parse JSON from LLM response, handling markdown code block wrappers."""
    import json as _json, re
    text = text.strip()
    # Strip ```json ... ``` or ``` ... ``` wrappers
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return _json.loads(text.strip())


def _llm_client(api_key: str, api_base: str, model: str):
    """Return an OpenAI-compatible client; auto-set base_url for Claude models."""
    import openai
    if not api_base:
        if model.startswith("claude"):
            api_base = "https://api.anthropic.com/v1"
        else:
            api_base = None
    return openai.OpenAI(api_key=api_key, base_url=api_base or None)


def page_news_sentiment():
    st.title("📰 新聞情報")
    st.caption("多來源財經新聞聚合、LLM 情緒分析、金融報告生成")

    tab1, tab2, tab3 = st.tabs(["Live News Feed", "Sentiment Analysis", "Financial Report"])

    FEEDS = {
        "Yahoo Finance":   "https://finance.yahoo.com/news/rssindex",
        "MarketWatch":     "https://feeds.marketwatch.com/marketwatch/topstories/",
        "Reuters":         "https://feeds.reuters.com/reuters/topNews",
        "CNBC":            "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "Bloomberg (BBG)": "https://feeds.bloomberg.com/markets/news.rss",
        "Seeking Alpha":   "https://seekingalpha.com/feed.xml",
        "Benzinga":        "https://www.benzinga.com/feed",
    }

    # ── Tab 1: Fetch ──────────────────────────────────────────────────
    with tab1:
        section("RSS / News Fetch")
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_feeds = st.multiselect(
                "新聞來源（可多選）",
                list(FEEDS.keys()),
                default=["Yahoo Finance", "MarketWatch"],
            )
        with col2:
            max_articles = st.number_input("每來源最多篇數", 5, 30, 10)

        if st.button("Fetch News", type="primary"):
            if not selected_feeds:
                st.warning("請至少選擇一個新聞來源")
            else:
                try:
                    import feedparser, re as _re
                    articles = []
                    for src in selected_feeds:
                        with st.spinner(f"Fetching from {src}…"):
                            try:
                                feed = feedparser.parse(FEEDS[src])
                                for entry in feed.entries[: int(max_articles)]:
                                    summary = _re.sub(r"<[^>]+>", "", entry.get("summary", ""))[:400]
                                    articles.append({
                                        "Title":     entry.get("title", ""),
                                        "Source":    src,
                                        "Published": entry.get("published", ""),
                                        "Summary":   summary,
                                        "Link":      entry.get("link", ""),
                                    })
                            except Exception as _e:
                                st.warning(f"{src} 載入失敗：{_e}")
                    if articles:
                        st.success(f"共取得 {len(articles)} 篇（{', '.join(selected_feeds)}）")
                        st.session_state["news_articles"] = articles
                        st.session_state["news_source"] = " + ".join(selected_feeds)
                        for a in articles:
                            with st.expander(f"[{a['Source']}] {a['Title']}"):
                                st.caption(a["Published"])
                                st.write(a["Summary"])
                                st.markdown(f"[Read more]({a['Link']})")
                    else:
                        st.warning("No articles found. Try different sources.")
                except ImportError:
                    st.error("feedparser not installed.")
                except Exception as e:
                    st.error(f"Feed error: {e}")

    # ── Tab 2: Sentiment ──────────────────────────────────────────────
    with tab2:
        section("LLM Sentiment Scoring")

        col1, col2 = st.columns([2, 1])
        with col1:
            api_key = st.text_input("API Key", type="password", placeholder="sk-… or Anthropic key",
                                    key="sent_api_key")
        with col2:
            model_choice = st.selectbox("Model", [
                "gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo",
                "claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022",
            ])
        api_base = st.text_input("API Base URL (leave blank for auto-detect)", "",
                                 key="sent_api_base")

        articles = st.session_state.get("news_articles", [])
        if not articles:
            st.info("Fetch news from the 'Live News Feed' tab first.")
        else:
            st.caption(f"{len(articles)} articles ready from {st.session_state.get('news_source','')}")
            if st.button("Score Sentiment", type="primary"):
                if not api_key:
                    st.error("Please enter your API key.")
                else:
                    results, errors = [], []
                    progress = st.progress(0)
                    status   = st.empty()
                    for i, a in enumerate(articles):
                        status.caption(f"Scoring {i+1}/{len(articles)}: {a['Title'][:60]}…")
                        try:
                            client = _llm_client(api_key, api_base, model_choice)
                            prompt = (
                                "Rate the financial market sentiment of this news headline.\n"
                                "Reply ONLY with valid JSON: {\"sentiment\": \"POSITIVE\"|\"NEGATIVE\"|\"NEUTRAL\", \"score\": <float -1.0 to 1.0>}\n\n"
                                f"Title: {a['Title']}\nSummary: {a['Summary'][:300]}"
                            )
                            resp = client.chat.completions.create(
                                model=model_choice,
                                messages=[{"role": "user", "content": prompt}],
                                temperature=0,
                                max_tokens=60,
                            )
                            text = resp.choices[0].message.content.strip()
                            parsed = _parse_llm_json(text)
                            sentiment = str(parsed.get("sentiment", "NEUTRAL")).upper()
                            if sentiment not in ("POSITIVE", "NEGATIVE", "NEUTRAL"):
                                sentiment = "NEUTRAL"
                            results.append({
                                "Title":     a["Title"][:80],
                                "Sentiment": sentiment,
                                "Score":     float(parsed.get("score", 0.0)),
                            })
                        except Exception as e:
                            results.append({"Title": a["Title"][:80], "Sentiment": "ERROR", "Score": 0.0})
                            errors.append(f"{a['Title'][:50]}: {e}")
                        progress.progress((i + 1) / len(articles))

                    status.empty()
                    st.session_state["sentiment_results"] = results

                    if errors:
                        with st.expander(f"⚠️ {len(errors)} errors (click to view)"):
                            for err in errors:
                                st.caption(err)

                    df_sent = pd.DataFrame(results)
                    valid = df_sent[df_sent["Sentiment"] != "ERROR"]
                    if not valid.empty:
                        avg = valid["Score"].mean()
                        label = "Positive" if avg > 0.1 else "Negative" if avg < -0.1 else "Neutral"
                        st.metric("Average Sentiment Score", f"{avg:+.3f}", delta=label)

                        fig = px.bar(
                            valid, x="Score", y="Title", orientation="h",
                            color="Score", color_continuous_scale="RdYlGn",
                            template="plotly_dark", text_auto=".2f",
                        )
                        fig.update_layout(**PLOTLY_LAYOUT, title="Sentiment Scores",
                                          height=max(300, 45 * len(valid)))
                        st.plotly_chart(fig, use_container_width=True)

                    st.dataframe(df_sent, use_container_width=True)

    # ── Tab 3: Financial Report ───────────────────────────────────────
    with tab3:
        section("AI Financial Report Summary")
        st.caption("Generate a structured financial market report from fetched news")

        col1, col2 = st.columns([2, 1])
        with col1:
            rep_api_key = st.text_input("API Key", type="password", placeholder="sk-… or Anthropic key",
                                        key="rep_api_key")
        with col2:
            rep_model = st.selectbox("Model", [
                "gpt-4o-mini", "gpt-4o",
                "claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022",
            ], key="rep_model")
        rep_api_base = st.text_input("API Base URL (leave blank for auto-detect)", "",
                                     key="rep_api_base")

        articles = st.session_state.get("news_articles", [])
        sentiment_results = st.session_state.get("sentiment_results", [])

        if not articles:
            st.info("Fetch news from the 'Live News Feed' tab first.")
        else:
            st.caption(f"{len(articles)} articles available. Sentiment data: {'✅' if sentiment_results else '⬜ (optional)'}")

            if st.button("Generate Financial Report", type="primary"):
                if not rep_api_key:
                    st.error("Please enter your API key.")
                else:
                    with st.spinner("Generating report…"):
                        try:
                            # Build article digest
                            digest_lines = []
                            for i, a in enumerate(articles[:20]):
                                sent_info = ""
                                if sentiment_results and i < len(sentiment_results):
                                    s = sentiment_results[i]
                                    if s["Sentiment"] != "ERROR":
                                        sent_info = f" [{s['Sentiment']}, score={s['Score']:+.2f}]"
                                digest_lines.append(f"{i+1}. {a['Title']}{sent_info}\n   {a['Summary'][:200]}")
                            digest = "\n\n".join(digest_lines)

                            from datetime import date as _date
                            today = _date.today().strftime("%B %d, %Y")
                            prompt = (
                                f"You are a senior financial analyst. Based on the following recent news articles (as of {today}), "
                                "write a concise structured financial market report in Traditional Chinese. "
                                "Include: (1) 市場概況 Market Overview, (2) 主要趨勢 Key Trends, "
                                "(3) 風險提示 Risk Factors, (4) 投資展望 Outlook. "
                                "Keep each section to 3-5 bullet points. Be specific and data-driven.\n\n"
                                f"NEWS DIGEST:\n{digest}"
                            )

                            client = _llm_client(rep_api_key, rep_api_base, rep_model)
                            resp = client.chat.completions.create(
                                model=rep_model,
                                messages=[{"role": "user", "content": prompt}],
                                temperature=0.3,
                                max_tokens=1200,
                            )
                            report_text = resp.choices[0].message.content.strip()
                            st.session_state["financial_report"] = report_text

                        except Exception as e:
                            st.error(f"Report generation failed: {e}")

            if st.session_state.get("financial_report"):
                st.markdown("---")
                st.markdown(f"### 📋 Financial Market Report — {date.today().strftime('%B %d, %Y')}")
                st.markdown(st.session_state["financial_report"])
                st.download_button(
                    "⬇ Download Report",
                    data=st.session_state["financial_report"],
                    file_name=f"rbs_report_{date.today()}.txt",
                    mime="text/plain",
                )


# ════════════════════════════════════════════════════════════════════
# PAGE: Institutional Stock Selector
# ════════════════════════════════════════════════════════════════════

def page_stock_selector():
    try:
        from stock_db import (
            ADB, MKTS, STRATS, MACRO_FACTORS, MACRO_BOOST,
            INSIGHTS, MWARN, MAVOID,
        )
    except ModuleNotFoundError:
        st.title("🏦 機構選股模型")
        st.error("❌ 找不到 `stock_db.py`，請在 Colab Cell 2 重新同步檔案：")
        st.code(
            "import urllib.request\n"
            "REPO   = 'RoyARVSA/RBS_claude_finance'\n"
            "BRANCH = 'claude/optimize-analysis-dashboard-NZUKB'\n"
            "url = 'https://raw.githubusercontent.com/' + REPO + '/' + BRANCH + '/stock_db.py'\n"
            "dst = DRIVE_BASE / 'stock_db.py'\n"
            "urllib.request.urlretrieve(url, dst)\n"
            "print('stock_db.py downloaded:', dst.stat().st_size, 'bytes')",
            language="python",
        )
        st.info("下載後重新整理頁面即可。")
        return

    st.title("🏦 機構選股模型")
    st.caption("六步驟系統化篩選流程，結合宏觀環境、策略偏好與產業輪動")

    # ── Persistent step state ────────────────────────────────────────
    if "ss_step" not in st.session_state:
        st.session_state.ss_step = 0
    if "ss_sel" not in st.session_state:
        st.session_state.ss_sel = {}

    sel = st.session_state.ss_sel

    # ── Progress bar ─────────────────────────────────────────────────
    STEPS = ["市場", "策略", "宏觀環境", "資產類型", "產業", "選股結果"]
    progress = st.session_state.ss_step / (len(STEPS) - 1)
    st.progress(progress)
    cols_step = st.columns(len(STEPS))
    for i, s in enumerate(STEPS):
        with cols_step[i]:
            if i < st.session_state.ss_step:
                st.markdown(f"<div style='text-align:center;color:#4CAF50;font-size:0.8rem'>✔ {s}</div>", unsafe_allow_html=True)
            elif i == st.session_state.ss_step:
                st.markdown(f"<div style='text-align:center;color:#1E88E5;font-weight:700;font-size:0.85rem'>● {s}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='text-align:center;color:#7B8599;font-size:0.8rem'>○ {s}</div>", unsafe_allow_html=True)

    st.markdown("---")

    # ────────────────────────────────────────────────────────────────
    # STEP 0 – Market
    # ────────────────────────────────────────────────────────────────
    if st.session_state.ss_step == 0:
        section("Step 1 of 6 — 選擇市場")
        st.markdown("請選擇您要分析的目標市場：")

        mkt_cols = st.columns(len(MKTS))
        for i, (label, code) in enumerate(MKTS.items()):
            with mkt_cols[i]:
                selected = sel.get("market") == code
                btn_style = "primary" if selected else "secondary"
                if st.button(label, key=f"mkt_{code}", type=btn_style, use_container_width=True):
                    sel["market"] = code

        if sel.get("market"):
            mkt_label = next(k for k, v in MKTS.items() if v == sel["market"])
            st.success(f"已選擇：{mkt_label}")
            if MWARN.get(sel["market"]):
                st.warning(f"⚠️ 風險提示：{MWARN[sel['market']]}")
            if st.button("下一步 →", type="primary"):
                st.session_state.ss_step = 1
                st.rerun()
        else:
            st.info("請點選上方市場按鈕")

    # ────────────────────────────────────────────────────────────────
    # STEP 1 – Strategy
    # ────────────────────────────────────────────────────────────────
    elif st.session_state.ss_step == 1:
        section("Step 2 of 6 — 選擇投資策略")

        strat_cols = st.columns(3)
        for i, (key, info) in enumerate(STRATS.items()):
            with strat_cols[i % 3]:
                selected = sel.get("strategy") == key
                card_border = "#1E88E5" if selected else "#3D4255"
                card_bg     = "#1E2847" if selected else "#1A1D27"
                st.markdown(
                    f"""<div style='border:2px solid {card_border};border-radius:10px;padding:14px;margin:4px 0;
                    background:{card_bg};'>
                    <div style='font-size:1.05rem;font-weight:700;color:#FAFAFA'>{info['label']}</div>
                    <div style='font-size:0.82rem;color:#C8CAD4;margin-top:6px;line-height:1.4'>{info['desc']}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
                if st.button("選擇" if not selected else "✔ 已選", key=f"strat_{key}",
                             type="primary" if selected else "secondary", use_container_width=True):
                    sel["strategy"] = key

        if sel.get("strategy"):
            strat_info = STRATS[sel["strategy"]]
            st.success(f"已選擇策略：{strat_info['label']}")
            st.info(f"⚠️ 注意：{MAVOID.get(sel['strategy'], '')}")

        col_back, col_next = st.columns(2)
        with col_back:
            if st.button("← 上一步", use_container_width=True):
                st.session_state.ss_step = 0
                st.rerun()
        with col_next:
            if sel.get("strategy") and st.button("下一步 →", type="primary", use_container_width=True):
                st.session_state.ss_step = 2
                st.rerun()

    # ────────────────────────────────────────────────────────────────
    # STEP 2 – Macro Environment
    # ────────────────────────────────────────────────────────────────
    elif st.session_state.ss_step == 2:
        section("Step 3 of 6 — 當前宏觀環境")
        st.markdown("請選擇目前您認為最相關的宏觀因素（可多選）：")

        macro_chosen = st.multiselect(
            "宏觀因素",
            MACRO_FACTORS,
            default=sel.get("macro", []),
            label_visibility="collapsed",
        )

        if macro_chosen:
            st.markdown("##### 宏觀解讀")
            for m in macro_chosen:
                if m in INSIGHTS:
                    st.markdown(f"**{m}**：{INSIGHTS[m]}")

        col_back, col_next = st.columns(2)
        with col_back:
            if st.button("← 上一步", use_container_width=True):
                sel["macro"] = macro_chosen
                st.session_state.ss_step = 1
                st.rerun()
        with col_next:
            if st.button("下一步 →", type="primary", use_container_width=True):
                sel["macro"] = macro_chosen
                st.session_state.ss_step = 3
                st.rerun()

    # ────────────────────────────────────────────────────────────────
    # STEP 3 – Asset Type
    # ────────────────────────────────────────────────────────────────
    elif st.session_state.ss_step == 3:
        section("Step 4 of 6 — 資產類型")
        st.markdown("請選擇想納入的資產類型（可多選）：")

        mkt_code = sel.get("market", "US")
        all_types_in_mkt = sorted({v["asset_type"] for v in ADB.get(mkt_code, {}).values()})

        asset_types = st.multiselect(
            "資產類型",
            all_types_in_mkt,
            default=sel.get("asset_types", all_types_in_mkt),
            label_visibility="collapsed",
        )

        col_back, col_next = st.columns(2)
        with col_back:
            if st.button("← 上一步", use_container_width=True):
                sel["asset_types"] = asset_types
                st.session_state.ss_step = 2
                st.rerun()
        with col_next:
            if asset_types and st.button("下一步 →", type="primary", use_container_width=True):
                sel["asset_types"] = asset_types
                st.session_state.ss_step = 4
                st.rerun()

        if not asset_types:
            st.warning("請至少選擇一種資產類型")

    # ────────────────────────────────────────────────────────────────
    # STEP 4 – Industry
    # ────────────────────────────────────────────────────────────────
    elif st.session_state.ss_step == 4:
        section("Step 5 of 6 — 產業篩選")

        mkt_code    = sel.get("market", "US")
        strategy    = sel.get("strategy", "growth")
        macro_list  = sel.get("macro", [])
        asset_types = sel.get("asset_types", [])

        industries_in_mkt = ADB.get(mkt_code, {})

        # Score each industry for relevance
        scored: list[tuple[str, int, str]] = []
        for ind_name, ind_data in industries_in_mkt.items():
            if ind_data["asset_type"] not in asset_types:
                continue
            score = 0
            # Strategy match
            if strategy in ind_data["strats"]:
                score += 3
            # Macro boost
            for m in macro_list:
                if ind_name in MACRO_BOOST.get(m, []):
                    score += 2
                if m in ind_data.get("macro", []):
                    score += 1
            scored.append((ind_name, score, ind_data.get("desc", "")))

        # Sort by relevance score
        scored.sort(key=lambda x: -x[1])

        if not scored:
            st.warning("目前篩選條件下沒有符合的產業，請返回調整設定。")
        else:
            st.markdown("依宏觀環境與策略適配度排序（⭐ 越高越匹配）：")
            industry_options = []
            for name, score, desc in scored:
                stars = "⭐" * min(score, 5) if score > 0 else "☆"
                industry_options.append(f"{stars} {name} — {desc}"[:90])

            # Map display → name
            display_map = {opt: scored[i][0] for i, opt in enumerate(industry_options)}

            # Pre-select previously chosen industries
            prev_ind = sel.get("industries", [])
            default_display = [opt for opt, name in display_map.items() if name in prev_ind]

            chosen_display = st.multiselect(
                "選擇一個或多個產業",
                industry_options,
                default=default_display,
                label_visibility="collapsed",
            )
            chosen_industries = [display_map[d] for d in chosen_display]

        col_back, col_next = st.columns(2)
        with col_back:
            if st.button("← 上一步", use_container_width=True):
                st.session_state.ss_step = 3
                st.rerun()
        with col_next:
            if scored and chosen_industries and st.button("查看選股結果 →", type="primary", use_container_width=True):
                sel["industries"] = chosen_industries
                st.session_state.ss_step = 5
                st.rerun()

    # ────────────────────────────────────────────────────────────────
    # STEP 5 – Results
    # ────────────────────────────────────────────────────────────────
    elif st.session_state.ss_step == 5:
        section("Step 6 of 6 — 選股結果")

        mkt_code    = sel.get("market", "US")
        strategy    = sel.get("strategy", "growth")
        macro_list  = sel.get("macro", [])
        industries  = sel.get("industries", [])

        mkt_label  = next((k for k, v in MKTS.items() if v == mkt_code), mkt_code)
        strat_label = STRATS.get(strategy, {}).get("label", strategy)

        # Summary chips
        chip_style = "background:#1E88E5;padding:3px 10px;border-radius:12px;font-size:0.75rem;margin:2px;display:inline-block"
        chips_html = "".join(
            f"<span style='{chip_style}'>{t}</span>"
            for t in [mkt_label, strat_label] + macro_list + industries
        )
        st.markdown(chips_html, unsafe_allow_html=True)
        st.markdown("")

        # Collect candidates
        mkt_db = ADB.get(mkt_code, {})
        candidates: list[str] = []
        industry_map: dict[str, str] = {}  # ticker → industry
        for ind in industries:
            for tkr in mkt_db.get(ind, {}).get("tickers", []):
                candidates.append(tkr)
                industry_map[tkr] = ind
        candidates = list(dict.fromkeys(candidates))  # deduplicate, preserve order

        if not candidates:
            st.warning("沒有找到候選股票，請返回調整篩選條件。")
            if st.button("← 重新選擇"):
                st.session_state.ss_step = 4
                st.rerun()
            return

        st.caption(f"候選標的 {len(candidates)} 檔：{', '.join(candidates[:12])}{'…' if len(candidates) > 12 else ''}")

        # ── Live price fetch ─────────────────────────────────────────
        section("即時行情")
        with st.spinner(f"抓取 {len(candidates)} 檔即時報價…"):
            try:
                import yfinance as yf
                raw = yf.download(
                    candidates, period="1y", auto_adjust=True, progress=False
                )
                if isinstance(raw.columns, pd.MultiIndex):
                    px_close = raw["Close"].dropna(how="all")
                else:
                    px_close = raw[["Close"]].rename(columns={"Close": candidates[0]})
                valid = [c for c in candidates if c in px_close.columns and not px_close[c].dropna().empty]
            except Exception as e:
                st.error(f"行情下載失敗：{e}")
                valid = []
                px_close = pd.DataFrame()

        if valid:
            # Build summary table
            rows = []
            for tkr in valid:
                s = px_close[tkr].dropna()
                if len(s) < 10:
                    continue
                last = float(s.iloc[-1])
                chg1d = float((s.iloc[-1] / s.iloc[-2] - 1)) if len(s) >= 2 else np.nan
                chg1m = float((s.iloc[-1] / s.iloc[-22] - 1)) if len(s) >= 22 else np.nan
                chg3m = float((s.iloc[-1] / s.iloc[-63] - 1)) if len(s) >= 63 else np.nan
                ann_vol = float(s.pct_change().dropna().std() * np.sqrt(252))
                rows.append({
                    "代碼": tkr,
                    "產業": industry_map.get(tkr, ""),
                    "現價": last,
                    "1日%": chg1d,
                    "1月%": chg1m,
                    "3月%": chg3m,
                    "年化波動": ann_vol,
                })

            tbl = pd.DataFrame(rows).set_index("代碼")
            fmt = {
                "現價": "{:.2f}",
                "1日%": "{:.2%}",
                "1月%": "{:.2%}",
                "3月%": "{:.2%}",
                "年化波動": "{:.2%}",
            }

            def color_ret(val):
                if isinstance(val, float) and not np.isnan(val):
                    return "color: #4CAF50" if val > 0 else "color: #F44336"
                return ""

            styled = (
                tbl.style
                .format(fmt, na_rep="—")
                .map(color_ret, subset=["1日%", "1月%", "3月%"])
            )
            st.dataframe(styled, use_container_width=True)

            # ── Performance chart ────────────────────────────────────
            section("相對績效走勢 (近1年 = 100)")
            show_tickers = st.multiselect(
                "選擇標的對比",
                valid,
                default=valid[:min(6, len(valid))],
                key="chart_tickers",
            )
            if show_tickers:
                fig = go.Figure()
                colors = px.colors.qualitative.Plotly
                for i, tkr in enumerate(show_tickers):
                    s = px_close[tkr].dropna()
                    norm = s / s.iloc[0] * 100
                    fig.add_trace(go.Scatter(
                        x=norm.index, y=norm.values, name=tkr,
                        mode="lines", line=dict(width=2, color=colors[i % len(colors)]),
                    ))
                fig.update_layout(
                    **PLOTLY_LAYOUT,
                    title="Indexed Price Performance (Base = 100)",
                    height=420,
                    hovermode="x unified",
                    yaxis_title="Index (100 = start)",
                )
                st.plotly_chart(fig, use_container_width=True)

            # ── Return distribution ──────────────────────────────────
            if len(show_tickers) > 1:
                section("報酬分佈對比")
                rets_df = px_close[show_tickers].pct_change().dropna()
                fig2 = go.Figure()
                colors = px.colors.qualitative.Plotly
                for i, tkr in enumerate(show_tickers):
                    fig2.add_trace(go.Violin(
                        y=rets_df[tkr].values, name=tkr,
                        box_visible=True, meanline_visible=True,
                        line_color=colors[i % len(colors)], opacity=0.7,
                    ))
                fig2.update_layout(
                    **PLOTLY_LAYOUT, title="Daily Return Distribution",
                    height=380, yaxis_title="Daily Return",
                )
                st.plotly_chart(fig2, use_container_width=True)

        # ── AI Analysis ──────────────────────────────────────────────
        section("AI 深度分析")
        with st.expander("使用 LLM 生成機構選股報告", expanded=False):
            col1, col2 = st.columns([2, 1])
            with col1:
                ai_key = st.text_input("API Key", type="password",
                                       placeholder="sk-… 或 Anthropic key",
                                       key="stk_api_key")
            with col2:
                ai_model = st.selectbox("模型", [
                    "gpt-4o-mini", "gpt-4o",
                    "claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022",
                ], key="stk_model")
            ai_base = st.text_input("API Base URL（空白自動偵測）", "", key="stk_base")

            if st.button("生成選股報告", type="primary", key="stk_gen"):
                if not ai_key:
                    st.error("請填入 API Key")
                else:
                    # Build context
                    ind_lines = []
                    for ind in industries:
                        ind_data = mkt_db.get(ind, {})
                        tickers_str = ", ".join(ind_data.get("tickers", [])[:8])
                        ind_lines.append(f"- {ind}：{ind_data.get('desc', '')}（代表標的：{tickers_str}）")

                    macro_str = "、".join(macro_list) if macro_list else "無特定宏觀偏好"
                    strat_str = STRATS.get(strategy, {}).get("label", strategy)

                    prompt = (
                        f"你是一位資深機構投資分析師。請根據以下投資框架，"
                        f"以繁體中文撰寫一份專業選股報告：\n\n"
                        f"**目標市場**：{mkt_label}\n"
                        f"**投資策略**：{strat_str}\n"
                        f"**宏觀環境**：{macro_str}\n"
                        f"**篩選產業**：\n" + "\n".join(ind_lines) + "\n\n"
                        "請提供：\n"
                        "1. 📊 市場環境評估（2-3段）\n"
                        "2. 🎯 重點標的推薦（每個產業各2-3檔，說明邏輯）\n"
                        "3. ⚠️ 主要風險提示\n"
                        "4. 📅 時間框架建議（短中長期配置比例）\n"
                        "報告應具體、數據導向、符合機構投資標準。"
                    )

                    with st.spinner("AI 正在生成報告…"):
                        try:
                            client = _llm_client(ai_key, ai_base, ai_model)
                            resp = client.chat.completions.create(
                                model=ai_model,
                                messages=[{"role": "user", "content": prompt}],
                                temperature=0.3,
                                max_tokens=1500,
                            )
                            ai_report = resp.choices[0].message.content.strip()
                            st.session_state["stk_report"] = ai_report
                        except Exception as e:
                            st.error(f"報告生成失敗：{e}")

            if st.session_state.get("stk_report"):
                st.markdown("---")
                st.markdown(st.session_state["stk_report"])
                st.download_button(
                    "⬇ 下載選股報告",
                    data=st.session_state["stk_report"],
                    file_name=f"stk_report_{date.today()}.txt",
                    mime="text/plain",
                )

        # ── Navigation ───────────────────────────────────────────────
        st.markdown("---")
        col_back, col_reset = st.columns(2)
        with col_back:
            if st.button("← 修改產業選擇", use_container_width=True):
                st.session_state.ss_step = 4
                st.rerun()
        with col_reset:
            if st.button("🔄 重新開始", use_container_width=True):
                st.session_state.ss_step = 0
                st.session_state.ss_sel = {}
                if "stk_report" in st.session_state:
                    del st.session_state["stk_report"]
                st.rerun()


# ════════════════════════════════════════════════════════════════════
# PAGE: Export Report
# ════════════════════════════════════════════════════════════════════

def page_export():
    st.title("📦 匯出報告")
    st.caption("Generate and download a summary zip of charts and KPI tables")

    st.markdown(
        """
        ### How to use
        1. Navigate to each analysis page and run the calculations
        2. Come back here to package your session data into a downloadable report

        > **Tip:** Run the full analysis pipeline first, then export.
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        report_name = st.text_input("Report Name", f"rbs_report_{date.today()}")
    with col2:
        include_raw = st.checkbox("Include raw price data", value=False)

    if st.button("Generate Report ZIP", type="primary"):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            meta = pd.DataFrame(
                {
                    "Field": ["Generated", "Report Name", "Note"],
                    "Value": [str(date.today()), report_name, "RBS Finance Dashboard v2.0"],
                }
            )
            z.writestr("metadata.csv", meta.to_csv(index=False))

            if include_raw:
                placeholder = pd.DataFrame({"info": ["No active session data to export. Run analysis pages first."]})
                z.writestr("session_data.csv", placeholder.to_csv(index=False))

            readme = (
                "RBS Finance Dashboard – Export\n"
                "==============================\n"
                f"Generated: {date.today()}\n\n"
                "Files:\n"
                "  metadata.csv       – Report metadata\n"
                "  session_data.csv   – Session data (if enabled)\n\n"
                "頁面：市場總覽 / 持倉分析 / 風險管理 / 股票研究 / 公司分析 /\n"
                "      即時警報 / 交易工具 / 模擬交易 / 機構選股 / 新聞情報\n"
            )
            z.writestr("README.txt", readme)

        st.download_button(
            "⬇ Download Report ZIP",
            data=buf.getvalue(),
            file_name=f"{report_name}.zip",
            mime="application/zip",
        )
        st.success("Report generated!")


# ════════════════════════════════════════════════════════════════════
# PAGE: Market Overview (Home Dashboard)
# ════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=120)
def _fetch_market_snapshot():
    import yfinance as yf

    INDICES = {
        "^GSPC":     ("S&P 500",    "index"),
        "^IXIC":     ("NASDAQ",     "index"),
        "^DJI":      ("Dow Jones",  "index"),
        "^RUT":      ("Russell 2k", "index"),
        "^VIX":      ("VIX",        "fear"),
        "^TNX":      ("10Y Yield",  "rate"),
        "GLD":       ("Gold",       "commodity"),
        "USO":       ("Oil ETF",    "commodity"),
        "DX-Y.NYB":  ("USD Index",  "fx"),
        "EURUSD=X":  ("EUR/USD",    "fx"),
        "JPY=X":     ("USD/JPY",    "fx"),
    }
    SECTORS = {
        "XLK": "科技", "XLF": "金融", "XLE": "能源",
        "XLV": "醫療", "XLY": "非必需消費", "XLP": "必需消費",
        "XLI": "工業", "XLU": "公用事業", "XLRE": "房地產",
        "XLB": "原物料", "XLC": "通訊",
    }
    def _batch_closes(tickers: list[str], period: str) -> dict:
        """一次批次下載多檔的收盤序列，穩健處理 MultiIndex 兩種版面。"""
        try:
            raw = yf.download(tickers, period=period, auto_adjust=True,
                              progress=False, threads=True)
        except Exception:
            return {}
        if raw is None or raw.empty:
            return {}
        out = {}
        is_multi = isinstance(raw.columns, pd.MultiIndex)
        for tkr in tickers:
            try:
                if is_multi:
                    if ("Close", tkr) in raw.columns:
                        s = raw[("Close", tkr)].dropna()
                    elif (tkr, "Close") in raw.columns:
                        s = raw[(tkr, "Close")].dropna()
                    else:
                        continue
                else:                              # 單檔 → 平面欄位
                    s = raw["Close"].squeeze().dropna()
                if len(s) >= 2:
                    out[tkr] = s
            except Exception:
                continue
        return out

    # 指數與板塊兩批並行下載（22 次序列 → 2 次批次，快 3-5 倍）
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_idx = ex.submit(_batch_closes, list(INDICES.keys()), "5d")
        f_sec = ex.submit(_batch_closes, list(SECTORS.keys()), "2mo")
        idx_closes = f_idx.result()
        sec_closes = f_sec.result()

    snapshot: dict = {}
    for tkr, (name, cat) in INDICES.items():
        s = idx_closes.get(tkr)
        if s is not None and len(s) >= 2:
            last, prev = float(s.iloc[-1]), float(s.iloc[-2])
            snapshot[name] = {"price": last, "chg": last / prev - 1, "cat": cat}

    sectors: dict = {}
    for tkr, name in SECTORS.items():
        s = sec_closes.get(tkr)
        if s is not None and len(s) >= 2:
            chg = float(s.iloc[-1] / s.iloc[-2] - 1)
            chg_1m = float(s.iloc[-1] / s.iloc[max(0, len(s) - 22)] - 1) if len(s) >= 22 else chg
            sectors[name] = {"chg": chg, "chg_1m": chg_1m}

    return snapshot, sectors


def page_market_overview():
    st.title("🏠 市場總覽")
    now_str = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    st.caption(f"全球市場概覽 · 資料快取 2 分鐘 · 更新：{now_str}")

    if st.button("🔄 刷新數據"):
        st.cache_data.clear()
        st.rerun()

    with st.spinner("載入市場數據…"):
        snapshot, sectors = _fetch_market_snapshot()

    if not snapshot:
        st.error("無法載入市場數據，請稍後再試。")
        return

    # ── Major indices ──────────────────────────────────────────────
    section("主要指數")
    index_items = [(n, v) for n, v in snapshot.items() if v["cat"] in ("index", "fear")]
    cols_idx = st.columns(len(index_items))
    for i, (name, data) in enumerate(index_items):
        with cols_idx[i]:
            chg = data["chg"]
            color = "#4CAF50" if chg >= 0 else "#F44336"
            arrow = "▲" if chg >= 0 else "▼"
            price_str = f"{data['price']:.1f}" if name == "VIX" else f"{data['price']:,.2f}"
            st.markdown(
                f"<div class='metric-card' style='border-color:{color}60'>"
                f"<div class='metric-label'>{name}</div>"
                f"<div class='metric-value' style='font-size:1.3rem'>{price_str}</div>"
                f"<div style='color:{color};font-weight:600;font-size:0.9rem'>{arrow} {chg:+.2%}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── Macro: rates / commodities / FX ───────────────────────────
    section("宏觀指標")
    macro_items = [(n, v) for n, v in snapshot.items() if v["cat"] in ("rate", "commodity", "fx")]
    cols_mac = st.columns(len(macro_items))
    for i, (name, data) in enumerate(macro_items):
        with cols_mac[i]:
            chg = data["chg"]
            color = "#4CAF50" if chg >= 0 else "#F44336"
            arrow = "▲" if chg >= 0 else "▼"
            price_str = f"{data['price']:.4f}" if "/" in name else f"{data['price']:.2f}"
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-label'>{name}</div>"
                f"<div class='metric-value' style='font-size:1.2rem'>{price_str}</div>"
                f"<div style='color:{color};font-size:0.85rem'>{arrow} {chg:+.2%}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── Sector performance ─────────────────────────────────────────
    section("美股板塊表現")
    if sectors:
        sec_df = pd.DataFrame([
            {"板塊": k, "今日%": v["chg"] * 100, "近1月%": v["chg_1m"] * 100}
            for k, v in sectors.items()
        ])
        tab_1d, tab_1m = st.tabs(["今日", "近1個月"])
        for tab_s, col_s, title_s in [
            (tab_1d, "今日%",  "板塊今日漲跌（%）"),
            (tab_1m, "近1月%", "板塊近1個月漲跌（%）"),
        ]:
            with tab_s:
                fig = px.bar(
                    sec_df.sort_values(col_s),
                    x=col_s, y="板塊", orientation="h",
                    color=col_s, color_continuous_scale="RdYlGn",
                    color_continuous_midpoint=0, text_auto=".2f",
                )
                fig.update_layout(
                    **PLOTLY_LAYOUT, height=380,
                    coloraxis_showscale=False,
                    xaxis_title=title_s, yaxis_title="",
                )
                st.plotly_chart(fig, use_container_width=True)

    # ── Macro (FRED) ───────────────────────────────────────────────
    section("總經指標（FRED）")
    import os as _os
    fred_key = _os.environ.get("FRED_API_KEY", "")
    try:
        fred_key = fred_key or st.secrets.get("FRED_API_KEY", "")
    except Exception:
        pass
    if not fred_key:
        fred_key = st.text_input(
            "FRED API Key（免費申請 fred.stlouisfed.org，或設 Secret FRED_API_KEY）",
            type="password", key="fred_key", placeholder="填入後顯示總經數據",
        )
    st.session_state["macro_data"] = {}
    if fred_key:
        try:
            import macro as _macro
            with st.spinner("載入總經數據…"):
                md = _macro.fetch_macro(fred_key)
            st.session_state["macro_data"] = md
            order = ["fed_funds", "y10", "y2", "curve", "cpi", "unemploy"]
            items = ([(k, md[k]) for k in order if k in md and md[k].get("value") is not None]
                     if md else [])
            if items:
                cols_m = st.columns(len(items))
                for col, (k, d) in zip(cols_m, items):
                    with col:
                        chg = d.get("chg")
                        color = "#9EA3B0" if chg is None else ("#4CAF50" if chg >= 0 else "#F44336")
                        arrow = "" if chg is None else ("▲" if chg >= 0 else "▼")
                        chg_s = "" if chg is None else f"{arrow} {chg:+.2f}"
                        st.markdown(
                            f"<div class='metric-card'><div class='metric-label'>{d['label']}</div>"
                            f"<div class='metric-value' style='font-size:1.2rem'>{d['value']:.2f}%</div>"
                            f"<div style='color:{color};font-size:0.8rem'>{chg_s}</div></div>",
                            unsafe_allow_html=True,
                        )
                reg = _macro.macro_regime(md)
                icon = {"caution": "🔴", "neutral": "🟡", "ok": "🟢"}.get(reg["risk"], "🟡")
                st.markdown(f"{icon} " + "　".join(reg["signals"]))
            else:
                st.caption("無法載入總經數據，請確認 API key。")
        except Exception as e:
            st.caption(f"總經數據載入失敗：{e}")
    else:
        st.caption("設定 FRED API key 後可顯示 Fed 利率、殖利率曲線、CPI、失業率等總經指標。")

    # ── Quick news ─────────────────────────────────────────────────
    section("市場快訊")
    try:
        import feedparser
        feed = feedparser.parse("https://feeds.marketwatch.com/marketwatch/topstories/")
        for entry in feed.entries[:5]:
            st.markdown(
                f"**[{entry.get('title','')}]({entry.get('link','#')})**  \n"
                f"<small style='color:#B8C0D0'>{entry.get('published','')}</small>",
                unsafe_allow_html=True,
            )
            st.markdown("---")
    except Exception:
        st.info("新聞載入失敗，請使用「📰 新聞情報」頁面。")

    # ── AI Market Intelligence ─────────────────────────────────────
    section("🤖 AI 市場智能分析")
    with st.expander("展開 AI 自主市場分析（需要 API Key）", expanded=False):
        ai_key_ov = st.text_input("API Key", type="password", key="ov_ai_key",
                                   placeholder="sk-… 或 Anthropic key")
        ai_model_ov = st.selectbox("模型", [
            "claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022",
            "gpt-4o-mini", "gpt-4o",
        ], key="ov_ai_model")
        ai_base_ov = st.text_input("API Base URL（留空自動判斷）", "", key="ov_ai_base")

        if st.button("🚀 執行全市場 AI 分析", type="primary", key="ov_ai_run"):
            if not ai_key_ov:
                st.error("請輸入 API Key")
            else:
                with st.spinner("AI 正在分析市場數據，約需 20–40 秒…"):
                    try:
                        # Build a comprehensive market context for the AI
                        ctx_lines = ["=== 當前市場快照 ==="]
                        for name, data in snapshot.items():
                            ctx_lines.append(f"{name}: {data['price']:.2f}  ({data['chg']:+.2%})")
                        if sectors:
                            ctx_lines.append("\n=== 板塊今日表現 ===")
                            sorted_sec = sorted(sectors.items(), key=lambda x: -x[1]["chg"])
                            for name, data in sorted_sec:
                                ctx_lines.append(f"{name}: {data['chg']:+.2%}  (近1月:{data['chg_1m']:+.2%})")
                        # Macro (FRED) context
                        _md = st.session_state.get("macro_data") or {}
                        if _md:
                            try:
                                import macro as _macro
                                ctx_lines.append("\n=== 總經指標（FRED）===")
                                ctx_lines.append(_macro.macro_summary_text(_md))
                                ctx_lines.extend(_macro.macro_regime(_md)["signals"])
                            except Exception:
                                pass
                        # Fetch top headlines for context
                        try:
                            import feedparser as _fp, re as _re
                            _feed = _fp.parse("https://feeds.marketwatch.com/marketwatch/topstories/")
                            headlines = [_re.sub(r"<[^>]+>","",e.get("title","")) for e in _feed.entries[:8]]
                            ctx_lines.append("\n=== 今日重要新聞標題 ===")
                            ctx_lines.extend(headlines)
                        except Exception:
                            pass

                        market_ctx = "\n".join(ctx_lines)
                        prompt = f"""你是一位頂尖的跨市場量化分析師。根據以下即時市場數據，提供全面的自主市場分析。

{market_ctx}

請完成以下分析（繁體中文回覆）：

1. **整體市場情緒**：Risk-on 或 Risk-off？判斷依據是什麼？
2. **最強/最弱板塊**：今日及近1個月的輪動方向，背後邏輯是什麼？
3. **宏觀信號解讀**：VIX、10年期殖利率、美元指數、黃金、油價的綜合訊息
4. **值得關注的機會**：基於當前數據，哪些板塊或主題有潛在機會？
5. **主要風險提示**：當前市場最大的3個潛在風險因子
6. **短期展望（1-2週）**：基於技術面與基本面，簡短預判市場走向

每項分析要具體且有依據，避免泛泛而談。"""

                        client = _llm_client(ai_key_ov, ai_base_ov, ai_model_ov)
                        resp = client.chat.completions.create(
                            model=ai_model_ov,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.3,
                            max_tokens=1500,
                        )
                        analysis = resp.choices[0].message.content
                        st.session_state["ov_ai_analysis"] = analysis
                    except Exception as e:
                        st.error(f"AI 分析失敗：{e}")

        if "ov_ai_analysis" in st.session_state:
            st.markdown(
                f"<div style='background:#1A1D27;border:1px solid #2D3142;border-radius:10px;"
                f"padding:20px;margin-top:10px;color:#E8EAF0;line-height:1.7'>"
                f"{st.session_state['ov_ai_analysis'].replace(chr(10),'<br>')}"
                f"</div>",
                unsafe_allow_html=True,
            )


# ════════════════════════════════════════════════════════════════════
# PAGE: Risk Management (VaR + Backtest + Scenarios + Correlation)
# ════════════════════════════════════════════════════════════════════

def page_risk_management():
    st.title("⚠️ 風險管理")
    st.caption("VaR / CVaR · Monte Carlo · Kupiec 回測 · 壓力測試 · 相關性分析")

    with st.sidebar:
        st.markdown("### ⚠️ 風險設定")
        raw_tickers = st.text_input("Tickers（逗號分隔）", "AAPL,MSFT,GOOGL,AMZN", key="rm_tickers")
        tickers = [t.strip().upper() for t in raw_tickers.split(",") if t.strip()]
        default_w = ",".join([f"{1/max(len(tickers),1):.4f}"] * len(tickers)) if tickers else "1.0"
        raw_w = st.text_input("權重（自動正規化）", default_w, key="rm_weights")
        start = st.date_input("開始日期", value=date(2020, 1, 1), key="rm_start")
        alpha = st.slider("信心水準", 0.80, 0.99, 0.95, 0.01, key="rm_alpha")
        hold_days = st.number_input("持有天數", 1, 60, 1, key="rm_hold")
        window = st.number_input("回顧窗口（天）", 60, 2000, 252, key="rm_window")
        cov_method = st.selectbox(
            "共變異數方法", ["hist", "ewma", "lw"],
            format_func=lambda x: {"hist": "Historical", "ewma": "EWMA", "lw": "Ledoit-Wolf"}[x],
            key="rm_cov",
        )
        lam = st.slider("λ (EWMA)", 0.80, 0.99, 0.94, 0.01, key="rm_lam")
        notional = st.number_input("名目金額 (USD)", 1_000.0, 1e9, 100_000.0, step=1_000.0, key="rm_notional")

    if not tickers:
        st.info("請在側欄填入股票代碼。")
        return

    try:
        ws = np.array([float(x.strip()) for x in raw_w.split(",") if x.strip()])
        ws = ws / ws.sum()
        if len(ws) != len(tickers):
            raise ValueError
    except Exception:
        ws = np.repeat(1 / len(tickers), len(tickers))

    with st.spinner("載入市場數據…"):
        try:
            px_df = load_price_data(tickers, start=str(start))
        except Exception as e:
            st.error(f"資料載入失敗：{e}")
            return

    w = pd.Series(ws, index=px_df.columns)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 VaR / CVaR", "🔁 Kupiec 回測", "💥 壓力測試", "🔗 相關性分析",
    ])

    # ── Tab 1: VaR / CVaR ─────────────────────────────────────────
    with tab1:
        with st.spinner("計算風險指標…"):
            res = portfolio_var(
                px_df, w, alpha=alpha, hold_days=int(hold_days),
                window=int(window), cov_method=cov_method, lam=lam,
                as_of_value=float(notional),
            )
        section("投資組合風險指標")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: metric_card("VaR (%)",     f"{res.var_pct:.3%}",  positive=False)
        with c2: metric_card("CVaR (%)",    f"{res.cvar_pct:.3%}", positive=False)
        with c3: metric_card("年化波動",     f"{res.vol_ann:.3%}")
        with c4: metric_card("VaR (USD)",   f"${res.value_var:,.0f}",  positive=False)
        with c5: metric_card("CVaR (USD)",  f"${res.value_cvar:,.0f}", positive=False)

        ct1, ct2 = st.tabs(["共變異數矩陣", "Monte Carlo 模擬"])
        with ct1:
            fig = px.imshow(res.cov, text_auto=".4f", color_continuous_scale="Blues", aspect="auto")
            fig.update_layout(**PLOTLY_LAYOUT, title="Covariance Matrix", height=400)
            st.plotly_chart(fig, use_container_width=True)
        with ct2:
            do_mc = st.checkbox("執行 Monte Carlo 模擬", value=True, key="rm_mc")
            n_mc = st.select_slider("模擬路徑數", [1000, 5000, 10000, 50000], value=10000, key="rm_nmc")
            if do_mc:
                with st.spinner(f"執行 {n_mc:,} 條路徑…"):
                    pnl = mc_portfolio_pnl(
                        px_df, w, days=int(hold_days), n=int(n_mc),
                        cov_method=cov_method, lam=lam, window=int(window),
                    )
                mc_var = np.percentile(pnl, (1 - alpha) * 100)
                mc_cvar = float(pnl[pnl <= mc_var].mean())
                mc1, mc2 = st.columns(2)
                with mc1: metric_card("MC VaR (USD)",  f"${-mc_var:,.0f}",  positive=False)
                with mc2: metric_card("MC CVaR (USD)", f"${-mc_cvar:,.0f}", positive=False)
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=pnl, nbinsx=80, marker_color="#1E88E5", opacity=0.8, name="P&L"))
                fig.add_vline(x=mc_var,  line_color="#F44336", line_dash="dash", annotation_text=f"VaR {mc_var:,.0f}")
                fig.add_vline(x=mc_cvar, line_color="#FF9800", line_dash="dot",  annotation_text=f"CVaR {mc_cvar:,.0f}")
                fig.update_layout(**PLOTLY_LAYOUT, title=f"MC P&L Distribution ({n_mc:,} paths)", height=420)
                st.plotly_chart(fig, use_container_width=True)

    # ── Tab 2: Kupiec Backtest ─────────────────────────────────────
    with tab2:
        bt_window = st.number_input("回測滾動窗口（天）", 60, 2000, 250, key="rm_bt_w")
        if st.button("執行回測", type="primary", key="rm_run_bt"):
            with st.spinner("計算滾動 VaR…"):
                try:
                    var_series = rolling_portfolio_var(
                        px_df, w, alpha=alpha, window=int(bt_window),
                        cov_method=cov_method, lam=lam,
                    )
                    port_ret = (px_df.pct_change().dropna() @ w).reindex(var_series.index)
                    kup = kupiec_pof_test(port_ret, var_series, alpha)
                    st.session_state["rm_kupiec"] = (kup, port_ret, var_series)
                except Exception as e:
                    st.error(f"回測失敗：{e}")

        if st.session_state.get("rm_kupiec"):
            kup, port_ret, var_series = st.session_state["rm_kupiec"]
            p_ok = (kup.p_value > 0.05) if pd.notna(kup.p_value) else None
            k1, k2, k3, k4 = st.columns(4)
            with k1: metric_card("例外次數", str(kup.exceptions))
            with k2: metric_card("預期次數", f"{kup.expected:.1f}")
            with k3: metric_card("例外比率", f"{kup.ratio:.4f}")
            with k4: metric_card("p-value",  f"{kup.p_value:.4f}" if pd.notna(kup.p_value) else "N/A", positive=p_ok)
            if pd.notna(kup.p_value):
                if kup.p_value > 0.05:
                    st.success("✅ 模型未被拒絕（p > 0.05），VaR 模型有效")
                else:
                    st.warning("⚠️ 模型被拒絕（p ≤ 0.05），建議重新校準")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=port_ret.index, y=port_ret.values, name="組合報酬",
                                     line=dict(color="#1E88E5", width=1), opacity=0.7))
            fig.add_trace(go.Scatter(x=var_series.index, y=var_series.values,
                                     name=f"VaR ({alpha:.0%})", line=dict(color="#F44336", width=2)))
            exc = port_ret[port_ret < var_series.reindex(port_ret.index)]
            fig.add_trace(go.Scatter(x=exc.index, y=exc.values, mode="markers", name="例外",
                                     marker=dict(color="#FF9800", size=8, symbol="x")))
            fig.update_layout(**PLOTLY_LAYOUT, title="滾動 VaR 回測", height=450, yaxis_title="日報酬")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("點擊「執行回測」開始計算。")

    # ── Tab 3: Stress Testing ──────────────────────────────────────
    with tab3:
        data_min = px_df.index.min().date()
        data_max = px_df.index.max().date()
        st.caption(f"數據範圍：{data_min} → {data_max}")
        st3a, st3b, st3c = st.tabs(["自訂情境", "歷史重播", "預設壓力"])

        with st3a:
            section("逐資產衝擊設定")
            shock_df = pd.DataFrame({"Ticker": list(w.index), "Shock (%)": [0.0] * len(w)})
            edited = st.data_editor(shock_df, use_container_width=True, key="rm_shock")
            shocks = {row["Ticker"]: float(row.get("Shock (%)", 0.0)) / 100.0 for _, row in edited.iterrows()}
            if st.button("執行情境", type="primary", key="rm_run_scn"):
                sv = np.array([shocks.get(t, 0.0) for t in w.index])
                port_r_scn = float(np.dot(w.values, sv))
                pnl_scn = scenario_pnl_value(w, shocks, float(notional))
                s1, s2, s3 = st.columns(3)
                with s1: metric_card("組合報酬",  f"{port_r_scn:.2%}", positive=port_r_scn >= 0)
                with s2: metric_card("P&L (USD)", f"${pnl_scn:,.0f}",  positive=pnl_scn >= 0)
                with s3: metric_card("新組合價值", f"${notional*(1+port_r_scn):,.0f}", positive=port_r_scn >= 0)
                contrib = pd.DataFrame({
                    "Weight %": w.values * 100,
                    "Shock %":  sv * 100,
                    "Contribution (bp)": w.values * sv * 10_000,
                }, index=w.index)
                fig = px.bar(contrib.reset_index(), x="index", y="Contribution (bp)",
                             color="Contribution (bp)", color_continuous_scale="RdYlGn",
                             template="plotly_dark", text_auto=".1f")
                fig.update_layout(**PLOTLY_LAYOUT, title="各資產貢獻度（基點）", height=350)
                st.plotly_chart(fig, use_container_width=True)

        with st3b:
            section("歷史期間重播")
            rb1, rb2 = st.columns(2)
            with rb1:
                sdate = st.date_input("起始日", value=max(data_min, date(2020, 1, 1)),
                                      min_value=data_min, max_value=data_max, key="rm_rep_s")
            with rb2:
                edate = st.date_input("結束日", value=data_max,
                                      min_value=data_min, max_value=data_max, key="rm_rep_e")
            if st.button("執行重播", type="primary", key="rm_run_rep"):
                res_rep = historical_replay(px_df, w, str(sdate), str(edate), notional=float(notional))
                r1, r2, r3, r4 = st.columns(4)
                with r1: metric_card("區間報酬",  f"{res_rep.get('Return',0):.2%}", positive=res_rep.get('Return',0)>=0)
                with r2: metric_card("P&L (USD)", f"${res_rep.get('PnL',0):,.0f}",  positive=res_rep.get('PnL',0)>=0)
                with r3: metric_card("最大回撤",  f"{res_rep.get('MaxDD',0):.2%}",  positive=False)
                with r4: metric_card("使用天數",  str(int(res_rep.get('Rows',0))))

        with st3c:
            section("預設壓力情境")
            SCENARIOS = {
                "COVID 崩盤 (Mar 2020, −35%)":    {t: -0.35 for t in tickers},
                "金融海嘯 (GFC 2008, −40%)":       {t: -0.40 for t in tickers},
                "科技股大跌 (−20%)":               {t: -0.20 for t in ["AAPL","MSFT","GOOGL","NVDA","META","AMZN"] if t in tickers},
                "暴力升息 (+200bp，債券 −10%)":     {t: -0.10 for t in ["AGG","BND","TLT","MBB","SHY"] if t in tickers},
                "溫和多頭 (+10%)":                 {t:  0.10 for t in tickers},
                "半導體循環下行 (−25%)":            {t: -0.25 for t in ["NVDA","AMD","AVGO","QCOM","AMAT","SMH","SOXX"] if t in tickers},
            }
            results_scn = []
            for sc_name, sc_shocks in SCENARIOS.items():
                sv2 = np.array([sc_shocks.get(t, 0.0) for t in w.index])
                r_val = float(np.dot(w.values, sv2))
                results_scn.append({"情境": sc_name, "報酬": r_val, "P&L (USD)": float(notional) * r_val})
            scn_df = pd.DataFrame(results_scn)
            fig = px.bar(scn_df, x="情境", y="P&L (USD)", color="P&L (USD)",
                         color_continuous_scale="RdYlGn", template="plotly_dark", text_auto=",.0f")
            fig.update_layout(**PLOTLY_LAYOUT, title="壓力情境彙總", height=380)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(scn_df.style.format({"報酬": "{:.2%}", "P&L (USD)": "${:,.0f}"}),
                         use_container_width=True)

    # ── Tab 4: Correlation & Beta ──────────────────────────────────
    with tab4:
        r_df = px_df.pct_change().dropna()
        roll_w_c = st.number_input("滾動窗口（天）", 30, 1000, 126, key="rm_rollw")
        bench_c  = st.text_input("基準代碼", "^GSPC", key="rm_bench")
        ct_a, ct_b, ct_c, ct_d = st.tabs(["相關係數矩陣", "滾動 Beta", "散佈矩陣", "⚖️ 風險平價配置"])

        with ct_a:
            corr = r_df.corr()
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                            zmin=-1, zmax=1, aspect="auto")
            fig.update_layout(**PLOTLY_LAYOUT, title="Pairwise Correlation", height=450)
            st.plotly_chart(fig, use_container_width=True)

        with ct_d:
            section("等風險貢獻配置（Risk Parity / ERC）")
            st.caption(
                "讓每檔標的對組合的風險貢獻相等，而非等權重。"
                "高波動標的自動減碼，整體分散更佳（學習自 Riskfolio-Lib）。"
            )
            try:
                import quant_tools as _qt
            except ImportError:
                _qt = None
                st.error("找不到 quant_tools.py，請確認已同步。")
            if _qt is not None and r_df.shape[1] >= 2:
                cov_ann = (r_df.cov() * 252).values
                tickers_rp = list(r_df.columns)
                w_erc = _qt.risk_parity_weights(cov_ann)
                w_iv  = _qt.inverse_vol_weights(np.sqrt(np.diag(cov_ann)))
                w_eq  = np.full(len(tickers_rp), 1.0 / len(tickers_rp))
                rc_erc = _qt.risk_contributions(w_erc, cov_ann)
                rc_eq  = _qt.risk_contributions(w_eq, cov_ann)

                alloc = pd.DataFrame({
                    "標的": tickers_rp,
                    "等風險權重": w_erc,
                    "反波動權重": w_iv,
                    "等權重": w_eq,
                    "風險貢獻%(ERC)": rc_erc / rc_erc.sum() if rc_erc.sum() else w_eq,
                    "風險貢獻%(等權)": rc_eq / rc_eq.sum() if rc_eq.sum() else w_eq,
                }).set_index("標的")

                st.dataframe(
                    alloc.style.format({
                        "等風險權重": "{:.1%}", "反波動權重": "{:.1%}", "等權重": "{:.1%}",
                        "風險貢獻%(ERC)": "{:.1%}", "風險貢獻%(等權)": "{:.1%}",
                    }),
                    use_container_width=True,
                )

                fig_rp = go.Figure()
                fig_rp.add_trace(go.Bar(x=tickers_rp, y=w_erc, name="等風險權重",
                                        marker_color="#1E88E5"))
                fig_rp.add_trace(go.Bar(x=tickers_rp, y=w_eq, name="等權重",
                                        marker_color="#888"))
                fig_rp.update_layout(**PLOTLY_LAYOUT, height=380, barmode="group",
                                     title="權重比較：風險平價 vs 等權")
                st.plotly_chart(fig_rp, use_container_width=True)

                st.info(
                    "💡 **等權重**下，高波動標的會貢獻過多風險（看「風險貢獻%(等權)」欄差異）；"
                    "**等風險貢獻**讓每檔貢獻接近一致，降低單一標的主導組合波動。"
                )
            elif _qt is not None:
                st.warning("請在 Tickers 加入至少 2 檔標的。")

        with ct_b:
            if bench_c in r_df.columns:
                fig = go.Figure()
                for t in [x for x in r_df.columns if x != bench_c]:
                    df_b = pd.concat([r_df[t], r_df[bench_c]], axis=1).dropna()
                    beta_roll = (
                        df_b[t].rolling(int(roll_w_c)).cov(df_b[bench_c])
                        / df_b[bench_c].rolling(int(roll_w_c)).var()
                    )
                    fig.add_trace(go.Scatter(x=beta_roll.index, y=beta_roll.values,
                                             name=t, mode="lines"))
                fig.add_hline(y=1, line_dash="dot", line_color="gray")
                fig.update_layout(**PLOTLY_LAYOUT, title=f"Rolling Beta vs {bench_c}", height=420)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"請在 Tickers 中加入 {bench_c}")

        with ct_c:
            show_c = r_df.columns[:6].tolist()
            fig = px.scatter_matrix(r_df[show_c], dimensions=show_c, template="plotly_dark")
            fig.update_layout(**PLOTLY_LAYOUT, height=600, title="Pairwise Return Scatter")
            st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════
# PAGE: Stock Research (Screener + Individual AI Deep Dive)
# ════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def _cached_ticker_data(ticker: str, period: str):
    """快取個股 info + 歷史 + 新聞（5 分鐘），避免每次改滑桿都重抓。"""
    import yfinance as yf
    tk = yf.Ticker(ticker)
    try:
        info = tk.info or {}
    except Exception:
        info = {}
    try:
        hist = tk.history(period=period, auto_adjust=True)
    except Exception:
        hist = pd.DataFrame()
    try:
        news = tk.news or []
    except Exception:
        news = []
    return info, hist, news


def page_stock_research():
    import yfinance as yf

    st.title("🔍 股票研究")
    st.caption("個股深度分析 · K 線 · RSI · AI 研究報告 · 市場篩選器")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 個股深度分析", "🔎 市場篩選器", "🧪 訊號回測", "📺 TradingView"])

    # ── Tab 4: TradingView 互動圖表（免費嵌入 widget，無需 key）──────
    with tab4:
        section("TradingView 專業互動圖表")
        st.caption("縮放、畫線、數百種指標；直接嵌入 TradingView 免費 widget。")
        tv_in = st.text_input("代碼（美股直接打 AAPL；台股 2330.TW；港股 0700.HK）",
                              "AAPL", key="tv_sym").upper().strip()

        def _tv_symbol(t: str) -> str:
            if t.endswith(".TW"):
                return "TWSE:" + t[:-3]
            if t.endswith(".TWO"):
                return "TPEX:" + t[:-4]
            if t.endswith(".HK"):
                return "HKEX:" + t[:-3].lstrip("0")
            if t.endswith(".T"):
                return "TSE:" + t[:-2]
            return t   # 美股 TradingView 會自動解析

        tv_sym = _tv_symbol(tv_in) if tv_in else "AAPL"
        html = """
        <div class="tradingview-widget-container">
          <div id="tv_chart"></div>
          <script src="https://s3.tradingview.com/tv.js"></script>
          <script>
          new TradingView.widget({
            "autosize": true, "symbol": "%s", "interval": "D",
            "timezone": "Asia/Taipei", "theme": "dark", "style": "1",
            "locale": "zh_TW", "toolbar_bg": "#0F1117",
            "enable_publishing": false, "allow_symbol_change": true,
            "hide_side_toolbar": false, "studies": ["RSI@tv-basicstudies","MASimple@tv-basicstudies"],
            "container_id": "tv_chart"
          });
          </script>
        </div>
        """ % tv_sym
        import streamlit.components.v1 as _components
        _components.html(html, height=600)
        st.caption(f"目前顯示：{tv_sym}　·　圖內可直接改代碼/週期/指標。資料由 TradingView 提供。")

    # ── Tab 1: Individual deep dive ────────────────────────────────
    with tab1:
        ci1, ci2 = st.columns([3, 1])
        with ci1:
            ticker_r = st.text_input(
                "輸入股票代碼", "AAPL",
                placeholder="AAPL / MSFT / 2330.TW / 0700.HK",
            ).upper().strip()
        with ci2:
            period_r = st.selectbox("分析期間", ["6mo", "1y", "2y", "5y", "max"], index=1)

        if not ticker_r:
            st.info("請輸入股票代碼。")
        else:
            with st.spinner(f"載入 {ticker_r} …"):
                try:
                    info, hist, news_data = _cached_ticker_data(ticker_r, period_r)
                except Exception as e:
                    st.error(f"資料載入失敗：{e}")
                    info, hist, news_data = {}, pd.DataFrame(), []

            if hist.empty:
                st.error(f"找不到 {ticker_r} 的歷史資料，請確認代碼。")
            else:
                name_r   = info.get("longName") or info.get("shortName") or ticker_r
                sector_r = info.get("sector", "")
                ind_r    = info.get("industry", "")
                st.markdown(f"## {name_r}")
                st.caption("  |  ".join(filter(None, [ticker_r, sector_r, ind_r])))

                # Key metrics row
                section("關鍵指標")
                mkt_cap = info.get("marketCap", 0)
                metrics_r = [
                    ("現價",     f"${float(hist['Close'].iloc[-1]):.2f}"),
                    ("市值",     f"${mkt_cap/1e9:.1f}B" if mkt_cap else "—"),
                    ("P/E",      f"{info['trailingPE']:.1f}x"    if info.get("trailingPE")       else "—"),
                    ("EPS",      f"${info['trailingEps']:.2f}"   if info.get("trailingEps")      else "—"),
                    ("殖利率",   f"{info['dividendYield']*100:.2f}%" if info.get("dividendYield") else "—"),
                    ("52W High", f"${info['fiftyTwoWeekHigh']:.2f}" if info.get("fiftyTwoWeekHigh") else "—"),
                    ("52W Low",  f"${info['fiftyTwoWeekLow']:.2f}"  if info.get("fiftyTwoWeekLow")  else "—"),
                    ("Beta",     f"{info['beta']:.2f}"            if info.get("beta")             else "—"),
                ]
                for row_m in [metrics_r[:4], metrics_r[4:]]:
                    cols_m = st.columns(len(row_m))
                    for col_m, (lbl, val) in zip(cols_m, row_m):
                        with col_m:
                            metric_card(lbl, val)

                # Price chart (candlestick + MAs + volume)
                section("K 線走勢")
                chart_type = st.radio("圖表類型", ["K線圖", "收盤線"], horizontal=True, key="res_ct")
                fig_k = make_subplots(rows=2, cols=1, row_heights=[0.75, 0.25],
                                      shared_xaxes=True, vertical_spacing=0.02)
                if chart_type == "K線圖":
                    fig_k.add_trace(go.Candlestick(
                        x=hist.index, open=hist["Open"], high=hist["High"],
                        low=hist["Low"], close=hist["Close"], name="Price",
                        increasing_line_color="#4CAF50", decreasing_line_color="#F44336",
                    ), row=1, col=1)
                else:
                    fig_k.add_trace(go.Scatter(x=hist.index, y=hist["Close"],
                                               name="Close", line=dict(color="#1E88E5", width=2)),
                                    row=1, col=1)
                for ma_d, c_ma in [(20, "#FF9800"), (50, "#9C27B0"), (200, "#E91E63")]:
                    if len(hist) >= ma_d:
                        ma_s = hist["Close"].rolling(ma_d).mean()
                        fig_k.add_trace(go.Scatter(x=ma_s.index, y=ma_s.values,
                                                   name=f"MA{ma_d}",
                                                   line=dict(color=c_ma, width=1.5, dash="dot"),
                                                   opacity=0.8), row=1, col=1)
                fig_k.add_trace(go.Bar(x=hist.index, y=hist["Volume"], name="Volume",
                                       marker_color="#1E88E5", opacity=0.35), row=2, col=1)
                fig_k.update_layout(**PLOTLY_LAYOUT, height=520, showlegend=True,
                                    xaxis_rangeslider_visible=False)
                fig_k.update_yaxes(title_text="Price", row=1, col=1)
                fig_k.update_yaxes(title_text="Volume", row=2, col=1)
                st.plotly_chart(fig_k, use_container_width=True)

                # Returns stats
                section("報酬與風險統計")
                rets_r   = hist["Close"].pct_change().dropna()
                ann_vol_r = float(rets_r.std() * np.sqrt(252))
                sharpe_r  = float(rets_r.mean() / rets_r.std() * np.sqrt(252)) if rets_r.std() > 0 else 0
                max_dd_r  = float((hist["Close"] / hist["Close"].cummax() - 1).min())
                total_r   = float(hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1)
                rs1, rs2, rs3, rs4 = st.columns(4)
                with rs1: metric_card("區間總報酬", f"{total_r:.2%}",    positive=total_r >= 0)
                with rs2: metric_card("年化波動",   f"{ann_vol_r:.2%}")
                with rs3: metric_card("Sharpe",     f"{sharpe_r:.2f}",   positive=sharpe_r > 1)
                with rs4: metric_card("最大回撤",   f"{max_dd_r:.2%}",   positive=False)

                # RSI
                section("動能指標 — RSI (14)")
                gain_r = rets_r.clip(lower=0)
                loss_r = (-rets_r).clip(lower=0)
                rs_val = gain_r.rolling(14).mean() / loss_r.rolling(14).mean().replace(0, np.nan)
                rsi_s  = 100 - 100 / (1 + rs_val)
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=rsi_s.index, y=rsi_s.values,
                                             name="RSI(14)", line=dict(color="#1E88E5", width=2)))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="#F44336", annotation_text="超買 70")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="#4CAF50", annotation_text="超賣 30")
                fig_rsi.update_layout(**PLOTLY_LAYOUT, height=260,
                                      title="RSI (14 天)", yaxis_range=[0, 100])
                st.plotly_chart(fig_rsi, use_container_width=True)
                cur_rsi = float(rsi_s.dropna().iloc[-1]) if not rsi_s.dropna().empty else 50
                if cur_rsi > 70:
                    st.warning(f"⚠️ RSI = {cur_rsi:.1f}，超買區間，注意回檔風險")
                elif cur_rsi < 30:
                    st.success(f"💡 RSI = {cur_rsi:.1f}，超賣區間，可能存在反彈機會")
                else:
                    st.info(f"RSI = {cur_rsi:.1f}，動能中性")

                # News
                section("近期新聞")
                try:
                    news_items = news_data or []
                    for item in news_items[:6]:
                        from datetime import datetime as _dt
                        pub_t   = item.get("providerPublishTime", 0)
                        pub_str = _dt.fromtimestamp(pub_t).strftime("%Y-%m-%d %H:%M") if pub_t else ""
                        st.markdown(
                            f"**[{item.get('title','')}]({item.get('link','#')})**  \n"
                            f"<small style='color:#B8C0D0'>{item.get('publisher','')} · {pub_str}</small>",
                            unsafe_allow_html=True,
                        )
                        st.markdown("---")
                except Exception:
                    st.info("無法載入新聞。")

                # AI deep dive
                section("AI 深度研究報告")
                with st.expander("使用 LLM 生成完整研究報告", expanded=False):
                    ak1, ak2 = st.columns([2, 1])
                    with ak1:
                        ai_key_r = st.text_input("API Key", type="password", key="res_key")
                    with ak2:
                        ai_model_r = st.selectbox("模型", [
                            "gpt-4o-mini", "gpt-4o",
                            "claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022",
                        ], key="res_model")
                    ai_base_r = st.text_input("API Base URL（空白自動偵測）", "", key="res_base")

                    if st.button("生成研究報告", type="primary", key="res_gen"):
                        if not ai_key_r:
                            st.error("請填入 API Key")
                        else:
                            metrics_ctx = "\n".join(f"- {l}: {v}" for l, v in metrics_r)
                            news_ctx = ""
                            try:
                                for ni in (news_data or [])[:5]:
                                    news_ctx += f"- {ni.get('title','')}\n"
                            except Exception:
                                pass
                            prompt_r = (
                                f"你是頂尖機構投資分析師，請對下列股票以繁體中文撰寫完整研究報告：\n\n"
                                f"股票：{name_r} ({ticker_r})\n"
                                f"產業：{sector_r} — {ind_r}\n\n"
                                f"關鍵財務指標：\n{metrics_ctx}\n\n"
                                f"技術面：區間報酬 {total_r:.2%} · 年化波動 {ann_vol_r:.2%} · "
                                f"Sharpe {sharpe_r:.2f} · 最大回撤 {max_dd_r:.2%} · RSI {cur_rsi:.1f}\n\n"
                                f"近期新聞：\n{news_ctx}\n\n"
                                "報告結構：\n"
                                "1. 📊 公司概況與核心競爭優勢\n"
                                "2. 💹 財務健康度評估\n"
                                "3. 📈 技術面分析與趨勢判斷\n"
                                "4. 🌍 產業趨勢與競爭格局\n"
                                "5. ⚠️ 主要風險因素（3-5點）\n"
                                "6. 🎯 投資建議（買進/持有/減碼）與目標價位思路\n"
                                "7. 📅 短期(1-3月) / 中期(3-12月) / 長期(1年+) 展望"
                            )
                            with st.spinner("AI 分析中…"):
                                try:
                                    client = _llm_client(ai_key_r, ai_base_r, ai_model_r)
                                    resp = client.chat.completions.create(
                                        model=ai_model_r,
                                        messages=[{"role": "user", "content": prompt_r}],
                                        temperature=0.3, max_tokens=2000,
                                    )
                                    st.session_state["res_report"] = resp.choices[0].message.content.strip()
                                except Exception as e:
                                    st.error(f"報告生成失敗：{e}")

                    if st.session_state.get("res_report"):
                        st.markdown("---")
                        st.markdown(st.session_state["res_report"])
                        st.download_button(
                            "⬇ 下載研究報告",
                            data=st.session_state["res_report"],
                            file_name=f"{ticker_r}_research_{date.today()}.txt",
                            mime="text/plain",
                        )

    # ── Tab 2: Market Screener ─────────────────────────────────────
    with tab2:
        section("市場篩選器")
        UNIVERSES = {
            "S&P 500 精選 (30檔)": [
                "AAPL","MSFT","NVDA","AMZN","META","GOOGL","BRK-B","LLY","AVGO","JPM",
                "UNH","XOM","V","TSLA","PG","MA","HD","COST","MRK","JNJ",
                "ABBV","BAC","CRM","NFLX","AMD","ACN","WMT","KO","PEP","T",
            ],
            "科技龍頭": [
                "AAPL","MSFT","NVDA","AMD","AVGO","QCOM","INTC","MU","AMAT","LRCX",
                "KLAC","META","GOOGL","AMZN","NFLX","CRM","NOW","SNOW","PLTR","NET",
                "ZS","PANW","DDOG","MDB","WDAY",
            ],
            "AI / 半導體": [
                "NVDA","AMD","AVGO","QCOM","AMAT","LRCX","KLAC","MU","ASML","TSM",
                "PLTR","AI","SOUN","MSFT","GOOGL","META","AMZN","ORCL","IBM","ARM",
            ],
            "高成長/動量": [
                "NVDA","TSLA","META","AMZN","NFLX","PLTR","COIN","MSTR","HOOD","SOFI",
                "RBLX","SHOP","SQ","PYPL","AFRM","UPST","DKNG","LYFT","ABNB","UBER",
            ],
            "台股半導體": [
                "2330.TW","2303.TW","2308.TW","2454.TW","2317.TW",
                "2382.TW","3711.TW","2376.TW","2049.TW","6770.TW",
            ],
            "台股寬基": [
                "2330.TW","2317.TW","2454.TW","2412.TW","2881.TW","2882.TW",
                "2886.TW","1301.TW","1303.TW","2002.TW","2308.TW","3711.TW",
                "2382.TW","2357.TW","4938.TW","6505.TW","2912.TW","2884.TW",
            ],
            "全球 ETF": [
                "SPY","QQQ","IWM","EEM","GLD","TLT","VNQ","SMH","SOXX",
                "EWJ","FXI","EWT","AGG","HYG","DBA","USO","XLK","XLF","XLE",
            ],
            "板塊 ETF": [
                "XLK","XLF","XLE","XLV","XLY","XLP","XLI","XLU","XLRE","XLB","XLC",
            ],
            "高股息": [
                "O","SCHD","DVY","VIG","JEPI","DIVO","T","VZ","KO","PEP",
                "JNJ","MO","XOM","CVX","IBM","MCD","PG","MMM","ABBV","WMT",
            ],
            "中概 ADR": [
                "BABA","JD","PDD","BIDU","NIO","XPEV","LI","TCOM","NTES","BILI",
                "VIPS","IQ","TIGR","FUTU","BOSS",
            ],
            "加密概念股": [
                "COIN","MSTR","MARA","RIOT","CLSK","HUT","BTBT","CIFR","BITF","SQ",
            ],
        }
        sc1, sc2 = st.columns([2, 1])
        with sc1:
            uni_choice = st.selectbox("股票池", list(UNIVERSES.keys()), key="sc_uni")
        with sc2:
            sc_period = st.selectbox("期間", ["3mo", "6mo", "1y"], index=2, key="sc_period")
        custom_sc = st.text_input("自訂清單（逗號分隔，覆蓋上方選擇）", "", key="sc_custom")
        screen_tickers = (
            [t.strip().upper() for t in custom_sc.split(",") if t.strip()]
            if custom_sc.strip()
            else UNIVERSES[uni_choice]
        )

        if st.button("開始篩選", type="primary", key="sc_run"):
            with st.spinner(f"篩選 {len(screen_tickers)} 檔，約需 15–30 秒…"):
                try:
                    rows_sc = []
                    for tkr_s in screen_tickers:
                        try:
                            _raw_s = yf.download(tkr_s, period=sc_period, auto_adjust=True, progress=False)
                            if _raw_s.empty:
                                continue
                            s_s = _raw_s["Close"].squeeze().dropna()
                        except Exception:
                            continue
                        if len(s_s) < 10:
                            continue
                        r_s = s_s.pct_change().dropna()
                        gain_s = r_s.clip(lower=0)
                        loss_s = (-r_s).clip(lower=0)
                        rs_s   = gain_s.rolling(14).mean() / loss_s.rolling(14).mean().replace(0, np.nan)
                        rsi_sv = float(100 - 100 / (1 + rs_s.dropna().iloc[-1])) if not rs_s.dropna().empty else np.nan
                        rows_sc.append({
                            "代碼":     tkr_s,
                            "現價":     float(s_s.iloc[-1]),
                            "1日%":    float(s_s.iloc[-1]/s_s.iloc[-2]-1) if len(s_s)>=2 else np.nan,
                            "1月%":    float(s_s.iloc[-1]/s_s.iloc[max(0,len(s_s)-22)]-1) if len(s_s)>=22 else np.nan,
                            "3月%":    float(s_s.iloc[-1]/s_s.iloc[max(0,len(s_s)-63)]-1) if len(s_s)>=63 else np.nan,
                            "區間%":   float(s_s.iloc[-1]/s_s.iloc[0]-1),
                            "年化波動": float(r_s.std()*np.sqrt(252)),
                            "Sharpe":  float(r_s.mean()/r_s.std()*np.sqrt(252)) if r_s.std()>0 else np.nan,
                            "最大回撤": float((s_s/s_s.cummax()-1).min()),
                            "RSI(14)": rsi_sv,
                        })
                    st.session_state["sc_result"] = pd.DataFrame(rows_sc).set_index("代碼")
                except Exception as e:
                    st.error(f"篩選失敗：{e}")

        if st.session_state.get("sc_result") is not None:
            df_sc = st.session_state["sc_result"]
            st.markdown("##### 條件篩選")
            sf1, sf2, sf3 = st.columns(3)
            with sf1:
                min_1m = st.slider("最低1月報酬%", -100, 100, -100, 5, key="sf_1m") / 100
            with sf2:
                max_v  = st.slider("最高年化波動%", 5, 200, 200, 5, key="sf_v") / 100
            with sf3:
                rsi_r  = st.slider("RSI 範圍", 0, 100, (0, 100), key="sf_rsi")
            sort_sc = st.selectbox("排序依據", ["3月%","1月%","Sharpe","RSI(14)","年化波動"], key="sf_sort")

            fdf = df_sc[
                (df_sc["1月%"].fillna(-999) >= min_1m) &
                (df_sc["年化波動"].fillna(999) <= max_v) &
                (df_sc["RSI(14)"].fillna(50) >= rsi_r[0]) &
                (df_sc["RSI(14)"].fillna(50) <= rsi_r[1])
            ].sort_values(sort_sc, ascending=False)

            st.caption(f"符合條件：{len(fdf)} / {len(df_sc)} 檔")

            def _cr(val):
                if isinstance(val, float) and not np.isnan(val):
                    return "color:#4CAF50" if val > 0 else "color:#F44336"
                return ""
            def _crsi(val):
                if isinstance(val, float) and not np.isnan(val):
                    if val > 70: return "color:#F44336"
                    if val < 30: return "color:#4CAF50"
                return ""

            fmt_sc = {
                "現價": "{:.2f}", "1日%": "{:.2%}", "1月%": "{:.2%}",
                "3月%": "{:.2%}", "區間%": "{:.2%}",
                "年化波動": "{:.2%}", "Sharpe": "{:.2f}",
                "最大回撤": "{:.2%}", "RSI(14)": "{:.1f}",
            }
            st.dataframe(
                fdf.style.format(fmt_sc, na_rep="—")
                   .map(_cr,   subset=["1日%","1月%","3月%","區間%"])
                   .map(_crsi, subset=["RSI(14)"]),
                use_container_width=True,
            )

            section("風險 vs 報酬 散佈圖")
            fig_sc = px.scatter(
                fdf.reset_index(), x="年化波動", y="3月%",
                hover_name="代碼", text="代碼",
                color="Sharpe", color_continuous_scale="RdYlGn",
                template="plotly_dark",
            )
            fig_sc.update_traces(textposition="top center", textfont_size=9)
            fig_sc.update_layout(
                **PLOTLY_LAYOUT, title="Risk vs Return (3M)", height=450,
                xaxis_title="Annual Volatility", yaxis_title="3-Month Return",
                xaxis_tickformat=".0%", yaxis_tickformat=".0%",
            )
            st.plotly_chart(fig_sc, use_container_width=True)
            st.download_button(
                "⬇ 下載篩選結果 CSV",
                data=fdf.reset_index().to_csv(index=False).encode("utf-8"),
                file_name=f"screener_{date.today()}.csv",
                mime="text/csv",
            )

    # ── Tab 3: Signal Backtest (Triple-Barrier) ────────────────────
    with tab3:
        section("訊號回測 — 三重關卡法 (Triple-Barrier)")
        st.caption(
            "每當訊號觸發，往後看 N 天設停利/停損/到期三道關卡，先碰到哪個決定勝負。"
            "方法源自 López de Prado《Advances in Financial ML》與實證量化研究。"
        )

        bt_c1, bt_c2, bt_c3, bt_c4 = st.columns(4)
        with bt_c1:
            bt_ticker = st.text_input("股票代碼", "AAPL", key="bt_tk").upper().strip()
        with bt_c2:
            bt_period = st.selectbox("回測期間", ["1y", "2y", "5y", "max"], index=1, key="bt_pd")
        with bt_c3:
            bt_tp = st.slider("停利 %", 1, 20, 5, key="bt_tp") / 100
        with bt_c4:
            bt_sl = st.slider("停損 %", 1, 15, 3, key="bt_sl") / 100
        bt_cc1, bt_cc2 = st.columns(2)
        with bt_cc1:
            bt_h = st.slider("持有上限（交易日）", 3, 30, 10, key="bt_h")
        with bt_cc2:
            bt_cost = st.slider("來回交易成本 ‰（手續費+滑價）", 0, 30, 10, key="bt_cost") / 1000

        bt_mtf = st.checkbox("加入週線確認（MTF）並對照效果", value=False, key="bt_mtf",
                             help="只保留週線同向的進場，並對照有/無 MTF 的勝率變化")

        st.markdown(
            "<small style='color:#B8C0D0'>💡 停利/停損比建議 ≥ 1.5:1（如停利5%/停損3%）。"
            "已採用<b>下一根進場</b>（消除前視偏誤）+ <b>扣交易成本</b> + "
            "<b>樣本外一致性</b>檢測（防過擬合）。交易數 <5 的規則統計上不可靠。</small>",
            unsafe_allow_html=True,
        )

        if st.button("🧪 執行回測", type="primary", key="bt_run"):
            try:
                import backtest as _bt
            except ImportError:
                st.error("找不到 backtest.py，請確認已同步該檔案（Colab 需更新 Cell 2）。")
                _bt = None

            if _bt and bt_ticker:
                with st.spinner(f"回測 {bt_ticker} 全部 9 種訊號規則…"):
                    try:
                        raw_bt = yf.download(bt_ticker, period=bt_period,
                                             auto_adjust=True, progress=False)
                        if raw_bt.empty:
                            st.error("無資料，請確認代碼。")
                        else:
                            df_bt = _bt.normalize_ohlc(raw_bt, bt_ticker)
                            res = _bt.backtest_all(df_bt, tp=bt_tp, sl=bt_sl,
                                                   horizon=bt_h, cost=bt_cost,
                                                   mtf_filter=bt_mtf)
                            # 加上樣本外一致性（穩健度）欄
                            try:
                                wf = _bt.walk_forward(df_bt, tp=bt_tp, sl=bt_sl,
                                                      horizon=bt_h, cost=bt_cost)
                                res["穩健度"] = [wf.get(r, np.nan) for r in res.index]
                            except Exception:
                                res["穩健度"] = np.nan
                            # MTF 對照：另跑一次無 MTF，附上勝率/獲利因子變化
                            mtf_cmp = None
                            if bt_mtf:
                                try:
                                    base = _bt.backtest_all(df_bt, tp=bt_tp, sl=bt_sl,
                                                            horizon=bt_h, cost=bt_cost,
                                                            mtf_filter=False)
                                    valid_b = base[base["trades"] >= 5]
                                    valid_m = res[res["trades"] >= 5]
                                    mtf_cmp = {
                                        "base_wr": float(valid_b["win_rate"].mean()) if not valid_b.empty else np.nan,
                                        "mtf_wr":  float(valid_m["win_rate"].mean()) if not valid_m.empty else np.nan,
                                        "base_pf": float(valid_b["profit_factor"].replace(np.inf, np.nan).mean()) if not valid_b.empty else np.nan,
                                        "mtf_pf":  float(valid_m["profit_factor"].replace(np.inf, np.nan).mean()) if not valid_m.empty else np.nan,
                                        "base_trades": int(base["trades"].sum()),
                                        "mtf_trades":  int(res["trades"].sum()),
                                    }
                                except Exception:
                                    mtf_cmp = None
                            st.session_state["bt_result"] = res
                            st.session_state["bt_mtf_cmp"] = mtf_cmp
                            st.session_state["bt_meta"] = (bt_ticker, bt_tp, bt_sl, bt_h)
                    except Exception as e:
                        st.error(f"回測失敗：{e}")

        if st.session_state.get("bt_result") is not None:
            res = st.session_state["bt_result"]
            mtk, mtp, msl, mh = st.session_state.get("bt_meta", (bt_ticker, bt_tp, bt_sl, bt_h))
            st.markdown(f"#### {mtk} · 停利{mtp:.0%} / 停損{msl:.0%} / 持有≤{mh}日")

            cmp = st.session_state.get("bt_mtf_cmp")
            if cmp:
                d_wr = (cmp["mtf_wr"] - cmp["base_wr"]) if (np.isfinite(cmp["mtf_wr"]) and np.isfinite(cmp["base_wr"])) else np.nan
                d_pf = (cmp["mtf_pf"] - cmp["base_pf"]) if (np.isfinite(cmp["mtf_pf"]) and np.isfinite(cmp["base_pf"])) else np.nan
                verdict = ("MTF 有幫助 ✅" if (np.isfinite(d_wr) and d_wr > 0.01) else
                           "MTF 影響不大 ➖" if (np.isfinite(d_wr) and abs(d_wr) <= 0.01) else
                           "MTF 反而變差 ⚠️")
                st.info(
                    f"**週線確認 (MTF) 對照**（僅計交易數≥5 的規則平均）\n\n"
                    f"勝率：{cmp['base_wr']:.0%} → **{cmp['mtf_wr']:.0%}**（{d_wr:+.0%}）　"
                    f"獲利因子：{cmp['base_pf']:.2f} → **{cmp['mtf_pf']:.2f}**（{d_pf:+.2f}）\n\n"
                    f"交易數：{cmp['base_trades']} → {cmp['mtf_trades']}（過濾掉週線背離的進場）　→ **{verdict}**"
                )

            disp = res.copy()
            disp_fmt = disp.rename(columns={
                "trades": "交易數", "win_rate": "勝率", "profit_factor": "獲利因子",
                "expectancy": "期望值/筆", "avg_win": "平均獲利", "avg_loss": "平均虧損",
                "avg_held": "平均持有", "total_ret": "累積報酬",
            })

            def _pf_color(v):
                if isinstance(v, float) and np.isfinite(v):
                    return "color:#4CAF50;font-weight:700" if v >= 1.5 else (
                        "color:#FFC107" if v >= 1.0 else "color:#F44336")
                return "color:#E8EAF0"
            def _wr_color(v):
                if isinstance(v, float) and not np.isnan(v):
                    return "color:#4CAF50" if v >= 0.55 else (
                        "color:#FFC107" if v >= 0.45 else "color:#F44336")
                return ""
            def _rob_color(v):
                if isinstance(v, float) and not np.isnan(v):
                    return "color:#4CAF50" if v >= 0.6 else (
                        "color:#FFC107" if v >= 0.4 else "color:#F44336")
                return ""

            _fmt = {
                "勝率": "{:.1%}", "獲利因子": "{:.2f}", "期望值/筆": "{:+.2%}",
                "平均獲利": "{:+.2%}", "平均虧損": "{:+.2%}",
                "平均持有": "{:.1f}", "累積報酬": "{:+.1%}",
            }
            _styler = disp_fmt.style
            if "穩健度" in disp_fmt.columns:
                _fmt["穩健度"] = "{:.0%}"
            _styler = _styler.format(_fmt, na_rep="—") \
                             .map(_pf_color, subset=["獲利因子"]) \
                             .map(_wr_color, subset=["勝率"])
            if "穩健度" in disp_fmt.columns:
                _styler = _styler.map(_rob_color, subset=["穩健度"])
            st.dataframe(_styler, use_container_width=True)

            # Best rule callout
            valid = res[res["trades"] >= 5].copy()
            if not valid.empty and np.isfinite(valid["profit_factor"].iloc[0]):
                best = valid.iloc[0]
                st.success(
                    f"🏆 最佳訊號：**{valid.index[0]}** — "
                    f"勝率 {best['win_rate']:.0%}，獲利因子 {best['profit_factor']:.2f}，"
                    f"{int(best['trades'])} 筆交易，期望值每筆 {best['expectancy']:+.2%}"
                )

            # Profit factor bar chart
            chart_df = valid.reset_index() if not valid.empty else res.reset_index()
            chart_df = chart_df[np.isfinite(chart_df["profit_factor"])]
            if not chart_df.empty:
                fig_bt = px.bar(
                    chart_df.sort_values("profit_factor"),
                    x="profit_factor", y="rule", orientation="h",
                    color="win_rate", color_continuous_scale="RdYlGn",
                    color_continuous_midpoint=0.5,
                    labels={"profit_factor": "獲利因子", "rule": "", "win_rate": "勝率"},
                )
                fig_bt.add_vline(x=1.0, line_dash="dash", line_color="#888")
                fig_bt.update_layout(**PLOTLY_LAYOUT, height=400,
                                     title="各訊號獲利因子（虛線=1.0 損益兩平）")
                st.plotly_chart(fig_bt, use_container_width=True)

            st.markdown(
                "<small style='color:#B8C0D0'>"
                "📖 **獲利因子** = 總獲利/總虧損，>1.5 佳、>1.0 才賺錢。"
                "**期望值** = 每筆交易平均報酬（已扣成本），正值代表長期有利。"
                "**穩健度** = 樣本外一致性，把資料切成 4 段看每段是否都賺，"
                "<60% 代表只在某些時期有效（過擬合風險）。"
                "回測採下一根進場（無前視）並已扣交易成本，但仍為歷史模擬，"
                "僅供訊號相對強弱參考。</small>",
                unsafe_allow_html=True,
            )

        # ── Parameter optimizer (mini-hyperopt) ───────────────────
        section("⚙️ 參數最佳化（找最適停利/停損/持有）")
        st.caption(
            "網格搜尋 27 組 (停利×停損×持有) 參數，依「期望值×獲利因子×樣本外一致性」"
            "綜合評分排名，自動避開只在歷史過擬合的組合。約需 20-40 秒。"
        )
        if st.button("🔍 搜尋最佳參數", key="opt_run"):
            try:
                import backtest as _bt2
            except ImportError:
                _bt2 = None
                st.error("找不到 backtest.py。")
            opt_tkr = st.session_state.get("bt_tk", "AAPL").upper().strip()
            if _bt2 and opt_tkr:
                with st.spinner(f"最佳化 {opt_tkr} 參數（27 組）…"):
                    try:
                        raw_o = yf.download(opt_tkr, period="2y",
                                            auto_adjust=True, progress=False)
                        if raw_o.empty:
                            st.error("無資料。")
                        else:
                            df_o = _bt2.normalize_ohlc(raw_o, opt_tkr)
                            opt = _bt2.optimize_params(df_o)
                            st.session_state["opt_result"] = (opt_tkr, opt)
                    except Exception as e:
                        st.error(f"最佳化失敗：{e}")

        if st.session_state.get("opt_result") is not None:
            o_tkr, opt = st.session_state["opt_result"]
            if opt is None or opt.empty:
                st.info("找不到足夠交易數的參數組合。")
            else:
                best = opt.iloc[0]
                st.success(
                    f"🏆 {o_tkr} 最佳參數：停利 **{best['tp']:.0%}** / 停損 **{best['sl']:.0%}** / "
                    f"持有 **{int(best['horizon'])}日** · 規則「{best['best_rule']}」\n\n"
                    f"勝率 {best['win_rate']:.0%}，獲利因子 {best['profit_factor']:.2f}，"
                    f"期望值 {best['expectancy']:+.2%}/筆，穩健度 {best['consistency']:.0%}"
                )
                opt_disp = opt.head(10).rename(columns={
                    "tp": "停利", "sl": "停損", "horizon": "持有",
                    "best_rule": "最佳規則", "trades": "交易數", "win_rate": "勝率",
                    "profit_factor": "獲利因子", "expectancy": "期望值", "consistency": "穩健度",
                    "objective": "綜合分",
                })
                st.dataframe(
                    opt_disp.style.format({
                        "停利": "{:.0%}", "停損": "{:.0%}", "勝率": "{:.0%}",
                        "獲利因子": "{:.2f}", "期望值": "{:+.2%}", "穩健度": "{:.0%}",
                        "綜合分": "{:.4f}",
                    }),
                    use_container_width=True,
                )


# ════════════════════════════════════════════════════════════════════
# PAGE: Real-time Alerts & Monitoring
# ════════════════════════════════════════════════════════════════════

ALERTS_FILE = BASE_DIR / "alerts_config.json"


def _load_alerts_config() -> dict:
    import json as _json
    if ALERTS_FILE.exists():
        try:
            return _json.loads(ALERTS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "watchlist": ["AAPL", "MSFT", "NVDA", "2330.TW", "SPY"],
        "alerts":    [],
        "telegram":  {"enabled": False, "token": "", "chat_id": ""},
        "email":     {"enabled": False, "smtp": "smtp.gmail.com", "port": 465,
                      "user": "", "password": "", "to": ""},
    }


def _save_alerts_config(cfg: dict) -> None:
    import json as _json
    ALERTS_FILE.write_text(
        _json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _send_telegram(token: str, chat_id: str, message: str) -> tuple[bool, str]:
    import requests
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        r = requests.post(
            url,
            json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"},
            timeout=10,
        )
        return (r.ok, r.text if not r.ok else "OK")
    except Exception as e:
        return (False, str(e))


def _send_email(smtp_host: str, port: int, user: str, pwd: str,
                to_addr: str, subject: str, body: str) -> tuple[bool, str]:
    import smtplib
    from email.mime.text import MIMEText
    try:
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"] = user
        msg["To"] = to_addr
        with smtplib.SMTP_SSL(smtp_host, int(port), timeout=15) as srv:
            srv.login(user, pwd)
            srv.send_message(msg)
        return (True, "OK")
    except Exception as e:
        return (False, str(e))


# ════════════════════════════════════════════════════════════════════
# PAGE: Company Fundamental Analysis
# ════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def _cached_fundamentals(ticker: str):
    """快取基本面 + 下次財報日（1 小時；基本面本就低頻）。"""
    import fundamentals as fa
    try:
        ed = fa.next_earnings_date(ticker)
    except Exception:
        ed = None
    return fa.fetch_fundamentals(ticker), ed


def page_company_analysis():
    st.title("🏢 公司分析")
    st.caption("基本面體質分析 · 財務健康評分 · 估值 · 三表趨勢 · AI 解讀（資料來源 yfinance）")

    try:
        import fundamentals as fa
    except ImportError:
        st.error("找不到 fundamentals.py，請確認已同步（Colab 需更新 Cell 2）。")
        return

    ca1, ca2 = st.columns([3, 1])
    with ca1:
        fa_tkr = st.text_input("輸入股票代碼", "AAPL",
                               placeholder="AAPL / MSFT / 2330.TW").upper().strip()
    with ca2:
        run_fa = st.button("分析", type="primary", use_container_width=True)

    if run_fa and fa_tkr:
        with st.spinner(f"抓取 {fa_tkr} 基本面資料…"):
            d, ed = _cached_fundamentals(fa_tkr)
            st.session_state["fa_data"] = d
            st.session_state["fa_earnings"] = ed
        st.session_state.pop("fa_ai_out", None)   # 換股時清掉舊的 AI 解讀

    data = st.session_state.get("fa_data")
    if not data:
        st.info("輸入代碼後按「分析」。ETF / 指數無公司基本面資料。")
        return

    # 提醒：輸入框與目前顯示資料的標的不一致（尚未按分析）
    if fa_tkr and data.get("ticker") != fa_tkr:
        st.info(f"目前顯示 **{data.get('ticker')}** 的資料；輸入框為 **{fa_tkr}**，請按「分析」更新。")

    if not data.get("ok"):
        st.warning(f"⚠️ {data.get('ticker','')}：{data.get('error') or '無法取得基本面資料'}")
        return

    cur = data.get("currency") or ""

    # ① 公司概況卡 ─────────────────────────────────────────────
    section(f"{data['name']}（{data['ticker']}）")
    info_bits = [b for b in [data.get("sector"), data.get("industry")] if b]
    if info_bits:
        st.caption(" · ".join(info_bits))
    o1, o2, o3, o4 = st.columns(4)
    with o1:
        metric_card("股價", f"{data['price']:.2f} {cur}" if data.get("price") else "—")
    with o2:
        mc = data.get("market_cap")
        mc_s = (f"{mc/1e12:.2f}T" if mc and mc >= 1e12 else
                f"{mc/1e9:.1f}B" if mc and mc >= 1e9 else
                f"{mc/1e6:.0f}M" if mc else "—")
        metric_card("市值", mc_s)
    with o3:
        hi, lo, pr = data.get("high_52w"), data.get("low_52w"), data.get("price")
        pos = f"{(pr-lo)/(hi-lo)*100:.0f}%" if (hi and lo and pr and hi > lo) else "—"
        metric_card("52週位階", pos)
    with o4:
        dy = data.get("dividend_yield")
        metric_card("股利率", f"{dy:.2f}%" if dy else "—")

    # 下次財報日
    ed = st.session_state.get("fa_earnings")
    if ed is not None:
        try:
            import datetime as _dt
            days_to = (ed - _dt.date.today()).days
            when = ("已過" if days_to < 0 else "今天" if days_to == 0 else f"{days_to} 天後")
            tag = "🔴" if 0 <= days_to <= 7 else "📅"
            st.caption(f"{tag} 下次財報：**{ed.isoformat()}**（{when}）")
        except Exception:
            pass

    # ② 財務健康評分 ──────────────────────────────────────────
    hs = fa.health_score(data)
    section("財務健康評分")
    if hs["score"] is None:
        st.warning("基本面資料不足，無法評分。")
    else:
        sc1, sc2 = st.columns([1, 2])
        with sc1:
            color = ("#4CAF50" if hs["score"] >= 75 else "#8BC34A" if hs["score"] >= 55
                     else "#FFC107" if hs["score"] >= 40 else "#F44336")
            st.markdown(
                f"<div class='metric-card' style='text-align:center;border-color:{color}80'>"
                f"<div class='metric-label'>綜合評分（{hs['covered']}/5 構面）</div>"
                f"<div style='font-size:2.4rem;font-weight:800;color:{color}'>{hs['score']:.0f}</div>"
                f"<div style='color:{color};font-weight:700;font-size:1.1rem'>{hs['rating']}</div>"
                f"</div>", unsafe_allow_html=True,
            )
        with sc2:
            bd = {k: v for k, v in hs["breakdown"].items() if v is not None}
            if bd:
                cats = list(bd.keys()) + [list(bd.keys())[0]]
                vals = list(bd.values()) + [list(bd.values())[0]]
                fig_r = go.Figure(go.Scatterpolar(r=vals, theta=cats, fill="toself",
                                                   line_color="#1E88E5"))
                fig_r.update_layout(**PLOTLY_LAYOUT, height=300,
                                    polar=dict(radialaxis=dict(range=[0, 100], color="#E8EAF0"),
                                               bgcolor="#14172A"),
                                    title="各構面評分（0~100）")
                st.plotly_chart(fig_r, use_container_width=True)

    tab_v, tab_p, tab_s, tab_ai = st.tabs(["估值", "獲利能力", "財報三表", "🤖 AI 解讀"])

    # ③ 估值 ─────────────────────────────────────────────────
    with tab_v:
        flags = fa.valuation_flags(data)
        flag_emoji = {"cheap": "🟢 偏便宜", "fair": "🟡 合理",
                      "expensive": "🔴 偏貴", "na": "— 無資料"}
        rows_v = [
            ("本益比 P/E", data.get("pe"), flags["pe"]),
            ("股價淨值比 P/B", data.get("pb"), flags["pb"]),
            ("PEG", data.get("peg"), flags["peg"]),
            ("EV/EBITDA", data.get("ev_ebitda"), flags["ev_ebitda"]),
        ]
        vdf = pd.DataFrame(
            [{"指標": n, "數值": (f"{v:.2f}" if v else "—"), "評估": flag_emoji[f]}
             for n, v, f in rows_v]
        ).set_index("指標")
        st.dataframe(vdf, use_container_width=True)
        st.caption("估值用絕對合理區間判斷，未做產業比較；不同產業合理區間本就不同，僅供參考。")

    # ④ 獲利能力 ──────────────────────────────────────────────
    with tab_p:
        p1, p2, p3 = st.columns(3)
        def _pct(x): return f"{x*100:.1f}%" if x is not None else "—"
        with p1:
            metric_card("ROE", _pct(data.get("roe")))
            metric_card("ROA", _pct(data.get("roa")))
        with p2:
            metric_card("毛利率", _pct(data.get("gross_margin")))
            metric_card("營業利益率", _pct(data.get("op_margin")))
        with p3:
            metric_card("淨利率", _pct(data.get("net_margin")))
            metric_card("FCF 利潤率", _pct(data.get("fcf_margin")))
        if data.get("roe_note"):
            st.caption(f"ℹ️ ROE {data['roe_note']}")
        gr1, gr2 = st.columns(2)
        with gr1:
            metric_card("營收成長", _pct(data.get("revenue_growth")),
                        positive=(data.get("revenue_growth") or 0) > 0)
        with gr2:
            metric_card("盈餘成長", _pct(data.get("earnings_growth")),
                        positive=(data.get("earnings_growth") or 0) > 0)

    # ⑤ 財報三表 ──────────────────────────────────────────────
    with tab_s:
        def _bar(title, series, color):
            periods, vals = series
            pairs = [(p, v) for p, v in zip(periods, vals) if v is not None]
            if not pairs:
                st.caption(f"{title}：無資料")
                return
            pp, vv = zip(*pairs)
            fig = go.Figure(go.Bar(x=list(pp), y=[v/1e9 for v in vv], marker_color=color))
            fig.update_layout(**PLOTLY_LAYOUT, height=240, title=f"{title}（十億 {cur}）")
            st.plotly_chart(fig, use_container_width=True)
        _bar("營收", data.get("revenue_series", ([], [])), "#1E88E5")
        _bar("淨利", data.get("netincome_series", ([], [])), "#4CAF50")
        _bar("自由現金流", data.get("fcf_series", ([], [])), "#FF9800")
        ana_t, ana_r = data.get("target_mean"), data.get("recommendation")
        if ana_t or ana_r:
            section("分析師")
            aa1, aa2, aa3 = st.columns(3)
            with aa1: metric_card("目標均價", f"{ana_t:.2f} {cur}" if ana_t else "—")
            with aa2:
                up = ((ana_t / data["price"] - 1) * 100) if (ana_t and data.get("price")) else None
                metric_card("上漲空間", f"{up:+.1f}%" if up is not None else "—",
                            positive=(up or 0) > 0)
            with aa3: metric_card("評等", str(ana_r).upper() if ana_r else "—")

    # ⑥ AI 解讀（lazy）────────────────────────────────────────
    with tab_ai:
        st.caption("把上述數字交給 LLM 生成白話財務體質解讀（不會自行抓資料，僅解讀給定數字）。")
        ak1, ak2 = st.columns([2, 1])
        with ak1:
            fa_key = st.text_input("API Key", type="password", key="fa_ai_key",
                                   placeholder="sk-… 或 Anthropic key")
        with ak2:
            fa_model = st.selectbox("模型", [
                "claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022",
                "gpt-4o-mini", "gpt-4o",
            ], key="fa_ai_model")
        fa_base = st.text_input("API Base URL（留空自動判斷）", "", key="fa_ai_base")
        if st.button("🤖 生成 AI 解讀", key="fa_ai_run"):
            if not fa_key:
                st.error("請輸入 API Key")
            else:
                facts = (
                    f"公司：{data['name']} ({data['ticker']})；產業：{data.get('sector')}/{data.get('industry')}\n"
                    f"財務健康評分：{hs['score']}（{hs['rating']}），各構面：{hs['breakdown']}\n"
                    f"估值：P/E={data.get('pe')}, P/B={data.get('pb')}, PEG={data.get('peg')}, "
                    f"EV/EBITDA={data.get('ev_ebitda')}\n"
                    f"獲利：ROE={data.get('roe')}, 淨利率={data.get('net_margin')}, "
                    f"毛利率={data.get('gross_margin')}\n"
                    f"成長：營收={data.get('revenue_growth')}, 盈餘={data.get('earnings_growth')}\n"
                    f"財務：負債權益比={data.get('debt_to_equity')}, 流動比={data.get('current_ratio')}, "
                    f"FCF={data.get('fcf')}\n"
                )
                prompt = (
                    "你是專業基本面分析師。僅根據以下數字解讀這家公司的財務體質，"
                    "不得編造未提供的事實。用繁體中文，分『優勢』『隱憂』『總評』三段，簡潔具體。\n\n"
                    + facts
                )
                with st.spinner("AI 解讀中…"):
                    try:
                        client = _llm_client(fa_key, fa_base, fa_model)
                        resp = client.chat.completions.create(
                            model=fa_model,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.3, max_tokens=900,
                        )
                        st.session_state["fa_ai_out"] = resp.choices[0].message.content
                    except Exception as e:
                        st.error(f"AI 解讀失敗：{e}")
        if st.session_state.get("fa_ai_out"):
            st.markdown(
                f"<div style='background:#1A1D27;border:1px solid #2D3142;border-radius:10px;"
                f"padding:18px;color:#E8EAF0;line-height:1.7'>"
                f"{st.session_state['fa_ai_out'].replace(chr(10),'<br>')}</div>",
                unsafe_allow_html=True,
            )

    st.markdown(
        "<small style='color:#B8C0D0'>⚠️ 資料來源 yfinance，季報更新、非即時；"
        "部分標的（尤其非美股）欄位可能缺漏。評分為相對輔助，非投資建議。</small>",
        unsafe_allow_html=True,
    )


def page_alerts():
    import yfinance as yf
    st.title("🚨 即時警報 & 監控")
    st.caption("監控清單 · 1 分鐘走勢 · 技術訊號掃描 · Telegram / Email 推播")

    cfg = _load_alerts_config()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 監控清單", "📊 即時走勢", "⚡ 訊號掃描", "🔔 通知設定",
    ])

    # ── Tab 1: Watchlist editor ────────────────────────────────────
    with tab1:
        section("我的監控清單")
        st.caption("在此維護要監控的股票/ETF/外匯代碼，會持久化到 alerts_config.json")
        wl_text = st.text_area(
            "代碼清單（每行一個）",
            value="\n".join(cfg.get("watchlist", [])),
            height=200,
            placeholder="AAPL\nMSFT\nNVDA\n2330.TW\nBTC-USD\nEURUSD=X",
        )
        if st.button("💾 儲存清單", type="primary", key="wl_save"):
            cfg["watchlist"] = [t.strip().upper() for t in wl_text.split("\n") if t.strip()]
            _save_alerts_config(cfg)
            st.success(f"已儲存 {len(cfg['watchlist'])} 檔標的到 {ALERTS_FILE.name}")

        if cfg.get("watchlist"):
            section("最新報價快照")
            with st.spinner("抓取報價…"):
                try:
                    rows = []
                    for tk in cfg["watchlist"]:
                        try:
                            _r = yf.download(tk, period="5d", auto_adjust=True, progress=False)
                            if _r.empty:
                                continue
                            s = _r["Close"].squeeze().dropna()
                            if len(s) >= 2:
                                chg1 = float(s.iloc[-1] / s.iloc[-2] - 1)
                                chg5 = float(s.iloc[-1] / s.iloc[0]  - 1)
                                rows.append({
                                    "代碼": tk,
                                    "現價": round(float(s.iloc[-1]), 2),
                                    "1日%": chg1,
                                    "5日%": chg5,
                                    "趨勢": "▲" if chg1 > 0 else "▼",
                                })
                        except Exception:
                            continue
                    if rows:
                        snap_df = pd.DataFrame(rows).set_index("代碼")
                        def _color(v):
                            if isinstance(v, float) and not np.isnan(v):
                                return "color:#4CAF50;font-weight:600" if v > 0 else "color:#F44336;font-weight:600"
                            return "color:#E8EAF0"
                        st.dataframe(
                            snap_df.style
                                .format({"現價": "{:.2f}", "1日%": "{:.2%}", "5日%": "{:.2%}"})
                                .map(_color, subset=["1日%", "5日%"]),
                            use_container_width=True,
                        )
                        gainers = sum(1 for r in rows if r["1日%"] > 0)
                        losers  = sum(1 for r in rows if r["1日%"] < 0)
                        st.caption(f"今日 🟢 上漲 {gainers} 支　🔴 下跌 {losers} 支　⚪ 持平 {len(rows)-gainers-losers} 支")
                except Exception as e:
                    st.error(f"報價抓取失敗：{e}")

    # ── Tab 2: Intraday chart ──────────────────────────────────────
    with tab2:
        ic1, ic2, ic3 = st.columns([2, 1, 1])
        with ic1:
            intra_t = st.selectbox(
                "選擇代碼", cfg.get("watchlist", ["AAPL"]) or ["AAPL"], key="intra_t"
            )
        with ic2:
            intra_int = st.selectbox(
                "間隔", ["1m", "2m", "5m", "15m", "30m", "60m"], index=2, key="intra_int"
            )
        with ic3:
            intra_per = st.selectbox(
                "範圍", ["1d", "2d", "5d", "1mo"], index=2, key="intra_per"
            )

        if st.button("🔄 刷新", key="intra_refresh"):
            st.cache_data.clear()

        if intra_t:
            with st.spinner(f"載入 {intra_t} 即時數據…"):
                try:
                    intra = yf.download(
                        intra_t, period=intra_per, interval=intra_int,
                        auto_adjust=True, progress=False,
                    )
                    if isinstance(intra.columns, pd.MultiIndex):
                        intra.columns = intra.columns.droplevel(1)
                except Exception as e:
                    st.error(f"載入失敗：{e}")
                    intra = pd.DataFrame()

            if not intra.empty:
                last = float(intra["Close"].iloc[-1])
                first = float(intra["Open"].iloc[0])
                day_chg = last / first - 1
                day_high = float(intra["High"].max())
                day_low  = float(intra["Low"].min())
                day_vol  = int(intra["Volume"].sum())
                ic_a, ic_b, ic_c, ic_d, ic_e = st.columns(5)
                with ic_a: metric_card("最新價",  f"{last:.2f}")
                with ic_b: metric_card("區間漲跌", f"{day_chg:+.2%}", positive=day_chg >= 0)
                with ic_c: metric_card("區間高",  f"{day_high:.2f}")
                with ic_d: metric_card("區間低",  f"{day_low:.2f}")
                with ic_e: metric_card("總量",    f"{day_vol:,}")

                fig_i = make_subplots(rows=2, cols=1, row_heights=[0.75, 0.25],
                                      shared_xaxes=True, vertical_spacing=0.02)
                fig_i.add_trace(go.Candlestick(
                    x=intra.index, open=intra["Open"], high=intra["High"],
                    low=intra["Low"], close=intra["Close"], name="Price",
                    increasing_line_color="#4CAF50", decreasing_line_color="#F44336",
                ), row=1, col=1)
                # VWAP
                if "Volume" in intra.columns and intra["Volume"].sum() > 0:
                    typ = (intra["High"] + intra["Low"] + intra["Close"]) / 3
                    vwap = (typ * intra["Volume"]).cumsum() / intra["Volume"].cumsum()
                    fig_i.add_trace(go.Scatter(
                        x=vwap.index, y=vwap.values, name="VWAP",
                        line=dict(color="#FFC107", width=2),
                    ), row=1, col=1)
                fig_i.add_trace(go.Bar(
                    x=intra.index, y=intra["Volume"], name="Vol",
                    marker_color="#1E88E5", opacity=0.4,
                ), row=2, col=1)
                fig_i.update_layout(
                    **PLOTLY_LAYOUT, height=520, showlegend=True,
                    xaxis_rangeslider_visible=False,
                    title=f"{intra_t} · {intra_int} 間隔 · {intra_per} 範圍",
                )
                st.plotly_chart(fig_i, use_container_width=True)
            else:
                st.info("此代碼沒有可用的盤中資料（亞股盤後/週末會空）。")

    # ── Tab 3: Signal scanner ──────────────────────────────────────
    with tab3:
        section("技術訊號掃描器")
        st.caption("掃描監控清單，找出符合條件的標的")
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            sig_rsi_lo = st.number_input("RSI 超賣門檻", 10, 50, 30, 5, key="sig_rl")
        with sc2:
            sig_rsi_hi = st.number_input("RSI 超買門檻", 50, 90, 70, 5, key="sig_rh")
        with sc3:
            sig_chg_th = st.number_input("單日漲跌警示 (±%)", 1, 20, 3, 1, key="sig_ch") / 100

        if st.button("🔍 開始掃描", type="primary", key="sig_scan"):
            wl = cfg.get("watchlist", [])
            if not wl:
                st.warning("請先在「監控清單」加入代碼")
            else:
                with st.spinner(f"掃描 {len(wl)} 檔標的…"):
                    try:
                        raw_sg = yf.download(
                            wl, period="3mo", auto_adjust=True, progress=False,
                        )
                        cl_sg = (raw_sg["Close"] if isinstance(raw_sg.columns, pd.MultiIndex)
                                 else raw_sg).ffill()
                        signals = []
                        for tk in wl:
                            if tk not in cl_sg.columns:
                                continue
                            s = cl_sg[tk].dropna()
                            if len(s) < 50:
                                continue
                            r = s.pct_change().dropna()
                            gain = r.clip(lower=0).rolling(14).mean()
                            loss = (-r).clip(lower=0).rolling(14).mean()
                            rs   = gain / loss.replace(0, np.nan)
                            rsi  = float((100 - 100 / (1 + rs)).dropna().iloc[-1]) if not rs.dropna().empty else np.nan
                            ma20 = float(s.rolling(20).mean().iloc[-1])
                            ma50 = float(s.rolling(50).mean().iloc[-1]) if len(s) >= 50 else np.nan
                            day_c = float(s.iloc[-1] / s.iloc[-2] - 1)
                            tags = []
                            if not np.isnan(rsi):
                                if rsi <= sig_rsi_lo: tags.append("🟢 RSI 超賣")
                                if rsi >= sig_rsi_hi: tags.append("🔴 RSI 超買")
                            if not np.isnan(ma50) and len(s) >= 51:
                                ma50_prev = float(s.rolling(50).mean().iloc[-2])
                                ma20_prev = float(s.rolling(20).mean().iloc[-2])
                                if ma20_prev < ma50_prev and ma20 > ma50: tags.append("🟢 黃金交叉")
                                if ma20_prev > ma50_prev and ma20 < ma50: tags.append("🔴 死亡交叉")
                            if abs(day_c) >= sig_chg_th:
                                tags.append(f"⚡ 大幅變動 {day_c:+.2%}")
                            if tags:
                                signals.append({
                                    "代碼": tk, "現價": float(s.iloc[-1]),
                                    "1日%": day_c, "RSI(14)": rsi,
                                    "訊號": " · ".join(tags),
                                })
                        if signals:
                            sg_df = pd.DataFrame(signals).set_index("代碼")
                            st.success(f"找到 {len(signals)} 檔觸發訊號")
                            st.dataframe(
                                sg_df.style.format({"現價": "{:.2f}", "1日%": "{:.2%}", "RSI(14)": "{:.1f}"}),
                                use_container_width=True,
                            )
                            st.session_state["last_signals"] = signals
                        else:
                            st.info("目前沒有觸發任何訊號。")
                    except Exception as e:
                        st.error(f"掃描失敗：{e}")

        # Push notifications
        if st.session_state.get("last_signals"):
            st.markdown("---")
            section("推送掃描結果")
            push_cols = st.columns(2)
            with push_cols[0]:
                if st.button("📨 推送到 Telegram", key="push_tg",
                             disabled=not cfg.get("telegram", {}).get("enabled")):
                    sigs = st.session_state["last_signals"]
                    msg_lines = ["🚨 *RBS 訊號掃描*", f"_時間：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}_", ""]
                    for s in sigs:
                        msg_lines.append(f"`{s['代碼']}` {s['現價']:.2f} ({s['1日%']:+.2%})")
                        msg_lines.append(f"  → {s['訊號']}")
                    msg = "\n".join(msg_lines)
                    ok, info = _send_telegram(
                        cfg["telegram"]["token"], cfg["telegram"]["chat_id"], msg
                    )
                    if ok: st.success("Telegram 已送出")
                    else:  st.error(f"送出失敗：{info}")
            with push_cols[1]:
                if st.button("📧 寄送 Email", key="push_em",
                             disabled=not cfg.get("email", {}).get("enabled")):
                    sigs = st.session_state["last_signals"]
                    body = f"RBS 訊號掃描結果\n時間：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n"
                    for s in sigs:
                        body += f"{s['代碼']:<10} {s['現價']:>10.2f}  {s['1日%']:>+7.2%}  {s['訊號']}\n"
                    em = cfg["email"]
                    ok, info = _send_email(
                        em["smtp"], em["port"], em["user"], em["password"],
                        em["to"], "RBS 訊號掃描", body,
                    )
                    if ok: st.success("Email 已送出")
                    else:  st.error(f"送出失敗：{info}")

    # ── Tab 4: Notification settings ───────────────────────────────
    with tab4:
        section("📨 Telegram Bot")
        st.markdown(
            """
            **快速設定（3 步）：**
            1. 在 Telegram 找 `@BotFather`，輸入 `/newbot` 取得 **Token**
            2. 把你的 Bot 加為好友後傳一句話給它
            3. 開瀏覽器訪問 `https://api.telegram.org/bot<TOKEN>/getUpdates`，找到你的 **chat_id**
            """
        )
        tg_en = st.checkbox("啟用 Telegram", value=cfg["telegram"].get("enabled", False), key="tg_en")
        tg_tk = st.text_input("Bot Token", value=cfg["telegram"].get("token", ""),
                              type="password", key="tg_tk")
        tg_id = st.text_input("Chat ID", value=cfg["telegram"].get("chat_id", ""), key="tg_id")
        tg_c1, tg_c2 = st.columns(2)
        with tg_c1:
            if st.button("💾 儲存 Telegram 設定", key="tg_save"):
                cfg["telegram"] = {"enabled": tg_en, "token": tg_tk, "chat_id": tg_id}
                _save_alerts_config(cfg)
                st.success("已儲存")
        with tg_c2:
            if st.button("🧪 發送測試訊息", key="tg_test"):
                ok, info = _send_telegram(tg_tk, tg_id, "🧪 *RBS Dashboard 測試訊息*\n設定成功！")
                st.success("已送出，請檢查 Telegram") if ok else st.error(f"失敗：{info}")

        st.markdown("---")
        section("📧 Email (SMTP)")
        st.caption("Gmail：需要先在 Google 帳戶 → 安全性 → 兩步驟驗證 → 應用程式密碼")
        em_en = st.checkbox("啟用 Email", value=cfg["email"].get("enabled", False), key="em_en")
        em_c1, em_c2 = st.columns(2)
        with em_c1:
            em_host = st.text_input("SMTP Host", value=cfg["email"].get("smtp", "smtp.gmail.com"), key="em_host")
            em_user = st.text_input("帳號 (寄件者)", value=cfg["email"].get("user", ""), key="em_user")
            em_to   = st.text_input("收件者", value=cfg["email"].get("to", ""), key="em_to")
        with em_c2:
            em_port = st.number_input("Port", 1, 65535, int(cfg["email"].get("port", 465)), key="em_port")
            em_pwd  = st.text_input("密碼/應用程式密碼", value=cfg["email"].get("password", ""),
                                    type="password", key="em_pwd")
        em_b1, em_b2 = st.columns(2)
        with em_b1:
            if st.button("💾 儲存 Email 設定", key="em_save"):
                cfg["email"] = {"enabled": em_en, "smtp": em_host, "port": int(em_port),
                                "user": em_user, "password": em_pwd, "to": em_to}
                _save_alerts_config(cfg)
                st.success("已儲存")
        with em_b2:
            if st.button("🧪 發送測試 Email", key="em_test"):
                ok, info = _send_email(em_host, int(em_port), em_user, em_pwd,
                                       em_to, "RBS 測試", "RBS Dashboard 測試成功！")
                st.success("已送出") if ok else st.error(f"失敗：{info}")


# ════════════════════════════════════════════════════════════════════
# PAGE: Paper Trading (Alpaca)
# ════════════════════════════════════════════════════════════════════

def page_paper_trading():
    st.title("📉 模擬交易（Alpaca Paper）")
    st.caption("追蹤訊號策略的模擬帳戶績效 · 純模擬不涉真錢")

    try:
        import alpaca_trader as at
    except ImportError:
        st.error("找不到 alpaca_trader.py，請確認已同步（Colab 需更新 Cell 2）。")
        return

    import os as _os
    key = _os.environ.get("ALPACA_KEY_ID", "")
    secret = _os.environ.get("ALPACA_SECRET_KEY", "")
    try:
        key = key or st.secrets.get("ALPACA_KEY_ID", "")
        secret = secret or st.secrets.get("ALPACA_SECRET_KEY", "")
    except Exception:
        pass
    if not key or not secret:
        c1, c2 = st.columns(2)
        with c1:
            key = st.text_input("Alpaca Key ID", type="password", key="alp_key")
        with c2:
            secret = st.text_input("Alpaca Secret Key", type="password", key="alp_sec")
        st.caption("免費申請 alpaca.markets → Paper Trading，或設 Secrets "
                   "ALPACA_KEY_ID / ALPACA_SECRET_KEY。")
    if not key or not secret:
        st.info("填入 Alpaca paper API key 後顯示帳戶績效。")
        return

    if not st.button("🔄 載入帳戶", type="primary", key="alp_load") and \
       not st.session_state.get("alp_loaded"):
        st.info("按「載入帳戶」抓取模擬帳戶資料。")
        return
    st.session_state["alp_loaded"] = True

    with st.spinner("讀取 Alpaca 帳戶…"):
        acc = at.get_account(key, secret)
        positions = at.get_positions(key, secret)
        orders = at.get_orders(key, secret, limit=20)
        hist = at.portfolio_history(key, secret, period="3M")

    if not acc:
        st.error("帳戶讀取失敗，請確認 key 是否為 paper trading key。")
        return

    # ── 帳戶摘要 ───────────────────────────────────────────────
    r = at.account_return(acc)
    section("帳戶摘要")
    m1, m2, m3, m4 = st.columns(4)
    with m1: metric_card("淨值", f"${r['equity']:,.0f}" if r["equity"] is not None else "—")
    with m2:
        dc = r["day_change"]
        metric_card("今日損益", f"{dc:+,.0f}" if dc is not None else "—",
                    positive=(dc or 0) >= 0)
    with m3:
        metric_card("今日%", f"{r['day_pct']*100:+.2f}%" if r["day_pct"] is not None else "—",
                    positive=(r["day_pct"] or 0) >= 0)
    with m4:
        cash = at._f(acc.get("cash"))
        metric_card("現金", f"${cash:,.0f}" if cash is not None else "—")

    # ── 權益曲線 vs SPY ────────────────────────────────────────
    if hist and hist.get("equity"):
        section("權益曲線 vs SPY")
        # 一起過濾 (timestamp, equity) 配對，避免濾掉中間的 None 造成軸錯位
        ts_raw = hist.get("timestamp", [])
        pairs = [(t, v) for t, v in zip(ts_raw, hist["equity"]) if v]
        if len(pairs) >= 2:
            import datetime as _dt
            tvals, eq = zip(*pairs)
            dates = [_dt.datetime.fromtimestamp(t).date() for t in tvals]
            eq_pct = [(v / eq[0] - 1) * 100 for v in eq]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=eq_pct, name="模擬帳戶",
                                     line=dict(color="#1E88E5", width=2)))
            try:
                import yfinance as yf
                spy = yf.download("SPY", start=str(dates[0]), auto_adjust=True, progress=False)
                sc = spy["Close"].squeeze().dropna()
                if len(sc) >= 2:
                    spy_pct = (sc / float(sc.iloc[0]) - 1) * 100
                    fig.add_trace(go.Scatter(x=spy_pct.index, y=spy_pct.values, name="SPY",
                                             line=dict(color="#888", width=1.5, dash="dot")))
            except Exception:
                pass
            fig.update_layout(**PLOTLY_LAYOUT, height=360, yaxis_title="累積報酬 %",
                              title="模擬帳戶 vs SPY（同期 %）")
            st.plotly_chart(fig, use_container_width=True)

    # ── 持倉 ───────────────────────────────────────────────────
    section("目前持倉")
    if positions:
        def _nan(x): return x if isinstance(x, (int, float)) else np.nan  # None → NaN，避免 formatter 崩潰
        rows = []
        for sym, p in positions.items():
            rows.append({
                "代碼": sym, "股數": _nan(p.get("qty")),
                "均價": _nan(p.get("avg_entry_price")), "市值": _nan(p.get("market_value")),
                "未實現損益": _nan(p.get("unrealized_pl")),
                "報酬%": (p.get("unrealized_plpc") or 0) * 100,
            })
        pdf = pd.DataFrame(rows).set_index("代碼")
        def _pl(v):
            if isinstance(v, (int, float)) and not np.isnan(v):
                return "color:#4CAF50" if v > 0 else ("color:#F44336" if v < 0 else "")
            return ""
        st.dataframe(
            pdf.style.format({"股數": "{:g}", "均價": "{:.2f}", "市值": "{:,.0f}",
                              "未實現損益": "{:+,.2f}", "報酬%": "{:+.2f}%"}, na_rep="—")
               .map(_pl, subset=["未實現損益", "報酬%"]),
            use_container_width=True,
        )
    else:
        st.info("目前無持倉。")

    # ── 近期訂單 ───────────────────────────────────────────────
    if orders:
        section("近期訂單")
        orows = []
        for o in orders[:15]:
            orows.append({
                "時間": (o.get("submitted_at") or "")[:16].replace("T", " "),
                "代碼": o.get("symbol"), "方向": o.get("side"),
                "股數": o.get("qty"), "狀態": o.get("status"),
                "成交價": o.get("filled_avg_price") or "—",
            })
        st.dataframe(pd.DataFrame(orows), use_container_width=True, hide_index=True)

    # ── 交易日誌（含下單原因/評分）─────────────────────────────
    section("交易日誌（為什麼交易）")
    journal = at.load_journal(BASE_DIR / "trade_journal.json")
    if journal:
        jrows = []
        for e in journal[-30:][::-1]:
            sc = e.get("score")
            jrows.append({
                "時間": (e.get("time") or "")[:16].replace("T", " "),
                "代碼": e.get("symbol"), "方向": e.get("side"),
                "股數": e.get("qty"),
                "評分": (round(sc, 2) if isinstance(sc, (int, float)) else None),
                "原因": e.get("reason"),
                "送出": "✅" if e.get("submitted") else "❌",
            })
        st.dataframe(pd.DataFrame(jrows), use_container_width=True, hide_index=True)
        st.caption("每筆自動交易的決策評分與觸發原因（由 Bot 記錄，與上方 Alpaca 實際成交對照）。")
    else:
        st.info("尚無自動交易紀錄。開啟 Bot 的 `/autotrade on` 後，每筆自動交易的評分與原因會記在這裡。")

    st.caption("⚠️ Alpaca paper 為模擬環境，成交為理想化（無真實滑價/流動性）。"
               "自動下單由 Bot 的 /autotrade 控制，此頁僅檢視。")


# ════════════════════════════════════════════════════════════════════
# PAGE: Trading Tools (Position Sizing / Kelly / R:R / Compound)
# ════════════════════════════════════════════════════════════════════

def page_trading_tools():
    st.title("🛠️ 交易工具")
    st.caption("部位大小計算 · Kelly 公式 · 風險報酬比 · 複利計算")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📐 部位大小", "🌊 波動率部位", "🎲 Kelly 公式", "🎯 風險報酬比", "💰 複利計算",
    ])

    # ── Tab 1: Position sizing ─────────────────────────────────────
    with tab1:
        section("部位大小計算（基於風險）")
        st.caption("公式：部位金額 = (帳戶 × 風險%) ÷ (進場價 − 停損價) × 進場價")
        ps1, ps2 = st.columns(2)
        with ps1:
            acct = st.number_input("帳戶總值 (USD)", 100.0, 1e9, 100_000.0, step=1_000.0, key="ps_acct")
            risk_pct = st.slider("單筆風險 (%)", 0.1, 5.0, 1.0, 0.1, key="ps_rp") / 100
        with ps2:
            entry = st.number_input("進場價", 0.01, 1e6, 100.0, step=0.5, key="ps_e")
            stop  = st.number_input("停損價", 0.01, 1e6,  95.0, step=0.5, key="ps_s")

        risk_per_share = abs(entry - stop)
        risk_amount = acct * risk_pct
        if risk_per_share > 0:
            shares = int(risk_amount / risk_per_share)
            position_value = shares * entry
            position_pct = position_value / acct
        else:
            shares = 0
            position_value = 0
            position_pct = 0

        st.markdown("---")
        ps_a, ps_b, ps_c, ps_d = st.columns(4)
        with ps_a: metric_card("可承受損失",  f"${risk_amount:,.0f}")
        with ps_b: metric_card("每股風險",    f"${risk_per_share:.2f}")
        with ps_c: metric_card("建議股數",    f"{shares:,}")
        with ps_d: metric_card("部位金額",    f"${position_value:,.0f}",
                               positive=position_pct < 0.5)
        st.info(f"此部位佔帳戶 **{position_pct:.1%}**" +
                ("，槓桿較重" if position_pct > 0.5 else ""))

    # ── Tab 2: Volatility-targeted sizing (ATR + vol target) ───────
    with tab2:
        section("波動率目標部位（ATR 風險基準）")
        st.caption(
            "依個股波動度反比配置：高波動少買、低波動多買，讓每筆風險一致。"
            "輸入代碼自動抓 ATR 與年化波動率。學習自 freqtrade 動態 stake。"
        )
        try:
            import quant_tools as _qt
            import yfinance as _yf
        except ImportError:
            _qt = None
            st.error("找不到 quant_tools.py，請確認已同步（Colab 需更新 Cell 2）。")

        if _qt:
            vt1, vt2, vt3 = st.columns(3)
            with vt1:
                vt_tkr = st.text_input("股票代碼", "AAPL", key="vt_tk").upper().strip()
                vt_acct = st.number_input("帳戶總值 (USD)", 100.0, 1e9, 100_000.0,
                                          step=1_000.0, key="vt_acct")
            with vt2:
                vt_risk = st.slider("單筆風險 (%)", 0.1, 5.0, 1.0, 0.1, key="vt_risk") / 100
                vt_atr_mult = st.slider("ATR 停損倍數", 0.5, 4.0, 1.5, 0.1, key="vt_am")
            with vt3:
                vt_target = st.slider("組合目標年化波動 (%)", 5, 40, 15, 1, key="vt_tg") / 100
                vt_maxlev = st.slider("最大槓桿", 1.0, 3.0, 2.0, 0.1, key="vt_ml")

            if st.button("計算波動率部位", type="primary", key="vt_run"):
                with st.spinner(f"抓取 {vt_tkr} 波動數據…"):
                    try:
                        raw_vt = _yf.download(vt_tkr, period="6mo",
                                              auto_adjust=True, progress=False)
                        if raw_vt.empty:
                            st.error("無資料，請確認代碼。")
                        else:
                            cl = raw_vt["Close"].squeeze().dropna()
                            hi = raw_vt["High"].squeeze().dropna() if "High" in raw_vt.columns else cl
                            lo = raw_vt["Low"].squeeze().dropna() if "Low" in raw_vt.columns else cl
                            price = float(cl.iloc[-1])
                            # ATR(14)
                            tr = pd.concat([hi - lo, (hi - cl.shift()).abs(),
                                            (lo - cl.shift()).abs()], axis=1).max(axis=1)
                            atr = float(tr.rolling(14).mean().iloc[-1])
                            ann_vol = float(cl.pct_change().dropna().std() * np.sqrt(252))

                            ps = _qt.atr_position_size(vt_acct, vt_risk, price, atr, vt_atr_mult)
                            lev = _qt.volatility_target(ann_vol, vt_target, vt_maxlev)
                            vol_adj_value = ps["position_value"] * lev

                            st.markdown("---")
                            m1, m2, m3, m4 = st.columns(4)
                            with m1: metric_card("現價",        f"${price:.2f}")
                            with m2: metric_card("ATR(14)",     f"${atr:.2f}")
                            with m3: metric_card("年化波動",     f"{ann_vol:.1%}")
                            with m4: metric_card("波動目標槓桿", f"{lev:.2f}×")

                            st.markdown("---")
                            n1, n2, n3, n4 = st.columns(4)
                            with n1: metric_card("建議股數",   f"{ps['shares']:,.0f}")
                            with n2: metric_card("停損價",     f"${ps['stop_price']:.2f}")
                            with n3: metric_card("ATR 部位",   f"${ps['position_value']:,.0f}",
                                                 positive=ps['pct_of_account'] < 0.5)
                            with n4: metric_card("波動調整後", f"${vol_adj_value:,.0f}")

                            st.info(
                                f"ATR 部位佔帳戶 **{ps['pct_of_account']:.1%}**（風險 ${ps['risk_amount']:,.0f}）。"
                                f"此股年化波動 {ann_vol:.0%} vs 目標 {vt_target:.0%} → "
                                f"建議{'放大' if lev > 1 else '縮小'}部位至 **{lev:.2f}×**。"
                            )
                    except Exception as e:
                        st.error(f"計算失敗：{e}")

    # ── Tab 3: Kelly criterion ─────────────────────────────────────
    with tab3:
        section("Kelly 公式 — 最佳下注比例")
        st.caption("公式：f* = (bp - q) / b，b = 賠率，p = 勝率，q = 1-p")
        ke1, ke2 = st.columns(2)
        with ke1:
            wr = st.slider("勝率 p (%)",  10.0, 90.0, 55.0, 0.5, key="k_wr") / 100
        with ke2:
            wlr = st.number_input("賠率 b（賺/賠 比）", 0.1, 20.0, 2.0, 0.1, key="k_b")

        kelly = (wlr * wr - (1 - wr)) / wlr if wlr > 0 else 0
        kelly = max(kelly, 0)

        ke_a, ke_b, ke_c = st.columns(3)
        with ke_a: metric_card("Full Kelly",    f"{kelly:.2%}")
        with ke_b: metric_card("Half Kelly",    f"{kelly/2:.2%}",  positive=True)
        with ke_c: metric_card("Quarter Kelly", f"{kelly/4:.2%}",  positive=True)

        st.markdown(
            "**建議**：實務上很少有人壓 Full Kelly（波動極大），"
            "多數機構與專業交易者用 **Half/Quarter Kelly** 以降低 Drawdown。"
        )

        # Kelly curve
        ws_arr = np.arange(0.01, 1.0, 0.01)
        f_arr = np.array([(wlr * x - (1 - x)) / wlr for x in ws_arr])
        f_arr = np.maximum(f_arr, 0)
        fig_k = go.Figure()
        fig_k.add_trace(go.Scatter(x=ws_arr * 100, y=f_arr * 100,
                                   name="Optimal Kelly %",
                                   line=dict(color="#1E88E5", width=2)))
        fig_k.add_vline(x=wr * 100, line_dash="dash", line_color="#F44336",
                        annotation_text=f"勝率 {wr:.1%}")
        fig_k.update_layout(**PLOTLY_LAYOUT, height=320,
                            title=f"Kelly 曲線（賠率 b = {wlr:.1f}）",
                            xaxis_title="勝率 (%)", yaxis_title="最佳下注 %")
        st.plotly_chart(fig_k, use_container_width=True)

    # ── Tab 4: Risk:Reward ─────────────────────────────────────────
    with tab4:
        section("風險報酬比 R:R 分析")
        rr1, rr2, rr3 = st.columns(3)
        with rr1: rr_e = st.number_input("進場價",   0.01, 1e6, 100.0, step=0.5, key="rr_e")
        with rr2: rr_s = st.number_input("停損價",   0.01, 1e6,  95.0, step=0.5, key="rr_s")
        with rr3: rr_t = st.number_input("目標價",   0.01, 1e6, 115.0, step=0.5, key="rr_t")

        risk = abs(rr_e - rr_s)
        reward = abs(rr_t - rr_e)
        ratio = reward / risk if risk > 0 else 0
        be_wr = 1 / (1 + ratio) if ratio > 0 else 1.0

        rr_a, rr_b, rr_c, rr_d = st.columns(4)
        with rr_a: metric_card("風險 R",    f"${risk:.2f}")
        with rr_b: metric_card("回報",      f"${reward:.2f}")
        with rr_c: metric_card("R:R",       f"1 : {ratio:.2f}", positive=ratio >= 2)
        with rr_d: metric_card("損益兩平勝率", f"{be_wr:.1%}")

        if ratio >= 3:
            st.success("✅ R:R ≥ 3，極佳的風險報酬結構")
        elif ratio >= 2:
            st.info("👍 R:R ≥ 2，符合多數策略最低門檻")
        elif ratio >= 1:
            st.warning("⚠️ R:R 偏低，需要高勝率才能獲利")
        else:
            st.error("❌ R:R < 1，數學期望值不利")

        # Expected value sweep
        wr_range = np.linspace(0.3, 0.8, 51)
        ev = wr_range * reward - (1 - wr_range) * risk
        fig_ev = go.Figure()
        fig_ev.add_trace(go.Scatter(x=wr_range * 100, y=ev,
                                    fill="tozeroy", line=dict(color="#1E88E5", width=2)))
        fig_ev.add_hline(y=0, line_dash="dash", line_color="#F44336")
        fig_ev.add_vline(x=be_wr * 100, line_dash="dot", line_color="#FF9800",
                         annotation_text=f"BE: {be_wr:.1%}")
        fig_ev.update_layout(**PLOTLY_LAYOUT, height=320,
                             title="期望值 vs 勝率",
                             xaxis_title="勝率 (%)", yaxis_title="每筆期望值 ($)")
        st.plotly_chart(fig_ev, use_container_width=True)

    # ── Tab 5: Compound interest ───────────────────────────────────
    with tab5:
        section("複利計算（含定期投入）")
        cm1, cm2, cm3 = st.columns(3)
        with cm1:
            cm_pv = st.number_input("初始本金", 0.0, 1e9, 100_000.0, step=10_000.0, key="cm_pv")
        with cm2:
            cm_pmt = st.number_input("每月定期投入", 0.0, 1e7, 1_000.0, step=500.0, key="cm_pmt")
        with cm3:
            cm_yrs = st.number_input("投資年數", 1, 60, 20, key="cm_yrs")
        cm_r = st.slider("年化報酬率 (%)", 1.0, 25.0, 8.0, 0.5, key="cm_r") / 100

        months = int(cm_yrs * 12)
        monthly_r = cm_r / 12
        balances = []
        bal = cm_pv
        contrib = cm_pv
        for m in range(months + 1):
            balances.append({"月": m, "餘額": bal, "累計投入": contrib})
            bal = bal * (1 + monthly_r) + cm_pmt
            contrib += cm_pmt
        bal_df = pd.DataFrame(balances)

        final_bal = bal_df["餘額"].iloc[-1]
        total_contrib = bal_df["累計投入"].iloc[-1]
        gain = final_bal - total_contrib

        fc_a, fc_b, fc_c, fc_d = st.columns(4)
        with fc_a: metric_card("最終餘額",   f"${final_bal:,.0f}",   positive=True)
        with fc_b: metric_card("累計投入",   f"${total_contrib:,.0f}")
        with fc_c: metric_card("純複利收益", f"${gain:,.0f}",        positive=True)
        with fc_d: metric_card("倍數",       f"{final_bal/cm_pv:.2f}x" if cm_pv > 0 else "—")

        fig_c = go.Figure()
        fig_c.add_trace(go.Scatter(x=bal_df["月"] / 12, y=bal_df["餘額"],
                                   name="總資產", fill="tozeroy",
                                   line=dict(color="#1E88E5", width=2)))
        fig_c.add_trace(go.Scatter(x=bal_df["月"] / 12, y=bal_df["累計投入"],
                                   name="累計投入",
                                   line=dict(color="#FF9800", width=2, dash="dash")))
        fig_c.update_layout(**PLOTLY_LAYOUT, height=420,
                            title=f"資產成長軌跡（年化 {cm_r:.1%}）",
                            xaxis_title="年", yaxis_title="USD")
        st.plotly_chart(fig_c, use_container_width=True)


# ════════════════════════════════════════════════════════════════════
# Router
# ════════════════════════════════════════════════════════════════════

PAGES = {
    "🏠 市場總覽":  page_market_overview,
    "📈 持倉分析":  page_portfolio_performance,
    "⚠️ 風險管理":  page_risk_management,
    "🔍 股票研究":  page_stock_research,
    "🏢 公司分析":  page_company_analysis,
    "🚨 即時警報":  page_alerts,
    "🛠️ 交易工具":  page_trading_tools,
    "📉 模擬交易":  page_paper_trading,
    "🏦 機構選股":  page_stock_selector,
    "📰 新聞情報":  page_news_sentiment,
    "📦 匯出報告":  page_export,
}

PAGES[page]()
