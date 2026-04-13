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
    .metric-card {
        background: #1A1D27;
        border: 1px solid #2D3142;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 4px 0;
    }
    .metric-label {
        font-size: 0.78rem;
        color: #9EA3B0;
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
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1E88E5;
        border-left: 3px solid #1E88E5;
        padding-left: 10px;
        margin: 16px 0 10px 0;
    }
    div[data-testid="stSidebar"] {
        background-color: #0D1117;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────── Helpers ────────────────────────────────

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0F1117",
    plot_bgcolor="#0F1117",
    font=dict(family="Inter, sans-serif", size=12, color="#FAFAFA"),
    margin=dict(l=40, r=20, t=40, b=40),
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
    page = st.radio(
        "Navigation",
        [
            "🏠 Overview",
            "📈 Portfolio Performance",
            "⚠️ Portfolio Risk",
            "🔁 VaR Backtest",
            "💥 Scenarios & Stress",
            "🔗 Correlation & Beta",
            "💳 Credit Model",
            "📰 News & Sentiment",
            "🏦 機構選股",
            "📦 Export Report",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("RBS Finance Dashboard v2.0")


# ════════════════════════════════════════════════════════════════════
# PAGE: Overview / Risk Dashboard
# ════════════════════════════════════════════════════════════════════

def page_overview():
    st.title("📊 Integrated Risk Dashboard")
    st.caption("Single-asset VaR / CVaR, volatility, and correlation overview")

    with st.sidebar:
        st.markdown("### Settings")
        tickers = st.multiselect(
            "Assets",
            ["AAPL", "MSFT", "GOOGL", "AMZN", "^GSPC", "SMH", "SOXX", "^VIX", "XSD", "SPY", "QQQ", "GLD"],
            default=["AAPL", "MSFT", "^GSPC"],
        )
        start_date = st.date_input("Start Date", value=date(2020, 1, 1))
        alpha = st.slider("Confidence Level", 0.80, 0.99, 0.95, 0.01)
        vol_window = st.number_input("Vol Window (days)", 10, 252, 20)

    if not tickers:
        st.info("Select at least one asset from the sidebar.")
        return

    with st.spinner("Loading market data…"):
        try:
            px_df = load_price_data(tickers, start=str(start_date))
        except Exception as e:
            st.error(f"Data load error: {e}")
            return

    rets = px_df.pct_change().dropna()

    # ── Summary table ──────────────────────────────────────────────
    section("Risk Summary")
    rows = []
    for t in px_df.columns:
        r = rets[t].dropna()
        if len(r) < 30:
            continue
        rows.append(
            {
                "Ticker": t,
                "Hist VaR": historical_var(r, alpha),
                "CVaR": conditional_var(r, alpha),
                "Δ-Normal VaR": delta_normal_var(r, alpha),
                "Ann. Vol": r.std(ddof=1) * np.sqrt(252),
                "Sharpe (ann.)": (r.mean() / r.std(ddof=1)) * np.sqrt(252) if r.std() else np.nan,
                "Max DD": (r.add(1).cumprod() / r.add(1).cumprod().cummax() - 1).min(),
            }
        )
    if rows:
        summary = pd.DataFrame(rows).set_index("Ticker")
        fmt = {c: "{:.2%}" for c in summary.columns}
        st.dataframe(summary.style.format(fmt).background_gradient(cmap="RdYlGn", subset=["Sharpe (ann.)"]), use_container_width=True)

    # ── Distribution charts ────────────────────────────────────────
    section("Return Distributions")
    cols = st.columns(min(len(px_df.columns), 3))
    for i, t in enumerate(px_df.columns):
        r = rets[t].dropna()
        if len(r) < 30:
            continue
        var_h = historical_var(r, alpha)
        cvar_h = conditional_var(r, alpha)
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=r.values, nbinsx=60, name=t, marker_color="#1E88E5", opacity=0.75))
        fig.add_vline(x=var_h, line_color="#F44336", line_dash="dash", annotation_text=f"VaR {var_h:.2%}")
        fig.add_vline(x=cvar_h, line_color="#FF9800", line_dash="dot", annotation_text=f"CVaR {cvar_h:.2%}")
        fig.update_layout(**PLOTLY_LAYOUT, title=f"{t} Returns", height=300, showlegend=False)
        cols[i % 3].plotly_chart(fig, use_container_width=True)

    # ── Rolling volatility ─────────────────────────────────────────
    section("Rolling Volatility (Annualised)")
    fig = go.Figure()
    for t in px_df.columns:
        rv = calculate_volatility(rets[t].dropna(), window=int(vol_window))
        fig.add_trace(go.Scatter(x=rv.index, y=rv.values, name=t, mode="lines"))
    fig.update_layout(**PLOTLY_LAYOUT, title=f"Rolling {vol_window}d Volatility", height=350)
    st.plotly_chart(fig, use_container_width=True)

    # ── Correlation heatmap ────────────────────────────────────────
    section("Correlation Matrix")
    corr = rets.corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        aspect="auto",
    )
    fig.update_layout(**PLOTLY_LAYOUT, title="Return Correlations", height=400)
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════
# PAGE: Portfolio Performance
# ════════════════════════════════════════════════════════════════════

def page_portfolio_performance():
    st.title("📈 Portfolio Performance")
    st.caption("Multi-asset portfolio analytics: equity curve, drawdown, Sharpe, IR, Beta")

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
            data = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
            data = data.dropna(how="all").ffill()
            if isinstance(data, pd.Series):
                data = data.to_frame()
            if freq in ["W", "M"]:
                data = data.resample(freq).last()
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
                rf_raw = rf_raw.resample(freq).last()
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
# PAGE: Portfolio Risk
# ════════════════════════════════════════════════════════════════════

def page_portfolio_risk():
    st.title("⚠️ Portfolio Risk")
    st.caption("Delta-Normal & Historical VaR/CVaR, EWMA/LW covariance, Monte Carlo simulation")

    with st.sidebar:
        st.markdown("### Settings")
        raw_tickers = st.text_input("Tickers (comma-separated)", "AAPL,MSFT,GOOGL,AMZN")
        tickers = [t.strip().upper() for t in raw_tickers.split(",") if t.strip()]
        if tickers:
            raw_w = st.text_input("Weights (comma-separated, will normalise)", ",".join([f"{1/len(tickers):.4f}"] * len(tickers)))
        start = st.date_input("Start Date", value=date(2020, 1, 1))
        alpha = st.slider("Confidence Level", 0.80, 0.99, 0.95, 0.01)
        hold_days = st.number_input("Holding Days", 1, 60, 1)
        window = st.number_input("Lookback Window", 60, 2000, 252)
        cov_method = st.selectbox("Covariance Method", ["hist", "ewma", "lw"], format_func=lambda x: {"hist": "Historical", "ewma": "EWMA (RiskMetrics)", "lw": "Ledoit-Wolf"}[x])
        lam = st.slider("λ (EWMA)", 0.80, 0.99, 0.94, 0.01)
        do_mc = st.checkbox("Monte Carlo simulation", value=True)
        n_mc = st.select_slider("MC Paths", options=[1000, 5000, 10000, 50000], value=10000)

    if not tickers:
        st.info("Enter tickers in the sidebar.")
        return

    try:
        ws = np.array([float(x.strip()) for x in raw_w.split(",") if x.strip()])
        ws = ws / ws.sum()
        if len(ws) != len(tickers):
            raise ValueError
    except Exception:
        ws = np.repeat(1 / len(tickers), len(tickers))

    with st.spinner("Loading data…"):
        try:
            px_df = load_price_data(tickers, start=str(start))
        except Exception as e:
            st.error(f"Data error: {e}")
            return

    w_series = pd.Series(ws, index=px_df.columns)
    with st.spinner("Computing risk metrics…"):
        res = portfolio_var(
            px_df, w_series, alpha=alpha, hold_days=int(hold_days),
            window=int(window), cov_method=cov_method, lam=lam,
            as_of_value=float((px_df.iloc[-1] * w_series).sum()),
        )

    section("Portfolio Risk Metrics")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: metric_card("VaR (%)", f"{res.var_pct:.3%}", positive=False)
    with c2: metric_card("CVaR (%)", f"{res.cvar_pct:.3%}", positive=False)
    with c3: metric_card("Ann. Vol", f"{res.vol_ann:.3%}")
    with c4: metric_card("VaR (USD)", f"${res.value_var:,.0f}", positive=False)
    with c5: metric_card("CVaR (USD)", f"${res.value_cvar:,.0f}", positive=False)

    tab1, tab2 = st.tabs(["Covariance", "Monte Carlo P&L"])

    with tab1:
        section("Covariance Matrix")
        fig = px.imshow(res.cov, text_auto=".4f", color_continuous_scale="Blues", aspect="auto")
        fig.update_layout(**PLOTLY_LAYOUT, title="Covariance Matrix", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if do_mc:
            with st.spinner(f"Running {n_mc:,} MC paths…"):
                pnl = mc_portfolio_pnl(px_df, w_series, days=int(hold_days), n=int(n_mc), cov_method=cov_method, lam=lam, window=int(window))
            mc_var = np.percentile(pnl, (1 - alpha) * 100)
            mc_cvar = pnl[pnl <= mc_var].mean()
            c1, c2 = st.columns(2)
            with c1: metric_card("MC VaR (USD)", f"${-mc_var:,.0f}", positive=False)
            with c2: metric_card("MC CVaR (USD)", f"${-mc_cvar:,.0f}", positive=False)
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=pnl, nbinsx=80, marker_color="#1E88E5", opacity=0.8, name="P&L"))
            fig.add_vline(x=mc_var, line_color="#F44336", line_dash="dash", annotation_text=f"VaR {mc_var:,.0f}")
            fig.add_vline(x=mc_cvar, line_color="#FF9800", line_dash="dot", annotation_text=f"CVaR {mc_cvar:,.0f}")
            fig.update_layout(**PLOTLY_LAYOUT, title=f"Monte Carlo P&L Distribution ({n_mc:,} paths)", height=420)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Enable Monte Carlo in the sidebar.")


# ════════════════════════════════════════════════════════════════════
# PAGE: VaR Backtest
# ════════════════════════════════════════════════════════════════════

def page_var_backtest():
    st.title("🔁 VaR Backtesting")
    st.caption("Kupiec POF test with rolling VaR vs realised returns")

    with st.sidebar:
        st.markdown("### Settings")
        raw_tickers = st.text_input("Tickers (comma-separated)", "AAPL,MSFT,GOOGL,AMZN")
        tickers = [t.strip().upper() for t in raw_tickers.split(",") if t.strip()]
        if tickers:
            raw_w = st.text_input("Weights", ",".join([f"{1/len(tickers):.4f}"] * len(tickers)))
        start = st.date_input("Start Date", value=date(2018, 1, 1))
        alpha = st.slider("Confidence Level", 0.80, 0.99, 0.95, 0.01)
        window = st.number_input("Rolling Window", 60, 2000, 250)
        cov_method = st.selectbox("Covariance", ["hist", "ewma", "lw"], format_func=lambda x: {"hist": "Historical", "ewma": "EWMA", "lw": "Ledoit-Wolf"}[x])
        lam = st.slider("λ (EWMA)", 0.80, 0.99, 0.94, 0.01)
        run = st.button("Run Backtest", use_container_width=True, type="primary")

    if not run:
        st.info("Configure parameters and click **Run Backtest**.")
        return

    try:
        ws = np.array([float(x.strip()) for x in raw_w.split(",") if x.strip()])
        ws = ws / ws.sum()
        if len(ws) != len(tickers):
            raise ValueError
    except Exception:
        ws = np.repeat(1 / len(tickers), len(tickers))

    with st.spinner("Loading data and computing rolling VaR…"):
        try:
            px_df = load_price_data(tickers, start=str(start))
            w = pd.Series(ws, index=px_df.columns)
            var_series = rolling_portfolio_var(px_df, w, alpha=alpha, window=int(window), cov_method=cov_method, lam=lam)
            port_ret = (px_df.pct_change().dropna() @ w).reindex(var_series.index)
            kup = kupiec_pof_test(port_ret, var_series, alpha)
        except Exception as e:
            st.error(f"Error: {e}")
            return

    section("Kupiec POF Test")
    c1, c2, c3, c4 = st.columns(4)
    p_ok = kup.p_value > 0.05 if pd.notna(kup.p_value) else None
    with c1: metric_card("Exceptions", str(kup.exceptions))
    with c2: metric_card("Expected", f"{kup.expected:.1f}")
    with c3: metric_card("Exc. Ratio", f"{kup.ratio:.4f}")
    with c4: metric_card("p-value", f"{kup.p_value:.4f}" if pd.notna(kup.p_value) else "N/A", positive=p_ok)

    if pd.notna(kup.p_value):
        if kup.p_value > 0.05:
            st.success("✅ Model not rejected at 5% significance (p > 0.05)")
        else:
            st.warning("⚠️ Model rejected at 5% significance (p ≤ 0.05) — consider recalibrating")

    section("Rolling VaR vs Realised Returns")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=port_ret.index, y=port_ret.values, name="Portfolio Return", line=dict(color="#1E88E5", width=1), opacity=0.7))
    fig.add_trace(go.Scatter(x=var_series.index, y=var_series.values, name=f"VaR ({alpha:.0%})", line=dict(color="#F44336", width=2)))

    exceptions = port_ret[port_ret < var_series.reindex(port_ret.index)]
    fig.add_trace(go.Scatter(x=exceptions.index, y=exceptions.values, mode="markers", name="Exceptions", marker=dict(color="#FF9800", size=8, symbol="x")))
    fig.update_layout(**PLOTLY_LAYOUT, title="Rolling VaR Backtest", height=450, yaxis_title="Daily Return")
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════
# PAGE: Scenarios & Stress
# ════════════════════════════════════════════════════════════════════

def page_scenarios():
    st.title("💥 Scenarios & Stress Testing")
    st.caption("Custom per-asset shocks and historical scenario replay")

    with st.sidebar:
        st.markdown("### Settings")
        raw_tickers = st.text_input("Tickers (comma-separated)", "AAPL,MSFT,GOOGL,AMZN")
        tickers = [t.strip().upper() for t in raw_tickers.split(",") if t.strip()]
        if tickers:
            raw_w = st.text_input("Weights", ",".join([f"{1/len(tickers):.4f}"] * len(tickers)))
        start_dl = st.date_input("Data Start", value=date(2020, 1, 1))
        notional = st.number_input("Portfolio Notional (USD)", min_value=1000.0, value=100_000.0, step=1000.0)

    if not tickers:
        st.info("Enter tickers in the sidebar.")
        return

    try:
        ws = np.array([float(x.strip()) for x in raw_w.split(",") if x.strip()])
        ws = ws / ws.sum()
        if len(ws) != len(tickers):
            raise ValueError
    except Exception:
        ws = np.repeat(1 / len(tickers), len(tickers))

    with st.spinner("Loading data…"):
        try:
            px_df = load_price_data(tickers, start=str(start_dl))
        except Exception as e:
            st.error(f"Data error: {e}")
            return

    w = pd.Series(ws, index=px_df.columns, dtype=float)
    data_min = px_df.index.min().date()
    data_max = px_df.index.max().date()
    st.caption(f"Data available: **{data_min}** → **{data_max}** ({len(px_df):,} rows)")

    tab1, tab2, tab3 = st.tabs(["Custom Scenario", "Historical Replay", "Predefined Scenarios"])

    # ── A: Custom shocks ───────────────────────────────────────────
    with tab1:
        section("Per-Asset Shock Inputs")
        shock_df = pd.DataFrame({"Ticker": list(w.index), "Shock (%)": [0.0] * len(w)})
        edited = st.data_editor(shock_df, use_container_width=True, key="shock_edit")
        shocks = {row["Ticker"]: float(row.get("Shock (%)", 0.0)) / 100.0 for _, row in edited.iterrows()}

        if st.button("Run Scenario", type="primary", key="run_scn"):
            shocks_vec = np.array([shocks.get(t, 0.0) for t in w.index], dtype=float)
            pnl = scenario_pnl_value(w, shocks, notional)
            port_ret_val = float(np.dot(w.values, shocks_vec))
            new_val = notional * (1 + port_ret_val)

            c1, c2, c3 = st.columns(3)
            with c1: metric_card("Portfolio Return", f"{port_ret_val:.2%}", positive=port_ret_val >= 0)
            with c2: metric_card("P&L (USD)", f"${pnl:,.0f}", positive=pnl >= 0)
            with c3: metric_card("New Value (USD)", f"${new_val:,.0f}", positive=new_val >= notional)

            section("Asset Contribution (basis points)")
            contrib = pd.DataFrame({
                "Weight (%)": w.values * 100,
                "Shock (%)": shocks_vec * 100,
                "Contribution (bp)": w.values * shocks_vec * 10_000,
            }, index=w.index)
            fig = px.bar(
                contrib.reset_index(), x="index", y="Contribution (bp)",
                color="Contribution (bp)", color_continuous_scale="RdYlGn",
                template="plotly_dark", text_auto=".1f",
            )
            fig.update_layout(**PLOTLY_LAYOUT, title="Contribution per Asset", height=350, xaxis_title="Ticker")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(contrib.style.format({"Weight (%)": "{:.2f}", "Shock (%)": "{:.2f}", "Contribution (bp)": "{:.1f}"}), use_container_width=True)

    # ── B: Historical replay ───────────────────────────────────────
    with tab2:
        section("Historical Replay (Real Returns)")
        c1, c2 = st.columns(2)
        with c1:
            sdate = st.date_input("Start", value=max(data_min, date(2020, 1, 1)), min_value=data_min, max_value=data_max, key="rep_s")
        with c2:
            edate = st.date_input("End", value=data_max, min_value=data_min, max_value=data_max, key="rep_e")

        if st.button("Run Replay", type="primary", key="run_rep"):
            if sdate > edate:
                st.error("Start must be before End.")
            else:
                res = historical_replay(px_df, w, str(sdate), str(edate), notional=notional)
                rows_used = int(res.get("Rows", 0) or 0)
                c1, c2, c3, c4 = st.columns(4)
                with c1: metric_card("Return", f"{res.get('Return', 0):.2%}", positive=res.get("Return", 0) >= 0)
                with c2: metric_card("P&L (USD)", f"${res.get('PnL', 0):,.0f}", positive=res.get("PnL", 0) >= 0)
                with c3: metric_card("Max Drawdown", f"{res.get('MaxDD', 0):.2%}", positive=False)
                with c4: metric_card("Rows Used", str(rows_used))

                if rows_used < 2:
                    st.warning(f"Insufficient data for selected period (rows = {rows_used}). Available: {data_min} ~ {data_max}")

    # ── C: Predefined scenarios ────────────────────────────────────
    with tab3:
        section("Predefined Stress Scenarios")
        SCENARIOS = {
            "COVID Crash (Mar 2020)": {t: -0.35 for t in tickers},
            "GFC 2008 Peak (-40%)": {t: -0.40 for t in tickers},
            "Tech Selloff (-20%)": {t: -0.20 for t in ["AAPL", "MSFT", "GOOGL", "AMZN"] if t in tickers},
            "Rate Shock (+200bp) – bonds -10%": {t: -0.10 for t in ["AGG", "BND", "MBB", "SHY"] if t in tickers},
            "Mild Rally (+10%)": {t: 0.10 for t in tickers},
        }
        results = []
        for name, shock in SCENARIOS.items():
            s_vec = np.array([shock.get(t, 0.0) for t in w.index])
            r = float(np.dot(w.values, s_vec))
            pnl = notional * r
            results.append({"Scenario": name, "Return": r, "P&L (USD)": pnl})

        scn_df = pd.DataFrame(results)
        fig = px.bar(scn_df, x="Scenario", y="P&L (USD)", color="P&L (USD)",
                     color_continuous_scale="RdYlGn", template="plotly_dark", text_auto=",.0f")
        fig.update_layout(**PLOTLY_LAYOUT, title="Scenario P&L Summary", height=380)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            scn_df.style.format({"Return": "{:.2%}", "P&L (USD)": "${:,.0f}"}),
            use_container_width=True,
        )


# ════════════════════════════════════════════════════════════════════
# PAGE: Correlation & Rolling Beta
# ════════════════════════════════════════════════════════════════════

def page_corr_beta():
    st.title("🔗 Correlation & Rolling Beta")

    with st.sidebar:
        st.markdown("### Settings")
        raw_tickers = st.text_input("Tickers (comma-separated)", "AAPL,MSFT,GOOGL,AMZN,^GSPC")
        tickers = [t.strip().upper() for t in raw_tickers.split(",") if t.strip()]
        start = st.date_input("Start Date", value=date(2020, 1, 1))
        roll_w = st.number_input("Rolling Window (days)", 30, 1000, 126)
        bench = st.text_input("Benchmark", "^GSPC")

    if not tickers:
        st.info("Enter tickers.")
        return

    with st.spinner("Loading data…"):
        try:
            px_df = load_price_data(tickers, start=str(start))
        except Exception as e:
            st.error(f"Data error: {e}")
            return

    r = px_df.pct_change().dropna()

    tab1, tab2, tab3 = st.tabs(["Correlation Heatmap", "Rolling Beta", "Return Scatter"])

    with tab1:
        section("Correlation Matrix")
        corr = r.corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
        fig.update_layout(**PLOTLY_LAYOUT, title="Pairwise Correlation", height=450)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if bench in r.columns:
            section(f"Rolling {roll_w}d Beta vs {bench}")
            fig = go.Figure()
            for t in [x for x in r.columns if x != bench]:
                df_b = pd.concat([r[t], r[bench]], axis=1).dropna()
                beta_roll = (
                    df_b[t].rolling(int(roll_w)).cov(df_b[bench])
                    / df_b[bench].rolling(int(roll_w)).var()
                )
                fig.add_trace(go.Scatter(x=beta_roll.index, y=beta_roll.values, name=t, mode="lines"))
            fig.add_hline(y=1, line_dash="dot", line_color="gray", annotation_text="Beta=1")
            fig.update_layout(**PLOTLY_LAYOUT, title=f"Rolling Beta vs {bench}", height=420)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"Benchmark '{bench}' not in loaded tickers. Add it to the tickers list.")

    with tab3:
        section("Return Scatter Matrix")
        if len(r.columns) <= 6:
            fig = px.scatter_matrix(r, dimensions=r.columns.tolist(), template="plotly_dark")
            fig.update_layout(**PLOTLY_LAYOUT, height=600, title="Pairwise Return Scatter")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Showing top 6 assets for readability.")
            fig = px.scatter_matrix(r.iloc[:, :6], dimensions=r.columns[:6].tolist(), template="plotly_dark")
            fig.update_layout(**PLOTLY_LAYOUT, height=600)
            st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════
# PAGE: Credit Model
# ════════════════════════════════════════════════════════════════════

def page_credit():
    st.title("💳 Credit Risk – Scorecard Pipeline")
    st.caption("WoE / IV feature analysis, Logistic Regression, KS / AUC, Score Transform")

    file = st.file_uploader("Upload CSV with a `default` (0/1) column", type=["csv"])
    if not file:
        st.info("Upload a credit dataset to get started. The file must include a `default` column (0 = good, 1 = default).")
        return

    df = pd.read_csv(file)
    if "default" not in df.columns:
        st.error("The file must contain a column named `default` (0/1).")
        return

    st.success(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
    st.dataframe(df.head(5), use_container_width=True)

    numeric_cols = [c for c in df.columns if c != "default" and pd.api.types.is_numeric_dtype(df[c])]
    feats = st.multiselect("Select features for modelling", numeric_cols, default=numeric_cols[:min(5, len(numeric_cols))])
    bins = st.slider("WoE Bins", 4, 20, 10)

    if not feats:
        st.info("Select at least one feature.")
        return

    if st.button("Run WoE/IV + Train LR", type="primary"):
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score

        # ── WoE / IV ────────────────────────────────────────────────
        section("Information Value (IV) Ranking")
        iv_list = []
        for f in feats:
            woe_df, iv = calculate_woe_iv(df[[*feats, "default"]], f, "default", bins=bins)
            iv_list.append({"Feature": f, "IV": iv, "Predictive Power": "Strong" if iv > 0.3 else "Medium" if iv > 0.1 else "Weak"})
        iv_df = pd.DataFrame(iv_list).sort_values("IV", ascending=False)
        fig = px.bar(iv_df, x="Feature", y="IV", color="Predictive Power",
                     color_discrete_map={"Strong": "#4CAF50", "Medium": "#FF9800", "Weak": "#F44336"},
                     template="plotly_dark", text_auto=".3f")
        fig.update_layout(**PLOTLY_LAYOUT, height=350, title="IV per Feature")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(iv_df, use_container_width=True)

        # ── Logistic Regression ──────────────────────────────────────
        section("Logistic Regression")
        X = df[feats].fillna(df[feats].median())
        y = df["default"].astype(int)
        model = LogisticRegression(max_iter=1000).fit(X, y)
        proba = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, proba)

        s = pd.DataFrame({"y": y, "score": proba}).sort_values("score")
        s["cum_pos"] = (s["y"] == 1).cumsum() / max((s["y"] == 1).sum(), 1)
        s["cum_neg"] = (s["y"] == 0).cumsum() / max((s["y"] == 0).sum(), 1)
        ks = (s["cum_pos"] - s["cum_neg"]).abs().max()

        c1, c2 = st.columns(2)
        with c1: metric_card("AUC (in-sample)", f"{auc:.3f}", positive=auc > 0.7)
        with c2: metric_card("KS Statistic", f"{ks:.3f}", positive=ks > 0.3)

        # ── ROC Curve ───────────────────────────────────────────────
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y, proba)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, fill="tozeroy", name=f"ROC (AUC={auc:.3f})", line=dict(color="#1E88E5")))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="gray"), name="Random"))
        fig.update_layout(**PLOTLY_LAYOUT, title="ROC Curve", height=380, xaxis_title="FPR", yaxis_title="TPR")
        st.plotly_chart(fig, use_container_width=True)

        # ── Score Transform ──────────────────────────────────────────
        section("Score Distribution")
        scores = scorecard_transform(proba)
        result_df = pd.DataFrame({"PD": proba, "Score": scores, "Default": y})
        fig = px.histogram(result_df, x="Score", color="Default", nbins=50,
                           color_discrete_map={0: "#1E88E5", 1: "#F44336"},
                           template="plotly_dark", barmode="overlay", opacity=0.7)
        fig.update_layout(**PLOTLY_LAYOUT, title="Score Distribution by Outcome", height=380)
        st.plotly_chart(fig, use_container_width=True)

        st.download_button(
            "⬇ Download Scores CSV",
            data=result_df.to_csv(index=False).encode("utf-8"),
            file_name="credit_scores.csv",
            mime="text/csv",
        )


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
    st.title("📰 News & Sentiment Analysis")
    st.caption("Fetch financial news and score sentiment via LLM API")

    tab1, tab2, tab3 = st.tabs(["Live News Feed", "Sentiment Analysis", "Financial Report"])

    FEEDS = {
        "Yahoo Finance":  "https://finance.yahoo.com/news/rssindex",
        "MarketWatch":    "https://feeds.marketwatch.com/marketwatch/topstories/",
        "Reuters Finance":"https://feeds.reuters.com/reuters/businessNews",
        "Investing.com":  "https://www.investing.com/rss/news.rss",
    }

    # ── Tab 1: Fetch ──────────────────────────────────────────────────
    with tab1:
        section("RSS / News Fetch")
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_feed = st.selectbox("News Source", list(FEEDS.keys()))
        with col2:
            max_articles = st.number_input("Max Articles", 5, 50, 15)

        if st.button("Fetch News", type="primary"):
            try:
                import feedparser
                with st.spinner(f"Fetching from {selected_feed}…"):
                    feed = feedparser.parse(FEEDS[selected_feed])
                articles = []
                for entry in feed.entries[: int(max_articles)]:
                    summary = entry.get("summary", "")
                    # Strip HTML tags from summary
                    import re
                    summary = re.sub(r"<[^>]+>", "", summary)[:400]
                    articles.append({
                        "Title":     entry.get("title", ""),
                        "Published": entry.get("published", ""),
                        "Summary":   summary,
                        "Link":      entry.get("link", ""),
                    })
                if articles:
                    st.success(f"Fetched {len(articles)} articles from {selected_feed}")
                    st.session_state["news_articles"] = articles
                    st.session_state["news_source"] = selected_feed
                    for a in articles:
                        with st.expander(a["Title"]):
                            st.caption(a["Published"])
                            st.write(a["Summary"])
                            st.markdown(f"[Read more]({a['Link']})")
                else:
                    st.warning("No articles found. Try a different news source.")
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
                st.markdown(f"<div style='text-align:center;color:#555;font-size:0.8rem'>○ {s}</div>", unsafe_allow_html=True)

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
                card_border = "#1E88E5" if selected else "#2D3142"
                st.markdown(
                    f"""<div style='border:2px solid {card_border};border-radius:10px;padding:14px;margin:4px 0;
                    background:#1A1D27;cursor:pointer;'>
                    <div style='font-size:1.1rem;font-weight:700'>{info['label']}</div>
                    <div style='font-size:0.78rem;color:#9EA3B0;margin-top:4px'>{info['desc']}</div>
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
                .applymap(color_ret, subset=["1日%", "1月%", "3月%"])
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
    st.title("📦 Export Report")
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
                "Pages available:\n"
                "  1. Overview / Risk Dashboard\n"
                "  2. Portfolio Performance\n"
                "  3. Portfolio Risk (VaR/CVaR/MC)\n"
                "  4. VaR Backtest (Kupiec)\n"
                "  5. Scenarios & Stress\n"
                "  6. Correlation & Rolling Beta\n"
                "  7. Credit Model (WoE/IV/LR)\n"
                "  8. News & Sentiment\n"
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
# Router
# ════════════════════════════════════════════════════════════════════

PAGES = {
    "🏠 Overview": page_overview,
    "📈 Portfolio Performance": page_portfolio_performance,
    "⚠️ Portfolio Risk": page_portfolio_risk,
    "🔁 VaR Backtest": page_var_backtest,
    "💥 Scenarios & Stress": page_scenarios,
    "🔗 Correlation & Beta": page_corr_beta,
    "💳 Credit Model": page_credit,
    "📰 News & Sentiment": page_news_sentiment,
    "🏦 機構選股": page_stock_selector,
    "📦 Export Report": page_export,
}

PAGES[page]()
