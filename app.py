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

def page_news_sentiment():
    st.title("📰 News & Sentiment Analysis")
    st.caption("Fetch financial news and score sentiment via LLM API")

    tab1, tab2 = st.tabs(["Live News Feed", "Sentiment Analysis"])

    with tab1:
        section("RSS / News Fetch")
        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input("Company / Ticker / Topic", "Apple AAPL")
        with col2:
            max_articles = st.number_input("Max Articles", 5, 50, 10)

        FEEDS = {
            "Reuters Finance": "https://feeds.reuters.com/reuters/businessNews",
            "Yahoo Finance": "https://finance.yahoo.com/news/rssindex",
            "MarketWatch": "https://feeds.marketwatch.com/marketwatch/topstories/",
            "Investing.com": "https://www.investing.com/rss/news.rss",
        }
        selected_feed = st.selectbox("News Source", list(FEEDS.keys()))

        if st.button("Fetch News", type="primary"):
            try:
                import feedparser

                feed = feedparser.parse(FEEDS[selected_feed])
                articles = []
                for entry in feed.entries[: int(max_articles)]:
                    articles.append(
                        {
                            "Title": entry.get("title", ""),
                            "Published": entry.get("published", ""),
                            "Summary": entry.get("summary", "")[:300] + "…",
                            "Link": entry.get("link", ""),
                        }
                    )
                if articles:
                    st.success(f"Fetched {len(articles)} articles from {selected_feed}")
                    st.session_state["news_articles"] = articles
                    for a in articles:
                        with st.expander(a["Title"]):
                            st.caption(a["Published"])
                            st.write(a["Summary"])
                            st.markdown(f"[Read more]({a['Link']})")
                else:
                    st.warning("No articles found.")
            except ImportError:
                st.error("feedparser not installed. Run: pip install feedparser")
            except Exception as e:
                st.error(f"Feed error: {e}")

    with tab2:
        section("LLM Sentiment Scoring")
        st.info(
            "Configure your LLM API key below to score the fetched articles. "
            "Supports OpenAI-compatible endpoints."
        )

        api_key = st.text_input("API Key", type="password", placeholder="sk-…")
        model_choice = st.selectbox(
            "Model",
            ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo", "claude-3-5-haiku-20241022"],
        )
        api_base = st.text_input("API Base URL (leave blank for OpenAI default)", "")

        articles = st.session_state.get("news_articles", [])
        if not articles:
            st.info("Fetch news from the 'Live News Feed' tab first.")
        elif st.button("Score Sentiment", type="primary"):
            if not api_key:
                st.error("Please enter your API key.")
            else:
                results = []
                progress = st.progress(0)
                for i, a in enumerate(articles):
                    try:
                        import openai

                        client = openai.OpenAI(
                            api_key=api_key,
                            base_url=api_base if api_base else None,
                        )
                        prompt = f"Rate the financial sentiment of this news headline and summary as POSITIVE, NEGATIVE, or NEUTRAL. Also give a score from -1.0 (very negative) to +1.0 (very positive). Reply in JSON: {{\"sentiment\": ..., \"score\": ...}}\n\nTitle: {a['Title']}\nSummary: {a['Summary']}"
                        resp = client.chat.completions.create(
                            model=model_choice,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0,
                        )
                        import json as _json
                        text = resp.choices[0].message.content.strip()
                        parsed = _json.loads(text)
                        results.append({
                            "Title": a["Title"][:80],
                            "Sentiment": parsed.get("sentiment", "N/A"),
                            "Score": parsed.get("score", 0.0),
                        })
                    except Exception as e:
                        results.append({"Title": a["Title"][:80], "Sentiment": "ERROR", "Score": 0.0})
                    progress.progress((i + 1) / len(articles))

                if results:
                    df_sent = pd.DataFrame(results)
                    avg = df_sent["Score"].mean()
                    st.metric("Average Sentiment Score", f"{avg:+.3f}", delta="Positive" if avg > 0.1 else "Negative" if avg < -0.1 else "Neutral")

                    fig = px.bar(
                        df_sent, x="Score", y="Title", orientation="h",
                        color="Score", color_continuous_scale="RdYlGn",
                        template="plotly_dark", text_auto=".2f",
                    )
                    fig.update_layout(**PLOTLY_LAYOUT, title="Sentiment Scores", height=max(300, 50 * len(df_sent)))
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(df_sent, use_container_width=True)


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
    "📦 Export Report": page_export,
}

PAGES[page]()
