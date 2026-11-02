"""
FinSight AI — Main Gradio Application
======================================
AI-Powered Financial Advisor & Risk Analyzer

Components:
  • Real-time stock data  via yfinance (free, unlimited)
  • LSTM price forecast   (60-step sliding window, 7-day autoregressive)
  • Risk analytics        (Sharpe, VaR, Beta, Max Drawdown, composite score)
  • Interactive Gradio UI (dark theme, tabbed layout)
  • MongoDB persistence   (user preferences + analysis history)
"""

import os
import sys

# ── Ensure project root is always on sys.path ─────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import gradio as gr
from dotenv import load_dotenv

from services.alpha_vantage import get_stock_data, get_company_info
from services.risk_calculator import full_risk_report
from model.lstm_predictor import predict_future_prices
from db.mongo_handler import save_analysis, save_preferences, get_analysis_history

load_dotenv()

# ═══════════════════════════════════════════════════════════════
# CHART HELPERS
# ═══════════════════════════════════════════════════════════════

DARK_BG    = "#0d0d1a"
PANEL_BG   = "#12122a"
ACCENT1    = "#4fc3f7"   # cyan-blue
ACCENT2    = "#7c4dff"   # violet
ACCENT3    = "#f06292"   # pink-red
GRID_COL   = "#1e1e3a"
TEXT_COLOR = "#c0c0d8"


def _style_fig(fig, ax_list):
    fig.patch.set_facecolor(DARK_BG)
    for ax in (ax_list if isinstance(ax_list, list) else [ax_list]):
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TEXT_COLOR, labelsize=8)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COL)
        ax.grid(True, color=GRID_COL, linewidth=0.5, linestyle="--", alpha=0.7)


def make_price_chart(df: pd.DataFrame, predictions: pd.DataFrame, symbol: str) -> plt.Figure:
    """Two-panel: price + forecast (top) | return distribution (bottom)."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 7),
        gridspec_kw={"height_ratios": [3, 1]},
        facecolor=DARK_BG
    )
    plt.subplots_adjust(hspace=0.4)
    _style_fig(fig, [ax1, ax2])

    # ─ Panel 1: Historical price + forecast ───────────────────
    ax1.plot(df.index, df["close"], color=ACCENT1, linewidth=1.8,
             label="Historical Close", zorder=3)
    ax1.fill_between(df.index, df["low"], df["high"],
                     color=ACCENT1, alpha=0.08, label="Daily Range")

    if not predictions.empty:
        ax1.plot(predictions.index, predictions["predicted"],
                 color=ACCENT3, linewidth=2, linestyle="--",
                 label="7-Day LSTM Forecast", zorder=4)
        ax1.fill_between(predictions.index,
                         predictions["lower_bound"],
                         predictions["upper_bound"],
                         color=ACCENT3, alpha=0.12, label="Confidence Band")

    ax1.set_title(f"{symbol.upper()} — Price History & 7-Day Forecast",
                  fontsize=13, fontweight="bold", pad=12)
    ax1.set_ylabel("Price (USD)", fontsize=9)
    ax1.legend(facecolor=PANEL_BG, edgecolor=GRID_COL,
               labelcolor=TEXT_COLOR, fontsize=8, loc="upper left")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.setp(ax1.get_xticklabels(), rotation=30, ha="right", fontsize=7)

    # ─ Panel 2: Return distribution ────────────────────────────
    returns = df["close"].pct_change().dropna() * 100
    ax2.hist(returns, bins=30, color=ACCENT2, alpha=0.8, edgecolor=GRID_COL)
    var_line = np.percentile(returns, 5)
    ax2.axvline(returns.mean(), color=ACCENT1, linestyle="--",
                linewidth=1.2, label=f"Mean {returns.mean():.2f}%")
    ax2.axvline(var_line, color=ACCENT3, linestyle="--",
                linewidth=1.2, label=f"VaR 5% {var_line:.2f}%")
    ax2.set_title("Daily Return Distribution", fontsize=10, pad=8)
    ax2.set_xlabel("Return (%)", fontsize=9)
    ax2.set_ylabel("Frequency", fontsize=9)
    ax2.legend(facecolor=PANEL_BG, edgecolor=GRID_COL,
               labelcolor=TEXT_COLOR, fontsize=8)

    plt.tight_layout()
    return fig


def make_gauge_chart(risk_score: float, label: str) -> plt.Figure:
    """Semicircular gauge: 0 (green) → 10 (red)."""
    fig, ax = plt.subplots(
        figsize=(5, 3.2), subplot_kw={"polar": True}, facecolor=DARK_BG
    )
    ax.set_facecolor(DARK_BG)

    theta = np.linspace(0, np.pi, 300)
    # Background track
    ax.plot(theta, np.ones(300) * 0.75, color=GRID_COL, linewidth=22,
            solid_capstyle="round")
    # Value arc
    fill_theta = np.linspace(0, np.pi * min(risk_score / 10, 1.0), 300)
    color = ("#22c55e" if risk_score < 3.5
             else "#eab308" if risk_score < 6.5
             else "#ef4444")
    ax.plot(fill_theta, np.ones(len(fill_theta)) * 0.75,
            color=color, linewidth=22, solid_capstyle="round")

    # Labels
    ax.text(np.pi / 2, 0.18, f"{risk_score:.1f}",
            ha="center", va="center", fontsize=26,
            fontweight="bold", color="white")
    ax.text(np.pi / 2, -0.05, "/ 10",
            ha="center", va="center", fontsize=12, color=TEXT_COLOR)
    clean_label = label.replace("🟢", "").replace("🟡", "").replace("🔴", "").strip()
    ax.text(np.pi / 2, -0.28, clean_label,
            ha="center", va="center", fontsize=11,
            fontweight="bold", color=color)

    ax.set_ylim(0, 1)
    ax.set_theta_zero_location("W")
    ax.set_theta_direction(1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["polar"].set_visible(False)
    fig.patch.set_facecolor(DARK_BG)
    return fig


def make_portfolio_chart(df_results: pd.DataFrame) -> plt.Figure:
    """Bar chart comparing risk scores across portfolio stocks."""
    if df_results.empty or "Risk Score" not in df_results.columns:
        fig, ax = plt.subplots(figsize=(8, 4), facecolor=DARK_BG)
        ax.set_facecolor(PANEL_BG)
        ax.text(0.5, 0.5, "No portfolio data", ha="center",
                va="center", color=TEXT_COLOR, transform=ax.transAxes)
        return fig

    fig, ax = plt.subplots(figsize=(10, 4), facecolor=DARK_BG)
    _style_fig(fig, ax)

    colors = ["#22c55e" if s < 3.5 else "#eab308" if s < 6.5 else "#ef4444"
              for s in df_results["Risk Score"]]
    bars = ax.bar(df_results["Symbol"], df_results["Risk Score"],
                  color=colors, edgecolor=GRID_COL, linewidth=0.5)
    ax.set_title("Portfolio Risk Scores", fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel("Risk Score (0–10)", fontsize=9)
    ax.set_ylim(0, 10)
    ax.axhline(3.5, color="#22c55e", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axhline(6.5, color="#ef4444", linestyle="--", linewidth=0.8, alpha=0.6)

    for bar, val in zip(bars, df_results["Risk Score"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                f"{val:.1f}", ha="center", va="bottom",
                color="white", fontsize=9, fontweight="bold")
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════
# CORE ANALYSIS
# ═══════════════════════════════════════════════════════════════

def analyze_stock(symbol: str, risk_tolerance: int, period: str, user_id: str):
    """
    Full pipeline:
      1. Fetch OHLCV (yfinance)
      2. Fetch company info
      3. LSTM forecast (lazy TF load, falls back to stats)
      4. Risk metrics (Sharpe, VaR, Beta, Drawdown)
      5. Charts
      6. Persist to MongoDB
    """
    symbol = symbol.strip().upper()
    if not symbol:
        return ({"error": "Enter a stock symbol (e.g. AAPL, TSLA, MSFT)"},
                None, None, "—", [], [])

    try:
        # ── 1. Market data ────────────────────────────────────
        df = get_stock_data(symbol, period=period)
        if df.empty:
            return ({"error": f"No data for '{symbol}'. Check the ticker symbol."},
                    None, None, "—", [], [])

        # ── 2. Company info ───────────────────────────────────
        info = get_company_info(symbol)
        company_header = (
            f"**{info['name']}** | {info['sector']} | {info['industry']}"
        )

        # ── 3. Forecast ───────────────────────────────────────
        predictions = predict_future_prices(df, symbol=symbol, days=7)

        # ── 4. Risk report ────────────────────────────────────
        report = full_risk_report(df, symbol=symbol, risk_tolerance=risk_tolerance)

        # ── 5. Charts ─────────────────────────────────────────
        price_fig = make_price_chart(df, predictions, symbol)
        gauge_fig = make_gauge_chart(report["risk_score"], report["risk_label"])

        # ── 6. Forecast table ─────────────────────────────────
        if not predictions.empty:
            forecast_rows = [
                [str(idx.date()), f"${row['predicted']:.2f}",
                 f"${row['lower_bound']:.2f}", f"${row['upper_bound']:.2f}"]
                for idx, row in predictions.iterrows()
            ]
        else:
            forecast_rows = []

        # ── 7. MongoDB (non-blocking) ─────────────────────────
        save_analysis(user_id, symbol, report)
        save_preferences(user_id, {
            "risk_tolerance": risk_tolerance,
            "last_symbol": symbol,
            "last_period": period
        })

        history = get_analysis_history(user_id, limit=8)
        history_rows = [
            [h.get("symbol", ""),
             str(h.get("timestamp", ""))[:19],
             h["analysis"].get("risk_label", ""),
             h["analysis"].get("risk_score", "")]
            for h in history if "analysis" in h
        ]

        return report, price_fig, gauge_fig, company_header, forecast_rows, history_rows

    except ConnectionError as e:
        return ({"error": str(e)}, None, None, "—", [], [])
    except Exception as e:
        return ({"error": f"Analysis failed: {str(e)}"}, None, None, "—", [], [])


def analyze_portfolio(file_path: str, risk_tolerance: int):
    """Analyze all stocks in a CSV portfolio file."""
    if not file_path:
        return pd.DataFrame({"message": ["Upload a CSV with a 'symbol' column"]}), None

    try:
        port_df = pd.read_csv(file_path)
        if "symbol" not in port_df.columns:
            return pd.DataFrame({"error": ["CSV must have a 'symbol' column"]}), None

        results = []
        for sym in port_df["symbol"].str.upper().unique():
            try:
                df = get_stock_data(sym, period="3mo")
                if df.empty:
                    results.append({"Symbol": sym, "Status": "No data"})
                    continue
                r = full_risk_report(df, sym, risk_tolerance)
                results.append({
                    "Symbol":      r["symbol"],
                    "Price":       f"${r['last_price']}",
                    "Sharpe":      r["sharpe_ratio"],
                    "Volatility":  r["annualized_volatility"],
                    "VaR 95%":     r["value_at_risk_95"],
                    "Max DD":      r["max_drawdown"],
                    "Beta":        r["beta"],
                    "Risk Score":  r["risk_score"],
                    "Risk":        r["risk_label"],
                    "Advice":      r["recommendation"]
                })
            except Exception as e:
                results.append({"Symbol": sym, "Error": str(e)})

        df_results = pd.DataFrame(results)
        chart = make_portfolio_chart(
            df_results[["Symbol", "Risk Score"]].dropna() if "Risk Score" in df_results else pd.DataFrame()
        )
        return df_results, chart

    except Exception as e:
        return pd.DataFrame({"error": [str(e)]}), None


# ═══════════════════════════════════════════════════════════════
# GRADIO UI
# ═══════════════════════════════════════════════════════════════

CUSTOM_CSS = """
/* ── Global ─────────────────────────────────────────── */
body, .gradio-container {
    background: #0d0d1a !important;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
}
.gr-panel, .gr-box, .svelte-1kyws56 { background: #12122a !important; }

/* ── Header ─────────────────────────────────────────── */
.finsight-header {
    text-align: center;
    padding: 24px 0 8px;
}
.finsight-title {
    font-size: 2.4em;
    font-weight: 800;
    background: linear-gradient(135deg, #4fc3f7 0%, #7c4dff 60%, #f06292 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}
.finsight-sub {
    color: #8080a0;
    font-size: 0.95em;
    margin-top: 6px;
}
.badge {
    display: inline-block;
    background: rgba(76,195,247,0.12);
    border: 1px solid rgba(76,195,247,0.3);
    color: #4fc3f7;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.78em;
    margin: 2px;
}

/* ── Primary button ──────────────────────────────────── */
.analyze-btn {
    background: linear-gradient(135deg, #4fc3f7, #7c4dff) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1em !important;
    letter-spacing: 0.03em !important;
    transition: opacity 0.2s, transform 0.1s !important;
}
.analyze-btn:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}

/* ── Metric cards ────────────────────────────────────── */
.metric-row { display: flex; gap: 12px; flex-wrap: wrap; margin: 8px 0; }
.metric-card {
    flex: 1; min-width: 120px;
    background: linear-gradient(135deg, #1a1a3e, #12122a);
    border: 1px solid #2a2a5a;
    border-radius: 10px;
    padding: 14px 16px;
    text-align: center;
}
.metric-label { color: #8080a0; font-size: 0.75em; margin-bottom: 4px; }
.metric-value { color: white; font-size: 1.4em; font-weight: 700; }

/* ── Hide footer ─────────────────────────────────────── */
footer { display: none !important; }
"""

with gr.Blocks(title="FinSight AI | Financial Advisor", css=CUSTOM_CSS) as demo:

    # ── Header ─────────────────────────────────────────────────
    gr.HTML("""
    <div class="finsight-header">
      <h1 class="finsight-title">📊 FinSight AI</h1>
      <p class="finsight-sub">AI-Powered Financial Advisor &amp; Risk Analyzer</p>
      <div style="margin-top:10px">
        <span class="badge">🧠 LSTM Forecasting</span>
        <span class="badge">📈 Real-Time Data</span>
        <span class="badge">⚡ Risk Analytics</span>
        <span class="badge">🗄️ MongoDB</span>
        <span class="badge">🐳 Docker Ready</span>
      </div>
    </div>
    """)

    # ── Main layout ─────────────────────────────────────────────
    with gr.Row(equal_height=False):

        # ── Left: Controls ──────────────────────────────────────
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("### 🎯 Analysis Parameters")

            company_banner = gr.Markdown("*Enter a symbol and click Analyze*")

            symbol_in = gr.Textbox(
                label="Stock Symbol",
                value="AAPL",
                placeholder="AAPL · MSFT · TSLA · GOOGL · AMZN",
                elem_id="symbol_input"
            )
            period_in = gr.Dropdown(
                choices=["1mo", "3mo", "6mo", "1y", "2y"],
                value="6mo",
                label="Historical Period",
                elem_id="period_select"
            )
            risk_in = gr.Slider(
                minimum=1, maximum=10, step=1, value=5,
                label="Risk Tolerance",
                info="1 = Very Conservative  ·  10 = Very Aggressive",
                elem_id="risk_slider"
            )
            user_in = gr.Textbox(
                label="User ID",
                value="demo_user",
                placeholder="Your ID for saving preferences",
                elem_id="user_input"
            )
            analyze_btn = gr.Button(
                "🔍  Analyze Stock",
                variant="primary",
                size="lg",
                elem_classes=["analyze-btn"]
            )

            gr.Markdown("---")
            gr.Markdown(
                "**Quick picks:** `AAPL` · `MSFT` · `TSLA` · `AMZN` · `GOOGL` · "
                "`NVDA` · `META` · `NFLX`"
            )

        # ── Right: Results ──────────────────────────────────────
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("📈 Forecast Chart", elem_id="tab_chart"):
                    chart_out = gr.Plot(elem_id="forecast_plot")

                with gr.Tab("🎯 Risk Gauge", elem_id="tab_gauge"):
                    gauge_out = gr.Plot(elem_id="gauge_plot")

                with gr.Tab("📋 Risk Report", elem_id="tab_report"):
                    report_out = gr.JSON(
                        label="Full Risk Analysis",
                        elem_id="risk_json"
                    )

                with gr.Tab("🔮 7-Day Forecast", elem_id="tab_forecast"):
                    forecast_table = gr.Dataframe(
                        headers=["Date", "Predicted Price", "Lower Bound", "Upper Bound"],
                        label="LSTM Price Forecast",
                        elem_id="forecast_table",
                        interactive=False
                    )

                with gr.Tab("🕐 History", elem_id="tab_history"):
                    history_table = gr.Dataframe(
                        headers=["Symbol", "Timestamp", "Risk Level", "Score"],
                        label="Your Analysis History",
                        elem_id="history_table",
                        interactive=False
                    )

    # ── Portfolio Section ───────────────────────────────────────
    with gr.Accordion("📂 Portfolio Analyzer", open=False, elem_id="portfolio_section"):
        gr.Markdown(
            "Upload a CSV with a **`symbol`** column. "
            "Each symbol is analyzed and risk-scored automatically."
        )
        gr.Markdown(
            "**Example CSV:**\n```\nsymbol\nAAPL\nMSFT\nTSLA\nGOOGL\n```"
        )
        with gr.Row():
            portfolio_file = gr.File(
                label="Portfolio CSV",
                file_types=[".csv"],
                type="filepath",
                elem_id="portfolio_file"
            )
        port_btn = gr.Button("Analyze Portfolio", variant="secondary",
                             elem_id="portfolio_btn")
        with gr.Row():
            port_table = gr.Dataframe(label="Portfolio Risk Summary",
                                      elem_id="portfolio_table")
            port_chart = gr.Plot(label="Risk Score Comparison",
                                 elem_id="portfolio_chart")

    # ── Footer ──────────────────────────────────────────────────
    gr.HTML("""
    <div style="text-align:center; padding:20px 0; color:#404060; font-size:0.82em">
      FinSight AI &nbsp;|&nbsp; Built with LSTM · yfinance · Gradio · MongoDB · Docker
      &nbsp;|&nbsp; Data via Yahoo Finance (real-time)
      &nbsp;|&nbsp; ⚠️ For educational purposes only — not financial advice
    </div>
    """)

    # ── Event Wiring ────────────────────────────────────────────
    analyze_btn.click(
        fn=analyze_stock,
        inputs=[symbol_in, risk_in, period_in, user_in],
        outputs=[report_out, chart_out, gauge_out,
                 company_banner, forecast_table, history_table]
    )

    port_btn.click(
        fn=analyze_portfolio,
        inputs=[portfolio_file, risk_in],
        outputs=[port_table, port_chart]
    )


# ═══════════════════════════════════════════════════════════════
# LAUNCH
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    print(f"\n🚀 Starting FinSight AI on http://localhost:{port}\n")
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True,
        share=False
    )