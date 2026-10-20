import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from datetime import datetime, timedelta

# -------------------------------
# App Layout
# -------------------------------
st.set_page_config(page_title="Stock Prediction Dashboard", layout="wide")
st.title("📈 Stock Prediction & Analysis Dashboard")
st.write("Real-time stock data, charts, and analysis with ML predictions")
st.write("Welcome! This app will show real-time predictions with charts.")

# -------------------------------
# Sidebar Inputs (Single Ticker)
# -------------------------------
st.sidebar.header("Select Stock")
ticker = st.sidebar.text_input("Enter Stock Symbol (e.g. AAPL, TSLA, MSFT)", "AAPL")

start_date = st.sidebar.date_input("Start Date", datetime.today() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", datetime.today())
ma1 = st.sidebar.slider("Short-term Moving Average (days)", 5, 50, 20)
ma2 = st.sidebar.slider("Long-term Moving Average (days)", 50, 200, 100)

# Optional: quick refresh button (not true realtime; grabs latest on click)
if st.sidebar.button("🔄 Refresh data"):
    st.cache_data.clear()

# -------------------------------
# Helpers
# -------------------------------
def _ensure_single_level_columns(df: pd.DataFrame) -> pd.DataFrame:
    """If yfinance returns a MultiIndex (e.g., ('Close','AAPL')), flatten to single level."""
    if isinstance(df.columns, pd.MultiIndex):
        # keep only the first level names: Open, High, Low, Close, Adj Close, Volume
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df

def _col_series(df: pd.DataFrame, name: str) -> pd.Series:
    """Return a 1D Series for a given column name even if df[name] is a 1-col DataFrame."""
    col = df[name]
    if isinstance(col, pd.DataFrame):
        col = col.iloc[:, 0]
    return pd.to_numeric(col, errors="coerce")

@st.cache_data
def load_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False, actions=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = _ensure_single_level_columns(df)
    # make sure index is tz-naive (sometimes yfinance returns tz-aware)
    try:
        df.index = df.index.tz_localize(None)
    except Exception:
        pass
    return df

# -------------------------------
# Fetch Data (Single Ticker)
# -------------------------------
data = load_data(ticker, start_date, end_date)

if data.empty:
    st.error("No data found! Check ticker symbol or date range.")
    st.stop()

# Extract Series for OHLC to avoid MultiIndex assignment issues
open_s  = _col_series(data, "Open")
high_s  = _col_series(data, "High")
low_s   = _col_series(data, "Low")
close_s = _col_series(data, "Close")
vol_s   = _col_series(data, "Volume")

# Precompute common fields
data[f"MA{ma1}"] = close_s.rolling(ma1).mean()
data[f"MA{ma2}"] = close_s.rolling(ma2).mean()
data["Returns"]  = close_s.pct_change()

# -------------------------------
# Tabs
# -------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Price Charts",
    "📉 Technical Indicators",
    "📈 Returns & Risk",
    "ℹ️ Metrics"
])

# -------------------------------
# Tab 1: Price Charts
# -------------------------------
with tab1:
    st.subheader(f"{ticker} — Line & Candlestick with Moving Averages")

    # Line Chart
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=data.index, y=close_s, mode='lines', name='Close'))
    fig_line.add_trace(go.Scatter(x=data.index, y=data[f"MA{ma1}"], mode='lines', name=f'{ma1}-Day MA'))
    fig_line.add_trace(go.Scatter(x=data.index, y=data[f"MA{ma2}"], mode='lines', name=f'{ma2}-Day MA'))
    fig_line.update_layout(hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_line, use_container_width=True)

    # Candlestick + MAs
    fig_candle = go.Figure(data=[go.Candlestick(
        x=data.index, open=open_s, high=high_s, low=low_s, close=close_s, name="Candlestick"
    )])
    fig_candle.add_trace(go.Scatter(x=data.index, y=data[f"MA{ma1}"], mode='lines', name=f'{ma1}-Day MA'))
    fig_candle.add_trace(go.Scatter(x=data.index, y=data[f"MA{ma2}"], mode='lines', name=f'{ma2}-Day MA'))
    fig_candle.update_layout(xaxis_rangeslider_visible=False, hovermode="x unified",
                             margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_candle, use_container_width=True)

# -------------------------------
# Tab 2: Technical Indicators
# -------------------------------
with tab2:
    st.subheader(f"{ticker} — RSI, MACD, Bollinger Bands")

    # --- RSI (14) ---
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    data["RSI"] = rsi

    # --- MACD (12,26,9) ---
    ema12 = close_s.ewm(span=12, adjust=False).mean()
    ema26 = close_s.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    data["MACD"] = macd
    data["Signal"] = signal

    # --- Bollinger Bands (20, 2σ) ---
    ma20 = close_s.rolling(20).mean()
    std20 = close_s.rolling(20).std()
    data["20MA"] = ma20
    data["Upper"] = ma20 + std20 * 2
    data["Lower"] = ma20 - std20 * 2

    # Plot Bollinger Bands with price
    fig_bb = go.Figure()
    fig_bb.add_trace(go.Scatter(x=data.index, y=close_s, mode='lines', name='Close'))
    fig_bb.add_trace(go.Scatter(x=data.index, y=data["Upper"], line=dict(dash='dot'), name='Upper BB'))
    fig_bb.add_trace(go.Scatter(x=data.index, y=data["Lower"], line=dict(dash='dot'), name='Lower BB'))
    fig_bb.update_layout(hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_bb, use_container_width=True)

    # RSI line
    st.line_chart(data["RSI"], height=220, use_container_width=True)

    # MACD vs Signal
    st.line_chart(data[["MACD", "Signal"]], height=220, use_container_width=True)

# -------------------------------
# Tab 3: Returns & Risk
# -------------------------------
with tab3:
    st.subheader(f"{ticker} — Returns & Risk")

    # Daily returns distribution
    fig, ax = plt.subplots()
    sns.histplot(data["Returns"].dropna(), bins=50, kde=True, ax=ax)
    ax.set_title(f"{ticker} Daily Returns Distribution")
    st.pyplot(fig)

    # Rolling 20-day volatility (std of returns)
    data["RollVol20"] = data["Returns"].rolling(20).std() * np.sqrt(252)
    st.line_chart(data["RollVol20"], height=220, use_container_width=True)

    # Cumulative returns
    cumret = (1 + data["Returns"].fillna(0)).cumprod() - 1
    st.line_chart(cumret, height=220, use_container_width=True)

    # OHLC correlation (single ticker)
    st.subheader("Correlation Matrix (OHLCV)")
    fig_corr, axc = plt.subplots()
    corr_df = data[["Open", "High", "Low", "Close", "Volume"]].apply(pd.to_numeric, errors="coerce").corr()
    sns.heatmap(corr_df, annot=True, cmap="coolwarm", ax=axc)
    st.pyplot(fig_corr)

# -------------------------------
# Tab 4: Performance Metrics
# -------------------------------
with tab4:
    st.subheader(f"{ticker} — Key Metrics")

    latest_close = float(close_s.dropna().iloc[-1])

    avg_return = float(data["Returns"].mean() * 100)
    volatility = float(data["Returns"].std() * np.sqrt(252) * 100)  # annualized
    total_volume = int(vol_s.sum())

    # Max Drawdown
    cum = (1 + data["Returns"].fillna(0)).cumprod()
    running_max = cum.cummax()
    drawdown = (cum / running_max) - 1
    max_dd = float(drawdown.min() * 100)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latest Closing Price", f"${latest_close:.2f}")
    col2.metric("Avg Daily Return", f"{avg_return:.2f}%")
    col3.metric("Annualized Volatility", f"{volatility:.2f}%")
    col4.metric("Total Volume Traded", f"{total_volume:,}")

    st.caption(f"Max Drawdown: **{max_dd:.2f}%**")

# -------------------------------
# Footer / Info
# -------------------------------
st.info("🚀 This dashboard can be extended with ML predictions using models saved in `results_archive/`. For now, it shows analytics & visualization.")
