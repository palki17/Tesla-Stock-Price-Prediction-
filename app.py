"""
Tesla Stock Price Prediction — Streamlit App
=============================================
Run with:  streamlit run streamlit_app.py

Put all these files in the SAME folder:
    - streamlit_app.py
    - TSLA.csv
    - tsla_simplernn_model.keras
    - tsla_lstm_model.keras
    - tsla_tuned_lstm_model.keras
    - tsla_scaler.pkl
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TSLA Stock Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e2130; padding: 10px; border-radius: 8px; }
    h1 { color: #e31937; }
</style>
""", unsafe_allow_html=True)

# ─── Title ────────────────────────────────────────────────────────────────────
st.title("🚗 Tesla (TSLA) Stock Price Predictor")
st.markdown("**Deep Learning prediction using SimpleRNN and LSTM models**")
st.markdown("---")

WINDOW_SIZE = 60


# ─── Load Data ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('TSLA.csv', parse_dates=['Date'], index_col='Date')
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df


# ─── Load Models & Scaler ─────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    from tensorflow.keras.models import load_model

    rnn_model  = load_model('tsla_simplernn_model.keras')
    lstm_model = load_model('tsla_lstm_model.keras')
    best_lstm  = load_model('tsla_tuned_lstm_model.keras')

    with open('tsla_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    return rnn_model, lstm_model, best_lstm, scaler


# ─── Load everything ──────────────────────────────────────────────────────────
df = load_data()

try:
    rnn_model, lstm_model, best_lstm, scaler = load_models()
    models_loaded = True
except Exception as e:
    st.error(f"⚠️ Could not load models. Make sure all .keras and .pkl files are in the same folder.\n\nError: {e}")
    models_loaded = False


# ─── Helper Functions ─────────────────────────────────────────────────────────
def predict_future(model, last_window_scaled, n_days, scaler):
    predictions  = []
    curr_window  = last_window_scaled.copy()
    for _ in range(n_days):
        x_in  = curr_window.reshape(1, WINDOW_SIZE, 1)
        pred  = model.predict(x_in, verbose=0)[0, 0]
        predictions.append(pred)
        curr_window = np.append(curr_window[1:], [[pred]], axis=0)
    return scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1)
    ).flatten()


# ─── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/e/e8/Tesla_logo.png", width=80)
st.sidebar.title("⚙️ Settings")

model_choice  = st.sidebar.selectbox(
    "Select Model",
    ["LSTM", "Tuned LSTM (Best)", "SimpleRNN"]
)
forecast_days = st.sidebar.slider("Forecast Days", 1, 30, 10)
show_volume   = st.sidebar.checkbox("Show Volume Chart", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Dataset Info")
st.sidebar.metric("Total Records",  f"{len(df):,}")
st.sidebar.metric("Date Range",     f"{df.index.min().year} – {df.index.max().year}")
st.sidebar.metric("Last Close",     f"${df['Adj Close'].iloc[-1]:.2f}")
st.sidebar.metric("Highest Ever",   f"${df['Adj Close'].max():.2f}")
st.sidebar.metric("Lowest Ever",    f"${df['Adj Close'].min():.2f}")

st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ by Palki | Deep Learning Project")


# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Price History",
    "🔮 Forecast",
    "📊 Model Comparison",
    "📋 Raw Data"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Price History
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Tesla Historical Stock Price (2010 – 2020)")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Price",  f"${df['Adj Close'].iloc[-1]:.2f}")
    col2.metric("All Time High",  f"${df['Adj Close'].max():.2f}")
    col3.metric("All Time Low",   f"${df['Adj Close'].min():.2f}")
    avg = df['Adj Close'].mean()
    col4.metric("Average Price",  f"${avg:.2f}")

    st.markdown("")

    # Moving averages
    ma30 = df['Adj Close'].rolling(30).mean()
    ma90 = df['Adj Close'].rolling(90).mean()

    rows = 2 if show_volume else 1
    fig, axes = plt.subplots(rows, 1,
                             figsize=(12, 7 if show_volume else 4),
                             sharex=True,
                             facecolor='#0e1117')

    if not show_volume:
        axes = [axes]

    for ax in axes:
        ax.set_facecolor('#0e1117')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')

    axes[0].plot(df.index, df['Adj Close'], color='#00d4ff', linewidth=1.2, label='Adj Close')
    axes[0].plot(df.index, ma30,            color='#ffa500', linewidth=1,   label='30-Day MA')
    axes[0].plot(df.index, ma90,            color='#e31937', linewidth=1,   label='90-Day MA')
    axes[0].set_ylabel("Price (USD)", color='white')
    axes[0].set_title("TSLA Adjusted Closing Price", color='white')
    axes[0].legend(facecolor='#1e2130', labelcolor='white')

    if show_volume:
        axes[1].bar(df.index, df['Volume'], color='#7b68ee', alpha=0.6, width=1)
        axes[1].set_ylabel("Volume",        color='white')
        axes[1].set_title("Daily Trading Volume", color='white')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Forecast
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader(f"🔮 {forecast_days}-Day Price Forecast — {model_choice}")

    if not models_loaded:
        st.warning("⚠️ Models not loaded. Please ensure all .keras and .pkl files are present.")
    else:
        model_map = {
            "LSTM":               lstm_model,
            "Tuned LSTM (Best)":  best_lstm,
            "SimpleRNN":          rnn_model,
        }
        selected_model = model_map[model_choice]

        # Prepare last window
        scaled_all  = scaler.transform(df[['Adj Close']].values)
        last_window = scaled_all[-WINDOW_SIZE:].reshape(WINDOW_SIZE, 1)

        with st.spinner(f"⏳ Forecasting next {forecast_days} trading days..."):
            future_prices = predict_future(selected_model, last_window, forecast_days, scaler)

        future_dates = pd.bdate_range(start=df.index[-1], periods=forecast_days + 1)[1:]

        # Metrics
        last_price  = df['Adj Close'].iloc[-1]
        final_price = future_prices[-1]
        change_pct  = ((final_price - last_price) / last_price) * 100
        direction   = "📈" if change_pct > 0 else "📉"

        col1, col2, col3 = st.columns(3)
        col1.metric("Last Known Price",       f"${last_price:.2f}")
        col2.metric(f"Predicted (Day {forecast_days})", f"${final_price:.2f}")
        col3.metric(f"{direction} Change",    f"{change_pct:+.2f}%",
                    delta_color="normal")

        st.markdown("")

        col_chart, col_table = st.columns([2, 1])

        with col_chart:
            fig2, ax = plt.subplots(figsize=(10, 4), facecolor='#0e1117')
            ax.set_facecolor('#0e1117')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_edgecolor('#444')

            last_actual = df['Adj Close'].iloc[-60:]
            ax.plot(last_actual.index, last_actual.values,
                    label='Actual (last 60 days)', color='#00d4ff', linewidth=1.8)
            ax.plot(future_dates, future_prices,
                    label=f'{forecast_days}-Day Forecast',
                    color='#ffa500', marker='o', linestyle='--', markersize=4)
            ax.axvline(x=df.index[-1], color='#e31937',
                       linestyle=':', linewidth=1.5, label='Forecast Start')
            ax.set_title(f"TSLA — {forecast_days}-Day Forecast using {model_choice}",
                         color='white')
            ax.set_ylabel("Price (USD)", color='white')
            ax.legend(facecolor='#1e2130', labelcolor='white')
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

        with col_table:
            st.markdown("**📅 Forecast Table**")
            forecast_df = pd.DataFrame({
                'Date':  [d.strftime('%b %d, %Y') for d in future_dates],
                'Price': [f"${p:.2f}" for p in future_prices]
            })
            st.dataframe(forecast_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Comparison
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("SimpleRNN vs LSTM — Architecture & Performance")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔴 SimpleRNN")
        st.markdown("""
        - Basic recurrent network
        - Passes hidden state forward
        - ❌ Forgets long-term patterns
        - ✅ Fast to train
        - ✅ Good for short sequences
        """)
        st.code("""
Input (60 days)
      ↓
SimpleRNN (64 units)
      ↓
Dropout (0.2)
      ↓
SimpleRNN (32 units)
      ↓
Dropout (0.2)
      ↓
Dense(1) → Price
        """)

    with col2:
        st.markdown("### 🔵 LSTM")
        st.markdown("""
        - Advanced recurrent network
        - Has 3 gates (forget/input/output)
        - ✅ Remembers long-term patterns
        - ✅ Better accuracy on stock data
        - ✅ Handles vanishing gradient
        """)
        st.code("""
Input (60 days)
      ↓
LSTM (64 units) ← 3 gates
      ↓
Dropout (0.2)
      ↓
LSTM (32 units)
      ↓
Dropout (0.2)
      ↓
Dense(1) → Price
        """)

    st.markdown("---")
    st.markdown("### 📊 Why LSTM wins on Tesla data")
    st.info("""
    Tesla stock had a **massive bull run in 2019–2020** (price went from ~$60 to ~$780).
    This required the model to remember patterns from **months ago** — not just days.
    
    - SimpleRNN forgets information after ~20 time steps (vanishing gradient problem)
    - LSTM's forget gate decides what to keep from months back
    - Result: LSTM captures the long-term upward trend much better
    
    **Expected results:** LSTM will have lower RMSE, lower MAE, and higher R² than SimpleRNN.
    """)

    st.markdown("### 🔧 Hyperparameters Tuned")
    tuning_df = pd.DataFrame({
        'Parameter':   ['Units', 'Dropout Rate', 'Learning Rate'],
        'Values Tried': ['32, 64', '0.2, 0.3', '0.001, 0.0005'],
        'What it controls': [
            'Model capacity / complexity',
            'Prevents overfitting',
            'How fast the model learns'
        ]
    })
    st.dataframe(tuning_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Raw Data
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("📋 Raw Dataset")

    col1, col2 = st.columns(2)
    col1.metric("Total Rows",    f"{len(df):,}")
    col2.metric("Total Columns", len(df.columns))

    st.markdown("**Filter by Year:**")
    years     = sorted(df.index.year.unique())
    sel_years = st.multiselect("Select years to view",
                               options=years,
                               default=[2019, 2020])

    if sel_years:
        filtered = df[df.index.year.isin(sel_years)]
    else:
        filtered = df

    st.dataframe(filtered.style.format("{:.2f}"),
                 use_container_width=True,
                 height=400)

    st.markdown(f"Showing **{len(filtered):,}** rows")
