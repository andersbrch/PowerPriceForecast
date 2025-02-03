import streamlit as st
import pandas as pd
import numpy as np
import requests
import statsmodels.api as sm
import matplotlib.pyplot as plt
from arch import arch_model
from scipy.stats import norm
import warnings

# Set matplotlib backend for non-interactive plotting
import matplotlib
matplotlib.use("Agg")

# Suppress warnings for a cleaner output
warnings.simplefilter("ignore")

# ---------------------------
# Manual Cache Clearing
# ---------------------------

if st.sidebar.button("Refresh Dashboard Data"):
    st.cache_data.clear()

# ---------------------------
# Data Loading and Preprocessing
# ---------------------------

@st.cache_data(ttl=3600)  
def load_raw_data(limit=16800):
    url = f'https://api.energidataservice.dk/dataset/Elspotprices?limit={limit}'
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    return pd.DataFrame(data.get("records", []))

@st.cache_data(ttl=3600)  
def preprocess_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.pivot_table(index="HourUTC", columns="PriceArea", values="SpotPriceEUR")
    df.index = pd.to_datetime(df.index)
    df['hour'] = df.index.hour
    df['sin_daily'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_daily'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['sin_weekly'] = np.sin(2 * np.pi * df['hour'] / 168)
    df['cos_weekly'] = np.cos(2 * np.pi * df['hour'] / 168)
    return df

@st.cache_data(ttl=3600)  
def filter_area_data(df: pd.DataFrame, area: str) -> pd.DataFrame:
    return df.dropna(subset=[area])

@st.cache_data(ttl=3600)  
def compute_expanding_forecast(y: pd.Series, X: pd.DataFrame, test_size: int = 24) -> pd.DataFrame:
    train_size = len(y) - test_size
    train_y, train_X = y.iloc[:train_size], X.iloc[:train_size]
    test_y, test_X = y.iloc[train_size:], X.iloc[train_size:]

    forecasts, actuals, timestamps = [], [], []
    current_train_y, current_train_X = train_y.copy(), train_X.copy()

    for i in range(len(test_y)):
        if pd.notna(test_y.iloc[i]):
            exog_future = test_X.iloc[[i]]
            try:
                model = sm.tsa.SARIMAX(current_train_y, order=(1, 0, 1), exog=current_train_X)
                model_fit = model.fit(disp=False)
                forecast = model_fit.forecast(steps=1, exog=exog_future)
            except Exception as e:
                st.error(f"Forecasting error at index {i}: {e}")
                continue
            forecasts.append(forecast.iloc[0])
            actuals.append(test_y.iloc[i])
            timestamps.append(test_y.index[i])
            current_train_y = pd.concat([current_train_y, pd.Series(test_y.iloc[i], index=[test_y.index[i]])])
            current_train_X = pd.concat([current_train_X, exog_future])

    return pd.DataFrame({'HourUTC': timestamps, 'Actual': actuals, 'Forecast': forecasts}).set_index('HourUTC')

# ---------------------------
# GARCH Model & VaR Functions
# ---------------------------

@st.cache_data(ttl=3600)  
def compute_garch_volatility(area_df: pd.DataFrame, area: str):
    df_ret = area_df[area].diff().dropna()
    if df_ret.empty:
        return None, None

    try:
        garch_model = arch_model(df_ret, vol='GARCH', p=1, q=1)
        garch_fit = garch_model.fit(disp="off")
        garch_volatility = garch_fit.conditional_volatility
        return garch_volatility, df_ret
    except Exception as e:
        st.error(f"GARCH model fitting failed: {e}")
        return None, None

def plot_garch_volatility(df_ret: pd.Series, garch_volatility: pd.Series):
    fig, ax = plt.subplots(figsize=(12, 6))
    last_7days = df_ret.index[-7 * 24:]
    ax.scatter(last_7days, np.abs(df_ret.loc[last_7days]), label="Absolute Returns", color="black", alpha=0.5)
    ax.plot(last_7days, garch_volatility.loc[last_7days], label="GARCH(1,1) Volatility", color="blue", linestyle="--")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility / Absolute Returns")
    ax.set_title("GARCH(1,1) Volatility vs. Absolute Returns")
    ax.legend()
    st.pyplot(fig)

def plot_var_forecast(df_ret: pd.Series, garch_volatility: pd.Series):
    alpha = 0.01
    z_score = norm.ppf(alpha)
    VaR_1step = z_score * garch_volatility

    fig, ax = plt.subplots(figsize=(12, 6))
    last_7days = df_ret.index[-7 * 24:]
    ax.scatter(last_7days, df_ret.loc[last_7days], color="black", alpha=0.5, label="Returns", s=10)
    ax.plot(last_7days, VaR_1step.loc[last_7days], color="red", linestyle="-", label="VaR (GARCH-N)")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Returns / VaR")
    ax.set_title("One-Step Ahead VaR Forecasts using GARCH Model")
    ax.legend()
    st.pyplot(fig)

# ---------------------------
# Streamlit App Layout
# ---------------------------

st.title("Electricity Price Forecasting Dashboard")
st.subheader("Anders Brodersen Christiansen")
st.write("Power spot price from [Energidataservice](https://www.energidataservice.dk/tso-electricity/Elspotprices)")

st.sidebar.header("Settings")

with st.spinner("Loading data..."):
    raw_df = load_raw_data()
    df = preprocess_data(raw_df)

if df.empty:
    st.error("No data available. Please try again later.")
    st.stop()

exclude_cols = ['hour', 'sin_daily', 'cos_daily', 'sin_weekly', 'cos_weekly']
price_areas = [col for col in df.columns if col not in exclude_cols]
selected_area = st.sidebar.selectbox("Select Power Price Area", price_areas)

area_df = filter_area_data(df, selected_area)

st.subheader("Actual Price Data")
plot_garch_volatility(area_df[selected_area], selected_area)

features = ['sin_daily', 'cos_daily', 'sin_weekly', 'cos_weekly']
X = sm.add_constant(area_df[features])
y = area_df[selected_area]

test_size = 24
train_split = len(area_df) - test_size

with st.spinner("Computing forecast..."):
    forecast_results = compute_expanding_forecast(y, X, test_size=test_size)

st.subheader("Seasonal ARMA(1,1) One-Step Ahead Forecast")
plot_var_forecast(area_df[selected_area], forecast_results)

garch_volatility, df_ret = compute_garch_volatility(area_df, selected_area)
if garch_volatility is not None:
    st.subheader("GARCH(1,1) Volatility")
    plot_garch_volatility(df_ret, garch_volatility)

    st.subheader("Value-at-Risk (VaR) Forecast")
    plot_var_forecast(df_ret, garch_volatility)
