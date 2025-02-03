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

# Add a button to manually clear cached data and force refresh
if st.button("Refresh Dashboard Data"):
    st.cache_data.clear()

# ---------------------------
# Data Loading and Preprocessing
# ---------------------------

@st.cache_data(ttl=3600)  # Auto-refresh cache every hour
def load_raw_data(limit=16800):
    """
    Load raw data from the Energidataservice API.
    """
    url = f'https://api.energidataservice.dk/dataset/Elspotprices?limit={limit}'
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    return pd.DataFrame(data.get("records", []))

@st.cache_data(ttl=3600)  # Ensure preprocessing updates with new data
def preprocess_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the raw data: pivot and add seasonal features.
    """
    df = raw_df.pivot_table(index="HourUTC", columns="PriceArea", values="SpotPriceEUR")
    df.index = pd.to_datetime(df.index)

    # Add an hour column (for seasonal feature creation)
    df['hour'] = df.index.hour
    df['sin_daily'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_daily'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['sin_weekly'] = np.sin(2 * np.pi * df['hour'] / 168)
    df['cos_weekly'] = np.cos(2 * np.pi * df['hour'] / 168)
    return df

@st.cache_data(ttl=3600)
def filter_area_data(df: pd.DataFrame, area: str) -> pd.DataFrame:
    """
    Filter data for the selected price area by dropping rows with missing values.
    """
    return df.dropna(subset=[area])

# ---------------------------
# Forecasting Functions
# ---------------------------

@st.cache_data(ttl=3600)
def compute_expanding_forecast(y: pd.Series, X: pd.DataFrame, test_size: int = 24) -> pd.DataFrame:
    """
    Perform an expanding window forecast using SARIMAX.
    Returns a DataFrame with actual values and forecasts.
    """
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
            # Expand the training set with the new observation
            current_train_y = pd.concat([current_train_y, pd.Series(test_y.iloc[i], index=[test_y.index[i]])])
            current_train_X = pd.concat([current_train_X, exog_future])

    results = pd.DataFrame({
        'HourUTC': timestamps,
        'Actual': actuals,
        'Forecast': forecasts
    }).set_index('HourUTC')

    return results

# ---------------------------
# Plotting Functions
# ---------------------------

def plot_actual_prices(area_df: pd.DataFrame, area: str):
    """Plot the last 7 days of actual price data."""
    fig, ax = plt.subplots(figsize=(12, 6))
    last_7days = area_df.index[-7 * 24:]
    ax.plot(last_7days, area_df[area].loc[last_7days], label='Actual Price', color='blue')
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (EUR)")
    ax.set_title("Last 7 Days of Actual Prices")
    ax.legend()
    st.pyplot(fig)

def plot_forecast(df: pd.DataFrame, forecast_df: pd.DataFrame, train_split: int, area: str):
    """Plot the forecasted prices alongside actual data."""
    fig, ax = plt.subplots(figsize=(12, 6))
    last_7days = df.index[-7 * 24:]
    ax.plot(last_7days, df[area].loc[last_7days], label='Actual', color='blue')

    if not forecast_df.empty:
        ax.plot(forecast_df.index, forecast_df['Forecast'], label='Forecast', color='red', linestyle="--")

    forecast_start = df.index[train_split]
    ax.axvline(forecast_start, color="black", linestyle="dashed", label="Forecast Start")

    ax.set_xlabel("Date")
    ax.set_ylabel("Price (EUR)")
    ax.set_title("Expanding Window Forecasting")
    ax.legend()
    st.pyplot(fig)

# ---------------------------
# Streamlit App Layout
# ---------------------------

st.title("Electricity Price Forecasting Dashboard")
st.subheader("Anders Brodersen Christiansen")
st.write("Power spot price from [Energidataservice](https://www.energidataservice.dk/tso-electricity/Elspotprices)")

# Sidebar for configuration
st.sidebar.header("Settings")

# Load and preprocess data
with st.spinner("Loading data..."):
    raw_df = load_raw_data()
    df = preprocess_data(raw_df)

if df.empty:
    st.error("No data available. Please try again later.")
    st.stop()

# Determine available price areas (excluding feature columns)
exclude_cols = ['hour', 'sin_daily', 'cos_daily', 'sin_weekly', 'cos_weekly']
price_areas = [col for col in df.columns if col not in exclude_cols]
selected_area = st.sidebar.selectbox("Select Power Price Area", price_areas)

# Filter data for the selected area
area_df = filter_area_data(df, selected_area)

# Plot the actual price data for the selected area
st.subheader("Actual Price Data")
plot_actual_prices(area_df, selected_area)

# Prepare forecasting features and target
features = ['sin_daily', 'cos_daily', 'sin_weekly', 'cos_weekly']
X = sm.add_constant(area_df[features])
y = area_df[selected_area]

# Determine train-test split for forecasting
test_size = 24
train_split = len(area_df) - test_size

# Compute the expanding window forecast
with st.spinner("Computing forecast..."):
    forecast_results = compute_expanding_forecast(y, X, test_size=test_size)

st.subheader("Seasonal ARMA(1,1) One-Step Ahead Forecast")
plot_forecast(area_df, forecast_results, train_split, selected_area)
