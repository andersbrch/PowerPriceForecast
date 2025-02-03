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
# Data Loading and Preprocessing
# ---------------------------

# Add a button to manually clear cached data and force refresh
if st.sidebar.button("Refresh Dashboard Data"):
    st.cache_data.clear()

@st.cache_data(ttl=3600)  # Refresh cache every hour
def load_raw_data(limit=16800):
    """
    Load raw data from the Energidataservice API.
    """
    url = f'https://api.energidataservice.dk/dataset/Elspotprices?limit={limit}'
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    return pd.DataFrame(data.get("records", []))


@st.cache_data(ttl=3600)  # Refresh cache every hour
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

@st.cache_data(ttl=3600)  # Refresh cache every hour
def filter_area_data(df: pd.DataFrame, area: str) -> pd.DataFrame:
    """
    Filter data for the selected price area by dropping rows with missing values.
    """
    return df.dropna(subset=[area])


# ---------------------------
# Forecasting Functions
# ---------------------------
@st.cache_data(show_spinner=True, ttl=3600)
def compute_expanding_forecast(y: pd.Series, X: pd.DataFrame, p: int = 1, d: int = 0, q: int = 1, test_size: int = 24) -> pd.DataFrame:
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
    """
    Plot the last 7 days of actual price data.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    last_7days = area_df.index[-7 * 24:]
    ax.plot(last_7days, area_df[area].loc[last_7days], label='Actual Price', color='blue')
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (EUR)")
    ax.set_title("Last 7 Days of Actual Prices")
    ax.legend()
    st.pyplot(fig)


def plot_forecast(df: pd.DataFrame, forecast_df: pd.DataFrame, train_split: int, area: str):
    """
    Plot the forecasted prices alongside actual data.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot actual prices for the last 7 days
    last_7days = df.index[-7 * 24:]
    ax.plot(last_7days, df[area].loc[last_7days], label='Actual', color='blue')

    # Plot forecasted values if available
    if not forecast_df.empty:
        ax.plot(forecast_df.index, forecast_df['Forecast'], label='Forecast', color='red', linestyle="--")

    # Mark the forecast start point
    forecast_start = df.index[train_split]
    ax.axvline(forecast_start, color="black", linestyle="dashed", label="Forecast Start")

    ax.set_xlabel("Date")
    ax.set_ylabel("Price (EUR)")
    ax.set_title("Expanding Window Forecasting")
    ax.legend()
    st.pyplot(fig)


def plot_garch_volatility(area_df: pd.DataFrame, area: str):
    """
    Plot GARCH(1,1) volatility versus absolute returns.
    """
    df_ret = area_df[area].diff().dropna()
    if df_ret.empty:
        st.write("Not enough return data available for GARCH modeling.")
        return None, None

    try:
        garch_model = arch_model(df_ret, vol='GARCH', p=1, q=1)
        garch_fit = garch_model.fit(disp="off")
        garch_volatility = garch_fit.conditional_volatility
    except Exception as e:
        st.error(f"GARCH model fitting failed: {e}")
        return None, None

    fig, ax = plt.subplots(figsize=(12, 6))
    last_7days = df_ret.index[-7 * 24:]
    ax.scatter(last_7days, np.abs(df_ret.loc[last_7days]), label="Absolute Returns", color="black", alpha=0.5)
    ax.plot(last_7days, garch_volatility.loc[last_7days], label="GARCH(1,1) Volatility", color="blue", linestyle="--")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility / Absolute Returns")
    ax.set_title("GARCH(1,1) Volatility vs. Absolute Returns")
    ax.legend()
    st.pyplot(fig)

    return garch_volatility, df_ret


def plot_var_forecast(df_ret: pd.Series, garch_volatility: pd.Series):
    """
    Plot the one-step ahead Value-at-Risk (VaR) forecast.
    """
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

# Adding option to change model
order_p = st.sidebar.number_input("SARIMAX Order p", min_value=0, max_value=5, value=1, step=1)
order_d = st.sidebar.number_input("SARIMAX Order d", min_value=0, max_value=2, value=0, step=1)
order_q = st.sidebar.number_input("SARIMAX Order q", min_value=0, max_value=5, value=1, step=1)

# Compute the expanding window forecast
with st.spinner("Computing forecast..."):
    forecast_results = compute_expanding_forecast(y, X, order_p, order_d, order_q, test_size=test_size)

st.subheader(f"Seasonal ARIMA({order_p},{order_d},{order_q}) One-Step Ahead Forecast")
plot_forecast(area_df, forecast_results, train_split, selected_area)

# GARCH volatility plot
st.subheader("GARCH(1,1) Volatility")
garch_volatility, df_ret = plot_garch_volatility(area_df, selected_area)

# VaR forecast plot (only if GARCH model ran successfully)
if (garch_volatility is not None) and (df_ret is not None):
    st.subheader("Value-at-Risk (VaR) Forecast")
    plot_var_forecast(df_ret, garch_volatility)
