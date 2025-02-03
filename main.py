import streamlit as st
import pandas as pd
import numpy as np
import requests
import statsmodels.api as sm
from arch import arch_model
from scipy.stats import norm
import warnings

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose

# ---------------------------
# Settings and Warnings
# ---------------------------
warnings.simplefilter("ignore")
st.set_page_config(layout="wide", page_title="Electricity Price Forecasting Dashboard")

# ---------------------------
# Caching Functions
# ---------------------------
if st.sidebar.button("Refresh Dashboard Data"):
    st.cache_data.clear()

@st.cache_data(ttl=3600)
def load_raw_data(limit=16800):
    """
    Load raw data from the Energidataservice API.
    """
    url = f'https://api.energidataservice.dk/dataset/Elspotprices?limit={limit}'
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    return pd.DataFrame(data.get("records", []))

@st.cache_data(ttl=3600)
def preprocess_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the raw data: pivot table and add seasonal features.
    """
    df = raw_df.pivot_table(index="HourUTC", columns="PriceArea", values="SpotPriceEUR")
    df.index = pd.to_datetime(df.index)
    # Add seasonal features
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
# Forecasting Function
# ---------------------------
@st.cache_data(show_spinner=True, ttl=3600)
def compute_expanding_forecast(y: pd.Series, X: pd.DataFrame, test_size: int = 24, order: tuple = (1, 0, 1)) -> pd.DataFrame:
    """
    Perform an expanding window forecast using SARIMAX.
    Returns a DataFrame with actual values and forecasts.
    """
    train_size = len(y) - test_size
    train_y, train_X = y.iloc[:train_size], X.iloc[:train_size]
    test_y, test_X = y.iloc[train_size:], X.iloc[train_size:]
    
    forecasts, actuals, timestamps = [], [], []
    current_train_y, current_train_X = train_y.copy(), train_X.copy()
    
    progress_bar = st.progress(0)
    total_steps = len(test_y)
    
    for i in range(total_steps):
        if pd.notna(test_y.iloc[i]):
            exog_future = test_X.iloc[[i]]
            try:
                model = sm.tsa.SARIMAX(current_train_y, order=order, exog=current_train_X)
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
        progress_bar.progress((i + 1) / total_steps)
    progress_bar.empty()

    results = pd.DataFrame({
        'HourUTC': timestamps,
        'Actual': actuals,
        'Forecast': forecasts
    }).set_index('HourUTC')

    return results

def compute_forecast_metrics(forecast_df: pd.DataFrame):
    """
    Compute forecast error metrics.
    """
    forecast_df = forecast_df.dropna()
    mae = np.mean(np.abs(forecast_df['Actual'] - forecast_df['Forecast']))
    rmse = np.sqrt(np.mean((forecast_df['Actual'] - forecast_df['Forecast'])**2))
    mape = np.mean(np.abs((forecast_df['Actual'] - forecast_df['Forecast']) / forecast_df['Actual'])) * 100
    return mae, rmse, mape

# ---------------------------
# Plotting Functions (using Plotly)
# ---------------------------
def plot_actual_prices(area_df: pd.DataFrame, area: str):
    """
    Plot the last 7 days of actual price data.
    """
    last_7days = area_df.index[-7 * 24:]
    fig = px.line(
        x=last_7days, 
        y=area_df.loc[last_7days, area],
        labels={"x": "Date", "y": "Price (EUR)"},
        title="Last 7 Days of Actual Prices"
    )
    fig.update_traces(name="Actual Price", line=dict(color="blue"))
    st.plotly_chart(fig, use_container_width=True)

def plot_forecast(df: pd.DataFrame, forecast_df: pd.DataFrame, train_split: int, area: str):
    """
    Plot the forecasted prices alongside actual data.
    """
    fig = go.Figure()
    last_7days = df.index[-7 * 24:]
    fig.add_trace(go.Scatter(
        x=last_7days,
        y=df.loc[last_7days, area],
        mode='lines',
        name='Actual',
        line=dict(color="blue")
    ))
    if not forecast_df.empty:
        fig.add_trace(go.Scatter(
            x=forecast_df.index,
            y=forecast_df['Forecast'],
            mode='lines',
            name='Forecast',
            line=dict(color="red", dash="dash")
        ))
    # Convert forecast_start to a native Python datetime to avoid issues
    forecast_start = df.index[train_split].to_pydatetime()
    fig.add_vline(
        x=forecast_start,
        line=dict(color="black", dash="dot"),
        annotation_text="Forecast Start"
    )
    fig.update_layout(
        title="Expanding Window Forecasting",
        xaxis_title="Date",
        yaxis_title="Price (EUR)"
    )
    st.plotly_chart(fig, use_container_width=True)

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

    last_7days = df_ret.index[-7 * 24:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=last_7days,
        y=np.abs(df_ret.loc[last_7days]),
        mode='markers',
        name="Absolute Returns",
        marker=dict(color="black", opacity=0.5)
    ))
    fig.add_trace(go.Scatter(
        x=last_7days,
        y=garch_volatility.loc[last_7days],
        mode='lines',
        name="GARCH(1,1) Volatility",
        line=dict(color="blue", dash="dash")
    ))
    fig.update_layout(
        title="GARCH(1,1) Volatility vs. Absolute Returns",
        xaxis_title="Date",
        yaxis_title="Volatility / Absolute Returns"
    )
    st.plotly_chart(fig, use_container_width=True)
    return garch_volatility, df_ret

def plot_var_forecast(df_ret: pd.Series, garch_volatility: pd.Series, confidence_level: float):
    """
    Plot the one-step ahead Value-at-Risk (VaR) forecast.
    """
    alpha = 1 - confidence_level  # e.g., confidence_level=0.99 => alpha=0.01
    z_score = norm.ppf(alpha)
    VaR_1step = z_score * garch_volatility

    last_7days = df_ret.index[-7 * 24:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=last_7days,
        y=df_ret.loc[last_7days],
        mode='markers',
        name="Returns",
        marker=dict(color="black", opacity=0.5)
    ))
    fig.add_trace(go.Scatter(
        x=last_7days,
        y=VaR_1step.loc[last_7days],
        mode='lines',
        name="VaR (GARCH-N)",
        line=dict(color="red")
    ))
    fig.add_hline(y=0, line=dict(color="black", dash="dot"))
    fig.update_layout(
        title="One-Step Ahead VaR Forecasts using GARCH Model",
        xaxis_title="Date",
        yaxis_title="Returns / VaR"
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_time_series_decomposition(area_df: pd.DataFrame, area: str):
    """
    Plot a time series decomposition (observed, trend, seasonal, residual).
    """
    try:
        # For hourly data, a period of 24 corresponds to daily seasonality.
        result = seasonal_decompose(area_df[area].dropna(), model='additive', period=24)
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                            subplot_titles=["Observed", "Trend", "Seasonal", "Residual"])
        fig.add_trace(go.Scatter(x=result.observed.index, y=result.observed, name="Observed"), row=1, col=1)
        fig.add_trace(go.Scatter(x=result.trend.index, y=result.trend, name="Trend"), row=2, col=1)
        fig.add_trace(go.Scatter(x=result.seasonal.index, y=result.seasonal, name="Seasonal"), row=3, col=1)
        fig.add_trace(go.Scatter(x=result.resid.index, y=result.resid, name="Residual"), row=4, col=1)
        fig.update_layout(height=800, title_text="Time Series Decomposition")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Time series decomposition failed: {e}")

def download_button(data: pd.DataFrame, filename: str, button_label: str):
    """
    Provide a download button for a DataFrame as CSV.
    """
    csv = data.to_csv().encode('utf-8')
    st.download_button(button_label, csv, file_name=filename, mime='text/csv')

# ---------------------------
# Main App Layout
# ---------------------------
st.title("Electricity Price Forecasting Dashboard")
st.subheader("Anders Brodersen Christiansen")
st.write("Power spot price data sourced from [Energidataservice](https://www.energidataservice.dk/tso-electricity/Elspotprices)")

# Sidebar: Forecast and Model Settings
st.sidebar.header("Settings")
forecast_horizon = st.sidebar.slider("Forecast Horizon (hours)", min_value=1, max_value=168, value=24)
order_p = st.sidebar.number_input("SARIMAX Order p", min_value=0, max_value=5, value=1, step=1)
order_d = st.sidebar.number_input("SARIMAX Order d", min_value=0, max_value=2, value=0, step=1)
order_q = st.sidebar.number_input("SARIMAX Order q", min_value=0, max_value=5, value=1, step=1)
var_confidence = st.sidebar.slider("VaR Confidence Level", min_value=0.90, max_value=0.99, value=0.99, step=0.01)

# ---------------------------
# Data Loading and Preprocessing
# ---------------------------
with st.spinner("Loading raw data..."):
    raw_df = load_raw_data()
    df = preprocess_data(raw_df)

if df.empty:
    st.error("No data available. Please try again later.")
    st.stop()

# Sidebar: Date Range Selection
min_date = df.index.min().date()
max_date = df.index.max().date()
date_range = st.sidebar.date_input("Select Date Range", value=(min_date, max_date))
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
    df = df[(df.index.date >= start_date) & (df.index.date <= end_date)]
else:
    st.error("Please select a valid date range.")
    st.stop()

# Determine available price areas (exclude feature columns)
exclude_cols = ['hour', 'sin_daily', 'cos_daily', 'sin_weekly', 'cos_weekly']
price_areas = [col for col in df.columns if col not in exclude_cols]
selected_area = st.sidebar.selectbox("Select Power Price Area", price_areas)

# Filter data for the selected area
area_df = filter_area_data(df, selected_area)

# ---------------------------
# Display Actual Price Data
# ---------------------------
st.subheader("Actual Price Data")
plot_actual_prices(area_df, selected_area)

# ---------------------------
# Forecasting
# ---------------------------
st.subheader("Seasonal ARMA Forecast (Expanding Window)")
features = ['sin_daily', 'cos_daily', 'sin_weekly', 'cos_weekly']
X = sm.add_constant(area_df[features])
y = area_df[selected_area]

train_split = len(area_df) - forecast_horizon

with st.spinner("Computing forecast..."):
    forecast_results = compute_expanding_forecast(
        y, X, test_size=forecast_horizon, order=(order_p, order_d, order_q)
    )

# Plot forecast results
plot_forecast(area_df, forecast_results, train_split, selected_area)

# Compute and display forecast error metrics
if not forecast_results.empty:
    mae, rmse, mape = compute_forecast_metrics(forecast_results)
    st.write(f"**Forecast Metrics:** MAE = {mae:.2f}, RMSE = {rmse:.2f}, MAPE = {mape:.2f}%")
    download_button(forecast_results, "forecast_results.csv", "Download Forecast Results CSV")

# ---------------------------
# GARCH Volatility & VaR Forecast
# ---------------------------
st.subheader("GARCH(1,1) Volatility Analysis")
garch_volatility, df_ret = plot_garch_volatility(area_df, selected_area)

if (garch_volatility is not None) and (df_ret is not None):
    st.subheader("Value-at-Risk (VaR) Forecast")
    plot_var_forecast(df_ret, garch_volatility, var_confidence)

# ---------------------------
# Time Series Decomposition
# ---------------------------
with st.expander("Show Time Series Decomposition"):
    plot_time_series_decomposition(area_df, selected_area)

# ---------------------------
# Download Raw Data Option
# ---------------------------
with st.expander("Download Raw Data"):
    download_button(raw_df, "raw_data.csv", "Download Raw Data CSV")
