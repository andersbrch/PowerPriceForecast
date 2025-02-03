import streamlit as st
import pandas as pd
import numpy as np
import requests
import statsmodels.api as sm
from arch import arch_model
from scipy.stats import norm
import warnings
import logging

import plotly.graph_objects as go

# Configure logging
logging.basicConfig(level=logging.INFO)

# Suppress warnings for a cleaner output
warnings.simplefilter("ignore")

# ---------------------------
# Data Loading and Preprocessing
# ---------------------------

# Add a button to manually clear cached data and force refresh
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
    logging.info("Data loaded successfully from Energidataservice API")
    return pd.DataFrame(data.get("records", []))

@st.cache_data(ttl=3600)
def preprocess_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the raw data: pivot and add seasonal features.
    """
    df = raw_df.pivot_table(index="HourUTC", columns="PriceArea", values="SpotPriceEUR")
    df.index = pd.to_datetime(df.index)
    
    # Add time-related features
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['hour_of_week'] = df.index.dayofweek * 24 + df.index.hour
    df['sin_daily'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_daily'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['sin_weekly'] = np.sin(2 * np.pi * df['hour_of_week'] / 168)
    df['cos_weekly'] = np.cos(2 * np.pi * df['hour_of_week'] / 168)
    
    logging.info("Data preprocessing complete with seasonal features.")
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
@st.cache_data(show_spinner=True, ttl=3600)
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
                # For demonstration, we use a fixed order (1,0,1)
                model = sm.tsa.SARIMAX(current_train_y, order=(1, 0, 1), exog=current_train_X)
                model_fit = model.fit(disp=False)
                forecast = model_fit.forecast(steps=1, exog=exog_future)
            except Exception as e:
                st.error(f"Forecasting error at index {i}: {e}")
                logging.error(f"Forecasting error at index {i}: {e}")
                continue
            forecasts.append(forecast.iloc[0])
            actuals.append(test_y.iloc[i])
            timestamps.append(test_y.index[i])
            # Expand training set with new observation
            current_train_y = pd.concat([current_train_y, pd.Series(test_y.iloc[i], index=[test_y.index[i]])])
            current_train_X = pd.concat([current_train_X, exog_future])
            
    results = pd.DataFrame({
        'HourUTC': timestamps,
        'Actual': actuals,
        'Forecast': forecasts
    }).set_index('HourUTC')
    
    logging.info("Expanding window forecast completed.")
    return results

# ---------------------------
# Plotting Functions using Plotly
# ---------------------------
def plot_actual_prices(area_df: pd.DataFrame, area: str):
    """
    Plot the last 7 days of actual price data using Plotly.
    """
    last_7days = area_df.index[-7 * 24:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=last_7days, y=area_df[area].loc[last_7days],
        mode='lines', name='Actual Price', line=dict(color='blue')
    ))
    fig.update_layout(
        title="Last 7 Days of Actual Prices",
        xaxis_title="Date",
        yaxis_title="Price (EUR)"
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_forecast(df: pd.DataFrame, forecast_df: pd.DataFrame, train_split: int, area: str):
    """
    Plot the forecasted prices alongside actual data using Plotly.
    """
    fig = go.Figure()
    # Plot actual prices for the last 7 days
    last_7days = df.index[-7 * 24:]
    fig.add_trace(go.Scatter(
        x=last_7days, y=df[area].loc[last_7days],
        mode='lines', name='Actual', line=dict(color='blue')
    ))
    # Plot forecasted values if available
    if not forecast_df.empty:
        fig.add_trace(go.Scatter(
            x=forecast_df.index, y=forecast_df['Forecast'],
            mode='lines', name='Forecast', line=dict(color='red', dash='dash')
        ))
    # Mark the forecast start point
    forecast_start = df.index[train_split]
    fig.add_vline(x=forecast_start, line=dict(color="black", dash="dot"), annotation_text="Forecast Start")
    fig.update_layout(
        title="Expanding Window Forecasting",
        xaxis_title="Date",
        yaxis_title="Price (EUR)"
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_garch_volatility(area_df: pd.DataFrame, area: str):
    """
    Plot GARCH(1,1) volatility versus absolute returns using Plotly.
    Returns the garch_volatility series and returns series (df_ret).
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
        logging.error(f"GARCH model fitting failed: {e}")
        return None, None

    last_7days = df_ret.index[-7 * 24:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=last_7days, y=np.abs(df_ret.loc[last_7days]),
        mode='markers', name="Absolute Returns", marker=dict(color="black", opacity=0.5)
    ))
    fig.add_trace(go.Scatter(
        x=last_7days, y=garch_volatility.loc[last_7days],
        mode='lines', name="GARCH(1,1) Volatility", line=dict(color="blue", dash="dash")
    ))
    fig.update_layout(
        title="GARCH(1,1) Volatility vs. Absolute Returns",
        xaxis_title="Date",
        yaxis_title="Volatility / Absolute Returns"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    return garch_volatility, df_ret

def plot_var_forecast(df_ret: pd.Series, garch_volatility: pd.Series):
    """
    Plot the one-step ahead Value-at-Risk (VaR) forecast using Plotly.
    """
    alpha = 0.01
    z_score = norm.ppf(alpha)
    VaR_1step = z_score * garch_volatility

    last_7days = df_ret.index[-7 * 24:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=last_7days, y=df_ret.loc[last_7days],
        mode='markers', name="Returns", marker=dict(color="black", opacity=0.5, size=5)
    ))
    fig.add_trace(go.Scatter(
        x=last_7days, y=VaR_1step.loc[last_7days],
        mode='lines', name="VaR (GARCH-N)", line=dict(color="red")
    ))
    fig.add_hline(y=0, line=dict(color="black", dash="dash"), annotation_text="Zero Line")
    fig.update_layout(
        title="One-Step Ahead VaR Forecasts using GARCH Model",
        xaxis_title="Date",
        yaxis_title="Returns / VaR"
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Streamlit App Layout
# ---------------------------

st.title("Electricity Price Forecasting Dashboard")
st.subheader("Anders Brodersen Christiansen")
st.write("Power spot price from [Energidataservice](https://www.energidataservice.dk/tso-electricity/Elspotprices)")

# Sidebar About section
st.sidebar.markdown("## About")
st.sidebar.info(
    "This dashboard demonstrates data ingestion, preprocessing, time series forecasting, "
    "and volatility modeling using electricity spot price data from Energidataservice. "
    "It includes seasonal ARMA forecasts and GARCH-based volatility and Value-at-Risk (VaR) analyses."
)

# Load and preprocess data
with st.spinner("Loading data..."):
    raw_df = load_raw_data()
    df = preprocess_data(raw_df)

if df.empty:
    st.error("No data available. Please try again later.")
    st.stop()

# Determine available price areas (excluding feature columns)
exclude_cols = ['hour', 'dayofweek', 'hour_of_week', 'sin_daily', 'cos_daily', 'sin_weekly', 'cos_weekly']
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

# Calculate and display forecast error metrics if forecast results are available
if not forecast_results.empty:
    mae = np.mean(np.abs(forecast_results['Actual'] - forecast_results['Forecast']))
    rmse = np.sqrt(np.mean((forecast_results['Actual'] - forecast_results['Forecast'])**2))
    st.write(f"**Forecast Error Metrics:** MAE = {mae:.2f}, RMSE = {rmse:.2f}")
    
    # Download button for forecast results
    csv = forecast_results.to_csv().encode('utf-8')
    st.download_button(
        label="Download Forecast Results as CSV",
        data=csv,
        file_name='forecast_results.csv',
        mime='text/csv'
    )

# GARCH volatility plot
st.subheader("GARCH(1,1) Volatility")
garch_volatility, df_ret = plot_garch_volatility(area_df, selected_area)

# VaR forecast plot (only if GARCH model ran successfully)
if (garch_volatility is not None) and (df_ret is not None):
    st.subheader("Value-at-Risk (VaR) Forecast")
    plot_var_forecast(df_ret, garch_volatility)
