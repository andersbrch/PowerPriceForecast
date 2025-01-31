import warnings
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import streamlit as st
from datetime import timedelta
from arch import arch_model
from scipy.stats import norm

# Suppress warnings
warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=sm.tools.sm_exceptions.ConvergenceWarning)

# Streamlit App Title
st.title("Electricity Price Forecasting Dashboard")

# Author
st.subheader("Anders Brodersen Christiansen")

# Description
st.write("Power spot price from Energinet: https://www.energidataservice.dk/tso-electricity/Elspotprices")

# Set limit on data to fetch
limit = 16800

@st.cache_data
def load_data():
    response = requests.get(url=f'https://api.energidataservice.dk/dataset/Elspotprices?limit={limit}')
    data = response.json()
    records = data.get("records", [])
    return pd.DataFrame(records)

raw_df = load_data()

# Pivoting the dataframe to get prices in Euros for each PriceArea for each timestamp
df = raw_df.pivot_table(index="HourUTC", columns="PriceArea", values="SpotPriceEUR")
df.index = pd.to_datetime(df.index)  # Ensure index is a DateTimeIndex

# Dropdown menu for selecting PowerPriceArea
power_price_area = st.selectbox("Select Power Price Area", df.columns)

# Handle missing values dynamically
df = df.dropna(subset=[power_price_area])  # Drop rows where selected area has missing values

# Feature Engineering
df['hour'] = df.index.hour / 60
df['sin_daily'] = np.sin(2 * np.pi * df['hour'] / 24)
df['cos_daily'] = np.cos(2 * np.pi * df['hour'] / 24)
df['sin_weekly'] = np.sin(2 * np.pi * df['hour'] / 168)
df['cos_weekly'] = np.cos(2 * np.pi * df['hour'] / 168)

y = df[power_price_area]
X_seasonal = df[['sin_daily', 'cos_daily', 'sin_weekly', 'cos_weekly']]
X_seasonal = sm.add_constant(X_seasonal)

# Train-Test Split
test_size = 24
train_size = len(df) - test_size  # Adjust dynamically based on available data

train_y = y.iloc[:train_size]
train_X = X_seasonal.iloc[:train_size]

test_y = y.iloc[train_size:]
test_X = X_seasonal.iloc[train_size:]

# Expanding Window Forecasting
expanding_forecast = []
expanding_actual = []
expanding_timestamps = []

current_train_y = train_y.copy()
current_train_X = train_X.copy()

for i in range(len(test_y)):  # Ensure valid index range
    if pd.notna(test_y.iloc[i]):  # Skip missing values
        exog_future = test_X.iloc[[i]]
        model = sm.tsa.SARIMAX(current_train_y, order=(1, 0, 1), exog=current_train_X).fit(disp=False)
        forecast = model.forecast(steps=1, exog=exog_future)
        expanding_forecast.append(forecast.iloc[0])
        expanding_actual.append(test_y.iloc[i])
        expanding_timestamps.append(test_y.index[i])

        # Append new data for expanding window
        current_train_y = pd.concat([current_train_y, pd.Series(test_y.iloc[i], index=[test_y.index[i]])])
        current_train_X = pd.concat([current_train_X, exog_future])

# Convert to DataFrame
expanding_results = pd.DataFrame({'HourUTC': expanding_timestamps, 'Actual': expanding_actual, 'Forecast': expanding_forecast}).set_index('HourUTC')

# ARMA Forecast Plot
st.subheader("Seasonal ARMA(1,1) one-step ahead forecast")
fig, ax = plt.subplots(figsize=(12, 6))

# Only plot available data
ax.plot(df.index[-7*24:], df[power_price_area].iloc[-7*24:], label='Actual', color='blue')

# Plot only available forecasted results
if not expanding_results.empty:
    ax.plot(expanding_results.index, expanding_results['Forecast'], label='Forecast', color='red', linestyle="--")

ax.axvline(df.index[train_size], color="black", linestyle="dashed", label="Forecast Start")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.set_title("Expanding Window Forecasting")
ax.legend()
st.pyplot(fig)

# GARCH Model
st.subheader("GARCH(1,1) Volatility")
df_ret = df[power_price_area].diff().dropna()

# Handle missing data in returns before GARCH modeling
if not df_ret.empty:
    garch_model = arch_model(df_ret, vol='GARCH', p=1, q=1)
    garch_fit = garch_model.fit(disp="off")
    garch_volatility = garch_fit.conditional_volatility

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(df_ret.index[-7*24:], np.abs(df_ret)[-7*24:], label="Absolute Returns", color="black", alpha=0.5)
    ax.plot(df_ret.index[-7*24:], garch_volatility[-7*24:], label="GARCH(1,1) Volatility", color="blue", linestyle="--")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility / Absolute Returns")
    ax.set_title("GARCH(1,1) Volatility vs. Absolute Returns")
    ax.legend()
    st.pyplot(fig)
else:
    st.write("Not enough return data available for GARCH modeling.")

# VaR Forecast
st.subheader("Value-at-Risk (VaR) Forecast")

if not df_ret.empty:
    alpha = 0.01
    z_score = norm.ppf(alpha)
    VaR_1step = z_score * garch_volatility

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(df_ret.index[-7*24:], df_ret[-7*24:], color="black", alpha=0.5, label="Returns", s=10)
    ax.plot(df_ret.index[-7*24:], VaR_1step[-7*24:], color="red", linestyle="-", label="VaR (GARCH-N)")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Returns / VaR")
    ax.set_title("One-Step Ahead VaR Forecasts using GARCH Model")
    ax.legend()
    st.pyplot(fig)
else:
    st.write("Not enough data available for VaR computation.")
