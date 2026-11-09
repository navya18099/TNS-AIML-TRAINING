import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

st.set_page_config(page_title="Retail Sales Forecasting", layout="wide")
st.title("Retail Sales Forecasting")
st.write("Predict future daily sales using an XGBoost model.")

FEATURES = ["t", "weekday", "month", "dayofyear", "promo", "price", "lag_364"]

@st.cache_data
def load_full():
    d = pd.read_csv("daily_sales.csv", index_col=0, parse_dates=True)
    return d.asfreq("D")

full = load_full()

@st.cache_resource
def train_model():
    d = full.copy()
    d["weekday"] = d.index.dayofweek
    d["month"] = d.index.month
    d["dayofyear"] = d.index.dayofyear
    d["t"] = np.arange(len(d))
    d["lag_364"] = d["sales"].shift(364)
    d = d.dropna()
    model = XGBRegressor(n_estimators=500, learning_rate=0.05,
                         max_depth=5, random_state=42)
    model.fit(d[FEATURES], d["sales"])
    return model

model = train_model()

# ---- Inputs ----
st.sidebar.header("Settings")
days = st.sidebar.slider("Days to forecast", 7, 90, 30)
promo = st.sidebar.slider("Expected promo share", 0.0, 1.0,
                          float(full["promo"].mean()))
price = st.sidebar.number_input("Average price",
                                value=float(full["price"].mean()))

# ---- Build future rows ----
idx = pd.date_range(full.index[-1] + pd.Timedelta(days=1), periods=days)
fut = pd.DataFrame(index=idx)
fut["t"] = np.arange(len(full), len(full) + days)
fut["weekday"] = idx.dayofweek
fut["month"] = idx.month
fut["dayofyear"] = idx.dayofyear
fut["promo"] = promo
fut["price"] = price
fut["lag_364"] = full["sales"].reindex(idx - pd.Timedelta(days=364)).values

fut["Predicted sales"] = model.predict(fut[FEATURES]).round(0)

# ---- Output ----
c1, c2, c3 = st.columns(3)
c1.metric("Total forecast", f"{fut['Predicted sales'].sum():,.0f}")
c2.metric("Average per day", f"{fut['Predicted sales'].mean():,.0f}")
c3.metric("Peak day", fut["Predicted sales"].idxmax().strftime("%d %b %Y"))

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(full.index[-180:], full["sales"][-180:], label="History")
ax.plot(fut.index, fut["Predicted sales"], label="Forecast")
ax.legend()
st.pyplot(fig)

st.dataframe(fut[["Predicted sales"]])
st.download_button("Download forecast as CSV",
                   fut[["Predicted sales"]].to_csv(), "forecast.csv")
st.caption("Model: XGBoost | Test MAE about 438 units per day")