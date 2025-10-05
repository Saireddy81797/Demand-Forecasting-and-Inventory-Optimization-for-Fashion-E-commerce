#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your dataset and model
df = pd.read_csv("sales_data_medium.csv")
df['order_date'] = pd.to_datetime(df['order_date'])

# Placeholder: replace with your trained LightGBM model
import lightgbm as lgb
model = lgb.Booster(model_file='model.txt')  # save your trained model first

st.title("Fashion Demand Forecasting & Inventory Optimization Demo")
st.write("Interactive demo for recruiters to explore SKU-level forecasts and inventory metrics.")

# Sidebar: select SKU and Warehouse
sku_list = df['sku'].unique().tolist()
warehouse_list = sorted(df['warehouse_id'].unique())

selected_sku = st.sidebar.selectbox("Select SKU", sku_list)
selected_wh = st.sidebar.selectbox("Select Warehouse", warehouse_list)

# Filter data
sku_data = df[(df['sku'] == selected_sku) & (df['warehouse_id'] == selected_wh)].copy()
sku_data = sku_data.sort_values('order_date')

# Features for prediction (use same features as model training)
features = ['qty','price','discount']  # modify according to your LightGBM features
X = sku_data[features]
sku_data['pred'] = model.predict(X)

# Plot Actual vs Forecast
st.subheader(f"Actual vs Predicted Sales for {selected_sku} (Warehouse {selected_wh})")
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(sku_data['order_date'], sku_data['qty'], label='Actual', marker='o')
ax.plot(sku_data['order_date'], sku_data['pred'], label='Predicted', linestyle='--', marker='x')
ax.set_xlabel('Date')
ax.set_ylabel('Sales')
ax.legend()
st.pyplot(fig)

# Inventory calculation
avg_demand = sku_data['qty'].mean()
std_demand = sku_data['qty'].std()
lead_time = 7
service_level = st.sidebar.slider("Service Level (%)", 80, 99, 95)
z = 1.65 if service_level == 95 else np.abs(np.round(np.random.normal(1.0,0.1),2))
safety_stock = z * std_demand * np.sqrt(lead_time)
reorder_point = avg_demand * lead_time + safety_stock

st.subheader("Inventory Metrics")
st.write(f"Average Daily Demand: {avg_demand:.2f}")
st.write(f"Demand Std Dev: {std_demand:.2f}")
st.write(f"Safety Stock: {safety_stock:.2f}")
st.write(f"Reorder Point: {reorder_point:.2f}")

