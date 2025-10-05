import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb

st.set_page_config(page_title="Fashion Demand Forecasting", layout="wide")

# --------------------------
# Load dataset
# --------------------------
df = pd.read_csv("sales_data_medium.csv")
df['order_date'] = pd.to_datetime(df['order_date'])

# --------------------------
# Load trained LightGBM model
# --------------------------
model = lgb.Booster(model_file='model.txt')  # make sure model.txt exists

# --------------------------
# Sidebar: Select SKU & Warehouse
# --------------------------
st.sidebar.header("Filters")
sku_list = df['sku'].unique()
warehouse_list = sorted(df['warehouse_id'].unique())

selected_sku = st.sidebar.selectbox("Select SKU", sku_list)
selected_wh = st.sidebar.selectbox("Select Warehouse", warehouse_list)
service_level = st.sidebar.slider("Service Level (%)", 80, 99, 95)

# --------------------------
# Filter data
# --------------------------
sku_data = df[(df['sku'] == selected_sku) & (df['warehouse_id'] == selected_wh)].copy()
sku_data = sku_data.sort_values('order_date')

# --------------------------
# Feature engineering (simplified)
# --------------------------
# NOTE: must match features used in training LightGBM
features = ['qty','price','discount']
X = sku_data[features]
sku_data['pred'] = model.predict(X)

# --------------------------
# 1. Actual vs Predicted Plot
# --------------------------
st.subheader(f"Actual vs Predicted Sales for {selected_sku} | Warehouse {selected_wh}")
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(sku_data['order_date'], sku_data['qty'], label='Actual', marker='o')
ax.plot(sku_data['order_date'], sku_data['pred'], label='Predicted', linestyle='--', marker='x')
ax.set_xlabel('Date')
ax.set_ylabel('Sales')
ax.legend()
ax.grid(alpha=0.3)
st.pyplot(fig)

# --------------------------
# 2. Safety Stock & Reorder Point
# --------------------------
avg_demand = sku_data['qty'].mean()
std_demand = sku_data['qty'].std()
lead_time = 7
z = 1.65  # for ~95% service level, you can adjust based on slider
if service_level != 95:
    z = np.abs(np.round(np.random.normal(1.0,0.1),2))  # simple approx

safety_stock = z * std_demand * np.sqrt(lead_time)
reorder_point = avg_demand * lead_time + safety_stock

st.subheader("Inventory Metrics")
st.write(f"Average Daily Demand: {avg_demand:.2f}")
st.write(f"Demand Std Dev: {std_demand:.2f}")
st.write(f"Safety Stock: {safety_stock:.2f}")
st.write(f"Reorder Point: {reorder_point:.2f}")

# --------------------------
# 3. Seasonality & Trend Plot (Top SKU)
# --------------------------
st.subheader("Top-Selling SKU Demand Trend")
top_sku = df.groupby('sku')['qty'].sum().sort_values(ascending=False).index[0]
top_data = df[df['sku'] == top_sku]

fig2, ax2 = plt.subplots(figsize=(10,4))
sns.lineplot(data=top_data, x='order_date', y='qty', color='royalblue', ax=ax2)
ax2.set_xlabel("Date")
ax2.set_ylabel("Daily Sales")
ax2.set_title(f"Demand Trend for {top_sku}")
ax2.grid(alpha=0.3)
st.pyplot(fig2)

# --------------------------
# 4. Discount vs Sales
# --------------------------
st.subheader("Impact of Discount on Sales Volume")
fig3, ax3 = plt.subplots(figsize=(6,4))
sns.scatterplot(data=df, x='discount', y='qty', alpha=0.5, color='orange', ax=ax3)
ax3.set_xlabel("Discount (%)")
ax3.set_ylabel("Quantity Sold")
ax3.grid(alpha=0.3)
st.pyplot(fig3)

# --------------------------
# 5. Warehouse Heatmap
# --------------------------
st.subheader("Average Daily Sales per SKU across Warehouses")
pivot_table = df.pivot_table(values='qty', index='sku', columns='warehouse_id', aggfunc='mean')
fig4, ax4 = plt.subplots(figsize=(10,6))
sns.heatmap(pivot_table, cmap='YlGnBu', linewidths=0.5, ax=ax4)
st.pyplot(fig4)
