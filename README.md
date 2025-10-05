# 🎯 FASHION DEMAND FORECASTING & INVENTORY OPTIMIZATION

### Interactive Live Demo for Fashion E-commerce Supply Chain Management

---

## 🔹 Overview
This project focuses on predicting future demand for fashion products and optimizing inventory using historical sales data and machine learning models. It simulates real-world e-commerce operations, providing insights into SKU-level sales trends and warehouse inventory management.

---

## 🔹 Key Features
- **Forecasting:** Predict SKU-level sales trends using LightGBM.  
- **Actual vs Predicted Visualization:** Compare actual sales with predictions.  
- **Inventory Metrics:** Calculate Safety Stock and Reorder Point for each SKU & warehouse.  
- **Trend Analysis:** Explore seasonality and demand trends of top-selling SKUs.  
- **Discount Analysis:** Understand the impact of discounts on sales volume.  
- **Warehouse Heatmap:** Visualize average daily sales across warehouses.  
- **Interactive Dashboard:** Fully interactive via Streamlit for real-time exploration.

---

## 🔹 Tech Stack
- **Python:** Data manipulation and computation  
- **Pandas & NumPy:** Dataset processing and numerical operations  
- **Streamlit:** Interactive live dashboard  
- **LightGBM:** Machine learning model for demand forecasting  
- **Matplotlib & Seaborn:** Professional data visualizations

---

## 🔹 Dataset
- Medium-sized synthetic dataset (`sales_data_medium.csv`) simulating fashion e-commerce sales.  
- Includes columns: `order_date`, `sku`, `warehouse_id`, `qty`, `price`, `discount`.

---

## 🔹 Live Demo
Experience the project live:  
[🌐 Click here for the Streamlit demo](https://saireddy81797-demand-forecasting-and-inventory-optim-app-has4ir.streamlit.app/)

---

## 🔹 How to Run Locally
1. Clone the repository:  
```bash
git clone https://github.com/yourusername/repo-name.git

pip install -r requirements.txt

streamlit run app.py
