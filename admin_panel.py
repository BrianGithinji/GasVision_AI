import streamlit as st
import pandas as pd
from pymongo import MongoClient
from datetime import datetime

# MongoDB Connection
@st.cache_resource
def init_connection():
    return MongoClient("mongodb://localhost:27017/")

client = init_connection()
db = client.gasvision_db
orders_collection = db.orders
customers_collection = db.customers

st.set_page_config(
    page_title="GasVision AI - Admin Panel",
    layout="wide"
)

st.title("🔧 GasVision AI - Admin Panel")

# Sidebar navigation
page = st.sidebar.selectbox("Select Page", ["Orders", "Customers", "Analytics"])

if page == "Orders":
    st.header("📦 Orders Management")
    
    # Display all orders
    all_orders = list(orders_collection.find())
    if all_orders:
        df_orders = pd.DataFrame(all_orders)
        df_orders = df_orders.drop('_id', axis=1)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Orders", len(all_orders))
        with col2:
            total_gas = sum(order['amount'] for order in all_orders)
            st.metric("Total Gas Ordered", f"{total_gas} kg")
        with col3:
            confirmed_orders = len([o for o in all_orders if o['status'] == 'Confirmed'])
            st.metric("Confirmed Orders", confirmed_orders)
        with col4:
            unique_customers = len(set(order['phone'] for order in all_orders))
            st.metric("Unique Customers", unique_customers)
        
        # Orders table
        st.dataframe(df_orders, use_container_width=True)
        
        # Export option
        if st.button("Export Orders to CSV"):
            df_orders.to_csv("orders_export.csv", index=False)
            st.success("Orders exported to orders_export.csv")
    else:
        st.info("No orders in database yet.")

elif page == "Customers":
    st.header("👥 Customers Management")
    
    all_customers = list(customers_collection.find())
    if all_customers:
        df_customers = pd.DataFrame(all_customers)
        df_customers = df_customers.drop('_id', axis=1)
        st.dataframe(df_customers, use_container_width=True)
    else:
        st.info("No customers in database yet.")

elif page == "Analytics":
    st.header("📊 Analytics Dashboard")
    
    all_orders = list(orders_collection.find())
    if all_orders:
        df = pd.DataFrame(all_orders)
        
        # Orders by day
        df['date'] = pd.to_datetime(df['date'])
        daily_orders = df.groupby(df['date'].dt.date).size()
        st.line_chart(daily_orders)
        
        # Gas amount distribution
        st.bar_chart(df['amount'].value_counts())
    else:
        st.info("No data available for analytics.")