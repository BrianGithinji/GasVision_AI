import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(
    page_title="GasVision AI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------
# CUSTOM CSS
# ---------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Roboto', sans-serif;
}

.main-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: linear-gradient(135deg, #004d40 0%, #00695c 100%);
    padding: 1rem 2rem;
    border-radius: 12px;
    margin-bottom: 2rem;
    color: white;
}

.logo-section {
    display: flex;
    align-items: center;
    gap: 1rem;
}

.navbar {
    display: flex;
    gap: 2rem;
}

.nav-button {
    background: rgba(255,255,255,0.1);
    border: 2px solid rgba(255,255,255,0.3);
    color: white;
    padding: 0.5rem 1.5rem;
    border-radius: 25px;
    text-decoration: none;
    font-weight: 500;
    transition: all 0.3s ease;
}

.nav-button:hover {
    background: rgba(255,204,0,0.2);
    border-color: #ffcc00;
    color: #ffcc00;
    transform: translateY(-2px);
}

.nav-button.active {
    background: #ffcc00;
    color: #004d40;
    border-color: #ffcc00;
}

.hero {
    background: linear-gradient(rgba(0,77,64,0.8), rgba(0,105,92,0.8)), url('https://images.unsplash.com/photo-1558618666-fcd25c85cd64?ixlib=rb-4.0.3') center/cover;
    color: white;
    text-align: center;
    padding: 6rem 2rem;
    border-radius: 20px;
    margin: 2rem 0;
}

.hero h1 {
    font-size: 3.5rem;
    font-weight: 700;
    margin-bottom: 1rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}

.hero p {
    font-size: 1.8rem;
    font-weight: 300;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
}

.section {
    background: white;
    color: #333;
    padding: 2rem;
    border-radius: 15px;
    margin: 2rem 0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}

.features-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 2rem;
    margin-top: 2rem;
}

.feature-card {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    padding: 2rem;
    border-radius: 15px;
    text-align: center;
    border: 1px solid #dee2e6;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.feature-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 25px rgba(0,77,64,0.15);
}

.feature-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
}

.order-form {
    background: #f8f9fa;
    color: #333;
    padding: 2rem;
    border-radius: 15px;
    border: 1px solid #dee2e6;
}

.prediction-card {
    background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
    padding: 2rem;
    border-radius: 15px;
    border-left: 5px solid #4caf50;
    margin: 1rem 0;
}

.warning-card {
    background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
    padding: 2rem;
    border-radius: 15px;
    border-left: 5px solid #ff9800;
    margin: 1rem 0;
}

.footer {
    background: #004d40;
    color: white;
    text-align: center;
    padding: 2rem;
    border-radius: 12px;
    margin-top: 3rem;
}

.stButton > button {
    background: linear-gradient(135deg, #004d40 0%, #00695c 100%);
    color: white;
    border: none;
    border-radius: 25px;
    padding: 0.75rem 2rem;
    font-weight: 500;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(0,77,64,0.3);
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# INITIALIZE SESSION STATE
# ---------------------------
if 'page' not in st.session_state:
    st.session_state.page = 'Home'
if 'orders' not in st.session_state:
    st.session_state.orders = []
if 'customers' not in st.session_state:
    st.session_state.customers = {}

# ---------------------------
# HEADER WITH LOGOS
# ---------------------------
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.image(r"greenwells logo.png", width=150)
    
with col2:
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <h1 style="color: #004d40; margin: 0;">GasVision AI</h1>
        <p style="color: #00695c; margin: 0;">In collaboration with Green Wells Energies</p>
    </div>
    """, unsafe_allow_html=True)
    
with col3:
    st.image(r"gasvision AI logo.png", width=100)

# ---------------------------
# NAVIGATION
# ---------------------------
col1, col2, col3 = st.columns([1,1,1])

with col1:
    if st.button("Home", key="home_btn", use_container_width=True):
        st.session_state.page = 'Home'
        
with col2:
    if st.button("Order", key="order_btn", use_container_width=True):
        st.session_state.page = 'Order'
        
with col3:
    if st.button("History", key="history_btn", use_container_width=True):
        st.session_state.page = 'History'

# ---------------------------
# SYNTHETIC DATA GENERATOR
# ---------------------------
def generate_synthetic_data():
    dates = pd.date_range('2024-01-01', periods=240, freq='D')
    np.random.seed(42)
    base_usage = 0.4
    seasonal_factor = 0.1 * np.sin(2 * np.pi * np.arange(240) / 365)
    noise = np.random.normal(0, 0.05, 240)
    daily_usage = base_usage + seasonal_factor + noise
    daily_usage = np.clip(daily_usage, 0.2, 0.8)
    
    remaining_gas = []
    current_gas = 13.0
    refill_days = []
    
    for i, usage in enumerate(daily_usage):
        if current_gas - usage <= 0.5:
            current_gas = 13.0
            refill_days.append(i)
        current_gas -= usage
        remaining_gas.append(current_gas)
    
    return pd.DataFrame({
        'date': dates,
        'day_of_week': dates.strftime('%A'),
        'daily_usage_kg': daily_usage.round(3),
        'remaining_gas_kg': np.array(remaining_gas).round(2),
        'is_refill_day': [i in refill_days for i in range(240)],
        'day_number': range(1, 241)
    })

# ---------------------------
# HOME PAGE
# ---------------------------
if st.session_state.page == 'Home':
    # Hero Section
    st.markdown("""
    <div class="hero">
        <h1>GasVision AI</h1>
        <p>Predict. Monitor. Save.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # About Section
    st.markdown("""
    <div class="section">
        <h2>About GasVision AI</h2>
        <p style="font-size: 1.2rem; line-height: 1.6;">GasVision AI is a revolutionary smart LPG delivery and monitoring system that combines Vision AI with predictive analytics. Our platform helps households and businesses optimize their gas consumption, predict refill needs, and never run out of gas unexpectedly.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features Section
    st.markdown("""
    <div class="section">
        <h2>Key Features</h2>
        <div class="features-grid">
            <div class="feature-card">
                <h3>Smart Monitoring</h3>
                <p>Real-time tracking of LPG levels with AI-powered consumption analysis</p>
            </div>
            <div class="feature-card">
                <h3>Smart Notifications</h3>
                <p>Get alerts before your gas runs out, never be caught off guard again</p>
            </div>
            <div class="feature-card">
                <h3>AI Predictions</h3>
                <p>Machine learning models predict your future gas consumption patterns</p>
            </div>
            <div class="feature-card">
                <h3>Usage History</h3>
                <p>Comprehensive tracking and visualization of your consumption data</p>
            </div>
            <div class="feature-card">
                <h3>Auto-Ordering</h3>
                <p>Seamless integration with suppliers for automatic refill scheduling</p>
            </div>
            <div class="feature-card">
                <h3>Vision AI (Coming Soon)</h3>
                <p>Monitor cylinder levels using smartphone camera technology</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------
# ORDER PAGE
# ---------------------------
elif st.session_state.page == 'Order':
    st.markdown("""
    <div class="section">
        <h2>Order LPG Refill</h2>
        <p>Place your gas refill order quickly and easily. We'll deliver fresh LPG cylinders right to your doorstep.</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("order_form", clear_on_submit=True):
        st.markdown('<div class="order-form">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            customer_name = st.text_input("Customer Name", placeholder="Enter your full name")
            phone = st.text_input("Phone Number", placeholder="e.g., +254 801 234 5678")
            
        with col2:
            amount = st.slider("Amount of Gas (kg)", 0, 100, 13, help="Standard cylinder is 13kg")
            location = st.text_area("Delivery Location", placeholder="Enter your complete address")
        
        delivery_time = st.selectbox("Preferred Delivery Time", 
                                   ["Morning (8AM-12PM)", "Afternoon (12PM-4PM)", "Evening (4PM-8PM)"])
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        submitted = st.form_submit_button("Place Order", use_container_width=True)
        
        if submitted:
            if customer_name and phone and location:
                order_id = f"GV{len(st.session_state.orders)+1:04d}"
                order = {
                    'id': order_id,
                    'customer': customer_name,
                    'phone': phone,
                    'amount': amount,
                    'location': location,
                    'delivery_time': delivery_time,
                    'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
                    'status': 'Confirmed'
                }
                st.session_state.orders.append(order)
                st.session_state.customers[phone] = customer_name
                
                st.success(f"""**Order Confirmed!**
                
**Order ID:** {order_id}  
**Customer:** {customer_name}  
**Amount:** {amount} kg LPG  
**Delivery:** {location}  
**Time:** {delivery_time}  
**Status:** Confirmed

SMS confirmation sent to {phone}""")
                
                # Delivery tracking simulation
                st.info("**Delivery Tracking:** Your order is being prepared and will be dispatched within 2 hours.")
            else:
                st.error("Please fill in all required fields.")
    
    # Recent Orders
    if st.session_state.orders:
        st.markdown("""
        <div class="section">
            <h3>Recent Orders</h3>
        </div>
        """, unsafe_allow_html=True)
        
        for order in st.session_state.orders[-3:]:
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; border-left: 4px solid #004d40;">
                <strong>Order {order['id']}</strong> - {order['customer']} - {order['amount']}kg - {order['status']}
            </div>
            """, unsafe_allow_html=True)

# ---------------------------
# HISTORY PAGE
# ---------------------------
elif st.session_state.page == 'History':
    st.markdown("""
    <div class="section">
        <h2>Gas Usage History</h2>
        <p>View your gas consumption and get predictions for future needs.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # User identification
    user_phone = st.text_input("Enter your phone number to view history:", placeholder="e.g., +254 801 234 5678")
    
    if user_phone and user_phone in st.session_state.customers:
        customer_name = st.session_state.customers[user_phone]
        
        # Get user orders
        user_orders = [order for order in st.session_state.orders if order['phone'] == user_phone]
        
        if user_orders:
            # Calculate total consumption
            total_consumed = sum(order['amount'] for order in user_orders)
            
            # Simple prediction based on average usage
            avg_usage = total_consumed / len(user_orders) if user_orders else 13
            predicted_next = avg_usage
            
            st.markdown(f"""
            <div class="section">
                <h3>Usage Summary for {customer_name}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="prediction-card">
                    <h4>Total Gas Consumed</h4>
                    <h2>{total_consumed} kg</h2>
                    <p>From {len(user_orders)} orders</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="prediction-card">
                    <h4>Predicted Next Order</h4>
                    <h2>{predicted_next:.1f} kg</h2>
                    <p>Based on your usage pattern</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Order history table
            st.markdown("""
            <div class="section">
                <h3>Order History</h3>
            </div>
            """, unsafe_allow_html=True)
            
            for order in user_orders:
                st.markdown(f"""
                <div style="background: #f8f9fa; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; border-left: 4px solid #004d40;">
                    <strong>{order['date']}</strong> - {order['amount']} kg - {order['status']}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info(f"No order history found for {customer_name}. Place your first order to see usage data.")
    
    elif user_phone:
        st.warning("Phone number not found. Please place an order first to create your usage history.")
    
    else:
        st.info("Enter your phone number to view your gas usage history and predictions.")

# ---------------------------
# FOOTER
# ---------------------------
st.markdown("""
<div class="footer">
    <p>© 2025 GasVision AI - In collaboration with Green Wells Energies</p>
    <p>Contact: info@gasvisionai.com | +254 800 GAS VISION</p>
    <p>Powered by AI • Built for smarter gas management</p>
</div>
""", unsafe_allow_html=True)
