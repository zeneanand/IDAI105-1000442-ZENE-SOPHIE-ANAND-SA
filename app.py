import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules

# --- PAGE CONFIG ---
st.set_page_config(page_title="Black Friday Insights", page_icon="🛒", layout="wide")

# Styling
st.markdown("""
    <style>
    .main {background-color: #f0f2f6;}
    .stMetric {background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);}
    </style>
    """, unsafe_allow_html=True)

# --- DATA GENERATION (Stage 2: Preprocessing) ---
@st.cache_data
def get_data():
    np.random.seed(42)
    size = 1500
    df = pd.DataFrame({
        'User_ID': np.random.randint(100, 500, size),
        'Gender': np.random.choice(['Male', 'Female'], size),
        'Age': np.random.choice(['0-17', '18-25', '26-35', '36-45', '46-55', '55+'], size),
        'City_Category': np.random.choice(['A', 'B', 'C'], size),
        'Product_Category_1': np.random.randint(1, 15, size),
        'Purchase': np.random.normal(9000, 3500, size)
    })
    # Add anomalies
    df.loc[np.random.choice(df.index, 15), 'Purchase'] = np.random.uniform(25000, 35000, 15)
    
    # Preprocessing
    df['Age_Code'] = df['Age'].astype('category').cat.codes
    return df

df = get_data()

# --- HEADER ---
st.title("🛒 Black Friday Sales: Business Intelligence Dashboard")
st.markdown("Exploring customer segments and purchasing behavior using Advanced Data Mining.")
st.divider()

# --- TOP METRICS ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Revenue", f"${df['Purchase'].sum():,.0f}")
c2.metric("Avg Purchase", f"${df['Purchase'].mean():,.2f}")
c3.metric("Total Customers", df['User_ID'].nunique())
c4.metric("High-Value Anomalies", len(df[df['Purchase'] > 18000]))

# --- SIDEBAR ---
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to:", ["EDA", "Clustering", "Associations", "Anomalies"])

# --- PAGES ---
if page == "EDA":
    st.header("📊 Exploratory Data Analysis")
    col_a, col_b = st.columns(2)
    
    with col_a:
        fig_hist = px.histogram(df, x="Purchase", color="Gender", marginal="box", 
                               title="Purchase Distribution by Gender", color_discrete_sequence=['#3498db', '#e74c3c'])
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with col_b:
        fig_age = px.bar(df.groupby('Age')['Purchase'].mean().reset_index(), 
                        x='Age', y='Purchase', title="Avg Spend by Age Group", color='Purchase')
        st.plotly_chart(fig_age, use_container_width=True)

elif page == "Clustering":
    st.header("🎯 Customer Segmentation (K-Means)")
    scaler = StandardScaler()
    df['Scaled_Purchase'] = scaler.fit_transform(df[['Purchase']])
    
    km = KMeans(n_clusters=3, random_state=42)
    df['Segment'] = km.fit_predict(df[['Age_Code', 'Scaled_Purchase']])
    df['Segment'] = df['Segment'].map({0: 'Value Shoppers', 1: 'Target Market', 2: 'Premium Spenders'})
    
    fig_cluster = px.scatter(df, x="Age", y="Purchase", color="Segment", 
                            title="Clustering Customers by Age and Spend",
                            color_discrete_map={'Premium Spenders':'#e74c3c', 'Target Market':'#f1c40f', 'Value Shoppers':'#2ecc71'})
    st.plotly_chart(fig_cluster, use_container_width=True)
    st.success("Business Insight: Focus marketing campaigns on the 'Premium Spenders' segment during the final hours of the sale.")

elif page == "Associations":
    st.header("🔗 Market Basket Analysis (Association Rules)")
    # Pivot for Apriori
    basket = df.groupby(['User_ID', 'Product_Category_1'])['Product_Category_1'].count().unstack().fillna(0)
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)
    
    freq_items = apriori(basket, min_support=0.07, use_colnames=True)
    rules = association_rules(freq_items, metric="lift", min_threshold=1)
    
    st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values('lift', ascending=False), use_container_width=True)
    st.info("The table above shows which Product Categories are bought together. A Lift > 1 suggests a strong cross-selling opportunity.")

elif page == "Anomalies":
    st.header("🚨 Anomaly Detection")
    threshold = df['Purchase'].mean() + 2.5 * df['Purchase'].std()
    df['Anomaly'] = df['Purchase'] > threshold
    
    fig_anomaly = px.scatter(df, x=df.index, y="Purchase", color="Anomaly", 
                            title="Flagging Extreme Purchase Outliers",
                            color_discrete_map={True: '#c0392b', False: '#bdc3c7'})
    st.plotly_chart(fig_anomaly, use_container_width=True)
    st.warning(f"Detected {len(df[df['Anomaly']])} extreme transactions above ${threshold:,.0f}. These should be investigated for potential fraud or VIP status.")
