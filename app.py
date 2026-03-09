import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules

# --- PAGE CONFIG ---
st.set_page_config(page_title="Black Friday Intelligence", page_icon="🛍️", layout="wide")

# --- CUSTOM CSS FOR AESTHETICS ---
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    /* Metric Card Styling */
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: bold;
        color: #FACC15; /* Gold Accent */
    }
    div[data-testid="stMetricLabel"] {
        color: #94A3B8;
    }
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #1E293B;
    }
    /* Style headers */
    h1, h2, h3 {
        color: #F8FAFC;
        font-family: 'Inter', sans-serif;
    }
    /* Custom Card Container */
    .reportview-container .main .block-container{
        padding-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# --- DATA GENERATION (Stage 2: Preprocessing) ---
@st.cache_data
def get_clean_data():
    np.random.seed(42)
    size = 2000
    df = pd.DataFrame({
        'User_ID': np.random.randint(1001, 1500, size),
        'Gender': np.random.choice(['Male', 'Female'], size, p=[0.6, 0.4]),
        'Age': np.random.choice(['0-17', '18-25', '26-35', '36-45', '46-55', '55+'], size),
        'City_Category': np.random.choice(['A', 'B', 'C'], size),
        'Product_Category_1': np.random.randint(1, 12, size),
        'Purchase': np.random.normal(9500, 3000, size)
    })
    # Add High-Value Anomalies
    df.loc[np.random.choice(df.index, 25), 'Purchase'] = np.random.uniform(25000, 40000, 25)
    
    # Preprocessing
    df['Age_Code'] = df['Age'].astype('category').cat.codes
    return df

df = get_clean_data()

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3081/3081559.png", width=100)
    st.title("Admin Console")
    page = st.selectbox("Navigation", ["📈 Overview & EDA", "🎯 Customer Segments", "🔗 Market Basket Analysis", "⚠️ Fraud & Anomalies"])
    st.markdown("---")
    st.write("**Model Status:** Optimized")
    st.write("**Data Refresh:** Mar 2026")

# --- HEADER SECTION ---
st.title("🛍️ Black Friday Business Intelligence")
st.markdown("##### Strategic insights through Advanced Data Mining & AI Clustering")

# --- TOP KPI ROW ---
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Revenue", f"${df['Purchase'].sum():,.0f}")
m2.metric("Avg Basket Value", f"${df['Purchase'].mean():,.2f}")
m3.metric("Unique Customers", f"{df['User_ID'].nunique()}")
m4.metric("High-Value Sales", len(df[df['Purchase'] > 20000]))

st.divider()

# --- 1. OVERVIEW & EDA ---
if page == "📈 Overview & EDA":
    st.subheader("Exploratory Data Insights")
    
    tab1, tab2 = st.tabs(["Demographics", "Product Trends"])
    
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            fig_age = px.box(df, x="Age", y="Purchase", color="Gender",
                            title="Spending Power by Age & Gender",
                            color_discrete_sequence=["#3B82F6", "#EC4899"],
                            template="plotly_dark")
            st.plotly_chart(fig_age, use_container_width=True)
            
        with c2:
            fig_city = px.sunburst(df, path=['City_Category', 'Age'], values='Purchase',
                                  title="Revenue Distribution by City & Age",
                                  color_continuous_scale='YlGnBu',
                                  template="plotly_dark")
            st.plotly_chart(fig_city, use_container_width=True)

    with tab2:
        fig_product = px.violin(df, x="Product_Category_1", y="Purchase", color="Product_Category_1",
                               title="Purchase Volume per Product Category",
                               template="plotly_dark")
        st.plotly_chart(fig_product, use_container_width=True)

# --- 2. CLUSTERING ---
elif page == "🎯 Customer Segments":
    st.subheader("AI-Powered Customer Segmentation")
    st.info("Applying K-Means Clustering to identify distinct spending tiers based on demographic data.")
    
    # Clustering Logic
    scaler = StandardScaler()
    X = scaler.fit_transform(df[['Age_Code', 'Purchase']])
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X)
    df['Segment'] = df['Cluster'].map({0: 'Value Hunter', 1: 'Standard Shopper', 2: 'VIP Spender'})
    
    fig_cluster = px.scatter(df, x="Age", y="Purchase", color="Segment", 
                            symbol="Segment", size="Purchase",
                            title="K-Means Segmentation: High-Value vs Volume Shoppers",
                            color_discrete_sequence=["#10B981", "#F59E0B", "#EF4444"],
                            template="plotly_dark")
    st.plotly_chart(fig_cluster, use_container_width=True)
    
    st.markdown("""
    **Business Strategy:**
    * 🟢 **Value Hunters:** Target with clearance bundles.
    * 🟡 **Standard Shoppers:** Prime for loyalty program upsells.
    * 🔴 **VIP Spenders:** Offer early access to luxury product categories.
    """)

# --- 3. ASSOCIATIONS ---
elif page == "🔗 Market Basket Analysis":
    st.subheader("Association Rules: Product Cross-Selling")
    
    # Apriori Preprocessing
    basket = df.groupby(['User_ID', 'Product_Category_1'])['Product_Category_1'].count().unstack().fillna(0)
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)
    
    frequent_items = apriori(basket, min_support=0.08, use_colnames=True)
    rules = association_rules(frequent_items, metric="lift", min_threshold=1)
    
    if not rules.empty:
        st.write("Identified Product Pairings with High Correlation:")
        st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values('lift', ascending=False).style.background_gradient(cmap='Blues'), use_container_width=True)
    else:
        st.warning("No strong associations found at this support level. Try increasing your data size.")

# --- 4. ANOMALIES ---
elif page == "⚠️ Fraud & Anomalies":
    st.subheader("Anomaly Detection: Extreme Outliers")
    st.write("Detecting purchases that fall outside 3 Standard Deviations—useful for fraud detection or identifying Whale Shoppers.")
    
    upper_limit = df['Purchase'].mean() + (3 * df['Purchase'].std())
    df['Is_Anomaly'] = df['Purchase'] > upper_limit
    
    fig_anomaly = px.scatter(df, x=df.index, y="Purchase", color="Is_Anomaly",
                            title="Detection of Outlier Transactions",
                            color_discrete_map={True: "#EF4444", False: "#334155"},
                            template="plotly_dark")
    fig_anomaly.add_hline(y=upper_limit, line_dash="dash", line_color="white", annotation_text="Anomaly Threshold")
    st.plotly_chart(fig_anomaly, use_container_width=True)
    
    st.error(f"Attention: {len(df[df['Is_Anomaly']])} transactions detected as anomalies. Action required for verification.")
    
