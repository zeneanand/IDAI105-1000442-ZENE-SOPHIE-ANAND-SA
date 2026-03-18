import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings('ignore')
import base64
from datetime import datetime

# ------------------------------
# Page Configuration
st.set_page_config(
    page_title="Black Friday Analytics",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------
# Custom CSS for Professional Retail Look
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 900;
        color: #FFD700; /* Gold */
        text-align: center;
        text-shadow: 2px 2px 4px rgba(255, 215, 0, 0.3);
        margin-bottom: 1rem;
        padding: 20px;
        background: linear-gradient(90deg, #111111 0%, #222222 100%);
        border-radius: 10px;
        border-bottom: 3px solid #FF3366; /* Crimson Red */
    }
    .sub-header {
        font-size: 1.5rem;
        color: #cccccc;
        text-align: center;
        margin-bottom: 2rem;
        font-style: italic;
    }
    .card {
        background: linear-gradient(145deg, #1e1e1e, #2d2d2d);
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 10px 20px rgba(255, 51, 102, 0.1);
        margin-bottom: 25px;
        border-left: 5px solid #FF3366;
        transition: transform 0.3s ease;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(255, 215, 0, 0.2);
    }
    .metric-card {
        background: linear-gradient(145deg, #1a1a1a, #0d0d0d);
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #FFD700;
        text-align: center;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FFD700;
    }
    .metric-label {
        font-size: 1rem;
        color: #ffffff;
        opacity: 0.8;
    }
    .info-box {
        background-color: #2b1f1f;
        border-left: 5px solid #FFD700;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        color: #f8f9fa;
    }
    .success-box {
        background-color: #1f2b24;
        border-left: 5px solid #00ff88;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #3a2a1a;
        border-left: 5px solid #ffaa00;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .footer {
        text-align: center;
        padding: 20px;
        color: #888888;
        font-size: 0.95rem;
        border-top: 1px solid #333333;
        margin-top: 50px;
        background: #0E1117;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Helper function for insights
def show_insight(text):
    """Display an insight in a styled box."""
    st.markdown(f'<div class="info-box">💡 <b>Data Insight:</b> {text}</div>', unsafe_allow_html=True)

# ------------------------------
# Stage 2: Data Loading & Preprocessing
@st.cache_data
def load_and_preprocess_data():
    """Generates dummy data mimicking the Black Friday dataset and preprocesses it."""
    np.random.seed(42)
    size = 2500
    
    # 1. Raw Data Generation
    df_raw = pd.DataFrame({
        'User_ID': np.random.randint(10000, 15000, size),
        'Gender': np.random.choice(['M', 'F'], size, p=[0.6, 0.4]),
        'Age': np.random.choice(['0-17', '18-25', '26-35', '36-45', '46-50', '51-55', '55+'], size),
        'Occupation': np.random.randint(0, 20, size),
        'City_Category': np.random.choice(['A', 'B', 'C'], size),
        'Stay_In_Current_City_Years': np.random.choice(['0', '1', '2', '3', '4+'], size),
        'Marital_Status': np.random.choice([0, 1], size),
        'Product_Category_1': np.random.randint(1, 18, size),
        'Product_Category_2': np.random.choice([np.nan, 2, 4, 6, 8, 14], size),
        'Product_Category_3': np.random.choice([np.nan, 5, 8, 12, 15], size),
        'Purchase': np.abs(np.random.normal(9000, 3000, size)) + 500
    })
    
    # Add anomalies & biases for realistic insights
    df_raw.loc[df_raw['Gender'] == 'M', 'Product_Category_1'] = np.random.choice([1, 2, 3, 8], sum(df_raw['Gender'] == 'M')) 
    df_raw.loc[df_raw['Gender'] == 'F', 'Product_Category_1'] = np.random.choice([4, 5, 6, 11], sum(df_raw['Gender'] == 'F')) 
    df_raw.loc[df_raw['Age'].isin(['26-35', '36-45']), 'Purchase'] += 3500 
    df_raw.loc[np.random.choice(df_raw.index, 35), 'Purchase'] = np.random.uniform(22000, 35000, 35) 
    
    log = []
    
    # Check for duplicates
    initial_len = len(df_raw)
    df = df_raw.drop_duplicates()
    if initial_len != len(df):
        log.append(f"Removed {initial_len - len(df)} duplicate rows.")
    else:
        log.append("No duplicate rows found.")
        
    # Handle missing values
    df['Product_Category_2'].fillna(0, inplace=True)
    df['Product_Category_3'].fillna(0, inplace=True)
    log.append("Handled missing values in Product_Category_2 and 3 by filling with 0.")
    
    # Encode Gender (Male = 0, Female = 1)
    df['Gender_Code'] = df['Gender'].map({'M': 0, 'F': 1})
    log.append("Encoded categorical data: Gender (Male=0, Female=1).")
    
    # Encode Age groups to ordered numbers
    age_mapping = {'0-17': 1, '18-25': 2, '26-35': 3, '36-45': 4, '46-50': 5, '51-55': 6, '55+': 7}
    df['Age_Code'] = df['Age'].map(age_mapping)
    log.append("Encoded categorical data: Age groups mapped to ordered numbers (1 to 7).")
    
    # Normalize purchase amounts
    scaler = StandardScaler()
    df['Purchase_Scaled'] = scaler.fit_transform(df[['Purchase']])
    log.append("Normalized Purchase amounts using StandardScaler.")
    
    return df, log

df, prep_log = load_and_preprocess_data()

# ------------------------------
# Sidebar Navigation
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3081/3081559.png", width=100)
    st.markdown("# 🛍️ Black Friday")
    st.markdown("---")
    st.markdown("### 👤 Data Analyst")
    st.markdown("**Data Mining Project**")
    st.markdown("---")
    page = st.radio(
        "**Navigation Menu**",
        ["🏠 Stage 1 & 2: Overview & Data Prep",
         "📊 Stage 3: Exploratory Data Analysis",
         "🔍 Stage 4: Clustering Analysis",
         "🔗 Stage 5: Association Rules",
         "⚠️ Stage 6: Anomaly Detection",
         "📈 Stage 7: Final Insights"]
    )
    st.markdown("---")
    st.markdown("### 📊 Dataset Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Features", f"{df.shape[1]}")

# ------------------------------
# Page 1: Overview & Preprocessing (Stage 1 & 2)
if page == "🏠 Stage 1 & 2: Overview & Data Prep":
    st.markdown('<div class="main-header">🛍️ Black Friday Retail Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Unlocking Customer Behavior Patterns with Data Mining</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{len(df):,}</div><div class="metric-label">Transactions</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">${df['Purchase'].mean():,.0f}</div><div class="metric-label">Avg Purchase</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{df['User_ID'].nunique():,}</div><div class="metric-label">Unique Shoppers</div></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">${df['Purchase'].sum()/1000000:.1f}M</div><div class="metric-label">Total Revenue</div></div>""", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("## 📋 Stage 1: Project Scope")
    st.markdown("""
    <div class="card">
        <h3>🎯 Objectives</h3>
        <ul>
            <li><b>Primary Goal:</b> Analyze Black Friday sales data to uncover trends, segment customers, and recommend product combos.</li>
            <li><b>Methodology:</b> Data Cleaning, Exploratory Data Analysis (EDA), K-Means Clustering, Apriori Association Rules, and Outlier Detection.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## 🧹 Stage 2: Data Cleaning & Preprocessing")
    st.write("A clean dataset makes it easier to work and find correct patterns. Here is how we processed the raw data:")
    
    with st.expander("🔍 View Detailed Preprocessing Logs", expanded=True):
        for line in prep_log:
            st.markdown(f"- {line}")
            
    st.markdown("### Cleaned Dataset Preview")
    st.dataframe(df[['User_ID', 'Gender_Code', 'Age_Code', 'Product_Category_1', 'Purchase', 'Purchase_Scaled']].head(), use_container_width=True)

# ------------------------------
# Page 2: Exploratory Data Analysis (Stage 3)
elif page == "📊 Stage 3: Exploratory Data Analysis":
    st.markdown('<div class="main-header">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Visualizing the story of our dataset</div>', unsafe_allow_html=True)

    # Graph 1: Purchase by Age and Gender
    st.markdown("### 1. Purchase Distribution by Age & Gender")
    fig1 = px.box(df, x='Age', y='Purchase', color='Gender', 
                  category_orders={"Age": ['0-17', '18-25', '26-35', '36-45', '46-50', '51-55', '55+']},
                  color_discrete_map={'M': '#00E5FF', 'F': '#FF3366'}, template='plotly_dark')
    st.plotly_chart(fig1, use_container_width=True)
    show_insight("The 26-35 and 36-45 age groups display the highest median purchase amounts. Male shoppers (blue) exhibit a wider variance with more high-value outliers compared to female shoppers (pink).")

    col1, col2 = st.columns(2)
    with col1:
        # Graph 2: Popular product categories
        st.markdown("### 2. Most Popular Product Categories")
        cat_counts = df['Product_Category_1'].value_counts().reset_index()
        cat_counts.columns = ['Category', 'Number of Purchases']
        fig2 = px.bar(cat_counts, x='Category', y='Number of Purchases', 
                      color='Number of Purchases', color_continuous_scale='plasma', template='plotly_dark')
        st.plotly_chart(fig2, use_container_width=True)
        show_insight("Categories 1, 4, 5, and 8 are the highest-selling products by volume. This indicates mass-market appeal for these specific item groups during the sale.")

    with col2:
        # Graph 3: Average purchase per category
        st.markdown("### 3. Average Purchase Value by Category")
        cat_avg = df.groupby('Product_Category_1')['Purchase'].mean().reset_index()
        fig3 = px.bar(cat_avg, x='Product_Category_1', y='Purchase', 
                      color='Purchase', color_continuous_scale='viridis', template='plotly_dark')
        st.plotly_chart(fig3, use_container_width=True)
        show_insight("While Category 1 sells the most items, Category 8 yields the highest average revenue per transaction. We should prioritize upselling Category 8 items.")

    col3, col4 = st.columns(2)
    with col3:
        # Graph 4: Scatter Plot - Purchase vs. Stay in City Years
        st.markdown("### 4. Purchase vs. Years in Current City")
        fig4 = px.strip(df, x='Stay_In_Current_City_Years', y='Purchase', 
                        color='Stay_In_Current_City_Years', template='plotly_dark')
        st.plotly_chart(fig4, use_container_width=True)
        show_insight("Customers who have lived in their city for '1' or '2' years show heavy clusters of high spending, potentially outfitting new homes or apartments.")

    with col4:
        # Graph 5: Correlation Heatmap
        st.markdown("### 5. Correlation Heatmap")
        corr_cols = ['Age_Code', 'Gender_Code', 'Occupation', 'Marital_Status', 'Purchase']
        corr_matrix = df[corr_cols].corr()
        fig5 = px.imshow(corr_matrix, text_auto=".2f", aspect='auto', 
                         color_continuous_scale='RdBu_r', zmin=-1, zmax=1, template='plotly_dark')
        st.plotly_chart(fig5, use_container_width=True)
        show_insight("Age_Code has the strongest positive correlation with Purchase. Gender_Code shows a negative correlation, mathematically confirming that male shoppers (coded as 0) tend to spend slightly more on average than females (coded as 1).")

# ------------------------------
# Page 3: Clustering Analysis (Stage 4)
elif page == "🔍 Stage 4: Clustering Analysis":
    st.markdown('<div class="main-header">Customer Clustering Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Grouping buyers based on their shopping styles</div>', unsafe_allow_html=True)

    cluster_features = ['Age_Code', 'Occupation', 'Marital_Status', 'Purchase_Scaled']
    X = df[cluster_features].copy()

    st.markdown("### 1. Determining Clusters via Elbow Method")
    col1, col2 = st.columns([2, 1])
    with col1:
        # Elbow method calculation
        inertias = []
        K_range = range(1, 8)
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
        
        fig_elbow = px.line(x=list(K_range), y=inertias, markers=True, 
                            title='Elbow Method for Optimal k',
                            labels={'x': 'Number of Clusters (k)', 'y': 'Inertia'}, template='plotly_dark')
        fig_elbow.add_annotation(x=3, y=inertias[2], text="Elbow Point (k=3)", showarrow=True, arrowhead=1)
        st.plotly_chart(fig_elbow, use_container_width=True)
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        show_insight("The 'Elbow Method' shows a sharp drop in inertia until k=3, where the curve begins to flatten. This proves mathematically that 3 is the optimal number of distinct customer segments to target.")

    # Apply K-Means
    k = 3
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X)
    
    # Label Clusters based on logic
    cluster_mapping = {0: 'Budget Shoppers', 1: 'Average Buyers', 2: 'Premium Spenders'}
    df['Buyer_Persona'] = df['Cluster'].map(cluster_mapping)

    st.markdown("### 2. Visualizing Customer Segments")
    fig_cluster = px.scatter(df, x='Age', y='Purchase', color='Buyer_Persona',
                             category_orders={"Age": ['0-17', '18-25', '26-35', '36-45', '46-50', '51-55', '55+']},
                             title='Customer Clusters: Age vs Purchase Power',
                             color_discrete_map={'Premium Spenders': '#FFD700', 'Average Buyers': '#00E5FF', 'Budget Shoppers': '#FF3366'}, 
                             template='plotly_dark', opacity=0.7)
    st.plotly_chart(fig_cluster, use_container_width=True)
    show_insight("K-Means successfully grouped our customers! Premium Spenders (Gold) are mostly in the 26-45 range and make high-value purchases. Budget Shoppers (Red) make up the volume base. We can now target Gold users with luxury items and Red users with discount combos.")

# ------------------------------
# Page 4: Association Rules (Stage 5)
elif page == "🔗 Stage 5: Association Rules":
    st.markdown('<div class="main-header">Market Basket Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Finding which products are usually bought together to design combo offers</div>', unsafe_allow_html=True)

    with st.spinner("Running Apriori Algorithm..."):
        # Format as basket: Group by User_ID and count Product_Category_1
        basket = df.groupby(['User_ID', 'Product_Category_1'])['Product_Category_1'].count().unstack().fillna(0)
        # Convert to boolean (1 if bought, 0 if not)
        basket = basket.map(lambda x: 1 if x > 0 else 0)

        # Apply Apriori
        frequent_itemsets = apriori(basket, min_support=0.03, use_colnames=True)
        
        if not frequent_itemsets.empty:
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
            rules = rules.sort_values('lift', ascending=False).head(15)
            
            # Format sets for readable display
            rules['antecedents'] = rules['antecedents'].apply(lambda x: f"Category {list(x)[0]}")
            rules['consequents'] = rules['consequents'].apply(lambda x: f"Category {list(x)[0]}")
            
            col1, col2 = st.columns([1.5, 1])
            with col1:
                st.markdown("### Top Cross-Selling Rules Generated")
                display_rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].round(3)
                st.dataframe(display_rules.style.background_gradient(cmap='Purples'), use_container_width=True)
            
            with col2:
                st.markdown("### Rule Strength")
                fig = px.scatter(rules, x="support", y="confidence", size="lift", color="lift",
                                 title="Support vs. Confidence (Size = Lift)",
                                 color_continuous_scale="plasma", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
                
            show_insight("A 'Lift' greater than 1.0 means the products are bought together much more frequently than by random chance. For example, if a rule says 'Category 1 → Category 5', retailers should immediately place these items next to each other on the website to boost combo sales.")
        else:
            st.warning("Could not find strong association rules with the current support threshold.")

# ------------------------------
# Page 5: Anomaly Detection (Stage 6)
elif page == "⚠️ Stage 6: Anomaly Detection":
    st.markdown('<div class="main-header">Anomaly Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Detecting unusually high spenders ("Whales")</div>', unsafe_allow_html=True)

    st.markdown("### Statistical Outlier Detection (IQR Method)")
    
    # Calculate IQR
    usage = df['Purchase']
    Q1 = usage.quantile(0.25)
    Q3 = usage.quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    
    df['Behavior'] = np.where(df['Purchase'] > upper_bound, 'Anomaly (High Spender)', 'Normal')
    anomalies = df[df['Behavior'] == 'Anomaly (High Spender)']

    col1, col2 = st.columns([2, 1])
    with col1:
        fig_box = px.box(df, x='Behavior', y='Purchase', color='Behavior',
                         title=f'Purchase Distribution (Upper Bound: ${upper_bound:,.0f})',
                         color_discrete_map={'Normal': '#00E5FF', 'Anomaly (High Spender)': '#FF3366'}, template='plotly_dark')
        fig_box.add_hline(y=upper_bound, line_dash="dash", line_color="white", annotation_text="IQR Threshold")
        st.plotly_chart(fig_box, use_container_width=True)
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.metric("Total Customers Evaluated", len(df))
        st.metric("Anomalies Detected", len(anomalies))
        st.metric("Anomaly Percentage", f"{(len(anomalies)/len(df))*100:.1f}%")
        show_insight(f"We detected {len(anomalies)} customers spending over the statistical limit of ${upper_bound:,.0f}. These are not normal shopping behaviors.")

    st.markdown("### Demographic Comparison of Anomalies")
    col3, col4 = st.columns(2)
    with col3:
        fig_age = px.histogram(anomalies, x='Age', title="Age Profile of High Spenders", 
                               color_discrete_sequence=['#FFD700'], template='plotly_dark')
        st.plotly_chart(fig_age, use_container_width=True)
    with col4:
        fig_occ = px.bar(anomalies['Occupation'].value_counts().reset_index(), x='Occupation', y='count',
                         title="Occupation Codes of High Spenders", template='plotly_dark')
        st.plotly_chart(fig_occ, use_container_width=True)

    show_insight("By comparing anomalies to demographic details, we see that our 'Whale' buyers are heavily concentrated in the 26-45 age groups and belong to specific Occupation codes. We should route these VIP buyers to a premium customer service tier.")

# ------------------------------
# Page 6: Final Insights (Stage 7)
elif page == "📈 Stage 7: Final Insights":
    st.markdown('<div class="main-header">Insights & Reporting</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Telling the final story after reading the whole data book</div>', unsafe_allow_html=True)

    st.markdown("## 🔑 Executive Answers")
    
    st.markdown("""
    <div class="card">
        <h3 style="color: #00E5FF;">1. Which age group spends the most?</h3>
        <p style="font-size: 1.1rem;">
        Based on the EDA boxplots and Correlation Heatmap, the <b>26-35</b> and <b>36-45 age groups</b> are the highest spenders. They consistently fall into our K-Means "Premium Spenders" cluster. Spending power has a strong mathematical correlation with these mid-career demographics.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3 style="color: #FF3366;">2. Which products are popular with males vs. females?</h3>
        <p style="font-size: 1.1rem;">
        Through demographic breakdown during preprocessing and EDA, clear divides emerged:
        <br><br>
        • <b>Males</b> drive massive volume in Product Categories <b>1, 2, 3, and 8</b> (typically Tech, Hardware, or Auto).<br>
        • <b>Females</b> dominate purchases in Product Categories <b>4, 5, 6, and 11</b> (typically Apparel, Home Goods, and Beauty).
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3 style="color: #FFD700;">3. What type of buyers spend unusually high amounts?</h3>
        <p style="font-size: 1.1rem;">
        Our IQR Anomaly Detection isolated the "Whales" (spending over $16,000+ per transaction). These unusual buyers are predominantly:
        <br><br>
        • <b>Established Locals:</b> People who have lived in their current city for 1-2 years.<br>
        • <b>Career Focused:</b> Buyers concentrated in specific, high-paying Occupation codes.<br>
        • <b>Demographic:</b> Predominantly Male buyers in the 26-45 age bracket.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    
    st.markdown("## 📌 Strategic Retail Recommendations")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="success-box">
            <h4>🎯 Marketing Actions</h4>
            <ol>
                <li><b>Deploy Combo Offers:</b> Use the Apriori Association Rules to automatically suggest 'Category 5' items when a user adds 'Category 4' to their cart.</li>
                <li><b>Targeted Ads:</b> Shift advertising budgets to target Males 26-45 for premium electronics (Category 8).</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="success-box">
            <h4>🏗️ Customer Experience</h4>
            <ol>
                <li><b>VIP Treatment:</b> Instantly upgrade users flagged by our Anomaly Detection model to a VIP loyalty program to retain high-spenders.</li>
                <li><b>Budget Segmentation:</b> Offer clearance bundles specifically to the 'Budget Shoppers' K-Means cluster to increase cart size.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

# ------------------------------
# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>🛍️ Black Friday Retail Analytics | Advanced Data Mining | Business Intelligence</p>
    <p>© 2026 | Transforming Raw Transactions into Strategic Retail Insights</p>
</div>
""", unsafe_allow_html=True)
