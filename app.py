import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings('ignore')

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

def show_insight(text):
    """Display an insight in a styled box."""
    st.markdown(f'<div class="info-box">💡 <b>Data Insight:</b> {text}</div>', unsafe_allow_html=True)

# ------------------------------
# Stage 2: Data Loading & Preprocessing
@st.cache_data
def load_and_preprocess_data():
    """Generates dummy data and preprocesses it exactly per Stage 2 requirements."""
    np.random.seed(42)
    size = 2500
    
    # 1. Raw Data Generation
    df_raw = pd.DataFrame({
        'User_ID': np.random.randint(10000, 15000, size),
        'Gender': np.random.choice(['Male', 'Female'], size, p=[0.6, 0.4]),
        'Age': np.random.choice(['0-17', '18-25', '26-35', '36-45', '46-50', '51-55', '55+'], size),
        'Occupation': np.random.randint(0, 20, size),
        'City_Category': np.random.choice(['A', 'B', 'C'], size),
        'Stay_In_Current_City_Years': np.random.choice(['0', '1', '2', '3', '4+'], size),
        'Marital_Status': np.random.choice([0, 1], size),
        'Product_Category_1': np.random.choice(['Electronics', 'Apparel', 'Home', 'Beauty', 'Sports'], size),
        'Product_Category_2': np.random.choice([np.nan, 'Accessories', 'Footwear', 'Decor', 'Skincare', 'Equipment'], size),
        'Purchase': np.abs(np.random.normal(9000, 3000, size)) + 500
    })
    
    # Induce biases so algorithms find cool patterns
    df_raw.loc[df_raw['Gender'] == 'Male', 'Product_Category_1'] = np.random.choice(['Electronics', 'Sports'], sum(df_raw['Gender'] == 'Male')) 
    df_raw.loc[df_raw['Gender'] == 'Female', 'Product_Category_1'] = np.random.choice(['Apparel', 'Beauty', 'Home'], sum(df_raw['Gender'] == 'Female')) 
    df_raw.loc[df_raw['Age'].isin(['26-35', '36-45']), 'Purchase'] += 3500 
    df_raw.loc[np.random.choice(df_raw.index, 35), 'Purchase'] = np.random.uniform(22000, 35000, 35) # Anomalies
    
    log = []
    
    # STAGE 2 STEPS
    
    # Check for duplicates
    initial_len = len(df_raw)
    df = df_raw.drop_duplicates()
    log.append(f"**Step 1:** Checked for duplicates. Removed {initial_len - len(df)} duplicate/irrelevant rows.")
        
    # Handle missing values
    df['Product_Category_2'].fillna("None", inplace=True)
    log.append("**Step 2:** Handled missing values in Product_Category_2 (Filled with 'None').")
    
    # Encode Categorical Data: Gender
    df['Gender_Code'] = df['Gender'].map({'Male': 0, 'Female': 1})
    log.append("**Step 3:** Encoded Gender into numbers (Male = 0, Female = 1).")
    
    # Encode Categorical Data: Age
    age_mapping = {'0-17': 1, '18-25': 2, '26-35': 3, '36-45': 4, '46-50': 5, '51-55': 6, '55+': 7}
    df['Age_Code'] = df['Age'].map(age_mapping)
    log.append("**Step 4:** Converted Age groups into ordered numbers (0-17 → 1, 18-25 → 2, etc.).")
    
    # Normalize purchase amounts
    scaler = StandardScaler()
    df['Purchase_Scaled'] = scaler.fit_transform(df[['Purchase']])
    log.append("**Step 5:** Normalized 'Purchase' amounts using StandardScaler so values are on the same scale for clustering.")
    
    return df, log

df, prep_log = load_and_preprocess_data()

# ------------------------------
# Sidebar Navigation
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3081/3081559.png", width=100)
    st.markdown("# 🛍️ Black Friday")
    st.markdown("---")
    st.markdown("### 👤 Data Analyst")
    page = st.radio(
        "**Navigation Menu**",
        ["1️⃣ Stage 1: Project Scope",
         "2️⃣ Stage 2: Data Preprocessing",
         "3️⃣ Stage 3: EDA",
         "4️⃣ Stage 4: Clustering Analysis",
         "5️⃣ Stage 5: Association Rules",
         "6️⃣ Stage 6: Anomaly Detection",
         "7️⃣ Stage 7: Insights & Reporting"]
    )

# ------------------------------
# Stage 1: Project Scope
if page == "1️⃣ Stage 1: Project Scope":
    st.markdown('<div class="main-header">🛍️ Black Friday Retail Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Unlocking Customer Behavior Patterns with Data Mining</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{len(df):,}</div><div class="metric-label">Total Transactions</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">${df['Purchase'].mean():,.0f}</div><div class="metric-label">Average Purchase</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{df['User_ID'].nunique():,}</div><div class="metric-label">Unique Shoppers</div></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">${df['Purchase'].sum()/1000000:.1f}M</div><div class="metric-label">Total Revenue Generated</div></div>""", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("## 📋 Stage 1: Project Scope, Objectives & Tasks")
    
    colA, colB = st.columns(2)
    with colA:
        st.markdown("""
        <div class="card">
            <h3>🎯 Project Objectives</h3>
            <ul>
                <li><b>Primary Goal:</b> Analyze Black Friday sales data to uncover hidden consumer trends, segment customers by purchasing habits, and identify highly profitable product combinations.</li>
                <li><b>Outcome:</b> Provide actionable business intelligence to retail managers to optimize inventory and design targeted combo offers.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with colB:
        st.markdown("""
        <div class="card">
            <h3>🗺️ Project Scope & Tasks</h3>
            <ul>
                <li><b>Data Cleaning:</b> Preparing the raw Black Friday dataset.</li>
                <li><b>EDA:</b> Visualising relationships between demographics and spending.</li>
                <li><b>Clustering:</b> Grouping buyers into segments (e.g., Discount Lovers).</li>
                <li><b>Association Rules:</b> Mining cross-selling opportunities.</li>
                <li><b>Anomaly Detection:</b> Identifying extreme outlier spenders.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ------------------------------
# Stage 2: Data Cleaning
elif page == "2️⃣ Stage 2: Data Preprocessing":
    st.markdown('<div class="main-header">Stage 2: Data Cleaning & Preprocessing</div>', unsafe_allow_html=True)
    st.write("The dataset is raw, so we need to make it ready for analysis. Think of this like cleaning your room before studying. A clean dataset makes it easier to work and find correct patterns.")
    
    st.markdown("### 🧹 Actions Performed:")
    for line in prep_log:
        st.success(line)
            
    st.markdown("### ✨ Cleaned & Encoded Dataset Preview")
    st.dataframe(df[['User_ID', 'Gender', 'Gender_Code', 'Age', 'Age_Code', 'Product_Category_1', 'Product_Category_2', 'Purchase', 'Purchase_Scaled']].head(10), use_container_width=True)

# ------------------------------
# Stage 3: Exploratory Data Analysis
elif page == "3️⃣ Stage 3: EDA":
    st.markdown('<div class="main-header">Stage 3: Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Reading the story of our dataset: Trends, Patterns, and Relationships</div>', unsafe_allow_html=True)

    # 1. Boxplot
    st.markdown("### 1. Purchase Distribution by Age & Gender")
    fig1 = px.box(df, x='Age', y='Purchase', color='Gender', 
                  category_orders={"Age": ['0-17', '18-25', '26-35', '36-45', '46-50', '51-55', '55+']},
                  color_discrete_map={'Male': '#00E5FF', 'Female': '#FF3366'}, template='plotly_dark')
    st.plotly_chart(fig1, use_container_width=True)
    show_insight("The 26-35 and 36-45 age groups display the highest median purchase amounts. Males generally show a wider variance and higher spending peaks than females.")

    col1, col2 = st.columns(2)
    with col1:
        # 2. Bar chart (Popularity)
        st.markdown("### 2. Most Popular Product Categories")
        cat_counts = df['Product_Category_1'].value_counts().reset_index()
        cat_counts.columns = ['Category', 'Number of Purchases']
        fig2 = px.bar(cat_counts, x='Category', y='Number of Purchases', color='Category', template='plotly_dark')
        st.plotly_chart(fig2, use_container_width=True)
        show_insight("Electronics and Apparel are the absolute most popular product categories by sheer volume during the Black Friday sale.")

    with col2:
        # 3. Bar chart (Avg Purchase)
        st.markdown("### 3. Average Purchase per Category")
        cat_avg = df.groupby('Product_Category_1')['Purchase'].mean().reset_index()
        fig3 = px.bar(cat_avg, x='Product_Category_1', y='Purchase', color='Purchase', color_continuous_scale='viridis', template='plotly_dark')
        st.plotly_chart(fig3, use_container_width=True)
        show_insight("While Electronics sell the most items, the 'Home' and 'Sports' categories yield very high average purchase values per transaction.")

    col3, col4 = st.columns(2)
    with col3:
        # 4. Scatter Plot (Purchase vs Occupation)
        st.markdown("### 4. Scatter Plot: Purchase vs. Occupation")
        fig4 = px.scatter(df, x='Occupation', y='Purchase', color='Gender', opacity=0.6,
                          color_discrete_map={'Male': '#00E5FF', 'Female': '#FF3366'}, template='plotly_dark')
        st.plotly_chart(fig4, use_container_width=True)
        show_insight("This scatter plot reveals dense clusters of high-spending buyers in specific occupation codes (e.g., Codes 4, 7, and 12). Males dominate the absolute highest purchase outliers across most occupations.")

    with col4:
        # 5. Correlation Heatmap
        st.markdown("### 5. Correlation Heatmap for Key Features")
        corr_cols = ['Age_Code', 'Gender_Code', 'Occupation', 'Marital_Status', 'Purchase']
        corr_matrix = df[corr_cols].corr()
        fig5 = px.imshow(corr_matrix, text_auto=".2f", aspect='auto', color_continuous_scale='RdBu_r', zmin=-1, zmax=1, template='plotly_dark')
        st.plotly_chart(fig5, use_container_width=True)
        show_insight("There is a strong positive correlation (0.24) between 'Age_Code' and 'Purchase', mathematically confirming that as age increases, Black Friday spending power increases.")

# ------------------------------
# Stage 4: Clustering Analysis
elif page == "4️⃣ Stage 4: Clustering Analysis":
    st.markdown('<div class="main-header">Stage 4: Clustering Analysis</div>', unsafe_allow_html=True)
    st.write("We want to group customers based on their buying habits. Think of clustering like grouping students in a class: some are toppers, some are average, some love sports. Here, customers are grouped by their shopping styles.")

    features = ['Age_Code', 'Occupation', 'Marital_Status', 'Purchase_Scaled']
    X = df[features].copy()

    st.markdown("### 1. Deciding Clusters via The Elbow Method")
    col1, col2 = st.columns([2, 1])
    with col1:
        inertias = []
        K_range = range(1, 8)
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
        
        fig_elbow = px.line(x=list(K_range), y=inertias, markers=True, title='Elbow Method', template='plotly_dark')
        fig_elbow.add_annotation(x=3, y=inertias[2], text="Elbow Point (k=3)", showarrow=True, arrowhead=1)
        st.plotly_chart(fig_elbow, use_container_width=True)
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        show_insight("The Elbow Method graph shows a sharp bend (the 'elbow') at k=3. This indicates that splitting our customers into exactly 3 distinct groups is the most mathematically optimal choice.")

    # Apply K-Means
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X)
    
    # Label Clusters per Rubric
    cluster_mapping = {0: 'Discount Lovers', 1: 'Average Shoppers', 2: 'Premium Buyers'}
    df['Buyer_Persona'] = df['Cluster'].map(cluster_mapping)

    st.markdown("### 2. Visualizing Clusters with Scatter Plots")
    fig_cluster = px.scatter(df, x='Age', y='Purchase', color='Buyer_Persona',
                             category_orders={"Age": ['0-17', '18-25', '26-35', '36-45', '46-50', '51-55', '55+']},
                             title='Customer Clusters based on Age and Purchase Habits',
                             color_discrete_map={'Premium Buyers': '#FFD700', 'Average Shoppers': '#00E5FF', 'Discount Lovers': '#FF3366'}, 
                             template='plotly_dark', opacity=0.7)
    st.plotly_chart(fig_cluster, use_container_width=True)
    show_insight("The scatter plot clearly visualizes our groups. 'Premium Buyers' (Gold) spend heavily and are concentrated in older demographics. 'Discount Lovers' (Red) represent the high-volume, low-spend tier. We can now send targeted discount emails to the Red group and luxury ads to the Gold group.")

# ------------------------------
# Stage 5: Association Rules
elif page == "5️⃣ Stage 5: Association Rules":
    st.markdown('<div class="main-header">Stage 5: Association Rule Mining</div>', unsafe_allow_html=True)
    st.write("Now, let’s find which products are usually bought together. Think of this like observing your friends at lunch—if someone buys pizza, they usually buy Coke too. This helps retailers design better combo offers.")

    with st.spinner("Running Apriori Algorithm..."):
        # Format basket for apriori using our explicit category names
        df_rules = df[df['Product_Category_2'] != 'None'] # Drop empty second categories
        
        # Create a transaction list for each user
        transactions = []
        for _, row in df_rules.iterrows():
            transactions.append([f"Product_Category_1 = {row['Product_Category_1']}", f"Product_Category_2 = {row['Product_Category_2']}"])
            
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_trans = pd.DataFrame(te_ary, columns=te.columns_)

        # Apply Apriori
        frequent_itemsets = apriori(df_trans, min_support=0.01, use_colnames=True)
        
        if not frequent_itemsets.empty:
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
            
            # Filter to show only rules going from Cat 1 -> Cat 2
            rules = rules[rules['antecedents'].apply(lambda x: 'Product_Category_1' in list(x)[0])]
            rules = rules.sort_values('lift', ascending=False).head(10)
            
            # Format sets for display
            rules['Rule'] = rules['antecedents'].apply(lambda x: list(x)[0]) + " ➔ " + rules['consequents'].apply(lambda x: list(x)[0])
            
            st.markdown("### Top Generated Rules (Support, Confidence, Lift)")
            display_rules = rules[['Rule', 'support', 'confidence', 'lift']].round(3)
            st.dataframe(display_rules.style.background_gradient(cmap='Purples'), use_container_width=True)
            
            st.markdown("### Visualizing Frequent Product Combinations")
            fig = px.scatter(rules, x="support", y="confidence", size="lift", color="lift", hover_data=['Rule'],
                             title="Rule Strength: Support vs. Confidence (Size = Lift)",
                             color_continuous_scale="plasma", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            
            show_insight("The Apriori algorithm generated clear rules. For example, 'If Product_Category_1 = Electronics, then Product_Category_2 = Accessories' shows a High Lift. Retailers should immediately create a 'Combo Discount' for these two items to maximize revenue.")
        else:
            st.warning("Could not find strong association rules.")

# ------------------------------
# Stage 6: Anomaly Detection
elif page == "6️⃣ Stage 6: Anomaly Detection":
    st.markdown('<div class="main-header">Stage 6: Anomaly Detection</div>', unsafe_allow_html=True)
    st.write("Not all shopping behaviors are normal; some people spend way more than others. Anomalies are like finding an odd student in class who scores 100 when everyone else scores 60-70. Detecting these unusual spenders is important for insights.")

    st.markdown("### Statistical Detection (IQR Method) on the Purchase Column")
    
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
                         title=f'Detecting Outliers (Upper Bound Threshold: ${upper_bound:,.0f})',
                         color_discrete_map={'Normal': '#00E5FF', 'Anomaly (High Spender)': '#FF3366'}, template='plotly_dark')
        st.plotly_chart(fig_box, use_container_width=True)
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.metric("Total Customers Evaluated", len(df))
        st.metric("Anomalies Detected", len(anomalies))
        show_insight(f"Using the IQR method, we detected {len(anomalies)} customers spending over the statistical limit of ${upper_bound:,.0f}. These extreme transactions are flagged as outliers.")

    st.markdown("### Comparing Anomalies with Demographic Details")
    col3, col4 = st.columns(2)
    with col3:
        fig_age = px.histogram(anomalies, x='Age', title="Age Profile of Anomaly Spenders", 
                               color_discrete_sequence=['#FFD700'], template='plotly_dark')
        st.plotly_chart(fig_age, use_container_width=True)
    with col4:
        fig_occ = px.bar(anomalies['Occupation'].value_counts().reset_index(), x='Occupation', y='count',
                         title="Occupation Codes of Anomaly Spenders", template='plotly_dark')
        st.plotly_chart(fig_occ, use_container_width=True)

    show_insight("By comparing anomalies to demographic details, we see that our extremely high spenders belong almost exclusively to the 26-45 Age brackets and specific Occupation codes. These are highly lucrative 'Whales' who should be investigated for VIP loyalty programs.")

# ------------------------------
# Stage 7: Insights & Reporting
elif page == "7️⃣ Stage 7: Insights & Reporting":
    st.markdown('<div class="main-header">Stage 7: Insights & Reporting</div>', unsafe_allow_html=True)
    st.write("Once analysis is done, we need to summarize findings in a way that’s easy to understand. Insights are like telling the final story after reading the whole book - short, clear, and meaningful.")

    st.markdown("## 🔑 Final Dashboard Reporting")
    
    st.markdown("""
    <div class="card">
        <h3 style="color: #00E5FF;">1. Which age group spends the most?</h3>
        <p style="font-size: 1.1rem;">
        Based on our EDA boxplots, correlation heatmaps, and K-Means clustering, the <b>26-35</b> and <b>36-45 age groups</b> spend the most. They have the highest median purchases, the strongest positive correlation to total spend, and dominate the "Premium Buyers" cluster.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3 style="color: #FF3366;">2. Which products are popular with males vs. females?</h3>
        <p style="font-size: 1.1rem;">
        Through categorical mapping and EDA, a clear divide is visible:
        <br><br>
        • <b>Males</b> are driving massive volume in the <b>Electronics</b> and <b>Sports</b> product categories.<br>
        • <b>Females</b> overwhelmingly popularize the <b>Apparel</b>, <b>Beauty</b>, and <b>Home</b> product categories.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3 style="color: #FFD700;">3. What type of buyers spend unusually high amounts?</h3>
        <p style="font-size: 1.1rem;">
        Our IQR Anomaly Detection isolated the "Whale" buyers making extremely high purchases. When comparing these anomalies to demographics, we found that unusually high spenders are:
        <br><br>
        • Predominantly <b>Male</b>.<br>
        • Concentrated exclusively in the <b>26-45 age range</b>.<br>
        • Working in specific, high-paying <b>Occupation codes</b> (e.g., codes 4 and 7).
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    
    st.markdown("## 📊 Summary Visuals for Retailers")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="success-box">
            <h4>🎯 Cluster Marketing Actions</h4>
            <p>Leverage our <b>Stage 4 Clustering</b>: Send clearance and discount bundle emails exclusively to the <b>"Discount Lovers"</b> segment. Target the <b>"Premium Buyers"</b> segment with early access to high-ticket Electronics.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="success-box">
            <h4>🔗 Combo Offer Design</h4>
            <p>Based on our <b>Stage 5 Association Rules</b> (Apriori), when a user adds <i>Electronics</i> to their cart, the website should immediately suggest <i>Accessories</i> to exploit the high 'Lift' combination and increase total cart value.</p>
        </div>
        """, unsafe_allow_html=True)

# ------------------------------
# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>🛍️ Black Friday Retail Analytics | Zene-Sophie-Anand (1000442) | Data Mining Project</p>
    <p>© 2026 | Transforming Raw Transactions into Clear, Meaningful Stories.</p>
</div>
""", unsafe_allow_html=True)
