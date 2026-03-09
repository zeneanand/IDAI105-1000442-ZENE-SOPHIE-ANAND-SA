import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
warnings.filterwarnings('ignore')

# --- PAGE SETUP ---
st.set_page_config(page_title="Black Friday Insights", page_icon="🛒", layout="wide")
st.title("🛒 Black Friday Sales: Data-Driven Business Intelligence")
st.markdown("Interactive dashboard to uncover shopping patterns, customer segments, and cross-selling opportunities.")

# --- DATA LOADING & PREPROCESSING (Stage 2) ---
@st.cache_data
def load_and_preprocess_data():
    """Generates dummy data mimicking the Black Friday dataset and preprocesses it."""
    np.random.seed(42)
    data_size = 2000
    
    # 1. Raw Data Generation
    df = pd.DataFrame({
        'User_ID': np.random.randint(1000, 2000, data_size),
        'Product_ID': ['P00' + str(i) for i in np.random.randint(1, 50, data_size)],
        'Gender': np.random.choice(['M', 'F'], size=data_size, p=[:6, .4]),
        'Age': np.random.choice(['0-17', '18-25', '26-35', '36-45', '46-50', '51-55', '55+'], size=data_size),
        'Occupation': np.random.randint(0, 20, data_size),
        'City_Category': np.random.choice(['A', 'B', 'C'], size=data_size),
        'Stay_In_Current_City_Years': np.random.choice(['0', '1', '2', '3', '4+'], size=data_size),
        'Marital_Status': np.random.choice([0, 1], size=data_size),
        'Product_Category_1': np.random.randint(1, 20, data_size),
        'Product_Category_2': np.random.choice([np.nan, 2, 4, 6, 8, 14, 16], size=data_size),
        'Product_Category_3': np.random.choice([np.nan, 5, 8, 12, 15, 18], size=data_size),
        'Purchase': np.random.normal(9000, 3000, data_size)
    })
    
    # Introduce some anomalies (extremely high spenders)
    anomaly_indices = np.random.choice(df.index, 20, replace=False)
    df.loc[anomaly_indices, 'Purchase'] = np.random.uniform(25000, 35000, 20)
    
    # 2. Preprocessing
    # Handle missing values in Product_Category 2 & 3
    df['Product_Category_2'].fillna(0, inplace=True)
    df['Product_Category_3'].fillna(0, inplace=True)
    
    # Encode Categorical Data
    df['Gender_Code'] = df['Gender'].map({'M': 0, 'F': 1})
    age_mapping = {'0-17': 1, '18-25': 2, '26-35': 3, '36-45': 4, '46-50': 5, '51-55': 6, '55+': 7}
    df['Age_Code'] = df['Age'].map(age_mapping)
    
    # Normalize Purchase amounts for clustering later
    scaler = StandardScaler()
    df['Purchase_Scaled'] = scaler.fit_transform(df[['Purchase']])
    
    return df

df = load_and_preprocess_data()

# --- SIDEBAR NAVIGATION ---
st.sidebar.header("Navigation")
menu = st.sidebar.radio("Select Analysis Stage:", [
    "1. Project Scope & Overview",
    "2. Exploratory Data Analysis (EDA)",
    "3. Customer Clustering",
    "4. Association Rule Mining",
    "5. Anomaly Detection"
])

st.sidebar.markdown("---")
st.sidebar.info(f"**Dataset loaded:** {df.shape[0]} rows, {df.shape[1]} columns")

# --- 1. PROJECT SCOPE ---
if menu == "1. Project Scope & Overview":
    st.header("Stage 1: Define the Project Scope")
    st.write("**Objective:** To understand shopping patterns during the Black Friday mega sale.")
    st.markdown("""
    * **Understand shopping preferences:** Discover how demographics influence purchases.
    * **Segment Customers Effectively:** Use K-Means clustering to identify distinct shopping groups.
    * **Identify Cross-Selling Opportunities:** Apply Apriori association rule mining to uncover frequent product combinations.
    * **Detect Anomalies:** Spot unusual purchase behavior (extremely high spenders) using statistical bounds.
    """)
    st.dataframe(df.head(10))

# --- 2. EDA ---
elif menu == "2. Exploratory Data Analysis (EDA)":
    st.header("Stage 3: Exploratory Data Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Purchase Distribution by Gender")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x='Gender', y='Purchase', data=df, palette='Set2', ax=ax)
        st.pyplot(fig)
        
    with col2:
        st.subheader("Purchase Amount by Age Group")
        fig, ax = plt.subplots(figsize=(6, 4))
        age_order = ['0-17', '18-25', '26-35', '36-45', '46-50', '51-55', '55+']
        sns.barplot(x='Age', y='Purchase', data=df, order=age_order, palette='viridis', ci=None, ax=ax)
        st.pyplot(fig)
        
    st.subheader("Correlation Heatmap of Key Features")
    fig, ax = plt.subplots(figsize=(8, 5))
    corr_cols = ['Age_Code', 'Gender_Code', 'Occupation', 'Product_Category_1', 'Purchase']
    sns.heatmap(df[corr_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)

# --- 3. CLUSTERING ---
elif menu == "3. Customer Clustering":
    st.header("Stage 4: Clustering Analysis (Customer Segmentation)")
    st.write("Using K-Means Clustering to group customers based on Age, Occupation, and Purchase behavior.")
    
    # Feature selection
    features = ['Age_Code', 'Occupation', 'Purchase_Scaled']
    X = df[features]
    
    # Apply K-Means
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X)
    
    # Label clusters logically based on Purchase
    cluster_mapping = {0: 'Budget Shoppers', 1: 'Standard Buyers', 2: 'Premium Spenders'}
    df['Customer_Segment'] = df['Cluster'].map(cluster_mapping)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='Age_Code', y='Purchase', hue='Customer_Segment', data=df, palette='deep', alpha=0.7, ax=ax)
    plt.xticks(ticks=[1, 2, 3, 4, 5, 6, 7], labels=['0-17', '18-25', '26-35', '36-45', '46-50', '51-55', '55+'])
    plt.title("Customer Segments by Age and Purchase Amount")
    st.pyplot(fig)
    
    st.success("Insight: We successfully segmented users. Premium spenders can be targeted with high-ticket electronics, while Budget Shoppers can be targeted with bundle discounts.")

# --- 4. ASSOCIATION RULES ---
elif menu == "4. Association Rule Mining":
    st.header("Stage 5: Association Rule Mining (Market Basket Analysis)")
    st.write("Finding which Product Categories are frequently bought together using the Apriori algorithm.")
    
    # Prepare data for Apriori (One-hot encoding categories per user)
    # Grouping categories bought by the same user
    basket = df.groupby(['User_ID', 'Product_Category_1'])['Product_Category_1'].count().unstack().reset_index().fillna(0).set_index('User_ID')
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)
    
    # Run Apriori
    frequent_itemsets = apriori(basket, min_support=0.05, use_colnames=True)
    
    if not frequent_itemsets.empty:
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
        st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))
        st.info("Insight: A high 'lift' (>1) indicates that customers buying the antecedent product category are highly likely to also buy the consequent category. Retailers should place these categories near each other or create bundle deals.")
    else:
        st.write("No strong associations found with the current support threshold.")

# --- 5. ANOMALY DETECTION ---
elif menu == "5. Anomaly Detection":
    st.header("Stage 6: Anomaly Detection (High Spenders)")
    st.write("Using statistical methods (Interquartile Range - IQR) to detect exceptionally high purchases.")
    
    Q1 = df['Purchase'].quantile(0.25)
    Q3 = df['Purchase'].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    
    df['Is_Anomaly'] = df['Purchase'] > upper_bound
    anomalies = df[df['Is_Anomaly']]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df['Purchase'], bins=50, color='lightgray', ax=ax, label='Normal Purchases')
    sns.scatterplot(x=anomalies['Purchase'], y=np.zeros(len(anomalies)), color='red', s=100, label='Anomalies', ax=ax)
    plt.axvline(upper_bound, color='red', linestyle='--', label=f'Threshold ({upper_bound:.0f})')
    plt.legend()
    st.pyplot(fig)
    
    st.warning(f"Detected {len(anomalies)} anomalies (extremely high spenders).")
    st.write("Sample of Anomalous Transactions:")
    st.dataframe(anomalies[['User_ID', 'Gender', 'Age', 'Occupation', 'Purchase']].head())
