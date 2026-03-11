
# 🛒 Mining the Future: Beyond Discounts Data Driven Black Friday Sales Insights

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-streamlit-link-here.streamlit.app)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## 🎓 **Student Information**
* **Name:** Zene-Sophie-Anand
* **Candidate Registration Number (WACP):** 1000442
* [cite_start]**CRS Name:** Artificial Intelligence [cite: 2]
* [cite_start]**Course Name:** Data Mining / IBCP (AI) [cite: 2]
* **School Name:** Aspen Nutan Academy

---

## 🚀 **1. Brief Project Title and Scope**
[cite_start]**Project Title:** Beyond Discounts Data Driven Black Friday Sales Insights[cite: 2].

[cite_start]**Project Scope:** The objective of this project is to act as a Data Analyst at InsightMart Analytics to understand shopping behavior during a Black Friday mega sale[cite: 2]. [cite_start]The scope involves analyzing customer purchase patterns, clustering shoppers into distinct groups, and identifying associations between product categories[cite: 2]. [cite_start]Additionally, the project aims to detect anomalies, such as unusually high spenders, to help retailers understand preferences, segment customers effectively, identify cross-selling opportunities, and ultimately deploy these insights via a Streamlit cloud app[cite: 2].

---

## 🌐 **2. Deployed Project on Streamlit (Overview & Functionality)**
**Streamlit App Link:** [Insert Your Streamlit App Link Here]

[cite_start]**Overview & Functionality:** This project is deployed as an interactive web application on Streamlit Cloud[cite: 2]. [cite_start]The app provides a user-friendly dashboard to visualize sales patterns and deliver actionable insights for decision-making[cite: 2]. 
* **Functionality:** Users can navigate through different stages of the data mining process (EDA, Clustering, Association Rules, and Anomaly Detection) via an interactive sidebar. The dashboard dynamically renders Plotly visualizations, statistical summaries, and highlighted anomalies for immediate business intelligence.

---

## ⚙️ **3. Key Preprocessing Steps**
[cite_start]To prepare the raw dataset for accurate analysis, several data cleaning and preprocessing steps were applied[cite: 2]:
* [cite_start]**Handling Missing Values:** Filled and handled missing or null values in `Product_Category_2` and `Product_Category_3`[cite: 2].
* [cite_start]**Encoding Categorical Data:** Converted text data into numerical formats, including converting `Gender` into binary numbers (Male = 0, Female = 1) and transforming `Age` groups into ordered numeric codes (e.g., 0–17 → 1, 18–25 → 2)[cite: 2].
* [cite_start]**Normalization:** Normalized the continuous `Purchase` amounts so that values are on the same standard scale for clustering algorithms[cite: 2].
* [cite_start]**Data Integrity:** Checked for and handled any duplicates or irrelevant data points[cite: 2].

---

## 📊 **4. Exploratory Data Analysis (EDA) & Visualizations**
[cite_start]Exploratory Data Analysis was conducted to uncover trends and relationships in the data[cite: 2]:
* [cite_start]**Distribution Analysis:** Created box plots and histograms to visualize `Purchase` amounts distributed by `Age` and `Gender`[cite: 2].
* [cite_start]**Product Popularity:** Utilized charts (such as violin plots) to determine which product categories are the most popular and yield the highest purchase volumes[cite: 2].
* [cite_start]**Relationship Mapping:** Drawn a correlation heatmap to identify statistical relationships between key numeric features like age, occupation, and purchase amount[cite: 2].

---

## 🧠 **5. Advanced Analytics (Clustering, Association & Anomalies)**

### **Clustering Analysis**
* [cite_start]**Technique Applied:** K-Means Clustering[cite: 2].
* [cite_start]**Implementation:** Grouped customers based on demographic and purchasing habits using features like Age, Occupation, and normalized Purchase amounts[cite: 2].
* [cite_start]**Outcome:** Labeled clusters into distinct business segments (e.g., "Value Hunters", "Premium Spenders") to allow for tailored marketing strategies[cite: 2].

### **Association Rule Mining**
* [cite_start]**Technique Applied:** Apriori Algorithm[cite: 2].
* [cite_start]**Implementation:** Generated frequent itemsets to discover which `Product_Category_1` items are usually bought together[cite: 2].
* [cite_start]**Outcome:** Evaluated combinations using Support, Confidence, and Lift metrics to find actionable cross-selling opportunities and bundle offers[cite: 2].

### **Anomaly Detection**
* [cite_start]**Technique Applied:** Statistical Outlier Detection[cite: 2].
* [cite_start]**Implementation:** Analyzed the `Purchase` column to find extremely high spenders whose behavior deviates significantly from the norm[cite: 2].
* [cite_start]**Outcome:** Isolated and flagged unusually large transactions to help the business spot "whale" buyers or potential bulk purchasing anomalies[cite: 2].

---

## 💡 **6. Main Insights & Findings**
* [cite_start]**Demographic Spending:** Visualizations revealed which specific age groups and genders spend the most during the Black Friday sale[cite: 2].
* **Customer Segmentation:** Successfully identified distinct buyer personas. High-value spenders can be targeted with premium categories, while budget-conscious clusters are ideal for discount bundles.
* [cite_start]**Product Combinations:** Discovered products frequently bought together (high 'Lift'), allowing the retailer to strategically place these items together or create combo deals[cite: 2].
* [cite_start]**Extreme Spenders:** Detected a subset of buyers making unusually large purchases compared to their demographic averages, providing an opportunity for VIP targeting or fraud review[cite: 2].

---

## 📁 **7. Repository Structure Guide**

```text
IDAI105(1000442)-zene-sophie-anand/
│
├── app.py                 # Main Streamlit application and ML pipeline
[cite_start]├── requirements.txt       # Python library dependencies (pandas, scikit-learn, streamlit, etc.) [cite: 2]
├── data/
[cite_start]│   └── dataset.csv        # The Black Friday Sales dataset used in the project [cite: 2]
│
[cite_start]└── README.md              # Detailed project documentation and insights [cite: 2]

```

---

## 📚 **8. References & Resources**

The following resources were utilized to support the methodologies and visualizations in this project:

* 
**Visualizations:** Data-to-Viz (https://www.data-to-viz.com/) 


* 
**Clustering:** K-Means Clustering Concepts (https://neptune.ai/blog/k-means-clustering) 


* 
**Market Basket Analysis:** Guide on Market Basket Analysis (https://www.analyticsvidhya.com/blog/2021/10/a-comprehensive-guide-on-market-basket-analysis/) 


* 
**Anomaly Detection:** DataCamp Anomaly Detection in Python (https://www.datacamp.com/courses/anomaly-detection-in-python) 


* 
**Libraries Used:** Pandas, Matplotlib, Seaborn, Plotly Graph Objects, Scikit-learn, Streamlit.

```
