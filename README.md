
# 🛍️ Beyond Discounts: Data-Driven Black Friday Sales Insights

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-streamlit-link-here.streamlit.app)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Data Mining](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange.svg)]()

An interactive Business Intelligence web application built with **Streamlit**. This project applies Advanced Data Mining techniques to analyze Black Friday sales data, segment customers, discover product associations, and detect anomalous spending behaviors.

---

## 🎓 **Academic Details**
* **Developer:** Zene-Sophie-Anand
* **WACP No:** 1000442
* **CRS Subject:** Artificial Intelligence
* **Course Name:** Data Mining / IBCP (AI)
* **Institution:** Aspen Nutan Academy

---

## 🚀 **Live Dashboard**
The interactive data mining dashboard is deployed live on Streamlit Community Cloud. 
👉 **[Click here to view the live app](https://idai105-1000442-zene-sophie-anand-sa.streamlit.app/)**

*(Note to grader: Please click the link above to view the interactive visualizations and insights).*

---

## 📋 **Project Scope & Objectives (Stage 1)**
As a Data Analyst at InsightMart Analytics, the objective of this project is to understand shopping behavior during a Black Friday mega sale. 
* **Goal:** Uncover hidden consumer trends, segment customers by purchasing habits, and identify highly profitable product combinations.
* **Outcome:** Provide actionable business intelligence to retail managers to optimize inventory and design targeted combo offers.

---

## 🧠 **Methodology & Advanced Analytics**

### **🧹 Stage 2: Data Cleaning & Preprocessing**
To ensure accurate machine learning models, the raw data was cleaned:
1. Checked for and removed duplicate/irrelevant rows.
2. Handled missing values in `Product_Category_2` and `Product_Category_3` by filling them.
3. Encoded `Gender` into binary numeric values (Male = 0, Female = 1).
4. Encoded `Age` groups into ordered numbers (0-17 → 1, 18-25 → 2, up to 7).
5. Normalized the continuous `Purchase` amounts using `StandardScaler` so values share the same scale for clustering.

### **📊 Stage 3: Exploratory Data Analysis (EDA)**
Visualized the dataset to uncover baseline trends using Plotly:
* **Box plots** of Purchase by Age and Gender.
* **Bar charts** revealing the most popular product categories by volume and average revenue.
* **Scatter plots** mapping Purchases against Occupations and Years in Current City.
* **Correlation Heatmap** proving the mathematical relationship between age, gender, and spending power.

### **🎯 Stage 4: Clustering Analysis (Customer Segmentation)**
* **Algorithm:** K-Means Clustering (`scikit-learn`).
* **Process:** Used the **Elbow Method** to mathematically determine the optimal number of clusters ($k=3$). 
* **Business Labels:** Segmented users into "Discount Lovers", "Average Shoppers", and "Premium Buyers" to allow for highly targeted email marketing.

### **🔗 Stage 5: Association Rule Mining (Market Basket)**
* **Algorithm:** Apriori Algorithm (`mlxtend`).
* **Process:** Mined frequent itemsets to discover which product categories are usually bought together.
* **Outcome:** Generated cross-selling rules (e.g., *If Electronics, then Accessories*) evaluated by Support, Confidence, and **Lift** to design combo offers.

### **⚠️ Stage 6: Anomaly Detection (Outliers)**
* **Method:** Interquartile Range (IQR) Statistical Method.
* **Process:** Isolated extremely high spenders ("Whales") who breached the upper statistical bound.
* **Outcome:** Compared anomalies to demographic details (Age & Occupation) to flag high-value VIP targets.

### **📈 Stage 7: Insights & Reporting**
The project successfully answers the core business questions:
1. **Age & Spending:** The 26-35 and 36-45 age groups spend the most on average.
2. **Gender Preferences:** Males drive volume in Electronics/Sports; Females drive volume in Apparel/Beauty/Home.
3. **Anomalies:** Unusually high spenders are predominantly 26-45 year-old males in specific, high-paying occupation codes.

---

## 🖥️ UI Dashboard

 ![Dashboard 1](./SCREEN%20SHOT/1.png)  ![Dashboard 2](./SCREEN%20SHOT/2.png)  ![Dashboard 3](./SCREEN%20SHOT/3.png) 

 ![Dashboard 4](./SCREEN%20SHOT/4.png)  ![Dashboard 5](./SCREEN%20SHOT/5.png)  ![Dashboard 6](./SCREEN%20SHOT/6.png) 

 ![Dashboard 7](./SCREEN%20SHOT/7.png)  ![Dashboard 8](./SCREEN%20SHOT/8.png)  ![Dashboard 9](./SCREEN%20SHOT/9.png) 

 ![Dashboard 10](./SCREEN%20SHOT/10.png)  ![Dashboard 11](./SCREEN%20SHOT/11.png)  ![Dashboard 12](./SCREEN%20SHOT/12.png) 

 ![Dashboard 13](./SCREEN%20SHOT/13.png)  ![Dashboard 14](./SCREEN%20SHOT/14.png) 

---
```text
IDAI105(1000442)-zene-sophie-anand/
│
├── app.py                 # Main Streamlit dashboard and ML pipeline
├── requirements.txt       # Python library dependencies 
├── SCREEN SHOT/           # Folder containing UI documentation images
│   ├── #1.png
│   └── ...
└── README.md              # Project documentation and rubric evidence
```

---

## 🛠️ **Installation & Local Setup**

To run this project on your local machine:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/zeneanand/IDAI105-1000442-ZENE-SOPHIE-ANAND.git](https://github.com/zeneanand/IDAI105-1000442-ZENE-SOPHIE-ANAND.git)
   cd IDAI105-1000442-ZENE-SOPHIE-ANAND
   ```

2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit server:**
   ```bash
   streamlit run app.py
   ```
```
