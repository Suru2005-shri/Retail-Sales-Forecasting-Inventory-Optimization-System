# Retail Sales Forecasting & Inventory Optimization System

> **An end-to-end machine learning system for demand forecasting and intelligent inventory management — built as industry-grade portfolio proof for Data Analyst, Business Analyst, and Data Science roles.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

##  Project Overview

This project simulates a **real-world retail intelligence system** used by companies like DMart, Reliance Retail, Amazon, and Flipkart. It ingests historical sales data, forecasts future demand using machine learning, and recommends optimal inventory decisions — all in an automated, reproducible pipeline.

**What it does:**
- Generates a realistic synthetic retail dataset (137,000+ records across 5 stores, 25 products, 3 years)
- Cleans and preprocesses data with proper feature extraction
- Performs Exploratory Data Analysis with 15 professional visualizations
- Engineers lag features, rolling averages, and time-series signals
- Trains and compares Linear Regression, Random Forest, and XGBoost models
- Applies inventory formulas: **Safety Stock, Reorder Point (ROP), Economic Order Quantity (EOQ)**
- Generates automated reorder alerts and business recommendations
- Delivers an interactive Streamlit dashboard

---

##  Problem Statement

Retailers face two critical inventory problems:
- **Stockouts** → lost sales, unhappy customers, revenue leakage
- **Overstock** → high holding costs, wastage, capital locked in unsold goods

**This system solves both** by forecasting future demand accurately and prescribing the exact quantity to order and when to order it.

---

##  Industry Relevance

| Company | How They Use This |
|---|---|
| Amazon / Flipkart | Demand forecasting per SKU per warehouse |
| DMart / Reliance Retail | Category-wise replenishment planning |
| Walmart | Multi-store inventory balancing |
| Swiggy / Zepto | Dark store stock optimization |
| FMCG Companies | Production planning and distribution |

---

##  Results Summary

| Metric | Value |
|---|---|
| Dataset Size | 137,000 records |
| Total Revenue Simulated | ₹2,588 Crore |
| Best Model | Random Forest Regressor |
| R² Score | **0.93** |
| MAE | **3.48 units/day** |
| RMSE | **5.34 units/day** |
| SKUs Needing Reorder | 56 out of 125 |
| Stockout Rate | 0.8% |

---

##  Tech Stack

```
Language      : Python 3.10+
Data          : Pandas, NumPy
Visualization : Matplotlib, Seaborn
ML Models     : Scikit-learn (Linear Regression, Random Forest)
               XGBoost (optional, install separately)
Inventory     : Custom Python logic (EOQ, ROP, Safety Stock)
Dashboard     : Streamlit
Serialization : Joblib
Environment   : Virtual Environment / Conda
```

---

##  Architecture

```
RAW SALES DATA
     │
     ▼
┌─────────────────────┐
│  Data Preprocessing  │  → date parsing, dtype correction, feature extraction
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│  EDA Module          │  → 7 charts: trends, seasonality, stockouts, promotions
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│  Feature Engineering │  → lag_7d/14d/30d, rolling_mean, EWM, encoding
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│  Forecasting Model   │  → Linear Regression → Random Forest → XGBoost
└─────────────────────┘
     │                          R² = 0.93, MAE = 3.48 units
     ▼
┌─────────────────────┐
│  Inventory Engine    │  → Safety Stock, ROP, EOQ per SKU
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│  Reorder Alerts +    │  → 56 SKUs flagged, ₹3.2Cr reorder value
│  Business Report     │
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│  Streamlit Dashboard │  → Interactive KPIs, filters, charts, tables
└─────────────────────┘
```

---

##  Folder Structure

```
Retail-Sales-Forecasting-Inventory-Optimization/
│
├── data/
│   ├── retail_sales_data.csv       ← Raw synthetic dataset (137K records)
│   ├── cleaned_data.csv            ← Preprocessed with date features
│   └── features.csv                ← ML-ready with lag/rolling features
│
├── src/
│   ├── generate_dataset.py         ← Synthetic data generator
│   ├── preprocess.py               ← Cleaning & feature extraction
│   ├── eda.py                      ← 7 EDA visualizations
│   ├── feature_engineering.py      ← Lag, rolling, EWM features
│   ├── forecasting_model.py        ← Model training & evaluation
│   ├── inventory_optimization.py   ← EOQ, ROP, Safety Stock logic
│   └── insights.py                 ← Dashboard image + executive report
│
├── app/
│   └── dashboard.py                ← Streamlit interactive dashboard
│
├── models/
│   └── best_model.pkl              ← Saved trained model (joblib)
│
├── outputs/
│   ├── graphs/                     ← 15 chart PNGs
│   ├── tables/                     ← predictions.csv, inventory_report.csv
│   └── reports/                    ← executive_summary.txt
│
├── images/                         ← Screenshots for README
├── notebooks/                      ← Jupyter notebooks (optional)
├── docs/                           ← Extra documentation
│
├── main.py                         ← ⭐ Single pipeline runner
├── requirements.txt
├── .gitignore
└── README.md
```

---

##  Installation & Setup

### Step 1: Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/Retail-Sales-Forecasting-Inventory-Optimization.git
cd Retail-Sales-Forecasting-Inventory-Optimization
```

### Step 2: Create virtual environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the complete pipeline
```bash
python main.py
```

### Step 5: Launch the dashboard
```bash
streamlit run app/dashboard.py
```

---

##  Dataset Details

The dataset is **synthetically generated** to simulate realistic retail conditions:

| Field | Description |
|---|---|
| `date` | Daily record from 2022-01-01 to 2024-12-31 |
| `store` | 5 Indian city stores (Mumbai, Delhi, Bangalore, Chennai, Pune) |
| `category` | Electronics, Clothing, Groceries, Home & Kitchen, Sports |
| `product` | 25 SKUs (5 per category) |
| `units_sold` | Daily units sold (includes seasonality + noise + promotions) |
| `unit_price` | Price with ±5% variation |
| `revenue` | units_sold × unit_price |
| `opening_stock` | Simulated stock level |
| `stockout_flag` | 1 if closing stock = 0 |
| `promotion` | 1 on promotion days (15% of days randomly) |

**Seasonality built-in:**
-  Electronics/Clothing spike in Oct–Dec (festival season)
-  Sports spike in May–June (summer)
-  Groceries boost in Jul–Aug (monsoon)

---

##  How to Run (Step by Step)

```bash
# Run individual modules
python src/generate_dataset.py        # Step 1: Create dataset
python src/preprocess.py              # Step 2: Clean data
python src/eda.py                     # Step 3: EDA charts
python src/feature_engineering.py    # Step 4: Build features
python src/forecasting_model.py      # Step 5: Train models
python src/inventory_optimization.py # Step 6: Inventory logic
python src/insights.py               # Step 7: Dashboard + report

# OR run everything at once
python main.py
```

---

##  Key Formulas Used

### Safety Stock
```
Safety Stock = Z × σ_demand × √Lead_Time
(Z = 1.65 for 95% service level)
```

### Reorder Point (ROP)
```
ROP = (Avg Daily Demand × Lead Time) + Safety Stock
```

### Economic Order Quantity (EOQ)
```
EOQ = √(2 × D × S / H)
D = Annual Demand  |  S = Ordering Cost  |  H = Holding Cost
```

---

##  Screenshots

*(After running `python main.py`, all charts are saved to `outputs/graphs/`)*

| Chart | Description |
|---|---|
| `01_monthly_revenue_trend.png` | Revenue trend across 3 years |
| `04_seasonality_heatmap.png` | Month × Category demand heatmap |
| `08_actual_vs_predicted.png` | Model accuracy visualization |
| `09_feature_importance.png` | Most impactful ML features |
| `11_reorder_alerts.png` | Top SKUs needing reorder |
| `15_insights_dashboard.png` | Full business dashboard |

---

##  Future Improvements

- [ ] Add XGBoost & Prophet time-series forecasting
- [ ] Multi-store demand transfer logic
- [ ] Price elasticity and promotional uplift modeling
- [ ] Weather/event-based demand signals
- [ ] Real-time API integration with ERP systems
- [ ] Anomaly detection for unusual sales spikes
- [ ] Region-wise heatmap with Folium maps
- [ ] Automated email reorder alerts

---

##  Learning Outcomes

After building this project you will understand:
- End-to-end ML pipeline design for business problems
- Time-series feature engineering (lags, rolling stats)
- Inventory management formulas used in real supply chains
- How to evaluate regression models (MAE, RMSE, R²)
- Building interactive dashboards with Streamlit
- Professional GitHub documentation and commit strategy

---

##  Author

**Shruti Srivastava**
- B.Tech
- Aspiring Data Analyst / Business Analyst / Data Scientist
- LinkedIn: www.linkedin.com/in/shruti-srivastava-36b26232a


---

##  License

This project is open source under the [MIT License](LICENSE).

