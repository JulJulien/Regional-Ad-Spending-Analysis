# 🥤 Coca-Cola Regional Ad Spending Analysis

## 📊 Overview

This project analyzes **Coca-Cola's advertising efficiency across global
regions** using machine learning.\
By simulating and evaluating ad spending data from 12 international
regions, the study identifies **which marketing channels (TV, Digital,
Outdoor, Retail)** and **regional strategies** yield the highest
**Return on Investment (ROI)**.

An **XGBoost Regression Model** was trained to predict ROI based on
spending habits, environmental variables, and competitor activity.\
Final model achieved **R² = 0.94**, revealing clear non-linear
relationships between spending and ROI.

------------------------------------------------------------------------

## 🎯 Objectives

-   Discover which **regions** deliver the highest ROI.\
-   Identify **optimal spending strategies** per advertising channel.\
-   Build a **predictive model** to forecast ROI given a region's ad
    spending conditions.\
-   Generate **data-driven recommendations** to maximize marketing
    returns.

------------------------------------------------------------------------

## 🧠 Methodology

### 1. **Data Simulation & Preparation**

-   Generated synthetic data for 12 Coca-Cola regions (Africa,
    Asia-Pacific, Europe, etc.) across 3 years (2022--2024).
-   Spending categories:
    -   `ad_spend_tv`\
    -   `ad_spend_digital`\
    -   `ad_spend_outdoor`\
    -   `ad_spend_retail`\
-   Additional factors: `avg_temp`, `holiday`, `pepsi_spend`,
    `sales_volume`, and calculated `roi` and `profit`.

### 2. **Exploratory Data Analysis (EDA)**

-   Distribution tests using **Shapiro-Wilk** confirmed non-normality (p
    \< 0.05).\
-   Correlation analysis revealed strong *negative* correlations between
    ROI and ad spending.\
-   Conclusion: Linear models inappropriate → non-linear ensemble
    methods preferred.

### 3. **Feature Engineering**

-   Added engineered columns:
    -   `profit = roi × total_spend`\
    -   Encoded categorical features (`region_encoded`, `year_encoded`)\
    -   Reordered features for model consistency.

### 4. **Modeling: XGBoost Regressor**

-   **Why XGBoost?**
    -   Handles non-normal data.\
    -   Detects complex, non-linear interactions.\
    -   Prevents overfitting via regularization and tree-based
        structure.

**Hyperparameters** \| Parameter \| Value \| Description \|
\|------------\|--------\|-------------\| \| n_estimators \| 500 \|
Boosting rounds \| \| max_depth \| 5 \| Shallow trees to prevent overfit
\| \| learning_rate \| 0.05 \| Slower, stable learning \| \| subsample
\| 0.8 \| Random row sampling \| \| colsample_bytree \| 0.8 \| Random
feature sampling \| \| random_state \| 42 \| Reproducibility \|

**Results**

    MAE: 0.179
    RMSE: 0.059
    R²: 0.944

Residual plots confirm stable variance and minimal bias.

### 5. **Prediction & Optimization**

-   Predicted ROI across all regions under varying spending conditions.
-   Identified **top 3 spending conditions per region** with highest
    predicted ROI.
-   Final dataframe provides **region-specific recommendations** (e.g.,
    ideal TV/Digital/Outdoor allocations).

------------------------------------------------------------------------

## 💼 Business Insights

-   **TV and Digital** are the top-performing ad types globally.\
-   ROI declines with overspending --- diminishing returns appear after
    moderate budgets.\
-   Some regions (e.g., **Europe, Northeast U.S.**) consistently yield
    higher ROI with lower ad spend.\
-   **Data-driven strategy**: Maintain moderate digital presence, reduce
    outdoor spending in underperforming regions, and reinvest savings
    into TV or high-ROI geographies.

------------------------------------------------------------------------

## 🧩 Tech Stack

-   **Languages:** Python\
-   **Libraries:**\
    `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `xgboost`,
    `scikit-learn`\
-   **Environment:** Jupyter Notebook\
-   **Dataset:** Synthetic (generated internally for model
    demonstration)

------------------------------------------------------------------------

## 📈 Key Visuals

-   Feature distribution plots (Bell Curves)\
-   Correlation heatmap\
-   Residual and importance plots from XGBoost\
-   Regional ROI bar charts

------------------------------------------------------------------------

## 🚀 How to Run

``` bash
# 1. Clone the repo
git clone https://github.com/JulJulien/Regional-Ad-Spending-Analysis.git

# 2. Navigate to the project
cd Regional-Ad-Spending-Analysis

# 3. Install dependencies
pip install -r requirements.txt

# 4. Open the notebook
jupyter notebook coca-cola.ipynb
```

------------------------------------------------------------------------

## 🧾 Files

  ----------------------------------------------------------------------------------
  File                             Description
  -------------------------------- -------------------------------------------------
  `coca-cola.ipynb`                Main analysis notebook

  `src/`                           Helper modules for data generation & plotting

  `synthetic_data_generation.py`   Script to build synthetic Coca-Cola dataset

  `modules.py`                     Custom plotting and EDA functions
  ----------------------------------------------------------------------------------

------------------------------------------------------------------------

## 🏁 Results Summary

  Metric            Value
  ----------------- -------------------------------------------------------
  Model Type        XGBoost Regressor
  R² Score          **0.944**
  MAE               0.179
  RMSE              0.059
  Top ROI Drivers   TV, Digital
  Key Finding       Optimal spending is *non-linear* and region-dependent

------------------------------------------------------------------------

## 🧃 Author

**Carlos \[JulJulien\]**\
📫 *Aspiring Data Scientist passionate about turning data into
decisions.*\
- 🧩 Focus Areas: Machine Learning, Marketing Analytics, and Predictive
Modeling\
- 🔗 [LinkedIn](https://www.linkedin.com/in/) \|
[GitHub](https://github.com/JulJulien)

------------------------------------------------------------------------

## ⭐ Future Work

-   Apply **SHAP analysis** for model interpretability.\
-   Introduce **budget optimization simulation** via linear
    programming.\
-   Compare against **Random Forest** and **LightGBM** models.\
-   Extend analysis to **multi-year forecasting**.
