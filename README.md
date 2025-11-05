# 30-Day-Churn-Prediction
A 30-day advance customer churn prediction system using XGBoost with dominant driver identification for high-LTV customers. Achieved Precision @ Top 10% = 0.843 and AUC = 0.746 (time-split validation). Enables targeted retention campaigns and ROI-driven customer insights.

# 30-Day Advance Churn Prediction  
**Predict customer churn 30 days in advance using transaction data**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-orange.svg)](https://xgboost.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Project Overview

This project predicts which **loyal customers** (≥2 purchases in last 90 days) will **not make a purchase in the next 30 days**, enabling **proactive retention campaigns**.

- **Dataset**: [Online Retail II](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II) (UCI ML Repository)  
- **Model**: XGBoost Classifier (hyperparameter-tuned)  
- **Key Metric**: **Precision @ Top 10% Risk = 0.843** (time-split validation)  
- **Validation**: June/July → August 2011 (no data leakage)

> **84.3% of flagged customers actually churn** — only **1 in 6.3 interventions is a false positive.**

---

## Business Impact

| Metric | Value |
|--------|-------|
| **Precision @ Top 10%** | **0.843** |
| **AUC** | **0.746** |
| **True churners saved per 1,000 targeted** | **843** |
| **Net profit (at $200 CLV, $5 cost)** | **$163,600** |
| **ROI** | **32.7x** |

> **$32.70 profit per $1 spent on retention**

---

## Project Structure
.
├── online_retail_II.csv              # Input dataset
├── churn_prediction.ipynb            # Full analysis (Jupyter)
├── 30_day_churn_model_v1.pkl         # Trained model
├── model_features_v1.pkl             # Feature list
├── cleaned_dataframe.csv             # Cleaned data
├── customer_summary.csv              # Aggregated customer stats
├── master_dataset.csv                # Final modeling dataset
└── README.md


---

## How It Works

1. **Data Cleaning**  
   - Remove missing `Customer ID`, negative `Quantity`/`Price`  
   - Create `total_amount = Quantity × Price`

2. **Snapshot-Based Labeling (June, July, August 2011)**  
   - **Observation Window**: 90 days before snapshot  
   - **Prediction Window**: 30 days after snapshot  
   - **Label**: `churn = 1` if no purchase in next 30 days

3. **Feature Engineering**  
   - `avg_basket`, `invoice` count, `spend_drop` (early vs late 45-day split)  
   - `total_spend` (lifetime), `country`, `driver` (heuristic: price, quality, adoption, competition)  
   - `high_ltv` flag (top 2% spenders)

4. **Modeling**  
   - XGBoost with `scale_pos_weight` for class imbalance  
   - Hyperparameter tuning (commented in notebook)  
   - Final model: `n_estimators=200`, `max_depth=5`, etc.

5. **Validation**  
   - **Time-split**: Train on June/July → Test on August  
   - **Result**: Precision @ 10% = **0.843**, AUC = **0.746**

---

## How to Run

```bash
# 1. Clone repo
git clone https://github.com/yourusername/30-day-churn-prediction.git
cd 30-day-churn-prediction

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install pandas numpy scikit-learn xgboost joblib jupyter

# 4. Run notebook
jupyter notebook churn_prediction.ipynb

import joblib
import pandas as pd

model = joblib.load("30_day_churn_model_v1.pkl")
features = joblib.load("model_features_v1.pkl")

# X_new: DataFrame with same features as training
X_new = X_new[features]  # Ensure column order & one-hot encoding
proba = model.predict_proba(X_new)[:, 1]
threshold = np.percentile(proba, 90)
X_new['churn_risk'] = proba
X_new['action_required'] = proba >= threshold

Author
Aniket
LinkedIn | GitHub

License
MIT License – Free to use, modify, and distribute.

Built with 💻 and ☕ in 2025
