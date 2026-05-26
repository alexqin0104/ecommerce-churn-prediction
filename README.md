# 🛒 E-commerce Customer Churn Prediction

End-to-end machine learning project predicting customer churn for a UK-based B2B gift retailer, with interactive Streamlit deployment.

**🚀 [Live Demo](https://your-app-url.streamlit.app)** *(deployment in progress)*

---

## 📊 Project Overview

**Dataset:** [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) — 1M+ transactions from 2009-2011

**Business question:** Which customers are likely to stop ordering in the next 6 months?

**Approach:**
- Time-based train/test split (observation: 2009-12 to 2011-05, prediction: 2011-06 to 2011-12)
- Churn definition: customer made no purchases during the 6-month prediction period
- Feature engineering: RFM + behavioral indicators (AvgPrice, OrderInterval)
- Model comparison across 6 versions (Logistic Regression vs XGBoost with various feature sets)
- Model interpretation with SHAP (global and local)
- Streamlit web app for interactive predictions

---

## 🎯 Key Results

| Model | Features | Test ROC-AUC | Notes |
|-------|----------|--------------|-------|
| Logistic Regression v1 | 3 (RFM only) | **0.7985** | Final production model |
| XGBoost v1 | 3 (RFM only) | 0.7935 | Slight overfitting |
| Logistic Regression v2 | 10 (full feature set) | 0.7982 | No gain from extra features |
| XGBoost v2 | 10 (regularized) | 0.7879 | Multicollinearity hurt performance |

**Final model: Logistic Regression** — selected for equivalent performance with superior interpretability and simplicity.

---

## 💡 Business Insights

1. **The "Second Purchase" Effect**: First-time customers churn at **73%** vs **35%** for repeat buyers. Operational priority: convert first-time buyers to repeat buyers.

2. **Counterintuitive AvgPrice Pattern**: High average item price correlates with **higher** churn risk — likely one-time gift buyers without retention incentive.

3. **Recency Dominates**: Recency is the strongest churn predictor (~1.7x more important than Monetary), confirming the importance of recent engagement over absolute spend.

---

## 🛠️ Tech Stack

- **Data Processing**: pandas, numpy
- **Modeling**: scikit-learn (Logistic Regression), XGBoost
- **Interpretability**: SHAP
- **Deployment**: Streamlit
- **Visualization**: matplotlib, seaborn

---

## 📁 Project Structure

   - `app.py` — Streamlit application
   - `01_data_exploration.ipynb` — Full analysis notebook
   - `app_files/` — Saved model artifacts
     - `model.pkl` — Trained Logistic Regression
     - `scaler.pkl` — StandardScaler
     - `explainer.pkl` — SHAP explainer
     - `feature_cols.pkl` — Feature names
     - `train_stats.pkl` — Training statistics
   - `requirements.txt` — Python dependencies
   - `README.md` — This file
---

## 🚀 Run Locally

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/ecommerce-churn.git
cd ecommerce-churn

# Install dependencies
pip install -r requirements.txt

# Launch the Streamlit app
streamlit run app.py
```

---

## 📈 Methodology Highlights

### Time-based Validation (avoiding data leakage)
Used a chronological split rather than random `train_test_split` — predicting future churn requires that the model only sees historical data, mimicking real production scenarios.

### Feature Engineering with Business Context
RFM (Recency, Frequency, Monetary) chosen as the foundation — a 40-year industry-standard customer analytics framework. Behavioral extensions (AvgPrice, OrderInterval) were tested and analyzed for multicollinearity.

### Model Selection by Occam's Razor
Despite XGBoost's general superiority on tabular data, Logistic Regression won here because:
- The 3 RFM features are nearly linearly separable for this problem
- Behavioral extensions showed high multicollinearity (Monetary↔TotalQuantity: r=0.92)
- Simpler models won at equivalent performance — better for production maintenance and stakeholder communication

### SHAP for Model Interpretation
- **Global SHAP**: identified Recency as the strongest churn driver across all customers
- **Local SHAP (waterfall plots)**: explained individual high-risk predictions for business stakeholders

---

## 📧 Contact

Built by [Your Name] — feel free to reach out for questions or collaboration.
