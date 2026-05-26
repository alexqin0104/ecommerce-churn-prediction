
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

# ============================================================
# 配置:页面标题和图标
# ============================================================
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🛒",
    layout="wide"
)

# ============================================================
# 加载所有保存的模型文件(用 @st.cache_resource 避免每次重跑)
# ============================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load('app_files/model.pkl')
    scaler = joblib.load('app_files/scaler.pkl')
    explainer = joblib.load('app_files/explainer.pkl')
    feature_cols = joblib.load('app_files/feature_cols.pkl')
    train_stats = joblib.load('app_files/train_stats.pkl')
    return model, scaler, explainer, feature_cols, train_stats

model, scaler, explainer, feature_cols, train_stats = load_artifacts()

# ============================================================
# 页面标题和介绍
# ============================================================
st.title("🛒 E-commerce Customer Churn Predictor")
st.markdown("""
This app predicts the probability that a B2B retail customer will **churn** 
(stop ordering in the next 6 months) based on their purchase behavior.

**Model:** Logistic Regression trained on UCI Online Retail II dataset (1M+ transactions).
""")

st.divider()

# ============================================================
# 输入区:用sidebar(侧边栏)收集用户输入
# ============================================================
st.sidebar.header("📊 Customer Profile Input")
st.sidebar.markdown("Enter the customer's behavioral metrics:")

# Recency
recency = st.sidebar.slider(
    "Recency (days since last purchase)",
    min_value=0, max_value=546, value=60,
    help="How many days ago was their most recent order?"
)

# Frequency  
frequency = st.sidebar.slider(
    "Frequency (number of unique orders)",
    min_value=1, max_value=200, value=3,
    help="How many separate orders has this customer placed?"
)

# Monetary
monetary = st.sidebar.number_input(
    "Monetary (total spend in £)",
    min_value=1.0, max_value=500000.0, value=775.0, step=10.0,
    help="Total amount spent across all orders"
)

# Average Price
avg_price = st.sidebar.number_input(
    "Average Item Price (£)",
    min_value=0.1, max_value=200.0, value=3.0, step=0.1,
    help="Average price per item purchased"
)

# Avg days between orders
avg_days = st.sidebar.slider(
    "Avg days between orders",
    min_value=1, max_value=546, value=30,
    help="Average gap between consecutive orders (high if only one order)"
)

# NumCountries
num_countries = st.sidebar.selectbox(
    "Number of countries served",
    options=[1, 2, 3, 4, 5],
    index=0
)

# 预测按钮
predict_button = st.sidebar.button("🔮 Predict Churn Risk", type="primary", use_container_width=True)

# ============================================================
# 预测区:点击按钮后显示结果
# ============================================================
if predict_button:
    # 构造特征向量(顺序要和训练时一致!)
    raw_features = {
        'Recency': recency,
        'NumCountries': num_countries,
        'log_Frequency': np.log1p(frequency),
        'log_Monetary': np.log1p(monetary),
        'log_AvgPrice': np.log1p(avg_price),
        'log_AvgDaysBetweenOrders': np.log1p(avg_days)
    }
    X_input = pd.DataFrame([raw_features])[feature_cols]

    # 标准化(必须用训练时的scaler)
    X_input_scaled = scaler.transform(X_input)

    # 预测
    churn_proba = model.predict_proba(X_input_scaled)[0, 1]
    churn_pred = "🚨 High Churn Risk" if churn_proba > 0.5 else "✅ Likely to Retain"

    # ===== 左右两列布局 =====
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Prediction Result")
        st.metric(
            label="Churn Probability",
            value=f"{churn_proba:.1%}",
            delta=churn_pred,
            delta_color="inverse" if churn_proba > 0.5 else "normal"
        )

        # 概率条
        st.progress(churn_proba)

        # 解读
        if churn_proba > 0.7:
            st.error("⚠️ Very high churn risk. Recommend immediate retention intervention.")
        elif churn_proba > 0.5:
            st.warning("⚠️ Moderate-to-high churn risk. Consider engagement campaign.")
        elif churn_proba > 0.3:
            st.info("ℹ️ Low-to-moderate churn risk. Monitor activity.")
        else:
            st.success("✅ Low churn risk. Customer appears engaged.")

    with col2:
        st.subheader("Customer Snapshot")
        snapshot_df = pd.DataFrame({
            'Metric': ['Recency', 'Frequency', 'Total Spend', 'Avg Item Price', 'Avg Days Between Orders', 'Countries'],
            'Value': [f"{recency} days", f"{frequency} orders", f"£{monetary:,.0f}", 
                      f"£{avg_price:.2f}", f"{avg_days} days", f"{num_countries}"]
        })
        st.dataframe(snapshot_df, hide_index=True, use_container_width=True)

    st.divider()

    # ===== SHAP 解释 =====
    st.subheader("📊 Why this prediction? (SHAP Explanation)")
    st.markdown("Each feature contributes to pushing the prediction higher (churn) or lower (retain).")

    shap_values = explainer.shap_values(X_input_scaled)

    # 画 waterfall plot
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=X_input_scaled[0],
            feature_names=feature_cols
        ),
        show=False
    )
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.caption("💡 Red bars push the prediction toward churn; blue bars push toward retention.")

else:
    # 没点按钮时显示一些项目说明
    st.info("👈 Enter customer profile in the sidebar and click **Predict** to see the result.")

    with st.expander("📖 About this project"):
        st.markdown("""
        **Dataset:** UCI Online Retail II — 1M+ transactions from a UK-based gift retailer (2009-2011)

        **Methodology:**
        - **Time-based split:** observation period (2009-12 to 2011-05), prediction period (2011-06 to 2011-12)
        - **Churn definition:** customer made no purchases during the 6-month prediction period
        - **Features:** RFM + behavioral indicators (AvgPrice, OrderInterval)
        - **Models compared:** Logistic Regression vs XGBoost (6 versions)
        - **Final model:** Logistic Regression (AUC 0.80) — chosen for simplicity, interpretability, and equivalent performance

        **Key business insights:**
        - **First-purchase customers churn at 73%** vs 35% for repeat buyers — the "second purchase" is the critical milestone
        - **High average item price correlates with higher churn** — likely one-time gift buyers
        - **Recency is the strongest predictor** — more important than total spend
        """)

    with st.expander("🛠️ Tech stack"):
        st.markdown("""
        - **Data processing:** pandas, numpy
        - **Modeling:** scikit-learn, XGBoost
        - **Interpretability:** SHAP
        - **Deployment:** Streamlit
        """)

# ============================================================
# 页脚
# ============================================================
st.divider()
st.caption("Built with Streamlit | Model: Logistic Regression | Dataset: UCI Online Retail II")
