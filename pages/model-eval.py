import streamlit as st

# Page config
st.set_page_config(page_title="Model Evaluation", layout="wide")

# Title
st.title("Model Evaluation")

## ---------------- Logistic Regression ---------------- ##
st.header("Logistic Regression")

st.subheader("Model Overview")
st.latex(r"""
P(Y = 1 | X) = \frac{1}{1 + e^{-(\beta_0 + \sum_{i=1}^{n} \beta_i X_i)}}
""")

st.subheader("Features Used")
st.markdown("""
- **Time Remaining (X₁):** Total seconds elapsed since game start.
- **Score Difference (X₂):** Home team score - Away team score.
- **Skater Difference (X₃):** Difference in number of skaters on ice.
- **Goalie Pulled (X₄):** Indicator if the home goalie is pulled.
- **Ice Time (X₅):** Total ice time for the key player involved in the play.
""")

st.subheader("Model Evaluation (Logistic Regression)")
st.latex(r"""
\text{Accuracy} = \frac{TP + TN}{TP + FP + TN + FN} = 0.6747
""")
st.latex(r"""
\text{Precision} = \frac{TP}{TP + FP} = 0.6353
""")
st.latex(r"""
\text{Recall} = \frac{TP}{TP + FN} = 0.8041
""")
st.latex(r"""
\text{ROC-AUC} = \int_{0}^{1} TPR \, d(FPR) = 0.7624
""")

col1, col2 = st.columns(2)
with col1:
    st.image("site-design/lr/logistic_confusion.png", width=500)
with col2:
    st.image("site-design/lr/logistic_roc.png", width=500)

## ---------------- XGBoost ---------------- ##
st.header("XGBoost Model")

st.subheader("Model Overview")
st.markdown("""
XGBoost is a **gradient boosting** algorithm that builds multiple decision trees sequentially, improving performance at each step.
""")

st.subheader("Mathematical Equation")
st.latex(r"""
\hat{y}_i = \sum_{k=1}^{K} f_k(X_i), \quad f_k \in \mathcal{F}
""")
st.latex(r"""
\mathcal{L}(\theta) = \sum_{i=1}^{n} l(y_i, \hat{y}_i) + \sum_{k=1}^{K} \Omega(f_k)
""")

st.subheader("Features Used")
st.markdown("""
- **Time Remaining (X₁):** Total seconds elapsed since game start.
- **Score Difference (X₂):** Home team score - Away team score.
- **Skater Difference (X₃):** Difference in number of skaters on ice.
- **Goalie Pulled (X₄):** Indicator if the home goalie is pulled.
- **Power Play (X₅):** Indicator if either team is on a power play.
- **Goal Difference Last 5 Minutes (X₆):** Average goal difference over the last 5 minutes.
- **Ice Time (X₇):** Total ice time for the key players involved in the play.
""")

st.subheader("Model Evaluation (XGBoost)")
st.latex(r"""
\text{Accuracy} = \frac{TP + TN}{TP + FP + TN + FN} = 0.6822
""")
st.latex(r"""
\text{Precision} = \frac{TP}{TP + FP} = 0.6655
""")
st.latex(r"""
\text{Recall} = \frac{TP}{TP + FN} = 0.7191
""")
st.latex(r"""
\text{ROC-AUC} = \int_{0}^{1} TPR \, d(FPR) = 0.7687
""")

col3, col4 = st.columns(2)
with col3:
    st.image("site-design/xgboost/xgboost_confusion.png", width=500)
with col4:
    st.image("site-design/xgboost/xgboost_roc.png", width=500)

## ---------------- Random Forest ---------------- ##
st.header("Random Forest Model")

st.subheader("Model Overview")
st.markdown("""
Random Forest is an **ensemble learning method** that builds multiple decision trees and averages their predictions to improve accuracy and reduce overfitting.
""")

st.subheader("Mathematical Equation")
st.latex(r"""
\hat{y} = \frac{1}{N} \sum_{i=1}^{N} h_i(X)
""")

st.subheader("Features Used")
st.markdown("""
- **Time Remaining (X₁):** Total seconds elapsed since game start.
- **Score Difference (X₂):** Home team score - Away team score.
- **Skater Difference (X₃):** Difference in number of skaters on ice.
- **Goalie Pulled (X₄):** Indicator if the home goalie is pulled.
- **Power Play (X₅):** Indicator if either team is on a power play.
- **Ice Time (X₆):** Total ice time for the key player involved in the play.
""")

st.subheader("Model Evaluation (Random Forest)")
st.latex(r"""
\text{Accuracy} = \frac{TP + TN}{TP + FP + TN + FN} = 0.7259
""")
st.latex(r"""
\text{ROC-AUC} = \int_{0}^{1} TPR \, d(FPR) = 0.8124
""")

st.markdown("### **Classification Report**")
st.code("""
               precision    recall  f1-score   support

           0       0.71      0.71      0.71     21168
           1       0.74      0.74      0.74     23263

    accuracy                           0.73     44431
   macro avg       0.73      0.73      0.73     44431
weighted avg       0.73      0.73      0.73     44431
""", language="plaintext")

col5, col6 = st.columns(2)
with col5:
    st.image("site-design/rf/rf_confusion.png", width=500)
with col6:
    st.image("site-design/rf/rf_roc.png", width=500)