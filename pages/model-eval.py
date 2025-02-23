import streamlit as st

# Page config
st.set_page_config(page_title="Model Evaluation", layout="wide")

# Title
st.title("Model Evaluation")

## ---------------- Logistic Regression ---------------- ##
st.header("Logistic Regression")

st.subheader("Model Overview")
st.markdown("""
Logistic regression is a simple yet effective model for binary classification tasks. It estimates the probability of an event (home team winning) by fitting a logistic curve to the data. This model was chosen because of its interpretability and ability to model the relationship between the features (e.g., score difference, time remaining) and the probability of the home team winning.
""")
st.latex(r"""
P(Y = 1 | X) = \frac{1}{1 + e^{-(\beta_0 + \sum_{i=1}^{n} \beta_i X_i)}}
""")

st.subheader("Features Used")
st.markdown("""
- **Time Remaining (X₁):** Total seconds elapsed since game start, capturing the urgency of the game as it progresses.
- **Score Difference (X₂):** Home team score minus away team score, directly influencing the likelihood of winning.
- **Skater Difference (X₃):** Difference in number of skaters on the ice, which affects the chances of scoring.
- **Goalie Pulled (X₄):** Indicator if the home goalie is pulled, which increases the chance of scoring but also risk.
- **Ice Time (X₅):** Total ice time for the key player, reflecting fatigue and the likelihood of critical plays.
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
XGBoost is a powerful gradient boosting algorithm known for its performance and scalability, particularly with large datasets. It was selected for its ability to model complex relationships in the data, making it ideal for predicting win probabilities in hockey where non-linear interactions between features exist.
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
- **Time Remaining (X₁):** Captures game progression.
- **Score Difference (X₂):** Reflects the lead in the game.
- **Skater Difference (X₃):** Indicates power play or penalty kill scenarios.
- **Goalie Pulled (X₄):** Affects scoring likelihood.
- **Power Play (X₅):** Affects the advantage during the game.
- **Goal Difference Last 5 Minutes (X₆):** Captures late-game momentum.
- **Ice Time (X₇):** Indicates key player fatigue and performance.
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
Random Forest is an ensemble learning method that constructs a multitude of decision trees, making it robust to overfitting and effective for handling high-dimensional data. It was chosen because it combines the advantages of multiple decision trees while reducing the risk of overfitting, making it ideal for a diverse set of features in hockey game data.
""")

st.subheader("Mathematical Equation")
st.latex(r"""
\hat{y} = \frac{1}{N} \sum_{i=1}^{N} h_i(X)
""")

st.subheader("Features Used")
st.markdown("""
- **Time Remaining (X₁):** Game progression impacts strategy.
- **Score Difference (X₂):** Directly influences the likelihood of victory.
- **Skater Difference (X₃):** Reveals situations like power plays or penalties.
- **Goalie Pulled (X₄):** Critical late-game factor.
- **Power Play (X₅):** Evaluates team advantage during power plays.
- **Ice Time (X₆):** Indicates key player fatigue or strength.
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
