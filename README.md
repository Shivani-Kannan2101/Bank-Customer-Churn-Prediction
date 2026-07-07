# Bank Customer Churn Prediction & Risk Scoring
A machine learning project that predicts customer churn risk for a European bank, with a live interactive dashboard for risk scoring and scenario simulation.

🔗 **Live App**: [https://your-app-name.streamlit.app  ](https://bank-customer-churn-prediction-9uutkhtwhnpvzxtuz22bqv.streamlit.app/)

## Overview
Banks lose revenue when customers churn — but by the time churn happens, it's too late  to act. This project builds a predictive model that flags at-risk customers *before* they leave, so retention teams can intervene proactively.

## Dataset
European Bank churn dataset 10,000 customers records with features including credit score, geography, age, tenure,account balance, number of products, and activity status.

**Primary Objectives**
• Predict customer churn with high accuracy
• Generate churn probability scores
• Identify key churn drivers

**Secondary Objectives**
• Reduce false positives in churn detection
• Improve interpretability of ML models
• Enable scenario-based churn risk analysis

**Approach**
1. **EDA** — explored churn patterns across demographics, geography, and product usage
2. **Feature Engineering** — created Balance-to-Salary ratio, Engagement×Product interaction, and Age×Tenure interaction features
3. **Modeling** — compared Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting
4. **Model Selection** — chose Random Forest over Gradient Boosting (despite a marginally lower ROC-AUC) because it catches 70% of actual churners vs. only 49% 
   — critical for a retention use case where missing 'at-risk' customers is costly.
5. **Explainability** — SHAP values and feature importance to identify churn drivers
6. **Deployment** — built and deployed a Streamlit dashboard with a churn risk 
   calculator, what-if simulator, feature importance view, and probability distribution

## Model Performance (Random Forest)
| Metric | Score |
|-----------|-------|
| Precision | 0.56  |
| Recall    | 0.70  |
| F1-Score  | 0.62  |
| ROC-AUC   | 0.87  |

## Tech Stack
Python, Pandas, Scikit-learn, SHAP, Streamlit, Matplotlib

## Skills used
Excel — used only for manual EDA (pivot tables, churn rate breakdowns) in the early steps.

Python — used for everything after that:
  Pandas/NumPy — data cleaning, feature engineering
  Scikit-learn — the actual machine learning: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting (all classic supervised ML algorithms)
  XGBoost — one additional ML library, same category of technique (gradient-boosted trees)
  SHAP — explainability technique, not a model itself — it interprets what your trained model is doing
  Matplotlib/Seaborn — charts and visualizations
  Streamlit — turning the trained model into a web app

## Files
- `European_Bank.csv` — dataset
- `Customer_Churn_Analysis.ipynb` — Exploratory Data Analysis,Preprocessing,Feature Engineering,Model Training, Evaluation,SHAP Explainability
- `requirements.txt` — dependencies
- `app.py` — Streamlit dashboard
- `[streamlit_app.py](https://bank-customer-churn-prediction-9uutkhtwhnpvzxtuz22bqv.streamlit.app/)` — interactive churn risk calculator dashboard
- `Research_paper` — EDA insights, model comparison, churn drivers, recommendations

  **Key Findings**
- Customers with 3–4 bank products churn at 80–100%, versus just 8% for customers with 2 products — this suggests over-bundling correlates with dissatisfaction, not loyalty.
- Age is the strongest churn driver — churned customers average 45 years old vs. 37 years old for retained customers.
- German customers churn at nearly 2x the rate of France/Spain.
