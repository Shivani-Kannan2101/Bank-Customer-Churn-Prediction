import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Churn Risk Intelligence", layout="wide")

model = joblib.load('best_model.pkl')
feature_columns = joblib.load('feature_columns.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Bank Customer Churn Risk Calculator")

tab1, tab2, tab3 = st.tabs(["Risk Calculator", "Feature Importance", "Probability Distribution"])

with tab1:
    st.subheader("Enter Customer Details")
    credit_score = st.slider("Credit Score", 300, 900, 650)
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.slider("Age", 18, 100, 40)
    tenure = st.slider("Tenure (years with bank)", 0, 15, 5)
    balance = st.number_input("Account Balance", min_value=0.0, value=75000.0)
    num_products = st.slider("Number of Products", 1, 4, 2)
    has_cr_card = st.checkbox("Has Credit Card", value=True)
    is_active = st.checkbox("Is Active Member", value=True)
    salary = st.number_input("Estimated Salary", min_value=1.0, value=100000.0)

    row = {
        "CreditScore": credit_score, "Age": age, "Tenure": tenure, "Balance": balance,
        "NumOfProducts": num_products, "HasCrCard": int(has_cr_card),
        "IsActiveMember": int(is_active), "EstimatedSalary": salary,
        "Geography_Germany": 1 if geography == "Germany" else 0,
        "Geography_Spain": 1 if geography == "Spain" else 0,
        "Gender_Male": 1 if gender == "Male" else 0,
    }
    row["Balance_to_Salary"] = balance / salary if salary != 0 else 0
    row["Engagement_product"] = int(is_active) * num_products
    row["Age_tenure"] = age * tenure

    X_input = pd.DataFrame([row])[feature_columns]
    proba = model.predict_proba(X_input)[0][1]

    st.subheader("Prediction")
    st.metric("Churn Probability", f"{proba:.1%}")
    if proba >= 0.5:
        st.error("⚠️ HIGH RISK — recommend proactive retention outreach")
    elif proba >= 0.3:
        st.warning("🟡 MODERATE RISK — monitor and consider engagement offer")
    else:
        st.success("✅ LOW RISK — customer likely to stay")

    st.markdown("---")
    st.subheader("What-if Scenario Simulator")
    sim_active = st.checkbox("Simulate: Make customer Active", value=is_active, key="sim_active")
    sim_products = st.slider("Simulate: Number of Products", 1, 4, num_products, key="sim_products")

    row_sim = row.copy()
    row_sim["IsActiveMember"] = int(sim_active)
    row_sim["NumOfProducts"] = sim_products
    row_sim["Engagement_product"] = int(sim_active) * sim_products
    X_sim = pd.DataFrame([row_sim])[feature_columns]
    sim_proba = model.predict_proba(X_sim)[0][1]
    st.metric("Simulated Churn Probability", f"{sim_proba:.1%}", delta=f"{sim_proba - proba:+.1%}")

with tab2:
    st.subheader("Feature Importance Dashboard")
    importances = pd.Series(model.feature_importances_, index=feature_columns).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(importances.index[-10:], importances.values[-10:], color="#6366f1")
    ax.set_title("Top 10 Churn Drivers")
    st.pyplot(fig)

with tab3:
    st.subheader("Probability Distribution Across All Customers")
    df_full = pd.read_csv("European_Bank.csv")
    df_full = df_full.drop(columns=["CustomerId", "Surname", "Year"])
    df_full = pd.get_dummies(df_full, columns=["Geography", "Gender"], drop_first=True)
    df_full["Balance_to_Salary"] = df_full["Balance"] / df_full["EstimatedSalary"]
    df_full["Engagement_product"] = df_full["IsActiveMember"] * df_full["NumOfProducts"]
    df_full["Age_tenure"] = df_full["Age"] * df_full["Tenure"]
    X_full = df_full[feature_columns]
    all_proba = model.predict_proba(X_full)[:, 1]

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.hist(all_proba[df_full["Exited"] == 0], bins=30, alpha=0.6, label="Retained", color="#3b82f6")
    ax2.hist(all_proba[df_full["Exited"] == 1], bins=30, alpha=0.6, label="Churned", color="#ef4444")
    ax2.set_xlabel("Predicted Churn Probability")
    ax2.set_ylabel("Number of Customers")
    ax2.legend()
    st.pyplot(fig2)