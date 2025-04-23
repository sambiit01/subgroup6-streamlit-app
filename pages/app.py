import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.title("📊 Loan Default Risk Prediction")

# Step 1: Generate simulated dataset (same logic as your notebook)
np.random.seed(42)
n = 1000
income = np.random.normal(60000, 15000, n)
loan_amount = np.random.normal(20000, 5000, n)
credit_score = np.random.normal(650, 50, n)
loan_term = np.random.choice([12, 24, 36, 48, 60], n)
age = np.random.normal(35, 10, n)

default = ((loan_amount / income > 0.4) & (credit_score < 620) | (age < 25)).astype(int)

df = pd.DataFrame({
    'income': income,
    'loan_amount': loan_amount,
    'credit_score': credit_score,
    'loan_term': loan_term,
    'age': age,
    'default': default
})

# Step 2: Split
X = df.drop('default', axis=1)
y = df['default']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Step 3: Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Train
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# App UI Inputs
st.subheader("🧾 Enter Applicant Details")
income_input = st.number_input("Income", min_value=0.0, step=1000.0)
loan_amount_input = st.number_input("Loan Amount", min_value=0.0, step=500.0)
credit_score_input = st.number_input("Credit Score (0–1000)", min_value=0.0, max_value=1000.0, step=1.0)
loan_term_input = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60], index=2)
age_input = st.number_input("Age", min_value=0.0, step=1.0)

# Prediction
if st.button("🔮 Predict Default Risk"):
    input_df = pd.DataFrame([{
        'income': income_input,
        'loan_amount': loan_amount_input,
        'credit_score': credit_score_input,
        'loan_term': loan_term_input,
        'age': age_input
    }])

    input_scaled = scaler.transform(input_df)
    pred_class = model.predict(input_scaled)[0]
    pred_prob = model.predict_proba(input_scaled)[0, 1]

    result = "❌ Will Default" if pred_class == 1 else "✅ Will Not Default"
    st.subheader(f"Prediction: {result}")
    st.write(f"Probability of Default: **{pred_prob:.2%}**")
