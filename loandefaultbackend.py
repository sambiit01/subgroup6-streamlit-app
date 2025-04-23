import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 1: Simulate Dataset
np.random.seed(42)
n = 1000
income = np.random.normal(60000, 15000, n)
loan_amount = np.random.normal(20000, 5000, n)
credit_score = np.random.normal(650, 50, n)
loan_term = np.random.choice([12, 24, 36, 48, 60], n)
age = np.random.normal(35, 10, n)

default = (
    (loan_amount / income > 0.4) &
    (credit_score < 620) |
    (age < 25)
).astype(int)

df = pd.DataFrame({
    'income': income,
    'loan_amount': loan_amount,
    'credit_score': credit_score,
    'loan_term': loan_term,
    'age': age,
    'default': default
})

# Step 2–4: Preprocess & Train
X = df.drop('default', axis=1)
y = df['default']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

def predict_default(income, loan_amount, credit_score, loan_term, age):
    """Returns prediction and probability for a new loan applicant."""
    input_df = pd.DataFrame([{
        'income': income,
        'loan_amount': loan_amount,
        'credit_score': credit_score,
        'loan_term': loan_term,
        'age': age
    }])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0, 1]
    return prediction, probability
