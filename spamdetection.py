import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

# Step 1: Load Dataset
df = pd.read_csv('network_spam_dataset (2).csv')

# Step 2–4: Prepare data
X = df.drop('is_spam', axis=1)
y = df['is_spam']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Step 5: Train Model
model = GaussianNB()
model.fit(X_train_scaled, y_train)

def predict_spam(input_dict):
    """Takes input as a dict and returns class and spam probability."""
    input_df = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    return prediction, probability
