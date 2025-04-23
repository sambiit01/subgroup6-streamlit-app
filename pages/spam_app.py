import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

st.title("🛡️ Network Spam Detection App")

# Load data
df = pd.read_csv("network_spam_dataset (2).csv")

X = df.drop('is_spam', axis=1)
y = df['is_spam']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = GaussianNB()
model.fit(X_train_scaled, y_train)

# --- Streamlit Inputs ---
st.subheader("📥 Enter Network Packet Details")

packet_size = st.number_input("Packet Size", value=100)
packet_frequency = st.number_input("Packet Frequency", value=444.06)
source_ip_entropy = st.number_input("Source IP Entropy", value=0.53)
destination_port = st.number_input("Destination Port", value=7022)
protocol_type_encoded = st.number_input("Protocol Type (Encoded)", value=2)
payload_length = st.number_input("Payload Length", value=5125)
ttl = st.number_input("Time To Live (TTL)", value=128)
http_request_ratio = st.number_input("HTTP Request Ratio", value=0.75)
dns_query_ratio = st.number_input("DNS Query Ratio", value=0.17)
packet_interval_mean = st.number_input("Packet Interval Mean", value=0.09)
packet_interval_std = st.number_input("Packet Interval Std", value=0.098)
unique_destinations = st.number_input("Unique Destinations", value=923)
spam_keywords_count = st.number_input("Spam Keywords Count", value=6)

if st.button("🔍 Predict Spam"):
    # Create DataFrame from inputs
    test_case = pd.DataFrame([{
        'packet_size': packet_size,
        'packet_frequency': packet_frequency,
        'source_ip_entropy': source_ip_entropy,
        'destination_port': destination_port,
        'protocol_type_encoded': protocol_type_encoded,
        'payload_length': payload_length,
        'ttl': ttl,
        'http_request_ratio': http_request_ratio,
        'dns_query_ratio': dns_query_ratio,
        'packet_interval_mean': packet_interval_mean,
        'packet_interval_std': packet_interval_std,
        'unique_destinations': unique_destinations,
        'spam_keywords_count': spam_keywords_count
    }])

    # Preprocess & Predict
    scaled_input = scaler.transform(test_case)
    pred = model.predict(scaled_input)[0]
    proba = model.predict_proba(scaled_input)[0][1]

    # Output
    result = "🚨 SPAM" if pred == 1 else "✅ NOT SPAM"
    st.subheader(f"Prediction: {result}")
    st.write(f"Probability of Spam: **{proba:.2%}**")
