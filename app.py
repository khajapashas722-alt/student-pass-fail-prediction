import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Student Pass Prediction", layout="centered")

st.title("🎓 Student Exam Pass Prediction")
st.write("Predict whether a student will **PASS or FAIL**")

# -----------------------------
# Load Dataset (FIXED)
# -----------------------------
@st.cache_data
def load_data():
    # IMPORTANT FIX: CSV is comma-separated
    return pd.read_csv("student-mat.csv")

df = load_data()

# -----------------------------
# Dataset Preview
# -----------------------------
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# -----------------------------
# Feature Selection
# -----------------------------
features = ["studytime", "absences", "G1", "G2"]

# Safety check
missing_features = [col for col in features if col not in df.columns]
if missing_features:
    st.error(f"❌ Missing columns: {missing_features}")
    st.stop()

X = df[features]

# PASS if final grade G3 >= 10
y = (df["G3"] >= 10).astype(int)

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Scaling
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Logistic Regression Model
# -----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# -----------------------------
# Accuracy
# -----------------------------
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("✅ Model Accuracy")
st.write(f"Accuracy: **{accuracy:.2f}**")

# -----------------------------
# User Input
# -----------------------------
st.subheader("🧑‍🎓 Enter Student Details")

studytime = st.number_input("Study Time (1 = low, 4 = high)", 1, 4, 2)
absences = st.number_input("Absences", 0, 100, 5)
g1 = st.number_input("G1 Score", 0, 20, 10)
g2 = st.number_input("G2 Score", 0, 20, 10)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔮 Predict"):
    user_data = np.array([[studytime, absences, g1, g2]])
    user_scaled = scaler.transform(user_data)

    prediction = model.predict(user_scaled)[0]
    probability = model.predict_proba(user_scaled)[0][1]

    if prediction == 1:
        st.success(f"🎉 PASS (Probability: {probability:.2f})")
    else:
        st.error(f"❌ FAIL (Probability: {probability:.2f})")

st.caption("Mini Project | Logistic Regression | Student Performance Dataset")
