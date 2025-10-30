import streamlit as st
import pandas as pd
import joblib
import os
from huggingface_hub import hf_hub_download

# ✅ LOAD MODEL (optimized for Streamlit)
@st.cache_resource
def load_model():
    local_path = "logistic_regression_model.joblib"

    # If model already exists locally → load silently
    if os.path.exists(local_path):
        model = joblib.load(local_path)
        return model

    # If not found, download once from Hugging Face
    with st.spinner("📦 Downloading model from Hugging Face Hub..."):
        model_path = hf_hub_download(
            repo_id="pratikpawar004/Customer-Churn-Model",
            filename="logistic_regression_model.joblib"
        )
        model = joblib.load(model_path)
        # Save a local copy so it won’t redownload next time
        joblib.dump(model, local_path)
    return model


# ---- LOAD THE MODEL ----
try:
    model = load_model()
    model_features = model.feature_names_in_
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# ---- STREAMLIT UI ----
st.title("📊 Customer Churn Prediction")
st.write("Enter the following details to predict whether the customer will churn:")

data = {}
data["gender"] = st.selectbox("Gender", ["Male", "Female"])
data["SeniorCitizen"] = st.selectbox("Senior Citizen", [0, 1])
data["Partner"] = st.selectbox("Partner", ["Yes", "No"])
data["Dependents"] = st.selectbox("Dependents", ["Yes", "No"])
data["tenure"] = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=1)

data["PhoneService"] = st.selectbox("Phone Service", ["Yes", "No"])
data["MultipleLines"] = st.selectbox("Multiple Lines", ["Yes", "No"])
data["InternetService"] = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
data["OnlineSecurity"] = st.selectbox("Online Security", ["Yes", "No"])
data["OnlineBackup"] = st.selectbox("Online Backup", ["Yes", "No"])
data["DeviceProtection"] = st.selectbox("Device Protection", ["Yes", "No"])
data["TechSupport"] = st.selectbox("Tech Support", ["Yes", "No"])
data["StreamingTV"] = st.selectbox("Streaming TV", ["Yes", "No"])
data["StreamingMovies"] = st.selectbox("Streaming Movies", ["Yes", "No"])

data["Contract"] = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
data["PaperlessBilling"] = st.selectbox("Paperless Billing", ["Yes", "No"])
data["PaymentMethod"] = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Credit card (automatic)", "Bank transfer (automatic)"]
)

data["MonthlyCharges"] = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)
data["TotalCharges"] = st.number_input("Total Charges", 0.0, 10000.0, 100.0)

input_df = pd.DataFrame([data])

# ✅ Convert Yes/No to 1/0
binary_cols = ["Partner", "Dependents", "PhoneService", "MultipleLines", "OnlineSecurity",
               "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
               "StreamingMovies", "PaperlessBilling"]
for col in binary_cols:
    input_df[col] = input_df[col].map({"Yes": 1, "No": 0})

# ✅ Convert gender to binary
input_df["gender"] = input_df["gender"].map({"Female": 0, "Male": 1})

# ✅ One-Hot Encoding for multiclass categorical features
cat_cols = ["InternetService", "Contract", "PaymentMethod"]
input_df = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)

# ✅ Align columns with model training
for col in model_features:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[model_features]  # Keep same order

# ---- PREDICT ----
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.subheader("📌 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ Customer will leave | Probability: {prob:.2f}")
    else:
        st.success(f"✅ Customer will stay | Probability: {(1 - prob):.2f}")
