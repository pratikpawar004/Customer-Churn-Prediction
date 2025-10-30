import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download


# Download the model file from your Hugging Face repo
model_path = hf_hub_download(
    repo_id="pratikpawar004/Customer-Churn-Model",
    filename="logistic_regression_model.joblib"
)

# ✅ Load the model from the downloaded path
model = joblib.load(model_path)

st.title("📊 Customer Churn Prediction")
st.write("Enter the following details to predict whether the customer will churn:")

# User Inputs
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
model_cols = model.feature_names_in_
for col in model_cols:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[model_cols]  # Keep correct order

# ✅ Predict Button
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.subheader("📌 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ Customer will leave | Probability: {prob:.2f}")
    else:
        st.success(f"✅ Customer will stay | Probability: {(1-prob):.2f}")
