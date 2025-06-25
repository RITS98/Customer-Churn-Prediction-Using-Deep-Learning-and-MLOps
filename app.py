import streamlit as st
import numpy as np
import pandas as pd
import pickle
import onnxruntime as ort
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# Load encoders and scaler
with open('label_encoder.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('one_hot_encoder.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Load ONNX model
session = ort.InferenceSession("churn_model_best.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Streamlit UI
st.set_page_config(page_title="Churn Prediction (ONNX)", layout="centered", page_icon="⚡")
st.title("Customer Churn Prediction")

with st.form("input_form"):
    st.subheader("Enter Customer Information")

    col1, col2 = st.columns(2)
    with col1:
        geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
        gender = st.selectbox('👤 Gender', label_encoder_gender.classes_)
        age = st.slider('🎂 Age', 18, 92, 35)
        credit_score = st.number_input('💳 Credit Score', value=600)
        estimated_salary = st.number_input('💰 Estimated Salary', value=50000)

    with col2:
        balance = st.number_input('🏦 Balance', value=0.0)
        tenure = st.slider('📆 Tenure (Years)', 0, 10, 3)
        num_of_products = st.slider('🛍️ Number of Products', 1, 4, 1)
        has_cr_card = st.radio('💳 Has Credit Card', {'Yes': 1, 'No': 0})
        has_cr_card = 1 if has_cr_card == 'Yes' else 0
        is_active_member = st.radio('✅ Is Active Member', {'Yes': 1, 'No': 0})
        is_active_member = 1 if is_active_member == 'Yes' else 0

    submitted = st.form_submit_button("Predict 🚀")

if submitted:
    # Preprocess input
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # One-hot encode Geography
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

    # Combine features
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # Scale
    input_scaled = scaler.transform(input_data)

    # Predict with ONNX
    input_array = input_scaled.astype(np.float32)
    prediction = session.run([output_name], {input_name: input_array})[0]
    prediction_proba = prediction[0][0]

    # Display Results
    st.markdown("---")
    st.subheader("📊 Prediction Result")
    st.progress(int(prediction_proba*100))
    st.metric(label="Churn Probability", value=f"{prediction_proba:.2%}",
              delta="High Risk" if prediction_proba > 0.5 else "Low Risk",
              delta_color="inverse")

    if prediction_proba > 0.5:
        st.error("⚠️ The customer is likely to churn.")
    else:
        st.success("✅ The customer is not likely to churn.")