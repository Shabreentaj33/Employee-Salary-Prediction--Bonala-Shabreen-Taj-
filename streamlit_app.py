
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("best_model.pkl")

st.set_page_config(page_title="Employee Salary Prediction", layout="centered")
st.title("💼 Employee Salary Prediction App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on employee attributes.")

# Sidebar Inputs
st.sidebar.header("Input Features")
age = st.sidebar.slider("Age", 18, 90, 30)
education_num = st.sidebar.slider("Education Level (Num)", 1, 16, 9)
hours_per_week = st.sidebar.slider("Hours per Week", 1, 99, 40)
capital_gain = st.sidebar.number_input("Capital Gain", 0, 100000, 0)
capital_loss = st.sidebar.number_input("Capital Loss", 0, 100000, 0)

# Dummy values for label-encoded fields
workclass = st.sidebar.selectbox("Workclass (Encoded)", [0,1,2,3,4,5,6])
marital_status = st.sidebar.selectbox("Marital Status (Encoded)", [0,1,2])
occupation = st.sidebar.selectbox("Occupation (Encoded)", list(range(15)))
relationship = st.sidebar.selectbox("Relationship (Encoded)", list(range(6)))
race = st.sidebar.selectbox("Race (Encoded)", list(range(5)))
gender = st.sidebar.selectbox("Gender (Encoded)", [0,1])
native_country = st.sidebar.selectbox("Native Country (Encoded)", list(range(5)))
fnlwgt = st.sidebar.number_input("FNLWGT", 10000, 1000000, 226802)

# Input DataFrame
input_df = pd.DataFrame([[
    age, workclass, fnlwgt, "Bachelors", education_num,
    marital_status, occupation, relationship, race, gender,
    capital_gain, capital_loss, hours_per_week, native_country
]], columns=[
    'age', 'workclass', 'fnlwgt', 'education', 'educational-num',
    'marital-status', 'occupation', 'relationship', 'race', 'gender',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'
])

# Drop unused text column if needed
if 'education' in input_df.columns:
    input_df = input_df.drop(columns=['education'])

st.write("### Input Preview")
st.write(input_df)

if st.button("Predict"):
    prediction = model.predict(input_df)
    st.success(f"Predicted Salary Class: {prediction[0]}")
