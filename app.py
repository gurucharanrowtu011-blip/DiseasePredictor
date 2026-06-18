import streamlit as st
import numpy as np
import joblib

# Models
diabetes_model = joblib.load("diabetes_model.pkl")
diabetes_scaler = joblib.load("scaler.pkl")

heart_model = joblib.load("heart_model.pkl")
heart_scaler = joblib.load("heart_scaler.pkl")

liver_model = joblib.load("liver_model.pkl")
liver_scaler = joblib.load("liver_scaler.pkl")

kidney_model = joblib.load("kidney_model.pkl")
kidney_scaler = joblib.load("kidney_scaler.pkl")

st.title("AI-Powered Multiple Disease Prediction System")

# Sidebar
st.sidebar.title("Navigation")
disease = st.sidebar.radio(
    "Select Disease",
    ["Diabetes", "Heart Disease", "Liver Disease", "Kidney Disease"]
)

# =========================
# DIABETES
# =========================
def diabetes_page():
    st.header("Diabetes Prediction")

    pregnancies = st.number_input("Pregnancies")
    glucose = st.number_input("Glucose")
    blood_pressure = st.number_input("Blood Pressure")
    skin_thickness = st.number_input("Skin Thickness")
    insulin = st.number_input("Insulin")
    bmi = st.number_input("BMI")
    dpf = st.number_input("Diabetes Pedigree Function")
    age = st.number_input("Age")

    if st.button("Predict Diabetes"):
        data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                          insulin, bmi, dpf, age]])

        data = diabetes_scaler.transform(data)
        prediction = diabetes_model.predict(data)

        if prediction[0] == 1:
            st.error("Diabetes Likely")
        else:
            st.success("No Diabetes")


# =========================
# HEART
# =========================
def heart_page():
    st.header("Heart Disease Prediction")

    age = st.number_input("Age")
    sex = st.number_input("Sex (0 = Female, 1 = Male)")
    cp = st.number_input("Chest Pain Type")
    trestbps = st.number_input("Resting BP")
    chol = st.number_input("Cholesterol")
    fbs = st.number_input("FBS")
    restecg = st.number_input("Rest ECG")
    thalach = st.number_input("Max Heart Rate")
    exang = st.number_input("Exercise Angina")
    oldpeak = st.number_input("Old Peak")
    slope = st.number_input("Slope")
    ca = st.number_input("Major Vessels")
    thal = st.number_input("Thal")

    if st.button("Predict Heart Disease"):
        data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                          thalach, exang, oldpeak, slope, ca, thal]])

        data = heart_scaler.transform(data)
        prediction = heart_model.predict(data)

        if prediction[0] == 1:
            st.error("Heart Disease Likely")
        else:
            st.success("No Heart Disease")


# =========================
# LIVER
# =========================
def liver_page():
    st.header("Liver Disease Prediction")

    age = st.number_input("Age")
    gender = st.number_input("Gender (0 = Female, 1 = Male)")
    total_bilirubin = st.number_input("Total Bilirubin")
    direct_bilirubin = st.number_input("Direct Bilirubin")
    alkaline_phosphotase = st.number_input("Alkaline Phosphotase")
    alamine_aminotransferase = st.number_input("ALT")
    aspartate_aminotransferase = st.number_input("AST")
    total_proteins = st.number_input("Total Proteins")
    albumin = st.number_input("Albumin")
    agr = st.number_input("A/G Ratio")

    if st.button("Predict Liver Disease"):
        data = np.array([[age, gender, total_bilirubin, direct_bilirubin,
                          alkaline_phosphotase, alamine_aminotransferase,
                          aspartate_aminotransferase, total_proteins,
                          albumin, agr]])

        data = liver_scaler.transform(data)
        prediction = liver_model.predict(data)

        if prediction[0] == 1:
            st.error("Liver Disease Likely")
        else:
            st.success("No Liver Disease")


# =========================
# KIDNEY
# =========================
def kidney_page():
    st.header("Kidney Disease Prediction")

    Age = st.number_input("Age")
    Blood_Pressure = st.number_input("Blood Pressure")
    Specific_Gravity = st.number_input("Specific Gravity")
    Albumin_Level = st.number_input("Albumin Level")
    Sugar_Level = st.number_input("Sugar Level")

    Red_Blood_Cells = st.number_input("Red Blood Cells")
    Pus_Cells = st.number_input("Pus Cells")
    Pus_Cells_Clumps = st.number_input("Pus Clumps")
    Bacteria = st.number_input("Bacteria")

    Random_Blood_Glucose = st.number_input("Glucose")
    Blood_Urea = st.number_input("Urea")
    Serum_Creatinine = st.number_input("Creatinine")
    Sodium = st.number_input("Sodium")
    Potassium = st.number_input("Potassium")

    Hemoglobin = st.number_input("Hemoglobin")
    Packed_Cell_Volume = st.number_input("PCV")
    WBC = st.number_input("WBC")
    RBC = st.number_input("RBC")

    Hypertension = st.number_input("Hypertension")
    Diabetes_Mellitus = st.number_input("Diabetes Mellitus")
    CAD = st.number_input("Coronary Artery Disease")
    Appetite = st.number_input("Appetite")
    Pedal_Edema = st.number_input("Pedal Edema")
    Anemia = st.number_input("Anemia")

    if st.button("Predict Kidney Disease"):
        data = np.array([[Age, Blood_Pressure, Specific_Gravity, Albumin_Level,
                          Sugar_Level, Red_Blood_Cells, Pus_Cells,
                          Pus_Cells_Clumps, Bacteria, Random_Blood_Glucose,
                          Blood_Urea, Serum_Creatinine, Sodium, Potassium,
                          Hemoglobin, Packed_Cell_Volume, WBC, RBC,
                          Hypertension, Diabetes_Mellitus, CAD,
                          Appetite, Pedal_Edema, Anemia]])

        data = kidney_scaler.transform(data)
        prediction = kidney_model.predict(data)

        if prediction[0] == 1:
            st.error("Kidney Disease Likely")
        else:
            st.success("No Kidney Disease")


# =========================
# ROUTER (IMPORTANT FIX)
# =========================
if disease == "Diabetes":
    diabetes_page()
elif disease == "Heart Disease":
    heart_page()
elif disease == "Liver Disease":
    liver_page()
elif disease == "Kidney Disease":
    kidney_page()