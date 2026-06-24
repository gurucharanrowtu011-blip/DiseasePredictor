import streamlit as st
import numpy as np
import joblib

# =========================
# LOAD MODELS
# =========================
diabetes_model = joblib.load("diabetes_model.pkl")
diabetes_scaler = joblib.load("scaler.pkl")

heart_model = joblib.load("heart_model.pkl")
heart_scaler = joblib.load("heart_scaler.pkl")

liver_model = joblib.load("liver_model.pkl")
liver_scaler = joblib.load("liver_scaler.pkl")

kidney_model = joblib.load("kidney_model.pkl")
kidney_scaler = joblib.load("kidney_scaler.pkl")

# =========================
# TITLE
# =========================
st.title("AI-Powered Multiple Disease Prediction System")

# =========================
# SIDEBAR
# =========================
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

    with st.form("diabetes_form"):
        pregnancies = st.text_input("Pregnancies")
        glucose = st.text_input("Glucose")
        blood_pressure = st.text_input("Blood Pressure")
        skin_thickness = st.text_input("Skin Thickness")
        insulin = st.text_input("Insulin")
        bmi = st.text_input("BMI")
        dpf = st.text_input("Diabetes Pedigree Function")
        age = st.text_input("Age")

        submitted = st.form_submit_button("Predict")

    if submitted:
        try:
            pregnancies = float(pregnancies)
            glucose = float(glucose)
            blood_pressure = float(blood_pressure)
            skin_thickness = float(skin_thickness)
            insulin = float(insulin)
            bmi = float(bmi)
            dpf = float(dpf)
            age = float(age)
        except:
            st.error("Please enter valid numeric values")
            st.stop()

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

    age = st.text_input("Age")
    sex = st.text_input("Sex (0 = Female, 1 = Male)")
    cp = st.text_input("Chest Pain Type")
    trestbps = st.text_input("Resting BP")
    chol = st.text_input("Cholesterol")
    fbs = st.text_input("FBS")
    restecg = st.text_input("Rest ECG")
    thalach = st.text_input("Max Heart Rate")
    exang = st.text_input("Exercise Angina")
    oldpeak = st.text_input("Old Peak")
    slope = st.text_input("Slope")
    ca = st.text_input("Major Vessels")
    thal = st.text_input("Thal")

    if st.button("Predict Heart Disease"):
        try:
            age = float(age)
            sex = float(sex)
            cp = float(cp)
            trestbps = float(trestbps)
            chol = float(chol)
            fbs = float(fbs)
            restecg = float(restecg)
            thalach = float(thalach)
            exang = float(exang)
            oldpeak = float(oldpeak)
            slope = float(slope)
            ca = float(ca)
            thal = float(thal)
        except:
            st.error("Please enter valid numeric values")
            st.stop()

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

    age = st.text_input("Age")
    gender = st.text_input("Gender (0 = Female, 1 = Male)")
    total_bilirubin = st.text_input("Total Bilirubin")
    direct_bilirubin = st.text_input("Direct Bilirubin")
    alkaline_phosphotase = st.text_input("Alkaline Phosphotase")
    alamine_aminotransferase = st.text_input("ALT")
    aspartate_aminotransferase = st.text_input("AST")
    total_proteins = st.text_input("Total Proteins")
    albumin = st.text_input("Albumin")
    agr = st.text_input("A/G Ratio")

    if st.button("Predict Liver Disease"):
        try:
            age = float(age)
            gender = float(gender)
            total_bilirubin = float(total_bilirubin)
            direct_bilirubin = float(direct_bilirubin)
            alkaline_phosphotase = float(alkaline_phosphotase)
            alamine_aminotransferase = float(alamine_aminotransferase)
            aspartate_aminotransferase = float(aspartate_aminotransferase)
            total_proteins = float(total_proteins)
            albumin = float(albumin)
            agr = float(agr)
        except:
            st.error("Please enter valid numeric values")
            st.stop()

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

    Age = st.text_input("Age")
    Blood_Pressure = st.text_input("Blood Pressure")
    Specific_Gravity = st.text_input("Specific Gravity")
    Albumin_Level = st.text_input("Albumin Level")
    Sugar_Level = st.text_input("Sugar Level")

    Red_Blood_Cells = st.text_input("Red Blood Cells")
    Pus_Cells = st.text_input("Pus Cells")
    Pus_Cells_Clumps = st.text_input("Pus Clumps")
    Bacteria = st.text_input("Bacteria")

    Random_Blood_Glucose = st.text_input("Glucose")
    Blood_Urea = st.text_input("Urea")
    Serum_Creatinine = st.text_input("Creatinine")
    Sodium = st.text_input("Sodium")
    Potassium = st.text_input("Potassium")

    Hemoglobin = st.text_input("Hemoglobin")
    Packed_Cell_Volume = st.text_input("PCV")
    WBC = st.text_input("WBC")
    RBC = st.text_input("RBC")

    Hypertension = st.text_input("Hypertension")
    Diabetes_Mellitus = st.text_input("Diabetes Mellitus")
    CAD = st.text_input("Coronary Artery Disease")
    Appetite = st.text_input("Appetite")
    Pedal_Edema = st.text_input("Pedal Edema")
    Anemia = st.text_input("Anemia")

    if st.button("Predict Kidney Disease"):
        try:
            Age = float(Age)
            Blood_Pressure = float(Blood_Pressure)
            Specific_Gravity = float(Specific_Gravity)
            Albumin_Level = float(Albumin_Level)
            Sugar_Level = float(Sugar_Level)
            Red_Blood_Cells = float(Red_Blood_Cells)
            Pus_Cells = float(Pus_Cells)
            Pus_Cells_Clumps = float(Pus_Cells_Clumps)
            Bacteria = float(Bacteria)
            Random_Blood_Glucose = float(Random_Blood_Glucose)
            Blood_Urea = float(Blood_Urea)
            Serum_Creatinine = float(Serum_Creatinine)
            Sodium = float(Sodium)
            Potassium = float(Potassium)
            Hemoglobin = float(Hemoglobin)
            Packed_Cell_Volume = float(Packed_Cell_Volume)
            WBC = float(WBC)
            RBC = float(RBC)
            Hypertension = float(Hypertension)
            Diabetes_Mellitus = float(Diabetes_Mellitus)
            CAD = float(CAD)
            Appetite = float(Appetite)
            Pedal_Edema = float(Pedal_Edema)
            Anemia = float(Anemia)
        except:
            st.error("Please enter valid numeric values")
            st.stop()

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
# ROUTER
# =========================
if disease == "Diabetes":
    diabetes_page()
elif disease == "Heart Disease":
    heart_page()
elif disease == "Liver Disease":
    liver_page()
elif disease == "Kidney Disease":
    kidney_page()
