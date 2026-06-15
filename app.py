import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("model.pkl")

st.set_page_config(
    page_title="Phone Addiction Predictor",
    page_icon="📱",
    layout="centered"
)

st.title("📱 Phone Addiction Level Predictor")
st.write("Fill all details below:")

# ---- INPUTS ----
age = st.number_input("Age", min_value=10, max_value=80, value=25)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

daily_usage = st.number_input(
    "Daily Usage Hours",
    min_value=0.0,
    max_value=24.0,
    value=5.0
)

sleep_hours = st.number_input(
    "Sleep Hours",
    min_value=0.0,
    max_value=12.0,
    value=7.0
)

# Keep spelling exactly as training data
interllectual = st.number_input(
    "Interllectual Performance",
    min_value=0.0,
    max_value=10.0,
    value=5.0
)

social = st.number_input(
    "Social Interactions",
    min_value=0.0,
    max_value=10.0,
    value=5.0
)

exercise = st.number_input(
    "Exercise Hours",
    min_value=0.0,
    max_value=10.0,
    value=1.0
)

anxiety = st.number_input(
    "Anxiety Level",
    min_value=0.0,
    max_value=10.0,
    value=5.0
)

depression = st.number_input(
    "Depression Level",
    min_value=0.0,
    max_value=10.0,
    value=5.0
)

self_esteem = st.number_input(
    "Self Esteem",
    min_value=0.0,
    max_value=10.0,
    value=5.0
)

screen_bed = st.number_input(
    "Screen Time Before Bed",
    min_value=0.0,
    max_value=10.0,
    value=2.0
)

phone_checks = st.number_input(
    "Phone Checks Per Day",
    min_value=0,
    max_value=500,
    value=50
)

apps_used = st.number_input(
    "Apps Used Daily",
    min_value=0,
    max_value=50,
    value=10
)

social_media = st.number_input(
    "Time on Social Media",
    min_value=0.0,
    max_value=12.0,
    value=2.0
)

gaming = st.number_input(
    "Time on Gaming",
    min_value=0.0,
    max_value=12.0,
    value=1.0
)

education = st.number_input(
    "Time on Education",
    min_value=0.0,
    max_value=12.0,
    value=2.0
)



family = st.number_input(
    "Family Communication",
    min_value=0.0,
    max_value=10.0,
    value=5.0
)

weekend = st.number_input(
    "Weekend Usage Hours",
    min_value=0.0,
    max_value=24.0,
    value=6.0
)

# ---- CREATE INPUT DATA ----
purpose = st.selectbox(
    "Phone Usage Purpose",
    [
        "Social Media",
        "Gaming",
        "Education",
        "Other"
    ]
)

input_data = pd.DataFrame([{
    "Age": age,
    "Daily_Usage_Hours": daily_usage,
    "Sleep_Hours": sleep_hours,
    "Interllectual_Performance": interllectual,
    "Social_Interactions": social,
    "Exercise_Hours": exercise,
    "Anxiety_Level": anxiety,
    "Depression_Level": depression,
    "Self_Esteem": self_esteem,
    "Screen_Time_Before_Bed": screen_bed,
    "Phone_Checks_Per_Day": phone_checks,
    "Apps_Used_Daily": apps_used,
    "Time_on_Social_Media": social_media,
    "Time_on_Gaming": gaming,
    "Time_on_Education": education,
    "Family_Communication": family,
    "Weekend_Usage_Hours": weekend,

    "Gender_Male": 1 if gender == "Male" else 0,
    "Gender_Other": 0,

    "Phone_Usage_Purpose_Education": 1 if purpose == "Education" else 0,
    "Phone_Usage_Purpose_Gaming": 1 if purpose == "Gaming" else 0,
    "Phone_Usage_Purpose_Other": 1 if purpose == "Other" else 0,
    "Phone_Usage_Purpose_Social Media": 1 if purpose == "Social Media" else 0
}])

# ---- PREDICTION ----
if st.button("Predict Addiction Level"):

    try:
        prediction = model.predict(input_data)[0]

        st.success(
            f"🎯 Predicted Addiction Level: {prediction}"
        )

        if prediction <= 2:
            st.info("🟢 Low Addiction")
        elif prediction <= 4:
            st.warning("🟡 Moderate Addiction")
        else:
            st.error("🔴 High Addiction")

    except Exception as e:
        st.error(f"Error: {e}")