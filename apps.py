import streamlit as st
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load model
model = joblib.load('best_model.pkl')

# 🎨 Custom page config
st.set_page_config(
    page_title="Student Exam Predictor",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 🌟 Header
st.title("📊 Student Exam Predictor")
st.markdown(
    """
    <style>
    .main {background-color: #f9f9f9;}
    </style>
    """,
    unsafe_allow_html=True
)
st.write("Masukkan data di bawah untuk memprediksi skor ujianmu!")

# 📌 Sidebar untuk input
st.sidebar.header("📝 Input Data")
study_hours = st.sidebar.slider('Study Hours per Day', 0.0, 12.0, 5.0)
attendance = st.sidebar.slider("Attendance Percentage", 0.0, 100.0, 91.0)
mental_health = st.sidebar.slider('Mental Health Rating (1-10)', 1, 10, 8)
sleep_hours = st.sidebar.slider('Sleep Hours per Night', 0.0, 12.0, 7.0)
part_time_job = st.sidebar.radio("Part-Time Job", ['No', 'Yes'])

ptj_encoded = 1 if part_time_job == 'Yes' else 0

# 🔮 Prediction button
if st.button("🚀 Predict Exam Score"):
    input_data = np.array([[study_hours, attendance, mental_health, sleep_hours, ptj_encoded]])
    prediction = model.predict(input_data)[0]
    prediction = max(0, min(100, prediction))

    # 🎯 Output dengan progress bar
    st.subheader("📈 Hasil Prediksi")
    st.progress(int(prediction))
    st.success(f'Predicted Exam Score: **{prediction:.2f}** / 100')

    # 🎨 Extra feedback
    if prediction >= 85:
        st.balloons()
        st.info("🎉 Great job! Kamu punya peluang besar untuk sukses.")
    elif prediction >= 70:
        st.warning("⚡ Skor cukup bagus, tetap konsisten belajar ya!")
    else:
        st.error("📌 Skor masih rendah, coba tingkatkan jam belajar dan kesehatan mentalmu.")
