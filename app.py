import streamlit as st
import pandas as pd
import joblib

# ======================
# LOAD MODEL
# ======================
model = joblib.load("model_kesesuaian.pkl")

st.title("Prediksi Kesesuaian Bidang Alumni")

st.write("Masukkan data alumni:")

# ======================
# INPUT USER
# ======================
pendapatan = st.selectbox(
    "Pendapatan Bersih",
    ["1-3 juta", "3-5 juta", ">5 juta"]
)

ruang_lingkup = st.selectbox(
    "Ruang Lingkup Kerja",
    ["Lokal/wilayah", "Nasional", "Internasional"]
)

jenis_pekerjaan = st.selectbox(
    "Jenis Pekerjaan",
    ["Swasta", "Instansi Pemerintah", "Freelancer", "Wiraswasta"]
)

lama_tunggu = st.selectbox(
    "Lama Tunggu Kerja",
    ["0–6 bulan", "7–12 bulan", ">12 bulan"]
)

# ======================
# PREPROCESS INPUT
# ======================
def convert_income(x):
    if x == "1-3 juta":
        return 2000000
    if x == "3-5 juta":
        return 4000000
    return 6000000

input_df = pd.DataFrame([{
    "Pendapatan_num": convert_income(pendapatan),
    "ruang_lingkup_kerja": ruang_lingkup,
    "Jenis_pekerjaan": jenis_pekerjaan,
    "Lama_tunggu_kerja": lama_tunggu
}])

# ======================
# PREDIKSI
# ======================
if st.button("Prediksi"):
    pred = model.predict(input_df)[0]

    if str(pred).lower().startswith("sesuai"):
        st.success("✅ Hasil: SESUAI BIDANG")
    else:
        st.error("❌ Hasil: TIDAK SESUAI BIDANG")

