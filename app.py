import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ======================
# LOAD DATA
# ======================
@st.cache_data
def load_data():
    return pd.read_excel("tracer_studi.xlsx")

df = load_data()

# ======================
# PREPROCESS
# ======================
def convert_income(x):
    if pd.isna(x): return np.nan
    x = str(x)
    if "1-3" in x: return 2000000
    if "3-5" in x: return 4000000
    if ">5" in x: return 6000000
    return np.nan

df["Pendapatan_num"] = df["Pendapatan_bersih"].apply(convert_income)

df_model = df[[
    "Pendapatan_num",
    "ruang_lingkup_kerja",
    "Jenis_pekerjaan",
    "Lama_tunggu_kerja",
    "Kesesuaian_bidang"
]].dropna()

X = df_model.drop("Kesesuaian_bidang", axis=1)
y = df_model["Kesesuaian_bidang"]

# ======================
# MODEL
# ======================
num_features = ["Pendapatan_num"]
cat_features = ["ruang_lingkup_kerja", "Jenis_pekerjaan", "Lama_tunggu_kerja"]

preprocess = ColumnTransformer([
    ("num", "passthrough", num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
])

model = ImbPipeline([
    ("prep", preprocess),
    ("smote", SMOTE(random_state=42)),
    ("rf", RandomForestClassifier(n_estimators=200, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

# ======================
# STREAMLIT UI
# ======================
st.title("Prediksi Kesesuaian Bidang Alumni")

pendapatan = st.selectbox("Pendapatan", ["1-3 juta", "3-5 juta", ">5 juta"])
ruang = st.selectbox("Ruang Lingkup Kerja", X["ruang_lingkup_kerja"].unique())
jenis = st.selectbox("Jenis Pekerjaan", X["Jenis_pekerjaan"].unique())
lama = st.selectbox("Lama Tunggu Kerja", X["Lama_tunggu_kerja"].unique())

if st.button("Prediksi"):
    input_df = pd.DataFrame([{
        "Pendapatan_num": convert_income(pendapatan),
        "ruang_lingkup_kerja": ruang,
        "Jenis_pekerjaan": jenis,
        "Lama_tunggu_kerja": lama
    }])

    hasil = model.predict(input_df)[0]
    st.success(f"Hasil Prediksi: **{hasil}**")

