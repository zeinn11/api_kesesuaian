import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

st.set_page_config(page_title="Tracer Study Alumni", layout="wide")

st.title("📊 Analisis Kesesuaian Bidang Alumni")

# =========================
# LOAD DATA
# =========================
@st.cache_data (ttl=60)
def load_data():
    return pd.read_excel("tracer_studi.xlsx")

df = load_data()

st.subheader("📁 Data Awal")
st.write(df.head())

# =========================
# PREPROCESSING
# =========================
def convert_income(x):
    if pd.isna(x):
        return np.nan
    x = str(x)
    if "1-3" in x:
        return 2000000
    if "3-5" in x:
        return 4000000
    if ">5" in x:
        return 6000000
    return np.nan

df["Pendapatan_num"] = df["Pendapatan_bersih"].apply(convert_income)

df_model = df[[
    "Pendapatan_num",
    "ruang_lingkup_kerja",
    "Jenis_pekerjaan",
    "Lama_tunggu_kerja",
    "Kesesuaian_bidang"
]].dropna()

st.subheader("🧹 Data Siap Pakai")
st.write("Jumlah data:", df_model.shape[0])

X = df_model.drop("Kesesuaian_bidang", axis=1)
y = df_model["Kesesuaian_bidang"]

# =========================
# MODEL
# =========================
num_features = ["Pendapatan_num"]
cat_features = ["ruang_lingkup_kerja", "Jenis_pekerjaan", "Lama_tunggu_kerja"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ]
)

model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("clf", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# =========================
# EVALUASI
# =========================
st.subheader("📈 Evaluasi Model")

st.write("Accuracy:", accuracy_score(y_test, y_pred))
st.write("Confusion Matrix:")
st.write(confusion_matrix(y_test, y_pred))

st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# =========================
# FEATURE IMPORTANCE
# =========================
feature_names = (
    model.named_steps["preprocess"]
    .get_feature_names_out()
)

importances = model.named_steps["clf"].feature_importances_

df_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

st.subheader("⭐ Faktor Paling Mempengaruhi")
st.dataframe(df_importance.head(10))


