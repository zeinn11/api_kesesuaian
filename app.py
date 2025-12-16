import streamlit as st
import pandas as pd

@st.cache_data
def load_data():
    return pd.read_excel("tracer_studi.xlsx")

st.title("Analisis Kesesuaian Bidang Alumni")

df = load_data()

st.write("Contoh 5 data:")
st.dataframe(df.head())

