import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import io

st.set_page_config(page_title="Z-Score Transformer", layout="centered")
st.title("📈 Z-Score Sample Transformer")

uploaded_file = st.file_uploader("📤 Upload an Excel file (.xlsx)", type=["xlsx"])

if uploaded_file:
    # Load Excel and get first sheet
    sheet = pd.ExcelFile(uploaded_file).sheet_names[0]
    df = pd.read_excel(uploaded_file, sheet_name=sheet)

    # Choose column
    col = st.selectbox("🔍 Select column with original sample values", df.columns)
    sample = df[col].dropna().astype(float).values

    st.write("📊 Original Sample", sample)

    # New mean between 74–76
    new_mean = st.number_input("🎯 New Mean", min_value=74.0, max_value=76.0, value=75.0)

    # Target % above 80 between 20% and 30%
    target_pct_above_80 = st.slider("🎯 % of values above 80", 0.20, 0.30, 0.25)

    # Compute z-scores
    mean_orig = np.mean(sample)
    std_orig = np.std(sample)
    z_scores = (sample - mean_orig) / std_orig

    # Calculate required std to meet target % > 80
    z_target = norm.ppf(1 - target_pct_above_80)
    required_std = (80 - new_mean) / z_target

    # Construct new sample
    new_sample = np.clip(z_scores * required_std + new_mean, 0, 100)

    # Add to DataFrame
    df['Z-score'] = pd.Series(z_scores, index=df.index[:len(z_scores)])
    df['New Sample'] = pd.Series(new_sample, index=df.index[:len(new_sample)])

    # Show results
    st.write("✅ Transformed Data")
    st.dataframe(df)

    # Allow download
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False, engine='openpyxl')
    st.download_button(
        label="📥 Download Transformed Excel File",
        data=buffer.getvalue(),
        file_name="transformed_sample.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )