import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import io

st.set_page_config(page_title="Z-Score Transformer", layout="centered")
st.title("📈 Z-Score Sample Transformer")

uploaded_file = st.file_uploader("📤 Upload an Excel file (.xlsx)", type=["xlsx"])

if uploaded_file:
    # Read from Excel, starting at row 9 (skip 8 rows), no header
    df_raw = pd.read_excel(uploaded_file, header=None, skiprows=8)

    # Column K = index 10 (0-based indexing)
    if len(df_raw.columns) <= 10:
        st.error("❌ Column K not found in the uploaded file.")
    else:
        sample_col = df_raw.iloc[:, 10]
        sample = sample_col.dropna().astype(float).values

        st.write("📊 Original Sample", sample)

        # 🎯 New Mean: 74–76
        new_mean = st.number_input("🎯 New Mean", min_value=74.0, max_value=76.0, value=75.0)

        # 🎯 % Above 80: 0.20–0.30
        target_pct_above_80 = st.slider("🎯 % of values above 80", 0.20, 0.30, 0.25)

        # Z-scores
        mean_orig = np.mean(sample)
        std_orig = np.std(sample)
        z_scores = (sample - mean_orig) / std_orig

        # Adjust std to hit % above 80
        z_target = norm.ppf(1 - target_pct_above_80)
        required_std = (80 - new_mean) / z_target
        new_sample = np.clip(z_scores * required_std + new_mean, 0, 100)

        # Output DataFrame
        df_out = pd.DataFrame({
            "Original Sample": sample,
            "Z-score": z_scores,
            "New Sample": new_sample
        })

        st.write("✅ Transformed Data")
        st.dataframe(df_out)

        # Download Excel
        buffer = io.BytesIO()
        df_out.to_excel(buffer, index=False, engine='openpyxl')
        st.download_button(
            label="📥 Download Transformed Excel File",
            data=buffer.getvalue(),
            file_name="transformed_sample.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )