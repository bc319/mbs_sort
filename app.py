import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import io

st.set_page_config(page_title="Z-Score Transformer", layout="centered")
st.title("📈 Z-Score Sample Transformer")

uploaded_file = st.file_uploader("📤 Upload an Excel file (.xlsx)", type=["xlsx"])

if uploaded_file:
    # Read from row 9 (skip 8 rows), no header
    df_raw = pd.read_excel(uploaded_file, header=None, skiprows=8)

    if len(df_raw.columns) <= 10:
        st.error("❌ Column K not found.")
    else:
        sample_col = df_raw.iloc[:, 10]

        # Show raw data
        st.write("📄 Raw Column K Data", sample_col)

        # Convert to numeric where possible
        numeric_sample = pd.to_numeric(sample_col, errors='coerce')
        valid_mask = ~numeric_sample.isna()

        # Extract valid numeric values
        sample_numeric = numeric_sample[valid_mask].values

        if len(sample_numeric) == 0:
            st.error("❌ Column K has no numeric values.")
            st.stop()

        # User inputs
        new_mean = st.number_input("🎯 New Mean", min_value=74.0, max_value=76.0, value=75.0)
        target_pct_above_80 = st.slider("🎯 % of values above 80", 0.20, 0.30, 0.25)

        # Z-scores for numeric values only
        mean_orig = np.mean(sample_numeric)
        std_orig = np.std(sample_numeric)
        z_scores = (sample_numeric - mean_orig) / std_orig

        z_target = norm.ppf(1 - target_pct_above_80)
        required_std = (80 - new_mean) / z_target
        adjusted_values = np.clip(z_scores * required_std + new_mean, 0, 100)

        # Create new columns aligned with original input
        zscore_full = pd.Series([None] * len(sample_col))
        new_sample_full = pd.Series([None] * len(sample_col))

        zscore_full[valid_mask] = z_scores
        new_sample_full[valid_mask] = adjusted_values
        new_sample_full[~valid_mask] = sample_col[~valid_mask]  # Copy non-numeric entries as-is

        # Create final DataFrame
        df_out = pd.DataFrame({
            "Original Sample": sample_col,
            "Z-score": zscore_full,
            "New Sample": new_sample_full
        })

        st.write("✅ Transformed Output")
        st.dataframe(df_out)

        # Allow download
        buffer = io.BytesIO()
        df_out.to_excel(buffer, index=False, engine='openpyxl')
        st.download_button(
            label="📥 Download Excel File",
            data=buffer.getvalue(),
            file_name="transformed_sample.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )