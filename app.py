import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Z-Score Transformer", layout="centered")
st.title("📈 Z-Score Sample Transformer")

uploaded_file = st.file_uploader("📤 Upload an Excel file (.xlsx)", type=["xlsx"])

if uploaded_file:
    # Read from Excel, starting at row 9 (skip 8 rows), no header
    df_raw = pd.read_excel(uploaded_file, header=None, skiprows=8)

    if len(df_raw.columns) <= 10:
        st.error("❌ Column K not found.")
    else:
        sample_col = df_raw.iloc[:, 10]  # Column K = index 10

        st.write("📄 Raw Column K Data", sample_col)

        # Convert to numeric; retain mask of valid values
        numeric_sample = pd.to_numeric(sample_col, errors='coerce')
        valid_mask = ~numeric_sample.isna()
        sample_numeric = numeric_sample[valid_mask].values

        if len(sample_numeric) == 0:
            st.error("❌ Column K has no numeric values.")
            st.stop()

        # User inputs
        new_mean = st.number_input("🎯 New Mean", min_value=74.0, max_value=76.0, value=75.0)
        target_pct_above_80 = st.slider("🎯 % of values above 80", 0.20, 0.30, 0.25)

        # Z-score transformation
        mean_orig = np.mean(sample_numeric)
        std_orig = np.std(sample_numeric)
        z_scores = (sample_numeric - mean_orig) / std_orig

        z_target = norm.ppf(1 - target_pct_above_80)
        required_std = (80 - new_mean) / z_target
        adjusted_values = np.clip(z_scores * required_std + new_mean, 0, 100)

        # Build final columns (same length as input, preserving non-numeric)
        zscore_full = pd.Series([None] * len(sample_col))
        new_sample_full = pd.Series([None] * len(sample_col))
        zscore_full[valid_mask] = z_scores
        new_sample_full[valid_mask] = adjusted_values
        new_sample_full[~valid_mask] = sample_col[~valid_mask]  # Copy text cells

        # Final output DataFrame
        df_out = pd.DataFrame({
            "Original Sample": sample_col,
            "Z-score": zscore_full,
            "New Sample": new_sample_full
        })

        st.write("✅ Transformed Output")
        st.dataframe(df_out)

        # Excel download
        buffer = io.BytesIO()
        df_out.to_excel(buffer, index=False, engine='openpyxl')
        st.download_button(
            label="📥 Download Transformed Excel File",
            data=buffer.getvalue(),
            file_name="transformed_sample.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # 📊 Histogram and summary
        adjusted_numeric = pd.to_numeric(new_sample_full, errors='coerce').dropna()

        # Histogram bins
        bins = [0, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
        counts, edges = np.histogram(adjusted_numeric, bins=bins)

        # Colors: orange if bin midpoint > 80
        bar_colors = ['#1f77b4' if (edges[i] + edges[i+1]) / 2 <= 80 else '#ff7f0e'
                      for i in range(len(counts))]

        # Plot
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(len(counts)), counts, color=bar_colors, width=1.0,
               edgecolor='black', align='center')

        # Bin labels
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels([f"{int(edges[i])}-{int(edges[i+1])}" for i in range(len(edges)-1)],
                           rotation=45)
        ax.set_xlabel("Adjusted Value Ranges")
        ax.set_ylabel("Frequency")
        ax.set_title("📊 Distribution of Adjusted Values")

        # Summary stats
        pct_above_80 = (adjusted_numeric > 80).mean() * 100
        mean_adj = adjusted_numeric.mean()

        st.write(f"**📈 Mean of Adjusted Values:** {mean_adj:.2f}")
        st.write(f"**🔥 Percentage Above 80:** {pct_above_80:.2f}%")

        # Show plot
        st.pyplot(fig)