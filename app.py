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
    df_raw = pd.read_excel(uploaded_file, header=None, skiprows=8)

    if len(df_raw.columns) <= 10:
        st.error("❌ Column K not found.")
    else:
        sample_col = df_raw.iloc[:, 10]  # Column K = index 10
        st.write("📄 Raw Column K Data", sample_col)

        numeric_sample = pd.to_numeric(sample_col, errors='coerce')
        valid_mask = ~numeric_sample.isna()
        sample_numeric = numeric_sample[valid_mask].values

        if len(sample_numeric) == 0:
            st.error("❌ Column K has no numeric values.")
            st.stop()

        # 🎯 User-defined inputs
        new_mean = st.slider("🎯 New Mean", min_value=74.0, max_value=76.0, value=75.0, step=0.1)
        target_pct_above_80 = st.slider("🎯 % of values above 80", 0.20, 0.30, 0.25)

        # 🔢 Z-score transformation
        mean_orig = np.mean(sample_numeric)
        std_orig = np.std(sample_numeric)
        z_scores = (sample_numeric - mean_orig) / std_orig

        z_target = norm.ppf(1 - target_pct_above_80)
        required_std = (80 - new_mean) / z_target
        adjusted_values = np.clip(z_scores * required_std + new_mean, 0, 100)

        # 🧱 Reconstruct full-length output
        zscore_full = pd.Series([None] * len(sample_col))
        new_sample_full = pd.Series([None] * len(sample_col))
        zscore_full[valid_mask] = z_scores
        new_sample_full[valid_mask] = adjusted_values
        new_sample_full[~valid_mask] = sample_col[~valid_mask]

        # 📄 DataFrame output
        df_out = pd.DataFrame({
            "Original Sample": sample_col,
            "Z-score": zscore_full,
            "New Sample": new_sample_full
        })

        st.write("✅ Transformed Output")
        st.dataframe(df_out)

        # 💾 Download Excel
        buffer = io.BytesIO()
        df_out.to_excel(buffer, index=False, engine='openpyxl')
        st.download_button(
            label="📥 Download Transformed Excel File",
            data=buffer.getvalue(),
            file_name="transformed_sample.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # 📊 Grade Histogram
        adjusted_numeric = pd.to_numeric(new_sample_full, errors='coerce').dropna()
        grade_bins = [0, 50, 65, 70, 75, 80, 100]
        grade_labels = ['F', 'P', 'H3', 'H2B', 'H2A', 'H1']
        colors = ['#d62728', '#9467bd', '#8c564b', '#e377c2', '#1f77b4', '#ff7f0e']  # H1 = orange

        grade_series = pd.cut(adjusted_numeric, bins=grade_bins, labels=grade_labels, right=False)
        grade_counts = grade_series.value_counts().reindex(grade_labels, fill_value=0)
        grade_percents = (grade_counts / len(adjusted_numeric) * 100).round(1)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(grade_labels, grade_counts, color=colors, edgecolor='black')
        ax.set_xlabel("Grade")
        ax.set_ylabel("Number of Students")
        ax.set_title("📊 Distribution of Adjusted Marks")
        st.pyplot(fig)

        # 📈 Stats summary
        mean_adj = adjusted_numeric.mean()
        pct_H1 = grade_percents['H1']

        st.write(f"**📈 Mean of Adjusted Marks:** {mean_adj:.2f}")
        st.write(f"**🔥 Percentage of H1s:** {pct_H1:.1f}%")

        # 📋 Summary table
        summary_df = pd.DataFrame({
            "Grade": grade_labels,
            "Number": grade_counts.values,
            "% of Class": grade_percents.values
        })

        total_results = grade_counts.sum()
        non_numeric = len(sample_col) - total_results
        total_students = len(sample_col)

        st.markdown("### 📋 Summary of Overall Results")
        st.dataframe(summary_df)

        st.markdown(f"""
        **Total Results:** {total_results}  
        **Non-numeric:** {non_numeric}  
        **Students:** {total_students}  
        **Mean:** {mean_adj:.2f}  
        **Standard Deviation:** {adjusted_numeric.std():.2f}  
        """)