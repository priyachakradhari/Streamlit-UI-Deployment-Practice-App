import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="CSV Data Analyzer", layout="wide")
st.title("📊 CSV Data Analyzer App")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:

    # -----------------------
    # Load data
    # -----------------------
    original_df = pd.read_csv(uploaded_file)
    df = original_df.copy()

    st.subheader("🔍 Raw Data Preview")
    st.dataframe(original_df.head())

    # -----------------------
    # Initial dataset info
    # -----------------------
    st.subheader("📌 Initial Dataset Info")

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", original_df.shape[0])
    col2.metric("Columns", original_df.shape[1])
    col3.metric("Missing Values", original_df.isnull().sum().sum())

    # -----------------------
    # Missing value handling
    # -----------------------
    st.subheader("🧹 Handle Missing Values")

    option = st.selectbox(
        "Choose cleaning method",
        ["Do Nothing", "Drop Rows", "Fill with Mean (numeric only)"]
    )

    if option == "Drop Rows":
        df = df.dropna()
        st.success("Rows with missing values removed.")

    elif option == "Fill with Mean (numeric only)":
        numeric_cols = df.select_dtypes(include=np.number).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        st.success("Numeric missing values filled with mean.")

    # -----------------------
    # Final missing values count
    # -----------------------
    st.subheader("📉 Missing Values After Cleaning")

    final_missing = df.isnull().sum().sum()

    if final_missing == 0:
        st.success("✅ No missing values remaining.")
    else:
        st.warning(f"⚠️ Missing values still present: {final_missing}")

    # -----------------------
    # Statistics
    # -----------------------
    st.subheader("📈 Statistical Summary (Cleaned Data)")
    st.dataframe(df.describe())

    # -----------------------
    # Visualization section
    # -----------------------
    st.subheader("📊 Visualization & Comparison")

    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

    if numeric_columns:

        selected_col = st.selectbox(
            "Select numeric column",
            numeric_columns
        )

        view_option = st.radio(
            "Choose data view",
            ["Original Data", "Cleaned Data", "Compare Both"]
        )

        if view_option == "Original Data":
            st.line_chart(original_df[selected_col])

        elif view_option == "Cleaned Data":
            st.line_chart(df[selected_col])

        else:
            compare_df = pd.DataFrame({
                "Original": original_df[selected_col],
                "Cleaned": df[selected_col]
            })
            st.line_chart(compare_df)

    else:
        st.warning("No numeric columns available for visualization.")

    # -----------------------
    # Filter data
    # -----------------------
    st.subheader("🔎 Filter Cleaned Data")

    filter_column = st.selectbox("Select column to filter", df.columns)
    filter_value = st.selectbox(
        "Select value",
        df[filter_column].dropna().unique()
    )

    filtered_df = df[df[filter_column] == filter_value]
    st.dataframe(filtered_df)

    # -----------------------
    # Download
    # -----------------------
    st.subheader("⬇️ Download Cleaned Data")

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download CSV",
        csv,
        "cleaned_data.csv",
        "text/csv"
    )

else:
    st.info("Please upload a CSV file to begin.")
