import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Production Analytics Dashboard", layout="wide")

st.title("📊 Production Report Analytics Dashboard")

# =========================
# File Upload
# =========================
uploaded_file = st.file_uploader(
    "Upload Production Excel File (.xlsx / .xlsm)",
    type=["xlsx", "xlsm"]
)

if uploaded_file:
    # Read sheet names
    xls = pd.ExcelFile(uploaded_file)
    sheet_name = st.selectbox("Select Sheet", xls.sheet_names)

    skip_rows = st.number_input(
        "Skip header rows (for reports with titles & notes)",
        min_value=0,
        max_value=50,
        value=0
    )

    df = pd.read_excel(uploaded_file, sheet_name=sheet_name, skiprows=skip_rows)

    # Clean empty rows & columns
    df = df.dropna(how="all")
    df = df.dropna(axis=1, how="all")

    st.subheader("🔍 Data Preview")
    st.dataframe(df, use_container_width=True)

    # =========================
    # Column Selection
    # =========================
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    st.sidebar.header("⚙️ Controls")

    x_axis = st.sidebar.selectbox("X-Axis", df.columns)
    y_axis = st.sidebar.selectbox("Y-Axis (Numeric)", numeric_cols)

    chart_type = st.sidebar.selectbox(
        "Chart Type",
        ["Line", "Bar", "Pie"]
    )

    # =========================
    # KPIs
    # =========================
    st.subheader("📌 Key Metrics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total", f"{df[y_axis].sum():,.2f}")
    col2.metric("Average", f"{df[y_axis].mean():,.2f}")
    col3.metric("Max", f"{df[y_axis].max():,.2f}")
    col4.metric("Min", f"{df[y_axis].min():,.2f}")

    # =========================
    # Charts
    # =========================
    st.subheader("📈 Visual Analytics")

    if chart_type == "Line":
        fig = px.line(df, x=x_axis, y=y_axis, markers=True)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Bar":
        fig = px.bar(df, x=x_axis, y=y_axis)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Pie":
        fig = px.pie(df, names=x_axis, values=y_axis)
        st.plotly_chart(fig, use_container_width=True)

    # =========================
    # Correlation Heatmap
    # =========================
    if len(numeric_cols) > 1:
        st.subheader("🔗 Correlation Heatmap")

        corr = df[numeric_cols].corr()

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # =========================
    # Insights Section
    # =========================
    st.subheader("🧠 Automated Insights")

    top_row = df.loc[df[y_axis].idxmax()]
    bottom_row = df.loc[df[y_axis].idxmin()]

    st.success(
        f"🔼 Highest **{y_axis}**: {top_row[x_axis]} → {top_row[y_axis]:,.2f}"
    )
    st.error(
        f"🔽 Lowest **{y_axis}**: {bottom_row[x_axis]} → {bottom_row[y_axis]:,.2f}"
    )

    if df[y_axis].iloc[-1] > df[y_axis].iloc[0]:
        st.info("📈 Overall trend shows an **increase**.")
    else:
        st.warning("📉 Overall trend shows a **decrease**.")
