import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Production Analytics Dashboard",
    layout="wide"
)

st.title("📊 Production Report Analytics Dashboard")

# =========================
# File Upload
# =========================
uploaded_file = st.file_uploader(
    "Upload Production Excel File (.xlsx / .xlsm)",
    type=["xlsx", "xlsm"]
)

if uploaded_file:

    # =========================
    # Sheet & Header Controls
    # =========================
    xls = pd.ExcelFile(uploaded_file)
    sheet_name = st.selectbox("Select Sheet", xls.sheet_names)

    skip_rows = st.number_input(
        "Skip rows before header (titles / notes)",
        min_value=0,
        max_value=50,
        value=0
    )

    header_rows = st.selectbox(
        "Number of header rows",
        options=[1, 2, 3],
        index=1
    )

    # =========================
    # Read Excel
    # =========================
    df = pd.read_excel(
        uploaded_file,
        sheet_name=sheet_name,
        skiprows=skip_rows,
        header=list(range(header_rows))
    )

    # =========================
    # Clean Empty Rows/Columns
    # =========================
    df = df.dropna(how="all").dropna(axis=1, how="all")

    # =========================
    # Flatten Multi-Level Headers
    # =========================
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join(
                [str(i).strip() for i in col if "Unnamed" not in str(i)]
            )
            for col in df.columns.values
        ]
    else:
        df.columns = [str(col).strip() for col in df.columns]

    df.columns = [col for col in df.columns if col != ""]

    # =========================
    # Convert Numeric Columns
    # =========================
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    # =========================
    # Data Preview
    # =========================
    st.subheader("🔍 Cleaned Data Preview")
    st.dataframe(df, use_container_width=True)

    # =========================
    # Column Classification
    # =========================
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    all_cols = df.columns.tolist()

    if not numeric_cols:
        st.error("No numeric columns detected. Check header rows.")
        st.stop()

    # =========================
    # Sidebar Controls
    # =========================
    st.sidebar.header("⚙️ Analytics Controls")

    x_axis = st.sidebar.selectbox("X-Axis", all_cols)
    y_axis = st.sidebar.selectbox("Y-Axis (Numeric)", numeric_cols)

    chart_type = st.sidebar.selectbox(
        "Chart Type",
        ["Line", "Bar", "Pie"]
    )

    # =========================
    # KPIs
    # =========================
    st.subheader("📌 Key Metrics")

    if pd.api.types.is_numeric_dtype(df[y_axis]):
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total", f"{df[y_axis].sum():,.2f}")
        col2.metric("Average", f"{df[y_axis].mean():,.2f}")
        col3.metric("Maximum", f"{df[y_axis].max():,.2f}")
        col4.metric("Minimum", f"{df[y_axis].min():,.2f}")
    else:
        st.warning("Selected Y-Axis column is not numeric")

    # =========================
    # Charts
    # =========================
    st.subheader("📈 Visual Analytics")

    try:
        if chart_type == "Line":
            fig = px.line(df, x=x_axis, y=y_axis, markers=True)
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Bar":
            fig = px.bar(df, x=x_axis, y=y_axis)
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Pie":
            fig = px.pie(df, names=x_axis, values=y_axis)
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Chart error: {e}")

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
    # Automated Insights
    # =========================
    st.subheader("🧠 Automated Insights")

    try:
        max_row = df.loc[df[y_axis].idxmax()]
        min_row = df.loc[df[y_axis].idxmin()]

        st.success(
            f"🔼 Highest **{y_axis}** → {max_row[x_axis]} : {max_row[y_axis]:,.2f}"
        )

        st.error(
            f"🔽 Lowest **{y_axis}** → {min_row[x_axis]} : {min_row[y_axis]:,.2f}"
        )

        if df[y_axis].iloc[-1] > df[y_axis].iloc[0]:
            st.info("📈 Overall trend shows an INCREASE")
        else:
            st.warning("📉 Overall trend shows a DECREASE")

    except Exception as e:
        st.warning(f"Insight generation skipped: {e}")

else:
    st.info("⬆️ Please upload a production Excel file to begin")
