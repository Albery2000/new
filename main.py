import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# App Configuration
# -------------------------------
st.set_page_config(
    page_title="Net BO Production Dashboard",
    layout="wide"
)

st.title("Net BO Production Dashboard")
st.write("Total and Field-wise Net BO Production")

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload Production Excel File (.xlsx or .xlsm)",
    type=["xlsx", "xlsm"]
)

if uploaded_file is not None:

    # -------------------------------
    # Read Excel File
    # -------------------------------
    df = pd.read_excel(uploaded_file)

    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

    # -------------------------------
    # Standardize Column Names
    # -------------------------------
    df.columns = df.columns.str.strip().str.lower()

    # Expected column names after normalization:
    # field
    # net bo production
    # date (optional)

    # -------------------------------
    # Filter Required Fields
    # -------------------------------
    fields_of_interest = [
        "abrar",
        "abrar south",
        "ganna",
        "ferdaus",
        "sidra"
    ]

    df_filtered = df[df["field"].str.lower().isin(fields_of_interest)]

    # -------------------------------
    # Aggregate Net BO by Field
    # -------------------------------
    field_summary = (
        df_filtered
        .groupby("field", as_index=False)["net bo production"]
        .sum()
    )

    field_summary["net bo production"] = field_summary["net bo production"].round(2)

    # -------------------------------
    # Total Net BO Production
    # -------------------------------
    total_net_bo = field_summary["net bo production"].sum()

    # -------------------------------
    # Display KPIs
    # -------------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Total Net BO Production",
            value=f"{total_net_bo:,.2f}"
        )

    with col2:
        st.metric(
            label="Number of Fields",
            value=len(field_summary)
        )

    # -------------------------------
    # Summary Table
    # -------------------------------
    st.subheader("Net BO Production by Field")
    st.dataframe(field_summary)

    # -------------------------------
    # Production Chart
    # -------------------------------
    st.subheader("Net BO Production Chart")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        field_summary["field"],
        field_summary["net bo production"]
    )

    ax.set_xlabel("Field")
    ax.set_ylabel("Net BO Production")
    ax.set_title("Net BO Production by Field")
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    st.pyplot(fig)

else:
    st.info("Please upload a production Excel file to begin.")
