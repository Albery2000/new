# =========================================================
# Oil & Gas AI Analytics Platform
# FULL AUTO-COLUMN DETECTION VERSION
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from statsmodels.tsa.arima.model import ARIMA
from pptx import Presentation
import io
import re

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Oil & Gas AI Analytics",
    page_icon="🛢️",
    layout="wide"
)

# =========================================================
# ALLOWED SHEETS
# =========================================================
ALLOWED_SHEETS = [
    "Report",
    "Remarks",
    "Well data",
    "Water Flooding",
    "Pico test",
    "Lab BS&W",
    "Test data",
    "Code list",
    "Fluid shots",
    "Tank Status",
    "Trucks & Tanks calculation",
    "NRA & Suez Shipping",
    "Yesterday data",
    "PRODUCTION DPR Report",
    "Station situation"
]

# =========================================================
# ROLES
# =========================================================
ROLES = {
    "Viewer": ["Dashboard"],
    "Engineer": ["Dashboard", "ML & Forecast"],
    "Admin": ["Dashboard", "ML & Forecast", "Admin"]
}

def login():
    st.sidebar.title("🔐 Login")
    user = st.sidebar.text_input("Username")
    role = st.sidebar.selectbox("Role", list(ROLES.keys()))
    return user, role

# =========================================================
# LOAD ONLY ALLOWED SHEETS
# =========================================================
@st.cache_data
def load_allowed_sheets(file):
    xls = pd.ExcelFile(file)
    sheets = {}
    for s in xls.sheet_names:
        if s in ALLOWED_SHEETS:
            sheets[s] = pd.read_excel(xls, sheet_name=s)
    return sheets

# =========================================================
# COLUMN NORMALIZER
# =========================================================
def normalize(col):
    return re.sub(r"[^a-z0-9]", "", str(col).lower())

# =========================================================
# AUTO COLUMN FINDER
# =========================================================
def find_column(columns, keywords):
    for col in columns:
        n = normalize(col)
        for kw in keywords:
            if kw in n:
                return col
    return None

# =========================================================
# HEALTH SCORE
# =========================================================
def health_score(net_bo, net_diff, wc):
    score = 100
    if net_bo <= 0:
        score -= 40
    elif net_bo < 100:
        score -= 20
    if net_diff < 0:
        score -= min(abs(net_diff) / 10, 30)
    if wc is not None:
        if wc > 80:
            score -= 30
        elif wc > 60:
            score -= 15
    return max(0, round(score))

# =========================================================
# FORECAST
# =========================================================
def forecast_series(series):
    try:
        model = ARIMA(series, order=(1,1,1))
        return model.fit().forecast(steps=5)
    except:
        return None

# =========================================================
# CLUSTERING
# =========================================================
def cluster_wells(df):
    features = df[['Net BO', 'Net Diff BO']].fillna(0)
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(features)
    return df

# =========================================================
# POWERPOINT
# =========================================================
def create_ppt(stats, df):
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = "Production Analytics Report"
    slide.placeholders[1].text = "Auto-generated"

    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Executive Summary"
    slide.placeholders[1].text_frame.text = (
        f"Total Wells: {stats['wells']}\n"
        f"Average Health Score: {stats['health']}\n"
        f"Zero Production Wells: {stats['zero']}"
    )

    buf = io.BytesIO()
    prs.save(buf)
    buf.seek(0)
    return buf

# =========================================================
# MAIN
# =========================================================
def main():
    user, role = login()

    st.title("🛢️ Oil & Gas AI Analytics Platform")
    uploaded = st.file_uploader("Upload Excel File", type=["xlsx","xlsm"])

    if not uploaded:
        st.info("Upload a file to start")
        return

    sheets = load_allowed_sheets(uploaded)
    if not sheets:
        st.error("❌ None of the required sheets were found")
        st.stop()

    # -----------------------------------------------------
    # SHEET SELECTION
    # -----------------------------------------------------
    st.sidebar.subheader("📄 Select Sheet")
    sheet_name = st.sidebar.selectbox("Production Sheet", list(sheets.keys()))
    df = sheets[sheet_name]

    # -----------------------------------------------------
    # AUTO COLUMN DETECTION
    # -----------------------------------------------------
    net_bo_col = find_column(df.columns, ["netbo", "netoil", "bo"])
    net_diff_col = find_column(df.columns, ["netdiff", "diffbo"])
    wc_col = find_column(df.columns, ["wc", "watercut", "bsw"])
    well_col = find_column(df.columns, ["well"])
    field_col = find_column(df.columns, ["field"])

    st.sidebar.subheader("🔍 Detected Columns")
    st.sidebar.write(f"Well: {well_col}")
    st.sidebar.write(f"Field: {field_col}")
    st.sidebar.write(f"Net BO: {net_bo_col}")
    st.sidebar.write(f"Net Diff BO: {net_diff_col}")
    st.sidebar.write(f"W/C: {wc_col}")

    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------
    if not net_bo_col or not net_diff_col or not well_col:
        st.error("❌ This sheet does not contain production columns.")
        st.info("💡 Try selecting 'Report', 'Yesterday data', or 'PRODUCTION DPR Report'")
        st.stop()

    # -----------------------------------------------------
    # RENAME
    # -----------------------------------------------------
    df = df.rename(columns={
        net_bo_col: "Net BO",
        net_diff_col: "Net Diff BO",
        well_col: "Well"
    })

    if wc_col:
        df = df.rename(columns={wc_col: "WC"})
    if field_col:
        df = df.rename(columns={field_col: "Field"})

    # -----------------------------------------------------
    # HEALTH SCORE
    # -----------------------------------------------------
    df['Health Score'] = df.apply(
        lambda r: health_score(
            r["Net BO"],
            r["Net Diff BO"],
            r["WC"] if "WC" in df.columns else None
        ),
        axis=1
    )

    # =====================================================
    # DASHBOARD
    # =====================================================
    st.header("📊 Production Dashboard")

    c1,c2,c3 = st.columns(3)
    c1.metric("Total Wells", len(df))
    c2.metric("Avg Health Score", round(df["Health Score"].mean(),1))
    c3.metric("Zero Production Wells", (df["Net BO"] == 0).sum())

    st.dataframe(df.sort_values("Health Score"))

    # =====================================================
    # ML & FORECAST
    # =====================================================
    df = cluster_wells(df)
    st.subheader("🧠 Well Clustering")
    st.dataframe(df[["Well","Cluster","Health Score"]])

    st.subheader("🔮 Forecast")
    well = st.selectbox("Select Well", df["Well"].unique())
    series = df[df["Well"] == well]["Net BO"]
    forecast = forecast_series(series)
    if forecast is not None:
        st.line_chart(pd.concat([series, forecast]))

    # =====================================================
    # PPT
    # =====================================================
    stats = {
        "wells": len(df),
        "health": round(df["Health Score"].mean(),1),
        "zero": (df["Net BO"] == 0).sum()
    }

    ppt = create_ppt(stats, df)
    st.download_button(
        "📥 Download PowerPoint",
        ppt,
        "production_report.pptx",
        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
    )

    st.success("✅ Analysis Completed Successfully")

# =========================================================
if __name__ == "__main__":
    main()
