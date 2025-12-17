import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from pptx import Presentation
from pptx.util import Inches

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Oil & Gas Analytics Dashboard",
    page_icon="🛢️",
    layout="wide"
)

# =============================================================================
# HELPERS
# =============================================================================

def find_column(df, level0, level1_contains=""):
    for col in df.columns:
        if (
            str(col[0]).strip().upper() == level0.upper()
            and level1_contains.upper() in str(col[1]).replace("\n", " ").upper()
        ):
            return col
    return None


def normalize_numeric(series):
    return (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .replace("", np.nan)
        .pipe(pd.to_numeric, errors="coerce")
    )

# =============================================================================
# DATA EXTRACTION
# =============================================================================

@st.cache_data(show_spinner=False)
def extract_data(file):
    df = pd.read_excel(
        file,
        sheet_name="Report",
        skiprows=6,
        header=[0, 1]
    )

    field = find_column(df, "Field")
    well = find_column(df, "RUNNING WELLS")
    net_bo = find_column(df, "TOTAL PRODUCTION", "Net BO")
    net_diff = find_column(df, "TOTAL PRODUCTION", "Net diff")
    wc = find_column(df, "W/C", "%")

    if None in [field, well, net_bo, net_diff]:
        return None, None

    df[net_bo] = normalize_numeric(df[net_bo])
    df[net_diff] = normalize_numeric(df[net_diff])
    if wc:
        df[wc] = normalize_numeric(df[wc])

    total_mask = df[field].astype(str).str.contains("TOTAL", na=False)
    if total_mask.any():
        df = df.loc[: total_mask.idxmax() - 1]

    df = df[
        df[well].notna() &
        (df[well].astype(str).str.strip() != "")
    ]

    return df.reset_index(drop=True), {
        "field": field,
        "well": well,
        "net_bo": net_bo,
        "net_diff": net_diff,
        "wc": wc
    }

# =============================================================================
# AI ANOMALY DETECTION (FINAL SAFE VERSION)
# =============================================================================

def detect_anomalies(df, cols):
    df = df.copy()

    # Z-score
    std = df[cols["net_diff"]].std()
    if std == 0 or pd.isna(std):
        df["z_net_diff"] = 0
    else:
        df["z_net_diff"] = (
            (df[cols["net_diff"]] - df[cols["net_diff"]].mean()) / std
        )

    # IQR
    q1 = df[cols["net_bo"]].quantile(0.25)
    q3 = df[cols["net_bo"]].quantile(0.75)
    iqr = q3 - q1

    lower_bo = q1 - 1.5 * iqr
    upper_bo = q3 + 1.5 * iqr

    flag_extreme_diff = df["z_net_diff"].abs() > 3
    flag_zero_bo = df[cols["net_bo"]] == 0
    flag_iqr_bo = (df[cols["net_bo"]] < lower_bo) | (df[cols["net_bo"]] > upper_bo)
    flag_high_wc = (df[cols["wc"]] > 75) if cols["wc"] else pd.Series(False, index=df.index)

    df["ANOMALY"] = (
        flag_extreme_diff |
        flag_zero_bo |
        flag_iqr_bo |
        flag_high_wc
    )

    df["REASON"] = ""

    df.loc[flag_extreme_diff, "REASON"] = (
        df.loc[flag_extreme_diff, "REASON"] +
        "Extreme Net Diff BO (Z-score > 3); "
    )

    df.loc[flag_zero_bo, "REASON"] = (
        df.loc[flag_zero_bo, "REASON"] +
        "Zero Net BO; "
    )

    df.loc[flag_iqr_bo, "REASON"] = (
        df.loc[flag_iqr_bo, "REASON"] +
        "Abnormal Net BO (IQR outlier); "
    )

    if cols["wc"]:
        df.loc[flag_high_wc, "REASON"] = (
            df.loc[flag_high_wc, "REASON"] +
            "High Water Cut (>75%); "
        )

    return df[df["ANOMALY"]].reset_index(drop=True)

# =============================================================================
# VISUALS
# =============================================================================

def create_visuals(df, cols):
    fig, axes = plt.subplots(1, 3, figsize=(32, 10))
    fig.set_dpi(300)

    diff_df = (
        df.assign(abs_diff=df[cols["net_diff"]].abs())
        .sort_values("abs_diff", ascending=False)
        .head(15)
    )

    colors = [
        "orange" if bo == 0 else ("green" if v > 0 else "red")
        for v, bo in zip(diff_df[cols["net_diff"]], diff_df[cols["net_bo"]])
    ]

    axes[0].bar(diff_df[cols["well"]], diff_df[cols["net_diff"]], color=colors)
    axes[0].axhline(0, color="black")
    axes[0].set_title("Top 15 Net Diff BO Wells")
    axes[0].tick_params(axis="x", rotation=45)

    if cols["wc"]:
        wc_df = df[df[cols["net_bo"]] > 0].nlargest(10, cols["wc"])
        axes[1].barh(wc_df[cols["well"]], wc_df[cols["wc"]])
        axes[1].set_title("Top 10 Wells by W/C")

    bo_df = df.nlargest(10, cols["net_bo"])
    axes[2].barh(bo_df[cols["well"]], bo_df[cols["net_bo"]])
    axes[2].set_title("Top 10 Net BO Wells")

    plt.tight_layout()
    return fig

# =============================================================================
# UI
# =============================================================================

st.title("🛢️ Oil & Gas Analytics Dashboard")

file = st.file_uploader("Upload Production Excel File", type=["xlsx", "xlsm"])

if file:
    df, cols = extract_data(file)

    if df is None:
        st.error("Required columns not found.")
        st.stop()

    anomalies = detect_anomalies(df, cols)
    fig = create_visuals(df, cols)

    st.subheader("📊 KPIs")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Wells", len(df))
    c2.metric("Anomalies", len(anomalies))
    c3.metric("Zero Net BO", int((df[cols["net_bo"]] == 0).sum()))
    c4.metric("High W/C", int((df[cols["wc"]] > 75).sum()) if cols["wc"] else 0)

    st.subheader("🚨 AI-Detected Anomalies")
    st.dataframe(anomalies, use_container_width=True)

    st.subheader("📈 Advanced Analytics")
    st.pyplot(fig)
    plt.close(fig)

else:
    st.info("Upload your production Excel file to start analysis")
