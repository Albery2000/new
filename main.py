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
        return None

    df[net_bo] = normalize_numeric(df[net_bo])
    df[net_diff] = normalize_numeric(df[net_diff])
    if wc:
        df[wc] = normalize_numeric(df[wc])

    # Stop at TOTAL
    total_row = df[field].astype(str).str.contains("TOTAL", na=False)
    if total_row.any():
        df = df.loc[:total_row.idxmax() - 1]

    df = df[df[well].notna() & (df[well].astype(str).str.strip() != "")]

    return df, {
        "field": field,
        "well": well,
        "net_bo": net_bo,
        "net_diff": net_diff,
        "wc": wc
    }

# =============================================================================
# AI ANOMALY DETECTION
# =============================================================================

def detect_anomalies(df, cols):
    df = df.copy()

    df["z_net_diff"] = (
        (df[cols["net_diff"]] - df[cols["net_diff"]].mean())
        / df[cols["net_diff"]].std()
    )

    q1 = df[cols["net_bo"]].quantile(0.25)
    q3 = df[cols["net_bo"]].quantile(0.75)
    iqr = q3 - q1

    conditions = (
        (df["z_net_diff"].abs() > 3) |
        (df[cols["net_bo"]] < q1 - 1.5 * iqr) |
        (df[cols["net_bo"]] > q3 + 1.5 * iqr) |
        (df[cols["net_bo"]] == 0) |
        ((df[cols["wc"]] > 75) if cols["wc"] else False)
    )

    df["ANOMALY"] = conditions

    df["REASON"] = ""
    df.loc[df["z_net_diff"].abs() > 3, "REASON"] += "Extreme Net Diff BO; "
    df.loc[df[cols["net_bo"]] == 0, "REASON"] += "Zero Net BO; "
    if cols["wc"]:
        df.loc[df[cols["wc"]] > 75, "REASON"] += "High W/C; "

    return df[df["ANOMALY"]]

# =============================================================================
# VISUALS
# =============================================================================

def create_visuals(df, cols):
    fig, axes = plt.subplots(1, 3, figsize=(32, 10))
    fig.set_dpi(300)

    # 1️⃣ Net Diff BO
    top_diff = (
        df.assign(abs_diff=df[cols["net_diff"]].abs())
        .sort_values("abs_diff", ascending=False)
        .head(15)
    )
    colors = [
        "orange" if nb == 0 else ("green" if v > 0 else "red")
        for v, nb in zip(top_diff[cols["net_diff"]], top_diff[cols["net_bo"]])
    ]
    axes[0].bar(top_diff[cols["well"]], top_diff[cols["net_diff"]], color=colors)
    axes[0].axhline(0, color="black")
    axes[0].set_title("Top 15 Net Diff BO")
    axes[0].tick_params(axis="x", rotation=45)

    # 2️⃣ W/C
    if cols["wc"]:
        wc_df = df[df[cols["net_bo"]] > 0].nlargest(10, cols["wc"])
        axes[1].barh(wc_df[cols["well"]], wc_df[cols["wc"]])
        axes[1].set_title("Top 10 W/C Wells")

    # 3️⃣ Net BO
    bo_df = df.nlargest(10, cols["net_bo"])
    axes[2].barh(bo_df[cols["well"]], bo_df[cols["net_bo"]])
    axes[2].set_title("Top 10 Net BO Wells")

    plt.tight_layout()
    return fig

# =============================================================================
# POWERPOINT
# =============================================================================

def create_ppt(df, anomalies, fig):
    prs = Presentation()

    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = "Production Analysis Report"
    slide.placeholders[1].text = "Automated Analytics Dashboard"

    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "AI Anomaly Summary"
    tf = slide.shapes.placeholders[1].text_frame
    tf.clear()
    for _, r in anomalies.iterrows():
        tf.add_paragraph().text = f"{r.iloc[1]} → {r['REASON']}"

    img = io.BytesIO()
    fig.savefig(img, format="png", dpi=300)
    img.seek(0)

    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.add_picture(img, Inches(0.5), Inches(0.5), width=Inches(9))

    ppt = io.BytesIO()
    prs.save(ppt)
    ppt.seek(0)
    return ppt

# =============================================================================
# UI
# =============================================================================

st.title("🛢️ Oil & Gas Analytics Dashboard")

file = st.file_uploader("Upload Production Excel File", type=["xlsx", "xlsm"])

if file:
    df, cols = extract_data(file)
    if df is None:
        st.error("Missing required columns")
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

    ppt = create_ppt(df, anomalies, fig)

    st.download_button(
        "📥 Download PowerPoint Report",
        ppt,
        "production_ai_report.pptx",
        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
    )

else:
    st.info("Upload your Excel production report to begin analysis")
