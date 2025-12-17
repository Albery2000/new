import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from pptx import Presentation
from pptx.util import Inches
from openpyxl import load_workbook

# =============================================================================
# CONFIG
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
    """
    Robust finder for MultiIndex Excel columns
    """
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
# MAIN EXTRACTION LOGIC
# =============================================================================

@st.cache_data(show_spinner=False)
def extract_production_data(uploaded_file):
    df = pd.read_excel(
        uploaded_file,
        sheet_name="Report",
        skiprows=6,
        header=[0, 1]
    )

    # Detect columns
    field_col = find_column(df, "Field")
    well_col = find_column(df, "RUNNING WELLS")
    net_bo_col = find_column(df, "TOTAL PRODUCTION", "Net BO")
    net_diff_bo_col = find_column(df, "TOTAL PRODUCTION", "Net diff")
    wc_col = find_column(df, "W/C", "%")

    required = {
        "Field": field_col,
        "Well Name": well_col,
        "Net BO": net_bo_col,
        "Net Diff BO": net_diff_bo_col
    }

    missing = [k for k, v in required.items() if v is None]
    if missing:
        st.error(f"❌ Missing required columns: {', '.join(missing)}")
        return None

    # Normalize numeric columns
    df[net_bo_col] = normalize_numeric(df[net_bo_col])
    df[net_diff_bo_col] = normalize_numeric(df[net_diff_bo_col])
    if wc_col:
        df[wc_col] = normalize_numeric(df[wc_col])

    # Stop at TOTAL row
    field_series = df[field_col].astype(str).str.upper()
    total_rows = field_series[field_series.str.contains("TOTAL", na=False)]

    if not total_rows.empty:
        stop_idx = total_rows.index[0]
        df = df.loc[:stop_idx - 1]

    # Clean wells
    df = df[
        df[well_col].notna() &
        (df[well_col].astype(str).str.strip() != "")
    ]

    # Zero Net BO
    zero_net_bo_df = df[df[net_bo_col] == 0]

    # Non-zero Net Diff BO
    df_non_zero = df[df[net_diff_bo_col] != 0]

    # Stats
    stats = {
        "Total Wells": len(df),
        "Non-Zero Net Diff BO Wells": len(df_non_zero),
        "Positive Net Diff BO": int((df_non_zero[net_diff_bo_col] > 0).sum()),
        "Negative Net Diff BO": int((df_non_zero[net_diff_bo_col] < 0).sum()),
        "Zero Net BO Wells": int((df[net_bo_col] == 0).sum()),
        "Total Net BO": df[net_bo_col].sum(),
        "Total Net Diff BO": df[net_diff_bo_col].sum()
    }

    return {
        "df_all": df,
        "df_non_zero": df_non_zero,
        "df_zero": zero_net_bo_df,
        "columns": {
            "field": field_col,
            "well": well_col,
            "net_bo": net_bo_col,
            "net_diff_bo": net_diff_bo_col,
            "wc": wc_col
        },
        "stats": stats
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_visuals(df, cols):
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.set_dpi(300)

    top = (
        df
        .assign(abs_diff=df[cols["net_diff_bo"]].abs())
        .sort_values("abs_diff", ascending=False)
        .head(15)
    )

    colors = ["green" if v > 0 else "red" for v in top[cols["net_diff_bo"]]]

    ax.bar(
        top[cols["well"]],
        top[cols["net_diff_bo"]],
        color=colors
    )

    ax.axhline(0, color="black", linewidth=2)
    ax.set_title("Top 15 Wells by Net Diff BO", fontsize=16, fontweight="bold")
    ax.set_ylabel("Net Diff BO")
    ax.set_xticklabels(top[cols["well"]], rotation=45, ha="right")

    plt.tight_layout()
    return fig


# =============================================================================
# POWERPOINT EXPORT
# =============================================================================

def create_ppt(result, fig):
    prs = Presentation()

    # Title slide
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = "Production Analysis Report"
    slide.placeholders[1].text = "Generated Automatically"

    # Stats slide
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Executive Summary"
    tf = slide.shapes.placeholders[1].text_frame
    tf.clear()

    for k, v in result["stats"].items():
        tf.add_paragraph().text = f"{k}: {v}"

    # Visualization slide
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

uploaded_file = st.file_uploader(
    "Upload Production Excel File",
    type=["xlsx", "xlsm"]
)

if uploaded_file:
    with st.spinner("Processing file..."):
        result = extract_production_data(uploaded_file)

    if result:
        st.success("✅ File processed successfully")

        st.subheader("📊 Key Metrics")
        cols = st.columns(5)
        for col, (k, v) in zip(cols, result["stats"].items()):
            col.metric(k, v)

        st.subheader("📋 Production Data")
        st.dataframe(result["df_all"], use_container_width=True)

        st.subheader("📈 Visual Analysis")
        fig = create_visuals(result["df_non_zero"], result["columns"])
        st.pyplot(fig)
        plt.close(fig)

        ppt = create_ppt(result, fig)

        st.download_button(
            "📥 Download PowerPoint",
            ppt,
            "production_analysis.pptx",
            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
        )

else:
    st.info("👆 Upload your production Excel file to start analysis")
