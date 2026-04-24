from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
import streamlit as st

from dml import run_dml_analysis
from preprocess import load_financial_data, prepare_dataset
from train import run_training_pipeline


st.set_page_config(page_title="Financial DML System", layout="wide")
st.title("Financial Decision-Making and Resource Allocation with DML")
st.caption("Forecasting, causal inference, and business recommendations in one workflow.")


# =========================
# 📂 INPUT
# =========================
uploaded_file = st.file_uploader("Upload financial dataset (CSV)", type=["csv"])
default_path = st.text_input(
    "Or provide a local CSV path",
    value=r"c:\Users\abhis\josh_su\Financial_companies_stratagies.csv",
)

df: pd.DataFrame | None = None
active_path: str | None = None

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, low_memory=False)
elif default_path:
    try:
        df = load_financial_data(default_path)
        active_path = default_path
    except Exception as exc:
        st.warning(f"Unable to load default dataset path yet: {exc}")


# =========================
# 🚀 MAIN UI
# =========================
if df is not None:

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), width="stretch")

    columns = list(df.columns)

    default_outcome = "Profit" if "Profit" in columns else columns[0]
    default_treatment = "Marketing Spend" if "Marketing Spend" in columns else columns[0]

    outcome_col = st.selectbox("Select outcome variable", columns, index=columns.index(default_outcome))
    treatment_col = st.selectbox("Select treatment variable", columns, index=columns.index(default_treatment))
    dml_type = st.selectbox("DML estimator", ["linear", "forest"], index=0)

    # =========================
    # ⚡ PERFORMANCE CONTROLS
    # =========================
    st.markdown("### ⚙️ Execution Settings")
    sample_rows = st.slider("Sample size (for faster execution)", 5000, 50000, 20000, step=5000)
    force_retrain = st.checkbox("🔁 Force Retrain Models", value=False)

    # =========================
    # ▶ RUN PIPELINE
    # =========================
    if st.button("Run End-to-End Pipeline", type="primary"):

        progress = st.progress(0, text="Starting pipeline...")

        try:
            if active_path is None:
                with NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
                    df.to_csv(tmp_file.name, index=False)
                    active_path = tmp_file.name

            progress.progress(10, text="Preparing data...")

            outcome = run_training_pipeline(
                data_path=active_path,
                treatment_col=treatment_col,
                regression_target=outcome_col,
                dml_model_type=dml_type,
                sample_rows=sample_rows,       # ✅ prevent freezing
                force_retrain=force_retrain,   # ✅ caching control
            )

            progress.progress(80, text="Finalizing results...")

            st.success("Pipeline completed successfully.")

            # =========================
            # 📊 RESULTS
            # =========================
            st.subheader("Best Models")
            st.write(
                {
                    "classification": outcome.best_classification_model,
                    "regression": outcome.best_regression_model,
                }
            )

            st.subheader("Classification Metrics")
            st.dataframe(outcome.classification_metrics, width="stretch")

            st.subheader("Regression Metrics")
            st.dataframe(outcome.regression_metrics, width="stretch")

            st.subheader("Causal Summary")
            st.json(outcome.dml_summary)

            st.subheader("Business Insights")
            for insight in outcome.business_insights:
                st.write(f"- {insight}")

            # =========================
            # 🖼 VISUALS
            # =========================
            image_names = [
                "correlation_heatmap.png",
                "classification_comparison.png",
                "regression_comparison.png",
                "prediction_vs_actual.png",
                "feature_importance.png",
                "causal_feature_importance.png",
                "causal_effect_distribution.png",
            ]

            st.subheader("Generated Visualizations")

            for image_name in image_names:
                image_path = Path("financial_dml_project") / "outputs" / image_name
                if image_path.exists():
                    st.image(str(image_path), caption=image_name, width="stretch")

            progress.progress(100, text="Done ✅")

        except Exception as exc:
            st.exception(exc)


    # =========================
    # 🔬 QUICK DML VIEW
    # =========================
    with st.expander("Standalone DML Quick View"):

        try:
            bundle = prepare_dataset(df, regression_target=outcome_col, treatment_col=treatment_col)

            dml_result = run_dml_analysis(
                bundle.data,
                treatment_col=treatment_col,
                outcome_col=outcome_col,
                confounders=bundle.confounders,
                model_type=dml_type,
            )

            st.write(
                {
                    "ATE": dml_result.ate,
                    "95% interval": dml_result.ate_interval,
                    "model": dml_result.model_name,
                }
            )

            if dml_result.inference_note:
                st.info(dml_result.inference_note)

            st.dataframe(dml_result.feature_importance, width="stretch")

        except Exception as exc:
            st.info(f"DML preview not available: {exc}")