from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from dml import generate_business_insights, run_dml_analysis
from preprocess import build_preprocessor, load_financial_data, prepare_dataset
from utils import (
    MODEL_DIR,
    OUTPUT_DIR,
    ensure_output_dirs,
    evaluate_classification,
    evaluate_regression,
    save_json,
    setup_logging,
)


# =========================
# 📦 DATA CLASS
# =========================
@dataclass
class TrainingOutcome:
    classification_metrics: pd.DataFrame
    regression_metrics: pd.DataFrame
    best_classification_model: str
    best_regression_model: str
    dml_summary: dict[str, Any]
    business_insights: list[str]
    artifacts: dict[str, str]


# =========================
# ⚙ MODEL CONFIGS (LIGHTWEIGHT)
# =========================
def get_classification_configs(random_state=42):
    return {
        "logistic_regression": {
            "estimator": LogisticRegression(max_iter=1000),
            "search": {"model__C": [0.1, 1, 5]},
            "use_pca": True,
            "sample_cap": 30000,
        },
        "random_forest": {
            "estimator": RandomForestClassifier(n_estimators=150, n_jobs=-1),
            "search": {"model__max_depth": [10, None]},
            "use_pca": False,
            "sample_cap": 50000,
        },
    }


def get_regression_configs(random_state=42):
    return {
        "linear_regression": {
            "estimator": LinearRegression(),
            "search": {},
            "use_pca": True,
            "sample_cap": 50000,
        },
        "random_forest": {
            "estimator": RandomForestRegressor(n_estimators=150, n_jobs=-1),
            "search": {"model__max_depth": [10, None]},
            "use_pca": False,
            "sample_cap": 50000,
        },
    }


# =========================
# 🔁 TRAIN FUNCTION
# =========================
def train_and_compare_models(df, target_col, task_type):

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    configs = (
        get_classification_configs()
        if task_type == "classification"
        else get_regression_configs()
    )

    results = []
    trained_models = {}

    for name, cfg in configs.items():

        preprocessor, _, _ = build_preprocessor(X_train, target_col)

        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("pca", PCA(n_components=0.95) if cfg["use_pca"] else "passthrough"),
                ("model", clone(cfg["estimator"])),
            ]
        )

        if cfg["search"]:
            search = RandomizedSearchCV(
                pipeline,
                cfg["search"],
                n_iter=3,
                cv=3,
                n_jobs=-1,
            )
            search.fit(X_train, y_train)
            model = search.best_estimator_
        else:
            model = pipeline.fit(X_train, y_train)

        preds = model.predict(X_test)

        metrics = (
            evaluate_classification(y_test, preds)
            if task_type == "classification"
            else evaluate_regression(y_test, preds)
        )

        results.append({"model": name, **metrics})
        trained_models[name] = model

    df_metrics = pd.DataFrame(results)
    best_model = df_metrics.sort_values(df_metrics.columns[1], ascending=False).iloc[0]["model"]

    return df_metrics, trained_models[best_model], best_model


# =========================
# 🚀 MAIN PIPELINE (CACHED)
# =========================
def run_training_pipeline(
    data_path: str,
    treatment_col=None,
    regression_target=None,
    dml_model_type="linear",
    sample_rows=None,
    force_retrain=False,
) -> TrainingOutcome:

    ensure_output_dirs()
    logger = setup_logging()

    clf_path = MODEL_DIR / "clf_model.joblib"
    reg_path = MODEL_DIR / "reg_model.joblib"
    summary_path = OUTPUT_DIR / "summary.json"

    # =========================
    # ✅ LOAD IF EXISTS
    # =========================
    if (
        not force_retrain
        and clf_path.exists()
        and reg_path.exists()
        and summary_path.exists()
    ):
        logger.info("Loading cached models...")

        import json

        with open(summary_path, "r") as f:
            summary = json.load(f)

        return TrainingOutcome(
            classification_metrics=pd.DataFrame(summary["classification_metrics"]),
            regression_metrics=pd.DataFrame(summary["regression_metrics"]),
            best_classification_model="loaded",
            best_regression_model="loaded",
            dml_summary=summary["dml_summary"],
            business_insights=summary["business_insights"],
            artifacts={},
        )

    # =========================
    # 🔥 TRAIN
    # =========================
    logger.info("Training models...")

    df = load_financial_data(data_path)

    if sample_rows:
        df = df.sample(sample_rows)

    bundle = prepare_dataset(df, regression_target, treatment_col)

    clf_metrics, clf_model, clf_name = train_and_compare_models(
        bundle.data, bundle.classification_target, "classification"
    )

    reg_metrics, reg_model, reg_name = train_and_compare_models(
        bundle.data, bundle.regression_target, "regression"
    )

    joblib.dump(clf_model, clf_path)
    joblib.dump(reg_model, reg_path)

    # =========================
    # 🔬 DML
    # =========================
    dml_result = run_dml_analysis(
        bundle.data,
        bundle.default_treatment,
        bundle.regression_target,
        bundle.confounders,
        model_type=dml_model_type,
    )

    dml_summary = {
        "ate": dml_result.ate,
        "model": dml_result.model_name,
    }

    insights = generate_business_insights(
        pd.DataFrame(), dml_result, reg_metrics
    )

    summary = {
        "classification_metrics": clf_metrics.to_dict("records"),
        "regression_metrics": reg_metrics.to_dict("records"),
        "dml_summary": dml_summary,
        "business_insights": insights,
    }

    save_json(summary, summary_path)

    return TrainingOutcome(
        clf_metrics,
        reg_metrics,
        clf_name,
        reg_name,
        dml_summary,
        insights,
        {},
    )


# =========================
# 🧠 CLI
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--force-retrain", action="store_true")
    args = parser.parse_args()

    result = run_training_pipeline(
        args.data_path,
        force_retrain=args.force_retrain,
        sample_rows=20000,
    )

    print("Done:", result.best_regression_model)


if __name__ == "__main__":
    main()