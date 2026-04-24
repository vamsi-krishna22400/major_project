from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
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
    extract_feature_importance,
    plot_causal_effect_distribution,
    plot_correlation_heatmap,
    plot_feature_importance,
    plot_model_comparison,
    plot_prediction_vs_actual,
    save_json,
    setup_logging,
    try_compute_shap_summary,
)


@dataclass
class TrainingOutcome:
    classification_metrics: pd.DataFrame
    regression_metrics: pd.DataFrame
    best_classification_model: str
    best_regression_model: str
    dml_summary: dict[str, Any]
    business_insights: list[str]
    artifacts: dict[str, str]


def get_classification_configs(random_state: int = 42) -> dict[str, dict[str, Any]]:
    return {
        "logistic_regression": {
            "estimator": LogisticRegression(max_iter=2000, random_state=random_state),
            "search": {
                "model__C": [0.1, 1.0, 5.0],
                "model__solver": ["lbfgs", "liblinear"],
            },
            "use_pca": True,
            "sample_cap": 60000,
        },
        "svm_rbf": {
            "estimator": SVC(kernel="rbf", probability=True, random_state=random_state),
            "search": {
                "model__C": [0.5, 1.0, 2.0],
                "model__gamma": ["scale", 0.1, 0.01],
            },
            "use_pca": True,
            "sample_cap": 12000,
        },
        "knn": {
            "estimator": KNeighborsClassifier(),
            "search": {
                "model__n_neighbors": [5, 9, 15],
                "model__weights": ["uniform", "distance"],
            },
            "use_pca": True,
            "sample_cap": 25000,
        },
        "decision_tree": {
            "estimator": DecisionTreeClassifier(random_state=random_state),
            "search": {
                "model__max_depth": [5, 10, 20, None],
                "model__min_samples_split": [2, 10, 20],
            },
            "use_pca": False,
            "sample_cap": 70000,
        },
        "random_forest": {
            "estimator": RandomForestClassifier(
                n_estimators=250, random_state=random_state, n_jobs=-1
            ),
            "search": {
                "model__max_depth": [8, 16, None],
                "model__min_samples_split": [2, 10],
                "model__min_samples_leaf": [1, 5],
            },
            "use_pca": False,
            "sample_cap": 70000,
        },
    }


def get_regression_configs(random_state: int = 42) -> dict[str, dict[str, Any]]:
    return {
        "linear_regression": {
            "estimator": LinearRegression(),
            "search": {},
            "use_pca": True,
            "sample_cap": 70000,
        },
        "svr_rbf": {
            "estimator": SVR(kernel="rbf"),
            "search": {
                "model__C": [0.5, 1.0, 3.0],
                "model__epsilon": [0.05, 0.1, 0.2],
            },
            "use_pca": True,
            "sample_cap": 12000,
        },
        "knn_regressor": {
            "estimator": KNeighborsRegressor(),
            "search": {
                "model__n_neighbors": [5, 9, 15],
                "model__weights": ["uniform", "distance"],
            },
            "use_pca": True,
            "sample_cap": 25000,
        },
        "decision_tree_regressor": {
            "estimator": DecisionTreeRegressor(random_state=random_state),
            "search": {
                "model__max_depth": [5, 10, 20, None],
                "model__min_samples_split": [2, 10, 20],
            },
            "use_pca": False,
            "sample_cap": 70000,
        },
        "random_forest_regressor": {
            "estimator": RandomForestRegressor(
                n_estimators=250, random_state=random_state, n_jobs=-1
            ),
            "search": {
                "model__max_depth": [8, 16, None],
                "model__min_samples_split": [2, 10],
                "model__min_samples_leaf": [1, 5],
            },
            "use_pca": False,
            "sample_cap": 70000,
        },
    }


def train_and_compare_models(
    df: pd.DataFrame,
    target_col: str,
    task_type: str,
    drop_columns: list[str] | None = None,
    scaler_type: str = "standard",
    random_state: int = 42,
    max_category_levels: int = 25,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], pd.Series, np.ndarray]:
    removable = [target_col] + list(drop_columns or [])
    X = df.drop(columns=[col for col in removable if col in df.columns])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y if task_type == "classification" else None,
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    configs = (
        get_classification_configs(random_state)
        if task_type == "classification"
        else get_regression_configs(random_state)
    )

    results: list[dict[str, Any]] = []
    trained_models: dict[str, Any] = {}
    test_predictions: dict[str, np.ndarray] = {}

    for model_name, config in configs.items():
        X_train_model, y_train_model = maybe_subsample(
            X_train,
            y_train,
            sample_cap=config["sample_cap"],
            stratify=task_type == "classification",
            random_state=random_state,
        )
        preprocessor, _, high_cardinality = build_preprocessor(
            X_train_model,
            target_col=target_col,
            scaler_type=scaler_type,
            max_category_levels=max_category_levels,
        )
        pca_step = PCA(n_components=0.95, random_state=random_state) if config["use_pca"] else "passthrough"
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("pca", pca_step),
                ("model", clone(config["estimator"])),
            ]
        )

        search_params = config["search"]
        if search_params:
            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=search_params,
                n_iter=min(6, np.prod([len(v) for v in search_params.values()])),
                cv=cv,
                scoring="f1" if task_type == "classification" else "r2",
                n_jobs=-1,
                random_state=random_state,
                refit=True,
            )
            search.fit(X_train_model, y_train_model)
            best_model = search.best_estimator_
            best_params = search.best_params_
        else:
            best_model = pipeline.fit(X_train_model, y_train_model)
            best_params = {}

        y_pred = best_model.predict(X_test)
        metrics = (
            evaluate_classification(y_test, y_pred)
            if task_type == "classification"
            else evaluate_regression(y_test, y_pred)
        )
        results.append(
            {
                "model": model_name,
                **metrics,
                "high_cardinality_excluded": ", ".join(high_cardinality) if high_cardinality else "",
                "best_params": best_params,
            }
        )
        trained_models[model_name] = best_model
        test_predictions[model_name] = y_pred

    metrics_df = pd.DataFrame(results)
    primary_metric = "f1" if task_type == "classification" else "r2"
    best_row = metrics_df.sort_values(primary_metric, ascending=False).iloc[0]
    best_model_name = str(best_row["model"])
    return (
        metrics_df,
        trained_models,
        {"name": best_model_name, "model": trained_models[best_model_name]},
        y_test,
        test_predictions[best_model_name],
    )


def maybe_subsample(
    X: pd.DataFrame,
    y: pd.Series,
    sample_cap: int,
    stratify: bool,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    if len(X) <= sample_cap:
        return X, y

    X_sub, _, y_sub, _ = train_test_split(
        X,
        y,
        train_size=sample_cap,
        random_state=random_state,
        stratify=y if stratify else None,
    )
    return X_sub, y_sub


def run_training_pipeline(
    data_path: str,
    output_dir: str | Path | None = None,
    treatment_col: str | None = None,
    regression_target: str | None = None,
    dml_model_type: str = "linear",
    sample_rows: int | None = None,
) -> TrainingOutcome:
    ensure_output_dirs()
    logger = setup_logging()
    output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Loading dataset from %s", data_path)

    raw_df = load_financial_data(data_path)
    if sample_rows is not None and len(raw_df) > sample_rows:
        raw_df = raw_df.sample(sample_rows, random_state=42).reset_index(drop=True)
        logger.info("Using sampled subset of %s rows for this run.", sample_rows)
    bundle = prepare_dataset(
        raw_df,
        regression_target=regression_target,
        treatment_col=treatment_col,
    )
    logger.info(
        "Prepared dataset with %s rows and %s columns; treatment=%s; outcome=%s",
        len(bundle.data),
        bundle.data.shape[1],
        bundle.default_treatment,
        bundle.regression_target,
    )

    classification_metrics, classification_models, best_classification, _, _ = train_and_compare_models(
        bundle.data,
        target_col=bundle.classification_target,
        task_type="classification",
        drop_columns=[bundle.regression_target],
    )
    regression_metrics, regression_models, best_regression, y_test_reg, y_pred_reg = train_and_compare_models(
        bundle.data,
        target_col=bundle.regression_target,
        task_type="regression",
        drop_columns=[bundle.classification_target],
    )

    classification_metrics.to_csv(output_dir / "classification_metrics.csv", index=False)
    regression_metrics.to_csv(output_dir / "regression_metrics.csv", index=False)
    joblib.dump(best_classification["model"], MODEL_DIR / "best_classification_model.joblib")
    joblib.dump(best_regression["model"], MODEL_DIR / "best_regression_model.joblib")

    plot_correlation_heatmap(bundle.data, bundle.regression_target, output_dir / "correlation_heatmap.png")
    plot_prediction_vs_actual(y_test_reg, y_pred_reg, output_dir / "prediction_vs_actual.png")
    plot_model_comparison(
        classification_metrics,
        metric_col="f1",
        title="Classification Model Comparison (F1)",
        path=output_dir / "classification_comparison.png",
        ascending=False,
    )
    plot_model_comparison(
        regression_metrics,
        metric_col="r2",
        title="Regression Model Comparison (R2)",
        path=output_dir / "regression_comparison.png",
        ascending=False,
    )

    best_regression_model = best_regression["model"]
    sample_features = bundle.data.drop(
        columns=[col for col in [bundle.regression_target, bundle.classification_target] if col in bundle.data.columns]
    ).sample(
        min(1000, len(bundle.data)),
        random_state=42,
    )
    sample_target = bundle.data.loc[sample_features.index, bundle.regression_target]
    feature_names = extract_transformed_feature_names(best_regression_model, sample_features)
    importance_df = extract_feature_importance(
        best_regression_model,
        feature_names,
        sample_features,
        sample_target,
        task_type="regression",
    )
    plot_feature_importance(
        importance_df,
        output_dir / "feature_importance.png",
        title=f"Feature Importance - {best_regression['name']}",
    )
    try_compute_shap_summary(
        best_regression_model,
        sample_features,
        output_dir / "shap_summary.png",
        logger=logger,
    )

    dml_result = run_dml_analysis(
        bundle.data,
        treatment_col=bundle.default_treatment,
        outcome_col=bundle.regression_target,
        confounders=bundle.confounders,
        model_type=dml_model_type,
    )
    plot_feature_importance(
        dml_result.feature_importance,
        output_dir / "causal_feature_importance.png",
        title=f"Causal Drivers - {dml_result.model_name}",
    )
    plot_causal_effect_distribution(dml_result.effect_values, output_dir / "causal_effect_distribution.png")

    business_insights = generate_business_insights(
        feature_importance=importance_df,
        dml_result=dml_result,
        regression_metrics=regression_metrics,
    )
    dml_summary = {
        "treatment": dml_result.treatment,
        "outcome": dml_result.outcome,
        "ate": dml_result.ate,
        "ate_interval": dml_result.ate_interval,
        "model_name": dml_result.model_name,
        "inference_note": dml_result.inference_note,
        "top_causal_features": dml_result.feature_importance.to_dict(orient="records"),
    }
    summary_payload = {
        "classification_metrics": classification_metrics.to_dict(orient="records"),
        "regression_metrics": regression_metrics.to_dict(orient="records"),
        "best_classification_model": best_classification["name"],
        "best_regression_model": best_regression["name"],
        "dml_summary": dml_summary,
        "business_insights": business_insights,
    }
    save_json(summary_payload, output_dir / "summary.json")
    logger.info("Pipeline completed successfully.")

    return TrainingOutcome(
        classification_metrics=classification_metrics,
        regression_metrics=regression_metrics,
        best_classification_model=best_classification["name"],
        best_regression_model=best_regression["name"],
        dml_summary=dml_summary,
        business_insights=business_insights,
        artifacts={
            "classification_metrics": str(output_dir / "classification_metrics.csv"),
            "regression_metrics": str(output_dir / "regression_metrics.csv"),
            "summary": str(output_dir / "summary.json"),
            "best_classification_model": str(MODEL_DIR / "best_classification_model.joblib"),
            "best_regression_model": str(MODEL_DIR / "best_regression_model.joblib"),
        },
    )


def extract_transformed_feature_names(fitted_pipeline: Pipeline, sample_features: pd.DataFrame) -> list[str]:
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    transformed_feature_names = list(preprocessor.get_feature_names_out())
    pca_step = fitted_pipeline.named_steps["pca"]
    if pca_step != "passthrough":
        n_components = pca_step.n_components_
        return [f"PC{i + 1}" for i in range(int(n_components))]
    return transformed_feature_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train financial ML and DML system.")
    parser.add_argument("--data-path", required=True, help="Path to the financial CSV dataset.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Directory for reports and plots.")
    parser.add_argument("--treatment-col", default=None, help="Treatment variable for DML.")
    parser.add_argument("--regression-target", default=None, help="Outcome target for regression.")
    parser.add_argument(
        "--dml-model-type",
        choices=["linear", "forest"],
        default="linear",
        help="EconML estimator to use.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="Optional row cap for quicker smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outcome = run_training_pipeline(
        data_path=args.data_path,
        output_dir=args.output_dir,
        treatment_col=args.treatment_col,
        regression_target=args.regression_target,
        dml_model_type=args.dml_model_type,
        sample_rows=args.sample_rows,
    )
    print("Best classification model:", outcome.best_classification_model)
    print("Best regression model:", outcome.best_regression_model)
    print("DML summary:", outcome.dml_summary)
    print("Business insights:")
    for insight in outcome.business_insights:
        print("-", insight)


if __name__ == "__main__":
    main()
