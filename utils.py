from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)


ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "outputs"
MODEL_DIR = ROOT_DIR / "models"


def setup_logging(log_file: str | Path | None = None) -> logging.Logger:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("financial_dml")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is None:
        log_file = OUTPUT_DIR / "pipeline.log"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [str(col).strip() for col in cleaned.columns]
    return cleaned


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def ensure_output_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_classification(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def evaluate_regression(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    rmse = root_mean_squared_error(y_true, y_pred)
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": rmse,
        "r2": r2_score(y_true, y_pred),
    }


def plot_correlation_heatmap(df: pd.DataFrame, target_col: str, path: str | Path) -> None:
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    if target_col not in numeric_df.columns or numeric_df.shape[1] < 2:
        return

    corr = numeric_df.corr(numeric_only=True)
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_prediction_vs_actual(
    y_true: pd.Series,
    y_pred: np.ndarray,
    path: str | Path,
    sample_size: int = 1500,
) -> None:
    plot_df = pd.DataFrame({"Actual": y_true, "Predicted": y_pred})
    if len(plot_df) > sample_size:
        plot_df = plot_df.sample(sample_size, random_state=42)

    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=plot_df, x="Actual", y="Predicted", alpha=0.6)
    diagonal_min = min(plot_df["Actual"].min(), plot_df["Predicted"].min())
    diagonal_max = max(plot_df["Actual"].max(), plot_df["Predicted"].max())
    plt.plot([diagonal_min, diagonal_max], [diagonal_min, diagonal_max], "r--")
    plt.title("Prediction vs Actual")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_model_comparison(
    metrics_df: pd.DataFrame,
    metric_col: str,
    title: str,
    path: str | Path,
    ascending: bool,
) -> None:
    if metrics_df.empty or metric_col not in metrics_df.columns:
        return

    ordered = metrics_df.sort_values(metric_col, ascending=ascending)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=ordered, x="model", y=metric_col, hue="model", palette="viridis", legend=False)
    plt.xticks(rotation=30, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def extract_feature_importance(
    fitted_pipeline,
    feature_names: list[str],
    X_sample: pd.DataFrame | np.ndarray,
    y_sample: pd.Series | np.ndarray,
    task_type: str,
    top_n: int = 15,
) -> pd.DataFrame:
    estimator = fitted_pipeline.named_steps["model"]
    importances: np.ndarray | None = None

    if hasattr(estimator, "feature_importances_"):
        importances = np.asarray(estimator.feature_importances_)
    elif hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_)
        importances = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
    else:
        scoring = "f1" if task_type == "classification" else "r2"
        result = permutation_importance(
            fitted_pipeline,
            X_sample,
            y_sample,
            n_repeats=5,
            random_state=42,
            scoring=scoring,
        )
        importances = result.importances_mean

    feature_count = min(len(feature_names), len(importances))
    importance_df = pd.DataFrame(
        {
            "feature": feature_names[:feature_count],
            "importance": importances[:feature_count],
        }
    )
    return importance_df.sort_values("importance", ascending=False).head(top_n)


def plot_feature_importance(importance_df: pd.DataFrame, path: str | Path, title: str) -> None:
    if importance_df.empty:
        return

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=importance_df.sort_values("importance", ascending=True),
        x="importance",
        y="feature",
        hue="feature",
        palette="crest",
        legend=False,
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_causal_effect_distribution(effect_values: np.ndarray, path: str | Path) -> None:
    if effect_values.size == 0:
        return

    plt.figure(figsize=(10, 6))
    sns.histplot(effect_values, kde=True, color="teal")
    plt.axvline(float(np.mean(effect_values)), color="red", linestyle="--", label="ATE")
    plt.legend()
    plt.title("Estimated Treatment Effect Distribution")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def try_compute_shap_summary(
    fitted_pipeline,
    X_sample: pd.DataFrame,
    path: str | Path,
    logger: logging.Logger,
) -> str | None:
    try:
        import shap
    except ImportError:
        logger.info("SHAP is not installed; skipping SHAP plot.")
        return None

    estimator = fitted_pipeline.named_steps["model"]
    if not hasattr(estimator, "predict"):
        return None

    transformed = fitted_pipeline.named_steps["preprocessor"].transform(X_sample)
    if "pca" in fitted_pipeline.named_steps and fitted_pipeline.named_steps["pca"] != "passthrough":
        transformed = fitted_pipeline.named_steps["pca"].transform(transformed)

    sample_matrix = transformed[: min(len(X_sample), 300)]
    try:
        explainer = shap.Explainer(estimator, sample_matrix)
        shap_values = explainer(sample_matrix)
        shap.summary_plot(shap_values, sample_matrix, show=False)
        plt.tight_layout()
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
        return str(path)
    except Exception as exc:  # pragma: no cover - optional visualization branch
        logger.info("Unable to compute SHAP values: %s", exc)
        plt.close("all")
        return None
