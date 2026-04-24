from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML, LinearDML
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


@dataclass
class DMLResult:
    treatment: str
    outcome: str
    ate: float
    ate_interval: tuple[float, float]
    effect_values: np.ndarray
    feature_importance: pd.DataFrame
    model_name: str
    inference_note: str | None = None


def run_dml_analysis(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    confounders: list[str],
    model_type: str = "linear",
    sample_size: int = 20000,
    random_state: int = 42,
) -> DMLResult:
    if treatment_col not in df.columns or outcome_col not in df.columns:
        raise ValueError("Treatment and outcome columns must exist in the dataframe.")
    if not confounders:
        raise ValueError("At least one confounder is required for DML estimation.")

    subset = df[[treatment_col, outcome_col] + confounders].copy()
    subset = subset.replace([np.inf, -np.inf], np.nan).dropna()
    if subset.empty:
        raise ValueError("No valid rows remain for DML after dropping missing values.")
    if len(subset) > sample_size:
        subset = subset.sample(sample_size, random_state=random_state)

    y = pd.to_numeric(subset[outcome_col], errors="coerce").to_numpy()
    t = pd.to_numeric(subset[treatment_col], errors="coerce").to_numpy()
    x = pd.get_dummies(subset[confounders], drop_first=True)

    if model_type == "forest":
        estimator = CausalForestDML(
            model_y=RandomForestRegressor(
                n_estimators=200,
                min_samples_leaf=10,
                random_state=random_state,
                n_jobs=-1,
            ),
            model_t=RandomForestRegressor(
                n_estimators=200,
                min_samples_leaf=10,
                random_state=random_state,
                n_jobs=-1,
            ),
            n_estimators=300,
            min_samples_leaf=10,
            random_state=random_state,
        )
        model_name = "CausalForestDML"
    else:
        estimator = LinearDML(
            model_y=RandomForestRegressor(
                n_estimators=200,
                random_state=random_state,
                n_jobs=-1,
            ),
            model_t=RandomForestRegressor(
                n_estimators=200,
                random_state=random_state,
                n_jobs=-1,
            ),
            random_state=random_state,
        )
        model_name = "LinearDML"

    x_train, x_eval, y_train, y_eval, t_train, t_eval = train_test_split(
        x, y, t, test_size=0.25, random_state=random_state
    )
    inference_note: str | None = None
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", UserWarning)
        estimator.fit(y_train, t_train, X=x_train)
        effect_values = estimator.effect(x_eval)
        ate = float(np.mean(effect_values))
        ate_interval_array = estimator.ate_interval(X=x_eval)
        ate_interval = (float(ate_interval_array[0]), float(ate_interval_array[1]))

    invalid_inference = any(
        "Co-variance matrix is underdetermined" in str(warning.message)
        for warning in caught_warnings
    )
    if invalid_inference:
        ate_interval = (float("nan"), float("nan"))
        inference_note = (
            "EconML could estimate the treatment effect, but the confidence interval is not reliable "
            "for this sample and feature set."
        )

    if hasattr(estimator, "feature_importances_"):
        importance_values = np.asarray(estimator.feature_importances_)
    elif hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_)
        importance_values = np.abs(coef).ravel()
    else:
        importance_values = np.zeros(x.shape[1])

    feature_importance = (
        pd.DataFrame({"feature": x.columns, "importance": importance_values})
        .sort_values("importance", ascending=False)
        .head(15)
    )

    return DMLResult(
        treatment=treatment_col,
        outcome=outcome_col,
        ate=ate,
        ate_interval=ate_interval,
        effect_values=effect_values,
        feature_importance=feature_importance,
        model_name=model_name,
        inference_note=inference_note,
    )


def generate_business_insights(
    feature_importance: pd.DataFrame,
    dml_result: DMLResult,
    regression_metrics: pd.DataFrame,
) -> list[str]:
    insights: list[str] = []
    if not feature_importance.empty:
        top_feature = feature_importance.iloc[0]
        insights.append(
            f"The strongest predictive driver of {dml_result.outcome} is '{top_feature['feature']}', "
            f"indicating this variable deserves close monitoring in allocation decisions."
        )
    if dml_result.inference_note:
        insights.append(
            f"The estimated average causal effect of '{dml_result.treatment}' on '{dml_result.outcome}' is "
            f"{dml_result.ate:.4f}. Confidence intervals were not reported because inference was unstable for this run."
        )
    else:
        insights.append(
            f"The estimated average causal effect of '{dml_result.treatment}' on '{dml_result.outcome}' is "
            f"{dml_result.ate:.4f}, with a 95% interval of [{dml_result.ate_interval[0]:.4f}, {dml_result.ate_interval[1]:.4f}]."
        )
    if not dml_result.feature_importance.empty:
        causal_driver = dml_result.feature_importance.iloc[0]["feature"]
        insights.append(
            f"'{causal_driver}' appears to moderate the treatment effect most strongly, so segmenting resource allocation by this factor should improve ROI."
        )
    if not regression_metrics.empty:
        best_model = regression_metrics.sort_values("r2", ascending=False).iloc[0]
        insights.append(
            f"For forecasting, '{best_model['model']}' delivered the strongest regression fit (R2={best_model['r2']:.4f}), making it the best candidate for operational deployment."
        )
    insights.append(
        "Risk analysis: high-cardinality operational identifiers were excluded from causal and predictive training to reduce noise and leakage risk; monitor drift in spend, fulfillment status, and regional mix."
    )
    insights.append(
        f"Resource allocation recommendation: prioritize budget changes through '{dml_result.treatment}' only after validating response segments with the largest estimated uplift and acceptable uncertainty."
    )
    return insights
