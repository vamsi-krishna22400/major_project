from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

from utils import sanitize_columns


NUMERIC_COERCE_CANDIDATES = {"PCS", "GROSS AMT"}
KNOWN_OUTCOME_COLUMNS = ("Profit", "Amount")
KNOWN_TREATMENTS = ("Marketing Spend", "R&D Spend", "Administration")


@dataclass
class DatasetBundle:
    data: pd.DataFrame
    regression_target: str
    classification_target: str
    default_treatment: str
    confounders: list[str]
    low_cardinality_categoricals: list[str]
    high_cardinality_columns: list[str]
    numeric_features: list[str]


def load_financial_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = sanitize_columns(df)
    return df


def prepare_dataset(
    df: pd.DataFrame,
    regression_target: str | None = None,
    treatment_col: str | None = None,
    classification_target: str = "profit_class",
    rolling_window: int = 14,
    max_category_levels: int = 25,
) -> DatasetBundle:
    work_df = sanitize_columns(df).copy()
    _coerce_numeric_columns(work_df)

    if regression_target is None:
        regression_target = _pick_first_existing(work_df.columns, KNOWN_OUTCOME_COLUMNS)
    if regression_target is None:
        raise ValueError("No numeric outcome column found. Please specify a regression target.")

    if treatment_col is None:
        treatment_col = _pick_first_existing(work_df.columns, KNOWN_TREATMENTS)
    if treatment_col is None:
        raise ValueError("No treatment column found. Please specify a treatment variable.")

    work_df = add_financial_indicators(work_df, base_column=regression_target, window=rolling_window)
    work_df[classification_target] = (
        work_df[regression_target] >= work_df[regression_target].median()
    ).astype(int)

    low_cardinality_categoricals, high_cardinality_columns = split_categorical_columns(
        work_df,
        max_category_levels=max_category_levels,
        excluded={classification_target, regression_target, treatment_col},
    )
    numeric_features = [
        col
        for col in work_df.select_dtypes(include=[np.number, "bool"]).columns
        if col not in {classification_target, regression_target}
    ]
    confounders = [
        col
        for col in work_df.columns
        if col not in {classification_target, regression_target, treatment_col}
        and col not in {"Order ID", "ASIN", "SKU"}
        and col in set(low_cardinality_categoricals + numeric_features)
    ]

    return DatasetBundle(
        data=work_df,
        regression_target=regression_target,
        classification_target=classification_target,
        default_treatment=treatment_col,
        confounders=confounders,
        low_cardinality_categoricals=low_cardinality_categoricals,
        high_cardinality_columns=high_cardinality_columns,
        numeric_features=numeric_features,
    )


def _coerce_numeric_columns(df: pd.DataFrame) -> None:
    for column in df.columns:
        if column in NUMERIC_COERCE_CANDIDATES:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        elif df[column].dtype == object:
            numeric_version = pd.to_numeric(df[column], errors="coerce")
            if numeric_version.notna().mean() > 0.95:
                df[column] = numeric_version


def add_financial_indicators(df: pd.DataFrame, base_column: str, window: int = 14) -> pd.DataFrame:
    if base_column not in df.columns:
        raise ValueError(f"Base column '{base_column}' not found for indicator engineering.")

    enriched = df.copy()
    price = pd.to_numeric(enriched[base_column], errors="coerce").ffill().bfill()
    if price.isna().all():
        raise ValueError(f"Unable to create indicators because '{base_column}' cannot be converted to numeric.")

    delta = price.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)

    sma = price.rolling(window=window, min_periods=1).mean()
    rolling_std = price.rolling(window=window, min_periods=1).std().fillna(0)
    ema_short = price.ewm(span=12, adjust=False).mean()
    ema_long = price.ewm(span=26, adjust=False).mean()
    macd = ema_short - ema_long
    signal = macd.ewm(span=9, adjust=False).mean()

    enriched[f"{base_column}_ma_{window}"] = sma
    enriched[f"{base_column}_rsi_{window}"] = 100 - (100 / (1 + rs.fillna(0)))
    enriched[f"{base_column}_bollinger_upper"] = sma + 2 * rolling_std
    enriched[f"{base_column}_bollinger_lower"] = sma - 2 * rolling_std
    enriched[f"{base_column}_macd"] = macd
    enriched[f"{base_column}_macd_signal"] = signal
    return enriched


def split_categorical_columns(
    df: pd.DataFrame,
    max_category_levels: int = 25,
    excluded: Iterable[str] | None = None,
) -> tuple[list[str], list[str]]:
    excluded = set(excluded or [])
    categoricals = [
        col
        for col in df.select_dtypes(include=["object", "category"]).columns
        if col not in excluded
    ]
    low_cardinality = [col for col in categoricals if df[col].nunique(dropna=False) <= max_category_levels]
    high_cardinality = [col for col in categoricals if col not in low_cardinality]
    return low_cardinality, high_cardinality


def build_preprocessor(
    df: pd.DataFrame,
    target_col: str,
    treatment_col: str | None = None,
    scaler_type: str = "standard",
    max_category_levels: int = 25,
) -> tuple[ColumnTransformer, list[str], list[str]]:
    excluded = {target_col}
    if treatment_col is not None:
        excluded.add(treatment_col)

    low_cardinality, high_cardinality = split_categorical_columns(
        df,
        max_category_levels=max_category_levels,
        excluded=excluded,
    )
    numeric_features = [
        col
        for col in df.select_dtypes(include=[np.number, "bool"]).columns
        if col not in excluded
    ]

    scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", scaler),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, low_cardinality),
        ],
        remainder="drop",
        sparse_threshold=0,
    )
    return transformer, numeric_features + low_cardinality, high_cardinality


def build_pca_component(
    enabled: bool,
    n_components: float | int = 0.95,
) -> PCA | str:
    if not enabled:
        return "passthrough"
    return PCA(n_components=n_components, random_state=42)


def _pick_first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    column_set = set(columns)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    return None

