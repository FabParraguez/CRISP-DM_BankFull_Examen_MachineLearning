from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "Data" / "bank-full.csv"
TARGET = "y"
POST_CONTACT_FEATURES = ("duration", "campaign", "day", "month", "contact")


@dataclass(frozen=True)
class DataSplits:
    x_train: pd.DataFrame
    x_validation: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series


def load_bank_data(data_path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load Bank Marketing data and apply only deterministic record-level cleaning."""
    path = Path(data_path)
    if not path.is_file():
        raise FileNotFoundError(f"Bank Marketing dataset was not found: {path}")

    data = pd.read_csv(path, sep=";")
    missing_columns = {"y"}.difference(data.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)}"
        )

    data = data.drop_duplicates().reset_index(drop=True)
    string_columns = data.select_dtypes(include=["object", "string"]).columns
    for column in string_columns:
        data[column] = data[column].str.strip()

    target = data[TARGET].map({"yes": 1, "no": 0})
    if target.isna().any():
        invalid_values = data.loc[target.isna(), TARGET].drop_duplicates().tolist()
        raise ValueError(f"Unexpected target values in '{TARGET}': {invalid_values}")
    data[TARGET] = target.astype("int8")
    return data


def get_model_features(
    data: pd.DataFrame, selected_features: list[str] | tuple[str, ...] | None = None
) -> pd.DataFrame:
    """Select pre-contact predictors, excluding the outcome and post-contact fields."""
    if TARGET not in data:
        raise ValueError(f"Dataset must contain the target column '{TARGET}'.")

    available = [
        column
        for column in data.columns
        if column != TARGET and column not in POST_CONTACT_FEATURES
    ]
    features = available if selected_features is None else list(selected_features)
    if not features:
        raise ValueError("Select at least one predictor.")
    invalid = sorted(set(features).difference(available))
    if invalid:
        raise ValueError(
            "Predictors are unknown or unavailable before customer contact: "
            f"{invalid}"
        )
    return data.loc[:, features].copy()


def split_data(
    data: pd.DataFrame,
    selected_features: list[str] | tuple[str, ...] | None = None,
    test_size: float = 0.2,
    validation_size: float = 0.2,
    random_state: int = 42,
) -> DataSplits:
    """Create stratified train/validation/test partitions with a final blind test."""
    if not 0 < test_size < 1 or not 0 < validation_size < 1:
        raise ValueError("Test and validation proportions must be between 0 and 1.")
    if test_size + validation_size >= 1:
        raise ValueError("Train, validation, and test partitions must all be non-empty.")

    x = get_model_features(data, selected_features)
    y = data[TARGET].astype("int8")
    x_development, x_test, y_development, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    validation_fraction_of_development = validation_size / (1 - test_size)
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_development,
        y_development,
        test_size=validation_fraction_of_development,
        random_state=random_state,
        stratify=y_development,
    )
    return DataSplits(
        x_train=x_train,
        x_validation=x_validation,
        x_test=x_test,
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test,
    )