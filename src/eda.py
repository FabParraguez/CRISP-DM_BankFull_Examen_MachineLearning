from __future__ import annotations

from math import ceil
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.etl import TARGET


def summarize_data(data: pd.DataFrame) -> dict[str, Any]:
    """Return the main structure and data-quality checks for the Bank dataset."""
    if TARGET not in data:
        raise ValueError(f"Dataset must contain the target column '{TARGET}'.")

    numeric_columns = data.select_dtypes(include="number").columns.tolist()
    categorical_columns = data.select_dtypes(exclude="number").columns.tolist()
    target_counts = data[TARGET].value_counts(dropna=False).sort_index()
    target_percentages = (
        data[TARGET].value_counts(normalize=True, dropna=False).sort_index() * 100
    )
    return {
        "shape": data.shape,
        "duplicate_rows": int(data.duplicated().sum()),
        "missing_by_column": data.isna().sum().sort_values(ascending=False),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "numeric_summary": data[numeric_columns].describe().T,
        "target_counts": target_counts,
        "target_percentages": target_percentages.round(2),
    }


def count_iqr_outliers(data: pd.DataFrame) -> pd.DataFrame:
    """Count IQR outliers without deleting legitimate high-value customers."""
    numeric = data.select_dtypes(include="number").drop(
        columns=[TARGET], errors="ignore"
    )
    rows = []
    for column in numeric:
        q1 = numeric[column].quantile(0.25)
        q3 = numeric[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((numeric[column] < lower) | (numeric[column] > upper)).sum())
        rows.append(
            {
                "Variable": column,
                "Limite inferior IQR": lower,
                "Limite superior IQR": upper,
                "Valores atipicos": count,
                "Porcentaje": count / len(numeric) * 100 if len(numeric) else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(
        "Valores atipicos", ascending=False
    ).reset_index(drop=True)


def plot_data_overview(data: pd.DataFrame) -> list[plt.Figure]:
    """Create target, numeric distribution, outlier, and correlation plots."""
    if TARGET not in data:
        raise ValueError(f"Dataset must contain the target column '{TARGET}'.")
    sns.set_theme(style="whitegrid")
    figures: list[plt.Figure] = []

    target_figure, target_axis = plt.subplots(figsize=(6, 4))
    sns.countplot(data=data, x=TARGET, hue=TARGET, legend=False, ax=target_axis)
    target_axis.set(
        title="Distribucion de la suscripcion",
        xlabel="Suscripcion",
        ylabel="Clientes",
    )
    target_figure.tight_layout()
    figures.append(target_figure)

    numeric = data.select_dtypes(include="number").drop(
        columns=[TARGET], errors="ignore"
    )
    if not numeric.empty:
        distribution_figure, distribution_axes = plt.subplots(
            ceil(len(numeric.columns) / 3), 3, figsize=(14, 4 * ceil(len(numeric.columns) / 3))
        )
        for axis, column in zip(distribution_axes.flat, numeric.columns):
            sns.histplot(data=data, x=column, bins=30, ax=axis)
            axis.set_title(f"Distribucion de {column}")
        for axis in distribution_axes.flat[len(numeric.columns) :]:
            axis.set_visible(False)
        distribution_figure.tight_layout()
        figures.append(distribution_figure)

        outlier_figure, outlier_axis = plt.subplots(figsize=(12, 5))
        sns.boxplot(data=numeric, orient="h", ax=outlier_axis)
        outlier_axis.set_title("Revision visual de valores atipicos")
        outlier_figure.tight_layout()
        figures.append(outlier_figure)

        correlation_figure, correlation_axis = plt.subplots(figsize=(9, 7))
        sns.heatmap(
            numeric.corr(),
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            vmin=-1,
            vmax=1,
            ax=correlation_axis,
        )
        correlation_axis.set_title("Correlacion entre variables numericas")
        correlation_figure.tight_layout()
        figures.append(correlation_figure)
    return figures