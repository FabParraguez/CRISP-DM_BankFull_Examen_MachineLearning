from __future__ import annotations

import json
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.etl import DEFAULT_DATA_PATH, DataSplits, load_bank_data, split_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "Modelos"
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2
MODEL_LABELS = {
    "logistic_regression": "Regresion logistica",
    "decision_tree": "Arbol de decision",
    "random_forest": "Random Forest",
}


@dataclass
class ExperimentResult:
    models: dict[str, Pipeline]
    splits: DataSplits
    validation_metrics: pd.DataFrame
    test_metrics: pd.DataFrame
    test_predictions: pd.DataFrame
    feature_importance: pd.DataFrame
    feature_ranking: pd.DataFrame
    search_results: pd.DataFrame
    best_model_name: str
    selected_features: list[str]
    random_state: int
    test_size: float
    validation_size: float


def make_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric_features = features.select_dtypes(include="number").columns.tolist()
    categorical_features = [
        column for column in features.columns if column not in numeric_features
    ]
    transformers = []
    if numeric_features:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            )
        )
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )


def rank_features(x_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """Rank source features using mutual information from training rows only."""
    encoded = x_train.copy()
    discrete_mask = []
    for feature in encoded.columns:
        if not pd.api.types.is_numeric_dtype(encoded[feature]):
            encoded[feature] = encoded[feature].astype("category").cat.codes
            discrete_mask.append(True)
        else:
            discrete_mask.append(False)
    scores = mutual_info_classif(
        encoded,
        y_train,
        discrete_features=discrete_mask,
        random_state=RANDOM_STATE,
    )
    return (
        pd.DataFrame({"Variable": x_train.columns, "Importancia": scores})
        .sort_values("Importancia", ascending=False)
        .reset_index(drop=True)
    )


def _pipeline(
    model_name: str,
    features: pd.DataFrame,
    iteration: int,
    random_state: int,
) -> Pipeline:
    if model_name == "logistic_regression":
        estimator = LogisticRegression(max_iter=1500, random_state=random_state)
    elif model_name == "decision_tree":
        estimator = DecisionTreeClassifier(random_state=random_state)
    elif model_name == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=1,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    steps: list[tuple[str, Any]] = [
        ("preprocessing", make_preprocessor(features))
    ]
    if iteration == 2:
        steps.append(("selection", SelectPercentile(score_func=f_classif)))
        if model_name == "logistic_regression":
            estimator.set_params(class_weight="balanced")
        elif model_name == "decision_tree":
            estimator.set_params(class_weight="balanced")
        else:
            estimator.set_params(class_weight="balanced_subsample")
    elif iteration != 1:
        raise ValueError(f"Unsupported iteration: {iteration}")
    steps.append(("classifier", estimator))
    return Pipeline(steps=steps)


def _search_grid(model_name: str) -> dict[str, list[Any]]:
    shared = {"selection__percentile": [50, 75]}
    if model_name == "logistic_regression":
        return {**shared, "classifier__C": [0.1, 1.0, 10.0]}
    if model_name == "decision_tree":
        return {
            **shared,
            "classifier__max_depth": [6, 12, None],
            "classifier__min_samples_leaf": [5, 20],
        }
    if model_name == "random_forest":
        return {
            **shared,
            "classifier__n_estimators": [80, 120],
            "classifier__max_depth": [10, None],
            "classifier__min_samples_leaf": [2, 8],
        }
    raise ValueError(f"Unknown model: {model_name}")


def _model_key(iteration: int, model_name: str) -> str:
    return f"Iteracion {iteration} | {MODEL_LABELS[model_name]}"


def _evaluate(
    models: dict[str, Pipeline], x: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    prediction_rows = pd.DataFrame({"target_real": y.to_numpy()}, index=x.index)
    for name, model in models.items():
        predicted = model.predict(x)
        probabilities = model.predict_proba(x)[:, 1]
        metric_rows.append(
            {
                "Iteracion / modelo": name,
                "Accuracy": accuracy_score(y, predicted),
                "Precision": precision_score(y, predicted, zero_division=0),
                "Recall": recall_score(y, predicted, zero_division=0),
                "F1": f1_score(y, predicted, zero_division=0),
                "ROC-AUC": roc_auc_score(y, probabilities),
                "PR-AUC": average_precision_score(y, probabilities),
            }
        )
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        prediction_rows[f"{slug}_prediccion"] = predicted
        prediction_rows[f"{slug}_probabilidad"] = probabilities
    return pd.DataFrame(metric_rows).set_index("Iteracion / modelo"), prediction_rows


def _raw_feature_importance(
    model: Pipeline, feature_names: list[str]
) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocessing"]
    transformed_names = preprocessor.get_feature_names_out()
    classifier = model.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        values = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        values = np.abs(classifier.coef_).mean(axis=0)
    else:
        raise TypeError("The selected classifier does not expose feature importance.")

    if "selection" in model.named_steps:
        selected_mask = model.named_steps["selection"].get_support()
        transformed_names = transformed_names[selected_mask]
    if len(values) != len(transformed_names):
        raise ValueError("Model importance and transformed feature sizes do not match.")

    rows = []
    for transformed_name, importance in zip(transformed_names, values):
        encoded_name = transformed_name.split("__", maxsplit=1)[-1]
        source_feature = next(
            (
                feature
                for feature in sorted(feature_names, key=len, reverse=True)
                if encoded_name == feature or encoded_name.startswith(f"{feature}_")
            ),
            encoded_name,
        )
        rows.append({"Variable": source_feature, "Importancia": float(importance)})
    return (
        pd.DataFrame(rows)
        .groupby("Variable", as_index=False)["Importancia"]
        .sum()
        .sort_values("Importancia", ascending=False)
        .reset_index(drop=True)
    )


def fit_experiment(
    data: pd.DataFrame,
    selected_features: list[str] | tuple[str, ...] | None = None,
    random_state: int = RANDOM_STATE,
    test_size: float = TEST_SIZE,
    validation_size: float = VALIDATION_SIZE,
    n_jobs: int = 2,
) -> ExperimentResult:
    """Fit three baseline and three refined models using train/validation/test splits."""
    splits = split_data(
        data,
        selected_features=selected_features,
        test_size=test_size,
        validation_size=validation_size,
        random_state=random_state,
    )
    selected = splits.x_train.columns.tolist()
    models: dict[str, Pipeline] = {}
    search_rows = []
    for model_name in MODEL_LABELS:
        baseline_name = _model_key(1, model_name)
        baseline = _pipeline(
            model_name, splits.x_train, iteration=1, random_state=random_state
        )
        baseline.fit(splits.x_train, splits.y_train)
        models[baseline_name] = baseline

        refined_name = _model_key(2, model_name)
        search = GridSearchCV(
            estimator=_pipeline(
                model_name, splits.x_train, iteration=2, random_state=random_state
            ),
            param_grid=_search_grid(model_name),
            scoring="f1",
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state),
            n_jobs=n_jobs,
            refit=True,
            return_train_score=True,
            error_score="raise",
        )
        search.fit(splits.x_train, splits.y_train)
        models[refined_name] = search.best_estimator_
        search_rows.append(
            {
                "Iteracion / modelo": refined_name,
                "F1 validacion cruzada": search.best_score_,
                "Mejores parametros": json.dumps(
                    search.best_params_, ensure_ascii=False
                ),
            }
        )

    validation_metrics, _ = _evaluate(
        models, splits.x_validation, splits.y_validation
    )
    test_metrics, test_predictions = _evaluate(models, splits.x_test, splits.y_test)
    refined_names = [
        name for name in validation_metrics.index if name.startswith("Iteracion 2")
    ]
    best_model_name = (
        validation_metrics.loc[refined_names]
        .sort_values(["F1", "Recall", "ROC-AUC"], ascending=False)
        .index[0]
    )
    feature_importance = _raw_feature_importance(models[best_model_name], selected)
    feature_ranking = rank_features(splits.x_train, splits.y_train)
    return ExperimentResult(
        models=models,
        splits=splits,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        test_predictions=test_predictions,
        feature_importance=feature_importance,
        feature_ranking=feature_ranking,
        search_results=pd.DataFrame(search_rows).set_index("Iteracion / modelo"),
        best_model_name=best_model_name,
        selected_features=selected,
        random_state=random_state,
        test_size=test_size,
        validation_size=validation_size,
    )


def save_experiment(
    result: ExperimentResult, output_dir: str | Path = DEFAULT_MODEL_DIR
) -> dict[str, Path]:
    """Persist all six fitted pipelines and reproducible evaluation summaries."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    model_files = {}
    for name, model in result.models.items():
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        path = directory / f"{slug}.joblib"
        joblib.dump(model, path)
        written[name] = path
        model_files[name] = path.name

    result.validation_metrics.to_csv(directory / "metricas_validacion.csv")
    result.test_metrics.to_csv(directory / "metricas_generalizacion.csv")
    result.search_results.to_csv(directory / "busqueda_hiperparametros.csv")
    result.feature_importance.to_csv(
        directory / "importancia_variables_modelo_seleccionado.csv", index=False
    )
    result.feature_ranking.to_csv(
        directory / "ranking_variables_entrenamiento.csv", index=False
    )
    result.test_predictions.to_csv(directory / "predicciones_generalizacion.csv")
    metadata = {
        "dataset": str(DEFAULT_DATA_PATH.relative_to(PROJECT_ROOT)),
        "dataset_sha256": hashlib.sha256(DEFAULT_DATA_PATH.read_bytes()).hexdigest(),
        "target": "y",
        "positive_class": "yes",
        "selected_features": result.selected_features,
        "excluded_post_contact_features": [
            "duration",
            "campaign",
            "day",
            "month",
            "contact",
        ],
        "train_fraction": 1 - result.test_size - result.validation_size,
        "validation_fraction": result.validation_size,
        "test_fraction": result.test_size,
        "random_state": result.random_state,
        "selection_metric": "F1 on validation; ties resolved by Recall and ROC-AUC",
        "selected_model": result.best_model_name,
        "models": model_files,
    }
    metadata_path = directory / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    written["metadata"] = metadata_path
    return written


def train_and_save(
    data_path: str | Path = DEFAULT_DATA_PATH,
    output_dir: str | Path = DEFAULT_MODEL_DIR,
    random_state: int = RANDOM_STATE,
    n_jobs: int = 2,
) -> ExperimentResult:
    data = load_bank_data(data_path)
    result = fit_experiment(data, random_state=random_state, n_jobs=n_jobs)
    save_experiment(result, output_dir)
    return result


def load_saved_models(
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> tuple[dict[str, Pipeline], dict[str, Any]]:
    directory = Path(model_dir)
    metadata_path = directory / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Model metadata was not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    models = {}
    for name, filename in metadata["models"].items():
        model_path = directory / filename
        if not model_path.is_file():
            raise FileNotFoundError(f"Serialized model was not found: {model_path}")
        models[name] = joblib.load(model_path)
    return models, metadata
