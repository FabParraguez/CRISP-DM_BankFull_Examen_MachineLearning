from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path(__file__).resolve().parent.parent / "Data" / "bank-full.csv"
TARGET = "y"
MODEL_SPECS = {
    "Iteración 1 · Regresión logística": LogisticRegression(
        class_weight="balanced", max_iter=1200, random_state=42
    ),
    "Iteración 2 · Random Forest": RandomForestClassifier(
        n_estimators=250,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    ),
    "Iteración 3 · Extra Trees": ExtraTreesClassifier(
        n_estimators=250,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    ),
}


st.set_page_config(page_title="Predicción de depósitos bancarios", page_icon="📊", layout="wide")


@st.cache_data
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el conjunto de datos requerido: {DATA_PATH}"
        )

    data = pd.read_csv(DATA_PATH, sep=";")
    if TARGET not in data.columns:
        raise ValueError(f"El archivo debe incluir la columna objetivo '{TARGET}'.")

    data = data.dropna(subset=[TARGET]).copy()
    data[TARGET] = data[TARGET].map({"yes": 1, "no": 0})
    if data[TARGET].isna().any():
        raise ValueError("La columna 'y' solo debe contener los valores 'yes' y 'no'.")
    return data


def make_preprocessor(features: list[str], data: pd.DataFrame) -> ColumnTransformer:
    numeric_features = [
        feature for feature in features if pd.api.types.is_numeric_dtype(data[feature])
    ]
    categorical_features = [
        feature for feature in features if feature not in numeric_features
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def ranked_features(
    x_train: pd.DataFrame, y_train: pd.Series
) -> pd.DataFrame:
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
        random_state=42,
    )
    return (
        pd.DataFrame({"Variable": encoded.columns, "Importancia": scores})
        .sort_values("Importancia", ascending=False)
        .reset_index(drop=True)
    )


@st.cache_resource
def fit_models(
    features: tuple[str, ...], test_fraction: float, random_state: int
) -> tuple[dict[str, Any], pd.DataFrame, pd.Series, pd.DataFrame]:
    data = load_data()
    x = data.drop(columns=TARGET)
    y = data[TARGET].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_fraction,
        random_state=random_state,
        stratify=y,
    )
    importance = ranked_features(x_train, y_train)

    fitted: dict[str, Any] = {}
    for name, estimator in MODEL_SPECS.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", make_preprocessor(list(features), x_train)),
                ("classifier", estimator),
            ]
        )
        pipeline.fit(x_train[list(features)], y_train)
        fitted[name] = pipeline

    return fitted, x_test[list(features)], y_test, importance


def evaluate_models(
    models: dict[str, Any], x_test: pd.DataFrame, y_test: pd.Series
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    predictions: dict[str, Any] = {}
    for name, model in models.items():
        predicted = model.predict(x_test)
        probabilities = model.predict_proba(x_test)[:, 1]
        predictions[name] = (predicted, probabilities)
        rows.append(
            {
                "Iteración / modelo": name,
                "Exactitud": accuracy_score(y_test, predicted),
                "Precisión": precision_score(y_test, predicted, zero_division=0),
                "Recall": recall_score(y_test, predicted, zero_division=0),
                "F1": f1_score(y_test, predicted, zero_division=0),
                "ROC AUC": roc_auc_score(y_test, probabilities),
                "PR AUC": average_precision_score(y_test, probabilities),
            }
        )
    return pd.DataFrame(rows).set_index("Iteración / modelo"), predictions


def main() -> None:
    st.title("Predicción de suscripción a depósitos bancarios")
    st.write(
        "Compara tres iteraciones de clasificación y consulta sus predicciones "
        "individuales. La evaluación se realiza sobre un conjunto de generalización "
        "separado antes del entrenamiento."
    )
    st.info(
        "No hay modelos previamente serializados en el proyecto; la aplicación "
        "entrena las tres iteraciones al iniciar o al cambiar la configuración."
    )

    try:
        data = load_data()
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        st.error(f"No se pudo cargar el conjunto de datos: {error}")
        st.stop()

    all_features = [column for column in data.columns if column != TARGET]
    default_features = [
        "age",
        "job",
        "marital",
        "education",
        "balance",
        "housing",
        "loan",
        "contact",
        "month",
        "duration",
        "campaign",
        "pdays",
        "previous",
        "poutcome",
    ]
    default_features = [feature for feature in default_features if feature in all_features]

    with st.sidebar:
        st.header("Configuración del experimento")
        test_fraction = st.slider(
            "Proporción para generalización",
            min_value=15,
            max_value=40,
            value=20,
            step=5,
            format="%d%%",
        )
        test_fraction /= 100
        random_state = st.number_input(
            "Semilla de aleatoriedad", min_value=0, max_value=99999, value=42
        )
        selected_features = st.multiselect(
            "Variables para construir los modelos",
            options=all_features,
            default=default_features,
            help="Las tres iteraciones usarán exactamente las variables seleccionadas.",
        )

    if not selected_features:
        st.warning("Selecciona al menos una variable para entrenar los modelos.")
        st.stop()

    with st.spinner("Entrenando los modelos y evaluando datos no vistos..."):
        models, x_test, y_test, importance = fit_models(
            tuple(selected_features), float(test_fraction), int(random_state)
        )
        metrics, _ = evaluate_models(models, x_test, y_test)

    st.subheader("Variables más informativas")
    st.caption(
        "Importancia por información mutua calculada únicamente con el subconjunto "
        "de entrenamiento. Selecciona las variables en la barra lateral."
    )
    importance_view = importance.set_index("Variable")
    st.bar_chart(importance_view["Importancia"].head(12), height=280)

    st.subheader("Comparación en datos de generalización")
    st.caption(
        f"Entrenamiento: {len(data) - len(y_test):,} registros · "
        f"Generalización: {len(y_test):,} registros · "
        f"Variables usadas: {len(selected_features)}"
    )
    st.dataframe(
        metrics.style.format("{:.3f}", subset=metrics.columns),
        width="stretch",
    )
    st.bar_chart(metrics[["F1", "ROC AUC", "PR AUC"]], height=300)
    st.caption(
        "PR AUC resume el equilibrio precisión-recall y es útil cuando la clase "
        "positiva está menos representada. Todas las métricas se calculan sobre "
        "el mismo conjunto reservado, que no participa en el ajuste."
    )

    st.subheader("Consulta en tiempo real")
    st.write(
        "Introduce los datos de un cliente. Los tres modelos devolverán su "
        "probabilidad de suscripción."
    )
    st.warning(
        "La variable `duration` representa la duración de la llamada y solo se "
        "conoce después del contacto. Para priorizar clientes antes de llamar, "
        "exclúyela en la selección de variables."
    )
    input_values: dict[str, Any] = {}
    with st.form("prediction_form"):
        columns = st.columns(2)
        for index, feature in enumerate(all_features):
            column = columns[index % 2]
            series = data[feature]
            with column:
                if pd.api.types.is_numeric_dtype(series):
                    value = float(series.median())
                    input_values[feature] = st.number_input(
                        feature,
                        value=value,
                        help=f"Rango observado: {series.min():g} a {series.max():g}.",
                    )
                else:
                    options = sorted(series.dropna().astype(str).unique().tolist())
                    default_value = str(series.mode().iloc[0]) if not series.mode().empty else options[0]
                    default_index = options.index(default_value) if default_value in options else 0
                    input_values[feature] = st.selectbox(
                        feature, options=options, index=default_index
                    )
        submitted = st.form_submit_button(
            "Consultar los tres modelos", type="primary", width="stretch"
        )

    if submitted:
        query = pd.DataFrame(
            [{feature: input_values[feature] for feature in selected_features}]
        )
        results = []
        for name, model in models.items():
            probability = float(model.predict_proba(query)[:, 1][0])
            prediction = "Sí" if probability >= 0.5 else "No"
            results.append(
                {
                    "Iteración / modelo": name,
                    "Predicción": prediction,
                    "Probabilidad de suscripción": probability,
                }
            )
        st.dataframe(
            pd.DataFrame(results).style.format(
                {"Probabilidad de suscripción": "{:.1%}"}
            ),
            width="stretch",
        )

    with st.expander("Matriz de confusión por modelo"):
        matrix_model = st.selectbox("Modelo", list(models))
        predicted, _ = evaluate_models(models, x_test, y_test)[1][matrix_model]
        matrix = confusion_matrix(y_test, predicted, labels=[0, 1])
        st.dataframe(
            pd.DataFrame(
                matrix,
                index=["Real: No", "Real: Sí"],
                columns=["Predicho: No", "Predicho: Sí"],
            ),
            width="stretch",
        )


if __name__ == "__main__":
    main()
