from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix, roc_curve


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.etl import (  # noqa: E402
    DEFAULT_DATA_PATH,
    POST_CONTACT_FEATURES,
    load_bank_data,
    split_data,
)
from src.modeling import (  # noqa: E402
    DEFAULT_MODEL_DIR,
    RANDOM_STATE,
    fit_experiment,
    load_saved_models,
)


st.set_page_config(
    page_title="Prediccion de depositos bancarios",
    page_icon="📊",
    layout="wide",
)


@st.cache_data
def load_data() -> pd.DataFrame:
    return load_bank_data(DEFAULT_DATA_PATH)


def _dataset_fingerprint() -> str:
    return hashlib.sha256(DEFAULT_DATA_PATH.read_bytes()).hexdigest()


@st.cache_resource(show_spinner=False)
def load_or_fit_experiment(
    selected_features: tuple[str, ...], random_state: int
) -> dict[str, Any]:
    data = load_data()
    metadata_path = DEFAULT_MODEL_DIR / "metadata.json"
    expected_features = list(selected_features)

    if metadata_path.is_file():
        models, metadata = load_saved_models(DEFAULT_MODEL_DIR)
        artifacts_match = (
            metadata.get("selected_features") == expected_features
            and metadata.get("random_state") == random_state
            and metadata.get("dataset_sha256") == _dataset_fingerprint()
        )
        if artifacts_match:
            splits = split_data(
                data,
                selected_features=selected_features,
                random_state=random_state,
            )
            validation_metrics = pd.read_csv(
                DEFAULT_MODEL_DIR / "metricas_validacion.csv", index_col=0
            )
            test_metrics = pd.read_csv(
                DEFAULT_MODEL_DIR / "metricas_generalizacion.csv", index_col=0
            )
            feature_importance = pd.read_csv(
                DEFAULT_MODEL_DIR / "importancia_variables_modelo_seleccionado.csv"
            )
            feature_ranking = pd.read_csv(
                DEFAULT_MODEL_DIR / "ranking_variables_entrenamiento.csv"
            )
            return {
                "using_saved_models": True,
                "models": models,
                "splits": splits,
                "validation_metrics": validation_metrics,
                "test_metrics": test_metrics,
                "feature_importance": feature_importance,
                "feature_ranking": feature_ranking,
                "best_model_name": metadata["selected_model"],
            }

    experiment = fit_experiment(
        data,
        selected_features=selected_features,
        random_state=random_state,
    )
    return {
        "using_saved_models": False,
        "models": experiment.models,
        "splits": experiment.splits,
        "validation_metrics": experiment.validation_metrics,
        "test_metrics": experiment.test_metrics,
        "feature_importance": experiment.feature_importance,
        "feature_ranking": experiment.feature_ranking,
        "best_model_name": experiment.best_model_name,
    }


def main() -> None:
    st.title("Suscripcion a depositos bancarios")
    st.write(
        "Proyecto CRISP-DM con dos iteraciones: tres algoritmos base y sus "
        "versiones refinadas. Las predicciones se comparan sobre los mismos "
        "datos de generalizacion."
    )
    st.info(
        "La variable `y` indica si el cliente suscribio un deposito a plazo "
        "(`yes` / `no`). Los resultados apoyan el analisis y no sustituyen "
        "la decision comercial."
    )

    try:
        data = load_data()
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        st.error(f"No se pudo cargar el conjunto de datos: {error}")
        st.stop()

    eligible_features = [
        feature
        for feature in data.columns
        if feature != "y" and feature not in POST_CONTACT_FEATURES
    ]
    default_features = eligible_features
    with st.sidebar:
        st.header("Variables y prediccion")
        ranked = pd.read_csv(DEFAULT_MODEL_DIR / "ranking_variables_entrenamiento.csv") \
            if (DEFAULT_MODEL_DIR / "ranking_variables_entrenamiento.csv").is_file() \
            else pd.DataFrame({"Variable": eligible_features})
        recommended = [
            feature for feature in ranked.get("Variable", []) if feature in eligible_features
        ][:8]
        st.caption(
            "Selecciona las variables disponibles antes de llamar. Los cambios "
            "en la seleccion ajustan los seis modelos sobre las mismas particiones."
        )
        selected_features = st.multiselect(
            "Variables predictoras",
            options=eligible_features,
            default=default_features,
            help=f"Variables de mayor relevancia en el entrenamiento: {', '.join(recommended)}",
        )
        threshold = st.slider(
            "Umbral para clasificar como suscripcion",
            min_value=0.10,
            max_value=0.90,
            value=0.50,
            step=0.05,
            help="El umbral cambia la clase predicha, no las probabilidades ni el ROC-AUC.",
        )

    if not selected_features:
        st.warning("Selecciona al menos una variable para entrenar los modelos.")
        st.stop()

    selected_tuple = tuple(selected_features)
    with st.spinner("Cargando o ajustando los modelos para la seleccion actual..."):
        try:
            experiment = load_or_fit_experiment(selected_tuple, RANDOM_STATE)
        except (FileNotFoundError, ValueError, TypeError) as error:
            st.error(f"No se pudo preparar el experimento: {error}")
            st.stop()

    models = experiment["models"]
    splits = experiment["splits"]
    st.caption(
        f"Registros de entrenamiento: {len(splits.y_train):,} · "
        f"validacion: {len(splits.y_validation):,} · "
        f"generalizacion: {len(splits.y_test):,} · "
        f"predictores: {len(selected_features)}"
    )
    if experiment["using_saved_models"]:
        st.success("Se cargaron los seis modelos persistidos en la carpeta Modelos.")
    else:
        st.warning(
            "La seleccion actual requiere un ajuste exploratorio de los modelos; "
            "los artefactos guardados no se modifican."
        )

    st.subheader("Seleccion de variables")
    st.caption(
        "El ranking se calcula solo con entrenamiento mediante informacion mutua. "
        "La seleccion refinada de la Iteracion 2 se aprende dentro de la validacion cruzada."
    )
    importance = experiment["feature_importance"]
    if not importance.empty:
        st.bar_chart(
            importance.set_index("Variable")["Importancia"].head(12),
            height=300,
        )
    else:
        st.info("No se pudo calcular la importancia para la seleccion actual.")
    st.dataframe(experiment["feature_ranking"].head(10), width="stretch")

    st.subheader("Comparacion de iteraciones en datos no vistos")
    validation_tab, generalization_tab = st.tabs(
        ["Validacion", "Generalizacion (test ciego)"]
    )
    with validation_tab:
        validation_metrics = experiment["validation_metrics"]
        st.dataframe(validation_metrics.style.format("{:.3f}"), width="stretch")
        st.bar_chart(validation_metrics[["F1", "ROC-AUC", "PR-AUC"]], height=300)
    with generalization_tab:
        test_metrics = experiment["test_metrics"]
        st.dataframe(test_metrics.style.format("{:.3f}"), width="stretch")
        st.bar_chart(test_metrics[["F1", "ROC-AUC", "PR-AUC"]], height=300)
        st.caption(
            "El conjunto de test se mantiene separado del entrenamiento, la "
            "seleccion de variables y la busqueda de hiperparametros."
        )

    st.subheader("Curvas ROC en generalizacion")
    false_positive_grid = np.linspace(0, 1, 101)
    roc_curves = {}
    for name, model in models.items():
        probabilities = model.predict_proba(splits.x_test)[:, 1]
        false_positive, true_positive, _ = roc_curve(
            splits.y_test, probabilities
        )
        roc_curves[name] = np.interp(
            false_positive_grid, false_positive, true_positive
        )
    roc_frame = pd.DataFrame(roc_curves, index=false_positive_grid)
    roc_frame.index.name = "Tasa de falsos positivos"
    st.line_chart(roc_frame, height=300)

    st.subheader("Consulta en tiempo real")
    st.write(
        "Ingresa un cliente y compara la probabilidad estimada por cada "
        "algoritmo en las dos iteraciones."
    )
    input_values: dict[str, Any] = {}
    with st.form("prediction_form"):
        columns = st.columns(2)
        for index, feature in enumerate(selected_features):
            series = data[feature]
            with columns[index % 2]:
                if pd.api.types.is_numeric_dtype(series):
                    input_values[feature] = st.number_input(
                        feature,
                        value=float(series.median()),
                        help=f"Rango observado: {series.min():g} a {series.max():g}.",
                    )
                else:
                    options = sorted(series.dropna().astype(str).unique().tolist())
                    mode_values = series.mode()
                    default = str(mode_values.iloc[0]) if not mode_values.empty else options[0]
                    input_values[feature] = st.selectbox(
                        feature,
                        options=options,
                        index=options.index(default) if default in options else 0,
                    )
        submitted = st.form_submit_button(
            "Consultar los seis modelos", type="primary", width="stretch"
        )

    if submitted:
        query = pd.DataFrame([input_values], columns=selected_features)
        prediction_rows = []
        for name, model in models.items():
            probability = float(model.predict_proba(query)[0, 1])
            prediction_rows.append(
                {
                    "Iteracion / modelo": name,
                    "Prediccion": "Si" if probability >= threshold else "No",
                    "Probabilidad de suscripcion": probability,
                }
            )
        st.dataframe(
            pd.DataFrame(prediction_rows).style.format(
                {"Probabilidad de suscripcion": "{:.1%}"}
            ),
            width="stretch",
        )

    with st.expander("Matriz de confusion en datos de generalizacion"):
        matrix_model = st.selectbox("Modelo", list(models))
        predicted = models[matrix_model].predict(splits.x_test)
        matrix = confusion_matrix(splits.y_test, predicted, labels=[0, 1])
        st.dataframe(
            pd.DataFrame(
                matrix,
                index=["Real: no", "Real: si"],
                columns=["Predicho: no", "Predicho: si"],
            ),
            width="stretch",
        )


if __name__ == "__main__":
    main()
