from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


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


PRIMARY = "#1F4E79"
ACCENT = "#F58518"
ITERATION_COLORS = {"Iteración 1": "#9DB4CE", "Iteración 2": ACCENT}
METRIC_HELP = {
    "Accuracy": "Proporción total de aciertos. Engañosa con clases desbalanceadas.",
    "Precision": "De los clientes que el modelo marca como 'sí', cuántos realmente suscriben.",
    "Recall": "De los clientes que sí suscriben, cuántos detecta el modelo.",
    "F1": "Equilibrio entre Precision y Recall.",
    "ROC-AUC": "Capacidad de ordenar clientes de mayor a menor propensión.",
    "PR-AUC": "Calidad de la clasificación centrada en la clase minoritaria.",
}

st.set_page_config(
    page_title="Depósitos bancarios | CRISP-DM",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
    .block-container {{padding-top: 1.6rem; max-width: 1280px;}}
    .hero {{
        background: linear-gradient(120deg, {PRIMARY} 0%, #2E75B6 100%);
        color: white; padding: 1.6rem 2rem; border-radius: 16px; margin-bottom: 1rem;
    }}
    .hero h1 {{color: white; margin: 0; font-size: 1.9rem;}}
    .hero p {{color: #E3EEF8; margin: .35rem 0 0 0; font-size: 1rem;}}
    .card {{
        border: 1px solid rgba(128,128,128,.25); border-radius: 14px;
        padding: 1rem 1.1rem; height: 100%;
    }}
    .card h4 {{margin: 0 0 .25rem 0;}}
    .card small {{opacity: .75;}}
    .pill {{
        display: inline-block; padding: .15rem .7rem; border-radius: 999px;
        font-weight: 600; font-size: .85rem; color: white;
    }}
    .pill-yes {{background: #2E9E5B;}}
    .pill-no {{background: #8A94A6;}}
    .bar-wrap {{background: rgba(128,128,128,.2); border-radius: 8px; height: 12px; overflow: hidden;}}
    .bar-fill {{height: 12px; border-radius: 8px;}}
    div[data-testid="stMetric"] {{
        border: 1px solid rgba(128,128,128,.25); border-radius: 14px; padding: .8rem 1rem;
    }}
    .stTabs [data-baseweb="tab"] {{font-size: 1rem; padding: .6rem 1rem;}}
    </style>
    """,
    unsafe_allow_html=True,
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
            return {
                "using_saved_models": True,
                "models": models,
                "splits": splits,
                "validation_metrics": pd.read_csv(
                    DEFAULT_MODEL_DIR / "metricas_validacion.csv", index_col=0
                ),
                "test_metrics": pd.read_csv(
                    DEFAULT_MODEL_DIR / "metricas_generalizacion.csv", index_col=0
                ),
                "feature_importance": pd.read_csv(
                    DEFAULT_MODEL_DIR / "importancia_variables_modelo_seleccionado.csv"
                ),
                "feature_ranking": pd.read_csv(
                    DEFAULT_MODEL_DIR / "ranking_variables_entrenamiento.csv"
                ),
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


def split_name(model_name: str) -> tuple[str, str]:
    iteration, algorithm = (part.strip() for part in model_name.split("|", 1))
    return iteration.replace("Iteracion", "Iteración"), algorithm


def metrics_long(metrics: pd.DataFrame) -> pd.DataFrame:
    frame = metrics.reset_index().rename(columns={metrics.index.name or "index": "Modelo"})
    frame[["Iteración", "Algoritmo"]] = frame["Modelo"].apply(
        lambda value: pd.Series(split_name(value))
    )
    return frame.melt(
        id_vars=["Modelo", "Iteración", "Algoritmo"],
        var_name="Métrica",
        value_name="Valor",
    )


def metrics_at_threshold(
    model: Any, x: pd.DataFrame, y: pd.Series, threshold: float
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    probabilities = model.predict_proba(x)[:, 1]
    predicted = (probabilities >= threshold).astype(int)
    scores = {
        "Accuracy": accuracy_score(y, predicted),
        "Precision": precision_score(y, predicted, zero_division=0),
        "Recall": recall_score(y, predicted, zero_division=0),
        "F1": f1_score(y, predicted, zero_division=0),
        "ROC-AUC": roc_auc_score(y, probabilities),
        "PR-AUC": average_precision_score(y, probabilities),
    }
    return scores, predicted, probabilities


def probability_card(model_name: str, probability: float, threshold: float) -> str:
    iteration, algorithm = split_name(model_name)
    is_yes = probability >= threshold
    color = ITERATION_COLORS[iteration] if iteration == "Iteración 1" else ACCENT
    pill = (
        '<span class="pill pill-yes">Suscribiría</span>'
        if is_yes
        else '<span class="pill pill-no">No suscribiría</span>'
    )
    return f"""
    <div class="card">
      <small>{iteration}</small>
      <h4>{algorithm}</h4>
      <div style="font-size:2rem;font-weight:700;color:{PRIMARY if is_yes else '#6B7280'}">
        {probability:.1%}
      </div>
      <div class="bar-wrap"><div class="bar-fill" style="width:{probability * 100:.1f}%;background:{color}"></div></div>
      <div style="margin-top:.6rem">{pill}</div>
    </div>
    """


def build_input_form(
    data: pd.DataFrame, features: list[str]
) -> tuple[dict[str, Any], bool]:
    values: dict[str, Any] = {}
    with st.form("prediction_form", border=True):
        columns = st.columns(3)
        for index, feature in enumerate(features):
            series = data[feature]
            with columns[index % 3]:
                if pd.api.types.is_numeric_dtype(series):
                    values[feature] = st.number_input(
                        feature,
                        value=float(series.median()),
                        help=f"Rango observado: {series.min():g} a {series.max():g}.",
                    )
                else:
                    options = sorted(series.dropna().astype(str).unique().tolist())
                    mode_values = series.mode()
                    default = str(mode_values.iloc[0]) if not mode_values.empty else options[0]
                    values[feature] = st.selectbox(
                        feature,
                        options=options,
                        index=options.index(default) if default in options else 0,
                    )
        submitted = st.form_submit_button(
            "🔮 Consultar los seis modelos", type="primary", width="stretch"
        )
    return values, submitted


def render_header() -> None:
    st.markdown(
        """
        <div class="hero">
          <h1>🏦 Predicción de suscripción a depósitos a plazo</h1>
          <p>Metodología CRISP-DM · 2 iteraciones · 3 algoritmos por iteración ·
          evaluación con datos de generalización</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def tab_summary(experiment: dict[str, Any], data: pd.DataFrame) -> None:
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("¿Qué hace esta aplicación?")
        st.write(
            "Estima la probabilidad de que un cliente suscriba un depósito a plazo "
            "usando solo información disponible **antes** de contactarlo, para "
            "priorizar la gestión comercial."
        )
        it1, it2 = st.columns(2)
        it1.markdown(
            """<div class="card"><small>Iteración 1</small><h4>Modelos base</h4>
            Regresión Logística, Árbol de Decisión y Random Forest con
            preprocesamiento estándar, sin balanceo ni ajuste.</div>""",
            unsafe_allow_html=True,
        )
        it2.markdown(
            """<div class="card"><small>Iteración 2</small><h4>Modelos refinados</h4>
            Selección de características, pesos de clase para el desbalance y
            <i>GridSearchCV</i> sobre los mismos tres algoritmos.</div>""",
            unsafe_allow_html=True,
        )
        st.caption(
            "Variables excluidas por describir el contacto actual (evitan fuga de "
            f"información): {', '.join(f'`{name}`' for name in POST_CONTACT_FEATURES)}."
        )
        st.info(
            "Los resultados apoyan el análisis y no sustituyen la decisión comercial."
        )
    with right:
        st.subheader("Distribución del objetivo")
        counts = data["y"].map({0: "No suscribe", 1: "Suscribe"}).value_counts().reset_index()
        counts.columns = ["Clase", "Clientes"]
        chart = (
            alt.Chart(counts)
            .mark_arc(innerRadius=60)
            .encode(
                theta="Clientes:Q",
                color=alt.Color(
                    "Clase:N",
                    scale=alt.Scale(domain=["No suscribe", "Suscribe"], range=["#9DB4CE", ACCENT]),
                ),
                tooltip=["Clase", alt.Tooltip("Clientes:Q", format=",")],
            )
            .properties(height=260)
        )
        st.altair_chart(chart, width="stretch")
        st.caption("La clase *Suscribe* es minoritaria (~12 %): por eso se priorizan Recall, F1 y ROC-AUC.")


def tab_variables(experiment: dict[str, Any], selected: list[str]) -> None:
    st.subheader("Importancia y selección de variables")
    st.caption(
        "Elige las variables en el panel lateral. El ranking se calcula solo con "
        "entrenamiento (información mutua); la selección refinada de la Iteración 2 "
        "se aprende dentro de la validación cruzada."
    )
    left, right = st.columns(2, gap="large")
    importance = experiment["feature_importance"]
    with left:
        st.markdown("**Importancia en el modelo seleccionado**")
        if importance.empty:
            st.info("No se pudo calcular la importancia para la selección actual.")
        else:
            top = importance.head(12)
            st.altair_chart(
                alt.Chart(top)
                .mark_bar(color=PRIMARY, cornerRadiusEnd=4)
                .encode(
                    x=alt.X("Importancia:Q", title="Importancia"),
                    y=alt.Y("Variable:N", sort="-x", title=None),
                    tooltip=["Variable", alt.Tooltip("Importancia:Q", format=".4f")],
                )
                .properties(height=340),
                width="stretch",
            )
    with right:
        st.markdown("**Ranking por información mutua (entrenamiento)**")
        ranking = experiment["feature_ranking"].copy()
        ranking["En uso"] = ranking["Variable"].isin(selected)
        st.dataframe(
            ranking,
            width="stretch",
            hide_index=True,
            height=340,
            column_config={
                "Importancia": st.column_config.ProgressColumn(
                    "Importancia",
                    min_value=0.0,
                    max_value=float(ranking["Importancia"].max()) or 1.0,
                    format="%.4f",
                ),
                "En uso": st.column_config.CheckboxColumn("En uso"),
            },
        )


def tab_realtime(
    experiment: dict[str, Any], data: pd.DataFrame, selected: list[str], threshold: float
) -> None:
    models = experiment["models"]
    single, batch = st.tabs(["Cliente individual", "Cargar archivo CSV"])

    with single:
        st.subheader("Consulta en tiempo real")
        st.caption("Ingresa los datos del cliente y compara la probabilidad de cada modelo.")
        values, submitted = build_input_form(data, selected)
        if submitted:
            query = pd.DataFrame([values], columns=selected)
            probabilities = {
                name: float(model.predict_proba(query)[0, 1]) for name, model in models.items()
            }
            consensus = sum(p >= threshold for p in probabilities.values())
            average = float(np.mean(list(probabilities.values())))
            summary = st.columns(3)
            summary[0].metric("Probabilidad promedio", f"{average:.1%}")
            summary[1].metric("Modelos que dicen 'sí'", f"{consensus} de {len(probabilities)}")
            summary[2].metric("Umbral aplicado", f"{threshold:.0%}")
            for iteration in ("Iteracion 1", "Iteracion 2"):
                st.markdown(f"##### {iteration.replace('Iteracion', 'Iteración')}")
                row = st.columns(3)
                names = [n for n in probabilities if n.startswith(iteration)]
                for column, name in zip(row, names):
                    column.markdown(
                        probability_card(name, probabilities[name], threshold),
                        unsafe_allow_html=True,
                    )
        else:
            st.info("Completa el formulario y presiona **Consultar los seis modelos**.")

    with batch:
        st.subheader("Predicción por lotes")
        st.caption(
            "Sube un CSV (separador `;` o `,`) con las columnas: "
            + ", ".join(f"`{c}`" for c in selected)
        )
        model_name = st.selectbox("Modelo a aplicar", list(models), key="batch_model")
        uploaded = st.file_uploader("Archivo CSV", type=["csv"])
        if uploaded is not None:
            try:
                frame = pd.read_csv(uploaded, sep=None, engine="python")
                missing = [c for c in selected if c not in frame.columns]
                if missing:
                    st.error(f"Faltan columnas en el archivo: {', '.join(missing)}")
                else:
                    probabilities = models[model_name].predict_proba(frame[selected])[:, 1]
                    result = frame.copy()
                    result["probabilidad_suscripcion"] = probabilities
                    result["prediccion"] = np.where(probabilities >= threshold, "Sí", "No")
                    st.success(f"{len(result):,} clientes evaluados.")
                    st.dataframe(
                        result,
                        width="stretch",
                        hide_index=True,
                        column_config={
                            "probabilidad_suscripcion": st.column_config.ProgressColumn(
                                "Probabilidad", min_value=0.0, max_value=1.0, format="%.1f%%"
                            )
                        },
                    )
                    st.download_button(
                        "⬇️ Descargar resultados",
                        result.to_csv(index=False).encode("utf-8"),
                        file_name="predicciones.csv",
                        mime="text/csv",
                    )
            except Exception as error:  # noqa: BLE001
                st.error(f"No se pudo procesar el archivo: {error}")


def tab_generalization(experiment: dict[str, Any], threshold: float) -> None:
    models = experiment["models"]
    splits = experiment["splits"]
    st.subheader("Ejecución con datos de generalización")
    st.caption(
        "Datos que ningún modelo vio durante el entrenamiento, la selección de "
        "variables ni la búsqueda de hiperparámetros."
    )
    model_name = st.selectbox(
        "Modelo a evaluar",
        list(models),
        index=list(models).index(experiment["best_model_name"]),
    )
    scores, predicted, _ = metrics_at_threshold(
        models[model_name], splits.x_test, splits.y_test, threshold
    )
    columns = st.columns(6)
    for column, (metric, value) in zip(columns, scores.items()):
        column.metric(metric, f"{value:.3f}", help=METRIC_HELP[metric])
    st.caption(f"Métricas calculadas con el umbral {threshold:.0%} sobre {len(splits.y_test):,} clientes.")

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Matriz de confusión**")
        matrix = confusion_matrix(splits.y_test, predicted, labels=[0, 1])
        cells = pd.DataFrame(
            [
                {"Real": "No suscribe", "Predicho": "No suscribe", "Clientes": matrix[0, 0]},
                {"Real": "No suscribe", "Predicho": "Suscribe", "Clientes": matrix[0, 1]},
                {"Real": "Suscribe", "Predicho": "No suscribe", "Clientes": matrix[1, 0]},
                {"Real": "Suscribe", "Predicho": "Suscribe", "Clientes": matrix[1, 1]},
            ]
        )
        base = alt.Chart(cells).encode(
            x=alt.X("Predicho:N", title="Predicho"),
            y=alt.Y("Real:N", title="Real", sort=["No suscribe", "Suscribe"]),
        )
        heat = base.mark_rect(cornerRadius=6).encode(
            color=alt.Color("Clientes:Q", scale=alt.Scale(scheme="blues"), legend=None)
        )
        text = base.mark_text(fontSize=18, fontWeight="bold").encode(
            text=alt.Text("Clientes:Q", format=","),
            color=alt.condition(alt.datum.Clientes > matrix.max() / 2, alt.value("white"), alt.value("black")),
        )
        st.altair_chart((heat + text).properties(height=300), width="stretch")
    with right:
        st.markdown("**Distribución de probabilidades por clase real**")
        probabilities = models[model_name].predict_proba(splits.x_test)[:, 1]
        dist = pd.DataFrame(
            {
                "Probabilidad": probabilities,
                "Clase real": np.where(splits.y_test.to_numpy() == 1, "Suscribe", "No suscribe"),
            }
        )
        hist = (
            alt.Chart(dist)
            .mark_bar(opacity=0.75)
            .encode(
                x=alt.X("Probabilidad:Q", bin=alt.Bin(maxbins=30)),
                y=alt.Y("count():Q", title="Clientes", stack=None),
                color=alt.Color(
                    "Clase real:N",
                    scale=alt.Scale(domain=["No suscribe", "Suscribe"], range=["#9DB4CE", ACCENT]),
                ),
            )
            .properties(height=300)
        )
        rule = alt.Chart(pd.DataFrame({"x": [threshold]})).mark_rule(color="red", strokeDash=[6, 4]).encode(x="x:Q")
        st.altair_chart(hist + rule, width="stretch")
        st.caption("La línea roja es el umbral elegido en el panel lateral.")


def tab_comparison(experiment: dict[str, Any]) -> None:
    models = experiment["models"]
    splits = experiment["splits"]
    st.subheader("Comparación de desempeño por iteración")

    source = st.radio(
        "Conjunto de evaluación",
        ["Generalización (test ciego)", "Validación"],
        horizontal=True,
    )
    metrics = (
        experiment["test_metrics"] if source.startswith("Generalización") else experiment["validation_metrics"]
    )
    metric_choice = st.multiselect(
        "Métricas a graficar",
        list(METRIC_HELP),
        default=["Recall", "F1", "ROC-AUC"],
    )
    if metric_choice:
        long = metrics_long(metrics)
        long = long[long["Métrica"].isin(metric_choice)]
        chart = (
            alt.Chart(long)
            .mark_bar(cornerRadiusEnd=3)
            .encode(
                x=alt.X("Algoritmo:N", title=None, axis=alt.Axis(labelAngle=-40, labelLimit=140)),
                xOffset="Iteración:N",
                y=alt.Y("Valor:Q", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color(
                    "Iteración:N",
                    scale=alt.Scale(domain=list(ITERATION_COLORS), range=list(ITERATION_COLORS.values())),
                ),
                column=alt.Column("Métrica:N", title=None),
                tooltip=["Modelo", "Métrica", alt.Tooltip("Valor:Q", format=".3f")],
            )
            .properties(width=170, height=280)
        )
        st.altair_chart(chart)

    st.markdown("**Tabla resumen** (el mejor valor de cada métrica aparece resaltado)")
    st.dataframe(
        metrics.style.format("{:.3f}").highlight_max(axis=0, color="#CDEBD6"),
        width="stretch",
    )
    best = experiment["best_model_name"]
    st.success(
        f"Modelo recomendado (elegido con validación, sin usar el test): **{best}**."
    )

    st.markdown("**Curvas ROC en generalización**")
    grid = np.linspace(0, 1, 101)
    rows = []
    for name, model in models.items():
        fpr, tpr, _ = roc_curve(splits.y_test, model.predict_proba(splits.x_test)[:, 1])
        iteration, algorithm = split_name(name)
        for x, y in zip(grid, np.interp(grid, fpr, tpr)):
            rows.append({"FPR": x, "TPR": y, "Modelo": name, "Iteración": iteration, "Algoritmo": algorithm})
    roc = pd.DataFrame(rows)
    lines = (
        alt.Chart(roc)
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X("FPR:Q", title="Tasa de falsos positivos"),
            y=alt.Y("TPR:Q", title="Tasa de verdaderos positivos"),
            color=alt.Color("Algoritmo:N"),
            strokeDash=alt.StrokeDash("Iteración:N"),
            tooltip=["Modelo", alt.Tooltip("FPR:Q", format=".2f"), alt.Tooltip("TPR:Q", format=".2f")],
        )
        .properties(height=380)
    )
    diagonal = (
        alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]}))
        .mark_line(color="gray", strokeDash=[4, 4])
        .encode(x="x:Q", y="y:Q")
    )
    st.altair_chart(lines + diagonal, width="stretch")


def main() -> None:
    render_header()

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
    ranking_path = DEFAULT_MODEL_DIR / "ranking_variables_entrenamiento.csv"
    recommended = (
        [v for v in pd.read_csv(ranking_path)["Variable"] if v in eligible_features][:5]
        if ranking_path.is_file()
        else eligible_features[:5]
    )

    with st.sidebar:
        st.header("⚙️ Configuración")
        selected_features = st.multiselect(
            "Variables predictoras",
            options=eligible_features,
            default=eligible_features,
            help="Variables disponibles antes de contactar al cliente.",
        )
        st.caption(f"Más relevantes en entrenamiento: {', '.join(recommended)}")
        threshold = st.slider(
            "Umbral de decisión",
            min_value=0.10,
            max_value=0.90,
            value=0.50,
            step=0.05,
            help="Probabilidad mínima para clasificar como 'suscribe'. No cambia el ROC-AUC.",
        )
        st.divider()
        st.caption(
            "Con la selección completa se usan los modelos guardados en `Modelos/`. "
            "Si cambias las variables, se entrena una configuración exploratoria en "
            "caché sin sobrescribir los artefactos."
        )

    if not selected_features:
        st.warning("Selecciona al menos una variable en el panel lateral.")
        st.stop()

    with st.spinner("Cargando o ajustando los modelos para la selección actual..."):
        try:
            experiment = load_or_fit_experiment(tuple(selected_features), RANDOM_STATE)
        except (FileNotFoundError, ValueError, TypeError) as error:
            st.error(f"No se pudo preparar el experimento: {error}")
            st.stop()

    splits = experiment["splits"]
    best = experiment["best_model_name"]
    best_test = experiment["test_metrics"].loc[best]

    with st.sidebar:
        if experiment["using_saved_models"]:
            st.success("Modelos persistidos cargados", icon="✅")
        else:
            st.warning("Ajuste exploratorio (no se guardó)", icon="🧪")

    best_iteration, best_algorithm = split_name(best)
    st.markdown(f"**Modelo recomendado:** {best_iteration} · {best_algorithm}")
    kpis = st.columns(4)
    kpis[0].metric("F1 (test)", f"{best_test['F1']:.3f}", help=METRIC_HELP["F1"])
    kpis[1].metric("Recall (test)", f"{best_test['Recall']:.3f}", help=METRIC_HELP["Recall"])
    kpis[2].metric("ROC-AUC (test)", f"{best_test['ROC-AUC']:.3f}", help=METRIC_HELP["ROC-AUC"])
    kpis[3].metric(
        "Clientes en el dataset",
        f"{len(splits.y_train) + len(splits.y_validation) + len(splits.y_test):,}",
        help=f"{len(selected_features)} variables predictoras en uso.",
    )

    tabs = st.tabs(
        [
            "🏠 Resumen",
            "🧩 Variables",
            "🔮 Consulta en tiempo real",
            "🧪 Generalización",
            "📈 Comparación",
        ]
    )
    with tabs[0]:
        tab_summary(experiment, load_data())
    with tabs[1]:
        tab_variables(experiment, selected_features)
    with tabs[2]:
        tab_realtime(experiment, data, selected_features, threshold)
    with tabs[3]:
        tab_generalization(experiment, threshold)
    with tabs[4]:
        tab_comparison(experiment)


if __name__ == "__main__":
    main()
