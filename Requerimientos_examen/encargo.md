# Estructura del Proyecto: Clasificación Machine Learning & Streamlit App

**Metodología base:** CRISP-DM (Cross-Industry Standard Process for Data Mining)  
**Entregables principales:** Informe de iteraciones de modelos y Aplicación interactiva en Streamlit.

---

##  FASE 1: Entendimiento del Negocio (Business Understanding)
* [ ] **Definición del Objetivo:** Establecer el problema de negocio y el propósito del modelo de clasificación.
* [ ] **Criterios de Éxito:** Definir métricas clave de desempeño (Accuracy, Precision, Recall, F1-Score, ROC-AUC) alineadas a las necesidades del negocio.
* [ ] **Plan del Proyecto:** Identificar recursos, herramientas (Python, Streamlit, Scikit-learn, etc.) y planificar las fases de trabajo.

---

## FASE 2: Entendimiento de los Datos (Data Understanding)
* [ ] **Recolección de Datos:** Carga e inspección inicial del conjunto de datos (*Dataset*).
* [ ] **Exploración Inicial (EDA previo):** Identificación de tipos de variables, dimensiones y volumen del dataset.
* [ ] **Auditoría de Calidad:** Verificación de valores faltantes, registros duplicados, inconsistencias y detección preliminar de valores atípicos (*outliers*).

---

## FASE 3: Preparación de los Datos (Data Preparation)
* [ ] **Limpieza de Datos:** Tratamiento de nulos (imputación/eliminación), eliminación de duplicados y manejo de outliers.
* [ ] **Ingeniería de Características (Feature Engineering):**
  * Codificación de variables categóricas (*One-Hot Encoding*, *Label Encoding*).
  * Escalamiento / Normalización de variables numéricas (*StandardScaler*, *MinMaxScaler*).
* [ ] **Selección de Variables Imput:** Identificación de las variables más relevantes/importantes para el problema.
* [ ] **División del Dataset:**
  * Datos de Entrenamiento (*Train Set*).
  * Datos de Validación (*Validation Set*).
  * Datos de Generalización / Test Ciego (*Holdout / Test Set* sin ver durante el entrenamiento).

---

## FASE 4: Modelado e Iteraciones (Modeling)

Se deben desarrollar **al menos dos iteraciones** de entrenamiento probando **3 algoritmos/modelos distintos** en total para comparar su desempeño:

### 🔹 Iteración 1: Modelo Base (Baseline)
* [ ] Entrenar 3 algoritmos de clasificación iniciales (ej: *Regresión Logística*, *Árbol de Decisión*, *Random Forest*).
* [ ] Evaluar métricas iniciales sobre el conjunto de validación.

### 🔹 Iteración 2: Optimización y Ajuste
* [ ] Aplicar **Selección de Características Avanzada** (*Feature Selection*) o reducción de dimensionalidad.
* [ ] Ajuste de hiperparámetros (*GridSearchCV* / *RandomizedSearchCV*).
* [ ] Manejo de desbalance de clases (si aplica, con *SMOTE*, pesos de clase, etc.).
* [ ] Re-entrenar los 3 modelos refinados.

---

## FASE 5: Evaluación (Evaluation)
* [ ] **Comparativa de Iteraciones:** Contrastar las métricas obtenidas entre la Iteración 1 y la Iteración 2.
* [ ] **Evaluación con Datos de Generalización:** Probar los modelos con el conjunto de datos no visto en el entrenamiento (*Test Set*).
* [ ] **Selección del Mejor Modelo:** Justificar cuál es el mejor algoritmo basándose en las métricas de rendimiento y la capacidad de generalizar.
* [ ] **Revisión de Negocio:** Asegurar que el modelo seleccionado cumple con los objetivos planteados en la Fase 1.

---

## FASE 6: Despliegue en Streamlit (Deployment)

Desarrollo de una plataforma web interactiva con la librería **Streamlit** que contenga los siguientes módulos funcionales:

### 1.  Selección de Variables (Feature Selection)
* [ ] Módulo/Panel para visualizar y seleccionar interactivamente las variables más importantes utilizadas para la construcción de los modelos.

### 2. Consultas en Tiempo Real (Inferencia)
* [ ] Interfaz que permita ingresar o cargar nuevos datos y realizar **predicciones en tiempo real** consultando a los **3 modelos desarrollados** durante las iteraciones.

### 3.  Evaluación de Generalización
* [ ] Sección dedicada a ejecutar los modelos de clasificación utilizando **datos de generalización** (datos inéditos/test set).

### 4.  Dashboard de Comparación de Desempeño
* [ ] Visualizaciones interactivas (gráficos de barras, matrices de confusión, curvas ROC) que muestren el **puntaje de desempeño (Score/Métricas)** obtenido en cada iteración.
* [ ] Tabla comparativa resumen para evaluar el rendimiento de todos los modelos lado a lado.