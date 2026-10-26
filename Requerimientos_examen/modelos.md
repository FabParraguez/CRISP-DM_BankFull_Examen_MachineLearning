 # Base de Conocimiento Técnica: Machine Learning Supervisado y No Supervisado (MLY1101)

Este documento sirve como base de conocimiento técnico y metodológico en para que un agente de Inteligencia Artificial (o desarrollador) diseñe, implemente, evalúe y optimice modelos de Machine Learning utilizando el stack científico de Python (`scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`).

---

## MÓDULO 1: Fundamentos y Flujo de Trabajo

### 1.1 Metodología CRISP-DM
Todo desarrollo de Machine Learning debe seguir las etapas del estándar CRISP-DM:
1. **Comprensión del Negocio:** Definir objetivos y traducir el problema a un alcance analítico (Clasificación, Regresión, Clustering, etc.).
2. **Comprensión de los Datos:** Exploración inicial (EDA), detección de nulos, outliers y distribución de variables.
3. **Preparación de Datos:** Limpieza, imputación, codificación de categóricas (One-Hot / Label Encoding), escalamiento y selección de características.
4. **Modelado:** Selección del algoritmo, partición del dataset (`train_test_split`), entrenamiento y ajuste de hiperparámetros.
5. **Evaluación:** Validación cruzada, cálculo de métricas ajustadas al problema y análisis de error/sesgo.
6. **Despliegue:** Consumo del modelo en producción y monitoreo continuo.

### 1.2 Partición de Datos (Data Splitting)
Para evitar la memorización y evaluar la capacidad de generalización:
- **Conjunto de Entrenamiento (Train):** Usualmente entre 70% y 80% de los datos.
- **Conjunto de Prueba (Test):** Usualmente entre 20% y 30% de los datos.
- **Estratificación (`stratify=y`):** Obligatoria en problemas de **clasificación** para conservar la proporción original de las clases en las particiones de entrenamiento y prueba.

---

## MÓDULO 2: Aprendizaje Supervisado (Supervised Learning)

El objetivo es aprender una función $f(X)$ que mapee variables de entrada $X$ a una variable objetivo etiquetada $y$.

```
                        ┌── Regresión ───► Variable Objetivo Continua (ej: Precio, Temperatura)
APRENDIZAJE SUPERVISADO ┤
                        └── Clasificación ─► Variable Objetivo Categórica (ej: Fraude/No Fraude, Categoría)
```

---

### 2.1 Regresión

#### A. Regresión Lineal Simple y Múltiple
Relación lineal entre la(s) variable(s) independiente(s) y la variable continua $y$.

$$\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n$$

*   **Premisa:** Asume relaciones lineales, homocedasticidad y falta de multicolinealidad.

#### B. Trade-off Sesgo-Varianza (Bias-Variance Tradeoff)
*   **Underfitting (Subajuste / Alto Sesgo):** El modelo es demasiado simple para capturar los patrones subyacentes. Presenta alto error tanto en train como en test.
*   **Overfitting (Sobreajuste / Alta Varianza):** El modelo memoriza el ruido del conjunto de entrenamiento. Presenta bajo error en train y alto error en test.

#### C. Regularización
Se agrega una penalización a la función de pérdida para controlar la complejidad del modelo y mitigar el sobreajuste:
*   **L1 (Lasso Regression):** $\text{Penalización} = \lambda \sum |\beta_j|$. Realiza selección de características al forzar algunos coeficientes exactamente a cero.
*   **L2 (Ridge Regression):** $\text{Penalización} = \lambda \sum \beta_j^2$. Reduce los coeficientes cercanos a cero sin eliminarlos por completo. Excelente para solucionar la multicolinealidad.

#### D. Métricas de Evaluación en Regresión
*   **Error Cuadrático Medio (MSE):** $MSE = \frac{1}{n} \sum (y_i - \hat{y}_i)^2$ (Penaliza errores grandes).
*   **Raíz del Error Cuadrático Medio (RMSE):** $RMSE = \sqrt{MSE}$ (En las mismas unidades que la variable $y$).
*   **Error Absoluto Medio (MAE):** $MAE = \frac{1}{n} \sum |y_i - \hat{y}_i|$ (Robusto ante outliers).
*   **Coeficiente de Determinación ($R^2$):** Proporción de la varianza explicada por el modelo ($0 \le R^2 \le 1$).

---

### 2.2 Clasificación

#### A. Regresión Logística
Utiliza la función sigmoide para transformar una combinación lineal en una probabilidad entre 0 y 1.

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

*   **Regla de Decisión:** Si $P(y=1|X) \ge 0.5 \implies$ Clase 1; en caso contrario, Clase 0.

#### B. Árboles de Decisión (Decision Trees)
Modelos no paramétricos que dividen el espacio de características mediante reglas booleanas consecutivas basadas en criterios de pureza:
*   **Gini Impurity:** $Gini = 1 - \sum p_i^2$
*   **Entropía (Ganancia de Información):** $H = -\sum p_i \log_2(p_i)$

#### C. K-Nearest Neighbors (KNN - Clasificación y Regresión)
Algoritmo basado en instancias (lazy learning). Clasifica una muestra según la clase mayoritaria entre sus $K$ vecinos más cercanos.
*   **Distancias Comunes:** Euclidiana ($d = \sqrt{\sum (x_i - y_i)^2}$) y Manhattan ($d = \sum |x_i - y_i|$).
*   **Requisito Crítico:** Requiere **estandarizar o normalizar los datos** previamente, ya que es altamente sensible a las escalas de las variables.

#### D. Métricas de Evaluación en Clasificación
Basadas en la **Matriz de Confusión**:

| | Predicción Positiva | Predicción Negativa |
|---|---|---|
| **Real Positivo** | Verdadero Positivo (VP) | Falso Negativo (FN) |
| **Real Negativo** | Falso Positivo (FP) | Verdadero Negativo (VN) |

*   **Accuracy (Exactitud):** $\frac{VP + VN}{VP + VN + FP + FN}$ (Engañosa en datasets desbalanceados).
*   **Precision (Precisión):** $\frac{VP}{VP + FP}$ (Importante cuando el costo de los Falsos Positivos es alto).
*   **Recall / Sensitivity (Sensibilidad):** $\frac{VP}{VP + FN}$ (Crítico cuando los Falsos Negativos son costosos, ej: detección de enfermedades).
*   **F1-Score:** Media armónica entre Precision y Recall: $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$.
*   **Curva ROC / AUC-ROC:** Evalúa el rendimiento del clasificador a través de diferentes umbrales de decisión.

---

## MÓDULO 3: Aprendizaje No Supervisado (Unsupervised Learning)

El objetivo es descubrir estructuras, agrupaciones o patrones ocultos en datos no etiquetados (sin variable objetivo $y$).

```
                           ┌── Agrupamiento (Clustering) ───► K-Means, DBSCAN, Jerárquico
APRENDIZAJE NO SUPERVISADO ┤
                           └── Reducción Dimensionalidad ───► PCA (Análisis de Componentes Principales)
```

---

### 3.1 Agrupamiento (Clustering)

#### A. Algoritmo K-Means
Partitiona $N$ observaciones en $K$ clusters donde cada observación pertenece al cluster con la media más cercana (centroide).

**Pasos:**
1. Definir arbitrariamente el número de clusters $K$.
2. Inicializar $K$ centroides aleatoriamente.
3. Asignar cada dato al centroide más cercano.
4. Recalcular el centroide de cada cluster tomando el promedio de sus puntos.
5. Repetir hasta la convergencia (cuando los centroides no cambien de posición).

#### B. Selección de $K$ Óptimo
*   **Método del Codo (Elbow Method):** Grafica la inercia (WCSS - Suma de cuadrados intra-cluster) en función de $K$. Se selecciona el punto donde la tasa de decrecimiento se desacelera abruptamente.
*   **Coeficiente de Silueta (Silhouette Score):** Mide qué tan similar es un objeto a su propio cluster en comparación con otros clusters. Varía de $-1$ a $+1$. Valores cercanos a $1$ indican agrupaciones bien separadas.

---

### 3.2 Reducción de Dimensionalidad

#### A. Análisis de Componentes Principales (PCA)
Técnica lineal que transforma un conjunto de variables correlacionadas en un número menor de variables no correlacionadas llamadas **Componentes Principales**, preservando la mayor varianza posible.

*   **Varianza Explicada Acumulada:** Porcentaje total de la varianza capturada por los $n$ componentes seleccionados (se busca comúnmente retener entre el 80% y 95% de la varianza original).
*   **Prerrequisito:** Requiere que las variables estén **estandarizadas** (`StandardScaler`), ya que variables con magnitudes grandes dominarían los componentes.

---

## MÓDULO 4: Implementación de Referencia en Python

El siguiente script contiene la plantilla de código reutilizable para entrenar, evaluar y comparar estos modelos en Python.

```python
# ==============================================================================
# PLANTILLA BASE DE IMPLEMENTACIÓN (SCIKIT-LEARN)
# ==============================================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    mean_squared_error, r2_score, silhouette_score
)

# ------------------------------------------------------------------------------
# 1. EJEMPLO SUPERVISADO: CLASIFICACIÓN
# ------------------------------------------------------------------------------
def flujo_clasificacion(X, y):
    """
    Recibe variables de entrada X y etiqueta y.
    Realiza partición, escalamiento, entrenamiento de KNN y evaluación.
    """
    # Partición estratificada
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Pipeline de Preprocesamiento y Modelo
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=5))
    ])
    
    # Entrenamiento
    pipeline.fit(X_train, y_train)
    
    # Predicción
    y_pred = pipeline.predict(X_test)
    
    # Evaluación
    print("--- REPORTE DE CLASIFICACIÓN ---")
    print(classification_report(y_test, y_pred))
    print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred))
    
    return pipeline

# ------------------------------------------------------------------------------
# 2. EJEMPLO NO SUPERVISADO: PCA + K-MEANS
# ------------------------------------------------------------------------------
def flujo_clustering_pca(X, n_clusters=3):
    """
    Estandariza los datos, reduce dimensionalidad con PCA y aplica K-Means.
    """
    # 1. Escalamiento obligatorio
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. PCA para retener el 90% de la varianza
    pca = PCA(n_components=0.90, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    print(f"Componentes creados por PCA: {X_pca.shape[1]}")
    print(f"Varianza explicada acumulada: {np.sum(pca.explained_variance_ratio_):.2f}")
    
    # 3. Agrupamiento con K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca)
    
    # 4. Evaluación de la calidad del agrupamiento
    sil_score = silhouette_score(X_pca, labels)
    print(f"Coeficiente de Silueta: {sil_score:.4f}")
    
    return labels, kmeans, pca
```

---

## MÓDULO 5: Guía de Decisión para la IA

Cuando el usuario plantee un requerimiento, aplica la siguiente lógica para seleccionar la mejor solución:

1. **¿El conjunto de datos tiene etiquetas explícitas ($y$)?**
   * **SÍ $\implies$ Aprendizaje Supervisado:**
     * ¿La etiqueta es un valor continuo? $\implies$ **Regresión** (Probar `LinearRegression`, `Ridge`/`Lasso` si hay sobreajuste).
     * ¿La etiqueta es discreta/categórica? $\implies$ **Clasificación** (Probar `LogisticRegression`, `DecisionTreeClassifier` o `KNeighborsClassifier`).
   * **NO $\implies$ Aprendizaje No Supervisado:**
     * ¿El objetivo es agrupar elementos similares? $\implies$ **Clustering** (`KMeans`, evaluar con *Elbow* y *Silhouette*).
     * ¿El dataset tiene demasiadas columnas correlacionadas? $\implies$ **Reducción de Dimensionalidad** (`PCA`).

2. **Verificaciones de Calidad Obligatorias:**
   * ¿Los algoritmos basados en distancia (KNN, K-Means, PCA, Regresión Logística) tienen datos escalados? **Exigir `StandardScaler` o `MinMaxScaler`**.
   * ¿Existe desbalance de clases en un problema de clasificación? **Priorizar $F1\text{-Score}$ y $Recall$ sobre el $Accuracy$**.