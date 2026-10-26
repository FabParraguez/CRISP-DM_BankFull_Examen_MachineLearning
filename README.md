# Clasificación de Clientes para Campañas de Marketing Directo (Depósitos a Plazo)

**Autores:** Tabatha Gamboa - Fabian Parraguez  
**Metodología:** CRISP-DM  
**Dataset:** `bank-full.csv` (Bank Marketing Dataset)  

---

## Descripción del Proyecto

Este proyecto aborda el problema de la baja eficiencia en las campañas de marketing directo (llamadas telefónicas) de una institución bancaria. Contactar a todos los clientes genera altos costos comerciales y baja tasa de conversión. Para resolver esto, se construyó un modelo de **aprendizaje automático supervisado (clasificación)** capaz de predecir la probabilidad de que un cliente contrate un **depósito a plazo** (`y`: `yes`/`no`).

El análisis excluye deliberadamente las variables de contacto de la campaña actual (`duration`, `campaign`, `day`, `month`, `contact`) para garantizar que la priorización de los clientes se realice **antes** de iniciar las llamadas telefónicas.

---

## Objetivos y Criterios de Éxito

### Objetivo General
Desarrollar un modelo de clasificación para estimar la probabilidad de contratación de un depósito a plazo por parte de un cliente y apoyar la toma de decisiones comerciales.

### Objetivos Específicos
1. Identificar las variables demográficas y financieras asociadas a la contratación del depósito.
2. Estimar la probabilidad de contratación individual de cada cliente.
3. Evaluar el desempeño del modelo en datos no vistos (conjunto de prueba).

### Criterios de Éxito
- **Éxito de Negocio:**
  - Aumentar la tasa de conversión focalizando los esfuerzos comerciales en clientes con mayor disposición a comprar.
  - Reducir costos operativos reduciendo llamadas no eficientes.
- **Éxito Técnico:**
  - **Recall (clase `yes`)** $\ge 0.50$ (capturar al menos el 50% de los compradores potenciales).
  - **ROC-AUC** $\ge 0.90$ como meta de capacidad discriminante general.

---

## Estructura del Notebook (CRISP-DM)

El Jupyter Notebook desarrolla una iteración completa del marco CRISP-DM:

1. **Fase 1: Comprensión del Negocio**  
   Definición del problema comercial, objetivos, supuesto de datos desbalanceados y restricciones operativas.
2. **Fase 2: Comprensión de los Datos**  
   - Carga del conjunto de datos (45,211 registros y 17 columnas).
   - Análisis de calidad (verificación de valores nulos y duplicados).
   - Análisis exploratorio del desbalance de la clase objetivo (`y`): ~88% `no` vs ~12% `yes`.
   - Estadística descriptiva de variables numéricas y categóricas.
3. **Fase 3: Preparación de los Datos**  
   - Mapeo de la variable objetivo (`yes` = 1, `no` = 0).
   - Filtrado de variables excluidas (`duration`, `campaign`, `day`, `month`, `contact`).
   - División estratificada de datos en **Entrenamiento (80%)** y **Prueba (20%)**.
   - Definición de transformadores mediante `ColumnTransformer` (Estandarización de numéricas con `StandardScaler` y codificación de categóricas con `OneHotEncoder`).
4. **Fase 4 y 5: Modelado y Evaluación** *(en desarrollo en la iteración)*  
   Uso de `Pipeline` de Scikit-Learn combinando el preprocesamiento con un clasificador `RandomForestClassifier` equilibrado (`class_weight='balanced'`).

---

## Tecnologías Utilizadas

- **Lenguaje:** Python 3.x
- **Entorno:** Jupyter Notebook
- **Librerías Principales:**
  - `pandas` y `numpy` para manipulación de datos.
  - `matplotlib` y `seaborn` para visualizaciones de datos.
  - `scikit-learn` para preprocesamiento, construcción de pipelines y modelado de Machine Learning.

---

## Variables Utilizadas en el Modelo

### Variables Seleccionadas (Predicción Pre-campaña)
- **Numéricas:** `age`, `balance`, `pdays`, `previous`
- **Categóricas:** `job`, `marital`, `education`, `default`, `housing`, `loan`, `poutcome`

### Variables Excluidas
- `duration`, `campaign`, `day`, `month`, `contact` (Excluidas por ocurrir durante/después de la interacción).

---

## Cómo Ejecutar el Proyecto

1. **Clonar o descargar el repositorio.**
2. **Asegurar la estructura de directorios:**
   ```text
   ├── Data/
   │   └── bank-full.csv
   └── Notebooks/
       └── taller_6_clasificacion.ipynb
   ```
3. **Instalar las dependencias:**
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```
4. **Ejecutar el Jupyter Notebook:**
   Abrir `taller_6_clasificacion.ipynb` en Jupyter Notebook o JupyterLab y ejecutar todas las celdas en orden.
