El **Análisis Exploratorio de Datos (EDA)** es la etapa obligatoria de auditoría clínica en la que se interroga el *dataset* antes de aplicar cualquier modelo estadístico o algoritmo de Machine Learning. 

El objetivo principal es maximizar la comprensión del conjunto de datos, revelar su estructura interna, detectar variables clave, limpiar anomalías, asegurar la calidad de la información e interpretar los resultados considerando aspectos éticos del negocio.

---

## 📌 Paso 1: Carga de Librerías y Datos

Carga el conjunto de datos inicial y realiza un primer diagnóstico general para conocer sus dimensiones y tipos de datos.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Carga de datos
df = pd.read_csv('tu_archivo.csv')

# Inspección básica de dimensiones y nombres
print("=== FORMA DEL DATASET ===")
print(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}\n")

print("=== INFORMACIÓN DE TIPOS DE DATOS ===")
print(df.info())

print("\n=== RESUMEN ESTADÍSTICO ===")
print(df.describe())
```

---

## 📌 Paso 2: Auditoría y Tratamiento de Calidad del Dato

Antes de responder preguntas de negocio, es imperativo tratar la información ausente y duplicada para garantizar la consistencia.

### 2.1 Manejo de Registros Duplicados
Los duplicados pueden sobreajustar (*overfitting*) los modelos al hacer que el algoritmo memorice filas repetidas en lugar de aprender el patrón general.

```python
# Conteo de duplicados
total_duplicados = df.duplicated().sum()
print(f"Filas duplicadas detectadas: {total_duplicados}")

# Eliminación de duplicados (manteniendo la primera ocurrencia)
if total_duplicados > 0:
    df.drop_duplicates(keep='first', inplace=True)
    df.reset_index(drop=True, inplace=True)
```

### 2.2 Tratamiento de Valores Faltantes (NaN)
* **Eliminación (`.dropna()`):** Útil solo si la pérdida es mínima (< 5%) y ocurre al azar.
* **Imputación (`.fillna()`):** Reemplazo mediante la mediana o media (para variables numéricas) o la moda (para variables categóricas).

```python
# Diagnóstico de nulos por columna
print("Valores nulos por columna:")
print(df.isnull().sum())

# Imputación de variables continuas con la mediana
mediana_edad = df['Age'].median()
df['Age'].fillna(mediana_edad, inplace=True)

# Imputación de variables categóricas con la moda
# moda_cat = df['Categoria'].mode()[0]
# df['Categoria'].fillna(moda_cat, inplace=True)
```

---

## 📌 Paso 3: Análisis Univariado

Examina el comportamiento, rango y tipo de cada variable aislada.

### 3.1 Distribución de Variables Numéricas (Histogramas)
El histograma agrupa variables numéricas continuas en intervalos (*bins*). Permite identificar si una variable sigue una distribución normal (campana simétrica) o presenta un sesgo pronunciado.

```python
plt.figure(figsize=(8, 4))
sns.histplot(df['Age'], kde=True, bins=15, color='skyblue')
plt.title("Distribución de la Variable Edad")
plt.xlabel("Edad")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.show()
```

---

## 📌 Paso 4: Detección e Imputación de Valores Atípicos (*Outliers*)

### 4.1 Identificación Gráfica mediante Diagrama de Caja (*Boxplot*)
Utiliza los cuartiles estadísticos para aislar visualmente las anomalías de los datos.

```python
plt.figure(figsize=(6, 4))
sns.boxplot(x=df['Balance'], color='coral')
plt.title("Detección de Outliers en Balance")
plt.tight_layout()
plt.show()
```

### 4.2 Aislamiento y Filtro Matemático (Regla IQR / Rango Intercuartílico)
El Rango Intercuartílico se define como:
$$IQR = Q3 - Q1$$

Cualquier valor fuera de los siguientes límites matemáticos se etiqueta como atípico:
* **Límite Superior:** $Ls = Q3 + 1.5 \times IQR$
* **Límite Inferior:** $Li = Q1 - 1.5 \times IQR$

```python
# Cálculo del Rango Intercuartílico (IQR)
Q1 = df['Balance'].quantile(0.25)
Q3 = df['Balance'].quantile(0.75)
IQR = Q3 - Q1

limite_inferior = Q1 - (1.5 * IQR)
limite_superior = Q3 + (1.5 * IQR)

# Filtrado lógico para aislar u omitir outliers
df_limpio = df[(df['Balance'] >= limite_inferior) & (df['Balance'] <= limite_superior)].copy()

print(f"Registros antes del filtro: {df.shape[0]}")
print(f"Registros después de remover outliers: {df_limpio.shape[0]}")
```

---

## 📌 Paso 5: Análisis Bivariado / Multivariado y Correlaciones

Analiza cómo interactúan dos o más variables simultáneamente.

### 5.1 Matriz de Correlación Lineal (Coeficiente de Pearson)
Calcula la correlación entre pares numéricos (entre $-1$ y $+1$). Permite detectar la **multicolinealidad** (variables predictoras altamente correlacionadas entre sí) para evitar redundancias.

```python
# Selección de variables numéricas para la matriz
cols_numericas = df_limpio.select_dtypes(include=[np.number]).columns
matriz_corr = df_limpio[cols_numericas].corr()

# Generación del Mapa de Calor (Heatmap)
plt.figure(figsize=(8, 6))
sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title("Matriz de Correlación Lineal (Dataset Limpio)")
plt.tight_layout()
plt.show()
```

---

## 📌 Paso 6: Evaluación Ética y Sesgos de Negocio

El tratamiento de datos es tanto un proceso técnico como un **imperativo ético y social**:

1. **Evitar la purga a ciegas:** No eliminar datos faltantes o *outliers* de grupos minoritarios o subrepresentados sin antes realizar una auditoría.
2. **Diferenciación de anomalías:** Asegurar que el filtro distinga correctamente entre un "error de sistema" (ruido) y una "realidad de negocio anómala pero legítima".
3. **Prevención de discriminación algorítmica:** Garantizar que los procesos de automatización no perpetúen sesgos o invisibilicen perfiles clave.