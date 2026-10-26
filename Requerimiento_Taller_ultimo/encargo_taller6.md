# Guía Paso a Paso: Taller 6 (Entregable Final CRISP-DM - BankFull)

El objetivo de este taller es consolidar **una iteración completa de la metodología CRISP-DM** dentro de un único notebook ubicado en:
`Taller6_examen/Ultima_Entrega_CRISPDM_BankFull.ipynb`

---

## Estructura del Notebook `Ultima_Entrega_CRISPDM_BankFull.ipynb`

---

### Paso 1: Configuración Inicial del Entorno

Abre el archivo `Ultima_Entrega_CRISPDM_BankFull.ipynb` e importa las librerías necesarias. Luego, carga el conjunto de datos `bank-full.csv`.
Revisa jupyters de carpeta Taller_ejemplo para hacerlos como enseñan ahi tambien revisa la carpeta requerimientos_examen los jupyter crispdm.md , eda.md, modelos.md


# Cargar el dataset bank-full.csv
df = pd.read_csv('../Data/bank-full.csv', sep=';')
df.head()
```

---

### Paso 2: Fase 1 - Comprensión del Negocio (*Business Understanding*)



---

### Paso 3: Fase 2 - Comprensión de los Datos (*Data Understanding*)

Analiza la estructura del dataset, revisa los tipos de variables, nulos y la distribución de la variable objetivo.


---

### Paso 4: Fase 3 - Preparación de los Datos (*Data Preparation*)

Prepara los datos para el modelado estructurando las transformaciones mediante un `ColumnTransformer`.

---

### Paso 5: Fase 4 - Modelado (*Modeling*)

Construye un `Pipeline` de scikit-learn para encapsular el preprocesamiento y el modelo predictivo (por ejemplo, `RandomForestClassifier`).



### Paso 6: Fase 5 - Evaluación (*Evaluation*)

Mide el rendimiento del modelo sobre el conjunto de prueba e interpreta los resultados.



### Paso 7: Fase 6 - Despliegue (*Deployment*)

> **Nota:** Incluye esta sección en una celda **Markdown** al final del notebook.

#### Resumen y Plan de Despliegue:
1. **Conclusión:** 
