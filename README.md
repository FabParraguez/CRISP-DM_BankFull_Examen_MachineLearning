# Proyecto Bank Marketing - CRISP-DM

Proyecto de clasificacion para analizar la suscripcion a depositos a plazo en
`Data/bank-full.csv`. La implementacion reutilizable de EDA, ETL y modelado esta
en `src/`; las seis fases se documentan en sus notebooks bajo `CRISP-DM/`.

## Entregables principales

- `CRISP-DM/fase1_Comprension_Negocio.ipynb` a `CRISP-DM/fase6_Despliegue.ipynb`:
  desarrollo del ciclo CRISP-DM.
- `Jupyter Notebook/eda.ipynb`, `etl.ipynb` y `seleccion_var.ipynb`: EDA,
  transformaciones y seleccion de variables, respectivamente.
- `src/eda.py`, `src/etl.py`, `src/modeling.py`: funciones reutilizables.
- `Modelos/`: seis pipelines serializados (tres algoritmos en cada una de dos
  iteraciones), metadatos, metricas y predicciones de generalizacion.
- `Streamlit app/app.py`: aplicacion local para comparar, seleccionar variables
  y consultar los modelos.
- `Taller6_examen/Ultima_Entrega_CRISPDM_BankFull.ipynb`: entregable especifico
  del Taller 6, que mantiene una unica iteracion como indico el docente.

## Reproducibilidad

Desde la raiz del repositorio:

```powershell
.\.venv\Scripts\python.exe -m src.train_models --n-jobs 2 --random-state 42
.\.venv\Scripts\python.exe -m streamlit run "Streamlit app\app.py"
```

Las particiones son estratificadas: 60% entrenamiento, 20% validacion y 20%
test ciego. `duration`, `campaign`, `day`, `month` y `contact` se excluyen para
evitar usar informacion que no esta disponible al priorizar clientes antes de
la llamada.
