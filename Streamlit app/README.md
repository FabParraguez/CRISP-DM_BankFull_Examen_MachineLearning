# Aplicacion Streamlit - Bank Marketing

La aplicacion permite explorar el proyecto CRISP-DM de suscripcion a depositos
bancarios. Compara tres algoritmos de clasificacion en dos iteraciones, muestra
metricas de validacion y generalizacion, permite seleccionar predictores
disponibles antes del contacto y consulta las probabilidades de los seis modelos.

Los modelos de la seleccion predeterminada se cargan desde `../Modelos/`.
Al cambiar las variables, la aplicacion ajusta una configuracion exploratoria
en cache de sesion; esos ajustes no reemplazan los artefactos del proyecto.
`duration`, `campaign`, `day`, `month` y `contact` se excluyen de la prediccion
previa a la llamada.

## Ejecucion

Desde la raiz del proyecto:

```powershell
.\.venv\Scripts\python.exe -m streamlit run "Streamlit app\app.py"
```

Para volver a generar los artefactos de las dos iteraciones:

```powershell
.\.venv\Scripts\python.exe -m src.train_models --n-jobs 2 --random-state 42
```

Los modelos, sus metadatos y los reportes de validacion/generalizacion quedan en
`Modelos/`. La aplicacion reserva las mismas particiones estratificadas y el
conjunto de test no interviene en el ajuste de los estimadores.
