# Aplicación Streamlit

La aplicación compara tres iteraciones de modelos para predecir si un cliente
suscribirá un depósito a plazo. Usa el archivo `../Data/bank-full.csv`, conserva
un conjunto estratificado de generalización y permite seleccionar las variables
antes de entrenar los tres modelos.

## Ejecución

Desde la raíz del proyecto:

```powershell
python -m pip install -r "Streamlit app/requirements.txt"
python -m streamlit run "Streamlit app/app.py"
```

Los modelos se entrenan y almacenan en caché en la sesión de Streamlit. La tabla
de métricas se obtiene exclusivamente del conjunto reservado para generalización;
la consulta interactiva devuelve la predicción y probabilidad de los tres modelos.
