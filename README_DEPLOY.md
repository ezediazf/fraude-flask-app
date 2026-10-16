# Despliegue
1. Copiar `Fraude.csv` en esta carpeta.
2. `python -m venv .venv`
3. Activar el entorno.
4. `pip install -r requirements.txt`
5. `python train_model.py`
6. `python app.py`
7. Abrir `http://127.0.0.1:5000`.

Para Render: subir esta carpeta a GitHub después de generar `fraud_model.joblib`; Build `pip install -r requirements.txt`; Start `gunicorn app:app`; Health `/health`.

El ZIP no incluye `Fraude.csv`, archivos temporales, copias duplicadas, presentaciones intermedias ni cachés.
