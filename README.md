# Glassdoor Feedback Analysis

Este proyecto tiene como objetivo analizar las reseñas laborales de Glassdoor, clasificándolas en positivas ("Pro") o negativas ("Con") mediante técnicas de NLP y análisis de sentimiento. El flujo incluye scraping automatizado, preprocesamiento, entrenamiento de modelo y registro MLOps con MLflow.

## Estructura del Proyecto

- `data/`: Contiene los datasets extraídos en formato `.csv`.
- `src/`: Contiene los scripts principales del proyecto:
  - `scraper.py`: Web scraper con Selenium + BeautifulSoup para Glassdoor.
  - `preprocessing.py`: Limpieza de texto, detección de idioma y eliminación de stopwords.
  - `model_training.py`: Entrenamiento de modelo con `TfidfVectorizer` y `LogisticRegression`.
  - `evaluation.py`: Reporte de métricas y visualización de la matriz de confusión.
  - `mlops_pipeline.py`: Registro del modelo y métricas en MLflow.
  - `sentiment_analysis.py`: Análisis de sentimiento con VADER.
- `requirements.txt`: Lista de dependencias necesarias para reproducir el proyecto.
- `README.md`: Instrucciones de uso.

## Instrucciones

1. Clona el repositorio:
    a. `git clone https://github.com/JessCasR/Challenge_2`
    b.  cd Challenge_2
2. Instala las dependencias: `pip install -r requirements.txt`.
3. Corre el siguiente comando en consola para ejecutar el pipeline: `python src/main.py`.
4. Sube los resultados a GitHub con `upload_results.py`.

## Instrucciones para visualizar los resultado en MLflow

1. Ejecuta el siguiente comando en consola: `mlflow ui`.
2. Abre el navegador y dirigete a http://localhost:5000 para visualizar los experimentos, modelos y métricas registrados.