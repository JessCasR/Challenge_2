import pandas as pd
from src.scraper import scrape_multiple
from src.preprocessing import preprocess_dataframe
from src.sentiment_analysis import analyze_sentiment
from src.model_training import train_classifier
from src.evaluation import evaluate_model
from src.mlops_pipeline import log_with_mlflow

def main():
    print("Análisis de reseñas laborales desde Glassdoor")

    # Scraping
    companies = [
    "Google-Reviews-E9079.htm",
    "Amazon-Reviews-E6036.htm",
    "Microsoft-Reviews-E1651.htm",
    "Meta-Reviews-E40772.htm",            
    "Apple-Reviews-E1138.htm",
    "IBM-Reviews-E354.htm",
    "Tesla-Reviews-E43129.htm",
    "Intel-Corporation-Reviews-E1519.htm",
    "Oracle-Reviews-E1737.htm",
    "Salesforce-Reviews-E11159.htm",
]
    scrape_multiple(companies)

    # Cargar datos
    df = pd.read_csv("data/glassdoor_reviews.csv")

    # Preprocesamiento
    df = preprocess_dataframe(df)  # crea columna 'processed'

    print("[DEBUG] Columnas del DataFrame:", df.columns)

   # Análisis de sentimiento
    print("Analizando sentimiento...")
    results = df.apply(lambda row: analyze_sentiment(row["processed"], row["language"]), axis=1)
    df[["sentiment", "sentiment_score"]] = pd.DataFrame(results.tolist(), index=df.index)

    # Clasificación tipo Pro/Con solo si POS o NEG
    df["type"] = df["sentiment"].apply(
        lambda x: "Pro" if x == "POS" else "Con" if x == "NEG" else "Neutral"
    )
    df = df[df["type"].isin(["Pro", "Con"])]

    # Mostrar distribución
    print("Distribución de clases después del filtrado:")
    print(df["type"].value_counts())

    # Verifica que haya suficientes datos
    if df["type"].nunique() < 2 or len(df) < 4:
        print("[ERROR] No hay suficientes reseñas etiquetadas como Pro y Con para entrenar.")
        return

    # Entrenamiento
    model, report, X_test, y_test, y_pred = train_classifier(df)

    # Evaluación
    evaluate_model(y_test, y_pred)

    # MLOps tracking
    log_with_mlflow(model, report)

    print("Pipeline completado")

if __name__ == "__main__":
    main()