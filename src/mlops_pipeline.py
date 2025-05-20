# Importamos las librerías necesarias para trabajar con MLflow, manejo de archivos y serialización JSON
import mlflow
import mlflow.sklearn
import json
import os

# Función para registrar un modelo y sus métricas en MLflow
# model: modelo de sklearn ya entrenado
# classification_report_dict: diccionario con métricas de evaluación (por ejemplo, salida de classification_report(output_dict=True))
# run_name: nombre del experimento en MLflow
def log_with_mlflow(model, classification_report_dict, run_name="glassdoor_model_run"):
    print("Registrando experimento en MLflow...")

    # Define o crea el experimento en MLflow donde se registrarán los datos
    mlflow.set_experiment("Glassdoor_Reviews_Analysis")

    # Inicia un nuevo "run" en MLflow con el nombre especificado
    with mlflow.start_run(run_name=run_name):
        # Registra el modelo entrenado en MLflow
        mlflow.sklearn.log_model(model, "model")

        # Registra el tipo de modelo como parámetro del experimento
        mlflow.log_param("model_type", "LogisticRegression")

        # Recorre las métricas de evaluación para cada clase ("Pro", "Con") y registra los valores
        for label in ["Pro", "Con"]:
            for metric in ["precision", "recall", "f1-score"]:
                value = classification_report_dict.get(label, {}).get(metric)
                if value is not None:
                    mlflow.log_metric(f"{label}_{metric}", value)

        # Crea una carpeta para guardar artefactos si no existe
        os.makedirs("mlruns_artifacts", exist_ok=True)

        # Guarda el reporte de métricas como archivo JSON
        report_path = "mlruns_artifacts/classification_report.json"
        with open(report_path, "w") as f:
            json.dump(classification_report_dict, f, indent=4)

        # Registra el archivo JSON como artefacto en MLflow
        mlflow.log_artifact(report_path)

        print("MLflow tracking completado.")
