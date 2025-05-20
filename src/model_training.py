# Importamos librerías necesarias para vectorización de texto, entrenamiento y evaluación de modelos
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Función para entrenar un modelo de clasificación de texto usando regresión logística
# df: DataFrame con los datos ya procesados
# text_column: columna que contiene el texto procesado
# label_column: columna que contiene las etiquetas ("Pro", "Con")
def train_classifier(df, text_column="processed", label_column="type"):
    print("Entrenando modelo de clasificación...")

    # Filtramos el dataframe para quedarnos solo con las clases "Pro" y "Con"
    df = df[df[label_column].isin(["Pro", "Con"])]

    # Dividimos los datos en entrenamiento y prueba (80%-20%)
    # Se asegura al menos 2 ejemplos en el test set
    X_train, X_test, y_train, y_test = train_test_split(
        df[text_column],
        df[label_column],
        test_size=max(2, int(len(df) * 0.2)),
        random_state=42
    )

    # Mostramos la distribución de clases en el conjunto de prueba
    print("Distribución en test set:", y_test.value_counts())

    # Definimos un pipeline: vectorización TF-IDF seguida por un clasificador de regresión logística
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=500, class_weight="balanced"))  # max_iter=500 para asegurar convergencia
    ])

    # Entrenamos el pipeline con los datos de entrenamiento
    pipeline.fit(X_train, y_train)

    # Realizamos predicciones sobre el conjunto de prueba
    y_pred = pipeline.predict(X_test)

    # Generamos un reporte de clasificación con métricas por clase
    report = classification_report(y_test, y_pred, output_dict=True)

    print("Clasificación terminada.")
    
    # Devolvemos el modelo entrenado, el reporte, y los datos para evaluación posterior
    return pipeline, report, X_test, y_test, y_pred
