import pandas as pd
import re
import string
from langdetect import detect
import nltk
from nltk.corpus import stopwords

# Verificamos si las stopwords ya están descargadas, si no, las descargamos
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# Cargamos listas de stopwords en inglés y español
stopwords_en = set(stopwords.words("english"))
stopwords_es = set(stopwords.words("spanish"))

# Función para limpiar el texto:
# convierte a minúsculas, elimina puntuación y números
def clean_text(text):
    text = text.lower()  # Minúsculas
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)  # Quita signos de puntuación
    text = re.sub(r"\d+", "", text)  # Quita números
    return text.strip()  # Quita espacios al inicio/final

# Función para detectar el idioma del texto
def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"  # Si falla, retorna "desconocido"

# Función para eliminar stopwords según el idioma detectado
def remove_stopwords(text, lang):
    tokens = text.split()  # Separamos el texto en palabras (tokens simples)
    stops = stopwords_en if lang == "en" else stopwords_es  # Elegimos las stopwords adecuadas
    filtered = [word for word in tokens if word not in stops]  # Quitamos las stopwords
    return " ".join(filtered)  # Reunimos las palabras filtradas

# Función para aplicar todo el preprocesamiento al DataFrame
def preprocess_dataframe(df):
    print("[INFO] Preprocesando reseñas...")

    # Limpieza básica del texto
    df["cleaned"] = df["review"].apply(clean_text)

    # Detección de idioma para cada reseña
    df["language"] = df["review"].apply(detect_language)

    # Eliminación de stopwords basadas en el idioma detectado
    df["processed"] = df.apply(lambda row: remove_stopwords(row["cleaned"], row["language"]), axis=1)

    return df
