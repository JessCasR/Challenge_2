from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Inicializar VADER
analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text, lang):

    # Aplica análisis de sentimiento con VADER, sin importar el idioma.
    # Devuelve la etiqueta ('POS', 'NEG', 'NEU') y el score de compound.
    score = analyzer.polarity_scores(text)
    label = "POS" if score["compound"] > 0.05 else "NEG" if score["compound"] < -0.05 else "NEU"
    return label, score["compound"]
