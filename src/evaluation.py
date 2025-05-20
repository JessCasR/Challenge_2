# Importamos librerías necesarias para graficar y evaluar modelos de clasificación
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Función para evaluar un modelo de clasificación
# Recibe los valores reales (y_test), las predicciones del modelo (y_pred),
# y las etiquetas de clase que se mostrarán en la matriz de confusión
def evaluate_model(y_test, y_pred, labels=["Pro", "Con"]):
    # Imprime en consola un reporte con métricas como precisión, recall y F1-score
    print("Resultados del modelo:")
    print(classification_report(y_test, y_pred))
    
    # Calcula la matriz de confusión con las etiquetas especificadas
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    
    # Crea un objeto para mostrar gráficamente la matriz de confusión
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    
    # Dibuja la matriz de confusión con un mapa de colores azules
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    
    # Muestra la gráfica
    plt.show()
