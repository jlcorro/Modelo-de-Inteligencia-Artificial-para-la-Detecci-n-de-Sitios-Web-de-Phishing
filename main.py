# ============================================================
# IMPORTACIÓN DE LIBRERÍAS
# ============================================================

# Pandas permite cargar, manipular y analizar datos en estructuras
# tipo DataFrame, facilitando el procesamiento del conjunto de datos.
import pandas as pd

# Matplotlib se utiliza para crear gráficos y visualizaciones.
import matplotlib.pyplot as plt

# Seaborn complementa a Matplotlib proporcionando gráficos estadísticos
# con una apariencia más atractiva y sencilla de generar.
import seaborn as sns

# Graphviz permite generar representaciones gráficas del árbol de decisión.
import graphviz

# Divide el conjunto de datos en entrenamiento y prueba.
from sklearn.model_selection import train_test_split

# Implementación del algoritmo Árbol de Decisión para clasificación.
from sklearn.tree import DecisionTreeClassifier

# Permite visualizar el árbol de decisión directamente con Matplotlib.
from sklearn.tree import plot_tree

# Herramienta para búsqueda de los mejores hiperparámetros
# mediante validación cruzada.
from sklearn.model_selection import GridSearchCV

# Exporta el árbol en formato DOT para ser visualizado con Graphviz.
from sklearn.tree import export_graphviz

# Métricas utilizadas para evaluar el rendimiento del modelo.
from sklearn.metrics import (
    accuracy_score,          # Exactitud del modelo
    precision_score,         # Precisión
    recall_score,            # Sensibilidad (Recall)
    f1_score,                # Puntaje F1
    confusion_matrix,        # Matriz de confusión
    classification_report    # Reporte completo de métricas
)

# ============================================================
# PRESENTACIÓN DEL PROYECTO
# ============================================================

# Imprime un encabezado para identificar el proyecto en la consola.
print("=" * 60)
print("PROYECTO DE DETECCIÓN DE PHISHING")
print("=" * 60)

# ============================================================
# CARGA DEL CONJUNTO DE DATOS
# ============================================================

# Se carga el archivo CSV que contiene las características de los
# sitios web y la clasificación que indica si son legítimos o phishing.
df = pd.read_csv("data/Website Phishing.csv")

# Confirma que el conjunto de datos fue cargado correctamente.
print("\nDataset cargado correctamente.\n")

# ============================================================
# EXPLORACIÓN INICIAL DEL DATASET
# ============================================================

# Muestra las primeras cinco filas para conocer la estructura
# del conjunto de datos.
print("Primeras 5 filas:\n")
print(df.head())

# ============================================================
# DISTRIBUCIÓN DE LAS CLASES
# ============================================================

# Cuenta cuántos registros pertenecen a cada clase
# (sitios legítimos y sitios phishing).
print("\nClases del dataset:")
print(df["Result"].value_counts())

# Muestra los valores únicos existentes en la variable objetivo.
print("\nValores únicos:")
print(df["Result"].unique())

# ============================================================
# INFORMACIÓN GENERAL DEL DATASET
# ============================================================

# Presenta información general como:
# - Número de registros
# - Tipo de dato de cada columna
# - Cantidad de valores no nulos
print("\nInformación del Dataset\n")
print(df.info())

# ============================================================
# VERIFICACIÓN DE VALORES FALTANTES
# ============================================================

# Cuenta la cantidad de valores nulos presentes en cada columna.
# Esto permite identificar si es necesario realizar un proceso
# de limpieza de datos.
print("\nValores nulos\n")
print(df.isnull().sum())

# ============================================================
# ESTADÍSTICAS DESCRIPTIVAS
# ============================================================

# Calcula estadísticas básicas para todas las variables numéricas,
# como media, desviación estándar, mínimo, máximo y cuartiles.
print("\nEstadísticas\n")
print(df.describe())

# ============================================================
# INFORMACIÓN SOBRE LAS VARIABLES
# ============================================================

# Muestra el nombre de todas las columnas del conjunto de datos.
print("\nColumnas del dataset\n")
print(df.columns)

# ============================================================
# DIMENSIONES DEL DATASET
# ============================================================

# Muestra el número total de registros (filas).
print("\nNúmero de filas:", df.shape[0])

# Muestra el número total de variables (columnas).
print("Número de columnas:", df.shape[1])

# ============================================================
# SEPARACIÓN DE VARIABLES
# ============================================================

# Se muestra un encabezado para indicar el inicio del proceso
# de separación de las variables del conjunto de datos.
print("\n")
print("=" * 60)
print("SEPARANDO VARIABLES")
print("=" * 60)

# Se crean las variables predictoras (X) eliminando la columna
# "Result", ya que esta corresponde a la variable objetivo.
X = df.drop("Result", axis=1)

# Se almacena la variable objetivo (etiqueta) en la variable y.
# Esta columna contiene la clasificación del sitio web
# (legítimo o phishing).
y = df["Result"]

# Se muestran las dimensiones de las variables predictoras.
print("Variables predictoras:", X.shape)

# Se muestran las dimensiones de la variable objetivo.
print("Variable objetivo:", y.shape)

# ============================================================
# DIVISIÓN DEL CONJUNTO DE DATOS
# ============================================================

# Se imprime un encabezado para indicar el inicio de la división
# del conjunto de datos.
print("\n")
print("=" * 60)
print("DIVIDIENDO EL DATASET")
print("=" * 60)

# Se divide el conjunto de datos en entrenamiento (80%)
# y prueba (20%).
#
# test_size=0.20      -> El 20% de los datos se reserva para pruebas.
# random_state=42     -> Garantiza que la división sea reproducible.
# stratify=y          -> Mantiene la misma proporción de clases
#                        en ambos conjuntos.
X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

# Se muestran las dimensiones del conjunto de entrenamiento.
print("Entrenamiento:", X_train.shape)

# Se muestran las dimensiones del conjunto de prueba.
print("Prueba:", X_test.shape)

# ============================================================
# CREACIÓN DEL MODELO
# ============================================================

# Se imprime un encabezado indicando el inicio de la construcción
# del modelo de Árbol de Decisión.
print("\n")
print("=" * 60)
print("CREANDO EL MODELO")
print("=" * 60)

# ============================================================
# OPTIMIZACIÓN DE HIPERPARÁMETROS
# ============================================================

# Se imprime un encabezado indicando el inicio del proceso de
# optimización del modelo.
print("\n")
print("=" * 60)
print("OPTIMIZANDO EL MODELO")
print("=" * 60)

# Se define el conjunto de hiperparámetros que serán evaluados
# mediante búsqueda exhaustiva (Grid Search).
parametros = {

    # Función utilizada para medir la calidad de las divisiones.
    "criterion": ["gini", "entropy"],

    # Profundidad máxima permitida para el árbol.
    "max_depth": [3, 5, 7, 10, None],

    # Número mínimo de muestras necesarias para dividir un nodo.
    "min_samples_split": [2, 5, 10],

    # Número mínimo de muestras que debe contener una hoja.
    "min_samples_leaf": [1, 2, 4]

}

# Se configura GridSearchCV para evaluar todas las combinaciones
# posibles de hiperparámetros utilizando validación cruzada.
grid = GridSearchCV(

    # Modelo base que será optimizado.
    estimator=DecisionTreeClassifier(random_state=42),

    # Conjunto de parámetros a evaluar.
    param_grid=parametros,

    # Validación cruzada de cinco particiones.
    cv=5,

    # Métrica utilizada para seleccionar el mejor modelo.
    scoring="accuracy",

    # Utiliza todos los núcleos disponibles del procesador.
    n_jobs=-1

)

# Se ejecuta la búsqueda del mejor conjunto de hiperparámetros
# utilizando únicamente los datos de entrenamiento.
grid.fit(X_train, y_train)

# Se almacena el mejor modelo encontrado.
modelo = grid.best_estimator_

# Se muestran los mejores hiperparámetros obtenidos.
print("\nMejores parámetros encontrados:\n")
print(grid.best_params_)

# Se muestra el mejor porcentaje de exactitud obtenido durante
# la validación cruzada.
print("\nMejor Accuracy (validación cruzada):")
print(round(grid.best_score_ * 100, 2), "%")

# ============================================================
# ENTRENAMIENTO FINAL DEL MODELO
# ============================================================

# Se entrena el árbol de decisión utilizando los mejores
# hiperparámetros encontrados durante la optimización.
modelo.fit(X_train, y_train)

# Confirma que el entrenamiento finalizó correctamente.
print("Modelo entrenado correctamente.")

# ============================================================
# REALIZAR PREDICCIONES
# ============================================================

# Se utiliza el modelo entrenado para predecir la clase de los
# registros que pertenecen al conjunto de prueba.
predicciones = modelo.predict(X_test)

# ============================================================
# EVALUACIÓN DEL MODELO
# ============================================================

# Se calcula la exactitud (Accuracy), que representa el porcentaje
# de predicciones realizadas correctamente por el modelo.
accuracy = accuracy_score(y_test, predicciones)

# Se muestra el Accuracy en porcentaje.
print("\nAccuracy:", round(accuracy * 100, 2), "%")

# Se calcula la Precisión (Precision), la cual indica qué proporción
# de las predicciones positivas realizadas por el modelo fueron correctas.
#
# average="weighted" pondera cada clase según su cantidad de muestras,
# siendo adecuado para conjuntos de datos con clases desbalanceadas.
precision = precision_score(
    y_test,
    predicciones,
    average="weighted"
)

# Se calcula el Recall o Sensibilidad, que mide la capacidad
# del modelo para identificar correctamente cada clase.
recall = recall_score(
    y_test,
    predicciones,
    average="weighted"
)

# Se calcula el F1-Score, métrica que combina la Precisión
# y el Recall mediante su media armónica.
f1 = f1_score(
    y_test,
    predicciones,
    average="weighted"
)

# Se muestran las principales métricas de evaluación.
print(f"\nAccuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# ============================================================
# REPORTE DE CLASIFICACIÓN
# ============================================================

# Se genera un reporte detallado con las métricas de cada clase:
# - Precision
# - Recall
# - F1-Score
# - Soporte (cantidad de registros)
#
# zero_division=0 evita errores cuando alguna clase no presenta
# predicciones o ejemplos durante la evaluación.
print("\nReporte de clasificación:\n")

print(classification_report(
    y_test,
    predicciones,
    target_names=[
        "Legítimo",
        "Sospechoso",
        "Phishing"
    ],
    zero_division=0
))

# ============================================================
# MATRIZ DE CONFUSIÓN
# ============================================================

# Se calcula la matriz de confusión para comparar las clases reales
# contra las clases predichas por el modelo.
cm = confusion_matrix(

    y_test,

    predicciones

)

# Se crea una figura para visualizar la matriz de confusión.
plt.figure(figsize=(6, 5))

# Se dibuja la matriz utilizando un mapa de calor.
#
# annot=True -> Muestra los valores dentro de cada celda.
# fmt="d"    -> Presenta los valores como números enteros.
# cmap       -> Define la paleta de colores utilizada.
sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues"

)

# Etiqueta del eje horizontal (predicciones del modelo).
plt.xlabel("Predicción")

# Etiqueta del eje vertical (valores reales).
plt.ylabel("Real")

# Título de la gráfica.
plt.title("Matriz de Confusión")

# Guarda la imagen de la matriz de confusión en alta resolución.
plt.savefig(
    "resultados/matriz_confusion.png",
    dpi=300,
    bbox_inches="tight"
)

# Muestra la gráfica en pantalla.
plt.show()

# ============================================================
# VISUALIZACIÓN DEL ÁRBOL SIMPLIFICADO
# ============================================================

# Se informa al usuario que se iniciará la generación de una
# versión simplificada del árbol de decisión.
print("\nGenerando árbol simplificado...")

# Se define el tamaño de la figura donde se visualizará el árbol.
plt.figure(figsize=(18, 10))

# Se dibujan únicamente los tres primeros niveles del árbol para
# facilitar su interpretación.
#
# feature_names -> Nombres de las variables predictoras.
# class_names   -> Etiquetas de las clases del problema.
# filled        -> Colorea los nodos según la clase predominante.
# rounded       -> Muestra los nodos con bordes redondeados.
# fontsize      -> Tamaño de la fuente.
# max_depth     -> Limita la visualización a los primeros niveles.
plot_tree(
    modelo,
    feature_names=X.columns,
    class_names=["Legítimo", "Sospechoso", "Phishing"],
    filled=True,
    rounded=True,
    fontsize=10,
    max_depth=3
)

# Agrega un título a la gráfica.
plt.title("Árbol de Decisión (Primeros 3 niveles)")

# Ajusta automáticamente los márgenes de la figura.
plt.tight_layout()

# Guarda la imagen del árbol simplificado en alta resolución.
plt.savefig(
    "resultados/arbol_simplificado.png",
    dpi=300,
    bbox_inches="tight"
)

# Muestra la imagen en pantalla.
plt.show()

# ============================================================
# EXPORTACIÓN DEL ÁRBOL COMPLETO CON GRAPHVIZ
# ============================================================

# Se imprime un encabezado indicando el inicio de la exportación
# del árbol completo.
print("\n")
print("=" * 60)
print("ÁRBOL COMPLETO (GRAPHVIZ)")
print("=" * 60)

# Convierte el árbol de decisión al formato DOT, utilizado por
# Graphviz para generar diagramas de alta calidad.
dot_data = export_graphviz(
    modelo,
    out_file=None,
    feature_names=X.columns,
    class_names=["Legítimo", "Sospechoso", "Phishing"],
    filled=True,
    rounded=True,
    special_characters=True
)

# Se crea el objeto gráfico a partir del código DOT.
graph = graphviz.Source(dot_data)

# Guarda el árbol completo en formato PDF.
graph.render(
    filename="arbol_completo",
    directory="resultados",
    format="pdf",
    cleanup=True
)

# Guarda el árbol completo en formato PNG.
graph.render(
    filename="arbol_completo",
    directory="resultados",
    format="png",
    cleanup=True
)

# Confirma que la exportación fue realizada correctamente.
print("Árbol completo guardado correctamente.")

# ============================================================
# IMPORTANCIA DE LAS VARIABLES
# ============================================================

# Se imprime un encabezado para el análisis de importancia
# de las variables predictoras.
print("\n")
print("=" * 60)
print("IMPORTANCIA DE LAS VARIABLES")
print("=" * 60)

# Se crea un DataFrame que relaciona cada característica con
# su nivel de importancia calculado por el árbol de decisión.
importancias = pd.DataFrame({
    "Característica": X.columns,
    "Importancia": modelo.feature_importances_
})

# Se ordenan las variables desde la más importante hasta la menos importante.
importancias = importancias.sort_values(
    by="Importancia",
    ascending=False
)

# Se muestra la tabla de importancia en la consola.
print(importancias)

# ============================================================
# GRÁFICO DE IMPORTANCIA DE VARIABLES
# ============================================================

# Se crea una figura para representar gráficamente la importancia
# de cada característica.
plt.figure(figsize=(10, 6))

# Se genera un gráfico de barras horizontales donde cada barra
# representa la importancia de una variable.
sns.barplot(
    data=importancias,
    x="Importancia",
    y="Característica",
    hue="Característica",
    palette="viridis",
    legend=False
)

# Título del gráfico.
plt.title("Importancia de las Variables")

# Etiquetas de los ejes.
plt.xlabel("Importancia")
plt.ylabel("Características")

# Ajusta automáticamente los márgenes.
plt.tight_layout()

# Guarda el gráfico en formato PNG.
plt.savefig(
    "resultados/importancia_variables.png",
    dpi=300,
    bbox_inches="tight"
)

# Muestra el gráfico.
plt.show()

# ============================================================
# TABLA DE IMPORTANCIA DE VARIABLES
# ============================================================

# Se informa al usuario que se generará una tabla con la
# importancia de cada característica.
print("\nGenerando tabla de importancia de variables...")

# Se crea una figura para mostrar la tabla.
fig, ax = plt.subplots(figsize=(8, 4))

# Se ocultan los ejes para mostrar únicamente la tabla.
ax.axis('off')

# Se crea la tabla utilizando los datos del DataFrame.
tabla = ax.table(
    cellText=importancias.round(4).values,
    colLabels=importancias.columns,
    loc='center'
)

# Configuración del tamaño de fuente y escala.
tabla.auto_set_font_size(False)
tabla.set_fontsize(11)
tabla.scale(1.2, 1.6)

# Agrega un título a la tabla.
plt.title("Tabla de Importancia de las Variables", fontsize=14)

# Guarda la tabla como imagen.
plt.savefig(
    "resultados/tabla_importancia_variables.png",
    dpi=300,
    bbox_inches="tight"
)

# Muestra la tabla.
plt.show()

# ============================================================
# EXPORTACIÓN DE RESULTADOS
# ============================================================

# Exporta la tabla de importancia de variables a un archivo Excel.
importancias.to_excel(
    "resultados/importancia_variables.xlsx",
    index=False
)

# Confirma que el archivo fue creado correctamente.
print("Tabla guardada en resultados/importancia_variables.xlsx")

# ============================================================
# GUARDAR MÉTRICAS DEL MODELO
# ============================================================

# Se crea un archivo de texto para almacenar las principales
# métricas obtenidas durante la evaluación del modelo.
with open("resultados/metricas.txt", "w", encoding="utf-8") as archivo:

    # Encabezado del archivo.
    archivo.write("RESULTADOS DEL MODELO\n")
    archivo.write("=========================\n\n")

    # Se escriben las métricas de desempeño.
    archivo.write(f"Accuracy : {accuracy:.4f}\n")
    archivo.write(f"Precision: {precision:.4f}\n")
    archivo.write(f"Recall   : {recall:.4f}\n")
    archivo.write(f"F1 Score : {f1:.4f}\n\n")

    # Se almacenan los mejores hiperparámetros encontrados
    # durante la optimización del modelo.
    archivo.write("Mejores parámetros encontrados:\n")
    archivo.write(str(grid.best_params_))