# Detección de Sitios Web Phishing mediante Árboles de Decisión

## Descripción

Este proyecto implementa un modelo de **Inteligencia Artificial** basado en **Árboles de Decisión** para clasificar sitios web en tres categorías:

* **Legítimo**
* **Sospechoso**
* **Phishing**

El modelo fue desarrollado utilizando **Python** y la biblioteca **Scikit-Learn**, aplicando un proceso completo de aprendizaje automático que incluye exploración de datos, entrenamiento, optimización de hiperparámetros, evaluación del modelo y generación de visualizaciones.

---

# Objetivo

Desarrollar un modelo de clasificación capaz de identificar páginas web potencialmente fraudulentas (phishing), optimizando su rendimiento mediante búsqueda de hiperparámetros y evaluando su desempeño utilizando diferentes métricas de clasificación.

---

# Tecnologías utilizadas

* Python 3.14.64
* Pandas
* Matplotlib
* Seaborn
* Scikit-Learn
* Graphviz

---

# Estructura del proyecto

```text
Proyecto/
│
├── data/
│   └── Website Phishing.csv
│
├── resultados/
│   ├── arbol_completo.pdf
│   ├── arbol_completo.png
│   ├── arbol_simplificado.png
│   ├── importancia_variables.png
│   ├── tabla_importancia_variables.png
│   ├── importancia_variables.xlsx
│   ├── matriz_confusion.png
│   └── metricas.txt
│
├── main.py
├── requirements.txt
└── README.md
```

---

# Dataset

El proyecto utiliza el conjunto de datos:

**Website Phishing.csv**

Cada registro contiene múltiples características extraídas de un sitio web y una variable objetivo denominada **Result**, que representa la clasificación del sitio.

Las clases utilizadas son:

| Valor | Clase      |
| ----- | ---------- |
| 0     | Legítimo   |
| 1     | Sospechoso |
| 2     | Phishing   |

---

# Flujo del proyecto

El desarrollo del modelo sigue las siguientes etapas:

1. Carga del conjunto de datos.
2. Exploración y análisis inicial.
3. Separación de variables predictoras y variable objetivo.
4. División del dataset en entrenamiento y prueba.
5. Optimización del Árbol de Decisión mediante GridSearchCV.
6. Entrenamiento del modelo.
7. Predicción sobre el conjunto de prueba.
8. Evaluación mediante métricas de clasificación.
9. Generación de la matriz de confusión.
10. Visualización del árbol simplificado.
11. Exportación del árbol completo con Graphviz.
12. Análisis de importancia de variables.
13. Exportación de resultados.

---

# Optimización del modelo

El modelo utiliza **GridSearchCV** para encontrar automáticamente la mejor combinación de hiperparámetros.

Se evaluaron los siguientes parámetros:

* Criterion:

  * gini
  * entropy

* Max Depth:

  * 3
  * 5
  * 7
  * 10
  * None

* Min Samples Split:

  * 2
  * 5
  * 10

* Min Samples Leaf:

  * 1
  * 2
  * 4

La selección del mejor modelo se realizó mediante **Validación Cruzada de 5 particiones (5-Fold Cross Validation)** utilizando la métrica Accuracy.

---

# Métricas utilizadas

El modelo es evaluado mediante:

* Accuracy
* Precision
* Recall
* F1-Score
* Matriz de Confusión
* Classification Report

---

# Resultados generados

Al ejecutar el proyecto se generan automáticamente los siguientes archivos dentro de la carpeta **resultados**:

| Archivo                         | Descripción                                               |
| ------------------------------- | --------------------------------------------------------- |
| matriz_confusion.png            | Matriz de confusión del modelo                            |
| arbol_simplificado.png          | Visualización de los primeros niveles del árbol           |
| arbol_completo.pdf              | Árbol completo en formato PDF                             |
| arbol_completo.png              | Árbol completo en formato imagen                          |
| importancia_variables.png       | Gráfico de importancia de variables                       |
| tabla_importancia_variables.png | Tabla de importancia como imagen                          |
| importancia_variables.xlsx      | Tabla de importancia en Excel                             |
| metricas.txt                    | Accuracy, Precision, Recall, F1 y mejores hiperparámetros |

---

# Instalación

Clonar el repositorio:

```bash
git clone <URL_DEL_REPOSITORIO>
```

Entrar al proyecto:

```bash
cd Proyecto
```

Instalar las dependencias:

```bash
pip install -r requirements.txt
```

---

# Ejecución

Ejecutar el archivo principal:

```bash
python main.py
```

---

# Librerías necesarias

```text
pandas
matplotlib
seaborn
graphviz
scikit-learn
openpyxl
```

También es necesario tener instalado **Graphviz** en el sistema operativo para exportar correctamente el árbol completo en formato PDF y PNG.

---

# Salidas del programa

Durante la ejecución el programa muestra información como:

* Información general del dataset.
* Estadísticas descriptivas.
* Valores nulos.
* Distribución de clases.
* Mejores hiperparámetros encontrados.
* Accuracy obtenido.
* Precision.
* Recall.
* F1-Score.
* Classification Report.

Además, genera automáticamente todas las imágenes y archivos de resultados.

---

# Autor

Proyecto desarrollado como actividad académica para la asignatura de **Inteligencia Artificial Avanzada**, utilizando un modelo de **Árbol de Decisión** para la detección de sitios web de phishing.
