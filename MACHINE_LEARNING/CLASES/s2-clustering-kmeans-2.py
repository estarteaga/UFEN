
"""
UNIVERSIDAD DE CHILE
DIPLOMADO EN CIENCIA DE DATOS PARA LAS FINANZAS
CURSO MACHINE LEARNING PARA FINANZAS
EXCLUSIVO USO CURSO PROFESOR GUILLERMO YAÑEZ

Esta vez desarrollaremos un ejercicio más completo

Preguntas:

Ud. quiere crear un modelo de clustering para dividir el mercado accionario en 3 segmentos de acciones riesgosas. 
Las de bajo riesgo, riesgo moderado y alto riesgo. Los clusters serán 0,1,2 respectivamente.
Para ello, debe simular 1000 acciones donde los features (atributos) son:
Beta de la acción, índice de endeudamiento, volatilidad de las ventas (desv. std.)
Simule todos estos datos usando numpy, scipy y/o sklearn.

Junte sus resultados en un DataFrame de Pandas

Ahora use sklearn para obtener un modelo predictivo de 100 de las 1000 observaciones para testeo (aleatorio)
Transforme sus series en los objetos que estime necesarios para su desarrollo.
Analice la capacidad predictiva del cluster al que pertenecen esas 100 acciones solo observando los features indicados.


"""

#
import numpy as np
# Ya lo conocemos Biblioteca que facilita la manipulación de arrays y operaciones matemáticas avanzadas.

import pandas as pd
# Ampliamente conocido por nosotros. Herramienta para la manipulación y análisis de datos tabulares, permitiendo la lectura, limpieza y análisis de dato

from sklearn.cluster import KMeans
#Algoritmo de clustering que agrupa datos en K clusters basados en características similares, les da atributos.

from sklearn.model_selection import train_test_split
#Divide los datos en conjuntos de entrenamiento y prueba para entrenar y evaluar un modelo.

from sklearn.metrics import accuracy_score
#Mide la precisión del modelo de clasificación comparando las predicciones con las etiquetas verdaderas.

import matplotlib.pyplot as plt
#Biblioteca para crear gráficos y visualizaciones de datos, como gráficos de líneas, barras y dispersión.

import seaborn as sns
# Apoyo visual para plt

from scipy.stats import norm
#utilizamos norm para simular la Beta de la acción (relación con el mercado)

from pylab import plt, mpl
# En pylab encontramos matplotlib completo, aprovechamos de cargarlo desde ahí

from sklearn.metrics import silhouette_score # Para determinar clusters por método silhouette

from sklearn.metrics import confusion_matrix
#El sub modulo confusion_matrix es especifico para poder obtener la matriz de confusión 

from sklearn.metrics import mean_squared_error
#Utilizamos el sub modulo mean_squared_error, para poder calcular el MSE de forma directa

#%%

# Simulación de datos

# Cambia la forma en que NumPy imprime sus arrays en pantalla:
# precision=4: muestra los números con 4 decimales.
# suppress=True: evita la notación científica (como 1.23e-04) en los arrays y muestra los números normalmente (como 0.0001).
np.set_printoptions(precision=4, suppress=True)

# Generar una variable latente (por ejemplo, un “factor de riesgo bajo”) 
# que influya negativamente sobre beta, endeudamiento y volatilidad.
# A cada variable, le sumamos un componente aleatorio para mantener la variabilidad.

# Semilla para reproducibilidad
np.random.seed(42)

# Número de observaciones
n_samples = 1000

# Variable latente: cuanto más alto, menor riesgo (y por tanto menor beta, endeudamiento y volatilidad)
# loc=0: es la media (μ = 0),
# scale=1: es la desviación estándar (σ = 1),

riesgo_latente = np.random.normal(loc=0, scale=1, size=n_samples)

# Simulación condicionada: más riesgo latente => menores valores en las 3 variables
beta = 1.5 - 0.4 * riesgo_latente + np.random.normal(0, 0.2, size=n_samples)
endeudamiento = 1.2 - 0.5 * riesgo_latente + np.random.normal(0, 0.1, size=n_samples)
volatilidad_ventas = 0.25 - 0.05 * riesgo_latente + np.random.normal(0, 0.02, size=n_samples)

# Límite inferior para que no haya valores negativos
# np.clip(array, min_value, max_value) None significa que no hay máximo
beta = np.clip(beta, 0.1, None)
endeudamiento = np.clip(endeudamiento, 0.05, None)
volatilidad_ventas = np.clip(volatilidad_ventas, 0.01, None)

#%%
# Crear DataFrame
data = pd.DataFrame({
    'Beta': beta,
    'Endeudamiento': endeudamiento,
    'Volatilidad_Ventas': volatilidad_ventas
})


#%%
# Aplicamos el algoritmo de KMeans para agrupar las acciones en 3 segmentos
kmeans = KMeans(n_clusters=3, random_state=0, n_init=100) 
# Itera 100 veces para minimizar la varianza de miembros de un cluster con su centroide

data['Cluster'] = kmeans.fit_predict(data)

#%%
# Visualización de los clusters
 
# Configuración del entorno de visualización de gráficos

mpl.rcParams['font.family'] = 'serif'

plt.figure(figsize=(12, 8))

# Gráfico de dispersión en 3D # solicite la ayuda de ChatGpt para poder graficarlo en 3D
ax = plt.axes(projection='3d')
# cmap: colormap (mapa de colores) que se usa para representar valores numéricos en gráficos donde los colores tienen un significado
ax.scatter(data['Beta'], data['Endeudamiento'], data['Volatilidad_Ventas'], 
           c=data['Cluster'], cmap='viridis', alpha=0.6)
ax.set_xlabel('Beta de la acción')
ax.set_ylabel('Índice de Endeudamiento')
ax.set_zlabel('Volatilidad de las Ventas')
ax.set_title('Clustering de 1000 acciones según riesgo')
plt.colorbar(ax.collections[0], label='Cluster')
plt.show()


# Mostramos las primeras filas del DataFrame con los clusters asignados
print(data.head())

#%%
# ahora continuamos con el modelo predictivo 

# El sub modulo train_test_split nos permite obtener facilmente la division de los datos en un % de entrenamiento y prueba

# Dividimos los datos en entrenamiento (90%) y prueba (10%)
X = data[['Beta', 'Endeudamiento', 'Volatilidad_Ventas']]
y = data['Cluster']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


#%%

# Métodos para determinar número de clusters:
    
# Método del codo (Elbow Method)
# Este método grafica la inercia (distancia dentro del cluster) 
# según el número de clusters y busca el punto donde la mejora comienza a ser marginal (el “codo”)

inertias = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Número de clusters (k)')
plt.ylabel('Inercia')
plt.title('Método del codo')
plt.show()


# Coeficiente de silueta (Silhouette Score)
# Evalúa qué tan bien se separan los clusters (valores entre -1 y 1). 
# Cuanto más alto, mejor agrupamiento.
# El Silhouette Score mide qué tan similares son los puntos dentro de un cluster comparado con otros clusters.
# Va de –1 a 1:
# Cerca de 1: puntos bien agrupados
# Cerca de 0: puntos en el límite entre clusters
# Negativo: puntos posiblemente mal asignados
# Si al aumentar el número de clusters el Silhouette Score cae bruscamente, eso suele indicar que:
# Los clusters adicionales no aportan estructura real al grupo.
# Se está sobreajustando, separando artificialmente lo que antes era un grupo coherente.

scores = []
K_range = range(2, 11) # (excluye el último , va de 2 a 10)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    scores.append(score)

# Para entender el formato go-
# g → color verde (green)
# o → marcador en forma de círculo (circle marker) en cada punto
# - → línea continua entre los puntos
plt.plot(K_range, scores, 'go-')
plt.xlabel('Número de clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Coeficiente de Silueta')
plt.show()

#%%
# # Entrenamos un modelo KMeans con los datos de entrenamiento
# Vamos a perseverar con 3 clusters
kmeans_model = KMeans(n_clusters=3, random_state=0, n_init=100)
kmeans_model.fit(X_train)

# O bien lo hacemos automático con el mejor silhouette score:

# de scores, seleccionamos el máximo pero reportamos el valor asociado al index
# best_k = K_range[scores.index(max(scores))]  # El k con mayor Silhouette Score

# # Entrenar modelo final con best_k
# kmeans_model = KMeans(n_clusters=best_k, random_state=0, n_init=100)
# kmeans_model.fit(X_train)
    
#%%
# Predicción en el conjunto de prueba
y_pred = kmeans_model.predict(X_test)

# Análisis de la capacidad predictiva

mse1 = mean_squared_error(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print(f"El error cuadrático medio (MSE) del modelo es: {mse1}")

# Visualización de los clusters en el conjunto de prueba
plt.figure(figsize=(12, 8))
plt.scatter(X_test['Beta'], X_test['Endeudamiento'], c=y_pred, cmap='viridis', alpha=0.6)
plt.xlabel('Beta de la acción')
plt.ylabel('Índice de Endeudamiento')
plt.title('Predicción de Clusters en el conjunto de prueba')
plt.colorbar(label='Cluster')
plt.show()

#%%

# El problema es que no sabemos si se está tomando el cluster 0,1,2 en el orden de bajo,medio,alto
# Lo más probable es que esté desordenado

# Ordenemos la matriz de confusión:
# KMeans no asigna etiquetas interpretables
# El algoritmo KMeans no sabe qué significa "riesgo bajo", "riesgo medio" o "riesgo alto". 
# Simplemente asigna etiquetas arbitrarias como 0, 1 y 2 según la cercanía a centroides 
# en el espacio de features.

# Por ejemplo:
# Un punto de "riesgo bajo" puede estar en el cluster 2.
# Otro "riesgo alto" puede estar en el cluster 0.
# Por esto, debemos primero ordenar

from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

# Crear matriz de confusión original
cm = confusion_matrix(y_test, y_pred)

# Encontrar la mejor asignación entre etiquetas verdaderas y predichas
# Esta función resuelve el problema de asignación óptima 
# Algoritmo del Hungaro (nos saltaremos la explicación)
row_ind, col_ind = linear_sum_assignment(-cm)  # Usamos el negativo porque queremos maximizar

# Crear un mapa de reasignación
mapping = dict(zip(col_ind, row_ind))

# Reasignar los valores predichos
y_pred_aligned = [mapping[label] for label in y_pred]

# Nueva matriz de confusión
cm_aligned = confusion_matrix(y_test, y_pred_aligned)

# Mostrar matriz corregida
sns.heatmap(cm_aligned, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Cluster Predicho (Reasignado)')
plt.ylabel('Cluster Verdadero')
plt.title('Matriz de Confusión Corregida')
plt.show()

#%%

# Veamos como queda el data frame para ir viendo el verdadero cluster y la prediccion:
    
# Mostramos las primeras filas del DataFrame con las predicciones y las etiquetas verdaderas
resultados = pd.DataFrame({
    'Beta': X_test['Beta'],
    'Endeudamiento': X_test['Endeudamiento'],
    'Volatilidad_Ventas': X_test['Volatilidad_Ventas'],
    'Cluster Verdadero': y_test,
    'Cluster Predicho': y_pred_aligned
})

print(resultados.head())


