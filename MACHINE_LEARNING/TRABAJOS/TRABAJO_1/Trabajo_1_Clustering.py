"""
UNIVERSIDAD DE CHILE
DIPLOMADO EN CIENCIA DE DATOS PARA LAS FINANZAS
CURSO MACHINE LEARNING PARA FINANZAS

TRABAJO 1 - CLUSTERING CON K-MEANS

Objetivo:
Segmentar clientes de la base BASE_CUSTOMER.xlsx para apoyar una decision
comercial. En este caso, se propone etiquetar clientes en grupos de valor
comercial usando ingresos por venta de vehiculos e ingresos por taller.

Flujo del analisis:
1. Carga y exploracion de datos
2. Preparacion de variables
3. Metodo del codo y silhouette score
4. Modelo K-Means
5. Interpretacion de clusters
6. Validacion simple train/test
7. Graficos y tablas de apoyo
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, mean_squared_error, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# Configuracion basica para que la salida en consola sea mas clara.
np.set_printoptions(precision=4, suppress=True)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)
sns.set_theme(style="whitegrid")


#%%
# 1. Carga de datos
# Se define la ruta del archivo de entrada y la carpeta donde se guardaran los graficos.
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "BASE_CUSTOMER.xlsx"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Se lee la base principal desde Excel.
df = pd.read_excel(DATA_PATH)

# Este bloque muestra una revision inicial de la base:
# cantidad de filas, primeras observaciones, tipos de datos y faltantes.
print("\n" + "=" * 80)
print("1. CARGA DE DATOS")
print("=" * 80)
print("Dimensiones de la base:", df.shape)
print("\nPrimeras filas:")
print(df.head())
print("\nTipos de datos:")
print(df.dtypes)
print("\nValores perdidos:")
print(df.isna().sum())
print("\nDescripcion estadistica:")
print(df[["REVENUE_TOTAL_VEHICULOS", "REVENUE_TALLER_TOTAL"]].describe())


#%%
# 2. Limpieza y preparacion
# Se trabaja sobre una copia para no alterar la base original.
df_modelo = df.copy()

# Los valores faltantes de ingreso por taller se reemplazan por la mediana.
# La mediana se usa porque es menos sensible a valores extremos.
df_modelo["REVENUE_TALLER_TOTAL"] = df_modelo["REVENUE_TALLER_TOTAL"].fillna(
    df_modelo["REVENUE_TALLER_TOTAL"].median()
)

# Se aplica transformacion logaritmica para reducir asimetria.
# Esto ayuda a que los montos muy grandes no dominen completamente el modelo.
df_modelo["LOG_REVENUE_VEHICULOS"] = np.log1p(df_modelo["REVENUE_TOTAL_VEHICULOS"])
df_modelo["LOG_REVENUE_TALLER"] = np.log1p(df_modelo["REVENUE_TALLER_TOTAL"])

# Se crea una variable adicional de ingreso total para interpretar mejor los segmentos.
df_modelo["TOTAL_REVENUE"] = (
    df_modelo["REVENUE_TOTAL_VEHICULOS"] + df_modelo["REVENUE_TALLER_TOTAL"]
)

# Estas son las variables que alimentan el algoritmo de clustering.
features = ["LOG_REVENUE_VEHICULOS", "LOG_REVENUE_TALLER"]
X = df_modelo[features]

# Estandarizar deja ambas variables en una escala comparable.
# Esto es importante porque K-Means usa distancias entre puntos.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n" + "=" * 80)
print("2. PREPARACION DE VARIABLES")
print("=" * 80)
print("\nVariables del modelo:")
print(X.head())


#%%
# 3. Busqueda de numero de clusters
# Se prueban distintos valores de k para comparar que tan bien separan los datos.
inertias = []
silhouette_scores = []
k_values_inertia = range(1, 11)
k_values_silhouette = range(2, 11)

# La inercia mide cuan compactos son los grupos.
for k in k_values_inertia:
    modelo_k = KMeans(n_clusters=k, random_state=42, n_init=50)
    modelo_k.fit(X_scaled)
    inertias.append(modelo_k.inertia_)

# El silhouette score mide que tan bien separado queda cada cluster respecto de los otros.
for k in k_values_silhouette:
    modelo_k = KMeans(n_clusters=k, random_state=42, n_init=50)
    labels_k = modelo_k.fit_predict(X_scaled)
    silhouette_scores.append(silhouette_score(X_scaled, labels_k))

# Se identifica el k con mejor silhouette score como referencia tecnica.
best_k = k_values_silhouette[silhouette_scores.index(max(silhouette_scores))]

print("\n" + "=" * 80)
print("3. SELECCION DE CLUSTERS")
print("=" * 80)
print(f"Mejor k segun silhouette score: {best_k}")
print("Silhouette scores por k:")
for k, score in zip(k_values_silhouette, silhouette_scores):
    print(f"k={k}: {score:.4f}")


#%%
# 4. Modelo final
# Se fija k=3 para construir segmentos faciles de interpretar:
# valor bajo, valor medio y valor alto.
n_clusters = 3
kmeans_final = KMeans(n_clusters=n_clusters, random_state=42, n_init=100)
clusters_raw = kmeans_final.fit_predict(X_scaled)

# El algoritmo entrega etiquetas numericas sin significado economico directo.
df_modelo["Cluster_raw"] = clusters_raw

# Se ordenan los clusters por ingreso total promedio para facilitar su interpretacion.
orden_clusters = (
    df_modelo.groupby("Cluster_raw")["TOTAL_REVENUE"].mean().sort_values().index.tolist()
)

# Aqui se reasignan las etiquetas para que 0 sea el segmento de menor valor
# y 2 sea el segmento de mayor valor.
mapping_clusters = {cluster: idx for idx, cluster in enumerate(orden_clusters)}
nombre_cluster = {
    0: "Valor bajo",
    1: "Valor medio",
    2: "Valor alto",
}

df_modelo["Cluster"] = df_modelo["Cluster_raw"].map(mapping_clusters)
df_modelo["Segmento"] = df_modelo["Cluster"].map(nombre_cluster)

# Los centroides representan el punto central de cada grupo.
centroides_escalados = kmeans_final.cluster_centers_

# Se vuelven a la escala original para interpretar los resultados en montos reales.
centroides_log = scaler.inverse_transform(centroides_escalados)
centroides_df = pd.DataFrame(centroides_log, columns=features)
centroides_df["Cluster_raw"] = range(n_clusters)
centroides_df["Cluster"] = centroides_df["Cluster_raw"].map(mapping_clusters)
centroides_df["Segmento"] = centroides_df["Cluster"].map(nombre_cluster)
centroides_df["Centroide_Revenue_Vehiculos"] = np.expm1(
    centroides_df["LOG_REVENUE_VEHICULOS"]
)
centroides_df["Centroide_Revenue_Taller"] = np.expm1(
    centroides_df["LOG_REVENUE_TALLER"]
)
centroides_df = centroides_df.sort_values("Cluster")

# Se resume cuantos clientes hay en cada segmento y cuales son sus ingresos tipicos.
print("\n" + "=" * 80)
print("4. RESULTADOS DEL MODELO")
print("=" * 80)
print("\nDistribucion por segmento:")
print(df_modelo["Segmento"].value_counts().sort_index())
print("\nCentroides interpretados en escala original:")
print(
    centroides_df[
        [
            "Cluster",
            "Segmento",
            "Centroide_Revenue_Vehiculos",
            "Centroide_Revenue_Taller",
        ]
    ]
)


#%%
# 5. Perfil de clientes por cluster
# Se construye una tabla descriptiva para comparar el comportamiento de cada grupo.
perfil_clusters = (
    df_modelo.groupby(["Cluster", "Segmento"])
    .agg(
        Clientes=("CUSTOMER_ID", "count"),
        Revenue_Vehiculos_Promedio=("REVENUE_TOTAL_VEHICULOS", "mean"),
        Revenue_Taller_Promedio=("REVENUE_TALLER_TOTAL", "mean"),
        Revenue_Total_Promedio=("TOTAL_REVENUE", "mean"),
        Revenue_Total_Mediana=("TOTAL_REVENUE", "median"),
    )
    .reset_index()
    .sort_values("Cluster")
)

print("\n" + "=" * 80)
print("5. PERFIL DE CLUSTERS")
print("=" * 80)
print(perfil_clusters)

print("\nPrimeros clientes etiquetados:")
print(
    df_modelo[
        [
            "CUSTOMER_ID",
            "KEY_ID",
            "REVENUE_TOTAL_VEHICULOS",
            "REVENUE_TALLER_TOTAL",
            "Cluster",
            "Segmento",
        ]
    ].head(15)
)

# Se guarda una tabla final con la clasificacion de cada cliente para su revision en Excel.
clientes_clasificados = df_modelo[
    [
        "CUSTOMER_ID",
        "KEY_ID",
        "REVENUE_TOTAL_VEHICULOS",
        "REVENUE_TALLER_TOTAL",
        "TOTAL_REVENUE",
        "Cluster",
        "Segmento",
    ]
].sort_values(["Cluster", "TOTAL_REVENUE"], ascending=[True, False])

clientes_clasificados.to_excel(
    OUTPUT_DIR / "clientes_clasificados.xlsx",
    index=False,
)

print("\nArchivo Excel generado con clasificacion de clientes:")
print(OUTPUT_DIR / "clientes_clasificados.xlsx")


#%%
# 6. Validacion simple train/test
# Se separan los datos en entrenamiento y prueba para revisar estabilidad del modelo.
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    df_modelo["Cluster"],
    test_size=0.2,
    random_state=42,
    stratify=df_modelo["Cluster"],
)

# Se entrena nuevamente K-Means solo con el subconjunto de entrenamiento.
kmeans_train = KMeans(n_clusters=n_clusters, random_state=42, n_init=100)
kmeans_train.fit(X_train)
y_pred_raw = kmeans_train.predict(X_test)

# Como las etiquetas de K-Means son arbitrarias, se alinean con la mejor correspondencia posible.
cm_base = confusion_matrix(y_test, y_pred_raw)
row_ind, col_ind = linear_sum_assignment(-cm_base)
mapping_test = dict(zip(col_ind, row_ind))
y_pred = np.array([mapping_test[label] for label in y_pred_raw])

# Se calculan medidas simples para comparar cluster real y cluster predicho.
mse = mean_squared_error(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 80)
print("6. VALIDACION TRAIN / TEST")
print("=" * 80)
print(f"MSE de etiquetas de cluster alineadas: {mse:.4f}")
print("\nMatriz de confusion alineada:")
print(cm)

resultados_test = pd.DataFrame(
    {
        "LOG_REVENUE_VEHICULOS": X_test[:, 0],
        "LOG_REVENUE_TALLER": X_test[:, 1],
        "Cluster_Real": y_test.to_numpy(),
        "Cluster_Predicho": y_pred,
    }
)
resultados_test["Segmento_Real"] = resultados_test["Cluster_Real"].map(nombre_cluster)
resultados_test["Segmento_Predicho"] = resultados_test["Cluster_Predicho"].map(
    nombre_cluster
)

print("\nPrimeras filas de resultados test:")
print(resultados_test.head(15))


#%%
# 7. Visualizaciones
# Grafico 1: metodo del codo y silhouette score.
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(list(k_values_inertia), inertias, "bo-")
axes[0].set_title("Metodo del codo")
axes[0].set_xlabel("Numero de clusters (k)")
axes[0].set_ylabel("Inercia")

axes[1].plot(list(k_values_silhouette), silhouette_scores, "go-")
axes[1].axvline(best_k, color="red", linestyle="--", label=f"Mejor k = {best_k}")
axes[1].set_title("Coeficiente de silueta")
axes[1].set_xlabel("Numero de clusters (k)")
axes[1].set_ylabel("Silhouette score")
axes[1].legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_elbow_silhouette.png", dpi=300, bbox_inches="tight")
plt.close()

# Grafico 2: dispersion de clientes por ingresos de vehiculos y taller.
plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=df_modelo,
    x="REVENUE_TOTAL_VEHICULOS",
    y="REVENUE_TALLER_TOTAL",
    hue="Segmento",
    palette="viridis",
    s=90,
)
plt.xscale("log")
plt.yscale("log")
plt.title("Clientes segmentados por valor comercial")
plt.xlabel("Revenue total vehiculos (escala log)")
plt.ylabel("Revenue taller total (escala log)")
plt.legend(title="Segmento")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_clientes_segmentados.png", dpi=300, bbox_inches="tight")
plt.close()

# Grafico 3: boxplots para comparar la distribucion de ingresos por segmento.
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.boxplot(
    data=df_modelo,
    x="Segmento",
    y="REVENUE_TOTAL_VEHICULOS",
    hue="Segmento",
    palette="viridis",
    legend=False,
    ax=axes[0],
)
axes[0].set_yscale("log")
axes[0].set_title("Revenue vehiculos por segmento")
axes[0].set_xlabel("Segmento")
axes[0].set_ylabel("Revenue vehiculos (escala log)")

sns.boxplot(
    data=df_modelo,
    x="Segmento",
    y="REVENUE_TALLER_TOTAL",
    hue="Segmento",
    palette="viridis",
    legend=False,
    ax=axes[1],
)
axes[1].set_yscale("log")
axes[1].set_title("Revenue taller por segmento")
axes[1].set_xlabel("Segmento")
axes[1].set_ylabel("Revenue taller (escala log)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_boxplots_segmentos.png", dpi=300, bbox_inches="tight")
plt.close()

# Grafico 4: matriz de confusion para visualizar el ajuste en el conjunto de prueba.
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Matriz de confusion de clusters alineados")
plt.xlabel("Cluster predicho")
plt.ylabel("Cluster real")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_matriz_confusion.png", dpi=300, bbox_inches="tight")
plt.close()


#%%
# 8. Conclusiones de negocio
# Se imprime una interpretacion breve orientada al uso comercial del modelo.
print("\n" + "=" * 80)
print("7. CONCLUSION DE NEGOCIO")
print("=" * 80)
print(
    "El modelo K-Means permite etiquetar clientes en tres segmentos de valor "
    "comercial: bajo, medio y alto. Esta clasificacion puede usarse para "
    "priorizar mantencion de cartera, campanas comerciales y estrategias "
    "diferenciadas de postventa."
)
print(
    "\nSugerencia practica: focalizar el segmento de valor alto con acciones "
    "de fidelizacion y cross-selling, mientras que el segmento medio puede "
    "ser trabajado con campanas de crecimiento y el segmento bajo con "
    "acciones de activacion."
)
print("\nGraficos guardados en:")
print(OUTPUT_DIR)
