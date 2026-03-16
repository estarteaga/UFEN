
"""

UNIVERSIDAD DE CHILE
DIPLOMADO EN CIENCIA DE DATOS PARA LAS FINANZAS
CURSO MACHINE LEARNING PARA FINANZAS
EXCLUSIVO USO CURSO PROFESOR GUILLERMO YAÑEZ

Density-Based Spatial Clustering of Applications with Noise.
DBSCAN

Density-Based → basado en densidad.
Los puntos que están en regiones con alta densidad de vecinos se agrupan en un mismo cluster.
Spatial Clustering → agrupamiento espacial.
No necesita que los clusters sean esféricos; puede encontrar formas arbitrarias (alargados, curvos).
Applications with Noise → aplicaciones con ruido.
Reconoce que algunos puntos no pertenecen a ningún cluster (se etiquetan como –1) y los trata como anomalías o ruido.

Es un modelo ideal para detectar individuos fuera de todo cluster, outliers o ruido

Un ejemplo de detección de fraude en compras públicas

Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996).
A density-based algorithm for discovering clusters in large spatial databases with noise.
In Proceedings of the Second International Conference on Knowledge Discovery and Data Mining (KDD-96), pp. 226–231.
Portland, Oregon: AAAI Press.

@inproceedings{Ester1996DBSCAN,
  author    = {Ester, Martin and Kriegel, Hans-Peter and Sander, Jörg and Xu, Xiaowei},
  title     = {A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise},
  booktitle = {Proceedings of the Second International Conference on Knowledge Discovery and Data Mining (KDD-96)},
  year      = {1996},
  pages     = {226--231},
  publisher = {AAAI Press},
  address   = {Portland, Oregon, USA},
  url       = {https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf}
}

Sander, J., Ester, M., Kriegel, HP. et al. Density-Based Clustering in Spatial Databases: 
    The Algorithm GDBSCAN and Its Applications. 
    Data Mining and Knowledge Discovery 2, 169–194 (1998). 
    https://doi.org/10.1023/A:1009745219419

"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

#%%
# ============================
# 1. Simulación de datos estilo ChileCompra pero hechos con simulación
# ============================
np.random.seed(42)

n = 200
df = pd.DataFrame({
    'ProveedorRUT': [f"prov_{i%20}" for i in range(n)],  # 20 proveedores
    'MontoAdjudicado': np.random.lognormal(mean=10, sigma=0.5, size=n), # montos sesgados
    'NumParticipantes': np.random.randint(1, 8, size=n), # 1 a 7 oferentes
    'Proceso_dias': np.random.randint(1, 60, size=n),    # duración entre 1 y 60 días
    'TamañoProveedor': np.random.choice(['pequeña','mediana','grande'], size=n, p=[0.5,0.3,0.2]),
    'Modalidad': np.random.choice(['Licitación Pública','Trato Directo','Convenio Marco'], size=n, p=[0.7,0.2,0.1])
})

#%%
# ============================
# 2. Variables adicionales de fraude
# ============================

# En el dataset simulado deberían aparecer como anomalías:
# Proveedores con FreqTratoDirecto > 0.8 (casi siempre trato directo).
# Contratos con monto extremadamente alto en relación al resto.
# Procesos con 1 solo oferente y montos grandes.

# Log del monto
df['LogMonto'] = np.log1p(df['MontoAdjudicado']) # log(1+x), evita problemas de 0
# Hay muchos contratos pequeños (miles de pesos/UF) y pocos contratos muy grandes (miles de millones).
# log comprime la diferencia
# Esa cola larga hace que un algoritmo como DBSCAN (basado en distancias) sea dominado por los valores extremos.

# Tamaño proveedor como numérico
df['TamañoProv_cat'] = df['TamañoProveedor'].map({'pequeña':1, 'mediana':2, 'grande':3})

# Binaria de trato directo
df['TratoDirecto'] = df['Modalidad'].apply(lambda x: 1 if 'Trato Directo' in str(x) else 0)

# Frecuencia de adjudicación por proveedor
count_prov = df['ProveedorRUT'].value_counts().rename('TotalProv')
df = df.merge(count_prov, left_on='ProveedorRUT', right_index=True)
df['FreqAdjudProc'] = df['TotalProv'] / n

# Frecuencia de trato directo por proveedor
freq_td = df.groupby('ProveedorRUT')['TratoDirecto'].mean().rename('FreqTratoDirecto')
df = df.merge(freq_td, left_on='ProveedorRUT', right_index=True)

#%%
# ============================
# 3. DBSCAN
# ============================

# Clusters (0,1,2,...) → comportamientos normales de compra 
# (licitaciones competitivas, montos habituales).
# Cluster -1 (ruido) → posibles anomalías:
# Montos desproporcionadamente altos.
# Pocos oferentes.
# Alta frecuencia de adjudicación.
# Uso excesivo de trato directo.

X = df[['LogMonto','NumParticipantes','Proceso_dias',
        'TamañoProv_cat','FreqAdjudProc','FreqTratoDirecto']]

scaler = StandardScaler() # Escala a normal N(0,1)
X_scaled = scaler.fit_transform(X)

# Hay otros scalers:
# MinMaxScaler
# Transforma los datos a un rango fijo (por defecto [0,1]).
# RobustScaler
# Escala usando mediana y IQR (percentiles 25–75).
# MaxAbsScaler
# Escala dividiendo por el máximo valor absoluto
# etc
    


# eps = radio de vecindad (ε).
# Define la distancia máxima que puede haber entre dos puntos para ser considerados vecinos.
# En términos prácticos:
# Si un punto está dentro de una esfera (o círculo en 2D) de radio eps alrededor de otro punto, 
# entonces son vecinos.
# Esa vecindad es la base para formar clusters.

db = DBSCAN(eps=1.5, min_samples=5)  # ajustar eps según datos reales
# min_samples (muestras mínimas) determina la densidad mínima que se requiere para 
# que un grupo de puntos forme un cluster válido.

labels = db.fit_predict(X_scaled)

# Con eps = 1.5:
# Solo los puntos a menos de 1.5 unidades de distancia se consideran conectados.
# Si eps es muy pequeño, habrá muchos puntos aislados, DBSCAN marcará casi todo como anomalía (-1).
# Si eps es muy grande, casi todos los puntos entrarán en un solo cluster, no se detecta anomalía

df['cluster'] = labels

#%%

# ==========================
# 3'. K-distance plot
# ==========================

# k-distance plot es la forma estándar de elegir un buen valor de eps para DBSCAN.

# Busca el “codo” de la curva (donde empieza a crecer más rápido).
# Ese valor en el eje Y ≈ buen candidato para eps.
# Ejemplo: si el codo está en 1.3 → probar eps=1.3.

min_samples = 5  # mismo que usarías en DBSCAN
neighbors = NearestNeighbors(n_neighbors=min_samples)
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)

# Distancia al k-ésimo vecino (última columna)
k_dist = np.sort(distances[:, min_samples-1])

# Graficar
plt.plot(k_dist)
plt.ylabel(f"Distancia al {min_samples}° vecino")
plt.xlabel("Puntos ordenados")
plt.title("K-distance plot para elegir eps en DBSCAN")
plt.show()


#%%
# ============================
# 4. Resultados
# ============================
print("Clusters detectados:", set(labels))
print("Número de anomalías (cluster = -1):", sum(df['cluster'] == -1))

# Ver algunos casos anómalos
print(df[df['cluster'] == -1].head(10))

# ============================
# 5. Visualización
# ============================
sns.scatterplot(data=df, x='LogMonto', y='NumParticipantes', hue='cluster', palette='tab10')
plt.title("DBSCAN aplicado a Compras Públicas (simulado)")
plt.show()
