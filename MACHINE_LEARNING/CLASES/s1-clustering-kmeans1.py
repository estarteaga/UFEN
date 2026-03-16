"""

UNIVERSIDAD DE CHILE
DIPLOMADO EN CIENCIA DE DATOS PARA LAS FINANZAS
CURSO MACHINE LEARNING PARA FINANZAS
EXCLUSIVO USO CURSO PROFESOR GUILLERMO YAÑEZ

Clustering (ejemplo simple)

Modelo K-means

Creado por el profesor Guillermo Yañez basado en:

Machine Learning in
Business:
An Introduction to the World of Data
Science
Second Edition
John C. Hull
Capítulo 2

"""

# Un caso con 5 empresas para entender cómo se define la distancia al centroide

# Lo primero es definir los atributos o features que son las variables independientes
# típicamente para nosotros en finanzas

# Features (atributos):

liquidez = [1,1.2,1.5,2,1.8]

Endeudamiento = [0.2,0.7,1,1.2,2]

ROE = [0.02,0.05,0.1,0.15,0.2] # Rentabilidad del patrimonio


#%%

# Calculamos el promedio de cada atributo

avg_liquidez = round(sum(liquidez) / len(liquidez),2)
avg_Endeudamiento = round(sum(Endeudamiento) / len(Endeudamiento),2)
avg_ROE = round(sum(ROE) / len(ROE),4)

print("Liquidez promedio:", avg_liquidez)
print("Endeudamiento promedio:", avg_Endeudamiento)
print("ROE promedio:", avg_ROE)


#%%

# Distancia al centroide (distancia euclidea al promedio):
    # La distancia euclidea es lo que aprendimos con el triangulo de Pitágoras
    # Cuando lo vemos en dos dimensiones. 
    # Ojo, aquí son 3 ya que son 3 atributos
    
def distancia(x1, y1, z1, x2, y2, z2):
    return ((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)**0.5

# Calculemos para cada una de las 5 empresas
distances_to_centroid = []
for i in range(len(liquidez)):
    distance = distancia(liquidez[i], Endeudamiento[i], ROE[i], avg_liquidez, avg_Endeudamiento, avg_ROE)
    distances_to_centroid.append(distance)

print("Distancia a centroide: ", distances_to_centroid)

#%%

# Una tabla:
    
import pandas as pd

# Agregamos el promedio para tener una última fila con el centroide
data = {
    'Firma': ['Firma 1', 'Firma 2', 'Firma 3', 'Firma 4', 'Firma 5', 'Centroide'],
    'Liquidez': liquidez + [avg_liquidez],
    'Endeudamiento': Endeudamiento + [avg_Endeudamiento],
    'ROE': ROE + [avg_ROE],
    'Distancia a centroide': distances_to_centroid + [0]  # 0 distancia del centroide con si mismo (lo dejamos en 0)
}

df = pd.DataFrame(data)
print(df)

#%%


# ¿Qué nos falta para que sea un modelo k-means? (el más usual en clustering)

# No hemos puesto etiquetas a las empresas y agruparlas en clusters

# Asignar empresas a clústeres: Basado en las distancias al centroide, asignamos 
# cada empresa al clúster más cercano.
# Recalcular el centroide: Después de asignar las empresas a los clústeres, 
# recalculamos el centroide de cada clúster tomando el promedio de los puntos dentro de ese clúster.
# Repetir: Repetir los pasos anteriores hasta que los centroides dejen de 
# cambiar significativamente (convergencia).

#%%
# Por ejemplo:
    
from sklearn.cluster import KMeans
import pandas as pd

# Creamos un DataFrame con los datos
df = pd.DataFrame({
    'Liquidez': liquidez,
    'Endeudamiento': Endeudamiento,
    'ROE': ROE
})

#%%
# Aplicamos KMeans con 2 clústeres, por ejemplo
kmeans = KMeans(n_clusters=2, random_state=0).fit(df)

# Agregamos la columna de los clústeres asignados
df['Cluster'] = kmeans.labels_

# Mostramos los resultados
print("Centroides:\n", kmeans.cluster_centers_)
print("Predicciones de clúster:\n", df)


