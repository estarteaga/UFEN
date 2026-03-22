# -*- coding: utf-8 -*-
"""
UNIVERSIDAD DE CHILE
DIPLOMADO EN CIENCIA DE DATOS PARA LAS FINANZAS
CURSO MACHINE LEARNING PARA FINANZAS
EXCLUSIVO USO CURSO PROFESOR GUILLERMO YAÑEZ

Ojo que tensorflow podría funcionar solo
entre versiones 3.11 y 3.12 de Python

UN MODELO DE CNN PARA DETECTAR TENDENCIA EN PRECIOS DE ACCIONES

"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

#%%

# --- 1. SIMULACIÓN DE DATOS ALCISTAS Y BAJISTAS ---
# Creamos "fotos" de precios de 10 días.
# Tendencia Alcista (Up): Números que van creciendo (ej: 1, 2, 3...) + un poco de ruido aleatorio
# Tendencia Bajista (Down): Números que van decreciendo (ej: 10, 9, 8...) + un poco de ruido

def crear_datos(n_muestras=1000):
    X = []
    y = []
    for _ in range(n_muestras):
        # Aleatoriamente decidimos si crear una muestra alcista (1) o bajista (0)
        es_alcista = np.random.randint(0, 2) 
        
        if es_alcista:
            # Crea una secuencia de 0 a 10 y le suma ruido
            # 0 (Inicio): El valor con el que empezamos. Precio de la acción el Día 1.
            # 10 (Fin): El valor al que queremos llegar. Precio el Día 10.
            # 10 (Cantidad de datos): Cuántos números queremos generar en total. 10 días.
            patron = np.maximum(0, np.linspace(0, 10, 10) + np.random.normal(0, 1, 10))
            etiqueta = 1 # 1 = Compra/Sube
        else:
            # Crea una secuencia de 10 a 0 y le suma ruido
            patron = np.maximum(0, np.linspace(10, 0, 10) + np.random.normal(0, 1, 10))
            etiqueta = 0 # 0 = Venta/Baja
            
        X.append(patron)
        y.append(etiqueta)
    
    return np.array(X), np.array(y)

#%%
# Generamos 1000 ejemplos para que la red estudie
X, y = crear_datos()

#%%
# --- 2. PREPROCESAMIENTO (CRUCIAL PARA CNN) ---
# Las CNN esperan datos en 3 dimensiones: [Muestras, Pasos de Tiempo, Características]
# Aquí: [1000 muestras, 10 días, 1 precio de cierre]
# X.shape, responde: (1000, 10).
# El primer número (0) es la cantidad de filas (muestras).
# El segundo número (1) es la cantidad de columnas (días por muestra).
X = X.reshape((X.shape[0], X.shape[1], 1))

#%%
# --- 3. ARQUITECTURA DEL MODELO  ---

model = Sequential() # (1) Iniciamos el modelo"

# --- Extracción de Características ---

# Conv1D (vs 2D): Usamos 1D porque el tiempo es una línea (ayer, hoy, mañana), 
# no un cuadrado como una imagen.

# (2) Capa Convolucional: Los "Analistas Especializados"

# "En inglés, a este proceso de pasar el filtro por encima de los datos se le llama 
# 'Sliding Window'.
# Al 'lente' o filtro que usamos lo llamarán 'Kernel', 
# y al paso que da para moverse (si avanza de uno en uno o salta días) se le llama 'Stride'."
# filters=32: Supongamos contratamos 32 analistas junior. 
# Cada uno aprenderá a buscar algo diferente en el gráfico: uno busca máximos, otro busca mínimos, otro busca líneas planas...
# kernel_size=3: Cada analista mira 3 días a la vez. Se desliza: mira días 1-2-3, luego 2-3-4, luego 3-4-5. Busca micro-tendencias.

model.add(Conv1D(filters=32,            # 32 tipos de patrones distintos a buscar (filtros deslizantes)
                 kernel_size=3,         # Ventana de observación de 3 días
                 activation='relu',     # (3) Función de activación
                 input_shape=(10, 1)))  # Recibe 10 días, 1 precio por día

# =============================================================================
# Pensemos en la función de activación de la siguiente manera:
# Analogía: Es un interruptor. 
# Si el analista encuentra un patrón interesante (valor positivo), 
# la neurona se "enciende" y pasa la información. 
# Si no encuentra nada o es ruido negativo, la neurona se apaga (se vuelve 0). 
# Ayuda a limpiar el ruido del mercado.
# =============================================================================

# (4) Capa de Pooling: El "Resumen Ejecutivo"

# Si en un periodo de 2 días hubo una señal de compra muy fuerte y una débil, 
# el Pooling descarta la débil y preserva la fuerte. 
# Reduce la cantidad de datos a procesar a la mitad, haciendo el modelo más rápido.

# =============================================================================
# Por ejemplo:
# Imaginemos esta secuencia de precios de 4 días. 
# El mercado está nervioso, sube y baja un poco, pero la tendencia de fondo es importante.
# Datos originales (4 días): [100, 102, 105, 103]
# Aplicando MaxPooling(2): La red toma parejas de días y se queda con el valor máximo (el más fuerte).
# Pareja 1: [100, 102] -> Se queda con 102.
# Pareja 2: [105, 103] -> Se queda con 105.
# Resultado (2 datos): [102, 105]
# ¿Qué logramos?
# Eliminamos el ruido: El 100 y el 103 (los valles menores) desaparecieron.
# Mantuvimos la señal: Los máximos (102 y 105) se preservaron. 
# La red sigue viendo que "la cosa va subiendo", pero con la mitad de datos que procesar.

# "Usar pool_size=2 es como tomar una foto de alta resolución y reducir su tamaño al 50% 
# para enviarla por WhatsApp. La imagen se ve más pequeña y pesa menos, 
# pero todavía se entiende perfectamente qué hay en la foto aunque esté filtrada (i.e., comprimida).
# =============================================================================

model.add(MaxPooling1D(pool_size=2))    # Reduce el tamaño a la mitad

# --- LA PARTE DEL "CEREBRO" (Clasificación) ---

# (5) Capa de Aplanamiento: Preparar los datos
# Flatten toma la estructura 3D/2D y la "aplasta" en una sola fila larga 
# de números para que pueda entrar a la siguiente fase de obtención de target (1 o 0 en nuestro caso).

model.add(Flatten())

# (6) Capa Densa (Oculta): El razonamiento

# Aquí es donde los "analistas" (las capas anteriores) le pasan sus informes al "Gerente".
# Esta capa conecta toda la información extraída (máximos, mínimos, mediana, tendencias) 
# y trata de encontrar relaciones complejas entre ellas.

model.add(Dense(10, activation='relu')) 

# (7) Capa de Salida: El veredicto final

# Solo queremos una neurona de salida. ¿Por qué? Porque la respuesta es binaria: o sube o baja.
# sigmoid: Esta función fuerza al resultado a ser un número entre 0 y 1.
# Cercano a 0 = Tendencia Bajista.
# Cercano a 1 = Tendencia Alcista.
# 0.5 = Indecisión.

model.add(Dense(1, activation='sigmoid')) 

# (8) Compilación: Las reglas del entrenamiento

# loss='binary_crossentropy': Es la forma de medir el error en preguntas de Sí/No (Binarias). 
# Si la red dice 0.9 (Sube) y el mercado bajó (0), esta función le da un castigo alto para que corrija.
# optimizer='adam': Es el algoritmo matemático que ajusta los pesos. 
# Le dice a la red exactamente cuánto ajustar sus parámetros para reducir el error rápidamente.
# Podríamos haber usado loss='MSE' pero en este caso, es util binary_crossentropy ya que es binario y 
# queremos castigar si se equivoca al decir 1 o 0 para incentivarlo a apuntar en la dirección correcta

model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

#%%
# --- 4. ENTRENAMIENTO ---
print("Entrenando la red neuronal...")
model.fit(X, y, epochs=10, batch_size=32, verbose=1)

#%%
# --- 5. PRUEBA DE PREDICCIÓN ---
# Vamos a inventar un nuevo patrón que claramente sube
nuevo_patron_subida = np.array([1, 1, 3, 4, 5, 2, 3, 7, 8, 6])
# Lo adaptamos al formato (1 muestra, 10 tiempos, 1 dato)
nuevo_patron_reshaped = nuevo_patron_subida.reshape((1, 10, 1))

prediccion = model.predict(nuevo_patron_reshaped)

print("\n--- RESULTADO ---")
print(f"Patrón de entrada: {nuevo_patron_subida}")
print(f"Probabilidad de ser Alcista (0-1): {prediccion[0][0]:.4f}")

if prediccion > 0.5:
    print("Predicción: TENDENCIA ALCISTA")
else:
    print("Predicción: TENDENCIA BAJISTA")