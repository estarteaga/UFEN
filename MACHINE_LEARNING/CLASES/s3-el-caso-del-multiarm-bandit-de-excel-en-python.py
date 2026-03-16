# -*- coding: utf-8 -*-
"""
UNIVERSIDAD DE CHILE
DIPLOMADO EN CIENCIA DE DATOS PARA LAS FINANZAS
CURSO MACHINE LEARNING PARA FINANZAS
EXCLUSIVO USO CURSO PROFESOR GUILLERMO YAÑEZ

Reinforcement learning: El caso multiarm bandit

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
# Configuración
np.random.seed(42)
n_arms = 4
n_rounds = 5000
# Conviene comenzar con epsilon = 1 para obtener varias rondas con elección 
# aleatoria y así va probando todas las palancas sin concentrarse demasiado rápido
# en una
epsilon = 1.0

#%%
# Para el decay, lo vamos a ir llevando a cero mediante una función exponencial 
# decreciente:

def epsilon_dec(t, epsilon_start=1.0, epsilon_end=0.001):
    """
    Calcula el valor de epsilon en el paso t, partiendo de epsilon_start y 
    decayendo hacia epsilon_end en total_rounds.
    
    Parámetros:
        t (int o array): número de iteración(es)
        epsilon_start (float): valor inicial de epsilon (por ejemplo 1.0)
        epsilon_end (float): valor objetivo mínimo de epsilon (por defecto 0.001)
        
    Retorna:
        float o array: valor(es) de epsilon en el paso t
    """
    k = -np.log(epsilon_end / epsilon_start) / t
    return 1 - epsilon_start * np.exp(-k * t)

epsilon_decay = epsilon_dec(t=n_rounds, epsilon_start=1.0, epsilon_end=0.001)

# Esto último calcula el decay a aplicar por periodo hasta el 5000 que es 0.999


#%%

# Valores esperanza de ganancia de las palancas
true_means = [1.2 , 1, 0.8 ,1.4]

#%%
# Inicialización
counts = [0] * n_arms
values = [0.0] * n_arms # Estos son los Q o recompensas acumuladas por palanca
history = []

# Simulación
for t in range(1, n_rounds + 1): # Aquí la numeración comienza en 1 hasta 5000
    random_number = np.random.rand()
    if random_number < epsilon:
        chosen_arm = np.random.randint(n_arms)
    else:
        chosen_arm = np.argmax(values)
        # np.random.randn() es una normal std 0,1
    reward_random = np.random.randn() + true_means[chosen_arm]
    # counts[chosen_arm] += 1 incrementa en uno el contador de veces que 
    # se ha elegido una determinada palanca
    counts[chosen_arm] += 1
    n = counts[chosen_arm]
    # values es una lista donde se guarda la recompensa media estimada para cada 
    # palanca.
    # Cada vez que se elige una palanca (chosen_arm), se usa values[chosen_arm] 
    # para consultar cuál es la estimación corriente de su valor esperado.
    # Luego, este valor (value) se usa en la fórmula de actualización de la media:
    value = values[chosen_arm] # obtiene valor estimado de la palanca elegida en esa ronda
    new_value = value + (reward_random - value) / n
    values[chosen_arm] = new_value

    history.append({
        'Ronda': t,
        'Epsilon': epsilon,
        'Aleatorio para palanca': random_number,
        'Palanca elegida': chosen_arm + 1,
        'Aleatorio para recompensa': reward_random - true_means[chosen_arm],
        'Recompensa': reward_random,
        'Ganancia media': np.mean([h['Recompensa'] for h in history]),
        **{f'Conteo Palanca {i+1}': counts[i] for i in range(n_arms)},
        **{f'Valor estimado Palanca {i+1}': values[i] for i in range(n_arms)},
    })

    # Aplicar decaimiento a epsilon
    epsilon *= epsilon_decay

# Guardar a Excel
df = pd.DataFrame(history)


#%% Gráficos

# 1. Evolución del valor estimado de cada palanca
plt.figure(figsize=(12, 6))
for i in range(n_arms):
    plt.plot(df['Ronda'], df[f'Valor estimado Palanca {i+1}'], label=f'Palanca {i+1}')
plt.title('Evolución del Valor Estimado de Cada Palanca')
plt.xlabel('Ronda')
plt.ylabel('Valor Estimado')
plt.legend()
plt.grid(True)
plt.show()

# 2. Distribución de las recompensas por cada palanca
plt.figure(figsize=(12, 6))
recompensas_por_palanca = df[['Recompensa', 'Palanca elegida']].groupby('Palanca elegida').mean()
plt.bar(recompensas_por_palanca.index, recompensas_por_palanca['Recompensa'], color='skyblue')
plt.title('Distribución de Recompensas por Palanca')
plt.xlabel('Palanca')
plt.ylabel('Recompensa Media')
plt.xticks(range(1, n_arms + 1))
plt.grid(True)
plt.show()

# 3. Evolución de epsilon
plt.figure(figsize=(12, 6))
plt.plot(df['Ronda'], df['Epsilon'], color='green')
plt.title('Evolución de Epsilon')
plt.xlabel('Ronda')
plt.ylabel('Epsilon')
plt.grid(True)
plt.show()

# 4. Conteo de elecciones por palanca
plt.figure(figsize=(12, 6))
conteo_palancas = df[[f'Conteo Palanca {i+1}' for i in range(n_arms)]].iloc[-1]
plt.bar(conteo_palancas.index, conteo_palancas, color='orange')
plt.title('Conteo de Elecciones por Palanca')
plt.xlabel('Palanca')
plt.ylabel('Número de Elecciones')
plt.xticks(range(1, n_arms + 1))
plt.grid(True)
plt.show()


#%%
df.to_excel("simulacion_bandido_decay.xlsx", index=False)
print("Simulación guardada en 'simulacion_bandido_decay.xlsx'")


#%%

"""
Esta parte es opcional
Hasta aquí podríamos decir que hemos realizado el código correctamente
pero resulta útil agregar un algoritmo para detenernos una vez que detectamos la mejor palanca 
la 4 en este caso.
Repitamos todo el código

"""

#%% Configuración
np.random.seed(42)
# Mantengamos los mismos parámetros iniciales ya definidos

threshold = 0.05  # Umbral para la diferencia en el valor estimado
consecutive_rounds = 100  # Número de rondas consecutivas para detenerse

#%% Inicialización
counts = [0] * n_arms
values = [0.0] * n_arms
history = []
no_improvement_rounds = 0  # Contador de rondas sin mejora

#%% Simulación
for t in range(1, n_rounds + 1):
    random_number = np.random.rand()
    if random_number < epsilon:
        chosen_arm = np.random.randint(n_arms)
    else:
        chosen_arm = np.argmax(values)
    
    reward_random = np.random.randn() + true_means[chosen_arm]
    counts[chosen_arm] += 1
    n = counts[chosen_arm]
    value = values[chosen_arm]
    new_value = value + (reward_random - value) / n
    values[chosen_arm] = new_value

    # Almacenamos los resultados de cada ronda
    history.append({
        'Ronda': t,
        'Epsilon': epsilon,
        'Aleatorio para palanca': random_number,
        'Palanca elegida': chosen_arm + 1,
        'Aleatorio para recompensa': reward_random - true_means[chosen_arm],
        'Recompensa': reward_random,
        'Ganancia media': np.mean([h['Recompensa'] for h in history]) if history else 0,
        **{f'Conteo Palanca {i+1}': counts[i] for i in range(n_arms)},
        **{f'Valor estimado Palanca {i+1}': values[i] for i in range(n_arms)},
    })

    # Verificar si la diferencia en el valor estimado de la mejor palanca es mayor que el umbral
    best_arm_value = max(values)
    diff = [best_arm_value - value for value in values]
    if diff[3] > threshold:  # Comprobar si la palanca 4 (índice 3) es significativamente mejor
        no_improvement_rounds += 1
    else:
        no_improvement_rounds = 0

    # Si la palanca 4 ha sido significativamente mejor durante "consecutive_rounds" rondas, detener
    if no_improvement_rounds >= consecutive_rounds:
        print(f"Deteniéndose en la ronda {t}. La palanca 4 es la mejor.")
        break

    # Aplicar decaimiento a epsilon
    epsilon *= epsilon_decay

df = pd.DataFrame(history)

#%% Gráficos

# 1. Evolución del valor estimado de cada palanca
plt.figure(figsize=(12, 6))
for i in range(n_arms):
    plt.plot(df['Ronda'], df[f'Valor estimado Palanca {i+1}'], label=f'Palanca {i+1}')
plt.title('Evolución del Valor Estimado de Cada Palanca')
plt.xlabel('Ronda')
plt.ylabel('Valor Estimado')
plt.legend()
plt.grid(True)
plt.show()

# 2. Distribución de las recompensas por cada palanca
plt.figure(figsize=(12, 6))
recompensas_por_palanca = df[['Recompensa', 'Palanca elegida']].groupby('Palanca elegida').mean()
plt.bar(recompensas_por_palanca.index, recompensas_por_palanca['Recompensa'], color='skyblue')
plt.title('Distribución de Recompensas por Palanca')
plt.xlabel('Palanca')
plt.ylabel('Recompensa Media')
plt.xticks(range(1, n_arms + 1))
plt.grid(True)
plt.show()

# 3. Evolución de epsilon
plt.figure(figsize=(12, 6))
plt.plot(df['Ronda'], df['Epsilon'], color='green')
plt.title('Evolución de Epsilon')
plt.xlabel('Ronda')
plt.ylabel('Epsilon')
plt.grid(True)
plt.show()

# 4. Conteo de elecciones por palanca
plt.figure(figsize=(12, 6))
conteo_palancas = df[[f'Conteo Palanca {i+1}' for i in range(n_arms)]].iloc[-1]
plt.bar(conteo_palancas.index, conteo_palancas, color='orange')
plt.title('Conteo de Elecciones por Palanca')
plt.xlabel('Palanca')
plt.ylabel('Número de Elecciones')
plt.xticks(range(1, n_arms + 1))
plt.grid(True)
plt.show()
