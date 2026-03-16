"""

UNIVERSIDAD DE SANTIAGO
CURSO MACHINE LEARNING PARA FINANZAS
EXCLUSIVO USO CURSO PROFESOR GUILLERMO YAÑEZ

Basado parcialmente en:
    
Artificial Intelligence in Finance
Dr Yves J Hilpisch | The AI Machine
http://aimachine.io


"""

#%%


import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8')
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'
np.set_printoptions(precision=4, suppress=True)


#%%

"""
Un modelo simple de Reinforcement Learning
Es un modelo del tipo probability matching

El siguiente ejemplo se basa en un juego de lanzamiento de moneda que se juega 
con una moneda cargada que cae el 80% de las veces en cara y el 20% de las veces en sello. 
El juego de lanzamiento de moneda está fuertemente sesgado para enfatizar los 
beneficios del aprendizaje en comparación con un algoritmo de referencia sin 
información. 
El algoritmo de referencia sin información, apuesta al azar y distribuye 
equitativamente 
entre cara y sello, estimando lograr una recompensa total de alrededor de 50, en promedio, 
por cada ronda o iteración (epoch) de 100 apuestas realizadas.
La idea es que al ir considerando la información, el modelo va mejorando su apuesta
No obstante, en el ejemplo inicial solo lo hará aleatorio (naif)

"""

#%%

# El espacio de posibles estados:
# 1 es cara, 0 es sello
ssp = [1, 1, 1, 1, 0] # Este es el state space
# La naturaleza elige un estado en cada ronda

# El espacio de acciones:
# Apostar a cara o sello
# El agente decide (lo haremos aleatorio y luego se puede cambiar a explorar/explotar)
asp = [1, 0] # este es el action space

#%%
# Una función para cada ronda:
# función que simula 100 rondas de apuestas, 
# y suma la recompensa total que obtiene el agente
# Ejecutaremos a su vez esta función para 15 rondas de 100 apuestas (la aplicaremos 15 veces)
def epoch():
    tr = 0 # recompensa
    for i in range(100):
        a = np.random.choice(asp) # Se elige una acción aleatoriamente
        s = np.random.choice(ssp) # Se elige un estado aleatoriamente
        if a == s:
            tr += 1 # recompensa
    return tr


#%%
# Definimos un cierto número de rondas a jugar:
    # En este caso son 15 rondas de 100 apuestas cada una
# El agente no sabe que la moneda está cargada. Solo elige aleatoriamente.

rl = np.array([epoch() for i in range(15)])
rl.size
rl

#%%
# Calculemos el promedio de recompensa:

promedio = rl.mean()
promedio

"""
Hasta aquí, todo ha sido aleatorio

El agente acierta solo cuando su acción coincide con el resultado de la moneda (estado).
Hay dos formas en que esto puede pasar:
Apostó a cara (50% de las veces) y salió cara (80%) → probabilidad conjunta:
𝑃(cara y acierto)=0.5×0.8=0.4

Apostó a sello (50%) y salió sello (20%) → probabilidad conjunta:
P(sello y acierto)=0.5×0.2=0.1

Probabilidad total de acierto por apuesta:
P(acierto)=0.4+0.1=0.5

Resultado esperado en 100 apuestas:
100×0.5=50 aciertos esperados

El agente gana cerca de 50 veces porque sus acciones son aleatorias (50/50), 
mientras que la moneda está cargada, pero el emparejamiento de ambas distribuciones 
da como resultado una tasa de acierto del 50%.

"""

#%%

    
# =============================================================================
#     Ahora el algoritmo va aprendiendo en base a lo que está pasando
#       Un modelo simple de probability matching 
# =============================================================================

# Aquí tras cada estado y la decisión tomada, el algoritmo intentará aprender e
# ir corrigiendo su decisión.
# En este modelo simple, solo aprenderá que hay un sesgo en la moneda
# Comenzamos con la hipótesis de que la moneda pudiera estar cargada.

# Usemos los mismos módulos y opciones iniciales

"""

Si el agente aprendiera que la moneda está cargada y apostara siempre a cara (1), entonces:
P(acierto) = P(apostar a cara y sale cara)=1×0.8=0.8
Ganaría 80 veces por ronda de 100, mucho más que 50.

"""

#%%
# Definición del sesgo en los resultados de la naturaleza
ssp = [1, 1, 1, 1, 0]  # Moneda sesgada, 80% cara (1) y 20% sello (0)

#%%
def epoch():
    tr = 0
    # Inicialmente las probabilidades son iguales (50/50 para cara o sello)
    asp = [1, 0]  # Lista que contiene las acciones anteriores
    prob_actions = [0.5, 0.5]  # Probabilidad inicial de elegir cara o sello

    for _ in range(100):
        # Elegir acción basada en las probabilidades actuales
        a = np.random.choice([1, 0], p=prob_actions)
        
        # Simular el estado de la naturaleza usando ssp
        s = np.random.choice(ssp)
        
        if a == s:
            tr += 1  # Incrementar la recompensa si coincide la acción con el estado

        # Registrar el estado observado en las decisiones futuras
        asp.append(s)

        # Si el estado observado fue 1 (cara), incrementar la probabilidad de elegir 1
        if s == 1:
            prob_actions[0] += 0.04  # Incrementar la probabilidad de elegir cara
            prob_actions[1] -= 0.04  # Disminuir la probabilidad de elegir sello
        else:
            prob_actions[1] += 0.04  # Incrementar la probabilidad de elegir sello
            prob_actions[0] -= 0.04  # Disminuir la probabilidad de elegir cara

        # Normalizar las probabilidades para asegurarse de que sumen 1
        prob_actions = np.clip(prob_actions, 0.01, 0.99)  # Limitar entre 0.01 y 0.99
        prob_actions = prob_actions / np.sum(prob_actions)  # Reescalar para que sumen 1

    return tr




#%%
# Resultado (Simular 15 rondas):


rl2 = np.array([epoch() for _ in range(15)])
rl2

# Ahora veamos si le ganamos al modelo puramente aleatorio
promedio2 = rl2.mean()
promedio2

"""
Si la moneda NO está cargada (es justa: 50% cara, 50% sello), 
este algoritmo "aprende mal" y en promedio seguirá ganando alrededor de 50 veces 
por cada 100 rondas, igual que un agente completamente aleatorio.

"""
# Este algoritmo es un ejemplo simple de método "exploración-explotación" 
# (en este caso, probability matching)



#%%

"""
Transformemos este código ahora en un modelo de trading de acciones muy simple.
(para tener un ejemplo simple, luego nos pasaremos a Q-learning que es lo más usual)

En este primer caso la acción es puramente aleatoria y el precio también (estado s)

"""

def epoch(stock_prices):
    tr = 0 # recompensa inicial
    asp = [-1, 0, 1]  # Acciones: -1 (vender), 0 (mantener), 1 (comprar)
    for _ in range(len(stock_prices)):
        a = np.random.choice(asp)  # Selección aleatoria de acción
        s = np.random.choice(asp)  # Selección aleatoria de estado 
        
        if a == s:
            tr += 1  # Acumular recompensa basada en la acción y 
            #          estado coincidentes (simulado)
        asp.append(s)
    return tr

#%%
# Precios de acciones 

num_periods = 100  # Número de períodos (días, horas, etc.)
min_price = 90     # Precio mínimo
max_price = 110    # Precio máximo

# Simular precios de acciones a partir de una distribución uniforme
stock_prices = np.random.uniform(low=min_price, high=max_price, size=num_periods)

#%%
# Ejecutar épocas o rondas y acumular recompensas
rl = np.array([epoch(stock_prices) for _ in range(1000)])
print("Recompensas totales por ronda:", rl)
print("Recompensa promedio por ronda:", rl.mean())



#%%

"""
Veamos ahora una representación un poco más elaborada de lo anterior
(probability matching)

En el ejemplo anterior, la acción se mueve de manera aleatoria
Por lo que no será posible ganarle al mercado pero sirve
para entender probability matching

"""

#%%
# Simulación de precios de acciones
np.random.seed(50)
num_periods = 100

# Esto genera precios con ondas (ciclos), donde comprar en mínimos y 
# vender en máximos es posible si el agente detecta el patrón.

t = np.arange(num_periods)
stock_prices = 100 + 5 * np.sin(2 * np.pi * t / 20) + np.random.normal(0, 1, size=num_periods)

plt.plot(stock_prices)
plt.title("Serie de precios simulada")
plt.xlabel("Tiempo")
plt.ylabel("Precio")
plt.grid(True)
plt.show()

#%%
# Definición de acciones
actions = [0, 1, 2]  # 0: vender, 1: mantener, 2: comprar

def epoch():
    tr = 100
    # Inicialmente las probabilidades son iguales (33.3% para cada acción)
    prob_actions = [1/3, 1/3, 1/3]  # Probabilidades iniciales para [vender, mantener, comprar]

    # Estado inicial (comienza con 1 acción)
    owned_shares = 1

    for t in range(num_periods - 1):
        # Elegir acción basada en las probabilidades actuales
        a = np.random.choice(actions, p=prob_actions)

        # Simular el estado del precio de la acción
        current_price = stock_prices[t]
        next_price = stock_prices[t + 1]

        if a == 2:  # Comprar
            if owned_shares == 0:  # Solo puede comprar si no tiene acciones
                owned_shares += 1
                tr -= current_price  # Desembolsa dinero para comprar

        elif a == 0:  # Vender
            if owned_shares > 0:  # Solo puede vender si tiene acciones (no hay posiciones cortas)
                owned_shares -= 1
                tr += next_price  # Gana dinero por la venta

        # Evaluar la recompensa
        reward = tr

        # Aprender y ajustar las probabilidades
        if reward > 0:  # Si ganó dinero, ajusta las probabilidades a favor de la acción
            if a == 2:  # Comprar
                prob_actions[2] += 0.1  # Aumentar probabilidad de comprar
                prob_actions[0] -= 0.1  # Disminuir probabilidad de vender
            elif a == 0:  # Vender
                prob_actions[0] += 0.1  # Aumentar probabilidad de vender
                prob_actions[2] -= 0.1  # Disminuir probabilidad de comprar
        else:  # Si perdió dinero, ajustar en sentido contrario
            if a == 2:  # Comprar
                prob_actions[2] -= 0.1
                prob_actions[0] += 0.1
            elif a == 0:  # Vender
                prob_actions[0] -= 0.1
                prob_actions[2] += 0.1

        # Normalizar las probabilidades para que no se vayan a los extremos
        prob_actions = np.clip(prob_actions, 0.01, 0.99)  # Limitar entre 0.01 y 0.99
        prob_actions = prob_actions / np.sum(prob_actions)  # Reescalar para que sumen 1

    return tr

#%%
# Simular 150 rondas
results = np.array([epoch() for _ in range(150)])

# Mostrar resultados
print("Resultados de cada ronda:", results)
print("Promedio de recompensas:", results.mean())


#%%

# =============================================================================
# Elaboremos un poco más con modelo tipo Thompson sampling / probability matching:
# =============================================================================
    

import numpy as np

# --- Simulación de precios: ciclo + ruido ---
np.random.seed(150)
num_periods = 100
t = np.arange(num_periods)

# Comenzamos nuevamente con precios oscilatorios (tipo función seno)
stock_prices = 100 + 5 * np.sin(2 * np.pi * t / 20) + np.random.normal(0, 1, size=num_periods)

plt.plot(stock_prices)
plt.title("Serie de precios simulada")
plt.xlabel("Tiempo")
plt.ylabel("Precio")
plt.grid(True)
plt.show()

# --- Acciones ---
SELL, HOLD, BUY = 0, 1, 2
actions = [SELL, HOLD, BUY]

# --- Parámetros de trading ---
fee = 0.05          # costo por trade (por simplicidad, 0; se puede agregar 0.05, etc.)
allow_short = False # sin venta corta
max_shares = 1     # posición discreta: 0 o 1 acción

def run_epoch():
    cash = 100.0    # Comenzamos con 100 en efectivo o el primer precio de la acción, etc
    shares = 0      # comienza sin posición esta vez, solo efectivo
    wealth = cash   # como no hay posición, wealth = cash inicial

    # Priors para función de repartición Beta por acción: α=1, β=1 (uniforme)
    alpha = np.array([1.0, 1.0, 1.0])  # SELL, HOLD, BUY
    beta  = np.array([1.0, 1.0, 1.0])

    for t in range(num_periods - 1):
        p_t = stock_prices[t]
        p_next = stock_prices[t+1]

        # Thompson sampling (probability matching Bayesiano)
        theta = np.random.beta(alpha, beta)  # una muestra por acción
        a = int(np.argmax(theta))

        # Ejecutar acción al precio actual
        cash_prev, shares_prev = cash, shares
        wealth_prev = cash + shares * p_t

        if a == BUY and shares < max_shares:
            # compra a p_t, paga fee
            cash -= (p_t + fee)
            shares += 1

        elif a == SELL and shares > 0:
            # vende a p_t, paga fee
            cash += (p_t - fee)
            shares -= 1

        elif a == HOLD:
            pass  # sin cambios

        # Valorización al cierre del paso (usamos p_next para marcar a mercado)
        wealth_curr = cash + shares * p_next
        reward = wealth_curr - wealth_prev  # contribución marginal de la decisión

        # Definir "éxito" por acción según el signo del retorno neto
        success = (reward > 0)

        # Actualizar Beta para la acción tomada
        if success:
            alpha[a] += 1.0
        else:
            beta[a]  += 1.0

        # (Opcional) pequeña amortiguación para evitar sobreconfianza:
        # alpha = alpha * 0.999 + 0.001
        # beta  = beta  * 0.999 + 0.001

        wealth = wealth_curr

    return wealth  # wealth final de la epoch

# Simular varias rondas
results = np.array([run_epoch() for _ in range(150)])
print("Wealth final por ronda:", results)
print("Promedio:", results.mean(), "Desv. Est.:", results.std())

#%%
"""

El modelo anterior no garantiza ganancia ya que toma decisiones
donde la realidad es puramente aleatoria.
Lo hizo suponiendo de que si gana,
repite la decisión , lo que es una estrategia que refuerza y además
en un contexto donde el precio es aleatorio y no tiene estructura


Hasta aquí hemos visto probability matching
y el multiarm bandit es un ejemplo de Q-learning básico tipo epsilon-greedy
Donde además el estado es fijo en cada periodo

Los Métodos populares de reinforcement learning:
    
  
- probability matching
- Q - learning
- MonteCarlo
- Redes Neuronales Profundas (DQN) 
- métodos de Gradiente de Políticas

Se pueden combinar y es usual hacerlo

"""

