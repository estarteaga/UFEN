# -*- coding: utf-8 -*-
"""
UNIVERSIDAD DE CHILE
DIPLOMADO EN CIENCIA DE DATOS PARA LAS FINANZAS
CURSO MACHINE LEARNING PARA FINANZAS
EXCLUSIVO USO CURSO PROFESOR GUILLERMO YAÑEZ



SIMULACIÓN Y REGRESIÓN LINEAL
MODELO DE FACTORES
MODELO INDICE (FAMA 1970)
FAMA & FRENCH (1992-1993)
SIMULACIÓN
MODELOS DE APRENDIZAJE SUPERVISADO

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate # Para hacer tablas con un DataFrame de pandas
import statsmodels.api as sm

# Con esta opción de numpy impedimos que salga e-00
np.set_printoptions(precision=4, suppress=True) 


#%%
# Simulemos retornos

np.random.seed(244466666) # Una semilla

# Definamos ruido para las series:
    
emkt = np.random.normal(0,0.05,1000) # de acuerdo a un modelo de factor, este es
# un portafolio razonablemente diversificado

ei = np.random.normal(0,0.06,1000)

mkt = 0.03 + emkt

ri = 0.01 + 0.8*mkt + ei


#%%

# Supongamos por un momento que la relación fuera determinística,
# la correlación sería 1 o perfecta.

# Número de observaciones
n = 1000

# Caso 1: Mkt fijo en 0.03
mkt1 = np.full(n, 0.03)

ri1 = 0.01 + 0.8 * mkt1

data1 = {"MKT": mkt1,"ri": ri1}
  
# Concatenamos las series:
    
base1 = pd.DataFrame(data1)

# Matriz de correlaciones con NumPy
corr1 = np.corrcoef(base1.T)
print("La matriz de correlaciones:")
print(corr1)

# Si quisiéramos solo el valor:
corr1val = round(corr1[0,1],2)
print("La correlación es: ",str(corr1val))

#%%

# Veamos ahora el caso con ruido (ya no es determinístico):
    
# Previamente vamos a crear un diccionario con las series:
    
data = {"MKT": mkt,"ri": ri}
  
# Concatenamos las series:
    
base = pd.DataFrame(data)

# Matriz de correlaciones con NumPy
corr = np.corrcoef(base.T)
print("La matriz de correlaciones:")
print(corr)

# Gráfico sencillo
plt.imshow(corr, cmap="bwr", interpolation="none")
plt.colorbar(label="Covarianza")
plt.xticks([0, 1], ["Mkt", "Ri"])
plt.yticks([0, 1], ["Mkt", "Ri"])
plt.title("Matriz de Covarianza")

# Agregar valores dentro de cada cuadro
for i in range(corr.shape[0]): # muestra el número de filas
    for j in range(corr.shape[1]): # muestra el número de columnas
        plt.text(j, i, f"{corr[i, j]:.4f}",
                 ha="center", va="center", color="white")
plt.show()

# Si quisiéramos solo el valor:
corrval = round(corr[0,1],2)
print("La correlación es: ",str(corrval))

# tarea: Pruebe distintos valores para la varianza de ei y cómo cambia la correlación

#%%

# Un gráfico de histograma para el mercado y nuestro activo i

plt.hist(mkt, bins=15, alpha=0.5, label='MKT')
plt.hist(ri, bins=15, alpha=0.5, label='ri')
plt.title("Histograma mercado y activo i")
plt.legend()
plt.show()

#%%

# Algo más de estadística descriptiva:
    
# Matriz de covarianza con NumPy
V = np.cov(base.T)
print("Matriz de covarianza:")
print(V)

# Gráfico sencillo
plt.imshow(V, cmap="coolwarm", interpolation="none")
plt.colorbar(label="Covarianza")
plt.xticks([0, 1], ["Mkt", "Ri"])
plt.yticks([0, 1], ["Mkt", "Ri"])
plt.title("Matriz de Covarianza")

# Agregar valores dentro de cada cuadro
for i in range(V.shape[0]): # muestra el número de filas
    for j in range(V.shape[1]): # muestra el número de columnas
        plt.text(j, i, f"{V[i, j]:.4f}",
                 ha="center", va="center", color="white")
plt.show()

#%%
# Estadística descriptiva
stats = round(base.describe(),2)
print("Cuadro estádistica descriptiva:")
print(stats)

# Vamos a hacer una tabla para la media, desv.std y el ratio de Sharpe

# Uno a uno
Emkt = base[['MKT']].mean(axis=0)
Eri = base[['ri']].mean(axis=0)

# De una vez ambas medias:
mu = base.mean(axis=0) # Esto queda como serie

std = base.std(ddof=0,axis=0) # De igual forma para la desviación estándar

sharpe = mu / std # Calculemos el ratio de Sharpe (retorno por riesgo)

indice = ['Retorno','std','Sharpe']
tabla = pd.DataFrame([mu,std,sharpe],
                      index = indice)

print(tabulate(tabla, headers=['mkt','ri'],
                                tablefmt="rst",
                                showindex="always"))

#%%

# -------- Gráfico bonito con imshow --------
fig, ax = plt.subplots(figsize=(6,4))

im = ax.imshow(tabla.values, cmap="viridis", aspect="auto")

# Barra de color
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Valor", rotation=270, labelpad=15)

# Etiquetas
ax.set_xticks(range(len(tabla.columns)))
ax.set_xticklabels(tabla.columns, fontsize=10)
ax.set_yticks(range(len(tabla.index)))
ax.set_yticklabels(tabla.index, fontsize=10)

# Mostrar valores dentro de las celdas
for i in range(tabla.shape[0]):
    for j in range(tabla.shape[1]):
        ax.text(j, i, f"{tabla.values[i, j]:.4f}",
                ha="center", va="center", color="white", fontsize=9,
                bbox=dict(facecolor="black", alpha=0.3, boxstyle="round,pad=0.2"))

ax.set_title("Tabla de Retorno, Riesgo y Sharpe", fontsize=12, pad=10)
plt.tight_layout()
plt.show()

# -------- Gráfico con seaborn --------
plt.figure(figsize=(6,2))
sns.heatmap(tabla, annot=True, fmt=".4f", cmap="Blues",
            cbar=False, linewidths=0.5, linecolor="white", # esta vez quitamos la gradiente de color y cbar
            annot_kws={"size":9, "color":"grey"})

plt.title("Tabla de Retorno, Riesgo y Sharpe", fontsize=12, pad=10)
plt.tight_layout()
plt.show()

#%%

# =============================================================================
# Regresión lineal
# =============================================================================

# Usemos statsmodels para regresión lineal:
    # Tiene la ventaja de que nos muestra la tabla con todos los valores en la consola

# Lo primero es tomar x1 y agregar una columna de 1 para que represente la constante o alfa
x=sm.add_constant(base["MKT"])

# Corremos el modelo
results=sm.OLS(base["ri"],x).fit()

# Lo mostramos en la consola
print(results.summary())

# Omnibus muestra un test F para saber si hay al menos una variable significativa distinta de 0
# Otra que B0

#%%

# Extraemos algunos datos para un cuadro con colores:
    
# Extraer tabla con coeficientes
tabla_reg = pd.DataFrame({
    "Coeficiente": results.params,
    "StdErr": results.bse,
    "t-Stat": results.tvalues,
    "p-Value": results.pvalues
})

print(tabla_reg)

# --- Visualización con seaborn ---
plt.figure(figsize=(6,3))
sns.heatmap(tabla_reg, annot=True, fmt=".4f", cmap="RdBu_r",
            cbar=False, linewidths=0.5, linecolor="white",
            annot_kws={"size":9, "color":"white"})

plt.title("Resultados de la Regresión (OLS)", fontsize=12, pad=10)
plt.tight_layout()
plt.show()

#%%

# Un gráfico de dispersión

alpha = results.params["const"]
beta = results.params["MKT"] 

plt.plot(base["MKT"], base["ri"], 'x', color='black', markersize=3) # Probar cambiar "x" por "o"
plt.plot(base["MKT"], alpha + beta*base["MKT"], color='blue', label=f"Recta OLS (β={beta:.2f})")
plt.axhline(y = float(mu["ri"]), color = 'red', linestyle = '-', label="Promedio ri")
plt.xlabel("Retorno del mercado (MKT)")
plt.ylabel("Retorno activo (ri)")
plt.title("Regresión modelo de índice: Activo vs. Mercado")
plt.legend()

plt.show()


#%%

  
# =============================================================================
# Veamos ahora el MSE para comparar modelos
# =============================================================================

# Todo lo que vimos precedentemente es dentro de muestra pero no hemos dicho nada respecto
# a la capacidad predictiva del modelo

# Esta es la base para lo que viene en machine learning

from sklearn.metrics import mean_squared_error
import statsmodels.formula.api as smf

# Ahora bien, si quisiéramos separar en periodos específicos para 
# estimación y testeo:
    
# Determinemos el corte en 80% / 20%
split_index = int(len(base) * 0.8)

# Separemos los datos en dos ventanas
train_data = base[:split_index]
test_data = base[split_index:]

# Completar el ejemplo de esta forma para determinar el MSE

# El modelo:
    
formula = "ri ~ MKT"
result = smf.ols(formula, data=train_data).fit()

# Predicción sobre datos de testeo
pred = result.predict(test_data)

# MSE en datos de testeo
mse = mean_squared_error(test_data['ri'], pred)

print(f"Error cuadrático medio del modelo : {mse}")

#%%

# AGREGAR MAE (modelo índice 1-factor) ====
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.formula.api as smf

# Ya teníamos 'base' con columnas ['MKT','ri'] y el split 80/20:
split_index = int(len(base) * 0.8)
train_data = base.iloc[:split_index].copy()
test_data  = base.iloc[split_index:].copy()

# Reestimamos:
result_1f = smf.ols("ri ~ MKT", data=train_data).fit()
pred_1f   = result_1f.predict(test_data)

mse_1f = mean_squared_error(test_data['ri'], pred_1f)
mae_1f = mean_absolute_error(test_data['ri'], pred_1f)

print("\n=== Modelo Índice (1 factor) - Métricas test ===")
print(f"MSE: {mse_1f:.6f}")
print(f"MAE: {mae_1f:.6f}")

# Cuadro resumen:
res_1f = pd.DataFrame({
    "Modelo": ["Índice (1F)"],
    "MSE": [mse_1f],
    "MAE": [mae_1f]
})
print("\nResumen métricas (1F):")
print(tabulate(res_1f, headers="keys", tablefmt="rst", showindex=False))

#%%
# Tarea: Simular modelo de Fama & French 3 factores

# SIMULACIÓN FAMA–FRENCH 3 FACTORES + REGRESIÓN + MSE/MAE ====
# Parámetros de simulación:
np.random.seed(244466666)
n = 1000

# Factores con primas medias y volatilidades razonables
emkt = np.random.normal(0, 0.05, n)
esmb = np.random.normal(0, 0.03, n)
ehml = np.random.normal(0, 0.03, n)

MKT = 0.03  + emkt   # prima de mercado ~3% 
SMB = 0.01  + esmb   # tamaño ~1%
HML = 0.012 + ehml   # valor ~1.2%

# Verdaderos parámetros del activo
alpha_true = 0.005
beta_mkt   = 0.80
beta_smb   = 0.30
beta_hml   = -0.20

# Ruido idiosincrático
eps = np.random.normal(0, 0.06, n)

# Retorno del activo conforme al modelo FF3
ri_ff3 = alpha_true + beta_mkt*MKT + beta_smb*SMB + beta_hml*HML + eps

# DataFrame FF3
ff3 = pd.DataFrame({
    "ri":  ri_ff3,
    "MKT": MKT,
    "SMB": SMB,
    "HML": HML
})

#%%
# Descriptivos rápidos :
print("\n=== Descriptivos factores (FF3) ===")
print(tabulate(ff3[["MKT","SMB","HML"]].describe().round(4), headers="keys", tablefmt="rst"))

# Regresión OLS FF3 (con constante implícita)
model_ff3 = smf.ols("ri ~ MKT + SMB + HML", data=ff3).fit()
print("\n=== Resultados OLS (FF3) - Muestra completa ===")
print(model_ff3.summary())

# Split 80/20 para métricas de generalización
split_index_ff3 = int(len(ff3) * 0.8)
train_ff3 = ff3.iloc[:split_index_ff3].copy()
test_ff3  = ff3.iloc[split_index_ff3:].copy()

model_ff3_tr = smf.ols("ri ~ MKT + SMB + HML", data=train_ff3).fit()
pred_ff3     = model_ff3_tr.predict(test_ff3)

mse_ff3 = mean_squared_error(test_ff3["ri"], pred_ff3)
mae_ff3 = mean_absolute_error(test_ff3["ri"], pred_ff3)

print("\n=== FF3 - Métricas test ===")
print(f"MSE: {mse_ff3:.6f}")
print(f"MAE: {mae_ff3:.6f}")

# Cuadro comparativo con el 1F
comp = pd.DataFrame({
    "Modelo": ["Índice (1F)", "Fama–French (3F)"],
    "MSE":    [mse_1f, mse_ff3],
    "MAE":    [mae_1f, mae_ff3]
})
print("\n=== Comparación de modelos (test) ===")
print(tabulate(comp, headers="keys", tablefmt="rst", showindex=False))

# Coeficientes estimados FF3 en una tabla tipo heatmap
tabla_reg_ff3 = pd.DataFrame({
    "Coeficiente": model_ff3.params,
    "StdErr":      model_ff3.bse,
    "t-Stat":      model_ff3.tvalues,
    "p-Value":     model_ff3.pvalues
}).round(4)

print("\nCoeficientes FF3 (muestra completa):")
print(tabulate(tabla_reg_ff3, headers="keys", tablefmt="rst"))

plt.figure(figsize=(6,3))
sns.heatmap(tabla_reg_ff3, annot=True, fmt=".4f", cmap="RdBu_r",
            cbar=False, linewidths=0.5, linecolor="white",
            annot_kws={"size":9, "color":"white"})
plt.title("Resultados de la Regresión (FF3)", fontsize=12, pad=10)
plt.tight_layout()
plt.show()

#%%

# ============================================================
# Selección automática de modelo (1F vs 3F) y predicción a 1 día
# ============================================================

from sklearn.metrics import mean_squared_error

# --- Modelo 1 factor (índice de mercado) ---
model_1f = smf.ols("ri ~ MKT", data=train_data).fit()
pred_1f  = model_1f.predict(test_data)
mse_1f   = mean_squared_error(test_data["ri"], pred_1f)

# --- Modelo Fama–French 3 factores ---
model_3f = smf.ols("ri ~ MKT + SMB + HML", data=train_ff3).fit()
pred_3f  = model_3f.predict(test_ff3)
mse_3f   = mean_squared_error(test_ff3["ri"], pred_3f)

print(f"MSE 1F: {mse_1f:.6f} | MSE FF3: {mse_3f:.6f}")

# --- Selección automática ---
if mse_1f <= mse_3f:
    best_model = model_1f
    best_name  = "Índice (1 factor)"
else:
    best_model = model_3f
    best_name  = "Fama–French 3 factores"

print(f"El mejor modelo fuera de muestra es: {best_name}")

# --- Predicción a 1 día ---
# Ejemplo: tomar el primer registro del set de test del modelo seleccionado
if best_name == "Índice (1 factor)":
    X_new = test_data.iloc[[0]][["MKT"]]
else:
    X_new = test_ff3.iloc[[0]][["MKT","SMB","HML"]]

pred_next = best_model.predict(X_new)*100
print(f"Predicción de retorno a 1 día ({best_name}): {pred_next.iloc[0]:.4f} %")

