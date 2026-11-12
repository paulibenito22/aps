
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 5 20:56:00 2026

@author: Paulina Benito
"""

import numpy as np
from scipy import signal
from scipy.interpolate import CubicSpline
import scipy.io as sio
import matplotlib.pyplot as plt


#%% CARGAR EL ECG COMPLETO

plt.close("all")

fs_ecgconruido = 1000  # Hz

mat_struct = sio.loadmat("ECG_TP4.mat")

# Vector de una dimensión para filtrar correctamente
ecg_conruido = mat_struct["ecg_lead"].flatten()
ecg_one_lead = ecg_conruido

N = len(ecg_conruido)
muestras = np.arange(N)
t_ecgconruido = muestras / fs_ecgconruido

# Detecciones QRS
qrs_indices = mat_struct["qrs_detections"].flatten().astype(int)

# Si las detecciones están expresadas como índices MATLAB,
# descomentá la siguiente línea:
# qrs_indices = qrs_indices - 1

qrs_indices = np.unique(qrs_indices)
qrs_indices = qrs_indices[
    (qrs_indices >= 0) & (qrs_indices < N)
]


#%% FILTRO DE MEDIANA SOBRE TODO EL ECG

b200 = signal.medfilt(ecg_conruido, kernel_size=201)
b600 = signal.medfilt(b200, kernel_size=601)

ecg_medfilt = ecg_conruido - b600

# Primer gráfico: etapas de mediana
plt.figure(figsize=(14, 5))

plt.plot(
    t_ecgconruido, ecg_conruido,
    color="gray", alpha=0.6, label="ECG sin filtrar"
)
plt.plot(
    t_ecgconruido, b200,
    color="tab:blue", label="Mediana — 201 muestras"
)
plt.plot(
    t_ecgconruido, b600,
    color="tab:orange", linewidth=2,
    label="Segunda mediana — 601 muestras"
)

plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.title("Estimación de línea de base por mediana — ECG completo")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# Segundo gráfico: ECG corregido por mediana
plt.figure(figsize=(14, 5))

plt.plot(
    t_ecgconruido, ecg_conruido,
    color="gray", alpha=0.6, label="ECG sin filtrar"
)
plt.plot(
    t_ecgconruido, ecg_medfilt,
    color="tab:blue", label="ECG corregido por mediana"
)

plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.title("Corrección por mediana — ECG completo")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()


#%% SPLINE SOBRE EL ECG ORIGINAL: UN PUNTO POR LATIDO

adelanto_ms = 60
n0 = int(round(adelanto_ms * fs_ecgconruido / 1000))

puntos_pq = qrs_indices - n0
puntos_pq = puntos_pq[puntos_pq >= 0]

if len(puntos_pq) < 4:
    raise ValueError("No hay suficientes puntos PQ para el spline.")

t_pq = t_ecgconruido[puntos_pq]
amplitud_pq = ecg_conruido[puntos_pq]

spline = CubicSpline(
    t_pq, amplitud_pq,
    bc_type="natural",
    extrapolate=False
)

# En los extremos se mantiene el valor del punto más cercano
t_evaluacion = np.clip(t_ecgconruido, t_pq[0], t_pq[-1])

base_spline = spline(t_evaluacion)
ecg_filtrado_spline = ecg_conruido - base_spline

# Primer gráfico: puntos PQ y spline
plt.figure(figsize=(14, 5))

plt.plot(
    t_ecgconruido, ecg_conruido,
    color="gray", alpha=0.7, label="ECG sin filtrar"
)
plt.plot(
    t_ecgconruido, base_spline,
    color="tab:orange", linewidth=2,
    label="Línea de base por spline"
)
plt.plot(
    t_pq, amplitud_pq,
    "og", markersize=3, label="Puntos PQ"
)
plt.plot(
    t_ecgconruido[qrs_indices], ecg_conruido[qrs_indices],
    "xr", markersize=3, label="Detecciones QRS"
)

plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.title("Estimación por spline — ECG completo")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# Segundo gráfico: ECG corregido por spline
plt.figure(figsize=(14, 5))

plt.plot(
    t_ecgconruido, ecg_conruido,
    color="gray", alpha=0.6, label="ECG sin filtrar"
)
plt.plot(
    t_ecgconruido, ecg_filtrado_spline,
    color="tab:blue", label="ECG corregido por spline"
)

plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.title("Corrección por spline — ECG completo")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()


#%% MÉTODO DEL PROFE: PROMEDIOS PQ SOBRE EL ECG CON MEDIANA

# Conservar QRS cuyas ventanas no comiencen antes del registro
qrd_det = qrs_indices[qrs_indices >= 60]

start_win = qrd_det - 60
end_win = qrd_det - 40

pq_hat = np.array([
    np.mean(ecg_medfilt[a:b])
    for a, b in zip(start_win, end_win)
])

pq_t_hat = (start_win + end_win) / 2

if len(pq_t_hat) < 4:
    raise ValueError("No hay suficientes ventanas PQ para el spline.")

# Graficar los promedios y las detecciones
plt.figure(figsize=(14, 5))

plt.plot(
    t_ecgconruido, ecg_medfilt,
    label="ECG corregido por mediana"
)
plt.plot(
    qrd_det / fs_ecgconruido, ecg_medfilt[qrd_det],
    "xr", markersize=3, label="Detecciones QRS"
)
plt.plot(
    pq_t_hat / fs_ecgconruido, pq_hat,
    "og", markersize=3, label="Promedios PQ"
)

plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.title("Promedios PQ — ECG completo")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()


#%% SPLINE DE LOS PROMEDIOS PQ

spline_promedios = CubicSpline(
    pq_t_hat, pq_hat,
    extrapolate=False
)

# Evaluar en todas las muestras sin extrapolar los extremos
muestras_evaluacion = np.clip(
    muestras, pq_t_hat[0], pq_t_hat[-1]
)

base_spline_residual = spline_promedios(muestras_evaluacion)

# Corrección adicional sobre el ECG ya filtrado por mediana
ecg_spline = ecg_medfilt - base_spline_residual

# Mostrar la línea de base residual
plt.figure(figsize=(14, 5))

plt.plot(
    t_ecgconruido, ecg_medfilt,
    color="gray", alpha=0.7,
    label="ECG corregido por mediana"
)
plt.plot(
    pq_t_hat / fs_ecgconruido, pq_hat,
    "og", markersize=3, label="Promedios PQ"
)
plt.plot(
    t_ecgconruido, base_spline_residual,
    color="tab:orange", linewidth=2,
    label="Spline residual"
)

plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.title("Spline después de la mediana — ECG completo")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# Mostrar el ECG final
plt.figure(figsize=(14, 5))

plt.plot(
    t_ecgconruido, ecg_spline,
    color="tab:blue",
    label="ECG corregido por mediana + spline"
)

plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.title("Corrección por mediana + spline — ECG completo")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.show()