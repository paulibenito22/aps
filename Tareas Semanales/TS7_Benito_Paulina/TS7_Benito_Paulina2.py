# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 20:35:23 2026

@author: Paulina Benito
"""

import numpy as np
from scipy import signal
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


#%%
plt.close("all")
#%%
# AGARRO EL ECG

fs_ecgconruido = 1000  # Hz

mat_struct = sio.loadmat("ECG_TP4.mat")
ecg_conruido = mat_struct["ecg_lead"].flatten()
t_ecgconruido = np.arange(len(ecg_conruido)) / fs_ecgconruido


#plantilla
wp = [1, 35]     # Hz  banda de paso
ws = [0.1, 50]  # Hz  rechazo: de 0 a 1 Hz (borde en 0.01 porque 0 exacto no se puede) y de 50 Hz en adelante
gpass = 3
gstop = 40

#%%
# FILTFILT

sos_butt = signal.iirdesign(wp, ws, gpass / 2, gstop / 2, ftype="butter", fs=fs_ecgconruido, output="sos")

sos_cheby = signal.iirdesign(
    wp=wp,
    ws=ws,
    gpass=gpass / 2,
    gstop=gstop / 2,
    ftype="cheby2",
    output="sos",
    fs=fs_ecgconruido
)
#%%
# FIR (Ventanas + Cuadrados Mínimos WLS) - PASABANDA

delta_w = 5
f_nyq = fs_ecgconruido / 2
gains = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]

freqs_hamming = [0.0, ws[0], wp[0], wp[1], wp[1] + delta_w, f_nyq]
b_win_hamming = signal.firwin2(2001, freqs_hamming, gains, window="hamming", fs=fs_ecgconruido)

bands_wls = [0.0, ws[0],  #rechazo inferior
             wp[0], wp[1], #paso
             wp[1],ws[1], #transición superior
             ws[1],f_nyq]  #rechazo superior

desired_wls = [0.0, 0.0,   #rechazo inferior ganancia 0
               1.0, 1.0,   #paso ganancia 1
               1.0, 0.0,   #trancision el descenso lineal deseado
               0.0,0.0]     

weights_wls = [1000.0,
               1.0,
               0.01,      #en la transición prioridad pequeña 
               1000.0]
b_wls = signal.firls(2001, bands_wls, desired_wls, weight=weights_wls, fs=fs_ecgconruido)

print(f"FIR hamming diseñado: {len(b_win_hamming)} coeficientes")
print(f"FIR WLS (firls) diseñado: {len(b_wls)} coeficientes")
#%% Calcular las respuestas antes de graficarlas

# Grilla logarítmica común para todos los filtros
frecuencias = np.concatenate((
    [0.0],
    np.logspace(
        np.log10(0.0001),
        np.log10(fs_ecgconruido / 2),
        num=65536
    )
))

# FIR Hamming
f_hamm, h_hamm = signal.freqz(
    b_win_hamming,
    a=1,
    worN=frecuencias,
    fs=fs_ecgconruido
)

modulo_hamm_db = 20 * np.log10(
    np.maximum(np.abs(h_hamm), 1e-12)
)

fase_hamm = np.unwrap(np.angle(h_hamm))

# FIR mínimos cuadrados
f_wls, h_wls = signal.freqz(
    b_wls,
    a=1,
    worN=frecuencias,
    fs=fs_ecgconruido
)

modulo_wls_db = 20 * np.log10(
    np.maximum(np.abs(h_wls), 1e-12)
)

fase_wls = np.unwrap(np.angle(h_wls))

# IIR Butterworth
f_butt, h_butt = signal.sosfreqz(
    sos_butt,
    worN=frecuencias,
    fs=fs_ecgconruido
)

# Respuesta efectiva de sosfiltfilt
modulo_butt_db = 20 * np.log10(
    np.maximum(np.abs(h_butt)**2, 1e-12)
)

# Fase de una pasada
fase_butt = np.unwrap(np.angle(h_butt))
#CHEBY
f_cheby, h_cheby = signal.sosfreqz(
    sos_cheby,
    worN=frecuencias,
    fs=fs_ecgconruido
)

# Respuesta efectiva de sosfiltfilt
modulo_cheby_db = 20 * np.log10(
    np.maximum(np.abs(h_cheby)**2, 1e-12)
)

#%%
# # calculos

# # FIR Hamming
# f_hamm, h_hamm = signal.freqz(b_win_hamming, a=1, worN=8192, fs=fs_ecgconruido)
# modulo_hamm_db = 20 * np.log10(np.maximum(np.abs(h_hamm), 1e-12))
# fase_hamm = np.unwrap(np.angle(h_hamm))

# # FIR WLS
# f_wls, h_wls = signal.freqz(b_wls, a=1, worN=2**20, fs=fs_ecgconruido)
# modulo_wls_db = 20 * np.log10(np.maximum(np.abs(h_wls), 1e-12))
# fase_wls = np.unwrap(np.angle(h_wls))

# # IIR Butterworth (sosfreqz al cuadrado en magnitud por filtfilt -> 2 * modulo_db)
# f_butt, h_butt = signal.sosfreqz(sos_butt, worN=2**20, fs=fs_ecgconruido)
# modulo_butt_db = 2 * (20 * np.log10(np.maximum(np.abs(h_butt), 1e-12)))
# fase_butt = np.unwrap(np.angle(h_butt))
#%%
# respuesta en modulo

fig, ax = plt.subplots(figsize=(11, 6))

ax.plot(f_butt, modulo_butt_db, label="IIR Butterworth", color="red", linestyle="--", lw=1.6)
ax.plot(f_hamm, modulo_hamm_db, label="FIR Hamming", color="crimson", linestyle="-", lw=1.6)
ax.plot(f_wls, modulo_wls_db, label="FIR Cuadrados Mínimos (WLS)", color="darkmagenta", linestyle="-", lw=1.6)
ax.plot(
    f_cheby, modulo_cheby_db,
    color="tab:green",
    linestyle="--",
    label="IIR Chebyshev II — sosfiltfilt"
)
f_max = fs_ecgconruido / 2
y_min = -100
y_max = 5

# Zona prohibida de la banda de paso: por debajo de -gpass dB, entre wp[0] y wp[1]
rect_pasante = Rectangle((wp[0], y_min), wp[1] - wp[0], -gpass - y_min, facecolor="red", alpha=0.15, edgecolor="red", hatch="xxx", lw=1.0, label="Zona prohibida")

# Zona prohibida de rechazo inferior: por encima de -gstop dB, de 0 a ws[0]
rect_rechazo_inf = Rectangle((0, -gstop), ws[0], y_max + gstop, facecolor="red", alpha=0.15, edgecolor="red", hatch="xxx", lw=1.0)

# Zona prohibida de rechazo superior: por encima de -gstop dB, de ws[1] en adelante
rect_rechazo_sup = Rectangle((ws[1], -gstop), f_max - ws[1], y_max + gstop, facecolor="red", alpha=0.15, edgecolor="red", hatch="xxx", lw=1.0)

ax.add_patch(rect_pasante)
ax.add_patch(rect_rechazo_inf)
ax.add_patch(rect_rechazo_sup)

# Bordes de las cuatro bandas
ax.axvline(ws[0], color="gray", linestyle=":", lw=1.2)
ax.axvline(wp[0], color="gray", linestyle=":", lw=1.2)
ax.axvline(wp[1], color="gray", linestyle=":", lw=1.2)
ax.axvline(ws[1], color="gray", linestyle=":", lw=1.2)

ax.axhline(-gpass, color="forestgreen", linestyle=":", lw=1.2, label=f"Atenuación máxima en paso: {gpass} dB")
ax.axhline(-gstop, color="firebrick", linestyle=":", lw=1.2, label=f"Atenuación mínima en rechazo: {gstop} dB")

ax.set_xlim(0, 100)
ax.set_ylim(-80, y_max)
ax.set_title("Respuesta en módulo y plantilla de tolerancia")
ax.set_xlabel("Frecuencia [Hz]")
ax.set_ylabel(r"Módulo [dB]: $20\log_{10}|H(f)|$")
ax.grid(True, linestyle="--", alpha=0.6)

ax.text((wp[0] + wp[1]) / 2, -8, f"Banda de paso\n{wp[0]} a {wp[1]} Hz", ha="center", va="top", fontsize=9)
ax.text((ws[1] + 100) / 2, -8, f"Banda de rechazo\n{ws[1]} Hz en adelante", ha="center", va="top", fontsize=9)

ax.legend(loc="lower left", fontsize=8.5, ncol=2)
fig.tight_layout()
plt.show()
# Comparar ambas grillas sobre el MISMO gráfico de módulo

f_pocos = np.linspace(
    0, fs_ecgconruido / 2,
    8192, endpoint=False
)

f_detalle = np.logspace(
    np.log10(0.0001),
    np.log10(fs_ecgconruido / 2),
    65536
)

filtros_comparar = [
    ("IIR Butterworth", sos_butt, "iir", "red"),
    ("FIR Hamming", b_win_hamming, "fir", "crimson"),
    ("FIR WLS", b_wls, "fir", "darkmagenta")
]

for nombre, coeficientes, tipo, color in filtros_comparar:

    if tipo == "iir":

        _, h_pocos = signal.sosfreqz(
            coeficientes, worN=f_pocos, fs=fs_ecgconruido
        )

        _, h_detalle = signal.sosfreqz(
            coeficientes, worN=f_detalle, fs=fs_ecgconruido
        )

        # Módulo efectivo de sosfiltfilt
        amplitud_pocos = np.abs(h_pocos)**2
        amplitud_detalle = np.abs(h_detalle)**2

    else:

        _, h_pocos = signal.freqz(
            coeficientes, worN=f_pocos, fs=fs_ecgconruido
        )

        _, h_detalle = signal.freqz(
            coeficientes, worN=f_detalle, fs=fs_ecgconruido
        )

        amplitud_pocos = np.abs(h_pocos)
        amplitud_detalle = np.abs(h_detalle)

    db_pocos = 20 * np.log10(
        np.maximum(amplitud_pocos, 1e-12)
    )

    db_detalle = 20 * np.log10(
        np.maximum(amplitud_detalle, 1e-12)
    )

    ax.plot(
        f_detalle, db_detalle,
        color=color,
        linewidth=1.6,
        label=f"{nombre} — grilla logarítmica"
    )

    # Mostrar los puntos originales solo cerca de las transiciones
    seleccion = (
        (f_pocos <= wp[0] + 0.5)
        |
        (
            (f_pocos >= wp[1] - 1)
            & (f_pocos <= ws[1] + 1)
        )
    )

    ax.plot(
        f_pocos[seleccion],
        db_pocos[seleccion],
        linestyle="none",
        marker="o",
        markersize=4,
        markerfacecolor="none",
        markeredgecolor="black",
        label=(
            "Puntos de grilla uniforme"
            if tipo == "iir" else "_nolegend_"
        )
    )

# %%
# respuesta en fase

plt.figure(figsize=(10, 5))

# Graficamos solo los filtros FIR
plt.plot(f_hamm, fase_hamm, label="FIR Hamming", color="crimson", lw=1.5)
plt.plot(f_wls, fase_wls, label="FIR Cuadrados Mínimos (WLS)", color="darkmagenta", lw=1.5)

plt.axvline(wp[0], color="forestgreen", linestyle="--", label=f"wp = {wp[0]} a {wp[1]} Hz")
plt.axvline(wp[1], color="forestgreen", linestyle="--")
plt.axvline(ws[0], color="firebrick", linestyle="--", label=f"ws = {ws[0]} y {ws[1]} Hz")
plt.axvline(ws[1], color="firebrick", linestyle="--")

plt.xlim(0, 100)
plt.title("Respuesta de fase de los filtros FIR (Fase lineal)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Fase [rad]")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(loc="best", fontsize=9)
plt.tight_layout()
plt.show()


# %%
# impulso

fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

# --- Hamming ---
n_taps_hamm = len(b_win_hamming)
t_ms_hamm = (np.arange(n_taps_hamm) / fs_ecgconruido) * 1000  # tiempo en milisegundos
markerline, stemlines, baseline = axs[0].stem(t_ms_hamm, b_win_hamming, linefmt="crimson", markerfmt="o", basefmt="gray")
plt.setp(markerline, markersize=2, color="crimson")
plt.setp(stemlines, linewidth=0.7, color="crimson")

tau_ms_hamm = ((n_taps_hamm - 1) / 2) / fs_ecgconruido * 1000
axs[0].axvline(tau_ms_hamm, color="black", linestyle="--", lw=1.0, label=f"Retardo = {tau_ms_hamm:.1f} ms ({n_taps_hamm} taps)")
axs[0].set_title("Respuesta al impulso: FIR Hamming", fontsize=10)
axs[0].set_ylabel(r"$h[n]$")
axs[0].grid(True, linestyle="--", alpha=0.5)
axs[0].legend(loc="upper right", fontsize=8)

# --- WLS ---
n_taps_wls = len(b_wls)
t_ms_wls = (np.arange(n_taps_wls) / fs_ecgconruido) * 1000
markerline, stemlines, baseline = axs[1].stem(t_ms_wls, b_wls, linefmt="darkmagenta", markerfmt="o", basefmt="gray")
plt.setp(markerline, markersize=2, color="darkmagenta")
plt.setp(stemlines, linewidth=0.7, color="darkmagenta")

tau_ms_wls = ((n_taps_wls - 1) / 2) / fs_ecgconruido * 1000
axs[1].axvline(tau_ms_wls, color="black", linestyle="--", lw=1.0, label=f"Retardo = {tau_ms_wls:.1f} ms ({n_taps_wls} taps)")
axs[1].set_title("Respuesta al impulso: FIR Cuadrados Mínimos (WLS)", fontsize=10)
axs[1].set_ylabel(r"$h[n]$")
axs[1].grid(True, linestyle="--", alpha=0.5)
axs[1].legend(loc="upper right", fontsize=8)

axs[-1].set_xlabel("Tiempo [ms]")

fig.tight_layout()
plt.show()

#%%
# filtros

# iir
ecg_filt_butt = signal.sosfiltfilt(sos_butt, ecg_conruido)

# fir
ecg_filt_winhamm = signal.lfilter(b_win_hamming, 1, ecg_conruido)
ecg_filt_wls = signal.lfilter(b_wls, 1, ecg_conruido)

# retardos fir
demora_hamm = (b_win_hamming.shape[0] - 1) // 2
demora_wls = (b_wls.shape[0] - 1) // 2

t_hamm = t_ecgconruido - (demora_hamm / fs_ecgconruido)
t_wls = t_ecgconruido - (demora_wls / fs_ecgconruido)

ecg_filt_cheby = signal.sosfiltfilt(
    sos_cheby, ecg_conruido
)
#%%
# original vs iir vs 2 fir

# Rango de visualización en muestras
n_1 = 300000
n_2 = 312000

# Segmentos de las señales para la ventana de visualización
seg_orig = ecg_conruido[n_1:n_2]

# Segmento IIR (retardo = 0 por sosfiltfilt)
seg_butt = ecg_filt_butt[n_1:n_2]

# Segmentos FIR (compensados con sus respectivas demoras)
seg_hamm = ecg_filt_winhamm[n_1 + demora_hamm : n_2 + demora_hamm]
seg_wls = ecg_filt_wls[n_1 + demora_wls : n_2 + demora_wls]

# Eje de muestras común para la ventana
n_eje = np.arange(n_1, n_2)

# Gráfico
plt.figure(figsize=(13, 6))

# Referencia: ECG con ruido
plt.plot(n_eje, seg_orig, label="Original con ruido", color="blue", lw=1.5, alpha=0.4)

# Filtro IIR (fase cero)
plt.plot(n_eje, seg_butt, label="IIR Butterworth", color="red", lw=1.2)

# Filtros FIR (con retardo compensado)
plt.plot(n_eje, seg_hamm, label=f"FIR Hamming (τ={demora_hamm})", color="limegreen", lw=1.2)
plt.plot(n_eje, seg_wls, label=f"FIR WLS (τ={demora_wls})", color="darkmagenta", lw=1.4)

plt.title(f"Comparación de filtros: Original vs IIR (fase cero) vs FIR (alineados) | Muestras [{n_1} - {n_2}]")
plt.xlabel("Muestra [n]")
plt.ylabel("Amplitud [mV]")
plt.xlim(n_1, n_2)
plt.autoscale(enable=True, axis="y")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(loc="upper right", fontsize=8.5)
plt.tight_layout()
plt.show()

#%%
# 6. RETARDO DE GRUPO

# FIR de fase lineal: retardo constante = (numtaps - 1) / 2 muestras
retardo_hamm_ms = ((len(b_win_hamming) - 1) / 2) / fs_ecgconruido * 1000
retardo_wls_ms = ((len(b_wls) - 1) / 2) / fs_ecgconruido * 1000

# IIR: se suma el retardo de grupo de cada sección SOS (una sola pasada)
f_gd = np.linspace(0.05, 100, 20000)
retardo_butt_ms = np.zeros_like(f_gd)
for seccion in sos_butt:
    _, gd_seccion = signal.group_delay((seccion[:3], seccion[3:]), w=f_gd, fs=fs_ecgconruido)
    retardo_butt_ms += gd_seccion / fs_ecgconruido * 1000
    
# Retardo de grupo de Chebyshev II: una pasada
retardo_cheby_ms = np.zeros_like(f_gd)

for seccion in sos_cheby:
    _, gd_seccion = signal.group_delay(
        (seccion[:3], seccion[3:]),
        w=f_gd,
        fs=fs_ecgconruido
    )

    retardo_cheby_ms += (
        gd_seccion / fs_ecgconruido * 1000
    )
    
    

plt.figure(figsize=(10, 5))

plt.plot(f_gd, retardo_butt_ms, label="IIR Butterworth (una pasada)", color="red", linestyle="--", lw=1.6)
plt.plot(f_gd, retardo_cheby_ms,label="IIR Chebyshev II (una pasada)",color="tab:green",linestyle="--",lw=1.6)
plt.axhline(retardo_hamm_ms, label=f"FIR Hamming ({retardo_hamm_ms:.0f} ms)", color="crimson", lw=1.6)
plt.axhline(retardo_wls_ms, label=f"FIR WLS ({retardo_wls_ms:.0f} ms)", color="darkmagenta", lw=1.6, linestyle=":")

plt.axvline(wp[0], color="forestgreen", linestyle="--", label=f"wp = {wp[0]} y {wp[1]} Hz")
plt.axvline(wp[1], color="forestgreen", linestyle="--")
plt.axvline(ws[0], color="firebrick", linestyle="--", label=f"ws = {ws[0]} y {ws[1]} Hz")
plt.axvline(ws[1], color="firebrick", linestyle="--")

plt.xlim(0, 70)
plt.ylim(0, 1700)
plt.title(r"Retardo de grupo $\tau_g(f)$")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Retardo de grupo [ms]")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(loc="best", fontsize=9)
plt.tight_layout()
plt.show()


