# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 21:30:31 2026

@author: Paulina Benito
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import signal as sig

import scipy.io as sio
from scipy.io.wavfile import write

# ----------------------------- Plantilla -----------------------------
fs = 1000.0          # Hz
f_pass = 40.0        # Hz
f_stop = 50.0       # Hz
alpha_max = 3.0      # dB  -> atenuación máxima permitida en la banda de paso (gpass)
alpha_min = 40.0     # dB  -> atenuación mínima exigida en la banda de rechazo (gstop)

ftype = 'butter'     # probá también: 'cheby1', 'cheby2', 'ellip'

# ----------------------------- Diseño --------------------------------
sos = signal.iirdesign(wp=f_pass, ws=f_stop, gpass=alpha_max, gstop=alpha_min,
                       analog=False, ftype=ftype, output='sos', fs=fs)

z, p, k = signal.sos2zpk(sos)
orden = int(np.sum(np.abs(p) > 1e-12))   # orden = cantidad de polos no triviales
print(f"Tipo: {ftype} | Orden: {orden} | Secciones SOS: {sos.shape[0]}")
print("SOS =\n", np.round(sos, 6))

# ----------------------------- Respuesta -----------------------------
w, h = signal.sosfreqz(sos, worN=8192, fs=fs)
mag_db = 20 * np.log10(np.maximum(np.abs(h), 1e-12))
fase = np.unwrap(np.angle(h))

# Verificación numérica de la plantilla
peor_paso = mag_db[w <= f_pass].min()      # el punto más atenuado dentro de la banda de paso
peor_stop = mag_db[w >= f_stop].max()      # el punto menos atenuado dentro de la banda de rechazo
cumple_paso = peor_paso >= -alpha_max - 1e-6
cumple_stop = peor_stop <= -alpha_min + 1e-6
print(f"Banda de paso:   mín |H| = {peor_paso:7.3f} dB  (exigido >= {-alpha_max} dB) -> {'OK' if cumple_paso else 'NO CUMPLE'}")
print(f"Banda de rechazo: máx |H| = {peor_stop:7.3f} dB  (exigido <= {-alpha_min} dB) -> {'OK' if cumple_stop else 'NO CUMPLE'}")

#---------------ecg------------------

#respuesta al impulso
delta=signal.unit_impulse(2000, idx=0)
deltafilt=signal.sosfilt(sos, delta, axis=0)

plt.plot(np.arange(2000), deltafilt)
plt.show()

sio.whosmat('ECG_TP4.mat')
mat_struct = sio.loadmat('./ECG_TP4.mat')

ecg = mat_struct['ecg_lead']
N = len(ecg)
#sosr=sos.reshape(11,2,1)
zi= signal.sosfilt_zi( sos ).reshape(11,2,1)

ecgfilt, zif= signal.sosfilt(sos,ecg, axis=0, zi=zi)


# ----------------------------- Gráficos ------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# --- Módulo ---
ax1.plot(w, mag_db, lw=2, label=f'{ftype}, orden {orden}')
y_bot = -alpha_min - 40
# zonas prohibidas de la plantilla (sombreadas)
ax1.fill_between([0, f_pass], -alpha_max, y_bot, color='red', alpha=0.12,
                 label='Zona prohibida')
ax1.fill_between([f_stop, fs / 2], -alpha_min, 10, color='red', alpha=0.12)
# líneas de la plantilla
ax1.axvline(f_pass, color='g', ls='--', lw=1.2, label=f'$f_{{pass}}$ = {f_pass:.0f} Hz')
ax1.axvline(f_stop, color='m', ls='--', lw=1.2, label=f'$f_{{stop}}$ = {f_stop:.0f} Hz')
ax1.hlines(-alpha_max, 0, f_pass, color='g', ls=':', lw=1.5,
           label=f'$\\alpha_{{max}}$ = {alpha_max} dB (g_pass)')
ax1.hlines(-alpha_min, f_stop, fs / 2, color='m', ls=':', lw=1.5,
           label=f'$\\alpha_{{min}}$ = {alpha_min} dB (g_stop)')
# puntos peor caso
ax1.plot(f_pass, np.interp(f_pass, w, mag_db), 'go')
ax1.plot(f_stop, np.interp(f_stop, w, mag_db), 'mo')
ax1.set_ylim(y_bot, 5)
ax1.set_ylabel('|H(f)| [dB]')
estado = 'CUMPLE' if (cumple_paso and cumple_stop) else 'NO CUMPLE'
ax1.set_title(f'Pasabajos IIR ({ftype}) – fs = {fs:.0f} Hz – Plantilla: {estado}')
ax1.grid(True, which='both', alpha=0.4)
ax1.legend(loc='lower left', fontsize=9)

# zoom de la banda de paso
axin = ax1.inset_axes([0.55, 0.55, 0.42, 0.38])
axin.plot(w, mag_db, lw=2)
axin.axhline(-alpha_max, color='g', ls=':')
axin.axvline(f_pass, color='g', ls='--')
axin.fill_between([0, f_pass], -alpha_max, -3, color='red', alpha=0.12)
axin.set_xlim(0, f_pass * 1.3)
axin.set_ylim(-3, 0.5)
axin.set_title('Zoom banda de paso', fontsize=8)
axin.tick_params(labelsize=7)
axin.grid(alpha=0.4)

# --- Fase ---
ax2.plot(w, np.degrees(fase), lw=2, color='tab:orange')
ax2.axvline(f_pass, color='g', ls='--', lw=1.2, label=f'$f_{{pass}}$')
ax2.axvline(f_stop, color='m', ls='--', lw=1.2, label=f'$f_{{stop}}$')
ax2.set_ylabel('Fase [°] (unwrap)')
ax2.set_xlabel('Frecuencia [Hz]')
ax2.grid(True, alpha=0.4)
ax2.legend(loc='lower left', fontsize=9)
ax2.set_xlim(0, fs / 2)

plt.tight_layout()
plt.savefig('respuesta_pasabajos.png', dpi=130)
plt.show()

#ecg y ecg filtrado

inicio, fin = 5000, 7000
t_ecg = np.arange(inicio, fin) / fs # Eje de tiempo en segundos

plt.figure(figsize=(12, 5))
plt.plot(t_ecg, ecg[inicio:fin], label='ECG Original (con ruido)', alpha=0.6, color='gray')
plt.plot(t_ecg, ecgfilt[inicio:fin], label='ECG Filtrado', color='blue', lw=1.8)

plt.title('Comparación en el Tiempo: ECG Original vs Filtrado')
plt.xlabel('Tiempo [segundos]')
plt.ylabel('Amplitud')
plt.grid(True, alpha=0.3)


# Calcular el espectro de frecuencias antes y después
f_orig, psd_orig = signal.welch(ecg, fs=fs, nperseg=1024, axis=0)
f_filt, psd_filt = signal.welch(ecgfilt, fs=fs, nperseg=1024, axis=0)

plt.figure(figsize=(10, 5))
plt.semilogy(f_orig, psd_orig, label='Espectro ECG Original', color='gray', alpha=0.7)
plt.semilogy(f_filt, psd_filt, label='Espectro ECG Filtrado', color='green', lw=2)

# Líneas de referencia de tu diseño
plt.axvline(f_pass, color='r', ls='--', label=f'Frecuencia de paso ({f_pass} Hz)')
plt.axvline(f_stop, color='m', ls='--', label=f'Frecuencia de parada ({f_stop} Hz)')

plt.title('Análisis de Frecuencias (PSD) - Efecto del Filtro')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Potencia Espectral [V^2/Hz]')
plt.xlim(0, fs/2) # Ver hasta la frecuencia de Nyquist
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.show()

plt.legend()
plt.show()

#%% respuesta al impulso 

N_imp = 500

delta1 = np.zeros(N_imp)
delta1[0] = 1

h = sig.sosfilt(sos, delta1)

n = np.arange(N_imp)

plt.figure()
plt.stem(n, h)
plt.xlabel('n [muestras]')
plt.ylabel('h[n]')
plt.title('Respuesta al impulso del filtro IIR')
plt.grid()
plt.show()

#%% respuesta del retardo
# =========================
# FASE Y DERIVADA DE FASE
# =========================

# Fase de H en radianes
fase = np.angle(h)

# Eliminar saltos artificiales de 2*pi
fase_unwrap = np.unwrap(fase)

# Pasar frecuencia de Hz a frecuencia angular digital [rad/muestra]
omega = 2*np.pi*w/fs

# Derivada de la fase respecto de omega
derivada_fase = np.gradient(fase_unwrap, omega)

# Retardo de grupo
retardo_grupo = -derivada_fase
plt.close("all")
plt.figure()

plt.plot(w, derivada_fase)

plt.xlabel('Frecuencia [Hz]')
plt.ylabel('dφ/dΩ [muestras]')
plt.title('Derivada de la fase')
plt.grid()

plt.xlim(0, 100)

plt.show()

