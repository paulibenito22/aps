# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 20:55:58 2026

@author: Paulina Benito
"""

import numpy as np
from scipy import signal as sig

import matplotlib.pyplot as plt
   
import scipy.io as sio
from scipy.io.wavfile import write


##################
# Lectura de ECG #
##################

fs_ecg = 1000 # Hz

##################
## ECG con ruido
##################

# para listar las variables que hay en el archivo
sio.whosmat('ECG_TP4.mat')
mat_struct = sio.loadmat('./ECG_TP4.mat')

ecg_one_leadruido = mat_struct['ecg_lead'].flatten()*1.0
N = len(ecg_one_leadruido)
ecg_one_leadruido = np.squeeze(ecg_one_leadruido)
# hb_1 = mat_struct['heartbeat_pattern1']
# hb_2 = mat_struct['heartbeat_pattern2']

# plt.figure()
# plt.plot(ecg_one_leadruido[5000:12000])
# plt.title("ecg con ruido")
# #[5000:12000]

# plt.figure()
# plt.plot(hb_1)
# plt.title("heartbeat_pattern1")

# plt.figure()
# plt.plot(hb_2)
# plt.title("heartbeat_pattern2")

# =========================
# FILTRO PASA BAJOS
# =========================

wp = 40
ws = 50
gpass = 3
gstop = 40

sos = sig.iirdesign(
    wp=wp,
    ws=ws,
    gpass=gpass,
    gstop=gstop,
    analog=False,
    ftype='butter',
    output='sos',
    fs=fs_ecg
)

#cada fila corresponde a orden 2 entonces segun la cantidad de filas tendre el orden de la transferencia
# Respuesta en frecuencia
w, H = sig.sosfreqz(sos, worN=2048, fs=fs_ecg)

Hdb = 20*np.log10(np.abs(H))

plt.figure()
plt.plot(w, Hdb)

plt.axvline(wp, linestyle='--', label='wp = 40 Hz')
plt.axvline(ws, linestyle='--', label='ws = 50 Hz')

plt.axhline(-3, linestyle='--', label='-3 dB')
plt.axhline(-40, linestyle='--', label='-40 dB')

plt.xlim(0, 100)
plt.ylim(-80, 5)

plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud [dB]')
plt.title('Respuesta del filtro Butterworth')
plt.grid()
plt.legend()

plt.show()

#%%
#la ecuacion de diferencias finitas es lo que me ayuda a aplicar el filtro
#es importante cuando usas varios bloques, guardar los condiciones finales para usarlas como iniciales del prox 
#esto es para la funcion sosfilt_zi 
#las condiciones iniciales es el transitorio
# Aplicar filtro al ECG
ecg_filtrado = sig.sosfiltfilt(sos, ecg_one_leadruido)

# Vector de tiempo
t = np.arange(len(ecg_one_leadruido)) / fs_ecg

# Graficar un tramo, por ejemplo de 5 a 12 segundos
inicio = 5000
fin = 7000

plt.figure()

plt.plot(t[inicio:fin],
         ecg_one_leadruido[inicio:fin],
         label='ECG con ruido')

plt.plot(t[inicio:fin],
         ecg_filtrado[inicio:fin],
         label='ECG filtrado')

plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.title('ECG antes y después del filtrado')
plt.grid()
plt.legend()

plt.show()

#%%
plt.figure()
plt.plot(t[inicio:fin], ecg_one_leadruido[inicio:fin])
plt.title('ECG con ruido')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.grid()
plt.show()

plt.figure()
plt.plot(t[inicio:fin], ecg_filtrado[inicio:fin])
plt.title('ECG filtrado')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.grid()
plt.show()

print(sos)
## pARA NORMALIZAR LA PLANTILLA DIVIDO TODO PPOR NYQUIST 
## EN ESTAS APLICACIONES SIRVE TENERLA DESNORMALIZADAS

#%%
#les paso una delta al filtro

N_imp = 500

delta = np.zeros(N_imp)
delta[0] = 1

h = sig.sosfilt(sos, delta)

n = np.arange(N_imp)

plt.figure()
plt.stem(n, h)
plt.xlabel('n [muestras]')
plt.ylabel('h[n]')
plt.title('Respuesta al impulso del filtro IIR')
plt.grid()
plt.show()

