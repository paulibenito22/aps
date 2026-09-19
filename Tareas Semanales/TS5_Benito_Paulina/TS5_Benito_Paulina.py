# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 11:35:40 2026

@author: Paulina Benito
"""


import numpy as np
from scipy import signal as sig

import matplotlib.pyplot as plt
   
import scipy.io as sio
from scipy.io.wavfile import write


#%%

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

ecg_one_leadruido = mat_struct['ecg_lead']
N = len(ecg_one_leadruido)

# hb_1 = mat_struct['heartbeat_pattern1']
# hb_2 = mat_struct['heartbeat_pattern2']

# plt.figure()
# plt.plot(ecg_one_lead[5000:12000])
# plt.title("ecg con ruido")
# #[5000:12000]

# plt.figure()
# plt.plot(hb_1)
# plt.title("heartbeat_pattern1")

# plt.figure()
# plt.plot(hb_2)
# plt.title("heartbeat_pattern2")

##################
## ECG sin ruido
##################

ecg_one_lead = np.load('ecg_sin_ruido.npy')

plt.figure()
plt.plot(ecg_one_lead)
plt.title("ecg sin ruido")

print(len(ecg_one_lead))


#%%

####################################
# Lectura de pletismografía (PPG)  #
####################################

# fs_ppg = 400 # Hz

# # ##################
# # ## PPG con ruido
# # ##################

# # # # Cargar el archivo CSV como un array de NumPy
# ppg = np.genfromtxt('PPG.csv', delimiter=',', skip_header=1)  # Omitir la cabecera si existe


# ##################
# ## PPG sin ruido
# ##################

# ppg = np.load('ppg_sin_ruido.npy')

# plt.figure()
# plt.plot(ppg)
# plt.title("ppg sin ruido")

# #%%

# ####################
# # Lectura de audio #
# ####################

# # Cargar el archivo CSV como un array de NumPy
# fs_audio, wav_data = sio.wavfile.read('la cucaracha.wav')
# fs_audio, wav_data = sio.wavfile.read('prueba psd.wav')
# fs_audio, wav_data = sio.wavfile.read('silbido.wav')

# plt.figure()
# plt.plot(wav_data)
# plt.title("sonidos")

# # si quieren oirlo, tienen que tener el siguiente módulo instalado
# # pip install sounddevice
# # import sounddevice as sd
# # sd.play(wav_data, fs_audio)

#%% Estimadores 
N=30000
L= np.array([N/33, N/35, N/37])
#nos gusto el de 35 :) 
K=N/L
arrwelch_ECG=[]
for l in L:
    welch_ECG = sig.welch(
        ecg_one_lead,
        fs_ecg,
        'boxcar',
        int(l),
        int(l/2),
        N
    )
    
    arrwelch_ECG.append(welch_ECG)


# GRAFICO

plt.figure(figsize=(12,6))

for i in range(len(arrwelch_ECG)):
    plt.plot(
        arrwelch_ECG[i][0],
        arrwelch_ECG[i][1],
        label=f'K = {K[i]:.0f}'
    )

plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD')
plt.title('Welch para distintos valores de K')
plt.legend()
plt.grid()

plt.show()

# plt.figure()
# plt.plot(welch_ECG[0], welch_ECG[1])
# plt.title(f"welch ECG con K={K}")
# plt.show()



