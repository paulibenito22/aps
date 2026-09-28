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

fs_ppg = 400 # Hz

# # ##################
# # ## PPG con ruido
# # ##################

# # Cargar el archivo CSV como un array de NumPy
 #ppg = np.genfromtxt('PPG.csv', delimiter=',', skip_header=1)  # Omitir la cabecera si existe


# ##################
# ## PPG sin ruido
# ##################
ppg = np.load('ppg_sin_ruido.npy')

# plt.figure()
# plt.plot(ppg)
# plt.title("ppg sin ruido")

# #%%

# ####################
# # Lectura de audio #
# ####################

# # Cargar el archivo CSV como un array de NumPy
fs_audio1, wav_data1 = sio.wavfile.read('la cucaracha.wav')
fs_audio2, wav_data2 = sio.wavfile.read('prueba psd.wav')
fs_audio3, wav_data3 = sio.wavfile.read('silbido.wav')

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

# plt.figure(figsize=(12,6))

# for i in range(len(arrwelch_ECG)):
#     plt.plot(
#         arrwelch_ECG[i][0],
#         arrwelch_ECG[i][1],
#         label=f'K = {K[i]:.0f}'
#     )

# plt.xlabel('Frecuencia [Hz]')
# plt.ylabel('PSD')
# plt.title('Welch para distintos valores de K')
# plt.legend()
# plt.grid()

# plt.show()

# plt.figure()
# plt.plot(welch_ECG[0], welch_ECG[1])
# plt.title(f"welch ECG con K={K}")
# plt.show()

#%%ppg 
N = len(ppg)
K = np.array([20,25,28, 30])
L = N / K

arrwelch_PPG = []
for l in L:
    welch_PPG = sig.welch(
        ppg,
        fs_ppg,
        window='boxcar',
        nperseg=int(l),
        noverlap=int(l/2),
        nfft=N
    )
    
    arrwelch_PPG.append(welch_PPG)
# plt.figure(figsize=(12,6))

# for i in range(len(arrwelch_PPG)):
#     plt.plot(
#         arrwelch_PPG[i][0],
#         arrwelch_PPG[i][1],
#         label=f'K = {K[i]}'
#     )

# plt.xlabel('Frecuencia [Hz]')
# plt.ylabel('PSD')
# plt.title('Welch para distintos valores de K')
# plt.legend()
# plt.grid()
# plt.show()

#%% audio la cucaracha
N = len(wav_data1)
K = np.array([40, 43, 48, 50])
L = N / K

arrwelch_audio1 = []
for l in L:
    welch_audio1 = sig.welch(
        wav_data1,
        fs_audio1,
        window='boxcar',
        nperseg=int(l),
        noverlap=int(l/2),
        nfft=N
    )
    
    arrwelch_audio1.append(welch_audio1)
plt.figure(figsize=(14,5))

# ESPECTRO COMPLETO
# plt.subplot(1,2,1)

# for i in range(len(arrwelch_audio1)):
#     plt.plot(
#         arrwelch_audio1[i][0],
#         arrwelch_audio1[i][1],
#         label=f'K = {K[i]}'
#     )

# plt.xlabel('Frecuencia [Hz]')
# plt.ylabel('PSD')
# plt.title('PSD del audio - espectro completo')
# plt.legend()
# plt.grid()


# # ZOOM
# plt.subplot(1,2,2)

# for i in range(len(arrwelch_audio1)):
#     plt.plot(
#         arrwelch_audio1[i][0],
#         arrwelch_audio1[i][1],
#         label=f'K = {K[i]}'
#     )

# plt.xlim(0, 2100)

# plt.xlabel('Frecuencia [Hz]')
# plt.ylabel('PSD')
# plt.title('PSD del audio - detalle en bajas frecuencias')
# plt.legend()
# plt.grid()

# plt.tight_layout()
# plt.show()

#%% audio prueba psd 
N = len(wav_data2)
K = np.array([40, 43, 48, 50])
L = N / K

arrwelch_audio2 = []
for l in L:
    welch_audio2 = sig.welch(
        wav_data2,
        fs_audio2,
        window='boxcar',
        nperseg=int(l),
        noverlap=int(l/2),
        nfft=N
    )
    
    arrwelch_audio2.append(welch_audio2)
plt.figure(figsize=(14,5))

# # ESPECTRO COMPLETO
# plt.subplot(1,2,1)

# for i in range(len(arrwelch_audio2)):
#     plt.plot(
#         arrwelch_audio2[i][0],
#         arrwelch_audio2[i][1],
#         label=f'K = {K[i]}'
#     )

# plt.xlabel('Frecuencia [Hz]')
# plt.ylabel('PSD')
# plt.title('PSD del audio - espectro completo')
# plt.legend()
# plt.grid()


# # ZOOM
# plt.subplot(1,2,2)

# for i in range(len(arrwelch_audio2)):
#     plt.plot(
#         arrwelch_audio2[i][0],
#         arrwelch_audio2[i][1],
#         label=f'K = {K[i]}'
#     )

# plt.xlim(0, 2100)

# plt.xlabel('Frecuencia [Hz]')
# plt.ylabel('PSD')
# plt.title('PSD del audio - detalle en bajas frecuencias')
# plt.legend()
# plt.grid()

# plt.tight_layout(rect=[0,0.05,1,1])
# plt.figtext(0.5, 0.02,
#             "Figura 8. Estimación de la PSD mediante diferentes valores de K con el método Welch. ",
#             ha="center")
# plt.show()

#%% silbido
N = len(wav_data3)
K = np.array([43, 48, 50, 53])
L = N / K

arrwelch_audio3 = []
for l in L:
    welch_audio3 = sig.welch(
        wav_data3,
        fs_audio3,
        window='boxcar',
        nperseg=int(l),
        noverlap=int(l/2),
        nfft=N
    )
    
    arrwelch_audio3.append(welch_audio3)
plt.figure(figsize=(14,5))

# ESPECTRO COMPLETO
plt.subplot(1,2,1)

for i in range(len(arrwelch_audio3)):
    plt.plot(
        arrwelch_audio3[i][0],
        arrwelch_audio3[i][1],
        label=f'K = {K[i]}'
    )

plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD')
plt.title('PSD del audio - espectro completo')
plt.legend()
plt.grid()


# ZOOM
plt.subplot(1,2,2)

for i in range(len(arrwelch_audio3)):
    plt.plot(
        arrwelch_audio3[i][0],
        arrwelch_audio3[i][1],
        label=f'K = {K[i]}'
    )

plt.xlim(2500, 7000)

plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD')
plt.title('PSD del audio - detalle en bajas frecuencias')
plt.legend()
plt.grid()

plt.tight_layout(rect=[0,0.05,1,1])
plt.figtext(0.5, 0.02,
            "Figura 9. Estimación de la PSD mediante diferentes valores de K con el método Welch. ",
            ha="center")
plt.show()