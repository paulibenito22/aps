# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 20:33:14 2026

@author: Paulina Benito
"""

## para llegar a z al cuadrado, z y 1 /z al cuadrado tiene que ser a=1,0,0 y b=1,1,1
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

import scipy.io as sio
from scipy.io.wavfile import write
##################
# Lectura de ECG #
##################

fs_ecg = 1000 # Hz

##################
## ECG con ruido
##################

#para listar las variables que hay en el archivo
sio.whosmat('ECG_TP4.mat')
mat_struct = sio.loadmat('./ECG_TP4.mat')

ecg_one_lead = mat_struct['ecg_lead']
N = len(ecg_one_lead)

hb_1 = mat_struct['heartbeat_pattern1']
hb_2 = mat_struct['heartbeat_pattern2']

plt.figure()
plt.plot(ecg_one_lead[5000:12000])

# plt.figure()
# plt.plot(hb_1)

# plt.figure()
# plt.plot(hb_2)

# ##################
# ## ECG sin ruido
# ##################

# ecg_one_lead = np.load('ecg_sin_ruido.npy')

# plt.figure()
# plt.plot(ecg_one_lead)

