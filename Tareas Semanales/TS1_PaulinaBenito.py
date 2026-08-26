# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 08:49:39 2026

@author: Paulina Benito
"""

import numpy as np 
import matplotlib.pyplot as plt

#%%
def funcion_sen( vmax , dc, k , ph, N,  fs) : 
    nn= np.arange(0,N)
    xx = dc + vmax * np.sin(k*(2*np.pi*fs/N)*nn/fs + ph)

    return (nn,xx)

#%% EJERCICIO 1
N=1000
fs=20000
k=100
ph=0
dc=0
vmax=1
nn= np.arange(0,N)

nn,xx1=funcion_sen(vmax, dc, k, ph, N, fs)

plt.figure()
plt.plot(nn,xx1)
plt.xlabel("n")
plt.ylabel("x[n]")
plt.title("Función sen de f0=2kHz")
plt.grid()
plt.show 