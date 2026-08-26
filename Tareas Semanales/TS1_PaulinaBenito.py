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

#%% EJERCICIO 1: primer señal
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

plt.plot(nn[:50], xx1[:50])
plt.grid()
plt.show()

#calculo la fft 
X = np.fft.fft(xx1)
mag = np.abs(X) 
plt.figure()
plt.plot(mag)
plt.xlabel("k")
plt.ylabel("|X[k]|")
plt.title("Magnitud de la FFT")
plt.grid()
plt.show()

#%% Segunda señal: misma senoide pero con potencia media 2W y fase pi/2
N=1000
fs=20000
k=100
ph=np.pi/2
dc=0
vmax=2
nn= np.arange(0,N)

nn,xx2=funcion_sen(vmax, dc, k, ph, N, fs)

plt.figure()
plt.plot(nn,xx2)
plt.xlabel("n")
plt.ylabel("x[n]")
plt.title("Función sen con amplitud y fase cambiada")
plt.grid()
plt.show 

plt.plot(nn[:50], xx2[:50])
plt.grid()
plt.show()











