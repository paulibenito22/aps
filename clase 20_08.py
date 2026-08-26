# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 20:52:54 2026

@author: Paulina Benito
"""
import numpy as np 
import matplotlib.pyplot as plt 
#%%
def funcion_sen( vmax , dc, k , ph, N,  fs) : 
    nn= np.arange(0,N)
    xx = dc + vmax * np.sin(k*(2*np.pi*fs/N)*nn/fs + ph)

    return (nn,xx)

#%%
def ruido_general(x,snr):
    Ps = np.var(x)                 
    pr = Ps / (10**(snr/10)) 
    desvio= np.sqrt(pr)
    r2= np.random.normal(0, desvio, len(x))
    xx2= x + r2
    return xx2, pr

#%%

vmax=1
fs=1000
N=1000
k=4
dc=0
ph=0
nn= np.arange(0,N)

nn,xx= funcion_sen(vmax, dc, k, ph, N, fs)
plt.figure()
plt.plot(nn,xx)
plt.xlabel("n")
plt.ylabel("x[n]")
plt.title("Función sin ruido")
plt.grid()
plt.show 


snr=20 
xx2,pr= ruido_general(xx, snr)

plt.figure()
plt.plot(nn, xx2)
plt.xlabel("n")
plt.ylabel("x[n]")
plt.title("Función con ruido")
plt.grid()
plt.show()


#calculo la fft
X = np.fft.fft(xx)
print(X[4])

#para saber que tan grande es la componente frecuencial 
mag = np.abs(X) 
plt.figure()
plt.plot(mag)
plt.xlabel("k")
plt.ylabel("|X[k]|")
plt.title("Magnitud de la FFT")
plt.grid()
plt.show()

#normalizar la fft (normalizar por N hace que cada pico tenga A/2 de amplitud)
X = np.fft.fft(xx) / N



