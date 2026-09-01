# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 08:49:39 2026

@author: Paulina Benito
"""

import numpy as np 
import matplotlib.pyplot as plt
from scipy import signal

#%%
def funcion_sen( vmax , dc, k , ph, N,  fs) : 
    nn= np.arange(0,N)
    xx = dc + vmax * np.sin(k*(2*np.pi*fs/N)*nn/fs + ph)

    return (nn,xx)

#%% EJERCICIO 1: primer señal
N=1000
fs=20000 #Hz
k=100
ph=0
dc=0
vmax=1
nn= np.arange(0,N)

f0_kHz = (k * fs / N) / 1000
print(f0_kHz)

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

#%%Tercer señal (ruido normal )

def ruido_general(media, varianza, N):
    desvio= np.sqrt(varianza)
    ruido= np.random.normal(media, desvio, N)
    return ruido

N=1000
varianza= 0.1
media= 0
ruido= ruido_general(media, varianza, N)
n = np.arange(N)

plt.figure()
plt.plot(n, ruido, color="purple")
plt.xlabel("n")
plt.ylabel("Amplitud")
plt.title("Ruido normalmente distribuido")
plt.grid()
plt.show()


#%% Cuarta simulación (ruido uniforme)

def ruido_uniforme(media, varianza, N):  #lo escribí de forma general
    A = np.sqrt(3 * varianza)
    a= media -A                                    
    b= media +A
    ruido_uniforme = np.random.uniform(a, b, N)
    return ruido2

N=1000
media= 0
varianza= 0.1
ruido2= ruido_uniforme(media, varianza, N)
n = np.arange(N)

plt.hist(ruido2)
plt.figure()
plt.hist(ruido2, bins=20)
plt.xlabel("Amplitud")
plt.ylabel("Cantidad de muestras")
plt.title("Histograma del ruido uniforme")
plt.grid()
plt.show()

plt.figure()
plt.plot(n, ruido2, color="purple")
plt.xlabel("n")
plt.ylabel("Amplitud")
plt.title("Ruido uniformemente distribuído")
plt.grid()
plt.show()


def ruido_uniforme(media, varianza, N):  #lo escribí de forma general
    A = np.sqrt(3 * varianza)
    a= media -A                                    
    b= media +A
    ruido_uniforme = np.random.uniform(a, b, N)
    return ruido_uniforme

N=1000
media= 0
varianza= 0.1
ruido2= ruido_uniforme(media, varianza, N)
n = np.arange(N)

plt.hist(ruido2)
plt.figure()
plt.hist(ruido2, bins=20)
plt.xlabel("Amplitud")
plt.ylabel("Cantidad de muestras")
plt.title("Histograma del ruido uniforme")
plt.grid()
plt.show()

plt.figure()
plt.plot(n, ruido2, color="purple")
plt.xlabel("n")
plt.ylabel("Amplitud")
plt.title("Ruido uniformemente distribuído")
plt.grid()
plt.show()


def xx_chirp(N, fs,f0, f1):
    T = N / fs
    t = np.arange(N) / fs
    x_chirp = signal.chirp(t, f0=f0,f1=f1,t1=T, method="linear")
    return t, x_chirp


N=1000
fs=20000
f0=100
f1=2000
t, x_chirp= xx_chirp(N, fs,f0, f1)


plt.figure(figsize=(10,4))

plt.plot(t, x_chirp)

plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.title("Señal Chirp")

plt.grid()
plt.show()

