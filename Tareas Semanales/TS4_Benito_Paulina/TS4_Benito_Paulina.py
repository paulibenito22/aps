# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 19:20:41 2026

@author: Paulina Benito
"""


import numpy as np 
import matplotlib.pyplot as plt
from spicy import signal

#%%
fs=1000
A0= np.sqrt(2)
N=1000
deltaf=fs/N 
fr =np.random.uniform(-2,2,200)
SNR=32
omega0= 2*np.pi*(N/4)/N
omega= omega0+ fr * (2*np.pi)/2
#la varianza del ruido normal sale de los SNR . despejo la potencia y se calcula
Psen= (A0**2)/2 #porq es de 1W
Pr = Psen / (10**(SNR/10))
frec=(fs/N)*np.arange(N) #normalizada 
R=200
senos=[]
#%%
def funcion_sen( A0 ,N, Pr,R, fr, omega0) : 
    nn= (np.arange(0,N)).reshape(1, N)
    rr=(omega0+fr*(2*np.pi)/N).reshape(R, 1)

    desvio= np.sqrt(Pr)
    ruidonormal= np.random.normal(0,desvio,size=(R, N))
    vector=nn *rr
    xx=  A0 * np.sin(vector) + ruidonormal
    
    return xx

#%% estimadores 
# def estimador_energia(xx, w):
#     ventana = xx* w
#     Xventana= np.fft.fft(ventana)
#     estimador1= np.abs(Xventana)
    
#     return estimador1
#%%
def estimador_frecuencia(xx,w, deltaf):
    ventana = xx* w
    Xventana= np.fft.fft(ventana)
    modulo= np.abs(Xventana)
    indice= np.argmax(modulo)
    estimador2= (2*np.pi)*deltaf*indice

    return estimador2

#%% 1) 

##senos
senoidal=funcion_sen(A0, N, Pr, R, fr,omega0)


nn=np.arange(0, N)
# plt.figure()
# plt.title("Senos")
# plt.plot(nn, senoidal.T)  #si no traspongo no me dan las dimensiones
# plt.show()

##fft
Fsenoidal = np.fft.fft(senoidal, axis=1)/N
frec = (fs/N)*np.arange(N//2)

print(senoidal.shape)
print(Fsenoidal.shape)


plt.figure()
plt.plot(frec, (2*np.abs(Fsenoidal[:, :N//2])).T)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('|X(f)|')
plt.title("Módulo de la fft")
plt.grid()
plt.show()

#amplitud de db
plt.figure()
plt.plot(
     frec,
     (10*np.log10(2*np.abs(Fsenoidal[:, :N//2])).T))
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud [dB]')
plt.grid()
plt.show()


#densidad de potencia db
P = 2*np.abs(Fsenoidal[:, :N//2])**2
P_dB = 10*np.log10(P)
plt.figure()
plt.plot(
     frec,P_dB.T)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.grid()
plt.show()

#%% estimador a1 
senoidal=funcion_sen(A0, N, Pr, R, fr,omega0)
a1=np.abs(Fsenoidal[:, N//4])

psd=10*np.log10(2*(a1)**2)
frec2= np.arange(0,200)
plt.figure()
plt.plot(frec2, a1)
plt.show()

#valor esperado
esperanza= np.mean(a1)
print(esperanza)

sesgo= (np.sqrt(2)/2) - esperanza
print(sesgo)
#print(ventana)
# Xventana= np.fft.fft(ventana)
# estimador1= np.abs(Xventana)
    
#%% otra ventana: flattop

w= signal.window.flattop()







 