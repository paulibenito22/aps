# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 21:00:09 2026

@author: Paulina Benito
"""

import matplotlib.pyplot as plt 
import numpy as np 
from scipy import signal



#%%
def funcion_sen( vmax , dc, ff , ph, nn,  fs) : 
    xx = dc + vmax * np.sin(2*np.pi*ff*tt + ph)
 
    return (tt,xx)

def funcion_sen_ruido(vmax, dc, ff, ph, nn, fs, snr, pr):
    ts = 1/fs
    tt = np.arange(0, nn) * ts
    desvio = np.sqrt(pr)
    r = np.random.normal(0, desvio, len(tt))   #se usa el desvio porq en la normal se usa ese parametro
    xx1 = xx + r
    return tt, xx1

 
 #%%
A= np.sqrt(2) #vmax
dc=0 
ff= 1
ph=0 
nn= 1000
fs=1000
 
snr= 20
Ps = (A**2)/ 2
pr= Ps /(10**(snr/10))
ts=1/fs
tt=np.arange(0,nn) *ts
tt, xx = funcion_sen( A , dc , ff , ph, nn,  fs)
tt1,xx1= funcion_sen_ruido(A, dc, ff, ph, nn, fs, snr, pr)
 
 
 
plt.figure(figsize=(10, 4))
plt.plot(tt, xx, label="Original", linewidth=2)
plt.plot(tt1, xx1, label="Con Ruido", alpha=0.7)

plt.xlabel("t[s]")
plt.ylabel("Amplitud")
plt.grid(True)
plt.legend()
plt.show()
 

#%%ahora lo hago con np.var q deberia ser mas general

def ruido_general(x,snr):
    Ps = np.var(x)                 # potencia de la señal, calculada del vector real
    pr = Ps / (10**(snr/10)) 
    desvio= np.sqrt(pr)
    r2= np.random.normal(0, desvio, len(x))
    xx2= x + r2
    return xx2, pr

#%%    
A= np.sqrt(2) #vmax
dc=0 
ff= 1
ph=0 
nn= 1000
fs=1000
ts=1/fs
tt=np.arange(0,nn) *ts
snr=20

tt, xx = funcion_sen( A , dc , ff , ph, nn,  fs)  
xx2, pr= ruido_general(xx, snr)

plt.figure()
plt.plot(tt,xx, label="original")
plt.plot(tt,xx2, label="ruido")
plt.grid(True)
plt.show()

#%% pruebo el ruido con otra señal 

def funcion_cuadrada(vmax, dc, ff, ph, nn, fs):

    ts = 1/fs
    tt = np.arange(0, nn) * ts

    yy = dc + vmax * signal.square(2*np.pi*ff*tt + ph)

    return tt, yy

A= np.sqrt(2) #vmax
dc=0 
ff= 1
ph=0 
nn= 1000
fs=1000
ts=1/fs
tt=np.arange(0,nn) *ts
snr=20

tt, yy = funcion_cuadrada( A , dc , ff , ph, nn,  fs)  
xx3, pr= ruido_general(yy, snr)


plt.figure()
plt.plot(tt,yy, label="original")
plt.plot(tt,xx3, label="ruido")
plt.grid(True)
plt.show()

