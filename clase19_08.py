# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 21:00:09 2026

@author: Paulina Benito
"""

import matplotlib.pyplot as plt 
import numpy as np 



#%%
def funcion_sen( vmax , dc, ff , ph, nn,  fs) : 
    xx = dc + vmax * np.sin(2*np.pi*ff*tt + ph)
 
    return (tt,xx)

def mi_funcion_sen_ruido(vmax, dc, ff, ph, nn, fs, snr, pr):
    ts = 1/fs
    tt = np.arange(0, nn) * ts
    desvio = np.sqrt(pr)
    r = np.random.normal(0, desvio, len(tt))   #se usa el desvio porq en la normal se usa ese parametro
    print(f"Potencia de ruido (pr): {pr}") 
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
tt1,xx1= mi_funcion_sen_ruido(A, dc, ff, ph, nn, fs, snr, pr)
 
 
 
plt.figure(figsize=(10, 4))
plt.plot(tt, xx, label="Original", linewidth=2)
plt.plot(tt1, xx1, label="Con Ruido", alpha=0.7)

plt.xlabel("t[s]")
plt.ylabel("Amplitud")
plt.grid(True)
plt.legend()
plt.show()
 
 