# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 19:26:51 2026

@author: Paulina Benito
"""


import numpy as np 
import matplotlib.pyplot as plt

#%%
def funcion_sen( vmax , dc, k , ph, N,  fs) : 
    nn= np.arange(0,N,)
    xx = dc + vmax * np.sin(k*(2*np.pi*fs/N)*nn/fs + ph)

    return (nn,xx)

#%%
vmax= np.sqrt(2)
N=1000
k=N/4
ph=0
dc=0
fs=1000
nn,xx= funcion_sen(vmax, dc, k, ph, N, fs)
frec=np.arange(N//2) * fs/N
nXX = 1/N*np.fft.fft(xx)

plt.plot(frec, 10*np.log10(2*(np.abs(nXX[:N//2])**2)), ':o', label='k=N/4')
plt.grid()

#%%

nn,xx2= funcion_sen(vmax, dc, k+0.25, ph, N, fs)
eXX = 1/N*np.fft.fft(xx2)

plt.plot(frec, 10*np.log10(2*(np.abs(eXX[:N//2])**2)),':x')
plt.grid(True)


#%%

nn,xx3= funcion_sen(vmax, dc, k+0.5, ph, N, fs)
aXX = 1/N*np.fft.fft(xx3)

plt.plot(frec, 10*np.log10(2*(np.abs(aXX[:N//2])**2)),':v')
plt.grid(True)


#%% zero padding

ceros = np.zeros(9*N)
xxpadding = np.concatenate((xx, ceros))

frec2=np.arange(10*N//2) * fs/N
pXX=(1/N)*np.fft.fft(xxpadding)

plt.figure()
plt.plot(frec2, 10*np.log10(2*(np.abs(pXX[:10*N//2])**2)),'x')
plt.grid(True)
plt.show()
