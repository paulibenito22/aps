# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 09:53:03 2026

@author: Paulina Benito
"""



import numpy as np 
import matplotlib.pyplot as plt
from scipy import signal



#%%
wd = np.pi/4 #defino una frecuencia para poder graficar
n = np.arange(-10, 10) #para tener finitas muestras
x = np.cos(wd*n)
h = np.array([1, 0, 0, 0, -1]) #es el resultado de ir reezplazando
nh = np.arange(0, 5) #vector de posiciones de h para graficar 
y = np.convolve(x, h, mode='full')
ny = np.arange(n[0] + nh[0], n[-1] + nh[-1] + 1)

#entrada
plt.stem(n, x)
plt.xlabel('n')
plt.ylabel('x[n]')
plt.title("Entrada x[n]")
plt.grid()
plt.show()



#respuesta al impulso
plt.figure()
plt.stem(nh, h)
plt.axhline(0)
plt.xlabel('n')
plt.ylabel('h[n]')
plt.title('Respuesta al impulso h[n]')
plt.grid()

plt.show()

#convolucion
plt.figure()
plt.stem(ny, y)
plt.axhline(0)
plt.xlabel('n')
plt.ylabel('y[n]')
plt.title('Convolución y[n] = x[n] * h[n]')
plt.grid()
plt.show()



#%%
wd=np.pi/4
y= np.cos(wd* n) - np.cos(wd * (n-4)) 
n= np.arange(-10,10)


plt.figure()
plt.stem(n, y)
plt.axhline(0)
plt.xlabel('n')
plt.ylabel('y[n]')
plt.title('Convolución y[n] = x[n] * h[n]')
plt.grid()
plt.show()




