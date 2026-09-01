# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 20:22:07 2026

@author: Paulina Benito
"""

import matplotlib.pyplot as plt
import numpy as np

fs=1000 #HZ
N = 1000 #MUESTRAS

"""respuesta en frecuencia=fs/N=1
--> sen(k(fs/N)n)  se puede rrescribir el seno asi
si por ejemplo elijo la frecuencia ff=4--> por simetria al hacer la FFT voy a tener un valor en 
k=4 y su conjugado (osea lo mismo pero en negarivo) en 996
X(4)=j(N/2)delta(k-4)=j500
X(996)=-j(N/2)delta(k-4)??=-j500
"""



def mi_function_senconruido (vmax = 1, dc=0, kk=4, ph=90, nn=N,fs =fs, snr=20):
    n=np.arange(nn)
    xx =vmax*np.sin(kk*(2*np.pi*fs/nn)*n/fs + ph) +dc
    tt =np.arange(0, stop=nn/fs, step=1/fs)
    
    ps=np.var(xx)
    pr=ps/(10**(snr/10))
    
    
    muestras = nn
    var=pr
    rango=np.sqrt(12*var)
    a = -rango/2
    b=rango/2

    ruido = np.random.uniform(low=a, high=b, size=muestras)
    
    xxruido= xx+ruido

    return(tt,xx, xxruido, ruido)

# vector frecuencia
k= np.arange(N)
# genero la senoidal con ruido

tt,xx, xxruido, ruido=mi_function_senconruido(1, 0, 4, 0, N, fs, 20)

# genero la fft para la senoidal

ffty=np.fft.fft(xx)
fftx=np.fft.fftfreq(N, 1/fs )


magsen = np.abs(ffty) 

# genero la fft para el ruido

fftry=np.fft.fft(ruido)

magr = np.abs(fftry) 


# genero la fft para la senoidal con ruido

fftysenr=np.fft.fft(xxruido)

magsenr = np.abs(fftysenr) 
#fase de la fft de la senoidal con ruido
fase= np.angle(fftysenr)  #para q tome la max frec

# y = fase*k
print(fase)
# graficos:

fase3=np.angle(ffty)

plt.figure()
plt.title("señal limpia")
plt.plot(tt, xx)
plt.show()

plt.figure()
plt.title("ruido")
plt.plot(tt, ruido)
plt.show()


plt.figure()
plt.title("señal con ruido")
plt.plot(tt, xxruido)
plt.show()

plt.figure()
plt.title("fft de la senoidal")
plt.plot(fftx, magsen)
plt.show()


plt.figure()
plt.title("fft del ruido")
plt.plot(fftx, magr)
plt.show()

plt.figure()
plt.title("fft de la senoidal con ruido")
plt.plot(fftx, magsenr)
plt.show()

plt.figure()
plt.plot(fase)
plt.xlabel("k")
plt.ylabel("Fase [rad]")
plt.title("Fase de la FFT")
plt.grid()
plt.show()