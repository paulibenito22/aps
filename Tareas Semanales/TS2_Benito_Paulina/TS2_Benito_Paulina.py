# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 11:42:16 2026

@author: Paulina Benito
"""

import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import kstest
#from scipy import signal

#%% definiciones
vmax= np.sqrt(2)
Psen= vmax**2/2
N=1000
VF=2
B=4
k=10
ph=0
dc=0
fs=1000
#%%
def funcion_sen( vmax , dc, ph, N,  fs) : 
    nn= np.arange(0,N)
    xx = dc + vmax * np.sin(2*np.pi*(fs/N)*nn/fs + ph)

    return (nn,xx)

#%% ADC

qq = 2*VF/(2**B)
def gen_noise(k, qq, ur=0, nn=N):
    Pq = qq**2 / 12
    Pn = k * Pq
    desv = np.sqrt(Pn)
    ruido = np.random.normal(ur, desv, nn)

    return ruido, Pn

#%% Apartado a)

#funcion limpia sin ruido
nn,xx= funcion_sen(vmax, dc, ph, N, fs)

#funcion entrada ADC
ruido, Pn= gen_noise(k, qq)
noisy_xx= xx + ruido

plt.figure()
plt.plot(nn, noisy_xx, label="Entrada ADC")
plt.plot(nn, xx, label="Señal sin ruido")

plt.xlabel("Tiempo [n]")
plt.ylabel("Amplitud [V]")
plt.title("Comparación de señales")
plt.legend()
plt.grid()
plt.show()

#densidad espectral de potencia de entrada al ADC
frec=np.arange(N//2) * fs/N

#Calculo fft de la señal ruidosa
nXX = 1/N*np.fft.fft(noisy_xx)

A = nXX[:N//2]
DEP= 10*np.log10(2*(np.abs(nXX[:N//2])**2))

# Espectro del módulo
plt.figure()
plt.plot(frec, DEP) #normalizo el ruido a 0db
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Módulo")
plt.grid(True)
plt.show()

# Espectro de fase
plt.figure()
plt.plot(frec, np.angle(A))
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Fase [rad]")
plt.grid(True)

plt.tight_layout()
plt.show()

#señal salida del ADC: cuantizacion

xx_q= np.round(noisy_xx/qq)*qq
#ruido cuantizacion
ruido_q= xx_q - noisy_xx
plt.figure()
plt.plot(nn, noisy_xx, "x", label="Entrada ADC")
plt.plot(nn, xx_q, "v", label="Salida cuantizada")

plt.xlabel("Tiempo [n]")
plt.ylabel("Amplitud [V]")
plt.title("Efecto de la cuantización")

plt.legend()
plt.grid()
plt.show()

#zoom en la cuantización asi se ve el efecto
plt.figure()

plt.plot(nn, noisy_xx, "x", label="Entrada ADC")
plt.plot(nn, xx_q, "v", label="Salida cuantizada")

plt.xlabel("Tiempo [n]")
plt.ylabel("Amplitud [V]")
plt.title("Zoom del efecto de cuantización")
plt.legend()
plt.grid()
plt.xlim(0, 40)
plt.show()


#ruido de cuantizacion/error de cuantización
plt.figure()
plt.plot(ruido_q/qq,':x')
plt.title("Ruido de cuantización")
plt.grid()


#para verificar que el ruido sea incorrelado
ruido_centrado = ruido_q - np.mean(ruido_q)

R = (np.correlate(ruido_centrado, ruido_centrado, mode="full") / len(ruido_q))/N  #esto tiene largo 2N-1 // divido por N para que se pueda comparar con la varianza
#el retardo 0 quedaria en el medio
centro=len(R)//2
retardo_0= R[centro]

varianza = np.var(ruido_q)


retardo_neg = np.arange(-N+1, N)

plt.figure()
plt.plot(retardo_neg, R)


plt.plot(0, R[centro], "o")

plt.xlabel("Retardo")
plt.ylabel("Autocorrelación")
plt.title("Autocorrelación del ruido")
plt.grid()
plt.show()


a=-qq/2
b=qq/2

resultado= kstest(ruido_q, "uniform", args=(a, b-a))

alpha= 0.5

if resultado.pvalue < alpha: 
    print("La señal no es compatible con una distribución uniforme.")
else: 
    print("No hay evidencia suficiente para confirmar que la señal no sigue una distribución uniforme.")

#%% espectros 

def espectro_potencia(x):
    X = np.fft.fft(x) / N
    P = 2 * np.abs(X[:N//2])**2  #aca me quedo solo con el rango de 0 a nyquist
    P_db = 10 * np.log10(P + 1e-20) #el valor chico queda para que si P=0 no me tire error
    return P_db


P_s = espectro_potencia(xx)
P_in = espectro_potencia(noisy_xx)
P_q = espectro_potencia(xx_q)

P_ruido = espectro_potencia(ruido)
P_ruido_q = espectro_potencia(ruido_q)

#valores medios
piso_ruido = 10*np.log10(np.mean(10**(P_ruido/10)))
piso_ruido_q = 10*np.log10(np.mean(10**(P_ruido_q/10)))


plt.figure(figsize=(12,6))

plt.plot(frec, P_q,
         label="ADC out")

plt.plot(frec, P_ruido, "--",
         label= "piso analogico")
plt.plot(frec, P_ruido_q, ":",
         label="piso digital")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Densidad de Potencia [dB]")
plt.title("Señal muestreada por un ADC")

plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

#%% punto 2
B = 8
k= 1
VF = 2

qq = 2*VF/(2**B)
ruido, Pn = gen_noise(k, qq)
noisy_xx= xx + ruido

xx_q = np.round(noisy_xx/qq)*qq
ruido_q = xx_q - noisy_xx

plt.figure()
plt.plot(nn, noisy_xx, "x", label="Entrada ADC")
plt.plot(nn, xx_q, "v", label="Salida cuantizada")
plt.xlabel("Tiempo [n]")
plt.ylabel("Amplitud [V]")
plt.title("Señal cuantizada")
plt.subplots_adjust(bottom=0.18)
plt.figtext(0.5, 0.02,
            "Figura 9. Resultado de la cuantización.",
            ha="center")
plt.legend()
plt.grid()
plt.show()