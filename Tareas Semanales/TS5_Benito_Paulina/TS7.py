# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 20:45:11 2026

@author: Paulina Benito
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz, iirdesign, sosfreqz

# b = [1, 1, 1]
# a = [1]

# w, H = freqz(b, a, worN=1024)

# modulo = np.abs(H)
# fase = np.angle(H)

# plt.figure()
# plt.plot(w, modulo)
# plt.xlabel('Frecuencia angular [rad/muestra]')
# plt.ylabel('|H(e^jw)|')
# plt.title('Respuesta en módulo')
# plt.grid()
# plt.show()

# plt.figure()
# plt.plot(w, fase)
# plt.xlabel('Frecuencia angular [rad/muestra]')
# plt.ylabel('Fase [rad]')
# plt.title('Respuesta de fase')
# plt.grid()
# plt.show()

# modulo_db = 20*np.log10(np.abs(H))
# plt.figure()
# plt.plot(w, modulo_db)
# plt.xlabel('Frecuencia angular [rad/muestra]')
# plt.ylabel('Magnitud [dB]')
# plt.title('Respuesta en frecuencia')
# plt.grid()
# plt.show()



# #%% filtro IIR

# wp= 1/2
# ws=2/3

# b,a = iirdesign(wp,ws,3,40, analog=False, ftype='butter', output='ba', fs=2)
# w, H = freqz(b,a)

# modulo_db = 20*np.log10(np.abs(H))
# fase = np.unwrap(np.angle(H))

# plt.figure()
# plt.plot(w, modulo_db)
# plt.axvline(wp, linestyle='--', label='wp')
# plt.axvline(ws, linestyle='--', label='ws')
# plt.xlabel('Frecuencia')
# plt.ylabel('Magnitud [dB]')
# plt.title('Respuesta en módulo del filtro Butterworth')
# plt.grid()
# plt.legend()

# plt.show()


# plt.figure()
# plt.plot(w, fase)
# plt.xlabel('Frecuencia')
# plt.ylabel('Fase [rad]')
# plt.title('Respuesta en fase del filtro Butterworth')
# plt.grid()

# plt.show()

# #%% otro grafico
# filtro  = iirdesign(wp,ws,3,40, analog=False, ftype='butter', output='ba', fs=2)
# w, H = freqz(filtro, fs=2)

# Hdb = 20*np.log10(np.abs(H))

# plt.figure()

# plt.plot(w, Hdb)

# plt.axvline(wp, linestyle='--', label='wp')
# plt.axvline(ws, linestyle='--', label='ws')

# plt.axhline(-3, linestyle='--', label='-3 dB')
# plt.axhline(-40, linestyle='--', label='-40 dB')

# plt.ylim(-60, 5)

# plt.xlabel('Frecuencia')
# plt.ylabel('Magnitud [dB]')
# plt.title('Respuesta en módulo del filtro Butterworth')

# plt.grid()
# plt.legend()

# plt.show()

#%%     ver el tema de  modificar el orden del filtro a partir de los coeficientes a y b 
#cambio de orden modificando b y a 
#con solo acordar las recursion a 1 solo voy a tener un polo y no te deja poner polos complejos conjugados, por eso se usan las sos como minimo de recursividad

b=np.ones(31)/31
a= [1]

wp=4/100
ws=5/100

filtro = iirdesign(wp, ws, 3, 40,
                   analog=False,
                   ftype='butter',
                   output='ba',
                   fs=2)

b, a = filtro
w, H = freqz(b, a, fs=2)

Hdb = 20*np.log10(np.abs(H))
print(len(a)-1)

plt.figure()

plt.plot(w, Hdb)

plt.axvline(wp, linestyle='--', label='wp')
plt.axvline(ws, linestyle='--', label='ws')

plt.axhline(-3, linestyle='--', label='-3 dB')
plt.axhline(-40, linestyle='--', label='-40 dB')

plt.ylim(-60, 5)

plt.xlabel('Frecuencia')
plt.ylabel('Magnitud [dB]')
plt.title('Respuesta en módulo del filtro Butterworth')

plt.grid()
plt.legend()

plt.show()