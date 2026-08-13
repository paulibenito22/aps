# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 20:26:39 2026

@author: Paulina Benito
"""

import matplotlib.pyplot as plt 
import numpy as np 
#%%definiciones 

N= 1000 #Hz
fs= 1000 #Muestras 
ts= 1/fs 


#%%funciones
def mi_funcion_sen( vmax , dc, ff , ph, nn,  fs) : 
    xx = dc + vmax * np.sin(2*np.pi*ff*tt + ph)
    
    return (tt,xx)

#%%parte 1
vmax = 1.5
dc= 3
ff= 3
ph=0
nn=N


tt=np.arange(0,nn) *ts 
tt, xx = mi_funcion_sen( vmax , dc , ff , ph, nn,  fs)

plt.figure()
plt.plot(tt, xx)
plt.show

#%%
ff2 = 100
tt2, xx2 = mi_funcion_sen(vmax, dc, ff2, ph, nn, fs)

plt.figure()
plt.plot(tt2, xx2)
plt.show()

#%%
ff3 = 500
tt3, xx3 = mi_funcion_sen(vmax, dc, ff3, ph, nn, fs)

plt.figure()
plt.plot(tt3, xx3)
plt.show()




