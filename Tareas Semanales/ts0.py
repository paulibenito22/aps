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
dc= 0
ff= 3
ph=0
nn=N                          #en nyquist tengo 2 puntos
#para disminuir el ruido de cuantizacion cambio la frecuencia de muestreo 


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
                                    #nunca  le pegas al 0 con nyquist 
                                    #1e-12 es un ruido (amplitud mas chica)
plt.figure()
plt.plot(tt3, xx3)
plt.show()



#%%

ff4 = 999
tt4, xx4 = mi_funcion_sen(vmax, dc, ff4, ph, nn, fs)
                                   #al aumentar mucho la frecuencia esta a penas abajo del sampling (por fuera del ancho de banda digital)
                                   #el efecto es tener una senoidal en contrafase de la senoidal q tenia 
plt.figure()
plt.plot(tt4, xx4)
plt.show()

#%%
ff5=1001
tt5, xx5 = mi_funcion_sen(vmax, dc, ff5 , ph, nn, fs)
                                  #aca pasa lo mismo q en ff4 pero desfasad 

plt.figure()
plt.plot(tt5,xx5)
plt.show


#en el mundo digital se interpreta lo q pasa entre 0 y nyquist
#si quiero representar algo q no esta en nyquist se pone denuevo periodicamente (coincide con un multiplo entero del periodo)

#si no pongo un filtro ni fuezo q hay energia arriba del nyquist se mete en banda y se te mezcla (quitala perioricidad del espectro)




