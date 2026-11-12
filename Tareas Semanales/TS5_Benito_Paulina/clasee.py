# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 19:57:00 2026

@author: Paulina Benito
"""
#primero hacer la estimacion espectral y establecer la plantilla de filtrado

#%% 1. Bibliotecas

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
import scipy.io as sio


#%% 2. Lectura del ECG

fs_ecg = 1000  # Frecuencia de muestreo [Hz]

mat_struct = sio.loadmat('ECG_TP4.mat')

ecg_one_leadruido = (
    mat_struct['ecg_lead'].flatten().astype(float)
)

N = len(ecg_one_leadruido)
t = np.arange(N) / fs_ecg

print(f'Cantidad de muestras: {N}')
print(f'Duración del registro: {N / fs_ecg:.2f} s')


#%% 3. Plantilla pasa banda

ws1 = 0.1     # Fin del rechazo inferior [Hz] — provisional
wp1 = 1.0     # Inicio de la banda de paso [Hz] — provisional

wp2 = 40      # Fin de la banda de paso [Hz]
ws2 = 50      # Inicio del rechazo superior [Hz]

gpass = 1     # Pérdida máxima en banda de paso [dB]
gstop = 40    # Atenuación mínima en bandas de rechazo [dB]

nyq_frec = fs_ecg / 2

wp_banda = np.array([wp1, wp2])
ws_banda = np.array([ws1, ws2])

# Comprobar que los límites estén correctamente ordenados
assert 0 < ws1 < wp1 < wp2 < ws2 < nyq_frec


#%% 4. Diseño de los tres filtros IIR

tipos_filtros = [
    ('Butterworth', 'butter', 'tab:blue'),
    ('Chebyshev II', 'cheby2', 'tab:orange'),
    ('Cauer', 'ellip', 'tab:green')
]

filtros = {}

for nombre, tipo, color in tipos_filtros:

    filtros[nombre] = sig.iirdesign(
        wp=wp_banda,
        ws=ws_banda,
        gpass=gpass,
        gstop=gstop,
        analog=False,
        ftype=tipo,
        output='sos',
        fs=fs_ecg
    )

    print(
        f'{nombre}: '
        f'{filtros[nombre].shape[0]} secciones SOS'
    )


#%% 5. Respuestas en frecuencia

# Grilla con más resolución en bajas frecuencias
frecuencias = np.unique(
    np.r_[
        np.linspace(0, 2, 10001),
        np.linspace(2, nyq_frec, 30001),
        ws1, wp1, wp2, ws2
    ]
)

respuestas = {}

for nombre, tipo, color in tipos_filtros:

    f, H = sig.sosfreqz(
        filtros[nombre],
        worN=frecuencias,
        fs=fs_ecg
    )

    modulo_db = 20 * np.log10(
        np.maximum(np.abs(H), 1e-12)
    )

    # La fase no está definida en los ceros de transmisión.
    fase = np.unwrap(np.angle(H))
    fase[np.abs(H) < 1e-8] = np.nan

    respuestas[nombre] = {
        'f': f,
        'H': H,
        'modulo_db': modulo_db,
        'fase': fase
    }


#%% 6. Verificación numérica de la plantilla
# Esta verificación corresponde al filtro de una pasada.

print('\nVERIFICACIÓN DE LA PLANTILLA')

for nombre, tipo, color in tipos_filtros:

    f = respuestas[nombre]['f']
    modulo_db = respuestas[nombre]['modulo_db']

    banda_paso = (f >= wp1) & (f <= wp2)
    rechazo_inferior = f <= ws1
    rechazo_superior = f >= ws2

    minimo_paso = np.min(modulo_db[banda_paso])
    maximo_paso = np.max(modulo_db[banda_paso])

    atenuacion_inferior = -np.max(
        modulo_db[rechazo_inferior]
    )

    atenuacion_superior = -np.max(
        modulo_db[rechazo_superior]
    )

    tolerancia = 0.01  # dB: tolerancia numérica

    cumple = (
        minimo_paso >= -gpass - tolerancia
        and maximo_paso <= tolerancia
        and atenuacion_inferior >= gstop - tolerancia
        and atenuacion_superior >= gstop - tolerancia
    )

    print(f'\n{nombre}')
    print(
        f'Banda de paso: '
        f'{minimo_paso:.3f} a {maximo_paso:.3f} dB'
    )
    print(
        f'Atenuación mínima inferior: '
        f'{atenuacion_inferior:.3f} dB'
    )
    print(
        f'Atenuación mínima superior: '
        f'{atenuacion_superior:.3f} dB'
    )
    print(f'Cumple en la grilla evaluada: {cumple}')


#%% 7. Función para dibujar la plantilla

def dibujar_plantilla(ax):

    estilo = {
        'color': 'lightcoral',
        'alpha': 0.25,
        'hatch': 'xxx'
    }

    # Rechazo inferior: prohibido superar -gstop
    ax.fill_between(
        [0, ws1], -gstop, 5,
        label='Región prohibida',
        **estilo
    )

    # Paso: prohibido caer debajo de -gpass
    ax.fill_between(
        [wp1, wp2], -100, -gpass,
        **estilo
    )

    # Paso: prohibido superar 0 dB
    ax.fill_between(
        [wp1, wp2], 0, 5,
        **estilo
    )

    # Rechazo superior: prohibido superar -gstop
    ax.fill_between(
        [ws2, nyq_frec], -gstop, 5,
        **estilo
    )

    # Bordes de las bandas
    for borde in (ws1, wp1, wp2, ws2):
        ax.axvline(
            borde,
            color='gray',
            linestyle=':',
            linewidth=1
        )

    # Límites horizontales en las bandas correspondientes
    ax.hlines(
        [0, -gpass],
        wp1, wp2,
        colors='gray',
        linestyles='--',
        linewidth=1
    )

    ax.hlines(
        -gstop,
        0, ws1,
        colors='gray',
        linestyles='--',
        linewidth=1
    )

    ax.hlines(
        -gstop,
        ws2, nyq_frec,
        colors='gray',
        linestyles='--',
        linewidth=1
    )

    ax.set_ylim(-100, 5)
    ax.set_ylabel('Módulo [dB]')
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)


#%% 8. Gráfico de módulo y fase

fig, (ax_mod, ax_fase) = plt.subplots(
    2, 1,
    figsize=(12, 8),
    sharex=True,
    constrained_layout=True
)

for nombre, tipo, color in tipos_filtros:

    respuesta = respuestas[nombre]

    ax_mod.plot(
        respuesta['f'],
        respuesta['modulo_db'],
        color=color,
        linewidth=1.5,
        label=nombre
    )

    ax_fase.plot(
        respuesta['f'],
        respuesta['fase'],
        color=color,
        linewidth=1.5,
        label=nombre
    )

dibujar_plantilla(ax_mod)

ax_mod.set_title(
    'Filtros IIR pasa banda — respuesta de una pasada'
)

ax_mod.legend(loc='lower right')

for borde in (ws1, wp1, wp2, ws2):
    ax_fase.axvline(
        borde,
        color='gray',
        linestyle=':',
        linewidth=1
    )

ax_fase.set_ylabel('Fase [rad]')
ax_fase.set_xlabel('Frecuencia [Hz]')
ax_fase.set_xlim(0, nyq_frec)
ax_fase.grid(True, alpha=0.3)

plt.show()


#%% 9. Acercamientos del módulo y la plantilla

fig, axes = plt.subplots(
    2, 1,
    figsize=(12, 8),
    constrained_layout=True
)

for nombre, tipo, color in tipos_filtros:

    respuesta = respuestas[nombre]

    for ax in axes:
        ax.plot(
            respuesta['f'],
            respuesta['modulo_db'],
            color=color,
            linewidth=1.5,
            label=nombre
        )

for ax in axes:
    dibujar_plantilla(ax)
    ax.set_xlabel('Frecuencia [Hz]')
    ax.legend(loc='lower right')

axes[0].set_xlim(0, 100)
axes[0].set_title('Plantilla pasa banda — de 0 a 100 Hz')

axes[1].set_xlim(0, 2)
axes[1].set_title('Acercamiento de la transición inferior')

plt.show()


#%% 10. Aplicar los filtros al registro completo
# sosfiltfilt aplica el filtro hacia adelante y hacia atrás.
# Su respuesta efectiva tiene fase cero y módulo |H|**2.
# Por eso, las pérdidas y atenuaciones en dB se duplican.

ecg_filtrados = {}

for nombre, tipo, color in tipos_filtros:

    ecg_filtrados[nombre] = sig.sosfiltfilt(
        filtros[nombre],
        ecg_one_leadruido
    )

# Variables de referencia para Butterworth
sos = filtros['Butterworth']
ecg_filtrado = ecg_filtrados['Butterworth']


#%% 11. Registro completo: ECG sin filtrar y filtrado

fig, axes = plt.subplots(
    4, 1,
    figsize=(14, 10),
    sharex=True,
    sharey=True,
    constrained_layout=True
)

axes[0].plot(
    t,
    ecg_one_leadruido,
    color='gray',
    linewidth=0.6
)

axes[0].set_title('ECG con ruido — sin filtrar')

for ax, (nombre, tipo, color) in zip(
    axes[1:], tipos_filtros
):

    ax.plot(
        t,
        ecg_filtrados[nombre],
        color=color,
        linewidth=0.6
    )

    ax.set_title(f'ECG filtrado — {nombre}')

for ax in axes:
    ax.set_ylabel('Amplitud')
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Tiempo [s]')
axes[-1].set_xlim(t[0], t[-1])

plt.show()


#%% 12. Comparación con zoom
# Cambiá estos dos valores para elegir otro tramo.

inicio_seg = 5
fin_seg = 7

inicio = max(0, int(inicio_seg * fs_ecg))
fin = min(N, int(fin_seg * fs_ecg))

if fin <= inicio:
    raise ValueError(
        'El intervalo de zoom está fuera del registro.'
    )

fig, axes = plt.subplots(
    3, 1,
    figsize=(12, 8),
    sharex=True,
    sharey=True,
    constrained_layout=True
)

for ax, (nombre, tipo, color) in zip(
    axes, tipos_filtros
):

    ax.plot(
        t[inicio:fin],
        ecg_one_leadruido[inicio:fin],
        color='gray',
        alpha=0.7,
        linewidth=1,
        label='ECG sin filtrar'
    )

    ax.plot(
        t[inicio:fin],
        ecg_filtrados[nombre][inicio:fin],
        color=color,
        linewidth=1.4,
        label=f'ECG filtrado: {nombre}'
    )

    ax.set_ylabel('Amplitud')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

axes[0].set_title(
    f'Comparación del ECG — de {inicio_seg} a {fin_seg} s'
)

axes[-1].set_xlabel('Tiempo [s]')

plt.show()


#%% 13. Respuestas al impulso en un mismo gráfico
# Se usa sosfilt: respuesta causal de una pasada.
# Extendemos a 10 segundos para observar las colas lentas
# introducidas por el filtrado de bajas frecuencias.

N_imp = int(10 * fs_ecg)

delta = np.zeros(N_imp)
delta[0] = 1

tiempo_imp = np.arange(N_imp) / fs_ecg

fig, ax = plt.subplots(
    figsize=(12, 5),
    constrained_layout=True
)

for nombre, tipo, color in tipos_filtros:

    h = sig.sosfilt(
        filtros[nombre],
        delta
    )

    ax.plot(
        tiempo_imp,
        h,
        color=color,
        linewidth=1.4,
        label=nombre
    )

ax.set_title('Respuesta al impulso de los IIR pasa banda')
ax.set_xlabel('Tiempo [s]')
ax.set_ylabel('h[n]')
ax.grid(True, alpha=0.3)
ax.legend()

plt.show()


#%% 14. Retardo de grupo de los tres filtros
# Se calcula para una pasada.
# El retardo total es la suma de los retardos de las SOS.

frecuencias_rg = np.unique(
    np.r_[
        np.linspace(0.01, 2, 4000),
        np.linspace(2, 130, 8000)
    ]
)

fig, axes = plt.subplots(
    2, 1,
    figsize=(12, 8),
    constrained_layout=True
)

for nombre, tipo, color in tipos_filtros:

    retardo_muestras = np.zeros_like(frecuencias_rg)

    for seccion in filtros[nombre]:

        b = seccion[:3]
        a = seccion[3:]

        f_rg, retardo_seccion = sig.group_delay(
            (b, a),
            w=frecuencias_rg,
            fs=fs_ecg
        )

        retardo_muestras += retardo_seccion

    retardo_segundos = retardo_muestras / fs_ecg

    _, H_rg = sig.sosfreqz(
        filtros[nombre],
        worN=frecuencias_rg,
        fs=fs_ecg
    )

    # Ocultar puntos cercanos a ceros de transmisión
    no_validos = (
        (np.abs(H_rg) < 1e-8)
        | ~np.isfinite(retardo_segundos)
    )

    retardo_segundos[no_validos] = np.nan

    for ax in axes:
        ax.plot(
            f_rg,
            retardo_segundos,
            color=color,
            linewidth=1.5,
            label=nombre
        )

for ax in axes:

    for borde, etiqueta in [
        (ws1, 'ws1'),
        (wp1, 'wp1'),
        (wp2, 'wp2'),
        (ws2, 'ws2')
    ]:

        ax.axvline(
            borde,
            color='gray',
            linestyle='--',
            linewidth=0.9,
            label=f'{etiqueta} = {borde:g} Hz'
        )

    ax.set_xlabel('Frecuencia [Hz]')
    ax.set_ylabel('Retardo de grupo [s]')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)

axes[0].set_xlim(0, 130)
axes[0].set_title('Retardo de grupo — vista general')

axes[1].set_xlim(wp1, 60)
axes[1].set_title(
    'Retardo de grupo — detalle desde la banda de paso'
)

# Ajustar la escala vertical del segundo panel
# usando solo los puntos visibles
valores_visibles = []

for linea in axes[1].lines[:len(tipos_filtros)]:

    x_linea = linea.get_xdata()
    y_linea = linea.get_ydata()

    mascara = (
        (x_linea >= wp1)
        & (x_linea <= 60)
        & np.isfinite(y_linea)
    )

    valores_visibles.extend(y_linea[mascara])

if valores_visibles:

    minimo = np.min(valores_visibles)
    maximo = np.max(valores_visibles)

    margen = max(0.05 * (maximo - minimo), 0.001)

    axes[1].set_ylim(
        minimo - margen,
        maximo + margen
    )

plt.show()


#%% Comparación de los tres IIR en las regiones de interés

# cant_muestras = len(ecg_one_leadruido)

# # Intervalos expresados en muestras.
# # El extremo final no se incluye.
# regiones = [
#     ('Con ruido', 4000, 5500),
#     ('Con ruido', 10000, 11000),

#     ('Sin ruido según consigna',
#      5 * 60 * fs_ecg, 5.2 * 60 * fs_ecg),

#     ('Sin ruido según consigna',
#      12 * 60 * fs_ecg, 12.4 * 60 * fs_ecg),

#     ('Sin ruido según consigna',
#      15 * 60 * fs_ecg, 15.2 * 60 * fs_ecg)
# ]

# for descripcion, muestra_inicio, muestra_fin in regiones:

#     # Convertir a enteros y limitar al registro disponible
#     inicio = max(0, int(round(muestra_inicio)))
#     fin = min(cant_muestras, int(round(muestra_fin)))

#     if fin <= inicio:
#         print(f'Región fuera del registro: {descripcion}')
#         continue

#     zoom_region = np.arange(inicio, fin)

#     # Pasar las muestras a segundos
#     tiempo_region = zoom_region / fs_ecg

#     fig, axes = plt.subplots(
#         3, 1,
#         figsize=(13, 8),
#         sharex=True,
#         sharey=True,
#         constrained_layout=True
#     )

#     for ax, (nombre, tipo, color) in zip(
#         axes, tipos_filtros
#     ):

#         ax.plot(
#             tiempo_region,
#             ecg_one_leadruido[zoom_region],
#             color='gray',
#             alpha=0.7,
#             linewidth=1,
#             label='ECG sin filtrar'
#         )

#         ax.plot(
#             tiempo_region,
#             ecg_filtrados[nombre][zoom_region],
#             color=color,
#             linewidth=1.3,
#             label=f'ECG filtrado: {nombre}'
#         )

#         ax.axhline(
#             0,
#             color='black',
#             linestyle=':',
#             linewidth=0.8
#         )

#         ax.set_ylabel('Amplitud')
#         ax.grid(True, alpha=0.3)
#         ax.legend(loc='upper right')

#     fig.suptitle(
#         f'{descripcion} — '
#         f'{inicio / fs_ecg:g} a {fin / fs_ecg:g} s'
#     )

#     axes[-1].set_xlabel('Tiempo [s]')
#     axes[-1].set_xlim(
#         inicio / fs_ecg,
#         fin / fs_ecg
#     )

#     plt.show()
   
#%% Regiones de interés: todas las señales superpuestas

cant_muestras = len(ecg_one_leadruido)

regiones = [
    ('Con ruido', 4000, 5500),
    ('Con ruido', 10000, 11000),

    ('Región de interés',
     5 * 60 * fs_ecg, 5.2 * 60 * fs_ecg),

    ('Región de interés',
     12 * 60 * fs_ecg, 12.4 * 60 * fs_ecg),

    ('Región de interés',
     15 * 60 * fs_ecg, 15.2 * 60 * fs_ecg)
]

for descripcion, muestra_inicio, muestra_fin in regiones:

    inicio = max(0, int(round(muestra_inicio)))
    fin = min(cant_muestras, int(round(muestra_fin)))

    if fin <= inicio:
        continue

    zoom_region = np.arange(inicio, fin)

    plt.figure(figsize=(12, 5))

    # ECG sin filtrar
    plt.plot(
        zoom_region,
        ecg_one_leadruido[zoom_region],
        color='tab:blue',
        linewidth=1.5,
        label='ECG sin filtrar'
    )

    # Butterworth
    plt.plot(
        zoom_region,
        ecg_filtrados['Butterworth'][zoom_region],
        color='tab:orange',
        linewidth=1.2,
        label='Butterworth'
    )

    # Chebyshev II
    plt.plot(
        zoom_region,
        ecg_filtrados['Chebyshev II'][zoom_region],
        color='tab:green',
        linewidth=1.2,
        label='Chebyshev II'
    )

    # Cauer
    plt.plot(
        zoom_region,
        ecg_filtrados['Cauer'][zoom_region],
        color='tab:red',
        linewidth=1.2,
        label='Cauer'
    )

    plt.title(
        f'{descripcion} — muestras {inicio} a {fin}'
    )

    plt.xlabel('Muestras [n]')
    plt.ylabel('Amplitud')
    plt.xlim(inicio, fin)
    plt.grid(True, alpha=0.25)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()