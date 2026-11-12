# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 13:55:04 2026

@author: Paulina Benito
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
import scipy.io as sio

#%% Lectura del ECG

fs_ecg = 1000  # Hz

mat_struct = sio.loadmat('ECG_TP4.mat')

ecg_one_leadruido = (
    mat_struct['ecg_lead'].flatten().astype(float)
)

N = len(ecg_one_leadruido)
t = np.arange(N) / fs_ecg

print(f'Cantidad de muestras: {N}')
print(f'Duración: {N / fs_ecg:.2f} s')

#%% ECG original con ruido: registro completo

t = np.arange(len(ecg_one_leadruido)) / fs_ecg

plt.figure(figsize=(12, 5))

plt.plot(
    t,
    ecg_one_leadruido,
    color='tab:blue',
    linewidth=0.8
)

plt.title('ECG original con ruido')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#%% Plantilla del filtro pasa banda

wp = 35       # Fin de banda de paso [Hz]
ws = 50       # Inicio de banda de rechazo [Hz]

gpass = 3     # Pérdida máxima en banda de paso [dB]
gstop = 40    # Atenuación mínima en banda de rechazo [dB]

nyq_frec = fs_ecg / 2

#%% Diseño de los filtros IIR

tipos_filtros = [
    ('Butterworth', 'butter', 'tab:blue'),
    ('Chebyshev II', 'cheby2', 'tab:orange'),
    ('Cauer', 'ellip', 'tab:green')
]

filtros = {}

for nombre, tipo, color in tipos_filtros:

    filtros[nombre] = sig.iirdesign(
        wp=wp,
        ws=ws,
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

#%% Respuesta en frecuencia: módulo y fase

fig, (ax_mod, ax_fase) = plt.subplots(
    2, 1,
    figsize=(12, 7),
    sharex=True,
    constrained_layout=True
)

frecuencias = np.unique(
    np.r_[np.linspace(0, nyq_frec, 16001), wp, ws]
)

for nombre, tipo, color in tipos_filtros:

    f, H = sig.sosfreqz(
        filtros[nombre],
        worN=frecuencias,
        fs=fs_ecg
    )

    modulo_db = 20 * np.log10(
        np.maximum(np.abs(H), 1e-12)
    )

    fase = np.unwrap(np.angle(H))

    ax_mod.plot(
        f, modulo_db,
        color=color,
        linewidth=1.6,
        label=nombre
    )

    ax_fase.plot(
        f, fase,
        color=color,
        linewidth=1.6
    )
#%% Dibujar la plantilla

db_min = -100
db_max = 5

estilo_plantilla = {
    'color': 'lightcoral',
    'alpha': 0.25,
    'hatch': 'xxx'
}

# Región prohibida debajo de la banda de paso
ax_mod.fill_between(
    [0, wp],
    db_min, -gpass,
    label='Región prohibida',
    **estilo_plantilla
)

# Región prohibida por encima de 0 dB
ax_mod.fill_between(
    [0, wp],
    0, db_max,
    **estilo_plantilla
)

# Región prohibida en la banda de rechazo
ax_mod.fill_between(
    [ws, nyq_frec],
    -gstop, db_max,
    **estilo_plantilla
)

# Límites horizontales
ax_mod.hlines(
    [0, -gpass],
    xmin=0, xmax=wp,
    colors='gray',
    linestyles='--',
    linewidth=1
)

ax_mod.hlines(
    -gstop,
    xmin=ws, xmax=nyq_frec,
    colors='gray',
    linestyles='--',
    linewidth=1
)

# Límites verticales y grillas
for ax in (ax_mod, ax_fase):

    ax.axvline(
        wp, color='gray',
        linestyle=':', linewidth=1.2
    )

    ax.axvline(
        ws, color='gray',
        linestyle=':', linewidth=1.2
    )

    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

ax_mod.set_title(
    'Filtros IIR pasa bajos: respuesta en frecuencia y plantilla'
)

ax_mod.set_ylabel('Módulo [dB]')
ax_mod.set_ylim(db_min, db_max)
ax_mod.legend(loc='lower left', framealpha=0.95)

ax_fase.set_ylabel('Fase [rad]')
ax_fase.set_xlabel('Frecuencia [Hz]')
ax_fase.set_xlim(0, nyq_frec)

ax_mod.text(
    wp / 2, -15,
    'Banda de\npaso',
    ha='center', fontsize=9
)

ax_mod.text(
    (ws + nyq_frec) / 2, -15,
    f'Banda de rechazo\nMódulo ≤ −{gstop} dB',
    ha='center', fontsize=10
)

plt.show()

#%% Aplicar los tres filtros al ECG (primero con sos.filt solo )
# sosfiltfilt aplica cada filtro hacia adelante y hacia atrás.

ecg_filtrados = {}

for nombre, tipo, color in tipos_filtros:

    ecg_filtrados[nombre] = sig.sosfilt(
        filtros[nombre],
        ecg_one_leadruido
    )

# Variables compatibles con tu código original
sos = filtros['Butterworth']
ecg_filtrado = ecg_filtrados['Butterworth']


#%% Comparar todas las señales en un mismo gráfico

inicio = 5000
fin = min(7000, N)

fig, ax = plt.subplots(
    figsize=(12, 5),
    constrained_layout=True
)

# ECG original: se dibuja una sola vez
ax.plot(
    t[inicio:fin],
    ecg_one_leadruido[inicio:fin],
    color='purple',
    alpha=0.6,
    linewidth=1.5,
    label='ECG sin filtrar'
)

# Superponer los tres ECG filtrados
for nombre, tipo, color in tipos_filtros:

    ax.plot(
        t[inicio:fin],
        ecg_filtrados[nombre][inicio:fin],
        color=color,
        linewidth=1.3,
        label=nombre
    )

ax.set_title('Comparación del ECG original y los tres filtros')
ax.set_xlabel('Tiempo [s]')
ax.set_ylabel('Amplitud')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right')

plt.show()

#%% Respuesta al impulso de los tres filtros

N_imp = 500

delta = np.zeros(N_imp)
delta[0] = 1

n = np.arange(N_imp)

plt.figure(figsize=(12, 5))

for nombre, tipo, color in tipos_filtros:

    h = sig.sosfilt(filtros[nombre], delta)

    plt.plot(
        n, h,
        color=color,
        linewidth=1.5,
        label=nombre
    )

plt.title('Respuesta al impulso de los filtros IIR')
plt.xlabel('Muestras [n]')
plt.ylabel('h[n]')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

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


#%% Ahora filtro con sosfiltfilt
#importante hacer la predistorsión para evitar cambios en el módulo


tipos_filtros = [
    ('Butterworth', 'butter', 'tab:blue'),
    ('Chebyshev II', 'cheby2', 'tab:orange'),
    ('Cauer', 'ellip', 'tab:green')
]

filtros = {}

fig, ax = plt.subplots(
    figsize=(12, 5),
    constrained_layout=True
)

for nombre, tipo, color in tipos_filtros:

    # Compensación para el filtrado de dos pasadas
    sos = sig.iirdesign(
        wp=wp,
        ws=ws,
        gpass=gpass / 2,
        gstop=gstop / 2,
        ftype=tipo,
        output='sos',
        fs=fs_ecg
    )

    filtros[nombre] = sos

    # Respuesta en frecuencia de una pasada
    f, h = sig.freqz_sos(
        sos,
        worN=8192,
        fs=fs_ecg
    )

    # Respuesta efectiva de sosfiltfilt
    modulo_db = 20 * np.log10(
        np.maximum(np.abs(h)**2, 1e-12)
    )

    ax.plot(
        f,
        modulo_db,
        color=color,
        linewidth=1.5,
        label=nombre
    )

# Plantilla: regiones prohibidas
ax.fill_between(
    [0, wp], -100, -gpass,
    color='lightcoral',
    alpha=0.25,
    hatch='xxx',
    label='Región prohibida'
)

ax.fill_between(
    [0, wp], 0, 5,
    color='lightcoral',
    alpha=0.25,
    hatch='xxx'
)

ax.fill_between(
    [ws, fs_ecg / 2], -gstop, 5,
    color='lightcoral',
    alpha=0.25,
    hatch='xxx'
)

# Bordes de la plantilla
ax.axvline(
    wp, color='gray', linestyle='--',
    label=f'wp = {wp:g} Hz'
)

ax.axvline(
    ws, color='black', linestyle='--',
    label=f'ws = {ws:g} Hz'
)

ax.hlines(
    [0, -gpass],
    0, wp,
    colors='gray',
    linestyles=':'
)

ax.hlines(
    -gstop,
    ws, fs_ecg / 2,
    colors='gray',
    linestyles=':'
)

ax.set_xlim(0, 100)
ax.set_ylim(-100, 5)

ax.set_title('Pasa bajos — respuesta efectiva de sosfiltfilt')
ax.set_xlabel('Frecuencia [Hz]')
ax.set_ylabel('Módulo [dB]')
ax.grid(True, alpha=0.3)
ax.set_axisbelow(True)
ax.legend(loc='lower right')

plt.show()

#%% Aplicar los tres filtros con sosfiltfilt

ecg_filtrados = {}

for nombre, tipo, color in tipos_filtros:

    ecg_filtrados[nombre] = sig.sosfiltfilt(
        filtros[nombre],
        ecg_one_leadruido
    )


#%% Comparar las cuatro señales en un mismo gráfico

t = np.arange(len(ecg_one_leadruido)) / fs_ecg

inicio = int(5 * fs_ecg)
fin = min(int(7 * fs_ecg), len(ecg_one_leadruido))

fig, ax = plt.subplots(
    figsize=(12, 5),
    constrained_layout=True
)

ax.plot(
    t[inicio:fin],
    ecg_one_leadruido[inicio:fin],
    color='purple',
    alpha=0.6,
    linewidth=1.5,
    label='ECG sin filtrar'
)

for nombre, tipo, color in tipos_filtros:

    ax.plot(
        t[inicio:fin],
        ecg_filtrados[nombre][inicio:fin],
        color=color,
        linewidth=1.3,
        label=f'{nombre} — sosfiltfilt'
    )

ax.set_title('ECG original y filtrado — pasa bajos con fase cero')
ax.set_xlabel('Tiempo [s]')
ax.set_ylabel('Amplitud')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right')

plt.show()

#%% filtro FIR con ventana

# Cantidad de coeficientes: valor inicial para comparar.
# No garantiza por sí solo cumplir la plantilla es lo que vamos a ir variando para "pegarle"
# numtaps = 151 #se cumplen al mismo tiempo con 6501

# # Longitud impar + simetría -> FIR tipo I
# orden_fir = numtaps - 1
# demora_fir = orden_fir // 2

# # Respuesta deseada en Hz y amplitud lineal
# frecs_fir = [0, wp, ws, fs_ecg / 2]
# gains_fir = [1, 1, 0, 0]

# ventanas_fir = [
#     ('Blackman', 'blackman', 'tab:blue'),
#     ('Hamming', 'hamming', 'tab:orange'),
#     ('Boxcar', 'boxcar', 'tab:green')
# ]

# filtros_fir = {}

# print(f'\nCantidad de coeficientes: {numtaps}')
# print(f'Orden: {orden_fir}')
# print(f'Demora: {demora_fir} muestras')
# print(f'Demora: {demora_fir / fs_ecg:.3f} s')

# for nombre, ventana, color in ventanas_fir:

#     filtros_fir[nombre] = sig.firwin2(
#         numtaps=numtaps,
#         freq=frecs_fir,
#         gain=gains_fir,
#         nfreqs=16385,
#         window=ventana,
#         antisymmetric=False,
#         fs=fs_ecg
#     )

#%% Diseño FIR con ajustes
# Usa wp, ws y fs_ecg de tu plantilla general.
# La plantilla no se modifica.

taps_win_han = 151
taps_win_hamm=191
taps_win_box = 191
delta_w = 5  # Hz

ventanas_fir = [
    ('Hann', 'hann', 'tab:blue'),
    ('Hamming', 'hamming', 'tab:orange'),
    ('Boxcar', 'boxcar', 'tab:green')
]

filtros_fir = {}

# Hann: reemplaza a Blackman
filtros_fir['Hann'] = sig.firwin2(
    numtaps=taps_win_han,
    freq=[0, wp, wp + delta_w, fs_ecg / 2],
    gain=[1, 1, 0, 0],
    window='hann',
    fs=fs_ecg
)

# Hamming
filtros_fir['Hamming'] = sig.firwin2(
    numtaps=taps_win_hamm,
    freq=[0, wp + 2, wp + delta_w, fs_ecg / 2],
    gain=[1, 1, 0, 0],
    window='hamming',
    fs=fs_ecg
)

# Boxcar
filtros_fir['Boxcar'] = sig.firwin2(
    numtaps=taps_win_box,
    freq=[0, wp - 2, wp + delta_w, fs_ecg / 2],
    gain=[1, 1, 0, 0],
    window='boxcar',
    fs=fs_ecg
)

# Mostrar longitud, orden y demora
for nombre, ventana, color in ventanas_fir:

    cantidad = len(filtros_fir[nombre])
    demora = (cantidad - 1) / 2

    print(f'\nFIR {nombre}')
    print(f'Coeficientes: {cantidad}')
    print(f'Orden: {cantidad - 1}')
    print(f'Demora: {demora:g} muestras')
    print(f'Demora: {demora / fs_ecg:.3f} s')

#%% Agregar FIR por mínimos cuadrados (Least Squares)

taps_firls = 191

filtros_fir['Mínimos cuadrados'] = sig.firls(
    numtaps=taps_firls,
    bands=[0, wp + 2, wp + delta_w, fs_ecg / 2],
    desired=[1, 1, 0, 0],
    weight=[1, 1],
    fs=fs_ecg
)

# Agregar el filtro a la lista sin duplicarlo.
ventanas_fir = [
    elemento for elemento in ventanas_fir
    if elemento[0] != 'Mínimos cuadrados'
]

ventanas_fir.append(
    ('Mínimos cuadrados', 'firls', 'tab:cyan')
)

# Demora para mostrar en consola.
# Más abajo, tu bucle calculará y guardará las de todos los FIR.
demora_ls = (taps_firls - 1) // 2

print('\nFIR Mínimos cuadrados')
print(f'Coeficientes: {taps_firls}')
print(f'Orden: {taps_firls - 1}')
print(f'Demora: {demora_ls} muestras')
print(f'Demora: {demora_ls / fs_ecg:.3f} s')
#%% Respuesta en frecuencia de los FIR: módulo y fase

fig, (ax_mod, ax_fase) = plt.subplots(
    2, 1,
    figsize=(12, 7),
    sharex=True,
    constrained_layout=True
)

for nombre, ventana, color in ventanas_fir:

    b = filtros_fir[nombre]

    f, h = sig.freqz(
        b,
        1,
        worN=8192,
        fs=fs_ecg
    )

    # Módulo en dB
    modulo_db = 20 * np.log10(
        np.maximum(np.abs(h), 1e-12)
    )

    # Fase en radianes
    fase = np.unwrap(np.angle(h))

    # Ocultar puntos muy próximos a ceros de transmisión
    fase[np.abs(h) < 1e-8] = np.nan

    ax_mod.plot(
        f,
        modulo_db,
        color=color,
        linewidth=1.3,
        label=f'FIR {nombre} — {len(b)} coef.'
    )

    ax_fase.plot(
        f,
        fase,
        color=color,
        linewidth=1.2
    )



ax_mod.fill_between(
    [0, wp], -100, -gpass,
    color='lightcoral',
    alpha=0.25,
    hatch='xxx',
    label='Plantilla'
)

ax_mod.fill_between(
    [0, wp], 0, 5,
    color='lightcoral',
    alpha=0.25,
    hatch='xxx'
)

ax_mod.fill_between(
    [ws, fs_ecg / 2], -gstop, 5,
    color='lightcoral',
    alpha=0.25,
    hatch='xxx'
)

ax_mod.hlines(
    [0, -gpass],
    0, wp,
    colors='gray',
    linestyles='--',
    linewidth=1
)

ax_mod.hlines(
    -gstop,
    ws, fs_ecg / 2,
    colors='gray',
    linestyles='--',
    linewidth=1
)

# Bordes de paso y rechazo en ambos paneles
for ax in (ax_mod, ax_fase):

    ax.axvline(
        wp,
        color='gray',
        linestyle=':',
        linewidth=1.2
    )

    ax.axvline(
        ws,
        color='gray',
        linestyle=':',
        linewidth=1.2
    )

    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)




ax_mod.set_title('Respuesta en frecuencia y plantilla — FIR')
ax_mod.set_ylabel('Módulo [dB]')
ax_mod.set_ylim(-100, 5)
ax_mod.legend(loc='lower left', fontsize=9)

ax_mod.text(
    wp / 2,
    -12,
    f'Banda pasante\n0 a −{gpass:g} dB',
    ha='center',
    fontsize=9
)

ax_mod.text(
    (ws + fs_ecg / 2) / 2,
    -12,
    f'Banda de rechazo\n≤ −{gstop:g} dB',
    ha='center',
    fontsize=9
)

ax_fase.set_ylabel('Fase [rad]')
ax_fase.set_xlabel('Frecuencia [Hz]')
ax_fase.set_xlim(0, fs_ecg / 2)

plt.show()
#%% Aplicar los FIR y calcular sus demoras

N = len(ecg_one_leadruido)
t = np.arange(N) / fs_ecg

ecg_filtrados_fir = {}
demoras_fir = {}

for nombre, ventana, color in ventanas_fir:

    b = filtros_fir[nombre]

    # Filtrado causal de una pasada
    ecg_filtrados_fir[nombre] = sig.lfilter(
        b,
        1,
        ecg_one_leadruido
    )

    # FIR simétricos de longitud impar: demora entera
    demoras_fir[nombre] = (len(b) - 1) // 2

    print(
        f'{nombre}: demora de '
        f'{demoras_fir[nombre]} muestras '
        f'({demoras_fir[nombre] / fs_ecg:.3f} s)'
    )


#%% Comparar el ECG y los FIR compensando la demora

inicio = int(5 * fs_ecg)

# Limitar el intervalo para no superar el registro
# al sumar la demora de cualquiera de los filtros.
demora_maxima = max(demoras_fir.values())
fin = min(int(7 * fs_ecg), N - demora_maxima)

if fin <= inicio:
    raise ValueError('El intervalo elegido no está disponible.')

zoom_region = np.arange(inicio, fin)

fig, ax = plt.subplots(
    figsize=(12, 5),
    constrained_layout=True
)

ax.plot(
    t[zoom_region],
    ecg_one_leadruido[zoom_region],
    color='purple',
    alpha=0.6,
    linewidth=1.5,
    label='ECG sin filtrar'
)

for nombre, ventana, color in ventanas_fir:

    demora = demoras_fir[nombre]

    ax.plot(
        t[zoom_region],
        ecg_filtrados_fir[nombre][zoom_region + demora],
        color=color,
        linewidth=1.3,
        label=f'FIR {nombre} — alineado'
    )

ax.set_title('ECG y filtros FIR — demora compensada')
ax.set_xlabel('Tiempo [s]')
ax.set_ylabel('Amplitud')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right')

plt.show()

#%% Graficar el original y los tres FIR juntos

inicio = int(5 * fs_ecg)
fin = min(int(7 * fs_ecg), N)

fig, ax = plt.subplots(
    figsize=(12, 5),
    constrained_layout=True
)

ax.plot(
    t[inicio:fin],
    ecg_one_leadruido[inicio:fin],
    color='purple',
    alpha=0.6,
    linewidth=1.5,
    label='ECG sin filtrar'
)

for nombre, ventana, color in ventanas_fir:

    ax.plot(
        t[inicio:fin],
        ecg_filtrados_fir[nombre][inicio:fin],
        color=color,
        linewidth=1.3,
        label=f'FIR {nombre}'
    )

ax.set_title('ECG y filtros FIR ')
ax.set_xlabel('Tiempo [s]')
ax.set_ylabel('Amplitud')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right')

plt.show()


#%%
taps_filrs=191
b_filrs= sig.filrs(taps_filrs)