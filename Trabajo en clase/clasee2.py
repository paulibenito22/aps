# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 19:20:36 2026

@author: Paulina Benito
"""


#%% Bibliotecas

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
import scipy.io as sio


#%% Lectura del ECG

fs_ecg = 1000  # Hz

# El archivo debe estar en la carpeta de trabajo de Spyder.
mat_struct = sio.loadmat('ECG_TP4.mat')

ecg_one_leadruido = mat_struct['ecg_lead'].flatten().astype(float)

N = len(ecg_one_leadruido)
t = np.arange(N) / fs_ecg

print(f'Cantidad de muestras: {N}')
print(f'Duración: {N / fs_ecg:.2f} s')


#%% Plantilla general: filtro pasa bajos

wp = 35       # Fin de banda de paso [Hz]
ws = 50       # Inicio de banda de rechazo [Hz]

gpass = 3     # Pérdida máxima en banda de paso [dB]
gstop = 40    # Atenuación mínima en banda de rechazo [dB]

nyq_frec = fs_ecg / 2

# Cambiar este valor a 100 para ampliar las bajas frecuencias.
fmax_grafico = nyq_frec

db_min = -100
db_max = 5

tipos_filtros = [
    ('Butterworth', 'butter', 'tab:blue'),
    ('Chebyshev II', 'cheby2', 'tab:orange'),
    ('Cauer', 'ellip', 'tab:green')
]

ventanas_fir = [
    ('Hann', 'hann', 'tab:red'),
    ('Hamming', 'hamming', 'tab:purple'),
    ('Boxcar', 'boxcar', 'tab:brown')
]


#%% Funciones para reutilizar los mismos gráficos

def dibujar_plantilla(ax):
    """Dibujar las regiones prohibidas de la plantilla general."""

    estilo = dict(
        color='lightcoral',
        alpha=0.25,
        hatch='xxx'
    )

    ax.fill_between(
        [0, wp], db_min, -gpass,
        label='Región prohibida',
        **estilo
    )

    ax.fill_between(
        [0, wp], 0, db_max,
        **estilo
    )

    ax.fill_between(
        [ws, nyq_frec], -gstop, db_max,
        **estilo
    )

    ax.hlines(
        [0, -gpass], 0, wp,
        colors='gray',
        linestyles='--',
        linewidth=1
    )

    ax.hlines(
        -gstop, ws, nyq_frec,
        colors='gray',
        linestyles='--',
        linewidth=1
    )

    ax.set_ylim(db_min, db_max)


def graficar_respuestas(respuestas, titulo):
    """
    respuestas contiene:
    (nombre, frecuencias, respuesta compleja, color).
    """

    fig, (ax_mod, ax_fase) = plt.subplots(
        2, 1,
        figsize=(12, 7),
        sharex=True,
        constrained_layout=True
    )

    for nombre, f, H, color in respuestas:

        modulo_db = 20 * np.log10(
            np.maximum(np.abs(H), 1e-12)
        )

        fase = np.unwrap(np.angle(H))

        # La fase no está definida en los ceros de transmisión.
        fase[np.abs(H) < 1e-8] = np.nan

        ax_mod.plot(
            f, modulo_db,
            color=color,
            linewidth=1.4,
            label=nombre
        )

        ax_fase.plot(
            f, fase,
            color=color,
            linewidth=1.3
        )

    dibujar_plantilla(ax_mod)

    for ax in (ax_mod, ax_fase):

        ax.axvline(wp, color='gray', linestyle=':')
        ax.axvline(ws, color='gray', linestyle=':')

        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    ax_mod.set_title(titulo)
    ax_mod.set_ylabel('Módulo [dB]')
    ax_mod.legend(loc='lower left', fontsize=9)

    ax_fase.set_ylabel('Fase [rad]')
    ax_fase.set_xlabel('Frecuencia [Hz]')
    ax_fase.set_xlim(0, fmax_grafico)

    plt.show()


def graficar_ecg(inicio, fin, curvas, titulo, en_muestras=False):
    """
    curvas contiene:
    (nombre, señal filtrada, color, estilo, demora a compensar).

    Si demora = 0, se grafica sin compensación.
    """

    demora_maxima = max(
        (demora for _, _, _, _, demora in curvas),
        default=0
    )

    inicio = max(0, int(round(inicio)))
    fin = min(int(round(fin)), N - demora_maxima)

    if fin <= inicio:
        print(f'Intervalo no disponible: {titulo}')
        return

    indices = np.arange(inicio, fin)

    if en_muestras:
        x = indices
        etiqueta_x = 'Muestras [n]'
    else:
        x = t[indices]
        etiqueta_x = 'Tiempo [s]'

    fig, ax = plt.subplots(
        figsize=(12, 5),
        constrained_layout=True
    )

    ax.plot(
        x,
        ecg_one_leadruido[indices],
        color='gray',
        alpha=0.65,
        linewidth=1.5,
        label='ECG sin filtrar'
    )

    for nombre, salida, color, estilo, demora in curvas:

        ax.plot(
            x,
            salida[indices + demora],
            color=color,
            linestyle=estilo,
            linewidth=1.2,
            label=nombre
        )

    ax.set_title(titulo)
    ax.set_xlabel(etiqueta_x)
    ax.set_ylabel('Amplitud')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.set_xlim(x[0], x[-1])

    plt.show()


#%% ECG sin filtrar: registro completo

fig, ax = plt.subplots(
    figsize=(12, 5),
    constrained_layout=True
)

ax.plot(t, ecg_one_leadruido, color='gray', linewidth=0.8)

ax.set_title('ECG sin filtrar — registro completo')
ax.set_xlabel('Tiempo [s]')
ax.set_ylabel('Amplitud')
ax.grid(True, alpha=0.3)

plt.show()


#%% Diseño IIR: una pasada y dos pasadas

# Guardamos ambos diseños por separado para no sobrescribirlos.
filtros_iir = {}
filtros_iir_ff = {}

for nombre, tipo, color in tipos_filtros:

    # Diseño para aplicar una sola pasada con sosfilt.
    filtros_iir[nombre] = sig.iirdesign(
        wp=wp,
        ws=ws,
        gpass=gpass,
        gstop=gstop,
        ftype=tipo,
        output='sos',
        fs=fs_ecg
    )

    # Diseño para sosfiltfilt:
    # al aplicar dos pasadas se duplican los valores en dB.
    filtros_iir_ff[nombre] = sig.iirdesign(
        wp=wp,
        ws=ws,
        gpass=gpass / 2,
        gstop=gstop / 2,
        ftype=tipo,
        output='sos',
        fs=fs_ecg
    )

    print(
        f'\n{nombre}: '
        f'{len(filtros_iir[nombre])} SOS para sosfilt; '
        f'{len(filtros_iir_ff[nombre])} SOS para sosfiltfilt'
    )


#%% Respuesta en frecuencia IIR: una pasada

# Compatibilidad con versiones anteriores de SciPy.
if hasattr(sig, 'freqz_sos'):
    respuesta_sos = sig.freqz_sos
else:
    respuesta_sos = sig.sosfreqz

respuestas_iir = []

for nombre, tipo, color in tipos_filtros:

    f, H = respuesta_sos(
        filtros_iir[nombre],
        worN=8192,
        fs=fs_ecg
    )

    respuestas_iir.append((nombre, f, H, color))

graficar_respuestas(
    respuestas_iir,
    'IIR — módulo y fase de una pasada'
)


#%% Respuesta efectiva de sosfiltfilt

respuestas_iir_ff = []

for nombre, tipo, color in tipos_filtros:

    f, H = respuesta_sos(
        filtros_iir_ff[nombre],
        worN=8192,
        fs=fs_ecg
    )

    # Respuesta ideal de las dos pasadas:
    # H_efectiva = H · conjugado(H) = |H|².
    # Su fase es cero donde el módulo no se anula.
    H_efectiva = np.abs(H)**2

    respuestas_iir_ff.append(
        (nombre, f, H_efectiva, color)
    )

graficar_respuestas(
    respuestas_iir_ff,
    'IIR — respuesta efectiva de sosfiltfilt, fase cero'
)


#%% Diseño FIR con ventanas y ajustes de frecuencia

taps_win = 131
taps_win_box = 171
delta_w = 5  # Hz

# Estos ajustes pertenecen al diseño.
# La plantilla de evaluación sigue siendo wp, ws, gpass y gstop.
configuracion_fir = {
    'Hann': (
        taps_win,
        [0, wp, wp + delta_w, nyq_frec]
    ),
    'Hamming': (
        taps_win,
        [0, wp + 2, wp + delta_w, nyq_frec]
    ),
    'Boxcar': (
        taps_win_box,
        [0, wp - 2, wp + delta_w, nyq_frec]
    )
}

filtros_fir = {}
demoras_fir = {}

for nombre, ventana, color in ventanas_fir:

    cantidad, frecuencias_diseno = configuracion_fir[nombre]

    filtros_fir[nombre] = sig.firwin2(
        numtaps=cantidad,
        freq=frecuencias_diseno,
        gain=[1, 1, 0, 0],
        window=ventana,
        antisymmetric=False,
        fs=fs_ecg
    )

    # Longitud impar y simetría: FIR tipo I.
    demoras_fir[nombre] = (cantidad - 1) // 2

    print(f'\nFIR {nombre}')
    print(f'Coeficientes: {cantidad}')
    print(f'Orden: {cantidad - 1}')
    print(f'Demora: {demoras_fir[nombre]} muestras')
    print(f'Demora: {demoras_fir[nombre] / fs_ecg:.3f} s')


#%% Respuesta en frecuencia FIR: módulo y fase

respuestas_fir = []

for nombre, ventana, color in ventanas_fir:

    b = filtros_fir[nombre]

    f, H = sig.freqz(
        b,
        1,
        worN=8192,
        fs=fs_ecg
    )

    respuestas_fir.append(
        (f'{nombre} — {len(b)} coef.', f, H, color)
    )

graficar_respuestas(
    respuestas_fir,
    'FIR — módulo, fase y plantilla'
)


#%% Respuestas al impulso: IIR y FIR

N_imp = max(
    500,
    max(len(b) for b in filtros_fir.values())
)

delta = np.zeros(N_imp)
delta[0] = 1

n = np.arange(N_imp)

fig, axes = plt.subplots(
    2, 1,
    figsize=(12, 7),
    sharex=True,
    constrained_layout=True
)

for nombre, tipo, color in tipos_filtros:

    h = sig.sosfilt(filtros_iir[nombre], delta)

    axes[0].plot(
        n, h,
        color=color,
        label=nombre
    )

for nombre, ventana, color in ventanas_fir:

    h = sig.lfilter(filtros_fir[nombre], 1, delta)

    axes[1].plot(
        n, h,
        color=color,
        label=nombre
    )

axes[0].set_title('Respuesta al impulso — IIR de una pasada')
axes[1].set_title('Respuesta al impulso — FIR sin compensar demora')

for ax in axes:
    ax.set_ylabel('h[n]')
    ax.grid(True, alpha=0.3)
    ax.legend()

axes[1].set_xlabel('Muestras [n]')

plt.show()


#%% Retardo de grupo de los filtros causales

# Se calcula para los IIR de una pasada y para los FIR.
# No corresponde al filtrado de fase cero con sosfiltfilt.

fig, axes = plt.subplots(
    2, 1,
    figsize=(12, 7),
    sharex=True,
    constrained_layout=True
)

# Evaluamos hasta 100 Hz para observar mejor la banda útil.
f_retardo = np.linspace(0, min(100, nyq_frec), 4096)

for nombre, tipo, color in tipos_filtros:

    sos = filtros_iir[nombre]
    retardo = np.zeros_like(f_retardo)

    # El retardo total de secciones en cascada es la suma.
    for seccion in sos:

        _, retardo_seccion = sig.group_delay(
            (seccion[:3], seccion[3:]),
            w=f_retardo,
            fs=fs_ecg
        )

        retardo += retardo_seccion

    _, H = respuesta_sos(
        sos,
        worN=f_retardo,
        fs=fs_ecg
    )

    # Ocultar zonas donde el módulo es prácticamente cero.
    retardo[np.abs(H) < 1e-6] = np.nan

    axes[0].plot(
        f_retardo,
        retardo,
        color=color,
        label=nombre
    )

for nombre, ventana, color in ventanas_fir:

    # Para estos FIR de fase lineal:
    # D = (cantidad de coeficientes - 1) / 2.
    retardo = np.full_like(
        f_retardo,
        demoras_fir[nombre],
        dtype=float
    )

    axes[1].plot(
        f_retardo,
        retardo,
        color=color,
        linestyle='--' if nombre == 'Hamming' else '-',
        label=f'{nombre}: {demoras_fir[nombre]} muestras'
    )

axes[0].set_title('Retardo de grupo — IIR con sosfilt')
axes[1].set_title('Retardo de grupo — FIR de fase lineal')

for ax in axes:

    ax.axvline(wp, color='gray', linestyle=':')
    ax.axvline(ws, color='gray', linestyle=':')

    ax.set_ylabel('Retardo [muestras]')
    ax.grid(True, alpha=0.3)
    ax.legend()

axes[1].set_xlabel('Frecuencia [Hz]')

plt.show()


#%% Aplicar todos los filtros al registro completo

ecg_iir_causal = {}
ecg_iir_fase_cero = {}
ecg_filtrados_fir = {}

for nombre, tipo, color in tipos_filtros:

    ecg_iir_causal[nombre] = sig.sosfilt(
        filtros_iir[nombre],
        ecg_one_leadruido
    )

    ecg_iir_fase_cero[nombre] = sig.sosfiltfilt(
        filtros_iir_ff[nombre],
        ecg_one_leadruido
    )

for nombre, ventana, color in ventanas_fir:

    ecg_filtrados_fir[nombre] = sig.lfilter(
        filtros_fir[nombre],
        1,
        ecg_one_leadruido
    )


#%% Preparar los grupos para reutilizarlos en los gráficos

curvas_iir_causal = []
curvas_iir_fase_cero = []
curvas_fir_sin_compensar = []
curvas_fir_alineadas = []

for nombre, tipo, color in tipos_filtros:

    curvas_iir_causal.append((
        f'{nombre} — sosfilt',
        ecg_iir_causal[nombre],
        color,
        '--',
        0
    ))

    curvas_iir_fase_cero.append((
        f'{nombre} — sosfiltfilt',
        ecg_iir_fase_cero[nombre],
        color,
        '--',
        0
    ))

for nombre, ventana, color in ventanas_fir:

    curvas_fir_sin_compensar.append((
        f'FIR {nombre}',
        ecg_filtrados_fir[nombre],
        color,
        '-',
        0
    ))

    curvas_fir_alineadas.append((
        f'FIR {nombre} — alineado',
        ecg_filtrados_fir[nombre],
        color,
        '-',
        demoras_fir[nombre]
    ))

# Compensar aquí significa desplazar los índices al graficar.
# Las señales guardadas conservan su demora causal.
curvas_comparacion = (
    curvas_iir_fase_cero + curvas_fir_alineadas
)


#%% Comparaciones del ECG entre 5 y 7 segundos

inicio = int(5 * fs_ecg)
fin = int(7 * fs_ecg)

comparaciones = [
    (
        'ECG e IIR — sosfilt, una pasada',
        curvas_iir_causal
    ),
    (
        'ECG e IIR — sosfiltfilt, fase cero',
        curvas_iir_fase_cero
    ),
    (
        'ECG y FIR — sin compensar demora',
        curvas_fir_sin_compensar
    ),
    (
        'ECG y FIR — demora compensada al graficar',
        curvas_fir_alineadas
    ),
    (
        'Comparación conjunta — IIR con fase cero y FIR alineados',
        curvas_comparacion
    )
]

for titulo, curvas in comparaciones:

    graficar_ecg(
        inicio,
        fin,
        curvas,
        titulo
    )


#%% Regiones de interés: original, tres IIR y tres FIR

# Intervalos de la consigna, expresados en muestras.
regiones = [
    (
        'Zona con ruido',
        4000,
        5500
    ),
    (
        'Zona con ruido',
        10000,
        11000
    ),
    (
        'Región de interés: 5 a 5.2 minutos',
        5 * 60 * fs_ecg,
        5.2 * 60 * fs_ecg
    ),
    (
        'Región de interés: 12 a 12.4 minutos',
        12 * 60 * fs_ecg,
        12.4 * 60 * fs_ecg
    ),
    (
        'Región de interés: 15 a 15.2 minutos',
        15 * 60 * fs_ecg,
        15.2 * 60 * fs_ecg
    )
]

for descripcion, inicio, fin in regiones:

    graficar_ecg(
        inicio,
        fin,
        curvas_comparacion,
        (
            f'{descripcion}\n'
            'IIR: sosfiltfilt | FIR: demora compensada'
        ),
        en_muestras=True
    )