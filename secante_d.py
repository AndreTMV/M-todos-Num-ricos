from __future__ import annotations

"""
Módulo: secante.py
------------------

Implementa el **método de la Secante** (método abierto) para aproximar raíces de f(x)=0,
utilizando dos aproximaciones iniciales x0 y x1 y actualizando por la fórmula:

    x_{k+1} = x_k - f(x_k) * (x_k - x_{k-1}) / (f(x_k) - f(x_{k-1}))

Incluye además una función para **animar** el proceso iterativo, mostrando la recta secante
entre (x_{k-1}, f(x_{k-1})) y (x_k, f(x_k)), y su intersección con el eje x (x_{k+1}).
Los paneles LaTeX (si están disponibles) se intercalan para explicar cada iteración.

Compatibilidad
~~~~~~~~~~~~~~
El historial de iteraciones usa las mismas llaves que bisección/posición falsa para
reutilizar tablas y animadores existentes:

- "xi"  := x_{k-1}, "fxi" := f(x_{k-1})
- "xd"  := x_{k},   "fxd" := f(x_{k})
- "xm"  := x_{k+1}, "fxm" := f(x_{k+1})

Notas teóricas
~~~~~~~~~~~~~~
- La secante **no requiere derivadas** (a diferencia de Newton).
- La convergencia típica es **superlineal** (~1.618) para raíces simples,
  siempre que las aproximaciones iniciales entren a la cuenca de atracción.
- Si f(x_k) ≈ f(x_{k-1}) la fórmula se vuelve inestable (denominador ~0).
  Aquí se incluye un *fallback* seguro: tomar el punto medio (bisección local).
"""

from typing import Callable, List, Tuple

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from utils import (
    Iteracion,
    criterio_cumplido,
    make_interleaved_frames,
    precompute_latex_panels,
)


def secante(
    f: Callable[[float], float],
    x0: float,
    x1: float,
    tol: float = 1e-3,
    criterio: str = "abs_fx",
    max_iter: int = 100,
) -> Tuple[float, int, List[Iteracion]]:
    """
    Ejecuta el **método de la Secante** (método abierto) para resolver f(x)=0.

    Parameters
    ----------
    f : Callable[[float], float]
        Función objetivo (se busca x* tal que f(x*)=0).
    x0 : float
        Primera aproximación inicial (x_{-1} en la notación del método).
    x1 : float
        Segunda aproximación inicial (x_{0} en la notación del método).
    tol : float, optional
        Tolerancia interpretada por `criterio_cumplido`. Por defecto 1e-3.
    criterio : str, optional
        Criterio de paro. Ejemplos (depende de `utils`):
          - "abs_fx": |f(x_k)| <= tol
          - "abs_dx": |x_k - x_{k-1}| <= tol
          - "rel_dx": |x_k - x_{k-1}|/max(1,|x_k|) <= tol
        Por defecto "abs_fx".
    max_iter : int, optional
        Máximo de iteraciones. Por defecto 100.

    Returns
    -------
    raiz_aprox : float
        Aproximación a la raíz encontrada (x_{k+1} más reciente).
    iter_usadas : int
        Número de iteraciones realizadas.
    historial : List[Iteracion]
        Lista de diccionarios con el trazo:
          - "i"   : índice k
          - "xi"  : x_{k-1}, "fxi": f(x_{k-1})
          - "xd"  : x_{k},   "fxd": f(x_{k})
          - "xm"  : x_{k+1}, "fxm": f(x_{k+1})

    Raises
    ------
    ValueError
        Si x0 == x1 (la secante colapsaría de inicio).

    Notas
    -----
    - Si f(x0)=0 o f(x1)=0, se retorna de inmediato con raíz exacta.
    - Si el denominador (f1 - f0) es 0 (o muy cercano), se usa un *fallback*
      robusto: el punto medio (x0 + x1)/2 para evitar división por cero y
      continuar el proceso de forma estable.
    """
    if x0 == x1:
        raise ValueError("x0 y x1 no deben ser iguales.")

    f0, f1 = f(x0), f(x1)

    # Detección temprana de raíz exacta
    if f0 == 0:
        return x0, 0, [{"i": 0, "xi": x0, "fxi": f0, "xd": x1, "fxd": f1, "xm": x0, "fxm": 0.0}]
    if f1 == 0:
        return x1, 0, [{"i": 0, "xi": x0, "fxi": f0, "xd": x1, "fxd": f1, "xm": x1, "fxm": 0.0}]

    historial: List[Iteracion] = []
    xm_prev: float | None = None  # para criterios basados en desplazamiento

    for i in range(0, max_iter):
        denom = (f1 - f0)
        if denom == 0:
            # Secante horizontal/colapsada: evita división por cero con un paso seguro
            xm = (x0 + x1) / 2.0
        else:
            # Fórmula de la secante: intersección con el eje x de la recta entre (x0,f0) y (x1,f1)
            xm = x1 - f1 * (x1 - x0) / denom

        fxm = f(xm)

        # Registro con el mismo "shape" que otros métodos del proyecto
        historial.append({
            "i": i,
            "xi": x0, "fxi": f0,
            "xd": x1, "fxd": f1,
            "xm": xm, "fxm": fxm
        })

        # Criterio de paro delegado a utils
        if criterio_cumplido(criterio, tol, fxm, xm, xm_prev):
            return xm, i, historial

        # Avance de ventana secante: (x_{k-1}, x_k) <- (x_k, x_{k+1})
        x0, f0 = x1, f1
        x1, f1 = xm, fxm
        xm_prev = xm

    # Fin por máximo de iteraciones: devuelve último estado
    return xm, i, historial  # type: ignore[name-defined]


def animar_secante(
    f: Callable[[float], float],
    hist: List[Iteracion],
    xi_inicial: float,
    xd_inicial: float,
    titulo: str = "Método de la Secante",
    guardar_gif: bool = False,
    nombre_gif: str = "secante.gif",
    fps: int = 2,
    intervalo_ms: int = 700,
    expr_text: str = "f(x)",
    panel_hold: int = 1,
) -> None:
    """
    Genera una **animación** del método de la Secante con paneles LaTeX por iteración.

    En cada paso se muestran:
      - Los puntos (x_{k-1}, f(x_{k-1})) y (x_k, f(x_k)).
      - La **recta secante** entre esos puntos.
      - El corte de la secante con el eje x: x_{k+1}.

    Parameters
    ----------
    f : Callable[[float], float]
        Función f(x) que se grafica de fondo para referencia.
    hist : List[Iteracion]
        Historial producido por `secante`.
    xi_inicial : float
        Valor inicial izquierdo (se usa solo para compatibilidad de firma).
    xd_inicial : float
        Valor inicial derecho (se usa solo para compatibilidad de firma).
    titulo : str, optional
        Título de la figura. Por defecto "Método de la Secante".
    guardar_gif : bool, optional
        Si True, intenta guardar la animación como GIF con PillowWriter.
        Si False, muestra la animación en una ventana. Por defecto False.
    nombre_gif : str, optional
        Nombre del archivo GIF. Por defecto "secante.gif".
    fps : int, optional
        Cuadros por segundo del GIF. Por defecto 2.
    intervalo_ms : int, optional
        Intervalo (ms) entre fotogramas. Por defecto 700 ms.
    expr_text : str, optional
        Texto LaTeX-friendly con la expresión de f(x) para paneles.
        Ej.: r"f(x)=\\cos x - x".
    panel_hold : int, optional
        Cuántos fotogramas mantener el panel por iteración al intercalar. Por defecto 1.

    Returns
    -------
    None
        Muestra la animación o guarda el GIF, según `guardar_gif`.

    Notas
    -----
    - `precompute_latex_panels` debe devolver rutas válidas a imágenes si se desea
      intercalar paneles LaTeX por iteración.
    """
    # Paneles por iteración (si están disponibles)
    latex_pngs = precompute_latex_panels(expr_text, hist, metodo="secante")

    # Series desde el historial (compatibles con bisección/pos. falsa)
    xa = np.array([h["xi"] for h in hist], dtype=float)   # x_{k-1}
    ya = np.array([h["fxi"] for h in hist], dtype=float)
    xb = np.array([h["xd"] for h in hist], dtype=float)   # x_{k}
    yb = np.array([h["fxd"] for h in hist], dtype=float)
    xr = np.array([h["xm"] for h in hist], dtype=float)   # x_{k+1}
    yr = np.array([h["fxm"] for h in hist], dtype=float)

    # Rango para graficar en función del recorrido observado
    xs_all = np.concatenate([xa, xb, xr])
    a, b = np.min(xs_all), np.max(xs_all)
    if a == b:  # fallback si todo colapsa a un punto
        a, b = a - 1.0, b + 1.0

    muestras = 401
    xs = np.linspace(a, b, muestras)
    fx = np.array([f(x) for x in xs])

    ymax, ymin = np.max(fx), np.min(fx)
    deltax = abs(b - a) if b != a else 1.0
    deltay = abs(ymax - ymin) if ymax != ymin else 1.0
    xlo, xhi = a - 0.08 * deltax, b + 0.08 * deltax
    ylo, yhi = ymin - 0.12 * deltay, ymax + 0.12 * deltay

    # Figura
    fig = plt.figure(figsize=(10, 5.5))
    ax = fig.add_axes([0.08, 0.10, 0.84, 0.80])
    ax.set_xlim([xlo, xhi])
    ax.set_ylim([ylo, yhi])
    ax.set_title(titulo)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(True, alpha=0.25)

    # Curva de f(x) y eje x
    (lineafx,) = ax.plot(xs, fx, label="f(x)")
    ax.axhline(0, color="k", linewidth=1)

    # Puntos y elementos dinámicos
    (puntoa,) = ax.plot([xa[0]], [ya[0]], "o", color="red",   label="xk-1")
    (puntob,) = ax.plot([xb[0]], [yb[0]], "o", color="green", label="xk")
    (puntoc,) = ax.plot([xr[0]], [yr[0]], "o", color="orange", label="xk+1")
    (linea_sec,) = ax.plot([xa[0], xb[0]], [ya[0], yb[0]],
                           color="orange", linewidth=1.6, label="secante")
    (linea_xr,) = ax.plot([xr[0], xr[0]], [0, yr[0]],
                          color="magenta", linestyle="--", linewidth=1.2)
    ax.legend(loc="best")

    # Texto guía con datos de la iteración
    texto = ax.text(
        0.02, 0.98, "",
        transform=ax.transAxes, va="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white",
                  alpha=0.9, edgecolor="0.85"),
    )

    # Panel LaTeX intercalado
    ax_panel = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax_panel.axis("off")
    im_panel = ax_panel.imshow(mpimg.imread(latex_pngs[0]), aspect="auto")
    ax_panel.set_visible(False)

    # Secuencia de fotogramas (intercala gráfica y panel)
    frames = make_interleaved_frames(len(hist), panel_hold=panel_hold)

    def _update_plot(i: int) -> None:
        """Actualiza puntos, secante y marcador de x_{k+1} para la iteración i."""
        puntoa.set_data([xa[i]], [ya[i]])
        puntob.set_data([xb[i]], [yb[i]])
        puntoc.set_data([xr[i]], [yr[i]])
        linea_sec.set_data([xa[i], xb[i]], [ya[i], yb[i]])
        linea_xr.set_data([xr[i], xr[i]], [0, yr[i]])

        texto.set_text(
            f"Iteración: {i}\n"
            f"x(k-1)={xa[i]:.8g}, f(x(k-1))={ya[i]:.8g}\n"
            f"xk={xb[i]:.8g}, f(xk)={yb[i]:.8g}\n"
            f"x(k+1)={xr[i]:.8g}, f(x(k+1))={yr[i]:.8g}"
        )

    def init():
        """Inicializa la animación ocultando panel y mostrando la gráfica."""
        ax_panel.set_visible(False)
        ax.set_visible(True)
        return []

    def update(frame):
        """
        Alterna entre:
        - 'plot': actualiza la gráfica (puntos, secante, corte).
        - 'panel': muestra el panel LaTeX correspondiente a la iteración.
        """
        kind, i = frame
        if kind == "plot":
            if not ax.get_visible():
                ax.set_visible(True)
            if ax_panel.get_visible():
                ax_panel.set_visible(False)
            _update_plot(i)
            return []
        else:
            if not ax_panel.get_visible():
                ax_panel.set_visible(True)
            if ax.get_visible():
                ax.set_visible(False)
            im_panel.set_data(mpimg.imread(latex_pngs[i]))
            return [im_panel]

    # Construcción de la animación
    ani = FuncAnimation(
        fig, update, frames=frames, init_func=init,
        interval=intervalo_ms, blit=False, repeat=False
    )

    # Guardar o mostrar
    if guardar_gif:
        try:
            from matplotlib.animation import PillowWriter
            ani.save(nombre_gif, writer=PillowWriter(fps=fps))
            print(f"GIF guardado como: {nombre_gif}")
        except Exception as e:
            print(
                f"No se pudo guardar el GIF: {e}\nMostrando animación en ventana...")
            plt.show()
    else:
        plt.show()
