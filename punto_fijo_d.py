from __future__ import annotations

"""
Módulo: punto_fijo.py
---------------------

Implementa el **método de Punto Fijo** para aproximar raíces de una ecuación f(x)=0
usando una transformación x = g(x), junto con una animación tipo *cobweb plot*.

Resumen teórico
~~~~~~~~~~~~~~~
Dado f(x)=0, se reescribe como x = g(x). A partir de un valor inicial x_0,
se define la iteración:

    x_{k+1} = g(x_k)

Bajo las condiciones estándar de convergencia local (existencia de un punto fijo x*,
g continua y derivable en un entorno de x*, y |g'(x*)| < 1), la sucesión {x_k}
converge a x*.

Este módulo expone:
- `punto_fijo`: ejecuta la iteración y devuelve aproximación, iteraciones y un
  historial estructurado.
- `animar_punto_fijo`: genera una animación *cobweb* alternando con paneles en
  LaTeX (si están disponibles) para explicar cada iteración.

Dependencias esperadas desde `utils`:
- `Iteracion`: alias de TypedDict o dict con llaves: i, xi, fxi, xd, fxd, xm, fxm.
- `criterio_cumplido(criterio, tol, fxm, xm, xm_prev)`: evalúa parada por
  criterio (p. ej. |f(x)|, error relativo, etc.).
- `make_interleaved_frames(n, panel_hold)`: secuencia de fotogramas (plot/panel)
  para intercalar gráfica y panel LaTeX.
- `precompute_latex_panels(expr_text, hist, metodo)`: genera/ubica PNG con
  fórmulas por iteración (opcional).

Notas prácticas
~~~~~~~~~~~~~~~
- La convergencia *no* está garantizada para cualquier g(x) y x0. Una mala
  elección puede producir oscilaciones o divergencia.
- Si f(x0)=0, se detecta raíz exacta de inmediato.
- Este diseño conserva el mismo esquema de historial que otros métodos
  (bisección/pos. falsa) para facilitar tablas y visualización.
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


def punto_fijo(
    f: Callable[[float], float],
    g: Callable[[float], float],
    x0: float,
    tol: float = 1e-3,
    criterio: str = "abs_fx",
    max_iter: int = 100,
) -> Tuple[float, int, List[Iteracion]]:
    """
    Ejecuta el **método de Punto Fijo** con la regla x_{k+1} = g(x_k).

    Parameters
    ----------
    f : Callable[[float], float]
        Función objetivo cuya raíz se desea, es decir, buscamos f(x*) = 0.
        Se usa solo para evaluar f en los puntos generados por g.
    g : Callable[[float], float]
        Transformación de punto fijo tal que x = g(x). Define la iteración.
    x0 : float
        Valor inicial (semilla) de la iteración.
    tol : float, optional
        Tolerancia para el criterio de paro (depende de `criterio`). Por defecto 1e-3.
    criterio : str, optional
        Nombre del criterio de paro, interpretado por `criterio_cumplido`.
        Comúnmente:
          - "abs_fx": detiene cuando |f(x_k)| <= tol,
          - "abs_dx": cuando |x_k - x_{k-1}| <= tol,
          - "rel_dx": cuando |x_k - x_{k-1}|/max(1,|x_k|) <= tol, etc.
        Por defecto "abs_fx".
    max_iter : int, optional
        Máximo de iteraciones permitidas. Por defecto 100.

    Returns
    -------
    raiz_aprox : float
        Último x_k (o x_{k+1}) calculado que satisface el criterio o el máximo de iteraciones.
    iter_usadas : int
        Número de iteraciones efectivamente realizadas (índice de la última).
    historial : List[Iteracion]
        Lista de diccionarios con el trazo de la iteración. Llaves:
          - "i"  : índice de iteración k
          - "xi" : x_k
          - "fxi": f(x_k)
          - "xd" : duplicado de x_k (por compatibilidad visual con otros métodos)
          - "fxd": f(x_k)
          - "xm" : x_{k+1} = g(x_k)
          - "fxm": f(x_{k+1})

    Notas
    -----
    - Convergencia local: si existe x* con g(x*)=x* y |g'(x*)|<1 en un entorno,
      para x0 cercano a x* la iteración converge.
    - Una g con |g'|>1 alrededor de la raíz suele divergir.

    Examples
    --------
    >>> # Resolver f(x)=cos(x)-x con g(x)=cos(x)
    >>> import math
    >>> f = lambda x: math.cos(x) - x
    >>> g = lambda x: math.cos(x)
    >>> x_aprox, it, hist = punto_fijo(f, g, x0=0.0, tol=1e-6, criterio="abs_fx")
    >>> round(x_aprox, 6)
    0.739085
    """
    # Inicialización de la iteración
    xk = float(x0)
    fk = f(xk)

    # Raíz exacta en el punto inicial
    if fk == 0.0:
        return xk, 0, [{"i": 0, "xi": xk, "fxi": fk, "xd": xk, "fxd": fk, "xm": xk, "fxm": 0.0}]

    historial: List[Iteracion] = []
    xm_prev: float | None = None  # para criterios basados en desplazamiento

    # Bucle principal de iteración
    for i in range(0, max_iter):
        x_next = g(xk)       # x_{k+1} = g(x_k)
        f_next = f(x_next)   # evaluación de f en el nuevo punto

        # Registro con el mismo "shape" que otros métodos (compatibilidad de tablas/animación)
        historial.append({
            "i": i,
            "xi": xk, "fxi": fk,
            "xd": xk, "fxd": fk,
            "xm": x_next, "fxm": f_next
        })

        # Evaluación de criterio de paro (delegado a utils.criterio_cumplido)
        if criterio_cumplido(criterio, tol, f_next, x_next, xm_prev):
            return x_next, i, historial

        # Avance de iteración
        xk, fk = x_next, f_next
        xm_prev = x_next

    # Si se alcanzó max_iter, devolver el último estado
    return x_next, i, historial  # type: ignore[name-defined]


def animar_punto_fijo(
    f: Callable[[float], float],
    g: Callable[[float], float],
    hist: List[Iteracion],
    titulo: str = "Método de Punto Fijo",
    guardar_gif: bool = False,
    nombre_gif: str = "punto_fijo.gif",
    fps: int = 2,
    intervalo_ms: int = 700,
    expr_text: str = "f(x), g(x)",
    panel_hold: int = 1,
) -> None:
    """
    Genera una animación del **método de Punto Fijo** mediante un *cobweb plot*.

    La animación intercala (1) la gráfica con la curva y=g(x) y la diagonal y=x,
    y (2) paneles LaTeX (si están disponibles) con las expresiones y datos por
    iteración. En la gráfica se dibujan los segmentos vertical y horizontal que
    conectan (x_k, x_k) -> (x_k, g(x_k)) -> (x_{k+1}, x_{k+1}).

    Parameters
    ----------
    f : Callable[[float], float]
        Función f(x) cuyo cero se estudia (se usa para información/etiquetas).
    g : Callable[[float], float]
        Función de iteración x = g(x).
    hist : List[Iteracion]
        Historial generado por `punto_fijo` (se toma de aquí la secuencia x_k).
    titulo : str, optional
        Título mostrado en la figura. Por defecto "Método de Punto Fijo".
    guardar_gif : bool, optional
        Si True, intenta guardar la animación como GIF con PillowWriter.
        Si False, muestra la animación en una ventana (plt.show()).
    nombre_gif : str, optional
        Nombre del archivo GIF a guardar. Por defecto "punto_fijo.gif".
    fps : int, optional
        Cuadros por segundo del GIF. Por defecto 2.
    intervalo_ms : int, optional
        Intervalo en milisegundos entre fotogramas en la animación. Por defecto 700 ms.
    expr_text : str, optional
        Texto (LaTeX-friendly) con las expresiones de f y g para los paneles.
        Ejemplo: r"f(x)=\\cos x - x,\\quad g(x)=\\cos x"
    panel_hold : int, optional
        Número de fotogramas seguidos que se mantiene visible el panel LaTeX
        por cada iteración (intercalado). Por defecto 1.

    Returns
    -------
    None
        Produce una animación en pantalla o un archivo GIF según `guardar_gif`.

    Notas
    -----
    - Requiere que `precompute_latex_panels` devuelva rutas válidas a imágenes PNG
      si se desean paneles LaTeX intercalados.
    - La escala de los ejes se ajusta automáticamente a los x_k y g(x) muestreados.

    Examples
    --------
    >>> # Suponiendo que ya corriste punto_fijo(...) y obtuviste `hist`
    >>> animar_punto_fijo(f, g, hist, titulo="Cobweb para x=cos(x)", guardar_gif=True)
    """
    # Paneles LaTeX por iteración (expr_text debe incluir f y g si se quiere mostrar)
    latex_pngs = precompute_latex_panels(expr_text, hist, metodo="puntofijo")

    # Secuencias desde el historial: x_k y x_{k+1}
    xk_arr = np.array([h["xi"] for h in hist], dtype=float)
    xr_arr = np.array([h["xm"] for h in hist], dtype=float)   # x_{k+1}

    # Rango horizontal basado en {x_k, x_{k+1}}
    xs_all = np.concatenate([xk_arr, xr_arr])
    xmin, xmax = float(np.min(xs_all)), float(np.max(xs_all))
    if xmin == xmax:
        # Si todos los puntos son iguales, abre una ventana de visualización mínima
        xmin, xmax = xmin - 1.0, xmax + 1.0
    dx = abs(xmax - xmin)
    a, b = xmin - 0.08*dx, xmax + 0.08*dx  # margen

    # Malla para graficar g(x) (y opcionalmente f(x))
    muestras = 401
    xs = np.linspace(a, b, muestras)
    gx = np.array([g(x) for x in xs])
    # no se grafica, pero es útil si se requiere
    fx = np.array([f(x) for x in xs])

    # Rango vertical considerando g(x) y la diagonal y=x
    ymax = float(np.max(np.concatenate([gx, xs])))
    ymin = float(np.min(np.concatenate([gx, xs])))
    dy = abs(ymax - ymin) if ymax != ymin else 1.0
    ylo, yhi = ymin - 0.12*dy, ymax + 0.12*dy

    # Figura y ejes principales
    fig = plt.figure(figsize=(10, 5.5))
    ax = fig.add_axes([0.08, 0.10, 0.84, 0.80])
    ax.set_xlim([a, b])
    ax.set_ylim([ylo, yhi])
    ax.set_title(titulo)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.25)

    # Curvas base: y=g(x) y diagonal y=x
    (line_g,) = ax.plot(xs, gx, label="y = g(x)")
    (line_diag,) = ax.plot(xs, xs, linestyle="--", label="y = x")

    # Puntos (x_k, x_k) y (x_{k+1}, x_{k+1}) sobre la diagonal
    (punto_k,) = ax.plot([xk_arr[0]], [xk_arr[0]],
                         "o", color="red", label="(xk, xk)")
    (punto_k1,) = ax.plot([xr_arr[0]], [xr_arr[0]],
                          "o", color="orange", label="(xk+1, xk+1)")

    # Segmentos del cobweb: vertical (x_k, x_k)->(x_k, g(x_k)) y horizontal (x_k, g)->(x_{k+1}, x_{k+1})
    (seg_vert,) = ax.plot([xk_arr[0], xk_arr[0]], [xk_arr[0], xr_arr[0]],
                          color="magenta", linewidth=1.6)
    (seg_horz,) = ax.plot([xk_arr[0], xr_arr[0]], [xr_arr[0], xr_arr[0]],
                          color="magenta", linewidth=1.6)

    ax.legend(loc="best")

    # Cuadro de texto con datos de la iteración actual
    texto = ax.text(
        0.02, 0.98, "",
        transform=ax.transAxes, va="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white",
                  alpha=0.9, edgecolor="0.85"),
    )

    # Eje secundario para panel LaTeX intercalado (oculto por defecto)
    ax_panel = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax_panel.axis("off")
    im_panel = ax_panel.imshow(mpimg.imread(latex_pngs[0]), aspect="auto")
    ax_panel.set_visible(False)

    # Secuencia de fotogramas intercalando 'plot' y 'panel' por iteración
    frames = make_interleaved_frames(len(hist), panel_hold=panel_hold)

    def _update_plot(i: int) -> None:
        """Actualiza los elementos gráficos para la iteración i."""
        xk = xk_arr[i]
        xkp1 = xr_arr[i]

        # Actualiza puntos en la diagonal (x, x)
        punto_k.set_data([xk], [xk])
        punto_k1.set_data([xkp1], [xkp1])

        # Cobweb: primero vertical hasta g(x_k)=x_{k+1}, luego horizontal
        seg_vert.set_data([xk, xk], [xk, xkp1])
        seg_horz.set_data([xk, xkp1], [xkp1, xkp1])

        # Texto con información resumida
        texto.set_text(
            f"Iteración: {i}\n"
            f"xk = {xk:.8g}\n"
            f"x(k+1) = g(xk) ≈ {xkp1:.8g}"
        )

    def init():
        """Función de inicialización para FuncAnimation."""
        ax_panel.set_visible(False)
        ax.set_visible(True)
        return []

    def update(frame):
        """
        Alterna entre:
        - 'plot': muestra la gráfica y actualiza cobweb para la iteración i.
        - 'panel': muestra la imagen LaTeX correspondiente a la iteración i.
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
    ani = FuncAnimation(fig, update, frames=frames, init_func=init,
                        interval=intervalo_ms, blit=False, repeat=False)

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
