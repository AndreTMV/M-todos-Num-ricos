from __future__ import annotations

"""
Módulo: newton_raphson.py
-------------------------

Implementa el **método de Newton–Raphson** para aproximar raíces de f(x)=0
mediante la iteración:

    x_{k+1} = x_k - f(x_k) / f'(x_k)

Además, incluye una función de **animación** que muestra la tangente en cada
iteración y su intersección con el eje x (corte que define x_{k+1}), con
paneles LaTeX intercalados (si están disponibles).

Dependencias esperadas desde `utils`:
- `Iteracion`: alias de TypedDict o dict con llaves: i, xi, fxi, xd, fxd, xm, fxm.
- `criterio_cumplido(criterio, tol, fxm, xm, xm_prev)`: evalúa el criterio de paro
  (p. ej. |f(x)|, error absoluto/relativo en x, etc.).
- `make_interleaved_frames(n, panel_hold)`: construye la secuencia de fotogramas
  para alternar “gráfica” y “panel LaTeX”.
- `precompute_latex_panels(expr_text, hist, metodo)`: genera/ubica PNG con fórmulas
  por iteración (opcional).

Notas teóricas
~~~~~~~~~~~~~~
- Convergencia local: si f es C^1 en un entorno de la raíz simple x* (f(x*)=0, f'(x*)≠0),
  Newton es típicamente **cuadráticamente** convergente para x0 cercano a x*.
- Si f'(x*)=0 (raíz múltiple) la convergencia puede degradarse a lineal.
- Si f'(x_k)≈0 o el punto cae fuera de la cuenca de atracción, puede haber
  divergencia u oscilaciones.
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


def _deriv_central(f: Callable[[float], float], x: float, h: float = 1e-6) -> float:
    """
    Aproximación de la derivada f'(x) por **diferencia central**.

    Parameters
    ----------
    f : Callable[[float], float]
        Función a derivar.
    x : float
        Punto de evaluación.
    h : float, optional
        Paso de la diferencia. Debe ser lo bastante pequeño para consistencia,
        pero no tan pequeño que induzca cancelación numérica (por defecto 1e-6).

    Returns
    -------
    float
        Aproximación numérica de f'(x) usando:
            (f(x+h) - f(x-h)) / (2h)

    Notas
    -----
    - La fórmula central es de **orden O(h^2)** (más precisa que forward/backward).
    - En práctica, h ≈ sqrt(eps) * escala(x) suele equilibrar error de redondeo
      y truncamiento (aquí se fija un valor razonable para doble precisión).
    """
    return (f(x + h) - f(x - h)) / (2.0 * h)


def newton_raphson(
    f: Callable[[float], float],
    x0: float,
    tol: float = 1e-3,
    criterio: str = "abs_fx",
    max_iter: int = 100,
    h_deriv: float = 1e-6,
) -> Tuple[float, int, List[Iteracion]]:
    """
    Ejecuta el **método de Newton–Raphson** para resolver f(x)=0.

    La derivada se aproxima numéricamente vía diferencia central `_deriv_central`
    con paso `h_deriv`.

    Parameters
    ----------
    f : Callable[[float], float]
        Función objetivo (se busca f(x*)=0).
    x0 : float
        Valor inicial de la iteración.
    tol : float, optional
        Tolerancia para el criterio de paro (interpretada por `criterio_cumplido`).
        Por defecto 1e-3.
    criterio : str, optional
        Nombre del criterio de paro. Ejemplos (dependen de `utils`):
          - "abs_fx": |f(x_k)| <= tol
          - "abs_dx": |x_k - x_{k-1}| <= tol
          - "rel_dx": |x_k - x_{k-1}|/max(1,|x_k|) <= tol
        Por defecto "abs_fx".
    max_iter : int, optional
        Máximo de iteraciones. Por defecto 100.
    h_deriv : float, optional
        Paso para la derivada numérica central. Por defecto 1e-6.

    Returns
    -------
    raiz_aprox : float
        Aproximación de la raíz al finalizar por criterio o por máximo de iteraciones.
    iter_usadas : int
        Iteraciones efectivamente realizadas.
    historial : List[Iteracion]
        Trazo de iteraciones con las llaves:
          - "i"   : índice k
          - "xi"  : x_k
          - "fxi" : f(x_k)
          - "xd"  : x_k          (duplicado por compatibilidad visual)
          - "fxd" : f(x_k)
          - "xm"  : x_{k+1}
          - "fxm" : f(x_{k+1})

    Notas
    -----
    - Si f(x0)=0, se detecta raíz exacta y se retorna de inmediato.
    - Si la derivada numérica sale ~0, se usa un **fallback mínimo** (1e-12) para
      evitar división por cero. Esto es un “escape” numérico, no una garantía de
      convergencia: idealmente, el usuario debe revisar x0 o reparametrizar f.
    - Convergencia cuadrática requiere raíz simple y x0 suficientemente cercano.
    """
    xk = float(x0)
    fk = f(xk)

    # Raíz exacta desde el inicio
    if fk == 0:
        return xk, 0, [{"i": 0, "xi": xk, "fxi": fk, "xd": xk, "fxd": fk, "xm": xk, "fxm": 0.0}]

    historial: List[Iteracion] = []
    xm_prev: float | None = None  # para criterios basados en desplazamiento

    for i in range(0, max_iter):
        # Derivada numérica en x_k
        d = _deriv_central(f, xk, h_deriv)
        if d == 0:
            # Fallback si la pendiente es ~0: evita división por cero.
            # (Advertencia: puede degradar el paso; revisar x0 o reescala)
            d = 1e-12

        # Paso de Newton: x_{k+1} = x_k - f(x_k)/f'(x_k)
        xnext = xk - fk / d
        fnext = f(xnext)

        # Registro con el mismo “shape” que otros métodos del proyecto
        historial.append({
            "i": i,
            "xi": xk, "fxi": fk,
            "xd": xk, "fxd": fk,
            "xm": xnext, "fxm": fnext
        })

        # Criterio de paro (delegado a utils)
        if criterio_cumplido(criterio, tol, fnext, xnext, xm_prev):
            return xnext, i, historial

        # Avance
        xk, fk = xnext, fnext
        xm_prev = xnext

    # Si se alcanzó max_iter, devuelve el último estado
    return xnext, i, historial  # type: ignore[name-defined]


def animar_newton(
    f: Callable[[float], float],
    hist: List[Iteracion],
    # Parámetros ignorados (compatibilidad de firma con un `main` externo):
    xi_inicial: float,
    xd_inicial: float,
    titulo: str = "Método de Newton–Raphson",
    guardar_gif: bool = False,
    nombre_gif: str = "newton.gif",
    fps: int = 2,
    intervalo_ms: int = 700,
    expr_text: str = "f(x)",
    panel_hold: int = 1,
) -> None:
    """
    Genera una animación del **método de Newton–Raphson** con paneles LaTeX.

    En cada iteración se muestra:
      1) El punto (x_k, f(x_k)).
      2) La **tangente** a la curva f en x_k.
      3) La intersección de esa tangente con el eje x, que define x_{k+1}.

    Parameters
    ----------
    f : Callable[[float], float]
        Función f(x) que se grafica para contexto visual.
    hist : List[Iteracion]
        Historial producido por `newton_raphson`.
    xi_inicial : float
        (Ignorado) incluido solo para compatibilidad con otras firmas de animación.
    xd_inicial : float
        (Ignorado) incluido solo para compatibilidad con otras firmas de animación.
    titulo : str, optional
        Título de la figura. Por defecto "Método de Newton–Raphson".
    guardar_gif : bool, optional
        Si True, intenta guardar la animación como GIF (PillowWriter).
        Si False, muestra la animación en una ventana. Por defecto False.
    nombre_gif : str, optional
        Nombre del archivo GIF de salida. Por defecto "newton.gif".
    fps : int, optional
        Cuadros por segundo para el GIF. Por defecto 2.
    intervalo_ms : int, optional
        Intervalo (ms) entre fotogramas en pantalla. Por defecto 700 ms.
    expr_text : str, optional
        Texto (LaTeX-friendly) con la expresión de f(x) para paneles.
        Ej.: r"f(x)=\\cos x - x".
    panel_hold : int, optional
        Cuántos fotogramas mantener visible el panel por iteración al intercalar.
        Por defecto 1.

    Returns
    -------
    None
        Muestra la animación o guarda el GIF según `guardar_gif`.

    Notas
    -----
    - La pendiente de la tangente en la figura se estima a partir de los puntos
      consecutivos (x_k, f(x_k)) y (x_{k+1}, f(x_{k+1})) como:
          m ≈ (f(x_{k+1}) - f(x_k)) / (x_{k+1} - x_k)
      con fines **visual-didácticos**.
    - Si `precompute_latex_panels` devuelve rutas válidas, se intercalan paneles
      LaTeX describiendo cada iteración.
    """
    # Prepara paneles LaTeX (si existen)
    latex_pngs = precompute_latex_panels(expr_text, hist, metodo="newton")

    # Extrae secuencias de iteración
    xk_arr = np.array([h["xi"] for h in hist], dtype=float)
    fk_arr = np.array([h["fxi"] for h in hist], dtype=float)
    xr_arr = np.array([h["xm"] for h in hist], dtype=float)
    fr_arr = np.array([h["fxm"] for h in hist], dtype=float)

    # Rango horizontal centrado en {x_k, x_{k+1}}
    xs_all = np.concatenate([xk_arr, xr_arr])
    a, b = np.min(xs_all), np.max(xs_all)
    if a == b:
        a, b = a - 1.0, b + 1.0

    # Malla para f(x) y rango vertical
    muestras = 401
    xs = np.linspace(a, b, muestras)
    fx = np.array([f(x) for x in xs])
    ymax, ymin = np.max(fx), np.min(fx)

    deltax = abs(b - a) if b != a else 1.0
    deltay = abs(ymax - ymin) if ymax != ymin else 1.0
    xlo, xhi = a - 0.08 * deltax, b + 0.08 * deltax
    ylo, yhi = ymin - 0.12 * deltay, ymax + 0.12 * deltay

    # Figura y ejes
    fig = plt.figure(figsize=(10, 5.5))
    ax = fig.add_axes([0.08, 0.10, 0.84, 0.80])
    ax.set_xlim([xlo, xhi])
    ax.set_ylim([ylo, yhi])
    ax.set_title(titulo)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(True, alpha=0.25)

    # Curva de f y eje x
    (lineafx,) = ax.plot(xs, fx, label="f(x)")
    ax.axhline(0, color="k", linewidth=1)

    # Elementos dinámicos: puntos, verticales y tangente
    (punto_xk,) = ax.plot([xk_arr[0]], [
        fk_arr[0]], "o", color="red", label="xk")
    (punto_xr,) = ax.plot([xr_arr[0]], [
        fr_arr[0]], "o", color="orange", label="xk+1")
    (vert_xk,) = ax.plot([xk_arr[0], xk_arr[0]], [0, fk_arr[0]],
                         linestyle="--", color="red", linewidth=1.2)
    (vert_xr,) = ax.plot([xr_arr[0], xr_arr[0]], [0, fr_arr[0]],
                         linestyle="--", color="orange", linewidth=1.2)
    (tangente,) = ax.plot([xk_arr[0], xk_arr[0]], [fk_arr[0], fk_arr[0]],
                          color="magenta", linewidth=1.6, label="tangente en xk")
    ax.legend(loc="best")

    # Texto con datos de la iteración
    texto = ax.text(
        0.02, 0.98, "",
        transform=ax.transAxes, va="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white",
                  alpha=0.9, edgecolor="0.85"),
    )

    # Eje secundario para panel LaTeX (oculto por defecto)
    ax_panel = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax_panel.axis("off")
    im_panel = ax_panel.imshow(mpimg.imread(latex_pngs[0]), aspect="auto")
    ax_panel.set_visible(False)

    # Secuencia de fotogramas (intercalando plot y panel)
    frames = make_interleaved_frames(len(hist), panel_hold=panel_hold)

    def _update_plot(i: int) -> None:
        """
        Actualiza los elementos de la gráfica para la iteración i:
        puntos (x_k, f(x_k)) y (x_{k+1}, f(x_{k+1})),
        verticales, y la recta tangente aproximada.
        """
        xk, fk = xk_arr[i], fk_arr[i]
        xr, fr = xr_arr[i], fr_arr[i]

        # Puntos y verticales
        punto_xk.set_data([xk], [fk])
        punto_xr.set_data([xr], [fr])
        vert_xk.set_data([xk, xk], [0, fk])
        vert_xr.set_data([xr, xr], [0, fr])

        # Tangente aproximada:
        # m ≈ (f(x_{k+1}) - f(x_k)) / (x_{k+1} - x_k) (solo con fin visual)
        m = (fr - fk) / (xr - xk) if (xr != xk) else 0.0
        xt = np.linspace(xlo, xhi, 50)
        yt = fk + m * (xt - xk)
        tangente.set_data(xt, yt)

        texto.set_text(
            f"Iteración: {i}\n"
            f"xk={xk:.8g}, f(xk)={fk:.8g}\n"
            f"x(k+1)={xr:.8g}, f(x(k+1))={fr:.8g}"
        )

    def init():
        """Inicializa la animación (muestra la gráfica y oculta panel)."""
        ax_panel.set_visible(False)
        ax.set_visible(True)
        return []

    def update(frame):
        """
        Alterna entre:
        - 'plot': actualiza la gráfica para la iteración i.
        - 'panel': muestra la imagen LaTeX de la iteración i.
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
