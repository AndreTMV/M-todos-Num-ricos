from __future__ import annotations

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
    """Derivada numérica central estable a doble precisión."""
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
    Método de Newton–Raphson.

    Historial mantiene compatibilidad de claves:
      - xi := x_k,     fxi := f(x_k)
      - xd := x_k,     fxd := f(x_k)   (duplicado a propósito)
      - xm := x_{k+1}, fxm := f(x_{k+1})
    """
    xk = float(x0)
    fk = f(xk)
    if fk == 0:
        return xk, 0, [{"i": 0, "xi": xk, "fxi": fk, "xd": xk, "fxd": fk, "xm": xk, "fxm": 0.0}]

    historial: List[Iteracion] = []
    xm_prev: float | None = None

    for i in range(0, max_iter):
        d = _deriv_central(f, xk, h_deriv)
        if d == 0:
            # Fallback si la pendiente es ~0: paso pequeño hacia atrás para intentar escapar
            d = 1e-12
        xnext = xk - fk / d
        fnext = f(xnext)

        historial.append({"i": i, "xi": xk, "fxi": fk,
                          "xd": xk, "fxd": fk, "xm": xnext, "fxm": fnext})

        if criterio_cumplido(criterio, tol, fnext, xnext, xm_prev):
            return xnext, i, historial

        xk, fk = xnext, fnext
        xm_prev = xnext

    return xnext, i, historial  # type: ignore[name-defined]


def animar_newton(
    f: Callable[[float], float],
    hist: List[Iteracion],
    # se ignora (sólo por compatibilidad de firma con main)
    xi_inicial: float,
    xd_inicial: float,   # se ignora
    titulo: str = "Método de Newton–Raphson",
    guardar_gif: bool = False,
    nombre_gif: str = "newton.gif",
    fps: int = 2,
    intervalo_ms: int = 700,
    expr_text: str = "f(x)",
    panel_hold: int = 1,
) -> None:
    """
    Animación del método de Newton–Raphson con paneles LaTeX por iteración.
    Muestra la tangente en (x_k, f(x_k)) y su intersección con el eje x (x_{k+1}).
    """
    latex_pngs = precompute_latex_panels(expr_text, hist, metodo="newton")

    xk_arr = np.array([h["xi"] for h in hist], dtype=float)
    fk_arr = np.array([h["fxi"] for h in hist], dtype=float)
    xr_arr = np.array([h["xm"] for h in hist], dtype=float)
    fr_arr = np.array([h["fxm"] for h in hist], dtype=float)

    xs_all = np.concatenate([xk_arr, xr_arr])
    a, b = np.min(xs_all), np.max(xs_all)
    if a == b:
        a, b = a - 1.0, b + 1.0

    muestras = 401
    xs = np.linspace(a, b, muestras)
    fx = np.array([f(x) for x in xs])
    ymax, ymin = np.max(fx), np.min(fx)
    deltax = abs(b - a) if b != a else 1.0
    deltay = abs(ymax - ymin) if ymax != ymin else 1.0
    xlo, xhi = a - 0.08 * deltax, b + 0.08 * deltax
    ylo, yhi = ymin - 0.12 * deltay, ymax + 0.12 * deltay

    fig = plt.figure(figsize=(10, 5.5))
    ax = fig.add_axes([0.08, 0.10, 0.84, 0.80])
    ax.set_xlim([xlo, xhi])
    ax.set_ylim([ylo, yhi])
    ax.set_title(titulo)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(True, alpha=0.25)
    (lineafx,) = ax.plot(xs, fx, label="f(x)")
    ax.axhline(0, color="k", linewidth=1)

    # Elementos: punto x_k, verticales y tangente
    (punto_xk,) = ax.plot([xk_arr[0]], [
        fk_arr[0]], "o", color="red", label="xk")
    (punto_xr,) = ax.plot([xr_arr[0]], [fr_arr[0]],
                          "o", color="orange", label="xk+1")
    (vert_xk,) = ax.plot([xk_arr[0], xk_arr[0]], [0, fk_arr[0]],
                         linestyle="--", color="red", linewidth=1.2)
    (vert_xr,) = ax.plot([xr_arr[0], xr_arr[0]], [0, fr_arr[0]],
                         linestyle="--", color="orange", linewidth=1.2)
    (tangente,) = ax.plot([xk_arr[0], xk_arr[0]], [fk_arr[0], fk_arr[0]],
                          color="magenta", linewidth=1.6, label="tangente en xk")
    ax.legend(loc="best")

    texto = ax.text(
        0.02, 0.98, "",
        transform=ax.transAxes, va="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white",
                  alpha=0.9, edgecolor="0.85"),
    )

    # Panel LaTeX
    ax_panel = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax_panel.axis("off")
    im_panel = ax_panel.imshow(mpimg.imread(latex_pngs[0]), aspect="auto")
    ax_panel.set_visible(False)

    frames = make_interleaved_frames(len(hist), panel_hold=panel_hold)

    def _update_plot(i: int) -> None:
        xk, fk = xk_arr[i], fk_arr[i]
        xr, fr = xr_arr[i], fr_arr[i]

        # actualizar puntos y líneas verticales
        punto_xk.set_data([xk], [fk])
        punto_xr.set_data([xr], [fr])
        vert_xk.set_data([xk, xk], [0, fk])
        vert_xr.set_data([xr, xr], [0, fr])

        # tangente en (xk, fk): y = fk + f'(xk)*(x - xk)
        # la dibujamos en un rango alrededor de xk
        # derivada numérica para la visual (mismo h que en el método)
        # m ≈ f'(xk) usando el salto NR
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
        ax_panel.set_visible(False)
        ax.set_visible(True)
        return []

    def update(frame):
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

    ani = FuncAnimation(fig, update, frames=frames, init_func=init,
                        interval=intervalo_ms, blit=False, repeat=False)

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
