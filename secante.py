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


def secante(
    f: Callable[[float], float],
    x0: float,
    x1: float,
    tol: float = 1e-3,
    criterio: str = "abs_fx",
    max_iter: int = 100,
) -> Tuple[float, int, List[Iteracion]]:
    """
    Método de la Secante (abierto).
    Devuelve (raiz_aprox, iter_usadas, historial).

    En el historial usamos:
      - xi := x_{k-1}, fxi := f(x_{k-1})
      - xd := x_{k},   fxd := f(x_{k})
      - xm := x_{k+1} (nuevo), fxm := f(x_{k+1})
    para mantener compatibilidad visual con bisección/pos. falsa.
    """
    if x0 == x1:
        raise ValueError("x0 y x1 no deben ser iguales.")

    f0, f1 = f(x0), f(x1)

    # si alguno ya es raíz exacta
    if f0 == 0:
        return x0, 0, [{"i": 0, "xi": x0, "fxi": f0, "xd": x1, "fxd": f1, "xm": x0, "fxm": 0.0}]
    if f1 == 0:
        return x1, 0, [{"i": 0, "xi": x0, "fxi": f0, "xd": x1, "fxd": f1, "xm": x1, "fxm": 0.0}]

    historial: List[Iteracion] = []
    xm_prev: float | None = None

    for i in range(0, max_iter):
        denom = (f1 - f0)
        if denom == 0:
            # Si la secante es horizontal o colapsa, forzamos un paso tipo "bisección local"
            xm = (x0 + x1) / 2.0
        else:
            xm = x1 - f1 * (x1 - x0) / denom
        fxm = f(xm)

        historial.append({"i": i, "xi": x0, "fxi": f0,
                          "xd": x1, "fxd": f1, "xm": xm, "fxm": fxm})

        if criterio_cumplido(criterio, tol, fxm, xm, xm_prev):
            return xm, i, historial

        # Avance de ventana secante: (x_{k-1}, x_k) <- (x_k, x_{k+1})
        x0, f0 = x1, f1
        x1, f1 = xm, fxm
        xm_prev = xm

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
    Animación del método de la Secante con paneles LaTeX por iteración.
    """
    # Paneles por iteración
    latex_pngs = precompute_latex_panels(expr_text, hist, metodo="secante")

    # Series (compatibles con claves del historial)
    xa = np.array([h["xi"] for h in hist], dtype=float)   # x_{k-1}
    ya = np.array([h["fxi"] for h in hist], dtype=float)
    xb = np.array([h["xd"] for h in hist], dtype=float)   # x_{k}
    yb = np.array([h["fxd"] for h in hist], dtype=float)
    # x_{k+1} (corte secante)
    xr = np.array([h["xm"] for h in hist], dtype=float)
    yr = np.array([h["fxm"] for h in hist], dtype=float)

    # Rango para graficar: toma todo el recorrido observado
    xs_all = np.concatenate([xa, xb, xr])
    a, b = np.min(xs_all), np.max(xs_all)
    if a == b:  # fallback si todo colapsa
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

    # Puntos y elementos
    (puntoa,) = ax.plot([xa[0]], [ya[0]], "o", color="red", label="xk-1")
    (puntob,) = ax.plot([xb[0]], [yb[0]], "o", color="green", label="xk")
    (puntoc,) = ax.plot([xr[0]], [yr[0]], "o", color="orange", label="xk+1")
    (linea_sec,) = ax.plot([xa[0], xb[0]], [ya[0], yb[0]],
                           color="orange", linewidth=1.6, label="secante")
    (linea_xr,) = ax.plot([xr[0], xr[0]], [0, yr[0]],
                          color="magenta", linestyle="--", linewidth=1.2)
    ax.legend(loc="best")

    # Texto guía
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
