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


def biseccion(
    f: Callable[[float], float],
    xi: float,
    xd: float,
    tol: float = 1e-3,
    criterio: str = "abs_fx",
    max_iter: int = 100,
) -> Tuple[float, int, List[Iteracion]]:
    """
    Método de bisección. Devuelve (raiz_aprox, iter_usadas, historial).
    """
    if xi == xd:
        raise ValueError("xi y xd no deben ser iguales.")
    if xi > xd:
        xi, xd = xd, xi

    fxi, fxd = f(xi), f(xd)
    if fxi == 0:
        return xi, 0, [{"i": 0, "xi": xi, "fxi": fxi, "xd": xd, "fxd": fxd, "xm": xi, "fxm": 0.0}]
    if fxd == 0:
        return xd, 0, [{"i": 0, "xi": xi, "fxi": fxi, "xd": xd, "fxd": fxd, "xm": xd, "fxm": 0.0}]

    historial: List[Iteracion] = []
    xm_prev: float | None = None

    for i in range(0, max_iter):
        xm = (xi + xd) / 2.0
        fxm = f(xm)

        historial.append({"i": i, "xi": xi, "fxi": fxi,
                         "xd": xd, "fxd": fxd, "xm": xm, "fxm": fxm})

        if criterio_cumplido(criterio, tol, fxm, xm, xm_prev):
            return xm, i, historial

        if fxm * fxi > 0:
            xi, fxi = xm, fxm
        else:
            xd, fxd = xm, fxm

        xm_prev = xm

    return xm, i, historial  # type: ignore[name-defined]


def animar_biseccion(
    f: Callable[[float], float],
    hist: List[Iteracion],
    xi_inicial: float,
    xd_inicial: float,
    titulo: str = "Método de Bisección",
    guardar_gif: bool = False,
    nombre_gif: str = "biseccion.gif",
    fps: int = 2,
    intervalo_ms: int = 700,
    expr_text: str = "f(x)",
    panel_hold: int = 1,
) -> None:
    """
    Animación del método de bisección con paneles explicativos por iteración.
    """
    # Pre-render de paneles
    latex_pngs = precompute_latex_panels(expr_text, hist, metodo="biseccion")

    # Serie para el gráfico
    xa = np.array([h["xi"] for h in hist], dtype=float)
    ya = np.array([h["fxi"] for h in hist], dtype=float)
    xb = np.array([h["xd"] for h in hist], dtype=float)
    yb = np.array([h["fxd"] for h in hist], dtype=float)
    xc = np.array([h["xm"] for h in hist], dtype=float)
    yc = np.array([h["fxm"] for h in hist], dtype=float)

    a0, b0 = float(xi_inicial), float(xd_inicial)
    a, b = min(a0, b0), max(a0, b0)

    # Curva de f(x)
    muestras = 401
    xs = np.linspace(a, b, muestras)
    fx = np.array([f(x) for x in xs])
    ymax, ymin = np.max(fx), np.min(fx)
    deltax = abs(b - a) if b != a else 1.0
    deltay = abs(ymax - ymin) if ymax != ymin else 1.0
    xlo, xhi = a - 0.08 * deltax, b + 0.08 * deltax
    ylo, yhi = ymin - 0.12 * deltay, ymax + 0.12 * deltay

    # Figura y elementos
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

    (puntoa,) = ax.plot([xa[0]], [ya[0]], "o", color="red", label="xi")
    (puntob,) = ax.plot([xb[0]], [yb[0]], "o", color="green", label="xf")
    (puntoc,) = ax.plot([xc[0]], [yc[0]], "o", color="orange", label="xm")
    (lineaa,) = ax.plot([xa[0], xa[0]], [0, ya[0]],
                        linestyle="--", color="red", linewidth=1.2)
    (lineab,) = ax.plot([xb[0], xb[0]], [0, yb[0]],
                        linestyle="--", color="green", linewidth=1.2)
    (lineac,) = ax.plot([xc[0], xc[0]], [0, yc[0]],
                        linestyle="--", color="orange", linewidth=1.2)
    (linea_ab,) = ax.plot([xa[0], xb[0]], [
        0, 0], linestyle=":", color="gold", linewidth=1.6, label="[xi,xf]")
    intervalo = ax.fill_between(
        [xa[0], xb[0]], y1=ylo, y2=yhi, color="tab:blue", alpha=0.06, label="intervalo")

    texto = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white",
                  alpha=0.9, edgecolor="0.85"),
    )
    ax.legend(loc="best")

    # Panel LaTeX
    ax_panel = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax_panel.axis("off")
    im_panel = ax_panel.imshow(mpimg.imread(latex_pngs[0]), aspect="auto")
    ax_panel.set_visible(False)

    frames = make_interleaved_frames(len(hist), panel_hold=panel_hold)

    def _update_plot(i: int) -> None:
        puntoa.set_data([xa[i]], [ya[i]])
        puntob.set_data([xb[i]], [yb[i]])
        puntoc.set_data([xc[i]], [yc[i]])
        lineaa.set_data([xa[i], xa[i]], [0, ya[i]])
        lineab.set_data([xb[i], xb[i]], [0, yb[i]])
        lineac.set_data([xc[i], xc[i]], [0, yc[i]])
        linea_ab.set_data([xa[i], xb[i]], [0, 0])
        # refrescar intervalo sombreado (eliminando el previo)
        for coll in list(ax.collections):
            if hasattr(coll, "get_alpha") and coll.get_alpha() == 0.06:
                try:
                    coll.remove()
                except Exception:
                    pass
        ax.fill_between([xa[i], xb[i]], y1=ylo, y2=yhi,
                        color="tab:blue", alpha=0.06)
        texto.set_text(
            f"Iteración: {i}\n"
            f"xi={xa[i]:.8g}, f(xi)={ya[i]:.8g}, xf={xb[i]:.8g}, f(xf)={yb[i]:.8g}\n"
            f"xm={xc[i]:.8g}, f(xm)={yc[i]:.8g}"
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
