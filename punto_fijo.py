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


def punto_fijo(
    f: Callable[[float], float],
    g: Callable[[float], float],
    x0: float,
    tol: float = 1e-3,
    criterio: str = "abs_fx",
    max_iter: int = 100,
) -> Tuple[float, int, List[Iteracion]]:
    """
    Método de Punto Fijo: x_{k+1} = g(x_k)
    Devuelve (raiz_aprox, iter_usadas, historial)

    Historial manteniendo el mismo esquema de claves:
      - xi := x_k
      - xd := x_k (duplicado intencionalmente para compatibilidad visual)
      - xm := x_{k+1} = g(x_k)
      - fxi, fxd := f(x_k)
      - fxm := f(x_{k+1})
    """
    xk = float(x0)
    fk = f(xk)
    if fk == 0.0:
        return xk, 0, [{"i": 0, "xi": xk, "fxi": fk, "xd": xk, "fxd": fk, "xm": xk, "fxm": 0.0}]

    historial: List[Iteracion] = []
    xm_prev: float | None = None

    for i in range(0, max_iter):
        x_next = g(xk)
        f_next = f(x_next)

        historial.append({
            "i": i,
            "xi": xk, "fxi": fk,
            "xd": xk, "fxd": fk,
            "xm": x_next, "fxm": f_next
        })

        if criterio_cumplido(criterio, tol, f_next, x_next, xm_prev):
            return x_next, i, historial

        xk, fk = x_next, f_next
        xm_prev = x_next

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
    Animación del método de Punto Fijo con 'cobweb plot':
    - Curva y = g(x)
    - Diagonal y = x
    - Segmentos vertical y horizontal que llevan de x_k a x_{k+1}
    """
    # Paneles LaTeX por iteración (expr_text debe incluir f y g)
    latex_pngs = precompute_latex_panels(expr_text, hist, metodo="puntofijo")

    # Secuencias desde el historial
    xk_arr = np.array([h["xi"] for h in hist], dtype=float)
    xr_arr = np.array([h["xm"] for h in hist], dtype=float)   # x_{k+1}

    xs_all = np.concatenate([xk_arr, xr_arr])
    xmin, xmax = float(np.min(xs_all)), float(np.max(xs_all))
    if xmin == xmax:
        xmin, xmax = xmin - 1.0, xmax + 1.0
    # margen
    dx = abs(xmax - xmin)
    a, b = xmin - 0.08*dx, xmax + 0.08*dx

    muestras = 401
    xs = np.linspace(a, b, muestras)
    gx = np.array([g(x) for x in xs])
    # opcional (no se grafica, pero útil si lo quieres)
    fx = np.array([f(x) for x in xs])

    # Rango vertical con base en g(x) y la diagonal
    ymax = float(np.max(np.concatenate([gx, xs])))
    ymin = float(np.min(np.concatenate([gx, xs])))
    dy = abs(ymax - ymin) if ymax != ymin else 1.0
    ylo, yhi = ymin - 0.12*dy, ymax + 0.12*dy

    # Figura
    fig = plt.figure(figsize=(10, 5.5))
    ax = fig.add_axes([0.08, 0.10, 0.84, 0.80])
    ax.set_xlim([a, b])
    ax.set_ylim([ylo, yhi])
    ax.set_title(titulo)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.25)

    (line_g,) = ax.plot(xs, gx, label="y = g(x)")
    (line_diag,) = ax.plot(xs, xs, linestyle="--", label="y = x")
    (punto_k,) = ax.plot([xk_arr[0]], [xk_arr[0]],
                         "o", color="red", label="(xk, xk)")
    (punto_k1,) = ax.plot([xr_arr[0]], [xr_arr[0]],
                          "o", color="orange", label="(xk+1, xk+1)")

    # Segmentos del "cobweb": vertical (x_k, x_k) -> (x_k, g(x_k)) y horizontal (x_k, g) -> (x_{k+1}, x_{k+1})
    (seg_vert,) = ax.plot([xk_arr[0], xk_arr[0]], [
        xk_arr[0], xr_arr[0]], color="magenta", linewidth=1.6)
    (seg_horz,) = ax.plot([xk_arr[0], xr_arr[0]], [
        xr_arr[0], xr_arr[0]], color="magenta", linewidth=1.6)

    ax.legend(loc="best")

    # Texto
    texto = ax.text(
        0.02, 0.98, "",
        transform=ax.transAxes, va="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white",
                  alpha=0.9, edgecolor="0.85"),
    )

    # Panel LaTeX (intercalado)
    ax_panel = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax_panel.axis("off")
    im_panel = ax_panel.imshow(mpimg.imread(latex_pngs[0]), aspect="auto")
    ax_panel.set_visible(False)

    frames = make_interleaved_frames(len(hist), panel_hold=panel_hold)

    def _update_plot(i: int) -> None:
        xk = xk_arr[i]
        xkp1 = xr_arr[i]

        # puntos en diagonal (x, x)
        punto_k.set_data([xk], [xk])
        punto_k1.set_data([xkp1], [xkp1])

        # cobweb: vertical hasta g(x_k) = x_{k+1} y luego horizontal
        seg_vert.set_data([xk, xk], [xk, xkp1])
        seg_horz.set_data([xk, xkp1], [xkp1, xkp1])

        texto.set_text(
            f"Iteración: {i}\n"
            f"xk={xk:.8g}\n"
            f"x(k+1)=g(xk)≈{xkp1:.8g}"
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
