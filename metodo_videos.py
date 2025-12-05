# ====== UTILIDADES DE ANIMACIÓN Y ESTILO ======
from matplotlib.animation import FuncAnimation
from matplotlib import rcParams

# Tipografía legible y render de fórmulas:
rcParams.update({
    "font.size": 11,
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
})


def _ease_in_out_cubic(t: float) -> float:
    """Easing suave para transiciones (0→1)."""
    return 4*t*t*t if t < 0.5 else 1 - pow(-2*t + 2, 3)/2


def _interpolar(a, b, t):
    return a + (b - a) * _ease_in_out_cubic(t)


def _construir_zoom_series(xi_hist, xd_hist, xs, fx, zoom_min_frac=0.18):
    """
    Devuelve límites animados (xlim, ylim) por frame.
    El zoom se calcula en función del tamaño del intervalo [xi, xd]:
    a intervalos pequeños, más zoom; a intervalos grandes, vista amplia.
    """
    import numpy as np
    fx_min, fx_max = float(np.min(fx)), float(np.max(fx))
    y_margin = 0.12 * max(1e-9, abs(fx_max - fx_min))
    ylo_base, yhi_base = fx_min - y_margin, fx_max + y_margin

    xlims = []
    ylims = []
    # Limites "globales" de partida (vista completa)
    xg0, xg1 = float(xs[0]), float(xs[-1])

    for xi, xd in zip(xi_hist, xd_hist):
        left, right = (min(xi, xd), max(xi, xd))
        span = max(1e-12, right - left)
        # Entre “span global” y “span de intervalo” generamos un objetivo a mezclar:
        base_span = max(1e-12, xg1 - xg0)
        # factor de zoom: si el intervalo es pequeño, acércate más (hasta ~zoom_min_frac del span global)
        target_span = max(zoom_min_frac * base_span, span * 1.6)
        cx = 0.5 * (left + right)
        xlims.append((cx - 0.5 * target_span, cx + 0.5 * target_span))
        ylims.append((ylo_base, yhi_base))

    # Suaviza los cambios de límites entre frames con easing:
    xlims_suaves, ylims_suaves = [xlims[0]], [ylims[0]]
    # frames intermedios virtuales para suavidad (no aumentan frames reales)
    steps_suavizado = 3
    for i in range(1, len(xlims)):
        x0, x1 = xlims_suaves[-1], xlims[i]
        y0, y1 = ylims_suaves[-1], ylims[i]
        # Solo guardamos el destino (el easing se aplica en tiempo real en update)
        xlims_suaves.append(x1)
        ylims_suaves.append(y1)
    return xlims_suaves, ylims_suaves


# ====== ANIMACIÓN: BISECCIÓN ======
def animar_biseccion(
    f, hist, xi_inicial, xd_inicial,
    titulo="Método de Bisección",
    guardar_gif=False, nombre_gif="biseccion.gif",
    fps=24, intervalo_ms=40
):
    import numpy as np
    import matplotlib.pyplot as plt

    # Historial → arreglos
    xa = np.array([h["xi"] for h in hist], dtype=float)
    ya = np.array([h["fxi"] for h in hist], dtype=float)
    xb = np.array([h["xd"] for h in hist], dtype=float)
    yb = np.array([h["fxd"] for h in hist], dtype=float)
    xc = np.array([h["xm"] for h in hist], dtype=float)
    yc = np.array([h["fxm"] for h in hist], dtype=float)

    a0, b0 = float(xi_inicial), float(xd_inicial)
    a, b = min(a0, b0), max(a0, b0)

    # Curva f(x)
    muestras = 600
    xs = np.linspace(a, b, muestras)
    fx = np.array([f(x) for x in xs], dtype=float)

    # Límites animados (zoom progresivo según [xi, xd])
    xlims, ylims = _construir_zoom_series(xa, xb, xs, fx)

    # ==== FIGURA ====
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    ax.set_title(titulo)
    ax.set_xlabel("x")
    ax.set_ylabel(r"$f(x)$")
    ax.grid(True, alpha=0.25)

    # Curva y eje
    (lineafx,) = ax.plot(xs, fx, label=r"$f(x)$")
    ax.axhline(0, linewidth=1)

    # Intervalo sombreado
    intervalo_poly = ax.fill_between(
        [xa[0], xb[0]], y1=ylims[0][0], y2=ylims[0][1], alpha=0.06, label="intervalo")

    # Puntos
    (puntoa,) = ax.plot([xa[0]], [ya[0]], "o", label=r"$x_i$")
    (puntob,) = ax.plot([xb[0]], [yb[0]], "o", label=r"$x_d$")
    (puntoc,) = ax.plot([xc[0]], [yc[0]], "o", label=r"$x_m$")

    # Guías verticales
    (lineaa,) = ax.plot([xa[0], xa[0]], [
        0, ya[0]], linestyle="--", linewidth=1.1)
    (lineab,) = ax.plot([xb[0], xb[0]], [
        0, yb[0]], linestyle="--", linewidth=1.1)
    (lineac,) = ax.plot([xc[0], xc[0]], [
        0, yc[0]], linestyle="--", linewidth=1.1)

    # Base en eje x
    (linea_ab,) = ax.plot([xa[0], xb[0]], [0, 0],
                          linestyle=":", linewidth=1.4, label="[xi, xd]")

    # Fórmulas (LaTeX/mathtext)
    ax.text(
        0.5, 1.02,
        r"$x_m=\frac{x_i+x_d}{2}\quad\;\;$"
        r"$\quad$Criterio: $|f(x_m)|\le\varepsilon$  ó  $\frac{|x_m-x_{m-1}|}{|x_m|}\le\varepsilon$",
        transform=ax.transAxes, ha="center", va="bottom"
    )

    # Caja de texto de iteración
    texto = ax.text(
        0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white",
                  alpha=0.92, edgecolor="0.85"),
    )

    ax.legend(loc="best")
    fig.tight_layout()

    # Para resaltar el frame final (criterio cumplido)
    idx_stop = len(xa) - 1

    def _set_intervalo(i, ylo, yhi):
        # Elimina el fill_between anterior y crea uno nuevo (para mantener etiqueta)
        for coll in ax.collections[:]:
            if getattr(coll, "get_label", lambda: "")() == "intervalo":
                coll.remove()
        ax.fill_between([xa[i], xb[i]], y1=ylo, y2=yhi,
                        alpha=0.06, label="intervalo")

    def init():
        ax.set_xlim(*xlims[0])
        ax.set_ylim(*ylims[0])
        return (puntoa, puntob, puntoc, lineaa, lineab, lineac, linea_ab, lineafx)

    def update(i):
        # Suaviza límites interpolando contra el objetivo del frame i
        x0, x1 = ax.get_xlim(), xlims[i]
        y0, y1 = ax.get_ylim(), ylims[i]
        t = 0.32  # cuánto se acerca por frame al destino (con easing)
        xl = (_interpolar(x0[0], x1[0], t), _interpolar(x0[1], x1[1], t))
        yl = (_interpolar(y0[0], y1[0], t), _interpolar(y0[1], y1[1], t))
        ax.set_xlim(*xl)
        ax.set_ylim(*yl)

        # Actualiza elementos
        puntoa.set_data([xa[i]], [ya[i]])
        puntob.set_data([xb[i]], [yb[i]])
        puntoc.set_data([xc[i]], [yc[i]])

        lineaa.set_data([xa[i], xa[i]], [0, ya[i]])
        lineab.set_data([xb[i], xb[i]], [0, yb[i]])
        lineac.set_data([xc[i], xc[i]], [0, yc[i]])
        linea_ab.set_data([xa[i], xb[i]], [0, 0])

        _set_intervalo(i, yl[0], yl[1])

        texto.set_text(
            f"Iteración: {i}\n"
            f"xi={xa[i]:.8g}, f(xi)={ya[i]:.8g}\n"
            f"xd={xb[i]:.8g}, f(xd)={yb[i]:.8g}\n"
            f"xm={xc[i]:.8g}, f(xm)={yc[i]:.8g}"
        )

        # Pulso sutil cuando se cumple el criterio (último frame del historial)
        if i == idx_stop:
            ms = 220  # tamaño de marcador para destacar
            puntoc.set_markersize(9.5)
            puntoc.set_markeredgewidth(1.4)
            ax.annotate(
                "¡Criterio cumplido!",
                xy=(xc[i], yc[i]), xycoords="data",
                xytext=(0.03, 0.15), textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", linewidth=1.2),
                bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.95)
            )
        return (puntoa, puntob, puntoc, lineaa, lineab, lineac, linea_ab, lineafx)

    ani = FuncAnimation(
        fig, update, frames=len(xa), init_func=init,
        interval=intervalo_ms, blit=False, repeat=False
    )

    if guardar_gif:
        try:
            from matplotlib.animation import PillowWriter
            ani.save(nombre_gif, writer=PillowWriter(fps=fps))
            print(f"GIF guardado como: {nombre_gif}")
        except Exception as e:
            print(f"No se pudo guardar el GIF: {e}\nMostrando animación...")
            plt.show()
    else:
        plt.show()


def animar_posicion_falsa(
    f, hist, xi_inicial, xd_inicial,
    titulo="Método de Posición Falsa",
    guardar_gif=False, nombre_gif="posicion_falsa.gif",
    fps=24, intervalo_ms=40
):
    import numpy as np
    import matplotlib.pyplot as plt

    xa = np.array([h["xi"] for h in hist], dtype=float)
    ya = np.array([h["fxi"] for h in hist], dtype=float)
    xb = np.array([h["xd"] for h in hist], dtype=float)
    yb = np.array([h["fxd"] for h in hist], dtype=float)
    # xr ≡ punto de intersección (regula falsi)
    xr = np.array([h["xm"] for h in hist], dtype=float)
    yr = np.array([h["fxm"] for h in hist], dtype=float)

    a0, b0 = float(xi_inicial), float(xd_inicial)
    a, b = min(a0, b0), max(a0, b0)

    muestras = 600
    xs = np.linspace(a, b, muestras)
    fx = np.array([f(x) for x in xs], dtype=float)

    xlims, ylims = _construir_zoom_series(xa, xb, xs, fx)

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    ax.set_title(titulo)
    ax.set_xlabel("x")
    ax.set_ylabel(r"$f(x)$")
    ax.grid(True, alpha=0.25)

    (lineafx,) = ax.plot(xs, fx, label=r"$f(x)$")
    ax.axhline(0, linewidth=1)

    # Puntos extremos y xr
    (puntoa,) = ax.plot([xa[0]], [ya[0]], "o", label=r"$x_i$")
    (puntob,) = ax.plot([xb[0]], [yb[0]], "o", label=r"$x_d$")
    (puntor,) = ax.plot([xr[0]], [yr[0]], "o", label=r"$x_r$")

    # Secante y proyección vertical de xr
    (lineaab,) = ax.plot([xa[0], xb[0]], [
        ya[0], yb[0]], linewidth=1.4, label="secante")
    (linear0,) = ax.plot([xr[0], xr[0]], [
        0, yr[0]], linestyle="--", linewidth=1.1)

    # Base y sombra del intervalo
    (linea_base,) = ax.plot([xa[0], xb[0]], [0, 0],
                            linestyle=":", linewidth=1.4, label="[xi, xd]")
    intervalo_poly = ax.fill_between(
        [xa[0], xb[0]], y1=ylims[0][0], y2=ylims[0][1], alpha=0.05, label="intervalo")

    ax.text(
        0.5, 1.02,
        r"$x_r=x_i-\dfrac{f(x_i)(x_d-x_i)}{f(x_d)-f(x_i)}\quad$"
        r"$\;$Criterio: $|f(x_r)|\le\varepsilon$  ó  $\frac{|x_r-x_{r-1}|}{|x_r|}\le\varepsilon$",
        transform=ax.transAxes, ha="center", va="bottom"
    )

    texto = ax.text(
        0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white",
                  alpha=0.92, edgecolor="0.85"),
    )
    ax.legend(loc="best")
    fig.tight_layout()

    idx_stop = len(xa) - 1

    def _set_intervalo(i, ylo, yhi):
        for coll in ax.collections[:]:
            if getattr(coll, "get_label", lambda: "")() == "intervalo":
                coll.remove()
        ax.fill_between([xa[i], xb[i]], y1=ylo, y2=yhi,
                        alpha=0.05, label="intervalo")

    def init():
        ax.set_xlim(*xlims[0])
        ax.set_ylim(*ylims[0])
        return (puntoa, puntob, puntor, lineaab, linear0, linea_base, lineafx)

    def update(i):
        # Easing en límites
        x0, x1 = ax.get_xlim(), xlims[i]
        y0, y1 = ax.get_ylim(), ylims[i]
        t = 0.32
        xl = (_interpolar(x0[0], x1[0], t), _interpolar(x0[1], x1[1], t))
        yl = (_interpolar(y0[0], y1[0], t), _interpolar(y0[1], y1[1], t))
        ax.set_xlim(*xl)
        ax.set_ylim(*yl)

        # Actualiza elementos
        puntoa.set_data([xa[i]], [ya[i]])
        puntob.set_data([xb[i]], [yb[i]])
        puntor.set_data([xr[i]], [yr[i]])

        lineaab.set_data([xa[i], xb[i]], [ya[i], yb[i]])
        linear0.set_data([xr[i], xr[i]], [0, yr[i]])

        linea_base.set_data([xa[i], xb[i]], [0, 0])
        _set_intervalo(i, yl[0], yl[1])

        texto.set_text(
            f"Iteración: {i}\n"
            f"xi={xa[i]:.8g}, f(xi)={ya[i]:.8g}\n"
            f"xd={xb[i]:.8g}, f(xd)={yb[i]:.8g}\n"
            f"xr={xr[i]:.8g}, f(xr)={yr[i]:.8g}"
        )

        if i == idx_stop:
            puntor.set_markersize(9.5)
            puntor.set_markeredgewidth(1.4)
            ax.annotate(
                "¡Criterio cumplido!",
                xy=(xr[i], yr[i]), xycoords="data",
                xytext=(0.03, 0.16), textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", linewidth=1.2),
                bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.95)
            )
        return (puntoa, puntob, puntor, lineaab, linear0, linea_base, lineafx)

    ani = FuncAnimation(
        fig, update, frames=len(xa), init_func=init,
        interval=intervalo_ms, blit=False, repeat=False
    )

    if guardar_gif:
        try:
            from matplotlib.animation import PillowWriter
            ani.save(nombre_gif, writer=PillowWriter(fps=fps))
            print(f"GIF guardado como: {nombre_gif}")
        except Exception as e:
            print(f"No se pudo guardar el GIF: {e}\nMostrando animación...")
            plt.show()
    else:
        plt.show()
