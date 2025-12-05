import math
import os
from io import BytesIO
from PIL import Image
import tempfile
import shutil
from pdf2image import convert_from_path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from pylatex import Document, NoEscape, Package


def _latex_doc_from_lines(lines):
    """
    Documento LaTeX sin márgenes (standalone) + graphicx para \resizebox.
    Recorta al contenido y deja matemáticas en \displaystyle.
    """
    doc = Document(
        documentclass="standalone",
        document_options=["border=0pt"],   # sin borde extra
    )
    doc.packages.append(Package("amsmath"))
    doc.packages.append(Package("amssymb"))
    doc.packages.append(Package("lmodern"))
    doc.packages.append(Package("graphicx"))   # <-- para \resizebox

    doc.preamble.append(NoEscape(r"\pagenumbering{gobble}"))
    doc.preamble.append(NoEscape(r"\everymath{\displaystyle}"))

    # pegamos 1 bloque grande (cada elemento de 'lines' ya será un bloque)
    for ln in lines:
        doc.append(NoEscape(ln))
        doc.append("\n")
    return doc


def _find_latex_compiler():
    for exe in ("latexmk", "pdflatex", "xelatex", "lualatex"):
        if shutil.which(exe):
            return exe
    return None


def _compile_latex_to_png(lines, out_dir, basename):
    os.makedirs(out_dir, exist_ok=True)
    compiler = _find_latex_compiler()
    if compiler is None:
        return _render_mathtext_png(lines, out_dir, basename, dpi=300, fontsize=28)

    pdf_base = os.path.join(out_dir, basename)
    doc = _latex_doc_from_lines(lines)
    try:
        if compiler == "latexmk":
            doc.generate_pdf(filepath=pdf_base, clean_tex=True, silent=True)
        else:
            doc.generate_pdf(filepath=pdf_base, clean_tex=True,
                             silent=True, compiler=compiler)
    except Exception:
        return _render_mathtext_png(lines, out_dir, basename, dpi=300, fontsize=28)

    try:
        images = convert_from_path(pdf_base + ".pdf", dpi=300)
        png_path = os.path.join(out_dir, basename + ".png")
        images[0].save(png_path)
        return png_path
    except Exception:
        return _render_mathtext_png(lines, out_dir, basename, dpi=300, fontsize=28)


def _render_mathtext_png(lines, out_dir, basename, dpi=300, fontsize=26):
    """
    Fallback sin LaTeX real: usa el motor mathtext de matplotlib.
    - Fuente GRANDE (fontsize).
    - Figura generosa para que el PNG ocupe pantalla.
    - Reemplaza $$...$$ por $...$ (mejor para mathtext).
    """
    os.makedirs(out_dir, exist_ok=True)

    # Lienzo grande; aumentamos tamaño para que escale bien
    fig = plt.figure(figsize=(8.0, 4.5), dpi=dpi)   # ~ 2400x1350 px a 300 dpi
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.axis("off")

    safe_lines = [ln.replace("$$", "$") for ln in lines]
    text = "\n".join(safe_lines)

    ax.text(0.02, 0.98, text, va="top", ha="left",
            fontsize=fontsize)  # <--- fuente grande

    buf = BytesIO()
    fig.savefig(buf, format="png", transparent=True,
                bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    buf.seek(0)
    im = Image.open(buf).convert("RGBA")
    png_path = os.path.join(out_dir, basename + ".png")
    im.save(png_path)
    return png_path


def _fmt_float(v, digits=8):
    try:
        return f"{float(v):.{digits}g}"
    except Exception:
        return str(v)


def _make_latex_panel_biseccion(expr, h, digits=8):
    def ff(v): return f"{float(v):.{digits}g}"
    xi, xd, xm = ff(h["xi"]), ff(h["xd"]), ff(h["xm"])
    fxi, fxd, fxm = ff(h["fxi"]), ff(h["fxd"]), ff(h["fxm"])
    expr_tex = expr.replace("*", r"\cdot ")

    block = rf"""
        \resizebox{{0.96\linewidth}}{{!}}{{%
        \begin{{minipage}}{{\linewidth}}
        \centering
        \textbf{{\Large Bisección}} \quad (\textit{{iteración {h['i']}}})\\[0.8em]
        $$ f(x) = {expr_tex} $$

        % Fórmula
        $$ x_m = \frac{{x_i + x_d}}{{2}} $$

        % Sustitución
        $$ x_m = \frac{{{xi} + {xd}}}{{2}} \;\approx\; {xm} $$

        % Evaluación
        $$ f(x_m) \;\approx\; {fxm}
        \quad\Big( f(x_i)={fxi},\; f(x_d)={fxd} \Big) $$
        \end{{minipage}}
        }}%
    """.strip()
    return [block]


def _make_latex_panel_posfalsa(expr, h, digits=8):
    def ff(v): return f"{float(v):.{digits}g}"
    xi, xd, xr = ff(h["xi"]), ff(h["xd"]), ff(h["xm"])
    fxi, fxd, fxr = ff(h["fxi"]), ff(h["fxd"]), ff(h["fxm"])
    expr_tex = expr.replace("*", r"\cdot ")

    block = rf"""
        \resizebox{{0.96\linewidth}}{{!}}{{%
        \begin{{minipage}}{{\linewidth}}
        \centering
        \textbf{{\Large Posición Falsa}} \quad (\textit{{iteración {h['i']}}})\\[0.8em]
        $$ f(x) = {expr_tex} $$

        % Fórmula
        $$ x_r = x_i \;-\; f(x_i)\,\frac{{x_d - x_i}}{{\,f(x_d) - f(x_i)\,}} $$

        % Sustitución
        $$ x_r = {xi} \;-\; ({fxi})\,\frac{{{xd}-{xi}}}{{\,{fxd}-{fxi}\,}}
        \;\approx\; {xr} $$

        % Evaluación
        $$ f(x_r) \;\approx\; {fxr} $$
        \end{{minipage}}
        }}%
    """.strip()
    return [block]


def _precompute_latex_panels(expr, hist, metodo_nombre):
    """
    Precompila un PNG por iteración con el panel LaTeX.
    Devuelve lista de rutas PNG (una por iteración).
    """
    tmp_dir = tempfile.mkdtemp(prefix="latex_panels_")
    pngs = []
    for h in hist:
        if metodo_nombre == "biseccion":
            lines = _make_latex_panel_biseccion(expr, h)
        else:
            lines = _make_latex_panel_posfalsa(expr, h)
        fname = f"{metodo_nombre}_iter_{h['i']:03d}"
        png_path = _compile_latex_to_png(lines, tmp_dir, fname)
        pngs.append(png_path)
    return pngs


def make_func(expr: str):
    """
    Convierte un string como 'x**3 - 2*x - 5' en f(x).
    """
    allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    allowed.update({"x": 0.0})
    code = compile(expr, "<user-f>", "eval")

    def f(x):
        allowed["x"] = x
        return eval(code, {"__builtins__": {}}, allowed)
    return f


def criterio_cumplido(criterio, tol, fxm, xm, xm_prev):
    if criterio == "abs_fx":
        return abs(fxm) <= tol
    if xm_prev is None:
        return False
    diff = abs(xm - xm_prev)
    if criterio == "abs":
        return diff <= tol
    if criterio == "rel":
        return (diff / (abs(xm) if xm != 0 else 1.0)) <= tol
    if criterio == "pct":
        return (diff / (abs(xm) if xm != 0 else 1.0)) * 100.0 <= tol
    raise ValueError("Criterio no reconocido")


def biseccion(f, xi, xd, tol=1e-3, criterio="abs_fx", max_iter=100):
    if xi == xd:
        raise ValueError("xi y xd no deben ser iguales.")
    if xi > xd:
        xi, xd = xd, xi
    fxi, fxd = f(xi), f(xd)
    if fxi == 0:
        return xi, 0, [{"i": 0, "xi": xi, "fxi": fxi, "xd": xd, "fxd": fxd, "xm": xi, "fxm": 0.0}]
    if fxd == 0:
        return xd, 0, [{"i": 0, "xi": xi, "fxi": fxi, "xd": xd, "fxd": fxd, "xm": xd, "fxm": 0.0}]

    historial = []
    xm_prev = None
    for i in range(0, max_iter):
        xm = (xi + xd) / 2.0
        fxm = f(xm)

        historial.append({
            "i": i, "xi": xi, "fxi": fxi, "xd": xd, "fxd": fxd, "xm": xm, "fxm": fxm
        })

        if criterio_cumplido(criterio, tol, fxm, xm, xm_prev):
            return xm, i, historial

        if fxm * fxi > 0:
            xi, fxi = xm, fxm
        else:
            xd, fxd = xm, fxm

        xm_prev = xm

    return xm, i, historial


def posicion_falsa(f, xi, xd, tol=1e-3, criterio="abs_fx", max_iter=100, usar_illinois=False):
    """
    Método de Posición Falsa (Regula Falsi) con variante Illinois opcional.
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

    historial = []
    xm_prev = None
    lado_fijo = None  # para Illinois

    for i in range(0, max_iter):
        denom = (fxd - fxi)
        if denom == 0:
            xm = (xi + xd) / 2.0
        else:
            xm = xi - fxi * (xd - xi) / denom

        fxm = f(xm)

        historial.append({
            "i": i, "xi": xi, "fxi": fxi, "xd": xd, "fxd": fxd, "xm": xm, "fxm": fxm
        })

        if criterio_cumplido(criterio, tol, fxm, xm, xm_prev):
            return xm, i, historial

        if fxm * fxi > 0:
            xi, fxi = xm, fxm
            if usar_illinois:
                if lado_fijo == "xi":
                    fxd *= 0.5
                lado_fijo = "xi"
        else:
            xd, fxd = xm, fxm
            if usar_illinois:
                if lado_fijo == "xd":
                    fxi *= 0.5
                lado_fijo = "xd"

        xm_prev = xm

    return xm, i, historial


def _make_interleaved_frames(n_iters: int, panel_hold: int = 1):
    """
    Genera frames intercalados:
      ('plot', i)  -> muestra el gráfico en la iteración i
      ('panel', i) -> muestra el panel LaTeX de la iteración i (panel_hold veces)
    """
    frames = []
    for i in range(n_iters):
        frames.append(("plot", i))
        for _ in range(panel_hold):
            frames.append(("panel", i))
    return frames


def animar_biseccion(
    f, hist, xi_inicial, xd_inicial,
    titulo="Método de Bisección",
    guardar_gif=False, nombre_gif="biseccion.gif",
    fps=2, intervalo_ms=700,
    expr_text="f(x)",
    panel_hold=1,
):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from matplotlib.animation import FuncAnimation

    # Pre-render de paneles LaTeX (uno por iteración)
    latex_pngs = _precompute_latex_panels(
        expr_text, hist, metodo_nombre="biseccion")

    # ---- Datos del historial ----
    xa = np.array([h["xi"] for h in hist], dtype=float)
    ya = np.array([h["fxi"] for h in hist], dtype=float)
    xb = np.array([h["xd"] for h in hist], dtype=float)
    yb = np.array([h["fxd"] for h in hist], dtype=float)
    xc = np.array([h["xm"] for h in hist], dtype=float)
    yc = np.array([h["fxm"] for h in hist], dtype=float)

    a0, b0 = float(xi_inicial), float(xd_inicial)
    a, b = min(a0, b0), max(a0, b0)

    # Curva f(x)
    muestras = 401
    xs = np.linspace(a, b, muestras)
    fx = np.array([f(x) for x in xs])
    ymax, ymin = np.max(fx), np.min(fx)
    deltax = abs(b - a) if b != a else 1.0
    deltay = abs(ymax - ymin) if ymax != ymin else 1.0
    xlo, xhi = a - 0.08 * deltax, b + 0.08 * deltax
    ylo, yhi = ymin - 0.12 * deltay, ymax + 0.12 * deltay

    # ---- Figura y ejes ----
    fig = plt.figure(figsize=(10, 5.5))
    # Eje del gráfico
    ax = fig.add_axes([0.08, 0.10, 0.84, 0.80])
    ax.set_xlim([xlo, xhi])
    ax.set_ylim([ylo, yhi])
    ax.set_title(titulo)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(True, alpha=0.25)
    (lineafx,) = ax.plot(xs, fx, label="f(x)")
    ax.axhline(0, color="k", linewidth=1)
    # Elementos del método
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
        0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white",
                  alpha=0.9, edgecolor="0.85")
    )
    ax.legend(loc="best")

    # Eje del panel (pantalla casi completa)
    ax_panel = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax_panel.axis("off")
    im_panel = ax_panel.imshow(mpimg.imread(latex_pngs[0]), aspect="auto")
    ax_panel.set_visible(False)  # arrancamos mostrando el gráfico

    # Frames intercalados
    frames = _make_interleaved_frames(len(hist), panel_hold=panel_hold)

    def _update_plot(i):
        # Actualiza elementos del método para la iteración i
        puntoa.set_data([xa[i]], [ya[i]])
        puntob.set_data([xb[i]], [yb[i]])
        puntoc.set_data([xc[i]], [yc[i]])
        lineaa.set_data([xa[i], xa[i]], [0, ya[i]])
        lineab.set_data([xb[i], xb[i]], [0, yb[i]])
        lineac.set_data([xc[i], xc[i]], [0, yc[i]])
        linea_ab.set_data([xa[i], xb[i]], [0, 0])
        # Refresca el sombreado del intervalo
        for coll in list(ax.collections):
            if hasattr(coll, "get_alpha") and coll.get_alpha() == 0.06:
                try:
                    coll.remove()
                except Exception:
                    pass
        ax.fill_between([xa[i], xb[i]], y1=ylo, y2=yhi,
                        color="tab:blue", alpha=0.06)
        # Texto
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
        else:  # 'panel'
            if not ax_panel.get_visible():
                ax_panel.set_visible(True)
            if ax.get_visible():
                ax.set_visible(False)
            im_panel.set_data(mpimg.imread(latex_pngs[i]))
            return [im_panel]

    ani = FuncAnimation(
        fig, update, frames=frames, init_func=init,
        interval=intervalo_ms, blit=False, repeat=False,
    )

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


def animar_posicion_falsa(
    f, hist, xi_inicial, xd_inicial,
    titulo="Método de Posición Falsa",
    guardar_gif=False, nombre_gif="posicion_falsa.gif",
    fps=2, intervalo_ms=700,
    expr_text="f(x)",
    panel_hold=1,
):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from matplotlib.animation import FuncAnimation

    # Pre-render de paneles LaTeX (uno por iteración)
    latex_pngs = _precompute_latex_panels(
        expr_text, hist, metodo_nombre="posfalsa")

    # ---- Datos del historial ----
    xa = np.array([h["xi"] for h in hist], dtype=float)
    ya = np.array([h["fxi"] for h in hist], dtype=float)
    xb = np.array([h["xd"] for h in hist], dtype=float)
    yb = np.array([h["fxd"] for h in hist], dtype=float)
    xr = np.array([h["xm"] for h in hist], dtype=float)
    yr = np.array([h["fxm"] for h in hist], dtype=float)

    a0, b0 = float(xi_inicial), float(xd_inicial)
    a, b = min(a0, b0), max(a0, b0)

    # Curva f(x)
    muestras = 401
    xs = np.linspace(a, b, muestras)
    fx = np.array([f(x) for x in xs])
    ymax, ymin = np.max(fx), np.min(fx)
    deltax = abs(b - a) if b != a else 1.0
    deltay = abs(ymax - ymin) if ymax != ymin else 1.0
    xlo, xhi = a - 0.08 * deltax, b + 0.08 * deltax
    ylo, yhi = ymin - 0.12 * deltay, ymax + 0.12 * deltay

    # ---- Figura y ejes ----
    fig = plt.figure(figsize=(10, 5.5))
    # Eje del gráfico
    ax = fig.add_axes([0.08, 0.10, 0.84, 0.80])
    ax.set_xlim([xlo, xhi])
    ax.set_ylim([ylo, yhi])
    ax.set_title(titulo)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(True, alpha=0.25)
    (lineafx,) = ax.plot(xs, fx, label="f(x)")
    ax.axhline(0, color="k", linewidth=1)

    # Elementos del método
    (puntoa,) = ax.plot([xa[0]], [ya[0]], "o", color="red", label="xi")
    (puntob,) = ax.plot([xb[0]], [yb[0]], "o", color="green", label="xf")
    (puntoc,) = ax.plot([xr[0]], [yr[0]], "o", color="orange", label="xr")
    (lineaab,) = ax.plot([xa[0], xb[0]], [ya[0], yb[0]],
                         color="orange", linewidth=1.6, label="secante")
    (lineac0,) = ax.plot([xr[0], xr[0]], [0, yr[0]],
                         color="magenta", linestyle="--", linewidth=1.2)
    (linea_base,) = ax.plot([xa[0], xb[0]], [
        0, 0], linestyle=":", color="gold", linewidth=1.4, label="[xi,xf]")
    intervalo = ax.fill_between(
        [xa[0], xb[0]], y1=ylo, y2=yhi, color="tab:purple", alpha=0.05, label="intervalo")
    texto = ax.text(
        0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white",
                  alpha=0.9, edgecolor="0.85"),
    )
    ax.legend(loc="best")

    # Eje del panel (pantalla casi completa)
    ax_panel = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax_panel.axis("off")
    im_panel = ax_panel.imshow(mpimg.imread(latex_pngs[0]), aspect="auto")
    ax_panel.set_visible(False)

    # Frames intercalados
    frames = _make_interleaved_frames(len(hist), panel_hold=panel_hold)

    def _update_plot(i):
        puntoa.set_data([xa[i]], [ya[i]])
        puntob.set_data([xb[i]], [yb[i]])
        puntoc.set_data([xr[i]], [yr[i]])
        lineaab.set_data([xa[i], xb[i]], [ya[i], yb[i]])
        lineac0.set_data([xr[i], xr[i]], [0, yr[i]])
        linea_base.set_data([xa[i], xb[i]], [0, 0])

        # Refresca el sombreado del intervalo
        for coll in list(ax.collections):
            if hasattr(coll, "get_alpha") and coll.get_alpha() == 0.05:
                try:
                    coll.remove()
                except Exception:
                    pass
        ax.fill_between([xa[i], xb[i]], y1=ylo, y2=yhi,
                        color="tab:purple", alpha=0.05)

        texto.set_text(
            f"Iteración: {i}\n"
            f"xi={xa[i]:.8g}, f(xi)={ya[i]:.8g}\n"
            f"xf={xb[i]:.8g}, f(xf)={yb[i]:.8g}\n"
            f"xr={xr[i]:.8g}, f(xr)={yr[i]:.8g}"
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
        else:  # 'panel'
            if not ax_panel.get_visible():
                ax_panel.set_visible(True)
            if ax.get_visible():
                ax.set_visible(False)
            im_panel.set_data(mpimg.imread(latex_pngs[i]))
            return [im_panel]

    ani = FuncAnimation(
        fig, update, frames=frames, init_func=init,
        interval=intervalo_ms, blit=False, repeat=False,
    )

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


if __name__ == "__main__":
    print("=== Métodos de Raíces (Bisección y Posición Falsa) ===")
    expr = input("Ingresa f(x), p.ej.: x**3 - 2*x - 5\nf(x) = ").strip()
    f = make_func(expr)

    xi = float(input("Ingresa xi: "))
    xd = float(input("Ingresa xd: "))

    metodo = input(
        "Elige método: (b)isección / (p)osición falsa: ").strip().lower()
    if metodo not in {"b", "p"}:
        raise ValueError("Opción inválida.")

    print("\nCriterios de paro disponibles:")
    print("- abs_fx : |f(xm)| <= tol")
    print("- abs    : |xm - xm_prev| <= tol")
    print("- rel    : |xm - xm_prev| / |xm| <= tol")
    print("- pct    : (|xm - xm_prev| / |xm|)*100 <= tol")
    criterio = input("Elige criterio (abs_fx/abs/rel/pct): ").strip().lower()
    if criterio not in {"abs_fx", "abs", "rel", "pct"}:
        raise ValueError("Criterio no válido.")

    tol = float(
        input("Ingresa tolerancia (ej. 0.001). Si elegiste 'pct', la tol es en %: "))

    try:
        if metodo == "b":
            raiz, iters, hist = biseccion(
                f, xi, xd, tol=tol, criterio=criterio, max_iter=200)
            titulo_anim = f"Método de Bisección | f(x) = {expr}"
            anim_fn = animar_biseccion
            nombre_gif = "biseccion.gif"
        else:
            usar_ill = input(
                "¿Usar variante Illinois para Posición Falsa? (s/n): ").strip().lower() == "s"
            raiz, iters, hist = posicion_falsa(
                f, xi, xd, tol=tol, criterio=criterio, max_iter=200, usar_illinois=usar_ill)
            titulo_anim = f"Método de Posición Falsa{' (Illinois)' if usar_ill else ''} | f(x) = {expr}"
            anim_fn = animar_posicion_falsa
            nombre_gif = "posicion_falsa.gif"
    except Exception as e:
        print(f"\nError: {e}")
    else:
        print("\n--- Iteraciones ---")
        for h in hist:
            i = h["i"]
            print(
                f"Iteracion#{i:>3} | "
                f"Xi={h['xi']:.7f},f(Xi)={h['fxi']:.7f} | "
                f"Xd={h['xd']:.7f},f(Xd)={h['fxd']:.7f} | "
                f"Xm={h['xm']:.7f},f(Xm)={h['fxm']:.7f}"
            )
        print("\n=== Resultado ===")
        print(f"Raíz aproximada: {raiz:.10f}")
        print(f"Iteraciones usadas: {iters}")
        print(f"f(raíz) ≈ {f(raiz):.10g}")
        print(f"Criterio: {criterio} con tolerancia {tol}")

        resp = input("\n¿Deseas ver la animación? (s/n): ").strip().lower()
        if resp == "s":
            guardar = input("¿Guardar GIF? (s/n): ").strip().lower()
            if guardar == "s":
                nombre = input(
                    f"Nombre de archivo GIF (ej. {nombre_gif}): ").strip() or nombre_gif
                anim_fn(
                    f, hist,
                    xi_inicial=hist[0]["xi"], xd_inicial=hist[0]["xd"],
                    titulo=titulo_anim, guardar_gif=True, nombre_gif=nombre,
                    expr_text=expr
                )
            else:
                anim_fn(
                    f, hist,
                    xi_inicial=hist[0]["xi"], xd_inicial=hist[0]["xd"],
                    titulo=titulo_anim, guardar_gif=False, expr_text=expr
                )
