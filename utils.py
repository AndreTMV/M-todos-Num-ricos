from __future__ import annotations

import math
import os
import shutil
import tempfile
from io import BytesIO
from typing import Callable, Dict, Iterable, List, Literal, TypedDict

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Dependencias opcionales (manejo seguro)
try:
    from pdf2image import convert_from_path  # type: ignore
except Exception:  # pragma: no cover
    convert_from_path = None  # type: ignore

try:
    from pylatex import Document, NoEscape, Package  # type: ignore
except Exception:  # pragma: no cover
    Document = NoEscape = Package = None  # type: ignore


# ------------------------------- Tipos ---------------------------------------

class Iteracion(TypedDict, total=False):
    i: int
    xi: float
    fxi: float
    xd: float
    fxd: float
    # xm para bisección y xr para posición falsa (nomenclatura común)
    xm: float
    fxm: float
    # Campos para integración
    area: float
    n_seg: int
    error: float


MetodoNombre = Literal["biseccion", "posfalsa",
                       "secante", "newton", "puntofijo", "trapecio"]


# ----------------------- Utilidades de evaluación ----------------------------

def make_func(expr: str) -> Callable[[float], float]:
    """
    Convierte un string como 'x**3 - 2*x - 5' en f(x).
    Expone 'x' y funciones de math en un entorno seguro.
    """
    allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    allowed.update({"x": 0.0})
    code = compile(expr, "<user-f>", "eval")

    def f(x: float) -> float:
        allowed["x"] = x
        return float(eval(code, {"__builtins__": {}}, allowed))

    return f


def criterio_cumplido(
    criterio: Literal["abs_fx", "abs", "rel", "pct"],
    tol: float,
    fxm: float,
    xm: float,
    xm_prev: float | None,
) -> bool:
    """
    Verifica si se cumple el criterio de paro.
    - abs_fx : |f(xm)| <= tol
    - abs    : |xm - xm_prev| <= tol
    - rel    : |xm - xm_prev| / |xm| <= tol
    - pct    : (|xm - xm_prev| / |xm|) * 100 <= tol
    """
    if criterio == "abs_fx":
        # Para integración, fxm podría interpretarse como el error relativo o absoluto si se pasa
        return abs(fxm) <= tol
    if xm_prev is None:
        return False
    diff = abs(xm - xm_prev)
    if criterio == "abs":
        return diff <= tol
    denom = abs(xm) if xm != 0 else 1.0
    if criterio == "rel":
        return (diff / denom) <= tol
    if criterio == "pct":
        return (diff / denom) * 100.0 <= tol
    raise ValueError("Criterio no reconocido")


# --------------------- Formateo y textos LaTeX / mathtext --------------------

def _fmt_float(v: float, digits: int = 8) -> str:
    try:
        return f"{float(v):.{digits}g}"
    except Exception:
        return str(v)


def _latex_doc_from_lines(lines: Iterable[str]):
    """Crea documento LaTeX 'standalone' con márgenes 0 y math display."""
    if Document is None:  # Fallback si no está pylatex
        return None
    doc = Document(documentclass="standalone", document_options=["border=0pt"])
    doc.packages.append(Package("amsmath"))
    doc.packages.append(Package("amssymb"))
    doc.packages.append(Package("lmodern"))
    doc.packages.append(Package("graphicx"))  # \resizebox
    doc.preamble.append(NoEscape(r"\pagenumbering{gobble}"))
    doc.preamble.append(NoEscape(r"\everymath{\displaystyle}"))
    for ln in lines:
        doc.append(NoEscape(ln))
        doc.append("\n")
    return doc


def _find_latex_compiler() -> str | None:
    for exe in ("latexmk", "pdflatex", "xelatex", "lualatex"):
        if shutil.which(exe):
            return exe
    return None


def _render_mathtext_png(
    lines: Iterable[str],
    out_dir: str,
    basename: str,
    dpi: int = 300,
    fontsize: int = 26,
) -> str:
    """
    Fallback sin LaTeX: usa mathtext (matplotlib) con lienzo amplio.
    """
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=(8.0, 4.5), dpi=dpi)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.axis("off")
    safe_lines = [ln.replace("$$", "$") for ln in lines]
    text = "\n".join(safe_lines)
    ax.text(0.02, 0.98, text, va="top", ha="left", fontsize=fontsize)
    buf = BytesIO()
    fig.savefig(buf, format="png", transparent=True,
                bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    buf.seek(0)
    png_path = os.path.join(out_dir, basename + ".png")
    # Intentar cargar para validar, aunque savefig ya escribió
    Image.open(buf).convert("RGBA").save(png_path)
    return png_path


def _compile_latex_to_png(lines: Iterable[str], out_dir: str, basename: str) -> str:
    """
    Compila a PDF+PNG si hay LaTeX; si no, hace fallback a mathtext.
    """
    compiler = _find_latex_compiler()
    have_pdf2image = convert_from_path is not None
    have_pylatex = Document is not None

    if not (compiler and have_pdf2image and have_pylatex):
        return _render_mathtext_png(lines, out_dir, basename, dpi=300, fontsize=28)

    os.makedirs(out_dir, exist_ok=True)
    pdf_base = os.path.join(out_dir, basename)
    doc = _latex_doc_from_lines(lines)
    try:
        if compiler == "latexmk":
            doc.generate_pdf(filepath=pdf_base, clean_tex=True, silent=True)
        else:
            doc.generate_pdf(filepath=pdf_base, clean_tex=True,
                             compiler=compiler, silent=True)
    except Exception:
        return _render_mathtext_png(lines, out_dir, basename, dpi=300, fontsize=28)

    try:
        images = convert_from_path(pdf_base + ".pdf", dpi=300)  # type: ignore
        png_path = os.path.join(out_dir, basename + ".png")
        images[0].save(png_path)
        return png_path
    except Exception:
        return _render_mathtext_png(lines, out_dir, basename, dpi=300, fontsize=28)


def make_latex_panel(expr: str, h: Iteracion, metodo: MetodoNombre, digits: int = 8) -> List[str]:
    """
    Genera el bloque LaTeX (como líneas) para una iteración, según el método.
    """
    def ff(v): return f"{float(v):.{digits}g}"
    
    # Valores comunes (pueden ser None/missing en algunos métodos)
    xi = ff(h.get("xi", 0))
    xd = ff(h.get("xd", 0))
    xm = ff(h.get("xm", 0))
    fxi = ff(h.get("fxi", 0))
    fxd = ff(h.get("fxd", 0))
    fxm = ff(h.get("fxm", 0))
    
    expr_tex = expr.replace("*", r"\cdot ")

    if metodo == "biseccion":
        block = rf"""
            \resizebox{{0.96\linewidth}}{{!}}{{%
            \begin{{minipage}}{{\linewidth}}
            \centering
            \textbf{{\Large Bisección}} \quad (\textit{{iteración {h['i']}}})\\[0.8em]
            $$ f(x) = {expr_tex} $$
            $$ x_m = \frac{{x_i + x_d}}{{2}} $$
            $$ x_m = \frac{{{xi} + {xd}}}{{2}} \;\approx\; {xm} $$
            $$ f(x_m) \;\approx\; {fxm}
              \quad \Big( f(x_i) = {fxi},\; f(x_d) = {fxd} \Big) $$
            \end{{minipage}}
            }}%
        """.strip()
    elif metodo == "posfalsa":  # posfalsa
        block = rf"""
            \resizebox{{0.96\linewidth}}{{!}}{{%
            \begin{{minipage}}{{\linewidth}}
            \centering
            \textbf{{\Large Posición Falsa}} \quad (\textit{{iteración {h['i']}}})\\[0.8em]
            $$ f(x) = {expr_tex} $$
            $$ x_r = x_i \;-\; f(x_i)\,\frac{{x_d - x_i}}{{\,f(x_d) - f(x_i)\,}} $$
            $$ x_r = {xi} \;-\; ({fxi})\,\frac{{{xd}-{xi}}}{{\,{fxd}-{fxi}\,}}
              \;\approx\; {xm} $$
            $$ f(x_r) \;\approx\; {fxm} $$
            \end{{minipage}}
            }}%
        """.strip()
    elif metodo == "secante":
        # Secante: xr = x_k - f(x_k) * (x_k - x_{k-1}) / (f(x_k) - f(x_{k-1}))
        block = rf"""
            \resizebox{{0.96\linewidth}}{{!}}{{%
            \begin{{minipage}}{{\linewidth}}
            \centering
            \textbf{{\Large Secante}} \quad (\textit{{iteración {h['i']}}})\\[0.8em]
            $$ f(x) = {expr_tex} $$
            $$ x_{{k+1}} \;=\; x_k \;-\; f(x_k)\,\frac{{x_k - x_{{k-1}}}}{{\,f(x_k) - f(x_{{k-1}})\,}} $$
            $$ x_{{k+1}} \;=\; {xd} \;-\; ({fxd})\,\frac{{{xd} - {xi}}}{{\,{fxd} - {fxi}\,}}
               \;\approx\; {xm} $$
            $$ f(x_{{k+1}}) \;\approx\; {fxm} $$
            \end{{minipage}}
            }}%
        """.strip()
    elif metodo == "newton":
        block = rf"""
            \resizebox{{0.96\linewidth}}{{!}}{{%
            \begin{{minipage}}{{\linewidth}}
            \centering
            \textbf{{\Large Newton–Raphson}} \quad (\textit{{iteración {h['i']}}})\\[0.8em]
            $$ f(x) = {expr_tex} $$
            $$ x_{{k+1}} \;=\; x_k \;-\; \frac{{f(x_k)}}{{f'(x_k)}} $$
            %\ \text{{(usando derivada numérica)}}
            \\[0.4em]
            $$ x_{{k+1}} \;\approx\; {xm} \qquad\qquad f(x_{{k+1}}) \;\approx\; {fxm} $$
            \end{{minipage}}
            }}%
        """.strip()
    elif metodo == "puntofijo":
        block = rf"""
            \resizebox{{0.96\linewidth}}{{!}}{{%
            \begin{{minipage}}{{\linewidth}}
            \centering
            \textbf{{\Large Punto Fijo}} \quad (\textit{{iteración {h['i']}}})\\[0.8em]
            $$ {expr_tex} $$
            $$ x_{{k+1}} \;=\; g(x_k) $$
            $$ x_k = {xi} \;\Rightarrow\; x_{{k+1}} = g(x_k) \;\approx\; {xm} $$
            $$ f(x_{{k+1}}) \;\approx\; {fxm} $$
            \end{{minipage}}
            }}%
        """.strip()
    elif metodo == "trapecio":
        area = ff(h.get("area", 0))
        n = h.get("n_seg", 1)
        err = ff(h.get("error", 0))
        
        # Diferenciar entre simple (n=1) y compuesto (n>1) en el título
        titulo = "Trapecio Simple" if n == 1 else "Trapecio Compuesto"
        formula = ""
        if n == 1:
            formula = rf"Area \approx \frac{{b-a}}{{2}}[f(a) + f(b)]"
        else:
            formula = rf"Area \approx \frac{{h}}{{2}}[f(a) + 2\sum f(x_i) + f(b)]"
            
        block = rf"""
            \resizebox{{0.96\linewidth}}{{!}}{{%
            \begin{{minipage}}{{\linewidth}}
            \centering
            \textbf{{\Large {titulo}}} \quad (n={n})\\[0.8em]
            $$ f(x) = {expr_tex} $$
            $$ {formula} $$
            $$ \text{{Area Aprox}} \approx {area} $$
            $$ \text{{Error Est.}} \approx {err} $$
            \end{{minipage}}
            }}%
        """.strip()
    else:
        raise ValueError("Método no soportado")

    return [block]


def precompute_latex_panels(expr: str, hist: List[Iteracion], metodo: MetodoNombre) -> List[str]:
    """
    Precompila un PNG por iteración con el panel LaTeX/mathtext. Devuelve rutas PNG.
    """
    tmp_dir = tempfile.mkdtemp(prefix=f"latex_panels_{metodo}_")
    pngs: List[str] = []
    for h in hist:
        lines = make_latex_panel(expr, h, metodo)
        fname = f"{metodo}_iter_{h['i']:03d}"
        png_path = _compile_latex_to_png(lines, tmp_dir, fname)
        pngs.append(png_path)
    return pngs


def make_interleaved_frames(n_iters: int, panel_hold: int = 1) -> List[tuple[str, int]]:
    """
    Genera frames intercalados: ('plot', i) y ('panel', i) * panel_hold.
    """
    frames: List[tuple[str, int]] = []
    for i in range(n_iters):
        frames.append(("plot", i))
        for _ in range(panel_hold):
            frames.append(("panel", i))
    return frames
