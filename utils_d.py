from __future__ import annotations

"""
Módulo: utils.py
----------------

Utilidades comunes para los métodos de raíces y sus animaciones:

- **Evaluación segura** de funciones escritas como texto por el usuario (`make_func`).
- **Criterios de paro** estandarizados para todos los métodos (`criterio_cumplido`).
- **Generación de paneles LaTeX/mathtext** por iteración para las animaciones
  (`make_latex_panel`, `_compile_latex_to_png`, `_render_mathtext_png`, etc.).
- **Secuencia de fotogramas intercalados** entre gráfica y panel LaTeX
  (`make_interleaved_frames`).

Diseño
~~~~~~
- El historial de cada método usa un **esquema de claves unificado** (TypedDict `Iteracion`):
  `i, xi, fxi, xd, fxd, xm, fxm`. Esto permite reusar tablas, paneles y animaciones.
- Los paneles con fórmulas se producen **usando LaTeX** si el entorno lo permite
  (compilador + `pylatex` + `pdf2image`), y si no, hay *fallback* a **mathtext** (matplotlib).
- `make_func` expone un entorno **restringido** con `math` y la variable `x` para
  evaluar expresiones como `x**3 - 2*x - 5`.

Requisitos opcionales
~~~~~~~~~~~~~~~~~~~~~
- LaTeX: un compilador disponible en PATH (`latexmk`, `pdflatex`, `xelatex` o `lualatex`).
- `pylatex` y `pdf2image` para el pipeline LaTeX→PDF→PNG.
- Si faltan, el sistema **degrada** a mathtext (no se rompe la ejecución).
"""

import math
import os
import shutil
import tempfile
from io import BytesIO
from typing import Callable, Dict, Iterable, List, Literal, TypedDict

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Dependencias opcionales (manejo seguro con fallback)
try:
    from pdf2image import convert_from_path  # type: ignore
except Exception:  # pragma: no cover
    convert_from_path = None  # type: ignore

try:
    from pylatex import Document, NoEscape, Package  # type: ignore
except Exception:  # pragma: no cover
    Document = NoEscape = Package = None  # type: ignore


# ------------------------------- Tipos ---------------------------------------

class Iteracion(TypedDict):
    """
    Trazo de una iteración con formato unificado para todos los métodos.

    Keys
    ----
    i : int
        Índice de iteración (k).
    xi : float
        Extremo/semilla izquierda o x_{k-1} (según el método).
    fxi : float
        f(xi).
    xd : float
        Extremo/semilla derecha o x_{k}.
    fxd : float
        f(xd).
    xm : float
        Nuevo punto (x_m en bisección, x_r en pos. falsa, x_{k+1} en secante/newton/p.fijo).
    fxm : float
        f(xm).
    """
    i: int
    xi: float
    fxi: float
    xd: float
    fxd: float
    # xm para bisección y xr para posición falsa (nomenclatura común)
    xm: float
    fxm: float


MetodoNombre = Literal["biseccion", "posfalsa",
                       "secante", "newton", "puntofijo"]


# ----------------------- Utilidades de evaluación ----------------------------

def make_func(expr: str) -> Callable[[float], float]:
    """
    Convierte una cadena como ``'x**3 - 2*x - 5'`` en una función ``f(x)``.

    La evaluación ocurre en un **entorno seguro**:
    - Sin `__builtins__`.
    - Solo se expone el módulo `math` (funciones y constantes públicas).
    - Variable `x` como flotante.

    Parameters
    ----------
    expr : str
        Expresión en sintaxis de Python (usa `**`, `math` y nombres válidos).

    Returns
    -------
    Callable[[float], float]
        Función `f(x)` evaluable en floats.

    Notes
    -----
    - Se usa `compile(..., 'eval')` y `eval` con *globals* vacíos y *locals*
      restringidos. Esto **no** protege frente a expresiones maliciosas que
      abusen de nombres de `math` en entornos no aislados, pero es suficiente
      para escenarios docentes típicos.
    - Si necesitas nombres adicionales (p. ej. `np`), es mejor **no** exponerlos
      aquí y mantener el entorno acotado.
    """
    # Diccionario de nombres permitidos: todo math público + placeholder de x
    allowed: Dict[str, float | Callable[..., float]] = {
        k: getattr(math, k) for k in dir(math) if not k.startswith("_")
    }
    allowed.update({"x": 0.0})

    # Compila la expresión una sola vez (más eficiente que compilar en cada llamada)
    code = compile(expr, "<user-f>", "eval")

    def f(x: float) -> float:
        # Actualiza x en el entorno y evalúa la expresión
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
    Verifica si se cumple el **criterio de paro** seleccionado.

    Criterios soportados
    --------------------
    - ``abs_fx`` :  |f(xm)| <= tol
    - ``abs``    :  |xm - xm_prev| <= tol
    - ``rel``    :  |xm - xm_prev| / |xm| <= tol
    - ``pct``    :  100 * |xm - xm_prev| / |xm| <= tol

    Parameters
    ----------
    criterio : {"abs_fx", "abs", "rel", "pct"}
        Nombre del criterio de paro.
    tol : float
        Tolerancia (si ``criterio='pct'``, se interpreta como **porcentaje**).
    fxm : float
        Valor f(xm) de la iteración actual.
    xm : float
        Punto actual xm (x_{k+1} en métodos abiertos).
    xm_prev : float | None
        Punto previo (para criterios basados en desplazamiento). Si es None,
        los criterios que lo requieren retornan False.

    Returns
    -------
    bool
        ``True`` si se cumple el criterio, ``False`` en caso contrario.

    Raises
    ------
    ValueError
        Si el nombre del criterio no es reconocido.
    """
    if criterio == "abs_fx":
        return abs(fxm) <= tol

    # Los criterios basados en desplazamiento requieren xm_prev
    if xm_prev is None:
        return False

    diff = abs(xm - xm_prev)

    if criterio == "abs":
        return diff <= tol

    # Evita división por cero en criterios relativos
    denom = abs(xm) if xm != 0 else 1.0
    if criterio == "rel":
        return (diff / denom) <= tol
    if criterio == "pct":
        return (diff / denom) * 100.0 <= tol

    raise ValueError("Criterio no reconocido")


# --------------------- Formateo y textos LaTeX / mathtext --------------------

def _fmt_float(v: float, digits: int = 8) -> str:
    """
    Formatea un float con `digits` cifras significativas (fallback a `str`).

    Parameters
    ----------
    v : float
        Valor a formatear.
    digits : int, optional
        Cifras significativas. Por defecto 8.

    Returns
    -------
    str
        Representación corta del número.
    """
    try:
        return f"{float(v):.{digits}g}"
    except Exception:
        return str(v)


def _latex_doc_from_lines(lines: Iterable[str]):
    """
    Crea un documento LaTeX *standalone* con márgenes cero y modo display.

    Usa `pylatex` si está disponible. Si no, retorna ``None`` y el llamador
    deberá usar el *fallback* a mathtext.

    Parameters
    ----------
    lines : Iterable[str]
        Líneas de contenido LaTeX (ya escapadas y en el orden deseado).

    Returns
    -------
    pylatex.Document | None
        Documento listo para compilar, o ``None`` si `pylatex` no está disponible.
    """
    if Document is None:  # Fallback si no está pylatex
        return None
    doc = Document(documentclass="standalone", document_options=["border=0pt"])
    # Paquetes comunes para fórmulas
    doc.packages.append(Package("amsmath"))
    doc.packages.append(Package("amssymb"))
    doc.packages.append(Package("lmodern"))
    doc.packages.append(Package("graphicx"))  # \resizebox
    # Quita numeración y fuerza displaystyle
    doc.preamble.append(NoEscape(r"\pagenumbering{gobble}"))
    doc.preamble.append(NoEscape(r"\everymath{\displaystyle}"))
    for ln in lines:
        doc.append(NoEscape(ln))
        doc.append("\n")
    return doc


def _find_latex_compiler() -> str | None:
    """
    Busca un compilador LaTeX disponible en PATH.

    Returns
    -------
    str | None
        Nombre del ejecutable encontrado (``latexmk``, ``pdflatex``, ``xelatex`` o ``lualatex``),
        o ``None`` si no hay ninguno.
    """
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
    Genera un PNG usando **mathtext** (matplotlib) como *fallback* sin LaTeX.

    Parameters
    ----------
    lines : Iterable[str]
        Líneas con fórmulas (``$...$``) y texto.
    out_dir : str
        Directorio de salida (se crea si no existe).
    basename : str
        Nombre base del archivo (sin extensión).
    dpi : int, optional
        Resolución del PNG. Por defecto 300.
    fontsize : int, optional
        Tamaño de fuente. Por defecto 26.

    Returns
    -------
    str
        Ruta absoluta del PNG generado (``.png``).

    Notes
    -----
    - Se normalizan ``$$``→``$`` para evitar conflictos con mathtext.
    - Se usa un lienzo amplio y transparente para embebido en la animación.
    """
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=(8.0, 4.5), dpi=dpi)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.axis("off")

    # mathtext no soporta $$...$$; se reemplaza por $...$
    safe_lines = [ln.replace("$$", "$") for ln in lines]
    text = "\n".join(safe_lines)

    # Escribe las líneas en la figura
    ax.text(0.02, 0.98, text, va="top", ha="left", fontsize=fontsize)

    # Exporta a buffer y luego a archivo (para forzar RGBA y transparencia)
    buf = BytesIO()
    fig.savefig(buf, format="png", transparent=True,
                bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    buf.seek(0)
    png_path = os.path.join(out_dir, basename + ".png")
    Image.open(buf).convert("RGBA").save(png_path)
    return png_path


def _compile_latex_to_png(lines: Iterable[str], out_dir: str, basename: str) -> str:
    """
    Compila **LaTeX→PDF→PNG** si el entorno lo permite; si no, usa *fallback* a mathtext.

    Requisitos para pipeline LaTeX:
    - Compilador disponible (`latexmk`/`pdflatex`/`xelatex`/`lualatex`)
    - `pylatex`
    - `pdf2image`

    Parameters
    ----------
    lines : Iterable[str]
        Líneas LaTeX.
    out_dir : str
        Directorio de salida.
    basename : str
        Nombre base del archivo (sin extensión).

    Returns
    -------
    str
        Ruta del PNG generado.

    Notes
    -----
    - El documento se compila como *standalone* con borde 0 para fácil incrustación.
    - Si cualquier paso falla, se recurre a `_render_mathtext_png` (robustez).
    """
    compiler = _find_latex_compiler()
    have_pdf2image = convert_from_path is not None
    have_pylatex = Document is not None

    # Si falta alguno, degrada a mathtext
    if not (compiler and have_pdf2image and have_pylatex):
        return _render_mathtext_png(lines, out_dir, basename, dpi=300, fontsize=28)

    os.makedirs(out_dir, exist_ok=True)
    pdf_base = os.path.join(out_dir, basename)
    doc = _latex_doc_from_lines(lines)

    try:
        # Compilación LaTeX (latexmk preferente si existe)
        if compiler == "latexmk":
            doc.generate_pdf(filepath=pdf_base, clean_tex=True, silent=True)
        else:
            doc.generate_pdf(filepath=pdf_base, clean_tex=True,
                             compiler=compiler, silent=True)
    except Exception:
        # Si falla, usa mathtext
        return _render_mathtext_png(lines, out_dir, basename, dpi=300, fontsize=28)

    try:
        # Conversión PDF→PNG
        images = convert_from_path(pdf_base + ".pdf", dpi=300)  # type: ignore
        png_path = os.path.join(out_dir, basename + ".png")
        images[0].save(png_path)
        return png_path
    except Exception:
        # Fallback final
        return _render_mathtext_png(lines, out_dir, basename, dpi=300, fontsize=28)


def make_latex_panel(expr: str, h: Iteracion, metodo: MetodoNombre, digits: int = 8) -> List[str]:
    """
    Construye un bloque LaTeX (como lista de líneas) para **una iteración** del método.

    El contenido se ajusta al método (`metodo`) e inserta valores numéricos formateados.

    Parameters
    ----------
    expr : str
        Expresión de f(x) en texto (se convierte a LaTeX-friendly).
    h : Iteracion
        Registro de la iteración (usa llaves estándar del proyecto).
    metodo : {"biseccion","posfalsa","secante","newton","puntofijo"}
        Método al que corresponde el panel.
    digits : int, optional
        Cifras significativas para valores numéricos. Por defecto 8.

    Returns
    -------
    List[str]
        Líneas LaTeX (listas para pasar a compilación).

    Raises
    ------
    ValueError
        Si el método no está soportado.
    """
    def ff(v): return f"{float(v):.{digits}g}"

    # Formatea números y adapta '*'→'\cdot' para tipografía LaTeX
    xi, xd, xm = ff(h["xi"]), ff(h["xd"]), ff(h["xm"])
    fxi, fxd, fxm = ff(h["fxi"]), ff(h["fxd"]), ff(h["fxm"])
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
    elif metodo == "posfalsa":  # posición falsa
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
    else:
        raise ValueError("Método no soportado")

    return [block]


def precompute_latex_panels(expr: str, hist: List[Iteracion], metodo: MetodoNombre) -> List[str]:
    """
    Precompila un **PNG por iteración** con el panel LaTeX/mathtext y devuelve sus rutas.

    Si el entorno no tiene LaTeX, `pylatex` o `pdf2image`, se usa mathtext como fallback.

    Parameters
    ----------
    expr : str
        Expresión de f(x) (texto).
    hist : List[Iteracion]
        Historial de iteraciones (se itera en orden de `i`).
    metodo : MetodoNombre
        Nombre del método (selecciona plantilla de panel).

    Returns
    -------
    List[str]
        Rutas absolutas de los PNG generados (uno por iteración).

    Notes
    -----
    - Se crea un directorio temporal por método con prefijo `latex_panels_<metodo>_`.
    - Cada archivo se nombra `"<metodo>_iter_<i>.png"`.
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
    Genera la secuencia de **fotogramas intercalados** para animación.

    El patrón por iteración es:
        ('plot', i) seguido por `panel_hold` veces ('panel', i)

    Parameters
    ----------
    n_iters : int
        Número de iteraciones (longitud del historial).
    panel_hold : int, optional
        Número de fotogramas de panel a mantener por cada iteración. Por defecto 1.

    Returns
    -------
    List[tuple[str, int]]
        Lista de tuplas (tipo, i) donde ``tipo`` ∈ {``"plot"``, ``"panel"``}.

    Examples
    --------
    >>> make_interleaved_frames(2, panel_hold=2)
    [('plot', 0), ('panel', 0), ('panel', 0), ('plot', 1), ('panel', 1), ('panel', 1)]
    """
    frames: List[tuple[str, int]] = []
    for i in range(n_iters):
        frames.append(("plot", i))
        for _ in range(panel_hold):
            frames.append(("panel", i))
    return frames
