from __future__ import annotations

from typing import Callable, List, Tuple

import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from utils import (
    Iteracion,
    criterio_cumplido,
    make_interleaved_frames,
    precompute_latex_panels,
)


def mostrar_explicacion():
    msg = r"""
El método de Simpson 1/3 aproxima la integral definida (\int_a^b f(x),dx) usando parábolas sobre intervalos igualmente espaciados. Para aplicarlo, el intervalo ([a,b]) se divide en (n) subintervalos iguales, donde (n) debe ser **par**. Se define:

[
\Delta x = \frac{b - a}{n}
]

Los puntos del intervalo se calculan como:

[
x_i = a + i(\Delta x), \quad i = 0, 1, 2, \dots, n
]

**Simpson 1/3 simple (n = 2):**

[
\int_{x_0}^{x_2} f(x),dx \approx \frac{\Delta x}{3}
\left[f(x_0) + 4f(x_1) + f(x_2)\right]
]

**Simpson 1/3 compuesto (n par):**

[
\int_a^b f(x),dx \approx \frac{\Delta x}{3} \left[
f(x_0)
+ 4\sum_{\text{i impar}} f(x_i)
+ 2\sum_{\text{i par},; i\neq 0,; i\neq n} f(x_i)
+ f(x_n)
  \right]
  ]

Reglas de coeficientes:

* (f(x_0)) y (f(x_n)) tienen coeficiente **1**.
* Los términos con índice impar tienen coeficiente **4**.
* Los términos con índice par (excepto (0) y (n)) tienen coeficiente **2**.

Procedimiento general:

1. Calcular (\Delta x).
2. Generar los puntos (x_i).
3. Evaluar la función en cada (x_i).
4. Aplicar los coeficientes correspondientes:
   * extremos -> 1
   * impares -> 4
   * pares internos -> 2
5. Multiplicar la suma obtenida por (\Delta x / 3).
"""
    print(msg)


def metodo_simpson_iterativo(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-4,
    criterio: str = "abs",
    max_iter: int = 10,
) -> Tuple[float, int, List[Iteracion]]:
    """
    Método de Simpson 1/3 Iterativo.
    """
    if a == b:
        return 0.0, 0, []
    
    historial: List[Iteracion] = []

    # ---------------------------------------------------------
    # Iteración -1: Repaso de Fórmulas (Intro)
    # ---------------------------------------------------------
    historial.append({
        "i": -1, # Flag for Intro
        "xi": a, "xd": b, "xm": 0,
        "fxi": 0, "fxd": 0, "fxm": 0,
        "area": 0,
        "n_seg": 0,
        "error": 0,
        "sum_odd": 0,
        "sum_even": 0
    })
    
    # ---------------------------------------------------------
    # Iteración 0: Simpson Simple (n=2)
    # ---------------------------------------------------------
    n = 2
    # 1. Calcular Delta x (h)
    h = (b - a) / n  
    
    # 2. Generar los puntos (x_i)
    x_puntos = np.linspace(a, b, n + 1)
    
    # 3. Evaluar la función en cada (x_i)
    y_puntos = np.array([f(x) for x in x_puntos])
    
    # 4. Aplicar coeficientes: 1, 4, 1
    # 5. Multiplicar por Delta x / 3
    # Para n=2, impares es y[1] (coef 4), pares es 0 (no hay pares internos entre 0 y 2)
    sum_odd_0 = y_puntos[1]
    sum_even_0 = 0.0
    area_prev = (h / 3.0) * (y_puntos[0] + 4*sum_odd_0 + y_puntos[2])
    
    historial.append({
        "i": 0,
        "xi": a, "xd": b, "xm": area_prev, 
        "fxi": y_puntos[0], "fxd": y_puntos[-1], "fxm": 0, # Storing f(a) and f(b)
        "area": area_prev,
        "n_seg": n,
        "error": 1.0,
        "sum_odd": sum_odd_0,
        "sum_even": sum_even_0
    })
    
    if max_iter == 0:
        return area_prev, 0, historial

    area_actual = area_prev
    
    for k in range(1, max_iter + 1):
        # Simpson Compuesto
        n_prev = n
        n = 2 * n_prev # Duplicar n (siempre par)
        
        # 1. Calcular Delta x
        h = (b - a) / n
        
        # 2. Generar puntos y 3. Evaluar
        x_puntos = np.linspace(a, b, n + 1)
        y_puntos = np.array([f(x) for x in x_puntos])
        
        # 4. Aplicar coeficientes:
        # Extremos (0 y n) -> 1
        # Impares -> 4
        # Pares internos -> 2
        suma_impares = np.sum(y_puntos[1:n:2])
        suma_pares = np.sum(y_puntos[2:n-1:2])
        
        # 5. Multiplicar por Delta x / 3
        area_actual = (h / 3.0) * (y_puntos[0] + 4*suma_impares + 2*suma_pares + y_puntos[-1])
        
        err = abs(area_actual - area_prev)
        
        historial.append({
            "i": k,
            "xi": a, "xd": b, "xm": area_actual,
            "fxi": y_puntos[0], "fxd": y_puntos[-1], "fxm": err, # Storing f(a) and f(b)
            "area": area_actual,
            "n_seg": n,
            "error": err,
            "sum_odd": suma_impares,
            "sum_even": suma_pares
        })
        
        if criterio == "abs":
            if err <= tol:
                return area_actual, k, historial
        elif criterio == "rel":
            if abs(area_actual) > 1e-12 and (err / abs(area_actual)) <= tol:
                return area_actual, k, historial
                
        area_prev = area_actual

    return area_actual, max_iter, historial


def _get_parabola_poly(x0, x1, x2, y0, y1, y2):
    """
    Devuelve poly(x) que interpola los 3 puntos.
    """
    def poly(x):
        l0 = (x - x1)*(x - x2) / ((x0 - x1)*(x0 - x2))
        l1 = (x - x0)*(x - x2) / ((x1 - x0)*(x1 - x2))
        l2 = (x - x0)*(x - x1) / ((x2 - x0)*(x2 - x1))
        return y0*l0 + y1*l1 + y2*l2
    return poly


def animar_simpson(
    f: Callable[[float], float],
    hist: List[Iteracion],
    xi_inicial: float,
    xd_inicial: float,
    titulo: str = "Método de Simpson 1/3",
    guardar_gif: bool = False,
    nombre_gif: str = "simpson.gif",
    fps: int = 1,
    intervalo_ms: int = 1500,
    expr_text: str = "f(x)",
    panel_hold: int = 1,
) -> None:
    """
    Animación de Simpson 1/3.
    """
    # Pre-render
    latex_pngs = precompute_latex_panels(expr_text, hist, metodo="simpson")
    
    a, b = float(xi_inicial), float(xd_inicial)
    if a > b:
        a, b = b, a
        
    # Curva real
    muestras = 500
    xs = np.linspace(a, b, muestras)
    ys = np.array([f(x) for x in xs])
    
    ymin, ymax = np.min(ys), np.max(ys)
    span_y = (ymax - ymin) if (ymax - ymin) > 1e-9 else 1.0
    ylo, yhi = ymin - 0.2 * span_y, ymax + 0.2 * span_y
    
    span_x = (b - a) if (b - a) > 1e-9 else 1.0
    xlo, xhi = a - 0.1 * span_x, b + 0.1 * span_x
    
    # Figura
    fig = plt.figure(figsize=(10, 6))
    
    # Axes plot
    ax = fig.add_axes([0.05, 0.10, 0.90, 0.80])
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_title(titulo, fontsize=14, fontweight='bold', color='#333333')
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("f(x)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3, color='#aaaaaa')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.axhline(0, color='black', linewidth=1)
    ax.plot(xs, ys, color='#2c3e50', linewidth=2.5, label="f(x)")
    
    # Info box
    info_box = ax.text(
        0.02, 0.95, "", transform=ax.transAxes,
        verticalalignment='top', fontsize=11,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', alpha=0.9, edgecolor='#dee2e6')
    )
    
    patches_collection = []
    lines_collection = []
    
    # Axes panel
    ax_panel = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax_panel.axis("off")
    im_panel = ax_panel.imshow(np.zeros((1,1)), aspect="auto")
    ax_panel.set_visible(False)
    
    frames = make_interleaved_frames(len(hist), panel_hold=panel_hold)
    
    def _draw_parabolas(n_seg: int):
        # Limpiar
        for p in patches_collection: p.remove()
        for l in lines_collection: l.remove()
        patches_collection.clear()
        lines_collection.clear()
        
        # Puntos base
        x_base = np.linspace(a, b, n_seg + 1)
        # Vamos de 2 en 2
        
        color_fill = '#e67e22' # Naranja para Simpson
        color_edge = '#d35400'
        alpha_fill = 0.3
        
        for i in range(0, n_seg, 2):
            # Triplete (x0, x1, x2) locales
            idx0, idx1, idx2 = i, i+1, i+2
            if idx2 >= len(x_base): break 
            
            x0, x1, x2 = x_base[idx0], x_base[idx1], x_base[idx2]
            y0, y1, y2 = f(x0), f(x1), f(x2)
            
            # Polinomio cuadrático que interpola
            poly = _get_parabola_poly(x0, x1, x2, y0, y1, y2)
            
            # Generar arco suave para dibujar
            x_arc = np.linspace(x0, x2, 20)
            y_arc = poly(x_arc)
            
            # Crear polígono cerrado: (x0,0) -> arco... -> (x2,0)
            verts = [(x0, 0)] + list(zip(x_arc, y_arc)) + [(x2, 0)]
            
            poly_patch = patches.Polygon(verts, closed=True, facecolor=color_fill, edgecolor=None, alpha=alpha_fill)
            ax.add_patch(poly_patch)
            patches_collection.append(poly_patch)
            
            # Dibujar el arco superior (borde)
            (line,) = ax.plot(x_arc, y_arc, color=color_edge, linewidth=1.5, linestyle='-')
            lines_collection.append(line)
            
            # Líneas verticales discontinuas en los extremos del par
            linev = ax.vlines(x=[x0, x1, x2], ymin=0, ymax=[y0, y1, y2], colors=color_edge, linestyles=':', alpha=0.5)
            lines_collection.append(linev)

    def _update_plot(i: int):
        h = hist[i]
        n_seg = h.get("n_seg", 2)
        area = h.get("area", 0)
        err = h.get("error", 0)
        
        _draw_parabolas(n_seg)
        
        info_text = (
            f"Iteración: {h['i']}\n"
            f"Segmentos (n): {n_seg}\n"
            f"Area Aprox: {area:.6f}"
        )
        info_box.set_text(info_text)

    def init():
        ax_panel.set_visible(False)
        ax.set_visible(True)
        return []

    def update(frame):
        kind, i = frame
        if kind == "plot":
            if not ax.get_visible(): ax.set_visible(True)
            if ax_panel.get_visible(): ax_panel.set_visible(False)
            _update_plot(i)
            return []
        else:
            if not ax_panel.get_visible(): ax_panel.set_visible(True)
            if ax.get_visible(): ax.set_visible(False)
            try:
                img = mpimg.imread(latex_pngs[i])
                im_panel.set_data(img)
            except Exception: pass
            return [im_panel]
            
    ani = FuncAnimation(fig, update, frames=frames, init_func=init,
                        interval=intervalo_ms, blit=False, repeat=False)
                        
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

if __name__ == "__main__":
    from utils import make_func
    
    mostrar_explicacion()
    
    print("=== Método de Simpson 1/3 (Iterativo) ===")
    expr = input("Ingresa f(x), p.ej.: sin(x) + 2: ").strip()
    f = make_func(expr)
    
    a_in = float(input("Ingresa a (limite inferior): "))
    b_in = float(input("Ingresa b (limite superior): "))
    # tol = float(input("Tolerancia (ej 1e-4): "))
    
    val, iters, hist = metodo_simpson_iterativo(f, a_in, b_in, tol=1e-4, max_iter=1)
    
    print(f"\nResultado: {val}")
    print(f"Iteraciones: {iters}")
    
    animar = input("\n¿Animar? (s/n): ").lower() == 's'
    if animar:
        save = input("¿Guardar GIF? (s/n): ").lower() == 's'
        animar_simpson(f, hist, a_in, b_in, guardar_gif=save, expr_text=expr)
