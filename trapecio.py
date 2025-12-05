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


def metodo_trapecio_iterativo(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-4,
    criterio: str = "abs",  # abs (diferencia con area previa)
    max_iter: int = 10,  # Ojo: iteracion 0 -> n=1, iter 1 -> n=2, etc.
) -> Tuple[float, int, List[Iteracion]]:
    """
    Método del Trapecio Iterativo (Simple -> Compuesto).
    Comienza con n=1 (Simple) y va doblando el número de segmentos: 1, 2, 4, 8...
    O simplemente incrementando. Para convergencia rápida y reuso, n=2^i.
    
    Devuelve (area_aprox, iter_usadas, historial).
    """
    if a == b:
        return 0.0, 0, []
    
    # Asegurar orden
    if a > b:
        a, b = b, a
        signo = -1.0
    else:
        signo = 1.0

    historial: List[Iteracion] = []
    
    # Iteración 0: Trapecio Simple (n=1)
    n = 1
    h = (b - a) / n
    area_prev = (f(a) + f(b)) * h / 2.0
    
    # Guardamos iter 0
    historial.append({
        "i": 0,
        "xi": a, "xd": b, "xm": area_prev, # Usamos 'xm' para guardar el area actual
        "fxi": 0.0, "fxd": 0.0, "fxm": 0.0, # Campos dummy para cumplir TypedDict si fuera estricto
        "area": area_prev * signo,
        "n_seg": n,
        "error": 1.0 # Error inicial grande
    })
    
    if max_iter == 0:
        return area_prev * signo, 0, historial

    # Iteraciones siguientes: Trapecio Compuesto
    area_actual = area_prev
    
    for k in range(1, max_iter + 1):
        # Estrategia: Duplicar n cada vez (Romberg style base) o incrementar linealmente?
        # Duplicar es mejor para visualizar "refinamiento".
        n_prev = n
        n = 2 * n_prev
        h = (b - a) / n
        
        # Suma de nuevos puntos (puntos impares en la nueva malla)
        # x_new = a + (2*j - 1)*h_new, j=1..n_prev
        # O cálculo directo de trapecio compuesto estándar:
        x_puntos = np.linspace(a, b, n + 1)
        y_puntos = np.array([f(x) for x in x_puntos])
        
        # Fórmula compuesta: h/2 * [f(a) + 2*sum(f_intermedios) + f(b)]
        suma_intermedios = np.sum(y_puntos[1:-1])
        area_actual = (h / 2.0) * (y_puntos[0] + 2 * suma_intermedios + y_puntos[-1])
        
        err = abs(area_actual - area_prev)
        
        historial.append({
            "i": k,
            "xi": a, "xd": b, "xm": area_actual,
            "fxi": 0, "fxd": 0, "fxm": err, # fxm guarda el error (diferencia)
            "area": area_actual * signo,
            "n_seg": n,
            "error": err
        })
        
        # Verificación de criterio de convergencia
        # Usamos 'abs' como |area_new - area_old| <= tol
        if criterio == "abs":
            if err <= tol:
                return area_actual * signo, k, historial
        elif criterio == "rel":
            if abs(area_actual) > 1e-12 and (err / abs(area_actual)) <= tol:
                return area_actual * signo, k, historial
        
        area_prev = area_actual

    return area_actual * signo, max_iter, historial


def animar_trapecio(
    f: Callable[[float], float],
    hist: List[Iteracion],
    xi_inicial: float, # a
    xd_inicial: float, # b
    titulo: str = "Método del Trapecio",
    guardar_gif: bool = False,
    nombre_gif: str = "trapecio.gif",
    fps: int = 1, # FPS bajo para apreciar cada n
    intervalo_ms: int = 1500,
    expr_text: str = "f(x)",
    panel_hold: int = 1,
) -> None:
    """
    Animación del método del Trapecio (Simple -> Compuesto)
    Muestra cómo al aumentar n, los trapecios se ajustan mejor a la curva.
    """
    # Pre-render de paneles LaTeX
    latex_pngs = precompute_latex_panels(expr_text, hist, metodo="trapecio")
    
    # Datos básicos
    a, b = float(xi_inicial), float(xd_inicial)
    if a > b:
        a, b = b, a
        
    # Configuración de límites y curva f(x)
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
    
    # Ejes principales con diseño profesional
    ax = fig.add_axes([0.05, 0.10, 0.90, 0.80])
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_title(titulo, fontsize=14, fontweight='bold', color='#333333')
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("f(x)", fontsize=12)
    
    # Estilo de grid y ejes
    ax.grid(True, linestyle='--', alpha=0.3, color='#aaaaaa')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Línea base y curva f(x)
    ax.axhline(0, color='black', linewidth=1)
    (lineafx,) = ax.plot(xs, ys, color='#2c3e50', linewidth=2.5, label="f(x)")
    
    # Colección de parches (trapecios) para actualizar
    patches_collection = []
    
    # Texto de info
    info_box = ax.text(
        0.02, 0.95, "", transform=ax.transAxes,
        verticalalignment='top', fontsize=11,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', alpha=0.9, edgecolor='#dee2e6')
    )
    
    # Panel LaTeX superpuesto
    ax_panel = fig.add_axes([0.05, 0.05, 0.35, 0.35]) # Panel abajo izquierda
    ax_panel.axis("off")
    im_panel = ax_panel.imshow(np.zeros((1,1)), aspect="auto") # Placeholder
    ax_panel.set_visible(False)
    # Reubicar panel LaTeX? Mejor que use logic de utils 'interleaved' => panel full screen o overlay?
    # El código actual de utils espera que el panel tenga su propio 'momento' fullscreen o similar.
    # En biseccion.py, el panel ocupa (0.02, 0.02, 0.96, 0.96) y switchea visibilidad con el plot.
    # Seguiremos ese patrón para consistencia cinematográfica.
    ax_panel.set_position([0.02, 0.02, 0.96, 0.96])
    
    frames = make_interleaved_frames(len(hist), panel_hold=panel_hold)
    
    def _draw_trapezoids(n_seg: int):
        # Limpiar trapecios anteriores
        for p in patches_collection:
            p.remove()
        patches_collection.clear()
        
        # Calcular puntos de trapecios
        x_trap = np.linspace(a, b, n_seg + 1)
        y_trap = [f(x) for x in x_trap]
        
        color_fill = '#3498db'
        color_edge = '#2980b9'
        alpha_fill = 0.3
        
        for i in range(n_seg):
            x0, x1 = x_trap[i], x_trap[i+1]
            y0, y1 = y_trap[i], y_trap[i+1]
            
            # Polígono del trapecio: (x0,0) -> (x0,y0) -> (x1,y1) -> (x1,0)
            verts = [(x0, 0), (x0, y0), (x1, y1), (x1, 0)]
            poly = patches.Polygon(verts, closed=True, facecolor=color_fill, edgecolor=color_edge, alpha=alpha_fill, linewidth=1.5)
            ax.add_patch(poly)
            patches_collection.append(poly)
            
            # Líneas verticales discontinuas para enfatizar segmentos
            # linev = ax.vlines(x=[x0, x1], ymin=0, ymax=[y0, y1], colors=color_edge, linestyles=':', alpha=0.6)
            # (No fácil de borrar si usamos vlines collection, mejor el borde del poligono ya hace eso)

    def _update_plot(i: int):
        h = hist[i]
        n_seg = h.get("n_seg", 1)
        area = h.get("area", 0)
        err = h.get("error", 0)
        
        _draw_trapezoids(n_seg)
        
        info_text = (
            f"Iteración: {h['i']}\n"
            f"Segmentos (n): {n_seg}\n"
            f"Area Aprox: {area:.6f}\n"
            f"Dif/Error: {err:.2e}"
        )
        info_box.set_text(info_text)

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
        else: # panel
            if not ax_panel.get_visible():
                ax_panel.set_visible(True)
            if ax.get_visible():
                ax.set_visible(False)
            # Cargar imagen
            try:
                img = mpimg.imread(latex_pngs[i])
                im_panel.set_data(img)
            except Exception:
                pass
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
