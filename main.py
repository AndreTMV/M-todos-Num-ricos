from utils import make_func
from biseccion import biseccion, animar_biseccion
from posicion_falsa import posicion_falsa, animar_posicion_falsa
from secante import secante, animar_secante
from newton import newton_raphson, animar_newton
from punto_fijo import punto_fijo, animar_punto_fijo
from trapecio import metodo_trapecio_iterativo, animar_trapecio

"""
Módulo: main.py
---------------

Interfaz de línea de comandos para ejecutar y visualizar **métodos de búsqueda de raíces** y **métodos numéricos**:

- Bisección
- Posición Falsa (con opción Illinois)
- Secante
- Newton–Raphson
- Punto Fijo
- Trapecio (Integración Numérica)

Flujo general
~~~~~~~~~~~~~
1) Se solicita al usuario la función f(x).
2) Se elige el método.
3) Se piden los datos de entrada (intervalos, valores iniciales, etc.).
4) Se ejecuta el método y se imprime el historial.
5) Se muestra el resultado.
6) Opcionalmente, se lanza una animación.
"""

if __name__ == "__main__":
    # Encabezado
    print("=== Métodos Numéricos: Raíces e Integración ===")

    # 1) Entrada de f(x)
    expr = input("Ingresa f(x), p.ej.: x**3 - 2*x - 5\nf(x) = ").strip()
    f = make_func(expr)

    # 2) Selección de método
    print("\nMétodos disponibles:")
    print("  Raíces: (b)isección / (p)osición falsa / (s)ecante / (n)ewton / (f)ijo")
    print("  Integración: (t)rapecio (Simple y Compuesto Iterativo)")
    
    metodo = input("Elige método: ").strip().lower()
    if metodo not in {"b", "p", "s", "n", "f", "t"}:
        raise ValueError("Opción inválida.")

    # 3) Lectura de parámetros
    criterio = "abs" # default
    tol = 1e-4      # default
    
    # Variables generales
    xi, xd, x0 = 0.0, 0.0, 0.0
    
    if metodo == "t":
        # Trapecio: requiere limites de integración [a, b]
        print("\n--- Configuración para Integración (Trapecio) ---")
        xi = float(input("Límite inferior (a): ")) # Usaremos xi como a
        xd = float(input("Límite superior (b): ")) # Usaremos xd como b
        
        # Opcional: criterio de convergencia para iterativo
        print("El método iterativo aumenta el número de segmentos (n=1,2,4,8...) hasta converger.")
        tol = float(input("Ingresa tolerancia ERROR (ej. 0.0001): "))
        
    elif metodo in {"n", "f"}:
        x0 = float(input("Ingresa x0: "))
        # Criterios para raíces
        print("\nCriterios de paro disponibles:")
        print("- abs_fx : |f(xm)| <= tol")
        print("- abs    : |xm - xm_prev| <= tol")
        print("- rel    : |xm - xm_prev| / |xm| <= tol")
        print("- pct    : (|xm - xm_prev| / |xm|)*100 <= tol")
        criterio = input("Elige criterio (abs_fx/abs/rel/pct): ").strip().lower()
        if criterio not in {"abs_fx", "abs", "rel", "pct"}:
            raise ValueError("Criterio no válido.")
        tol = float(input("Ingresa tolerancia: "))
        
    else: # b, p, s
        xi = float(input("Ingresa xi: "))
        xd = float(input("Ingresa xd: "))
        # Criterios para raíces
        print("\nCriterios de paro disponibles:")
        print("- abs_fx : |f(xm)| <= tol")
        print("- abs    : |xm - xm_prev| <= tol")
        print("- rel    : |xm - xm_prev| / |xm| <= tol")
        print("- pct    : (|xm - xm_prev| / |xm|)*100 <= tol")
        criterio = input("Elige criterio (abs_fx/abs/rel/pct): ").strip().lower()
        if criterio not in {"abs_fx", "abs", "rel", "pct"}:
            raise ValueError("Criterio no válido.")
        tol = float(input("Ingresa tolerancia: "))

    # Variables para animación
    anim_fn = None
    nombre_gif = None
    titulo_anim = None
    expr_text_pf = None
    g = None

    try:
        # 5) Ejecución del método
        raiz_o_area = 0.0
        iters = 0
        hist = []
        
        if metodo == "b":
            raiz_o_area, iters, hist = biseccion(
                f, xi, xd, tol=tol, criterio=criterio, max_iter=200)
            titulo_anim = f"Método de Bisección | f(x) = {expr}"
            anim_fn = animar_biseccion
            nombre_gif = "biseccion.gif"

        elif metodo == "p":
            usar_ill = input("¿Usar Illinois? (s/n): ").strip().lower() == "s"
            raiz_o_area, iters, hist = posicion_falsa(
                f, xi, xd, tol=tol, criterio=criterio, max_iter=200, usar_illinois=usar_ill)
            titulo_anim = f"Posición Falsa{' (Illinois)' if usar_ill else ''} | f(x) = {expr}"
            anim_fn = animar_posicion_falsa
            nombre_gif = "posicion_falsa.gif"

        elif metodo == "s":
            raiz_o_area, iters, hist = secante(
                f, xi, xd, tol=tol, criterio=criterio, max_iter=200)
            titulo_anim = f"Método de la Secante | f(x) = {expr}"
            anim_fn = animar_secante
            nombre_gif = "secante.gif"

        elif metodo == "n":
            raiz_o_area, iters, hist = newton_raphson(
                f, x0, tol=tol, criterio=criterio, max_iter=200)
            titulo_anim = f"Método de Newton–Raphson | f(x) = {expr}"
            anim_fn = animar_newton
            nombre_gif = "newton.gif"

        elif metodo == "f":
            gexpr = input("Ingresa g(x):\ng(x) = ").strip()
            g = make_func(gexpr)
            raiz_o_area, iters, hist = punto_fijo(
                f, g, x0, tol=tol, criterio=criterio, max_iter=200)
            titulo_anim = f"Método de Punto Fijo | f(x) = {expr}"
            expr_tex = expr.replace("*", "\\cdot ")
            gexpr_tex = gexpr.replace("*", "\\cdot ")
            expr_text_pf = f"f(x) = {expr_tex} \\quad g(x) = {gexpr_tex}"
            anim_fn = animar_punto_fijo
            nombre_gif = "punto_fijo.gif"
            
        elif metodo == "t":
            # Trapecio
            # max_iter controla cuantos doblajes de n hacer (n=1, 2, 4, 8, 16...)
            # Por defecto pongamos un límite razonable (ej. 12 iteraciones -> n=4096)
            raiz_o_area, iters, hist = metodo_trapecio_iterativo(
                f, xi, xd, tol=tol, criterio="abs", max_iter=12
            )
            titulo_anim = f"Método del Trapecio | f(x) = {expr}"
            anim_fn = animar_trapecio
            nombre_gif = "trapecio.gif"

    except Exception as e:
        print(f"\nError: {e}")
    else:
        # 6) Impresión de historial
        print("\n--- Historial ---")
        if metodo == "t":
            # Formato especial para Integración
            print(f"{'i':>3} | {'n':>6} | {'Area Aprox':>12} | {'Error Est.':>12}")
            for h in hist:
                print(f"{h['i']:>3} | {h['n_seg']:>6} | {h['area']:>12.7f} | {h['error']:>12.7f}")
        else:
            # Formato Raíces
            for h in hist:
                print(f"Iter#{h['i']:>3} | Xi={h['xi']:.5f} | Xd={h['xd']:.5f} | Xm={h['xm']:.5f} | f(Xm)={h['fxm']:.5g}")

        # 7) Resultado
        print("\n=== Resultado Final ===")
        label_res = "Área aproximada" if metodo == "t" else "Raíz aproximada"
        print(f"{label_res}: {raiz_o_area:.10f}")
        print(f"Iteraciones/Extensiones: {iters}")
        if metodo != "t":
            print(f"f(raíz) ≈ {f(raiz_o_area):.10g}")
        print(f"Tolerancia usada: {tol}")

        # 8) Animación
        resp = input("\n¿Deseas ver la animación? (s/n): ").strip().lower()
        if resp == "s":
            guardar = input("¿Guardar GIF? (s/n): ").strip().lower()
            if guardar == "s":
                nombre = input(f"Nombre GIF (ej. {nombre_gif}): ").strip() or nombre_gif
            else:
                nombre = nombre_gif

            if metodo == "f":
                anim_fn(f, g, hist, titulo=titulo_anim, guardar_gif=(guardar == "s"), nombre_gif=nombre, expr_text=expr_text_pf)
            elif metodo == "t":
                expr_text_simple = expr.replace("*", "\\cdot ")
                # animación trapecio usa firma similar (f, hist, xi, xd...)
                anim_fn(
                    f, hist, xi_inicial=xi, xd_inicial=xd, # xi=a, xd=b
                    titulo=titulo_anim, guardar_gif=(guardar == "s"),
                    nombre_gif=nombre, expr_text=expr_text_simple
                )
            else:
                expr_text_simple = expr.replace("*", "\\cdot ")
                anim_fn(
                    f, hist, xi_inicial=hist[0]["xi"], xd_inicial=hist[0]["xd"],
                    titulo=titulo_anim, guardar_gif=(guardar == "s"),
                    nombre_gif=nombre, expr_text=expr_text_simple
                )
