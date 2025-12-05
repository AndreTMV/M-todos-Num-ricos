from utils import make_func
from biseccion import biseccion, animar_biseccion
from posicion_falsa import posicion_falsa, animar_posicion_falsa
from secante import secante, animar_secante
from newton import newton_raphson, animar_newton
from punto_fijo import punto_fijo, animar_punto_fijo

"""
Módulo: main.py
---------------

Interfaz de línea de comandos para ejecutar y visualizar **métodos de búsqueda de raíces**:

- Bisección
- Posición Falsa (con opción Illinois)
- Secante
- Newton–Raphson
- Punto Fijo

Flujo general
~~~~~~~~~~~~~
1) Se solicita al usuario la función f(x) como texto en sintaxis de Python (p.ej. `x**3 - 2*x - 5`)
   y se compila con `make_func`.
2) Se elige el método y se piden los datos de entrada: (xi, xd) para métodos con intervalo o x0 para
   métodos abiertos (Newton, Punto Fijo).
3) Se selecciona el **criterio de paro** y la **tolerancia**.
4) Se ejecuta el método y se imprime el **historial** de iteraciones con un esquema unificado:
   - "xi", "fxi": punto izquierdo o x_{k-1}
   - "xd", "fxd": punto derecho o x_{k}
   - "xm", "fxm": punto medio/nuevo x_{k+1}
5) Se muestra el **resultado** (raíz aproximada, iteraciones y f(raíz)).
6) Opcionalmente, se lanza una **animación** (y se puede guardar en GIF).

Criterios de paro
~~~~~~~~~~~~~~~~~
- `abs_fx`:     |f(xm)| <= tol
- `abs`:        |xm - xm_prev| <= tol
- `rel`:        |xm - xm_prev| / |xm| <= tol
- `pct`:        100*|xm - xm_prev| / |xm| <= tol   (tol se interpreta como porcentaje)

Notas
~~~~~
- La animación de **Punto Fijo** necesita también g(x). Aquí se pide `g(x)` y se pasa a su animador.
- Para evitar problemas de escape de LaTeX, se reemplaza `*` por `\\cdot` en los textos de fórmulas
  mostrados en paneles LaTeX dentro de las animaciones.
- Este archivo no valida dominios ni discontinuidades; la responsabilidad de una buena
  elección de datos (intervalos y semillas) recae en el usuario.
"""

if __name__ == "__main__":
    # Encabezado
    print("=== Métodos de Raíces (Bisección, Posición Falsa, Secante, Newton y Punto fijo) ===")

    # 1) Entrada de f(x) como texto y compilación a callable con make_func
    expr = input("Ingresa f(x), p.ej.: x**3 - 2*x - 5\nf(x) = ").strip()
    f = make_func(expr)

    # 2) Selección de método
    metodo = input(
        "Elige método: (b)isección / (p)osición falsa / (s)ecante / (n)ewton / (f)ijo (Punto fijo): "
    ).strip().lower()
    if metodo not in {"b", "p", "s", "n", "f"}:
        raise ValueError("Opción inválida.")

    # 3) Lectura de parámetros según el método elegido
    #    - Newton y Punto Fijo: una sola semilla x0
    #    - Bisección, Posición Falsa y Secante: extremos/semillas xi, xd
    if metodo in {"n", "f"}:
        x0 = float(input("Ingresa x0: "))
    else:
        xi = float(input("Ingresa xi: "))
        xd = float(input("Ingresa xd: "))

    # 4) Selección de criterio de paro y tolerancia
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

    # Variables para animación (se determinan según el método)
    anim_fn = None          # referencia a la función de animación apropiada
    nombre_gif = None       # nombre por defecto del GIF según el método
    titulo_anim = None      # título a mostrar en la animación
    # cadena LaTeX-friendly para Punto Fijo (incluye f y g)
    expr_text_pf = None
    g = None                # g(x) para Punto Fijo

    try:
        # 5) Ejecución del método seleccionado y preparación de metadatos para animación
        if metodo == "b":
            # Bisección requiere intervalo [xi, xd] con f(xi)*f(xd) < 0 (no validado aquí)
            raiz, iters, hist = biseccion(
                f, xi, xd, tol=tol, criterio=criterio, max_iter=200)
            titulo_anim = f"Método de Bisección | f(x) = {expr}"
            anim_fn = animar_biseccion
            nombre_gif = "biseccion.gif"

        elif metodo == "p":
            # Posición Falsa (opcionalmente con corrección Illinois)
            usar_ill = input(
                "¿Usar variante Illinois para Posición Falsa? (s/n): ").strip().lower() == "s"
            raiz, iters, hist = posicion_falsa(
                f, xi, xd, tol=tol, criterio=criterio, max_iter=200, usar_illinois=usar_ill
            )
            titulo_anim = f"Método de Posición Falsa{' (Illinois)' if usar_ill else ''} | f(x) = {expr}"
            anim_fn = animar_posicion_falsa
            nombre_gif = "posicion_falsa.gif"

        elif metodo == "s":
            # Secante usa dos semillas x0=xi y x1=xd (no requiere intervalos con cambio de signo)
            raiz, iters, hist = secante(
                f, xi, xd, tol=tol, criterio=criterio, max_iter=200)
            titulo_anim = f"Método de la Secante | f(x) = {expr}"
            anim_fn = animar_secante
            nombre_gif = "secante.gif"

        elif metodo == "n":
            # Newton–Raphson usa una sola semilla x0
            raiz, iters, hist = newton_raphson(
                f, x0, tol=tol, criterio=criterio, max_iter=200)
            titulo_anim = f"Método de Newton–Raphson | f(x) = {expr}"
            anim_fn = animar_newton
            nombre_gif = "newton.gif"

        elif metodo == "f":
            # Punto Fijo requiere especificar g(x) tal que x = g(x)
            gexpr = input("Ingresa g(x) tal que x = g(x):\ng(x) = ").strip()
            g = make_func(gexpr)

            raiz, iters, hist = punto_fijo(
                f, g, x0, tol=tol, criterio=criterio, max_iter=200)
            titulo_anim = f"Método de Punto Fijo | f(x) = {expr} ; g(x) = {gexpr}"

            # Texto LaTeX para paneles (reemplaza '*' por '\cdot' para mejor tipografía)
            expr_tex = expr.replace("*", "\\cdot ")
            gexpr_tex = gexpr.replace("*", "\\cdot ")
            expr_text_pf = f"f(x) = {expr_tex} \\quad g(x) = {gexpr_tex}"

            anim_fn = animar_punto_fijo
            nombre_gif = "punto_fijo.gif"

    except Exception as e:
        # Manejo simple de errores (por ejemplo, entradas inválidas o fallos en el método)
        print(f"\nError: {e}")
    else:
        # 6) Impresión del historial de iteraciones con formato uniforme
        print("\n--- Iteraciones ---")
        for h in hist:
            i = h["i"]
            print(
                f"Iteracion#{i:>3} | "
                f"Xi={h['xi']:.7f},f(Xi)={h['fxi']:.7f} | "
                f"Xd={h['xd']:.7f},f(Xd)={h['fxd']:.7f} | "
                f"Xm={h['xm']:.7f},f(Xm)={h['fxm']:.7f}"
            )

        # 7) Resumen del resultado
        print("\n=== Resultado ===")
        print(f"Raíz aproximada: {raiz:.10f}")
        print(f"Iteraciones usadas: {iters}")
        print(f"f(raíz) ≈ {f(raiz):.10g}")
        print(f"Criterio: {criterio} con tolerancia {tol}")

        # 8) Animación opcional: en Punto Fijo la firma es distinta (requiere g y expr_text_pf)
        resp = input("\n¿Deseas ver la animación? (s/n): ").strip().lower()
        if resp == "s":
            guardar = input("¿Guardar GIF? (s/n): ").strip().lower()
            if guardar == "s":
                nombre = input(
                    f"Nombre de archivo GIF (ej. {nombre_gif}): ").strip() or nombre_gif
            else:
                nombre = nombre_gif

            if metodo == "f":
                # Punto Fijo: NO pasar xi_inicial/xd_inicial (no existen en la firma)
                anim_fn(
                    f=f,
                    g=g,
                    hist=hist,
                    titulo=titulo_anim,
                    guardar_gif=(guardar == "s"),
                    nombre_gif=nombre,
                    expr_text=expr_text_pf,
                )
            else:
                # Otros métodos: firma común con xi_inicial/xd_inicial para contextualizar la animación
                expr_text_simple = expr.replace("*", "\\cdot ")
                anim_fn(
                    f,
                    hist,
                    xi_inicial=hist[0]["xi"],
                    xd_inicial=hist[0]["xd"],
                    titulo=titulo_anim,
                    guardar_gif=(guardar == "s"),
                    nombre_gif=nombre,
                    expr_text=expr_text_simple,
                )

