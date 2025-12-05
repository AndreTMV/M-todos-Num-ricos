from utils import make_func
from biseccion import biseccion, animar_biseccion
from posicion_falsa import posicion_falsa, animar_posicion_falsa
from secante import secante, animar_secante
from newton import newton_raphson, animar_newton
from punto_fijo import punto_fijo, animar_punto_fijo

if __name__ == "__main__":
    print("=== Métodos de Raíces (Bisección, Posición Falsa, Secante, Newton y Punto fijo) ===")
    expr = input("Ingresa f(x), p.ej.: x**3 - 2*x - 5\nf(x) = ").strip()
    f = make_func(expr)

    metodo = input(
        "Elige método: (b)isección / (p)osición falsa / (s)ecante / (n)ewton / (f)ijo (Punto fijo): ").strip().lower()
    if metodo not in {"b", "p", "s", "n", "f"}:
        raise ValueError("Opción inválida.")

    # Entradas según método
    if metodo in {"n", "f"}:
        x0 = float(input("Ingresa x0: "))
    else:
        xi = float(input("Ingresa xi: "))
        xd = float(input("Ingresa xd: "))

    # Criterios de paro
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

    # Variables para animación
    anim_fn = None
    nombre_gif = None
    titulo_anim = None
    expr_text_pf = None  # expr_text específico para punto fijo
    g = None             # g para punto fijo

    try:
        if metodo == "b":
            raiz, iters, hist = biseccion(
                f, xi, xd, tol=tol, criterio=criterio, max_iter=200)
            titulo_anim = f"Método de Bisección | f(x) = {expr}"
            anim_fn = animar_biseccion
            nombre_gif = "biseccion.gif"

        elif metodo == "p":
            usar_ill = input(
                "¿Usar variante Illinois para Posición Falsa? (s/n): ").strip().lower() == "s"
            raiz, iters, hist = posicion_falsa(
                f, xi, xd, tol=tol, criterio=criterio, max_iter=200, usar_illinois=usar_ill
            )
            titulo_anim = f"Método de Posición Falsa{' (Illinois)' if usar_ill else ''} | f(x) = {expr}"
            anim_fn = animar_posicion_falsa
            nombre_gif = "posicion_falsa.gif"

        elif metodo == "s":
            raiz, iters, hist = secante(
                f, xi, xd, tol=tol, criterio=criterio, max_iter=200)
            titulo_anim = f"Método de la Secante | f(x) = {expr}"
            anim_fn = animar_secante
            nombre_gif = "secante.gif"

        elif metodo == "n":
            # Newton usa x0
            raiz, iters, hist = newton_raphson(
                f, x0, tol=tol, criterio=criterio, max_iter=200)
            titulo_anim = f"Método de Newton–Raphson | f(x) = {expr}"
            anim_fn = animar_newton
            nombre_gif = "newton.gif"

        elif metodo == "f":
            gexpr = input("Ingresa g(x) tal que x = g(x):\ng(x) = ").strip()
            g = make_func(gexpr)

            raiz, iters, hist = punto_fijo(
                f, g, x0, tol=tol, criterio=criterio, max_iter=200)
            titulo_anim = f"Método de Punto Fijo | f(x) = {expr} ; g(x) = {gexpr}"

            # Panel LaTeX mostrando f y g
            expr_tex = expr.replace("*", "\\cdot ")
            gexpr_tex = gexpr.replace("*", "\\cdot ")
            expr_text_pf = f"f(x) = {expr_tex} \\quad g(x) = {gexpr_tex}"

            anim_fn = animar_punto_fijo
            nombre_gif = "punto_fijo.gif"

    except Exception as e:
        print(f"\nError: {e}")
    else:
        # Impresión de historial
        print("\n--- Iteraciones ---")
        for h in hist:
            i = h["i"]
            print(
                f"Iteracion#{i:>3} | "
                f"Xi={h['xi']:.7f},f(Xi)={h['fxi']:.7f} | "
                f"Xd={h['xd']:.7f},f(Xd)={h['fxd']:.7f} | "
                f"Xm={h['xm']:.7f},f(Xm)={h['fxm']:.7f}"
            )

        # Resultado
        print("\n=== Resultado ===")
        print(f"Raíz aproximada: {raiz:.10f}")
        print(f"Iteraciones usadas: {iters}")
        print(f"f(raíz) ≈ {f(raiz):.10g}")
        print(f"Criterio: {criterio} con tolerancia {tol}")

        # Animación
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
                # Otros métodos: sí usan xi_inicial/xd_inicial
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
