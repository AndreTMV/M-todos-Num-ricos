from biseccion import biseccion
from math import e

xi = 50
xd = 70
g = 9.8
c = 15
t = 9


def f(m):
    return (((g*m)/c) * (1-e**-((c/m)*t))) - 35


try:
    raiz, iters, hist = biseccion(
        f, xi, xd, tol=.001, criterio="abs_fx", max_iter=200)
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
