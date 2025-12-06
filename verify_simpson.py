
import os
import sys
# Add parent dir to path to import modules
sys.path.append("/Users/minervarivera/Desktop/M-todos-Num-ricos")

from simpson import metodo_simpson_iterativo, animar_simpson, _get_parabola_poly
from utils import make_func

def test_numerical():
    print("Testing numerical accuracy...")
    # Integral x^2 from 0 to 1 = 1/3
    f = lambda x: x**2
    val, iters, hist = metodo_simpson_iterativo(f, 0, 1, tol=1e-6)
    expected = 1/3.0
    print(f"Computed: {val}, Expected: {expected}")
    assert abs(val - expected) < 1e-6, "Numerical test failed!"
    print("Numerical test passed.")

def test_animation_generation():
    print("Testing animation generation...")
    f = lambda x: x**2
    # Mock history
    val, iters, hist = metodo_simpson_iterativo(f, 0, 1, max_iter=2)
    
    gif_name = "test_simpson.gif"
    if os.path.exists(gif_name):
        os.remove(gif_name)
        
    try:
        animar_simpson(f, hist, 0, 1, guardar_gif=True, nombre_gif=gif_name, fps=1, intervalo_ms=100)
        if os.path.exists(gif_name):
            print(f"GIF generated successfully: {gif_name}")
            os.remove(gif_name) # Cleanup
        else:
            print("GIF file was not created.")
    except Exception as e:
        print(f"Animation generation failed: {e}")

if __name__ == "__main__":
    test_numerical()
    # Skip animation test in headless env if needed, but we try it
    # Matplotlib backend might need config, but let's see.
    try:
        import matplotlib
        matplotlib.use('Agg') # Use non-interactive backend for testing
        test_animation_generation()
    except Exception as e:
        print(f"Skipping animation test due to backend issues: {e}")
