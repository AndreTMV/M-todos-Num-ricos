
import subprocess
import sys
import numpy as np

def test_sign_correctness():
    # We will import the function directly since it's just a unit test
    # But it's in a file that runs if __main__.
    # Better to create a small test wrapper that imports it.
    
    # Adding path to sys.path
    sys.path.append("/Users/minervarivera/Desktop/M-todos-Num-ricos")
    from simpson import metodo_simpson_iterativo
    
    f = lambda x: x**2
    
    # Test 1: Positive direction [0, 1]
    # Expected: 1/3
    val_pos, _, _ = metodo_simpson_iterativo(f, 0, 1, tol=1e-5)
    print(f"Int(0->1) x^2 = {val_pos}")
    
    if abs(val_pos - 1/3) < 1e-5:
        print("PASS: Positive direction.")
    else:
        print("FAIL: Positive direction.")
        
    # Test 2: Negative direction [1, 0]
    # Expected: -1/3
    val_neg, _, _ = metodo_simpson_iterativo(f, 1, 0, tol=1e-5)
    print(f"Int(1->0) x^2 = {val_neg}")
    
    if abs(val_neg - (-1/3)) < 1e-5:
        print("PASS: Negative direction.")
    else:
        print(f"FAIL: Negative direction. Expected -0.333..., got {val_neg}")

if __name__ == "__main__":
    test_sign_correctness()
