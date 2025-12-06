
import subprocess
import sys

def test_explanation_output():
    print("Testing explanation output...")
    # Run simpson.py with some dummy input
    # Needs: f(x), a, b, tol, animar(n)
    input_str = "x**2\n0\n1\n1e-4\nn\n"
    
    process = subprocess.Popen(
        [sys.executable, "simpson.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd="/Users/minervarivera/Desktop/M-todos-Num-ricos"
    )
    stdout, stderr = process.communicate(input=input_str)
    
    if "El método de Simpson 1/3 aproxima la integral definida" in stdout:
        print("SUCCESS: Explanation text found in output.")
    else:
        print("FAILURE: Explanation text NOT found.")
        print("Output snippet:", stdout[:500])

if __name__ == "__main__":
    test_explanation_output()
