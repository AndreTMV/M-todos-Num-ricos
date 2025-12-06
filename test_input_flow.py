
import subprocess
import sys

def test_interactive_flow():
    print("Testing interactive flow (no tolerance)...")
    # Input: function, a, b, animate(n)
    # Note: Tolerance input should be skipped now.
    input_str = "x**2\n0\n1\nn\n"
    
    process = subprocess.Popen(
        [sys.executable, "simpson.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd="/Users/minervarivera/Desktop/M-todos-Num-ricos"
    )
    stdout, stderr = process.communicate(input=input_str)
    
    print("STDOUT:", stdout)
    if "Tolerancia" in stdout:
        print("FAILURE: Script still asked for tolerance.")
    else:
        print("SUCCESS: Tolerance prompt not found.")

if __name__ == "__main__":
    test_interactive_flow()
