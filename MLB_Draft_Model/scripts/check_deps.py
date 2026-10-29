import subprocess, sys
result = subprocess.run(["python3", "-c", "import numpy; print('numpy', numpy.__version__); import sklearn; print('sklearn', sklearn.__version__)"], capture_output=True, text=True, cwd="/Users/brendanwaterval/Projects/vt_baseball/MLB_Draft_Model")
print("stdout:", result.stdout)
print("stderr:", result.stderr)
print("returncode:", result.returncode)
# Also check for venv
import os
venv_bin = "/Users/brendanwaterval/Projects/vt_baseball/MLB_Draft_Model/.venv/bin/python3"
if os.path.exists(venv_bin):
    print("venv exists")
    result2 = subprocess.run([venv_bin, "-c", "import numpy; print('numpy', numpy.__version__); import sklearn; print('sklearn', sklearn.__version__)"], capture_output=True, text=True)
    print("venv stdout:", result2.stdout)
    print("venv stderr:", result2.stderr)
else:
    print("venv not found")
