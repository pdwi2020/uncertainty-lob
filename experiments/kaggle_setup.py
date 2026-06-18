"""Run this cell first in a Kaggle notebook or script session.

Installs missing packages and sets up the src/ path.
Copy-paste into a Kaggle code cell, or run as:
    !python experiments/kaggle_setup.py
"""

import subprocess
import sys
import os

PACKAGES = [
    "torchcde",
    "torchdiffeq",
    "hmmlearn",
    "mapie",
]

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

print("Installing packages...")
for pkg in PACKAGES:
    try:
        __import__(pkg.replace("-", "_"))
        print(f"  {pkg} already installed")
    except ImportError:
        print(f"  Installing {pkg}...")
        install(pkg)

# Add project root to path (when running from /kaggle/working/)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
print(f"\nProject root: {project_root}")
print("Setup complete.")
