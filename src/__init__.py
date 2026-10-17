"""Los Podadores root package."""

import sys

# Prevent foreign Python version site-packages (e.g. Nix ROS Python 3.14 leaking into Python 3.12)
_curr_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
sys.path = [p for p in sys.path if "/lib/python" not in p or f"/lib/{_curr_ver}" in p]
