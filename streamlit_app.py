"""
Root-level entry point for Streamlit Cloud deployment.
Adds src/ to the Python path and delegates to the actual frontend app.
"""
import sys
import os
from pathlib import Path

# ── Make src/ importable from project root ──────────────────────────────────
_ROOT = Path(__file__).resolve().parent
_SRC  = _ROOT / "src"

for p in [str(_ROOT), str(_SRC)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Set working directory to repo root so relative file paths work
os.chdir(str(_ROOT))

# ── Run the actual Streamlit app ─────────────────────────────────────────────
# We import and re-execute the frontend app module directly
import importlib.util

_app_path = _SRC / "frontend" / "app.py"
spec = importlib.util.spec_from_file_location("app", str(_app_path))
module = importlib.util.module_from_spec(spec)

# Inject the frontend dir into path so `import utils` works inside app.py
_frontend = str(_SRC / "frontend")
if _frontend not in sys.path:
    sys.path.insert(0, _frontend)

spec.loader.exec_module(module)
