Good catch — that error means our earlier “wrapper **function**” named `run_inpaint` shadowed what `main.py` expects: a **module** called `run_inpaint` that has a function like `inpaint_dataset(...)`.

Let’s swap the wrapper for a **lazy-module shim**, so `run_inpaint.inpaint_dataset` works again and nothing heavy is imported until it’s actually used.

## 1) Fix `src/inpainting/__init__.py`

Replace the entire file with this:

```python
# src/inpainting/__init__.py
# Expose submodules lazily so import-time doesn’t pull Tk/dotenv, etc.

import importlib
import importlib.util as _iu

class _LazyModule:
    def __init__(self, relmod):
        self._relmod = relmod
        self._m = None
    def _load(self):
        if self._m is None:
            self._m = importlib.import_module(self._relmod, __name__)
        return self._m
    def __getattr__(self, name):
        return getattr(self._load(), name)

# Prefer a submodule named "run_inpaint" if it exists; otherwise fall back to "inpaint".
if _iu.find_spec(__name__ + ".run_inpaint") is not None:
    run_inpaint = _LazyModule(".run_inpaint")
else:
    run_inpaint = _LazyModule(".inpaint")

# If the code references run_inpaint_scalebars as a callable, keep this helper:
def run_inpaint_scalebars(*args, **kwargs):
    mod = importlib.import_module(".inpaint_scalebars", __name__)
    return mod.run_inpaint_scalebars(*args, **kwargs)
```

## 2) (Recommended) Make `src/slicegan/__init__.py` lazy too

This prevents PlotOptiX/.NET from loading unless you explicitly animate.

```python
# src/slicegan/__init__.py
import importlib

class _LazyModule:
    def __init__(self, relmod):
        self._relmod = relmod
        self._m = None
    def _load(self):
        if self._m is None:
            self._m = importlib.import_module(self._relmod, __name__)
        return self._m
    def __getattr__(self, name):
        return getattr(self._load(), name)

run_slicegan = _LazyModule(".run_slicegan")
animations   = _LazyModule(".animations")   # imports PlotOptiX only if used
```

## 3) Keep the earlier small guard in `src/inpainting/util.py`

(You already patched Tk; keep this at the top so missing packages don’t crash.)

```python
# tkinter optional
try:
    from tkinter import image_types
except Exception:
    image_types = [('PNG','*.png'),('JPEG','*.jpg;*.jpeg'),('TIFF','*.tif;*.tiff')]

# dotenv optional
try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*a, **k): return False
```

## 4) Rerun headless (and fix the env var typo)

You typed `T_OPA_PLATFORM`; it should be `QT_QPA_PLATFORM`.

```bash
# from repo root
ls data/micrographs_png/000237.png  ||  ln -s micrographs_raw data/micrographs_png

export MPLBACKEND=Agg
export QT_QPA_PLATFORM=offscreen

python main.py inpaint
python main.py slicegan     # (skip animate for now)
```

## 5) Quick sanity check (optional)

Verify the shim now behaves like a module:

```bash
python - <<'PY'
from src.inpainting import run_inpaint
print("type:", type(run_inpaint))
print("has inpaint_dataset:", hasattr(run_inpaint, "inpaint_dataset"))
PY
```

You should see `has inpaint_dataset: True`. Then the pipeline should proceed without the AttributeError.

If the next error mentions a specific symbol, paste the first \~10 lines of the traceback and I’ll patch that spot, too.
