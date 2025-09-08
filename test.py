def run_steps(steps):
    env = dict(os.environ)  # add at top of file: import os
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    for step in steps:
        print(f"\n[RUN] main.py {step} (headless)")
        rc = subprocess.run([sys.executable, "main.py", step], env=env).returncode
        if rc != 0:
            sys.exit(f"[ERROR] Step '{step}' failed with exit code {rc}.")
        print(f"[OK] Step '{step}' completed.")



Good catch — the repo is importing `tkinter` directly inside `src/inpainting/util.py`, so even with a headless Matplotlib it still blows up. Let’s patch that line so Tk isn’t required at all.

## One-shot patch script (recommended)

Save this as `patch_tk_shim.py` in your **microlib** repo root and run it once:

```python
# patch_tk_shim.py
# Replaces "from tkinter import image_types" in src/inpainting/util.py
# with a headless-safe shim that works even if Tk isn't installed.

from pathlib import Path
import sys

repo = Path(__file__).resolve().parent
util_py = repo / "src" / "inpainting" / "util.py"

if not util_py.exists():
    sys.exit(f"[ERROR] Not found: {util_py}")

text = util_py.read_text(encoding="utf-8")

needle = "from tkinter import image_types"
shim = """# --- Headless shim for image_types (avoid hard tkinter dependency)
try:
    import tkinter as _tk  # optional; not required in headless
except Exception:
    _tk = None

def _image_types_impl():
    # Try to ask Tk for registered image types; fall back to common formats
    if _tk is not None:
        try:
            _root = _tk.Tk()
            types = tuple(_root.tk.call('image', 'types'))
            _root.destroy()
            return types
        except Exception:
            pass
    return ('png','gif','jpeg','bmp','tiff','ppm','pgm')

class image_types:  # supports both calling and iterating
    def __call__(self):
        return _image_types_impl()
    def __iter__(self):
        return iter(_image_types_impl())
# --- End shim ---
"""

if needle not in text:
    print("[INFO] 'from tkinter import image_types' not found; leaving file unchanged.")
    print(f"[INFO] Please open {util_py} to confirm it’s already patched.")
    sys.exit(0)

new = text.replace(needle, shim)
util_py.write_text(new, encoding="utf-8")
print("[OK] Patched:", util_py)
```



Yep—this is a Windows-only stdlib module. Something in your import chain is doing:

```python
from msilib.schema import Error
```

On Linux (your Docker), `msilib` doesn’t exist, so you get that error. We can unblock you in two safe ways:

---

# ✅ Fix A (recommended first): stop importing GUI/packaging stuff at import-time

Your trace still shows `src/inpainting/__init__.py` importing `util` eagerly. Replace that file with a **lazy wrapper** so `main.py import` doesn’t pull Tk/dotenv/msilib at import-time.

**Overwrite `src/inpainting/__init__.py` with exactly this:**

```python
# src/inpainting/__init__.py
# Lazy wrappers so importing this package doesn't pull GUI/Tk/dotenv/msilib deps.

def run_inpaint(*args, **kwargs):
    from .inpaint import run_inpaint as _run
    return _run(*args, **kwargs)

def run_inpaint_scalebars(*args, **kwargs):
    from .inpaint_scalebars import run_inpaint_scalebars as _run
    return _run(*args, **kwargs)
```

Then hard-guard the optional imports in `util.py` (keeps you headless-safe even when you do `inpaint`):

**Edit `src/inpainting/util.py`:**

```python
# --- Tk shim (avoid hard dependency) ---
try:
    from tkinter import image_types  # some environments don't have tkinter
except Exception:
    image_types = [('PNG', '*.png'), ('JPEG', '*.jpg;*.jpeg'), ('TIFF', '*.tif;*.tiff')]

# --- dotenv shim (optional) ---
try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*args, **kwargs):
        return False
```

(Keep any other Tk imports behind try/except too if they exist, e.g., `filedialog`, etc.)

Now run headless:

```bash
export MPLBACKEND=Agg
export QT_QPA_PLATFORM=offscreen

python main.py import
cp data/prelabelled_anns.json data/anns.json
python run_micro.py --query "000237.png" --steps inpaint slicegan animate
```

---

# ✅ Fix B (quick universal shim): stub `msilib` locally

If the error persists (some third-party lib insists on importing `msilib`), add a tiny **local stub** so the import succeeds. This doesn’t affect your training/rendering.

From your repo root:

```bash
mkdir -p msilib
printf "" > msilib/__init__.py
cat > msilib/schema.py <<'PY'
class Error(Exception):
    pass
PY
```

Because the current working directory is on `sys.path`, Python will import this stub instead of the (nonexistent) Windows module.

Then rerun:

```bash
export MPLBACKEND=Agg
export QT_QPA_PLATFORM=offscreen

python main.py import
cp data/prelabelled_anns.json data/anns.json
python run_micro.py --query "000237.png" --steps inpaint slicegan animate
```

---

## (Optional) Find the offender

If you’re curious which file was importing `msilib`, this finds it:

```bash
grep -Rni "msilib" src $(python -c "import site,sys; print(' '.join(site.getsitepackages()+[site.getusersitepackages()]))") 2>/dev/null | head -n 20
```

If it’s inside the repo, we’ll patch that file instead of stubbing.

---

### Quick recap

1. Make `inpainting/__init__.py` lazy (very important).
2. Guard Tk/dotenv in `util.py`.
3. If needed, add the `msilib` stub.
4. Run headless with `MPLBACKEND=Agg` and `QT_QPA_PLATFORM=offscreen`.

This combo removes all GUI/Windows-only dependencies from your path so you can finish **Option A** and run micro **000237** end-to-end.
