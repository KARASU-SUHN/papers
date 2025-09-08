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

Run it:

```bash
python patch_tk_shim.py
```

You should see `[OK] Patched: src/inpainting/util.py`.

---

## Now run the pipeline (still headless-safe)

If you already created `headless_run.py` earlier, use it; otherwise you can run `main.py` normally now that the Tk import is gone.

```bash
# 1) Download micrographs into repo's data/
python headless_run.py import
# or: python main.py import

# 2) Skip the GUI using the authors’ labels
cp data/prelabelled_anns.json data/anns.json    # (Windows PowerShell: Copy-Item ...)

# 3) Run ONLY the entry that ends with 000237.png
python run_micro.py --query "000237.png" --steps inpaint slicegan animate
```

That should produce:

* Inpainted image for that entry under `data/final_images/`
* A SliceGAN run folder under `data/slicegan_runs/...` with weights + GIF/MP4.

---

### If anything else still mentions `tkinter`

Run this quick search to spot any other hard Tk imports inside the repo:

```bash
grep -RniE "tkinter|ImageTk|TkAgg" src || true
```

If something shows up (besides the line we just patched), tell me the file/line and I’ll give you a tiny shim for that too.
