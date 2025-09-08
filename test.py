
Got it — here’s a drop-in Python script you can place in the repo root to run **only microstructure 237** (or any ID you pass). It filters `data/prelabelled_anns.json` → `data/anns.json` to just that one entry, then runs `main.py` steps for you.

Save it as `run_micro.py` in your `microlib/` folder.

```python
# run_micro.py
# Usage examples (from repo root):
#   python run_micro.py                  # defaults to --id 237 and runs: inpaint, slicegan, animate
#   python run_micro.py --id 237 --steps import inpaint slicegan animate
#   python run_micro.py --id 150 --steps inpaint slicegan          # different micro ID
#
# This script:
# 1) Reads data/prelabelled_anns.json
# 2) Filters it down to a single microstructure (--id)
# 3) Writes data/anns.json (backing up to data/anns.json.bak)
# 4) Runs main.py for the given pipeline steps

import argparse
import json
import pathlib
import re
import shutil
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
PRELABELLED = DATA_DIR / "prelabelled_anns.json"
ANNS = DATA_DIR / "anns.json"
BACKUP = DATA_DIR / "anns.json.bak"
MAIN = REPO_ROOT / "main.py"

def load_json(path: pathlib.Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        sys.exit(f"[ERROR] Not found: {path}. Did you download the repo data?")
    except json.JSONDecodeError as e:
        sys.exit(f"[ERROR] JSON decode failed for {path}: {e}")

def match_entry(entry: dict, micro_id: str) -> bool:
    """Return True if this annotation refers to the requested micro_id."""
    micro_id_l = micro_id.lower()
    candidate_keys = ("id","micro_id","microID","name","slug","micro_name","code","title","label","url")
    # direct field checks
    for k in candidate_keys:
        if k in entry:
            v = str(entry[k]).strip().lower()
            if v == micro_id_l or v == f"micro{micro_id_l}" or re.search(rf"(^|[^0-9]){re.escape(micro_id_l)}([^0-9]|$)", v):
                return True
    # scan all string fields as fallback
    for k, v in entry.items():
        if isinstance(v, str):
            vv = v.lower()
            if vv == micro_id_l or vv == f"micro{micro_id_l}" or re.search(rf"(^|[^0-9]){re.escape(micro_id_l)}([^0-9]|$)", vv):
                return True
    return False

def filter_anns(micro_id: str):
    src = PRELABELLED if PRELABELLED.exists() else ANNS
    if not src.exists():
        sys.exit(f"[ERROR] Neither {PRELABELLED} nor {ANNS} exists. Run data download/import first or check paths.")

    anns = load_json(src)
    print(f"[INFO] Loaded annotations from: {src}")

    # normalize list of entries
    if isinstance(anns, dict) and "items" in anns and isinstance(anns["items"], list):
        items = anns["items"]
        wrapper = True
    elif isinstance(anns, list):
        items = anns
        wrapper = False
    else:
        sys.exit("[ERROR] Unexpected JSON structure in annotations. Expected a list or a dict with 'items'.")

    # filter
    filtered = [e for e in items if isinstance(e, dict) and match_entry(e, micro_id)]
    if not filtered:
        # give one more chance: if micro_id numeric, try strict numeric equality on common integer keys
        try:
            mid_num = int(micro_id)
            numeric_keys = ("id","micro_id","microID")
            for e in items:
                if isinstance(e, dict):
                    for k in numeric_keys:
                        if k in e and isinstance(e[k], int) and e[k] == mid_num:
                            filtered.append(e)
                            break
        except ValueError:
            pass

    if not filtered:
        sys.exit(f"[ERROR] Could not find any entry matching micro '{micro_id}'. "
                 f"Open {src} and check which key holds the micro identifier.")

    out_obj = {"items": filtered} if wrapper else filtered

    # backup current anns.json if present
    if ANNS.exists():
        shutil.copy2(ANNS, BACKUP)
        print(f"[INFO] Backed up existing {ANNS} → {BACKUP}")

    ANNS.write_text(json.dumps(out_obj, indent=2), encoding="utf-8")
    print(f"[OK] Wrote filtered annotations for micro {micro_id} → {ANNS} "
          f"({1 if isinstance(out_obj, dict) else len(out_obj)} entry)")

def run_steps(steps):
    for step in steps:
        print(f"\n[RUN] python {MAIN.name} {step}")
        proc = subprocess.run([sys.executable, str(MAIN), step])
        if proc.returncode != 0:
            sys.exit(f"[ERROR] Step '{step}' failed with exit code {proc.returncode}. "
                     "Check the console logs above.")
        print(f"[OK] Step '{step}' completed.")

def main():
    parser = argparse.ArgumentParser(description="Filter microlib annotations to a single micro and run pipeline steps.")
    parser.add_argument("--id", default="237", help="Microstructure ID to run (default: 237).")
    parser.add_argument("--steps", nargs="+",
                        default=["inpaint", "slicegan", "animate"],
                        choices=["import", "preprocess", "inpaint", "slicegan", "animate"],
                        help="Pipeline steps to run after filtering (default: inpaint slicegan animate).")
    args = parser.parse_args()

    # sanity checks
    if not MAIN.exists():
        sys.exit(f"[ERROR] main.py not found at {MAIN}. Make sure you run this from the repo root.")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1) filter annotations
    filter_anns(args.id)

    # 2) run selected steps
    run_steps(args.steps)

    print("\n[DONE] All requested steps completed. "
          "Check data/final_images/ (after inpaint) and data/slicegan_runs/ (after slicegan/animate).")

if __name__ == "__main__":
    main()
```

### How to use

1. Put the file in your repo root:

```
microlib/
  ├─ main.py
  ├─ data/
  ├─ run_micro.py   <-- save here
  └─ ...
```

2. From the repo root, run:

```bash
# default (micro 237; steps: inpaint → slicegan → animate)
python run_micro.py

# include import (if you haven’t fetched assets yet)
python run_micro.py --steps import inpaint slicegan animate

# try a different micro
python run_micro.py --id 150 --steps inpaint slicegan animate
```

That’s it—no manual terminal JSON editing needed. If it errors saying it can’t find “237,” open `data/prelabelled_anns.json` once to see which field stores the ID (e.g., `id`, `name`, `slug`); the matcher is already flexible, but I can tweak it if your file format is different.
