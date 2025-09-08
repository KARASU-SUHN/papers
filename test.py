
```python
# run_micro.py  (search + inspect + filter)
# Examples:
#   python run_micro.py --inspect
#   python run_micro.py --search 237
#   python run_micro.py --query "spinodal"            # filter by substring match
#   python run_micro.py --id 237                      # strict numeric/text id match
#   python run_micro.py --query "237" --steps inpaint slicegan animate
#   python run_micro.py --id 237 --steps import inpaint slicegan animate

import argparse, json, pathlib, re, shutil, subprocess, sys
from typing import Any, Dict, Iterable, List, Tuple, Union

REPO_ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
PRELABELLED = DATA_DIR / "prelabelled_anns.json"
ANNS = DATA_DIR / "anns.json"
BACKUP = DATA_DIR / "anns.json.bak"
MAIN = REPO_ROOT / "main.py"
LIST_KEYS = ("items", "annotations", "microstructures", "anns", "data")

def load_json(path: pathlib.Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        sys.exit(f"[ERROR] Not found: {path}.")
    except json.JSONDecodeError as e:
        sys.exit(f"[ERROR] JSON decode failed for {path}: {e}")

def detect_container(obj):
    # returns (kind, accessor)
    if isinstance(obj, list):
        return "list", obj
    if isinstance(obj, dict):
        for k in LIST_KEYS:
            if k in obj and isinstance(obj[k], list):
                return "dictlist", (k, obj[k])
        vals = list(obj.values())
        if vals and sum(isinstance(v, dict) for v in vals) / len(vals) > 0.7:
            return "dictmap", obj
    return None, None

def stringify_id(v):
    try:
        return str(int(v))
    except Exception:
        return str(v).strip()

def iter_entries(kind, acc) -> Iterable[Tuple[Union[int, str], Dict[str, Any]]]:
    """Yield (key, entry_dict) across any supported container shape.
       key is index for list/dictlist; original key for dictmap.
    """
    if kind == "list":
        for i, e in enumerate(acc):
            if isinstance(e, dict):
                yield i, e
    elif kind == "dictlist":
        _, lst = acc
        for i, e in enumerate(lst):
            if isinstance(e, dict):
                yield i, e
    elif kind == "dictmap":
        for k, v in acc.items():
            if isinstance(v, dict):
                yield k, v

CAND_KEYS = ("id","micro_id","microID","name","slug","code","title","label","url")

def entry_label(entry: Dict[str, Any]) -> str:
    parts = []
    for k in CAND_KEYS:
        if k in entry:
            val = entry[k]
            if isinstance(val, (str, int)):
                parts.append(f"{k}={val}")
    # shorten
    s = ", ".join(parts) if parts else "(no common id/name fields)"
    return (s[:200] + "…") if len(s) > 200 else s

def match_entry_by_id(entry: Dict[str, Any], target_id: str) -> bool:
    t = target_id.lower()
    for k in CAND_KEYS:
        if k in entry:
            v = str(entry[k]).strip().lower()
            if v == t or v == f"micro{t}" or re.search(rf"(^|[^0-9]){re.escape(t)}([^0-9]|$)", v):
                return True
    try:
        ti = int(target_id)
        for k in ("id","micro_id","microID"):
            if k in entry and isinstance(entry[k], int) and entry[k] == ti:
                return True
    except ValueError:
        pass
    return False

def match_entry_by_query(entry: Dict[str, Any], query: str) -> bool:
    q = query.lower()
    for v in entry.values():
        if isinstance(v, (str, int, float)):
            if q in str(v).lower():
                return True
    return False

def write_filtered(src_obj, kind, acc, selected: List[Tuple[Union[int,str], Dict[str,Any]]]):
    if not selected:
        sys.exit("[ERROR] Nothing selected to write.")
    # backup
    if ANNS.exists():
        shutil.copy2(ANNS, BACKUP)
        print(f"[INFO] Backed up {ANNS} → {BACKUP}")

    if kind == "list":
        out = [e for _, e in selected]
    elif kind == "dictlist":
        key, _ = acc
        out = dict(src_obj)  # shallow copy of wrapper
        out[key] = [e for _, e in selected]
    elif kind == "dictmap":
        out = {k: e for k, e in selected}
    else:
        sys.exit("[ERROR] Unknown container kind after selection.")

    ANNS.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    count = len(selected)
    print(f"[OK] Wrote filtered annotations → {ANNS} ({count} entry{'s' if count!=1 else ''})")

def run_steps(steps: List[str]):
    for step in steps:
        print(f"\n[RUN] python {MAIN.name} {step}")
        rc = subprocess.run([sys.executable, str(MAIN), step]).returncode
        if rc != 0:
            sys.exit(f"[ERROR] Step '{step}' failed with exit code {rc}.")
        print(f"[OK] Step '{step}' completed.")

def main():
    ap = argparse.ArgumentParser(description="Search/inspect microlib annotations and run a subset pipeline.")
    ap.add_argument("--inspect", action="store_true", help="Print top-level structure and a few example entries, then exit.")
    ap.add_argument("--list-ids", action="store_true", help="List detected candidate IDs (best-effort), then exit.")
    ap.add_argument("--search", type=str, help="Substring search across all fields. Prints matches then exits.")
    ap.add_argument("--id", type=str, help="Strict match by id/name/slug/etc. (e.g., 237).")
    ap.add_argument("--query", type=str, help="Filter by substring (e.g., '237' or 'ferrite'). If multiple, keeps the first match.")
    ap.add_argument("--steps", nargs="+", default=["inpaint", "slicegan", "animate"],
                    choices=["import", "preprocess", "inpaint", "slicegan", "animate"])
    args = ap.parse_args()

    if not MAIN.exists():
        sys.exit(f"[ERROR] main.py not found at {MAIN}. Run from repo root.")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    src_path = PRELABELLED if PRELABELLED.exists() else ANNS
    src = load_json(src_path)
    kind, acc = detect_container(src)
    if kind is None:
        sys.exit("[ERROR] Unexpected annotations structure. Try --inspect and share the output.")

    print(f"[INFO] Loaded from {src_path} (kind={kind}).")

    if args.inspect:
        print("[INSPECT] First 3 entries (label fields):")
        for i, (k, e) in enumerate(iter_entries(kind, acc)):
            print(f"  - key={k!r}: {entry_label(e)}")
            if i >= 2: break
        return

    if args.list_ids:
        ids = []
        for k, e in iter_entries(kind, acc):
            # collect a single best id/name
            best = None
            for cand in ("id","micro_id","microID","name","slug","code","title","label"):
                if cand in e:
                    best = stringify_id(e[cand])
                    break
            ids.append(best if best is not None else f"[key={k}]")
        print(f"[INFO] Detected {len(ids)} candidates (showing up to 200):")
        for i, v in enumerate(ids[:200], 1):
            print(f"  {i:3d}. {v}")
        return

    if args.search:
        q = args.search.strip().lower()
        print(f"[SEARCH] matches containing '{q}':")
        shown = 0
        for k, e in iter_entries(kind, acc):
            label = entry_label(e).lower()
            if q in label or any(isinstance(v, str) and q in v.lower() for v in e.values()):
                print(f"  - key={k!r}: {entry_label(e)}")
                shown += 1
                if shown >= 200:
                    print("  [truncated at 200 matches]")
                    break
        if shown == 0:
            print("  (no matches)")
        return

    # Selection: by --id if provided; else by --query (substring); else default to "237" as id
    selected: List[Tuple[Union[int,str], Dict[str,Any]]] = []
    target_id = args.id if args.id else "237"
    if args.id:
        for k, e in iter_entries(kind, acc):
            if match_entry_by_id(e, args.id):
                selected.append((k, e))
                break
        if not selected:
            sys.exit(f"[ERROR] Could not find any entry matching '--id {args.id}'. Try --search '{args.id}'.")
    else:
        q = args.query if args.query else "237"
        for k, e in iter_entries(kind, acc):
            if match_entry_by_query(e, q):
                selected.append((k, e))
                break
        if not selected:
            sys.exit(f"[ERROR] Could not find any entry with substring '{q}'. "
                     f"Try: python run_micro.py --inspect  and  python run_micro.py --search {q}")

    write_filtered(src, kind, acc, selected)
    run_steps(args.steps)
    print("\n[DONE] Check data/final_images/ and data/slicegan_runs/.")

if __name__ == "__main__":
    main()
```

### How to use (quick)

1. Inspect structure and see sample labels:

```
python run_micro.py --inspect
```

2. Search for anything containing “237” across all fields:

```
python run_micro.py --search 237
```

This prints lines like:

```
- key=...: id=..., name=..., slug=..., url=...
```

Pick the unique string that clearly identifies your target (maybe it’s called “micro237”, or the number is only in the URL, or the dataset uses names instead of numbers).

3. Run with the thing that matches:

* If you saw a clean ID/name like `name=Micro 237`, try:

```
python run_micro.py --id 237
```

* If the number only appears inside another field (e.g., part of a URL or title), use a substring query:

```
python run_micro.py --query "237"
```

This writes a filtered `data/anns.json` and then runs `inpaint → slicegan → animate` for that single entry.

If it still can’t find the entry, please run:

```
python run_micro.py --inspect
python run_micro.py --search 237
```

and paste the printed lines (just a few). I’ll adapt the matcher to your file immediately.


