# run_micro.py  (robust version)
# Usage:
#   python run_micro.py                       # defaults to --id 237, steps: inpaint slicegan animate
#   python run_micro.py --id 237 --steps import inpaint slicegan animate
#   python run_micro.py --id 150 --steps inpaint slicegan
#   python run_micro.py --list-ids            # just show detected IDs and exit

import argparse, json, pathlib, re, shutil, subprocess, sys

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
        sys.exit(f"[ERROR] Not found: {path}. Did you download/extract the data?")
    except json.JSONDecodeError as e:
        sys.exit(f"[ERROR] JSON decode failed for {path}: {e}")

def detect_container(obj):
    """
    Return a tuple (kind, accessor) where:
      kind in {"list", "dictmap", "dictlist"}
      accessor:
        - for list: returns the list itself
        - for dictmap: returns dict (id->entry)
        - for dictlist: (keyname, listref)
    """
    # raw list?
    if isinstance(obj, list):
        return "list", obj

    if isinstance(obj, dict):
        # dict with a list under a known key?
        for k in LIST_KEYS:
            if k in obj and isinstance(obj[k], list):
                return "dictlist", (k, obj[k])
        # dict mapping id->entry?
        # Heuristic: all values are dicts (or most)
        vals = list(obj.values())
        if vals and sum(isinstance(v, dict) for v in vals) / len(vals) > 0.7:
            return "dictmap", obj

    return None, None

def stringify_id(v):
    try:
        return str(int(v))
    except Exception:
        return str(v).strip()

def match_entry(entry: dict, target_id: str) -> bool:
    t = target_id.lower()
    candidate_keys = ("id","micro_id","microID","name","slug","micro_name","code","title","label","url")
    # direct field checks
    for k in candidate_keys:
        if k in entry:
            v = str(entry[k]).strip().lower()
            if v == t or v == f"micro{t}" or re.search(rf"(^|[^0-9]){re.escape(t)}([^0-9]|$)", v):
                return True
    # scan all strings as fallback
    for v in entry.values():
        if isinstance(v, str):
            vv = v.lower()
            if vv == t or vv == f"micro{t}" or re.search(rf"(^|[^0-9]){re.escape(t)}([^0-9]|$)", vv):
                return True
    # strict numeric check for common integer keys
    try:
        ti = int(target_id)
        for k in ("id","micro_id","microID"):
            if k in entry and isinstance(entry[k], int) and entry[k] == ti:
                return True
    except ValueError:
        pass
    return False

def extract_all_ids(kind, accessor):
    ids = []
    if kind == "list":
        for e in accessor:
            if isinstance(e, dict):
                for k in ("id","micro_id","microID","name","slug","code","title","label"):
                    if k in e:
                        ids.append(stringify_id(e[k]))
                        break
    elif kind == "dictlist":
        _, lst = accessor
        for e in lst:
            if isinstance(e, dict):
                for k in ("id","micro_id","microID","name","slug","code","title","label"):
                    if k in e:
                        ids.append(stringify_id(e[k]))
                        break
    elif kind == "dictmap":
        ids = [stringify_id(k) for k in accessor.keys()]
    return sorted(set(ids), key=lambda x: (len(x), x))

def filter_and_write(src_obj, micro_id: str):
    kind, acc = detect_container(src_obj)
    if kind is None:
        raise SystemExit("[ERROR] Unexpected JSON structure in annotations.\n"
                         "Tip: run with --list-ids to see what I can detect.")

    # backup existing anns.json
    if ANNS.exists():
        shutil.copy2(ANNS, BACKUP)
        print(f"[INFO] Backed up existing {ANNS} → {BACKUP}")

    out_obj = None

    if kind == "list":
        filtered = [e for e in acc if isinstance(e, dict) and match_entry(e, micro_id)]
        if not filtered:
            raise SystemExit(f"[ERROR] Could not find any entry matching '{micro_id}'. "
                             f"Try --list-ids to view candidates.")
        out_obj = filtered  # preserve top-level list

    elif kind == "dictlist":
        key, lst = acc
        filtered = [e for e in lst if isinstance(e, dict) and match_entry(e, micro_id)]
        if not filtered:
            raise SystemExit(f"[ERROR] Could not find any entry matching '{micro_id}'. "
                             f"Try --list-ids to view candidates.")
        out_obj = dict(src_obj)  # shallow copy
        out_obj[key] = filtered  # preserve same wrapping key

    elif kind == "dictmap":
        # try by exact key first
        key_hit = None
        # allow both '237' and int 237 keys
        for k in list(acc.keys()):
            if stringify_id(k) == stringify_id(micro_id):
                key_hit = k
                break
        if key_hit is not None:
            out_obj = {key_hit: acc[key_hit]}
        else:
            # fall back: search values
            hit_items = []
            for k, v in acc.items():
                if isinstance(v, dict) and match_entry(v, micro_id):
                    hit_items.append((k, v))
            if not hit_items:
                raise SystemExit(f"[ERROR] Could not find any entry matching '{micro_id}'. "
                                 f"Try --list-ids to view candidates.")
            out_obj = {k: v for k, v in hit_items}

    ANNS.write_text(json.dumps(out_obj, indent=2, ensure_ascii=False), encoding="utf-8")
    # Count entries written
    n = len(out_obj) if isinstance(out_obj, list) else (len(next(iter(out_obj.values()))) if isinstance(out_obj, dict) and any(isinstance(v, list) for v in out_obj.values()) else len(out_obj))
    print(f"[OK] Wrote filtered annotations for micro {micro_id} → {ANNS} ({n} entry)")

def run_s_
