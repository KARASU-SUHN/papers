Yes—you can run **micro237** and **micro860** in parallel from VS Code. The two key things:

1. give each run its **own GPU** (so they don’t fight), and
2. make sure each run points `anns.json` at the **right PNG** before it starts.

Here’s a clean, safe way.

---

## A. Find the exact filenames to target

(So your filter matches the right entry.)

```bash
# from repo root
grep -nE '"data_path".*860\.png' data/prelabelled_anns.json | head
grep -nE '"data_path".*237\.png' data/prelabelled_anns.json | head
```

Note the exact names (e.g., `000860.png`, `000237.png`). If your importer wrote to `micrographs_raw/`, add this once:

```bash
[ -d data/micrographs_png ] || ln -s micrographs_raw data/micrographs_png
```

---

## B. Open **two terminals** in VS Code

### Terminal 1 → micro237 on GPU 0 (A5000, for example)

```bash
# pin this process to physical GPU 0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export MPLBACKEND=Agg
export QT_QPA_PLATFORM=offscreen

# use prelabels and run only 237 (skip PlotOptiX-heavy animate)
cp data/prelabelled_anns.json data/anns.json
python run_micro.py --query "000237.png" --steps inpaint slicegan
```

Wait until you see it print something like “training micro237” or the progress bar has started.

### Terminal 2 → micro860 on GPU 1 (A6000)

```bash
# pin this process to physical GPU 1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export MPLBACKEND=Agg
export QT_QPA_PLATFORM=offscreen

# now filter and run only 860 (this will rewrite anns.json for THIS process)
cp data/prelabelled_anns.json data/anns.json
python run_micro.py --query "000860.png" --steps inpaint slicegan
```

> Why this works: each Python process reads `data/anns.json` **at startup**. Once running, it doesn’t re-read it, so changing `anns.json` for the second terminal won’t affect the first. Outputs go to separate folders:
>
> * `data/inpaint_runs/micro237/...` and `data/slicegan_runs/...`
> * `data/inpaint_runs/micro860/...` and `data/slicegan_runs/...`

---

## C. Verify which GPU each run is using

In another terminal:

```bash
watch -n 0.5 nvidia-smi
```

You should see:

* GPU 0 memory/util climbing for the **237** run
* GPU 1 memory/util climbing for the **860** run

If you want to double-check inside Python:

```bash
python - <<'PY'
import torch
print("device_count =", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"cuda:{i} ->", torch.cuda.get_device_name(i))
PY
```

---

## D. Tips / gotchas

* If `run_micro.py` says “no match” for 860, search the exact name first:

  ```bash
  grep -n "860.png" data/prelabelled_anns.json | head
  ```

  then use that exact string in `--query`.
* Keep skipping `animate` unless you’ve installed PlotOptiX + Mono/.NET.
* If you want to force **CPU** for one run: `export CUDA_VISIBLE_DEVICES=""` in that terminal (slower).

Want me to drop a tiny script to make quick slice GIFs (no PlotOptiX) for both runs so you can visually compare 237 vs 860?
