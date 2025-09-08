Yes—apply **both** code patches and you’ll cover **both micro237 and micro860** (and likely any others with odd sizes).

### What to change (once)

1. **`src/inpainting/inpaint.py` – fix loss-size mismatches (affects micro860)**

   * Add helper at top:

   ```python
   def _align_hw(a, b):
       H = min(a.shape[-2], b.shape[-2])
       W = min(a.shape[-1], b.shape[-1])
       return a[..., :H, :W], b[..., :H, :W]
   ```

   * In `optimise_noise(...)`, just **before** `loss = (raw - target)**4`:

   ```python
   if raw.shape != target.shape:
       raw, target = _align_hw(raw, target)
   ```

   * (If a `mask` is also used in that loss block, align it too to `raw`.)

2. **`src/inpainting/inpaint.py` – fix final mask merge (affects micro237)**

   * At top (if not already there):

   ```python
   from PIL import Image
   import numpy as np

   def _resize_mask_to(mask, H, W):
       m = np.asarray(mask)
       if m.shape[:2] != (H, W):
           m_img = Image.fromarray((m > 0).astype(np.uint8) * 255)
           m_img = m_img.resize((W, H), resample=Image.NEAREST)
           m = (np.array(m_img) > 0)
       return m
   ```

   * Where it crashed (inside `inpaint(...)` near the end), replace the assignment with:

   ```python
   # Make shapes consistent
   H, W = final_img.shape[:2]
   mask_ip = _resize_mask_to(mask_ip, H, W)
   final_img, final_img_fresh = _align_hw(final_img, final_img_fresh)
   H, W = final_img.shape[:2]
   if mask_ip.shape[:2] != (H, W):
       mask_ip = _resize_mask_to(mask_ip, H, W)

   sel = mask_ip.astype(bool)
   final_img[sel] = final_img_fresh[sel]
   ```

   (Note: same mask on both sides—no `!=`.)

3. Keep the small robustness tweaks you already added:

   * **Create output dirs** in `train(...)` before saving.
   * **Default `crop`** in `anns.json` if missing.
   * **Headless backends** (`MPLBACKEND=Agg`, `QT_QPA_PLATFORM=offscreen`).
   * **Lazy imports** in `src/inpainting/__init__.py` and `src/slicegan/__init__.py`.

### Then run (can be in parallel)

Terminal for **237** (e.g., A5000):

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export MPLBACKEND=Agg
export QT_QPA_PLATFORM=offscreen
python run_micro.py --query "000237.png" --steps inpaint
```

Terminal for **860** (e.g., A6000):

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export MPLBACKEND=Agg
export QT_QPA_PLATFORM=offscreen
python run_micro.py --query "000860.png" --steps inpaint
```

### If anything else pops up

Add one-time debug prints right before the loss and before the final mask merge:

```python
print("[DBG] raw", tuple(raw.shape), "target", tuple(target.shape))
print("[DBG] final", final_img.shape, "fresh", final_img_fresh.shape, "mask", mask_ip.shape)
```

That will pinpoint any new size edge case fast—but with the two patches above, you should be good for both 237 and 860.

