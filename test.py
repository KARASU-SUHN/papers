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
