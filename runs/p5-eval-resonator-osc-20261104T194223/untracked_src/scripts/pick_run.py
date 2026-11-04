"""Print the unique passed full-pilot run dir (steps=1500, n_train=4000) for TASK MODEL SEED; exit 1 otherwise."""
import json, sys
from pathlib import Path
task, model, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
hits = []
for d in sorted(Path("runs").glob(f"p4-{task}-{model}-s{seed}-*")):
    try:
        man = json.loads((d / "manifest.json").read_text())
    except FileNotFoundError:
        continue
    c = man["config"]
    if man["status"] == "passed" and c.get("steps") == 1500 and c.get("n_train") == 4000 and (d / "best.pt").exists():
        hits.append(d)
if len(hits) != 1:
    sys.exit(f"pick_run: expected 1 run for {task}/{model}/s{seed}, found {[str(h) for h in hits]}")
print(hits[0])
