"""Plain local run records: manifest, source snapshot, metrics, experiment index (plan §8)."""
from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / "experiments" / "index.csv"
INDEX_FIELDS = [
    "run_id", "hypothesis", "plan_rev", "purpose", "command", "status", "start_utc", "end_utc",
    "wall_s", "device", "device_hours", "artifact_dir", "key_result",
]


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except Exception as exc:  # pragma: no cover
        return f"<git unavailable: {exc}>"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    return sha256_bytes(Path(p).read_bytes())


def config_hash(cfg: dict) -> str:
    return sha256_bytes(json.dumps(cfg, sort_keys=True, default=str).encode())


def plan_revision() -> str:
    plan = ROOT / "docs" / "RESEARCH_PLAN.md"
    for line in plan.read_text().splitlines():
        if line.startswith("Revision:"):
            return line.split("·")[0].replace("Revision:", "").strip()
    return "unknown"


def env_info(device: str) -> dict:
    import numpy
    import torch

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip(),
        "torch_threads": torch.get_num_threads(),
        "device": device,
        "mps_available": bool(torch.backends.mps.is_available()),
    }


class Run:
    """Reserve runs/<run_id>/, write manifest before launch, finalize afterward."""

    def __init__(self, prefix: str, hypothesis: str, purpose: str, cfg: dict, device: str = "cpu"):
        stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%S")
        run_id = f"{prefix}-{stamp}"
        d = ROOT / "runs" / run_id
        n = 1
        while d.exists():  # never reuse a run directory
            n += 1
            d = ROOT / "runs" / f"{run_id}-{n}"
        d.mkdir(parents=True)
        self.dir, self.run_id = d, d.name
        self.t0 = time.time()
        self.device = device
        self.cfg = cfg
        commit = _git("rev-parse", "HEAD").strip()
        status = _git("status", "--porcelain")
        dirty = bool(status.strip())
        src = {}
        if dirty:
            patch = _git("diff", "HEAD")
            (d / "source.patch").write_text(patch)
            untracked = [l[3:] for l in status.splitlines() if l.startswith("??")]
            keep = [p for p in untracked if p.endswith((".py", ".yaml", ".yml", ".toml")) and not p.startswith("runs/")]
            snap = d / "untracked_src"
            for p in keep:
                fp = ROOT / p
                if fp.is_file():
                    (snap / p).parent.mkdir(parents=True, exist_ok=True)
                    (snap / p).write_bytes(fp.read_bytes())
                    src[p] = sha256_file(fp)
        self.manifest: dict[str, Any] = {
            "run_id": self.run_id,
            "hypothesis": hypothesis,
            "plan_rev": plan_revision(),
            "purpose": purpose,
            "command": " ".join([Path(sys.executable).name, *sys.argv]),
            "cwd": os.getcwd(),
            "start_utc": _now(),
            "status": "running",
            "git_commit": commit,
            "git_dirty": dirty,
            "source_patch_sha256": sha256_file(d / "source.patch") if dirty else None,
            "untracked_src_sha256": src,
            "config": cfg,
            "config_sha256": config_hash(cfg),
            "env": env_info(device),
            "pid": os.getpid(),
        }
        self._write_manifest()
        self.log_path = d / "log.txt"

    @classmethod
    def reopen(cls, run_dir) -> "Run":
        """Resume an interrupted run in its own directory (appends to log; keeps manifest history)."""
        self = cls.__new__(cls)
        self.dir = Path(run_dir)
        self.run_id = self.dir.name
        self.manifest = json.loads((self.dir / "manifest.json").read_text())
        self.cfg = self.manifest["config"]
        self.device = self.manifest["env"]["device"]
        prev = float(self.manifest.get("wall_s") or 0.0)
        self.t0 = time.time() - prev
        self.manifest.setdefault("resumes", []).append({"utc": _now(), "pid": os.getpid(), "git_commit": _git("rev-parse", "HEAD").strip()})
        self.manifest["status"] = "running"
        self.log_path = self.dir / "log.txt"
        self._write_manifest()
        return self

    def _write_manifest(self) -> None:
        tmp = self.dir / "manifest.json.tmp"
        tmp.write_text(json.dumps(self.manifest, indent=2, default=str))
        tmp.replace(self.dir / "manifest.json")

    def log(self, msg: str) -> None:
        line = f"[{_now()}] {msg}"
        print(line, flush=True)
        with open(self.log_path, "a") as f:
            f.write(line + "\n")

    def save_json(self, name: str, obj: Any) -> Path:
        p = self.dir / name
        p.write_text(json.dumps(obj, indent=2, default=str))
        return p

    def finalize(self, status: str, metrics: dict, key_result: str, devices: int = 1) -> None:
        wall = time.time() - self.t0
        self.save_json("metrics.json", metrics)
        artifacts = {p.name: sha256_file(p) for p in sorted(self.dir.iterdir()) if p.is_file() and p.name != "manifest.json"}
        self.manifest.update(
            status=status, end_utc=_now(), wall_s=round(wall, 2),
            device_hours=round(wall * devices / 3600, 4), artifacts_sha256=artifacts, key_result=key_result,
        )
        self._write_manifest()
        INDEX.parent.mkdir(exist_ok=True)
        new = not INDEX.exists()
        with open(INDEX, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
            if new:
                w.writeheader()
            w.writerow({
                "run_id": self.run_id, "hypothesis": self.manifest["hypothesis"], "plan_rev": self.manifest["plan_rev"],
                "purpose": self.manifest["purpose"], "command": self.manifest["command"], "status": status,
                "start_utc": self.manifest["start_utc"], "end_utc": self.manifest["end_utc"], "wall_s": round(wall, 2),
                "device": self.device, "device_hours": round(wall * devices / 3600, 4),
                "artifact_dir": f"runs/{self.run_id}", "key_result": key_result,
            })


def load_config(path: str) -> dict:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)
