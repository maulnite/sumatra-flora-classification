"""Detached A/B supervisor: heartbeat, duplicate protection, same-batch OOM resume."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import yaml

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def write(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n"); temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_launch(config, out):
    spec, smoke = read(out / "spec.json"), read(out / "smoke_result.json")
    if not spec or not smoke.get("passed") or smoke.get("spec_sha256") != sha(out / "spec.json"):
        raise ValueError("Matching preparation and GPU smoke test are required")
    for path, expected in spec["code_sha256"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"Prepared code/config changed: {path}")
    if str(config.resolve()) not in spec["code_sha256"]:
        raise ValueError("Unrecognized configuration")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/LOCAL-PROP-004-SSL.yaml")
    parser.add_argument("--supervise", action="store_true")
    args = parser.parse_args(); cfg = yaml.safe_load(args.config.read_text())
    out = ROOT / cfg["output_root"]; out.mkdir(parents=True, exist_ok=True)
    validate_launch(args.config, out)
    if (out / "result.json").exists():
        print(json.dumps(dict(state="already_completed", result_file=str(out / "result.json")))); return
    environment = os.environ.copy()
    environment.update({key: str(cfg["execution"]["cpu_threads"]) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
    environment["NO_ALBUMENTATIONS_UPDATE"] = "1"
    if not args.supervise:
        existing = read(out / "job.json"); pid = existing.get("supervisor_pid")
        path = Path(f"/proc/{pid}/cmdline")
        if pid and path.exists() and Path(__file__).name in path.read_text(errors="replace"):
            print(json.dumps(existing)); return
        with (out / "supervisor.log").open("a") as stream:
            process = subprocess.Popen([cfg["python"], str(Path(__file__).resolve()), "--config", str(args.config.resolve()), "--supervise"],
                    cwd=ROOT, env=environment, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps(dict(supervisor_pid=process.pid, detached=True, output_root=str(out), launched_at=now()))); return
    with (out / "supervisor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        job = dict(supervisor_pid=os.getpid(), started_at=now(), output_root=str(out), state="starting",
                   heartbeat_seconds=cfg["execution"]["monitor_interval_seconds"], oom_restarts=0)
        while True:
            with (out / "runner.log").open("a") as stream:
                process = subprocess.Popen([cfg["python"], "-u", str(ROOT / cfg["execution"]["runner"]), "--config", str(args.config.resolve()), "--run"],
                        cwd=ROOT, env=environment, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
                job.update(runner_pid=process.pid, state="running", last_started_at=now())
                write(out / "job.json", job)
                while True:
                    write(out / "heartbeat.json", dict(at=now(), runner_pid=process.pid, return_code=process.poll(), status=read(out / "status.json")))
                    try:
                        code = process.wait(timeout=cfg["execution"]["monitor_interval_seconds"]); break
                    except subprocess.TimeoutExpired:
                        continue
            if code == cfg["execution"]["oom_exit_code"]:
                job.update(state="waiting_after_cuda_oom", oom_restarts=job["oom_restarts"]+1, last_oom_at=now())
                write(out / "job.json", job)
                write(out / "heartbeat.json", dict(at=now(), **job, status=read(out / "status.json")))
                time.sleep(cfg["execution"]["oom_retry_seconds"])
                continue
            job.update(state="completed" if code == 0 else "failed", return_code=code, finished_at=now())
            write(out / "job.json", job)
            write(out / "heartbeat.json", dict(at=now(), **job, status=read(out / "status.json")))
            write(out / "completion.json", dict(**job, screening_passed=read(out / "result.json").get("screening_passed")))
            return


if __name__ == "__main__":
    main()
