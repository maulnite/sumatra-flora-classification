"""Launch the authorized graph pilot detached and record a heartbeat every minute.

Use the team Python interpreter. Re-running the launcher returns the active job;
the supervisor lock prevents duplicate computation. Only the pilot's own files
and child process are managed. No automatic remote upload or git mutation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/LOCAL-PROP-004-PILOT.yaml")
    parser.add_argument("--supervise", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    out = ROOT / cfg["output_root"]; out.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS=str(cfg["features"]["cpu_threads"]), OPENBLAS_NUM_THREADS=str(cfg["features"]["cpu_threads"]),
                       MKL_NUM_THREADS=str(cfg["features"]["cpu_threads"]), NO_ALBUMENTATIONS_UPDATE="1")
    if not args.supervise:
        existing = read(out / "job.json")
        pid = existing.get("supervisor_pid")
        if pid and Path(f"/proc/{pid}/cmdline").exists():
            if "launch_retrieval_graph.py" in Path(f"/proc/{pid}/cmdline").read_text(errors="replace"):
                print(json.dumps(existing)); return
        with (out / "supervisor.log").open("a") as log:
            supervisor = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--config", str(args.config.resolve()), "--supervise"],
                    cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=environment, start_new_session=True)
        print(json.dumps(dict(supervisor_pid=supervisor.pid, output_root=str(out), detached=True, launched_at=now())))
        return
    with (out / "supervisor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (out / "runner.log").open("a") as log:
            runner = subprocess.Popen([sys.executable, "-u", str(ROOT / "scripts/pilot_retrieval_graph.py"), "--config", str(args.config.resolve())],
                                      cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=environment)
            job = dict(supervisor_pid=os.getpid(), runner_pid=runner.pid, started_at=now(),
                       heartbeat_seconds=cfg["execution"]["monitor_interval_seconds"], runner_log=str(out / "runner.log"),
                       status_file=str(out / "status.json"), output_root=str(out), state="running")
            write(out / "job.json", job)
            while True:
                state = read(out / "status.json")
                write(out / "heartbeat.json", dict(at=now(), runner_pid=runner.pid, return_code=runner.poll(), pilot_status=state))
                try:
                    code = runner.wait(timeout=cfg["execution"]["monitor_interval_seconds"])
                    break
                except subprocess.TimeoutExpired:
                    continue
            result = read(out / "result.json")
            job.update(finished_at=now(), return_code=code, state="completed" if code == 0 else "failed")
            write(out / "job.json", job)
            write(out / "heartbeat.json", dict(at=now(), runner_pid=runner.pid, return_code=code, pilot_status=read(out / "status.json")))
            write(out / "completion.json", dict(**job, graph_component_screen_passed=result.get("graph_component_screen_passed"),
                       primary_pooled_delta=result.get("primary_pooled_delta"), result_file=str(out / "result.json")))


if __name__ == "__main__":
    main()
