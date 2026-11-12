"""Experiment registry: validate per-experiment JSON metadata, render the central table, and
aggregate experiments from every remote branch.

Usage (from the repository root):
    python scripts/registry.py check                    # validate experiments/EXP-*.json in the working tree
    python scripts/registry.py render                   # rewrite the generated block in docs/EXPERIMENT_LOG.md
    python scripts/registry.py render --all-branches    # fetch origin and include experiments from every branch

Standard library only, so it runs in CI and on any teammate's machine without extra installs.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = "experiments"
LOG_PATH = ROOT / "docs" / "EXPERIMENT_LOG.md"
PROTOCOL_PATH = ROOT / "docs" / "protocols.json"
BEGIN, END = "<!-- BEGIN GENERATED REGISTRY -->", "<!-- END GENERATED REGISTRY -->"

STATUSES = {"planned", "running", "completed", "partial", "paused", "invalid"}
ID_RE = re.compile(r"EXP-\d{3,}")
REQUIRED = ("id", "title", "owner", "branch", "status", "date", "protocol", "results", "evidence")


def load_protocols(path: Path = PROTOCOL_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_score(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and 0.0 <= x <= 1.0


def validate(exp: dict, protocols: dict, base: Path = ROOT) -> list[str]:
    """Return a list of problems; an empty list means the experiment is valid."""
    errs = []
    missing = [k for k in REQUIRED if k not in exp]
    if missing:
        return [f"missing keys: {', '.join(missing)}"]
    eid = exp["id"]
    if not ID_RE.fullmatch(str(eid)):
        errs.append(f"id {eid!r} must look like EXP-004")
    if exp["status"] not in STATUSES:
        errs.append(f"status {exp['status']!r} not in {sorted(STATUSES)}")
    if exp["protocol"] not in protocols:
        errs.append(f"protocol {exp['protocol']!r} is not defined in docs/protocols.json")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(exp["date"])):
        errs.append("date must be YYYY-MM-DD")
    if not isinstance(exp["results"], list):
        errs.append("results must be a list")
        return errs
    scored = 0
    for i, r in enumerate(exp["results"]):
        if "name" not in r or "macro_f1" not in r:
            errs.append(f"results[{i}] needs name and macro_f1")
            continue
        if r["macro_f1"] is not None and not _is_score(r["macro_f1"]):
            errs.append(f"results[{i}] macro_f1 {r['macro_f1']!r} is not in [0, 1]")
        if r["macro_f1"] is not None:
            scored += 1
    lb = exp.get("leaderboard") or {}
    for key in ("public", "private"):
        if lb.get(key) is not None and not _is_score(lb[key]):
            errs.append(f"leaderboard.{key} {lb[key]!r} is not in [0, 1]")
    if exp["status"] == "completed":
        if scored == 0:
            errs.append("completed experiment needs at least one numeric macro_f1")
        report = exp["evidence"].get("report")
        if not report or not (base / report).exists():
            errs.append(f"evidence.report {report!r} does not exist")
    return errs


def best_oof(exp: dict):
    vals = [r["macro_f1"] for r in exp["results"] if r.get("scope") == "oof" and r.get("macro_f1") is not None]
    return max(vals) if vals else None


def render_table(exps: list[dict], protocols: dict) -> str:
    rows = ["| ID | Status | Owner / branch | Protocol | Best OOF macro F1 | Public LB | Note |",
            "| --- | --- | --- | --- | ---: | ---: | --- |"]
    for e in sorted(exps, key=lambda e: e["id"]):
        oof = best_oof(e)
        lb = (e.get("leaderboard") or {}).get("public")
        rows.append("| {id} | {st} | {ow} / `{br}` | `{pr}` | {oof} | {lb} | {note} |".format(
            id=e["id"], st=e["status"], ow=e["owner"], br=e["branch"], pr=e["protocol"],
            oof=f"{oof:.4f}" if oof is not None else "—",
            lb=f"{lb:.5f}" if lb is not None else "—",
            note=e.get("summary", "").replace("|", "/")))
    best = sorted((e for e in exps if best_oof(e) is not None), key=best_oof, reverse=True)[:5]
    lines = ["", "### Top 5 by OOF (same protocol only compare within a protocol)", "",
             "| ID | Protocol | OOF macro F1 | Public LB |", "| --- | --- | ---: | ---: |"]
    for e in best:
        lb = (e.get("leaderboard") or {}).get("public")
        lines.append(f"| {e['id']} | `{e['protocol']}` | {best_oof(e):.4f} | {f'{lb:.5f}' if lb is not None else '—'} |")
    lines += ["", "### Protocols", "", "| Protocol | Meaning |", "| --- | --- |"]
    used = sorted({e["protocol"] for e in exps})
    for p in used:
        lines.append(f"| `{p}` | {protocols.get(p, {}).get('description', 'UNDEFINED')} |")
    return "\n".join(rows + lines) + "\n"


def load_local(base: Path = ROOT) -> list[dict]:
    out = []
    for p in sorted((base / EXP_DIR).glob("EXP-*.json")):
        out.append(json.loads(p.read_text(encoding="utf-8")))
    return out


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout


def load_all_branches() -> list[dict]:
    """Read experiments/EXP-*.json from every origin/* branch (no checkout needed)."""
    _git("fetch", "origin", "--prune")
    refs = [r.removeprefix("refs/remotes/") for r in
            _git("for-each-ref", "--format=%(refname)", "refs/remotes/origin").split()
            if not r.endswith("/HEAD")]
    out, seen = [], {}
    for ref in refs:
        files = [f for f in _git("ls-tree", "-r", "--name-only", ref, EXP_DIR).split("\n")
                 if re.fullmatch(rf"{EXP_DIR}/EXP-\d+\.json", f)]
        for f in files:
            exp = json.loads(_git("show", f"{ref}:{f}"))
            exp["_source"] = ref
            if exp["id"] in seen:
                raise SystemExit(f"duplicate {exp['id']} on {seen[exp['id']]} and {ref}: renumber one of them")
            seen[exp["id"]] = ref
            out.append(exp)
    return out


def update_log(table: str, log: Path = LOG_PATH) -> None:
    text = log.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        raise SystemExit(f"{log} has no generated block markers")
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    log.write_text(f"{head}{BEGIN}\n\n{table}\n{END}{tail}", encoding="utf-8")


def cmd_check(_args) -> int:
    protocols = load_protocols()
    exps = load_local()
    bad, ids = 0, {}
    for e in exps:
        errs = validate(e, protocols)
        if e.get("id") in ids:
            errs.append(f"duplicate id also in {ids[e['id']]}")
        ids.setdefault(e.get("id"), e.get("branch"))
        for err in errs:
            bad += 1
            print(f"{e.get('id', '?')}: {err}")
    print(f"{len(exps)} experiment(s) checked, {bad} problem(s)")
    return 1 if bad else 0


def cmd_render(args) -> int:
    protocols = load_protocols()
    exps = load_all_branches() if args.all_branches else load_local()
    bad = [(e.get("id"), err) for e in exps for err in validate(e, protocols)]
    if bad:
        for eid, err in bad:
            print(f"{eid}: {err}")
        return 1
    update_log(render_table(exps, protocols))
    print(f"rendered {len(exps)} experiment(s) into {LOG_PATH.relative_to(ROOT)}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="validate local experiment metadata")
    r = sub.add_parser("render", help="regenerate the registry table")
    r.add_argument("--all-branches", action="store_true", help="include experiments from every origin/* branch")
    args = ap.parse_args(argv)
    return {"check": cmd_check, "render": cmd_render}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
