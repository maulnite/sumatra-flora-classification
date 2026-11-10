"""Create a new experiment Markdown document without overwriting existing work."""
from __future__ import annotations

import argparse
import re
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Scaffold a numbered experiment.")
    parser.add_argument("--id", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--title", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"EXP-\d{3,}", args.id):
        parser.error("Experiment ID must resemble EXP-004")
    output = Path("experiments") / f"{args.id}.md"
    if output.exists():
        parser.error(f"Will not overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    template = Path("docs/EXPERIMENT_TEMPLATE.md").read_text(encoding="utf-8")
    template = template.replace("EXP-XXX — Experiment title", f"{args.id} — {args.title}")
    marker = chr(96)
    template = template.replace("Author: " + marker + "@TBD" + marker,
                                "Author: " + marker + "@" + args.owner.lstrip("@") + marker)
    output.write_text(template, encoding="utf-8")
    print(f"Created {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
