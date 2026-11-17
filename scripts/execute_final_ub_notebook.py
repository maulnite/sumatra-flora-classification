"""Execute the final student notebook with the calling Python interpreter."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    args = parser.parse_args()
    path = args.notebook.resolve()
    local_jupyter = path.parent / ".jupyter"
    kernel_dir = local_jupyter / "kernels" / "dq-final-ub"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "DATERA final (project Python)", "language": "python",
    }, indent=2) + "\n")
    os.environ["JUPYTER_PATH"] = str(local_jupyter) + os.pathsep + os.environ.get("JUPYTER_PATH", "")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "4"
    notebook = nbformat.read(path, as_version=4)
    started = time.time()

    def progress(cell, cell_index, **kwargs):
        print(f"cell {cell_index + 1}/{len(notebook.cells)} {cell.cell_type} dispatched", flush=True)

    client = NotebookClient(notebook, kernel_name="dq-final-ub", timeout=1200,
                            resources={"metadata": {"path": str(path.parent)}},
                            on_cell_complete=progress)
    status = "error"
    try:
        client.execute()
        nbformat.validate(notebook)
        status = "success"
    finally:
        nbformat.write(notebook, path)
        (path.parent / "execution.json").write_text(json.dumps({
            "status": status, "python": sys.executable, "elapsed_seconds": time.time() - started,
            "notebook": path.name,
            "code_cells_executed": sum(c.cell_type == "code" and c.execution_count is not None for c in notebook.cells),
        }, indent=2) + "\n")
        print(status, flush=True)


if __name__ == "__main__":
    main()
