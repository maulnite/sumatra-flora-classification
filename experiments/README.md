# Experiments

Use one Markdown report per experiment to prevent three members from editing the same file simultaneously. Start from [template](../docs/EXPERIMENT_TEMPLATE.md) or run:

~~~bash
python scripts/new_experiment.py --id EXP-004 --owner your-handle --title "ResNet scratch baseline"
~~~

Report real measured metrics only; no fabricated numbers. Each experiment has two files: `EXP-###.md` (human report) and `EXP-###.json` (machine-readable metadata used by `scripts/registry.py`). Update the [central registry](../docs/EXPERIMENT_LOG.md) with `python scripts/registry.py render` once reviewed.

Branch `kaka` holds the backfilled history `EXP-004` to `EXP-025`. Those reports copy numbers from the original logs; the training code is not in this repository.
