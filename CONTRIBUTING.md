# Contributing — three-person team

Collaboration and code sharing **within our registered team** are allowed. Cross-team solution/code/prediction sharing is prohibited. This repo is **public**, so anyone—including competing teams—can read what we commit. Do not accidentally publish organizer data or sensitive assets.

## Workflow

1. Owner invites two teammates via GitHub Settings → Collaborators.
2. Clone, update `main` and create your personal branch:

~~~bash
git switch main
git pull origin main
git switch -c exp/your-handle/experiment-topic
~~~

3. Implement shared code in `src/`, notebook in `notebooks/`, config in `configs/` and a unique experiment report in `experiments/EXP-###.md`.
4. Record exact split, seed, architecture, augmentation, config, library versions, hardware, validation Macro F1 and measured runtime.
5. Commit and push:

~~~bash
git add .
git status
git commit -m "exp: document from-scratch baseline"
git push -u origin exp/your-handle/experiment-topic
~~~

6. Open a pull request to `main`, get a teammate review, and merge only after checks pass.

## Pull-request checklist

- [ ] One clear purpose and linked experiment ID
- [ ] No pretrained weights, external dataset, forbidden VLM/AutoML, or labeled test data
- [ ] Reproducible settings, seed, split and command recorded
- [ ] Metric values are real; placeholders clearly marked
- [ ] CI and local validator tests pass
- [ ] No dataset, predictions, model weights, credentials or large artifacts added
- [ ] At least one teammate reviewed

Use independent notebooks such as `EXP-001_member1_baseline.ipynb`, not one shared `train.ipynb`. Do not force-push `main`.

The organizer's ZIP with executed notebooks and matching CSV is a separate final deliverable, not a commit to GitHub.
