# Three-member teamwork

Members on the *same registered team* may share and review code. A **public repo** is readable by everyone, not just teammates, so coordinate what to publish. Never exchange predictions/solutions with competing teams.

## Suggested work division (adjust freely)
- Member 1: data inspection, EDA, splits, augmentation.
- Member 2: CNN baseline, optimization and regularization.
- Member 3: alternative architecture, metrics, inference and ensemble ideas.

## Daily workflow
1. Pull fresh `main`; create your own `exp/<handle>/<topic>` or `feat/<handle>/<topic>` branch.
2. Reserve a unique EXP ID; record hypothesis, config, seed, split, and equipment.
3. Run an experiment and capture **local Macro F1**, per-class patterns and reproducibility details.
4. Push code plus independent Markdown report; create a pull request; get a review.
5. Update registry and README after verification.

Before Kaggle submission, validate ID/label/order and stay within 3 submissions/day. Before final ZIP, rerun notebook end-to-end and ensure matching CSV.
