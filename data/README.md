# Local competition data (not tracked)

Download the official dataset using authorized Kaggle access to `data/raw/`.

Expected directory: `train/` (30 labeled species folders), `test/` (2010 unlabeled images), `train.csv`, `class_mapping.csv` and `sample_submission.csv`.

Do not commit images/CSVs or introduce outside data. Inspect locally via:

~~~bash
python scripts/inspect_dataset.py --root data/raw
~~~
