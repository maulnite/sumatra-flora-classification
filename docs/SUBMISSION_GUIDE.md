# Submission checklist

## CSV format
- Name `submission.csv` with **exactly 2,010 prediction rows** and one header.
- Exactly columns `id,label` in that order.
- IDs, order and spelling must be identical to official `sample_submission.csv`.
- Labels must be integer 0–29, without missing/NaN/blank values.
- Check the actual model-generated file:

~~~bash
python scripts/validate_submission.py --file outputs/submissions/submission.csv --template data/raw/sample_submission.csv
~~~

Max **3 Kaggle submissions per day**.

## Organizer final verification
- Upload `IDTim_NamaTim.zip` to organizer portal **17 Nov 2026 by 23:59 WIB**.
- ZIP contains fully executed `.ipynb` notebook(s) with training logs and validation metrics, and the corresponding `.csv`.
- Notebook must reproduce the exact leaderboard submission. Rerun cleanly and compare before packaging.

## Finalists
- `IDTim_NamaTim.pptx` deadline **26 Nov 2026 by 23:59 WIB**.
- 10-minute presentation + 10-minute Q&A online.

Review current organizer instructions for any updates.
