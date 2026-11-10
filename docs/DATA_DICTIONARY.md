# Dataset dictionary

| File/directory | Content | Columns/description |
| --- | --- | --- |
| `train/` | Labeled organizer-provided images | 30 class subfolders, e.g. `0_Albizia_saponaria/` |
| `test/` | 2,010 unlabeled test images | e.g. `test_0001.jpg` |
| `train.csv` | Training image metadata | `id` and `label` (0–29) |
| `class_mapping.csv` | Label to plant species mapping | `label`, `species`, `folder_name` |
| `sample_submission.csv` | Official prediction format/template | `id` and `label` |

Do not upload raw data to GitHub. Inspect sizes, balance, duplicates, camera conditions and train/validation leakage using only organizer data.
