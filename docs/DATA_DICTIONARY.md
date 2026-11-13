# Dataset dictionary

| File/directory | Content | Columns/description |
| --- | --- | --- |
| `train/` | Labeled organizer-provided images | 30 class subfolders, e.g. `0_Albizia_saponaria/` |
| `test/` | 2,010 unlabeled test images | e.g. `test_0001.jpg` |
| `train.csv` | Training image metadata | `id` is a path relative to `train/`, e.g. `0_Albizia_saponaria/0_1.jpg`; `label` is 0–29 |
| `class_mapping.csv` | Label to plant species mapping | `label`, `species`, `folder_name` |
| `sample_submission.csv` | Official prediction format/template | `id` and `label` |

Do not upload raw data to GitHub. Inspect sizes, balance, duplicates, camera conditions and train/validation leakage using only organizer data.

Verified on 12 Nov 2026: 3,014 training images, 2,010 test images, 30 classes, 47–169 training images per class. All images decode as RGB JPEG. Test JPEG quantization is substantially coarser than train; account for this shift when designing augmentation and validation. See [the dataset audit](COMPETITION_AUDIT_2026-11-12.md) for measurements and local artifact paths.
