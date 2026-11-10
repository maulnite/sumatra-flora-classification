# Competition rules — quick reference

These are notes transcribed from the organizer's DATERAQUEST 2.0 screenshots shared by the team (10 Nov 2026). Official Kaggle/organizer rules and subsequent updates take precedence.

## Technical
- Classify images of 30 plants from Kebun Raya ITERA using only the **official dataset**.
- All model training must be **from scratch** (random weights, no pretrained weights, no external training sources).
- Standard CNN/ResNet/EfficientNet/ConvNeXt/ViT architecture blueprints are allowed.
- No external datasets, manually labeled test samples, VLM/multimodal models such as CLIP/SigLIP/BioCLIP/BLIP, prediction APIs or instant AutoML.
- Optuna/KerasTuner hyperparameter tuning and conventional augmentation on official data are allowed.
- Random seeds should be fixed and results reproducible.
- **Within-team** sharing is permitted; cross-team sharing of code, predictions or collaboration is prohibited.
- Do not copy/fork competing public code for submission.

## Scoring
- Metric: Macro F1 over all 30 classes.
- Public leaderboard: 30% of test data; Private: 70% and decisive for advancement.
- Maximum 3 Kaggle submissions daily.

## Preliminary round
- Kaggle online competition.
- Final ZIP of fully executed `.ipynb` notebook(s) and matching `.csv`, named `IDTim_NamaTim.zip`.
- Organizer website deadline: **17 Nov 2026 at 23:59 WIB**.

## Final round (if qualified)
- Up to 10% of participating teams, capped at 3 teams per college, according to organizer.
- Submit `IDTim_NamaTim.pptx` by **26 Nov 2026 at 23:59 WIB**.
- Zoom session: 10-minute presentation and 10-minute Q&A.
- Scoring: presentation 30%, Q&A 40%, notebook narrative/explanation 30%.

Keep the official Kaggle rules open when preparing submissions.
