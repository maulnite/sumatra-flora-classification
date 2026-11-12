#!/usr/bin/env bash
# Screening arsitektur lain di fold 0 saja: resep R (300 ep, rot180, mix corrupt_v4, MixUp/CutMix 0.5, EMA), TTA flip.
# Semua dari nol (train.py: pretrained=False). Log ke W&B project "dateraquest" (entity <wandb-entity>).
# Pembanding fold 0: R_r18d 0.7693, R_e600 0.7585, R 0.6821 (F1 v4+rot).
# Kalau ada yang bagus, lanjut 4 fold sisanya + ensemble (fold 0 otomatis di-skip):
#   python train.py run --recipe R --arch <arch> --tta flip --tag R_<arch> --jobs 2 --wandb dateraquest
cd ~/inahuy && source ~/venvs/dq/bin/activate
for arch in ${@:-regnety_008 regnety_016 efficientnet_b0 efficientnet_b2}; do
  nohup python train.py run --recipe R --arch "$arch" --tta flip --folds 0 --jobs 1 --workers 32 \
    --tag "R_$arch" --wandb dateraquest > "R_$arch.log" 2>&1 &
  echo "launched R_$arch (pid $!)"
done
