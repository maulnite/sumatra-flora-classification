#!/usr/bin/env bash
# Gelombang 2026-11-12: cari model tunggal terbaik. Semua resep R (rot180, MixUp/CutMix 0.5, EMA), dari nol, TTA flip, W&B project dateraquest.
# Pembanding OOF @224: R_r18d 0.7524 (300 ep), R_e600 0.7509 (resnet18 600 ep).
cd ~/inahuy/dateruy && source ~/venvs/dq/bin/activate
go() { tag=$1; shift; nohup python train.py run --recipe R --tta flip --jobs 5 --workers 50 --tag "$tag" --wandb dateraquest "$@" > "$tag.log" 2>&1 & echo "launched $tag (pid $!)"; }
go R_r18d_e600      --arch resnet18d --epochs 600                      # arsitektur terbaik + epoch lebih panjang
go R_r34d_e600      --arch resnet34d --epochs 600                      # stem "d" di model lebih dalam
go R_r18d_i288      --arch resnet18d --img 288                         # training di resolusi lebih tinggi (300 ep, pembanding langsung R_r18d)
go R_r18d_v5        --arch resnet18d --cmode corrupt_v5                # + blur/downscale/abu-abu (300 ep, pembanding langsung R_r18d)
go R_r18d_i288_e600 --arch resnet18d --img 288 --epochs 600            # gabungan resolusi + epoch
