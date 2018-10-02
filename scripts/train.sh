#!/bin/bash

source /home/slyforce/src/MusicStyleTransfer/venv/bin/activate;

python -m music_style_transfer.GAN.main \
--batch-size 32 \
--out-samples /tmp/out \
--max-seq-len 64 \
--slices-per-quarter-note 4 \
--data ~/src/MusicStyleTransfer/work/data/guitar_bass \
--sampling-frequency 50 \
--epochs 10000 \
--discriminator-update-steps 5 \
--model-output test/ \
--g-learning-rate 0.00005 \
--g-n-layers 1 \
--g-rnn-hidden-dim 256 \
--g-emb-hidden-dim 256 \
--noise-dim 64 \
--d-learning-rate 0.00005 \
--d-n-layers 1 \
--d-rnn-hidden-dim 256 \
--d-emb-hidden-dim 256 --gpu --toy
