#!/bin/bash

source /home/slyforce/src/MusicStyleTransfer/venv/bin/activate;

python -m music_style_transfer.GAN.main \
--batch-size 32 \
--out-samples /tmp/out \
--max-seq-len 64 \
--slices-per-quarter-note 4 \
--data ~/src/MusicStyleTransfer/work/data/guitar_bass \
--checkpoint-frequency 1000 \
--model-output test/ \
--g-n-layers 1 \
--g-rnn-hidden-dim 256 \
--g-emb-hidden-dim 64 \
--d-n-layers 1 \
--d-rnn-hidden-dim 256 \
--d-emb-hidden-dim 64 
